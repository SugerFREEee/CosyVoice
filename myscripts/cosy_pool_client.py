"""
并发客户端：向 cosy_server_pool 发送请求（token_file + prompt_wav），打印每条请求的耗时。

示例：
CUDA_VISIBLE_DEVICES=2 \
    /home/wjs/workspace/miniconda3/envs/cosyvoice/bin/python\
    myscripts/cosy_pool_client.py \
    --input-lst /home/wjs/workspace/data/CowboyZ/seed-tts-eval/seedtts_testset/zh/meta.lst \
    --tokens-dir /home/wjs/workspace/data/seedtts_tokens \
    --zmq-address ipc:///tmp/cosy_pool.sock \
    --num-workers 1 \
    --num-steps 10 \
    --total-reqs 20 \
    --detail
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import msgpack
import zmq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cosy_server_pool 并发客户端")
    parser.add_argument("--input-lst", type=str, required=True, help="meta 列表（含 utt|...|prompt_wav）")
    parser.add_argument("--tokens-dir", type=str, required=True, help="tokens 目录，文件名形如 <utt>.txt")
    parser.add_argument("--zmq-address", type=str, default="ipc:///tmp/cosy_pool.sock", help="cosy_server_pool 地址")
    parser.add_argument("--num-workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--total-reqs", type=int, default=None, help="截取请求数量")
    parser.add_argument("--num-steps", type=int, default=None, help="覆盖 num_inference_steps")
    parser.add_argument("--detail", action="store_true", help="输出每个请求的时间戳")
    return parser.parse_args()


def load_items(input_lst: str, tokens_dir: str, total: int | None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    tokens_dir = Path(tokens_dir)
    with open(input_lst, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 5:
                utt, _, prompt_wav, _, _ = parts
            elif len(parts) == 4:
                utt, _, prompt_wav, _ = parts
            elif len(parts) == 3:
                utt, _, prompt_wav = parts
            elif len(parts) == 2:
                utt, prompt_wav = parts
            else:
                # 不符合预期格式，跳过
                continue
            if utt.endswith(".wav"):
                utt = utt[:-4]
            token_file = tokens_dir / f"{utt}.txt"
            if not token_file.is_file():
                print(f"[WARN] token file missing: {token_file}")
                continue
            if prompt_wav and not Path(prompt_wav).is_absolute():
                prompt_wav = str(Path(input_lst).parent / prompt_wav)
            if not Path(prompt_wav).is_file():
                print(f"[WARN] prompt wav missing: {prompt_wav}")
                continue
            items.append({"utt": utt, "token_file": str(token_file), "prompt_wav": prompt_wav})
            if total and len(items) >= total:
                break
    return items


def worker(item: Dict[str, Any], zmq_addr: str, num_steps: int | None) -> Dict[str, Any]:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(zmq_addr)
    ts: Dict[str, Any] = {"id": item["utt"], "start": time.time()}
    try:
        payload = {"token_file": item["token_file"], "prompt_wav": item["prompt_wav"]}
        if num_steps is not None:
            payload["num_inference_steps"] = num_steps
        sock.send(msgpack.packb(payload, use_bin_type=True))
        ts["sent"] = time.time()
        resp = sock.recv()
        ts["recv"] = time.time()
        out = msgpack.unpackb(resp, raw=False)
        if isinstance(out, dict) and "error" in out:
            ts["error"] = out["error"]
        ts["done"] = time.time()
    except Exception as e:  # pragma: no cover
        ts["error"] = str(e)
        ts["done"] = time.time()
    finally:
        sock.close()
    return ts


def main():
    args = parse_args()
    items = load_items(args.input_lst, args.tokens_dir, args.total_reqs)
    if not items:
        print("没有可发送的请求")
        return

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(worker, item, args.zmq_address, args.num_steps) for item in items]
        with tqdm(total=len(futures), desc="requests", unit="req") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                pbar.update(1)

    ok = [r for r in results if "error" not in r]
    fail = len(results) - len(ok)
    if args.detail:
        for r in results:
            sid = r.get("id", "?")
            if "error" in r:
                print(f"[detail][{sid}] error={r['error']}")
            else:
                print(f"[detail][{sid}] start={r.get('start'):.6f} sent={r.get('sent'):.6f} recv={r.get('recv'):.6f} done={r.get('done'):.6f}")
    if ok:
        total_ms = [(r["done"] - r["start"]) * 1000 for r in ok]
        avg = sum(total_ms) / len(total_ms)
        wall = max(r["done"] for r in results) - min(r["start"] for r in results)
        throughput = len(ok) / wall if wall > 0 else 0
        print(f"[summary] success {len(ok)}/{len(results)}, fail={fail}, avg={avg:.1f}ms, wall={wall:.2f}s, qps={throughput:.2f}")
    else:
        print(f"[summary] all failed: {len(results)}")


if __name__ == "__main__":
    main()
