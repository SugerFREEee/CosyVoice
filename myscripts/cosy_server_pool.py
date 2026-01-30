"""
cosy_server 池：启动多个 cosy_server worker（token->DiT->wav），前端提供 ZeroMQ ROUTER，
按轮询把请求分发给后端 worker。适配 vllm_server_pool。

示例：
CUDA_VISIBLE_DEVICES=1 \
    /home/wjs/workspace/miniconda3/envs/cosyvoice/bin/python\
    myscripts/cosy_server_pool.py \
    --num-workers 1 \
    --frontend-address ipc:///tmp/cosy_pool.sock \
    --backend-base-address ipc:///tmp/cosy_worker.sock \
    --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --vllm-address ipc:///tmp/vllm_pool.sock

客户端连接 frontend-address，发送 cosy_server.py 期望的 msgpack payload。
"""

from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import zmq


def launch_workers(num_workers: int, base_addr: str, cosy_args: list[str]) -> tuple[list[subprocess.Popen], list[zmq.Socket]]:
    ctx = zmq.Context.instance()
    procs: list[subprocess.Popen] = []
    backends: list[zmq.Socket] = []

    for i in range(num_workers):
        addr = f"{base_addr}.{i}"
        args = [sys.executable, str(Path(__file__).with_name("cosy_server.py"))] + cosy_args + ["--zmq-address", addr]
        proc = subprocess.Popen(args)
        procs.append(proc)

        sock = ctx.socket(zmq.DEALER)
        sock.connect(addr)
        backends.append(sock)

    return procs, backends


def proxy(frontend_addr: str, backend_socks: list[zmq.Socket]) -> None:
    ctx = zmq.Context.instance()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind(frontend_addr)

    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    for s in backend_socks:
        poller.register(s, zmq.POLLIN)

    rr_idx = 0
    n = len(backend_socks)
    print(f"[cosy_pool] Proxy started. frontend={frontend_addr}, backends={n}")

    try:
        while True:
            events = dict(poller.poll())

            if frontend in events:
                frames = frontend.recv_multipart()
                backend = backend_socks[rr_idx]
                rr_idx = (rr_idx + 1) % n
                backend.send_multipart(frames)

            for s in backend_socks:
                if s in events:
                    frames = s.recv_multipart()
                    frontend.send_multipart(frames)
    finally:
        frontend.close(0)
        for s in backend_socks:
            s.close(0)


def main():
    parser = argparse.ArgumentParser(description="CosyVoice worker pool (round-robin)")
    parser.add_argument("--num-workers", type=int, default=1, help="启动的 cosy_server 实例数")
    parser.add_argument("--frontend-address", type=str, default="ipc:///tmp/cosy_pool.sock", help="对外暴露地址")
    parser.add_argument(
        "--backend-base-address",
        type=str,
        default="ipc:///tmp/cosy_worker.sock",
        help="后端基础地址，实际会追加 .0/.1 等",
    )
    # 其余参数原样传给 cosy_server.py
    known, extra = parser.parse_known_args()
    num_workers = known.num_workers

    procs, backend_socks = launch_workers(num_workers, known.backend_base_address, extra)

    def _cleanup():
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()

    atexit.register(_cleanup)
    time.sleep(0.5)
    try:
        proxy(known.frontend_address, backend_socks)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
