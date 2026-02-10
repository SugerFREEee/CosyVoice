#!/usr/bin/env python
# Compare runtime between zero_shot (with prompt audio) and sft (cached speaker) inference paths.
"""

/home/wjs/workspace/miniconda3/envs/dev/bin/python myscripts/compare.py --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --spk-id my_spk4 \
  --prompt-wav /home/wjs/workspace/CosyVoice/asset/Run_zero_shot.wav \
  --instruct-text "You are a helpful assistant.<|endofprompt|>" \
  --prompt-text "祥子找到外祖父定治，却被告知，父亲遭遇了诈骗并造成了168亿日元的损失，已“不再是丰川家的人”。" \
  --tts-text "祥子决定和父亲一起搬到一间破旧公寓里住。然而父亲酗酒颓废的模样却让祥子感到气愤。家庭的变故使得祥子无心参与乐队的活动" \
  --stream \
  --warmup-iters 2

python myscripts/compare.py --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --spk-id my_spk4 \
  --prompt-wav /home/wjs/workspace/CosyVoice/asset/Run_zero_shot.wav \
  --instruct-text "You are a helpful assistant.<|endofprompt|>" \
  --prompt-text "祥子找到外祖父定治，却被告知，父亲遭遇了诈骗并造成了168亿日元的损失，已“不再是丰川家的人”。" \
  --tts-text "祥子决定和父亲一起搬到一间破旧公寓里住。然而父亲酗酒颓废的模样却让祥子感到气愤。家庭的变故使得祥子无心参与乐队的活动" \
  --stream \
  --warmup-iters 2 \
  --enable-cache-dit \
  --dbcache-steps 10 \
  --dbcache-warmup 0 \
  --dbcache-max-cached -1 \
  --dbcache-fn 1 \
  --dbcache-bn 0 \
  --dbcache-rdt 0.35


"""

import argparse
import time
import sys
from pathlib import Path

# Ensure repo root on sys.path when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
import torchaudio
import torch
from pathlib import Path
logging.getLogger().setLevel(logging.INFO)

PREFIX = "You are a helpful assistant.<|endofprompt|>"
OUT_DIR = Path("output_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)



def save_wav(prefix, kind, wav, sr):
    path = OUT_DIR / f"{prefix}_{kind}.wav"
    torchaudio.save(str(path), wav.cpu(), sr)


def run_zero_shot(model, tts_text, prompt_text, prompt_wav, stream, save_prefix=None):
    total_sec = 0.0
    chunks = []
    start = time.perf_counter()
    for out in model.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream):
        wav = out["tts_speech"]
        chunks.append(wav)
        total_sec += wav.shape[1] / model.sample_rate
    elapsed = time.perf_counter() - start
    if save_prefix is not None and chunks:
        save_wav(save_prefix, "zero_shot", torch.cat(chunks, dim=1), model.sample_rate)
    return elapsed, total_sec


def run_zero_shot_cached(model, tts_text, prompt_text, zero_shot_spk_id, stream, save_prefix=None):
    total_sec = 0.0
    chunks = []
    start = time.perf_counter()
    for out in model.inference_zero_shot(tts_text, "", "", zero_shot_spk_id=zero_shot_spk_id, stream=stream):
        # use returned audio to compute duration
        wav = out["tts_speech"]
        chunks.append(wav)
        total_sec += wav.shape[1] / model.sample_rate
    elapsed = time.perf_counter() - start
    if save_prefix is not None and chunks:
        save_wav(save_prefix, "zero_shot_cached", torch.cat(chunks, dim=1), model.sample_rate)
    return elapsed, total_sec


def run_instruct(model, tts_text, instruct_text, prompt_wav, spk_id, stream, save_prefix=None):
    # Ensure required key exists when spk2info was cached via zero_shot.
    total_sec = 0.0
    start = time.perf_counter()
    # Avoid cached prompt_text by not supplying zero_shot_spk_id; use prompt_wav directly.
    iterator = model.inference_instruct2(tts_text, instruct_text, prompt_wav=None, zero_shot_spk_id=spk_id, stream=stream)
    chunks = []
    for out in iterator:
        wav = out["tts_speech"]
        chunks.append(wav)
        total_sec += wav.shape[1] / model.sample_rate
    elapsed = time.perf_counter() - start
    if save_prefix is not None and chunks:
        save_wav(save_prefix, "instruct", torch.cat(chunks, dim=1), model.sample_rate)
    return elapsed, total_sec


def fmt(elapsed, sec):
    rtf = elapsed / sec if sec > 0 else float("nan")
    return f"time={elapsed:.3f}s, audio={sec:.3f}s, rtf={rtf:.3f}"

def _with_prefix(s: str) -> str:
    s = s or ""
    # If user leaves it empty, keep empty (do not auto-prefix).
    if s.strip() == "":
        return ""
    return s if s.startswith(PREFIX) else PREFIX + s


def run(model, tts_text, prompt_text, instruct_text, prompt_wav, spk_id, stream, iters, kind, mode):
    print(f"=== {mode} {kind} ===")
    if mode == "Run": iters = 1
    
    for _ in range(iters):
        if kind == "zero_shot":
            elapsed, sec = run_zero_shot(model, tts_text, prompt_text, prompt_wav, stream=stream,
                                         save_prefix=None if mode == "Warmup" else mode)
        elif kind == "zero_shot_cached":
            elapsed, sec = run_zero_shot_cached(model, tts_text, prompt_text, spk_id, stream=stream,
                                                save_prefix=None if mode == "Warmup" else mode)
        elif kind == "instruct":
            elapsed, sec = run_instruct(model, tts_text, instruct_text, prompt_wav, spk_id, stream=stream,
                                        save_prefix=None if mode == "Warmup" else mode)
    if mode == "Warmup":
        return 
    
    print(fmt(elapsed, sec))

def main():
    parser = argparse.ArgumentParser(description="Compare zero_shot vs sft inference speed.")
    parser.add_argument("--model-dir", required=True, help="Path to CosyVoice model directory.")
    parser.add_argument("--spk-id", default="my_spk3", help="Registered speaker id.")
    parser.add_argument("--prompt-wav", default="./asset/zero_shot_prompt.wav", help="Prompt audio for zero_shot baseline.")
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        help="Prompt text matching prompt_wav for zero_shot baseline.",
    )
    parser.add_argument(
        "--instruct-text",
        default="You are a helpful assistant.<|endofprompt|>",
        help="Instruct text for instruct path.",
    )
    parser.add_argument(
        "--tts-text",
        default="你好，这是一个流式推理速度对比测试，用于验证缓存是否生效。",
        help="Synthesis text to benchmark.",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode.")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Number of untimed warmup runs per path.")
    parser.add_argument("--enable-cache-dit", action="store_true", help="Enable cache-dit DBCache for DiT.")
    parser.add_argument("--dbcache-steps", type=int, default=10, help="DBCache num_inference_steps.")
    parser.add_argument("--dbcache-warmup", type=int, default=8, help="DBCache max_warmup_steps.")
    parser.add_argument("--dbcache-max-cached", type=int, default=-1, help="DBCache max_cached_steps.")
    parser.add_argument("--dbcache-fn", type=int, default=8, help="DBCache Fn_compute_blocks.")
    parser.add_argument("--dbcache-bn", type=int, default=0, help="DBCache Bn_compute_blocks.")
    parser.add_argument("--dbcache-rdt", type=float, default=0.12, help="DBCache residual_diff_threshold.")
    args = parser.parse_args()

    cosy = AutoModel(model_dir=args.model_dir)
    if args.enable_cache_dit:
        cosy.model.enable_cache_dit(
            {
                "num_inference_steps": args.dbcache_steps,
                "max_warmup_steps": args.dbcache_warmup,
                "max_cached_steps": args.dbcache_max_cached,
                "Fn_compute_blocks": args.dbcache_fn,
                "Bn_compute_blocks": args.dbcache_bn,
                "residual_diff_threshold": args.dbcache_rdt,
            }
        )

    prompt_text = _with_prefix(args.prompt_text)
    instruct_text = _with_prefix(args.instruct_text)
    tts_text = args.tts_text
    
    for kind in ["zero_shot", "zero_shot_cached", "instruct"]:
        run(cosy, tts_text, prompt_text, instruct_text, args.prompt_wav, args.spk_id, args.stream, args.warmup_iters, kind, "Warmup")
        run(cosy, tts_text, prompt_text, instruct_text, args.prompt_wav, args.spk_id, args.stream, args.warmup_iters, kind, "Run")


if __name__ == "__main__":
    main()
