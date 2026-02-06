#!/usr/bin/env python
# Register a zero-shot speaker and save spk2info.pt for later fast inference.
"""
python myscripts/register.py --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --spk-id my_spk4 \
  --prompt-wav /home/wjs/workspace/CosyVoice/output_compare/Run_zero_shot.wav \
  --prompt-text "You are a helpful assistant.<|endofprompt|>祥子找到外祖父定治，却被告知，父亲遭遇了诈骗并造成了168亿日元的损失，已“不再是丰川家的人”。"
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root on sys.path when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosyvoice.cli.cosyvoice import AutoModel


def main():
    parser = argparse.ArgumentParser(description="Register (cache) a zero-shot speaker for CosyVoice.")
    parser.add_argument("--model-dir", required=True, help="Path to CosyVoice model directory (contains model files).")
    parser.add_argument("--prompt-wav", default="./asset/zero_shot_prompt.wav", help="Reference audio for the target speaker.")
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        help="Transcription or prompt text that matches the reference audio.",
    )
    parser.add_argument("--spk-id", default="my_spk3", help="Identifier used later in inference_sft/inference_zero_shot.")
    args = parser.parse_args()

    cosy = AutoModel(model_dir=args.model_dir)
    ok = cosy.add_zero_shot_spk(args.prompt_text, args.prompt_wav, args.spk_id)
    # For CosyVoice3, frontend_sft expects key 'embedding'. Map from llm_embedding if missing.
    info = cosy.frontend.spk2info.get(args.spk_id, {})
    if "embedding" not in info and "llm_embedding" in info:
        info["embedding"] = info["llm_embedding"]
        cosy.frontend.spk2info[args.spk_id] = info
    cosy.save_spkinfo()
    status = "success" if ok else "failed"
    print(f"[register] {status}, spk_id={args.spk_id}, spk2info saved to {args.model_dir}/spk2info.pt")


if __name__ == "__main__":
    main()
