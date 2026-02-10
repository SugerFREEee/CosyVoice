"""
/home/wjs/workspace/miniconda3/envs/dev/bin/python myscripts/batch_generate.py \
    -i /home/wjs/workspace/data/CowboyZ/seed-tts-eval/seedtts_testset/zh/meta.lst \
    -o /cfs-czb184s7/jasonjswang/aggressive\
    --enable-cache-dit \
    --dbcache-steps 10 \
    --dbcache-warmup 0 \
    --dbcache-max-cached -1 \
    --dbcache-fn 2 \
    --dbcache-bn 0 \
    --dbcache-rdt 0.3
"""


import sys
import os
import argparse
import torch
import torchaudio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel


def batch_generate(
    input_lst,
    output_dir,
    model_dir='pretrained_models/CosyVoice3-0.5B',
    instruction='You are a helpful assistant.',
    stream=False,
    enable_cache_dit=False,
    dbcache_steps=10,
    dbcache_warmup=8,
    dbcache_max_cached=-1,
    dbcache_fn=8,
    dbcache_bn=0,
    dbcache_rdt=0.12
):
    """
    Batch generate audio files from a meta list file (seedtts format).

    Args:
        input_lst (str): Path to input meta list file
            Supported formats:
            - Format 1: utt|prompt_text|prompt_wav|infer_text|infer_wav (5 columns)
            - Format 2: utt|prompt_text|prompt_wav|infer_text (4 columns)
            - Format 3: utt|infer_text|prompt_wav (3 columns)
            - Format 4: utt|infer_text (2 columns)
        output_dir (str): Directory to save generated audio files
        model_dir (str): Path to model directory
        instruction (str): Instruction prefix for prompt content
        stream (bool): Use streaming mode
        enable_cache_dit (bool): Enable cache-dit DBCache
        dbcache_steps (int): DBCache num_inference_steps
        dbcache_warmup (int): DBCache max_warmup_steps
        dbcache_max_cached (int): DBCache max_cached_steps
        dbcache_fn (int): DBCache Fn_compute_blocks
        dbcache_bn (int): DBCache Bn_compute_blocks
        dbcache_rdt (float): DBCache residual_diff_threshold
    """

    # Create output directory if not exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {model_dir}...")
    cosyvoice = AutoModel(model_dir=model_dir)
    if enable_cache_dit:
        cosyvoice.model.enable_cache_dit(
            {
                "num_inference_steps": dbcache_steps,
                "max_warmup_steps": dbcache_warmup,
                "max_cached_steps": dbcache_max_cached,
                "Fn_compute_blocks": dbcache_fn,
                "Bn_compute_blocks": dbcache_bn,
                "residual_diff_threshold": dbcache_rdt,
            }
        )
    # return
    mode_desc = 'streaming' if stream else 'non-streaming'
    print(f"Model loaded. Running in {mode_desc} mode...")
    
    # Read meta list file
    print(f"Reading meta list from {input_lst}...")
    if not os.path.exists(input_lst):
        print(f"Error: Input list file not found: {input_lst}")
        return

    with open(input_lst, 'r', encoding='utf-8') as f:
        lines = f.readlines()
     
    # Parse all lines and prepare data items
    print(f"Parsing {len(lines)} samples...")
    data_items = []
    for index, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        # Parse different meta list formats
        parts = line.split('|')

        if len(parts) == 5:
            utt, prompt_text, prompt_wav, infer_text, infer_wav = parts
        elif len(parts) == 4:
            utt, prompt_text, prompt_wav, infer_text = parts
        elif len(parts) == 3:
            utt, infer_text, prompt_wav = parts
            prompt_text = ""  # Empty prompt text
            # Handle case where utt ends with .wav
            if utt.endswith(".wav"):
                utt = utt[:-4]
        elif len(parts) == 2:
            utt, infer_text = parts
            prompt_text = ""
            prompt_wav = ""
        else:
            print(f"Warning: Skipping line {index} with unexpected format: {line}")
            continue

        # Handle relative paths for prompt_wav
        if prompt_wav and not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(input_lst), prompt_wav)

        # Check if prompt_wav exists
        if prompt_wav and not os.path.exists(prompt_wav):
            print(f"Warning: Skipping line {index}, prompt_wav not found: {prompt_wav}")
            continue

        data_items.append((utt, prompt_text, prompt_wav, infer_text))

    total_samples = len(data_items)
    print(f"Found {total_samples} valid samples to process.")

    if total_samples == 0:
        print("No valid samples to process!")
        return
     
    # Process each item
    success_count = 0
    for idx, (utt, prompt_text, prompt_wav, infer_text) in enumerate(data_items, start=1):
        print(f"\n[{idx}/{total_samples}] Processing: {utt}")
        print(f"  Text: {infer_text[:50]}{'...' if len(infer_text) > 50 else ''}")

        # Format prompt_content with instruction prefix
        if prompt_text:
            prompt_content = f"{instruction}<|endofprompt|>{prompt_text}"
        else:
            prompt_content = f"{instruction}<|endofprompt|>"

        # Generate output file path
        output_file = output_path / f"{utt}.wav"

        try:
            # Generate audio
            audio_chunks = []
            for i, j in enumerate(cosyvoice.inference_zero_shot(
                infer_text,
                prompt_content,
                prompt_wav,
                stream=stream
            )):
                audio_chunks.append(j['tts_speech'])

            # Concatenate and save
            audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(str(output_file), audio, cosyvoice.sample_rate)
            print(f"  ✓ Saved to: {output_file}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error processing {utt}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"Total samples: {total_samples}")
    print(f"Successfully generated: {success_count}")
    print(f"Failed: {total_samples - success_count}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    """
    Main function for batch audio generation using seedtts meta list format.

    Usage examples:
        # Basic usage
        python batch_generate.py --input meta.lst --output ./output

        # With streaming mode
        python batch_generate.py --input meta.lst --output ./output --stream

        # With progressive prompt (requires streaming)
        python batch_generate.py --input meta.lst --output ./output --stream --use-progressive-prompt

        # Custom instruction prefix
        python batch_generate.py --input meta.lst --output ./output --instruction "You are a helpful assistant."

        # With TensorRT
        python batch_generate.py --input meta.lst --output ./output --trt

    Meta list format (pipe-separated):
        Format 1: utt|prompt_text|prompt_wav|infer_text|infer_wav (5 columns)
        Format 2: utt|prompt_text|prompt_wav|infer_text (4 columns)
        Format 3: utt|infer_text|prompt_wav (3 columns)
        Format 4: utt|infer_text (2 columns)
    """

    parser = argparse.ArgumentParser(
        description='Batch generate audio files from meta list file (seedtts format) using CosyVoice3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__
    )

    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input meta list file (seedtts format, pipe-separated)')
    parser.add_argument('--output', '-o', required=True,
                       help='Directory to save generated audio files')

    # Model arguments
    parser.add_argument('--model-dir', default='/home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                       help='Path to model directory (default: FunAudioLLM/Fun-CosyVoice3-0.5B-2512)')
    parser.add_argument('--instruction', default='You are a helpful assistant.',
                       help='Instruction prefix for prompt content (default: "You are a helpful assistant.")')

    # Inference mode arguments
    parser.add_argument('--stream', action='store_true',
                       help='Use streaming mode for chunk-by-chunk inference')
    parser.add_argument('--enable-cache-dit', action='store_true',
                       help='Enable cache-dit DBCache for DiT')
    parser.add_argument('--dbcache-steps', type=int, default=10,
                       help='DBCache num_inference_steps')
    parser.add_argument('--dbcache-warmup', type=int, default=8,
                       help='DBCache max_warmup_steps')
    parser.add_argument('--dbcache-max-cached', type=int, default=-1,
                       help='DBCache max_cached_steps')
    parser.add_argument('--dbcache-fn', type=int, default=8,
                       help='DBCache Fn_compute_blocks')
    parser.add_argument('--dbcache-bn', type=int, default=0,
                       help='DBCache Bn_compute_blocks')
    parser.add_argument('--dbcache-rdt', type=float, default=0.12,
                       help='DBCache residual_diff_threshold')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run batch generation
    batch_generate(
        input_lst=args.input,
        output_dir=args.output,
        model_dir=args.model_dir,
        instruction=args.instruction,
        stream=args.stream,
        enable_cache_dit=args.enable_cache_dit,
        dbcache_steps=args.dbcache_steps,
        dbcache_warmup=args.dbcache_warmup,
        dbcache_max_cached=args.dbcache_max_cached,
        dbcache_fn=args.dbcache_fn,
        dbcache_bn=args.dbcache_bn,
        dbcache_rdt=args.dbcache_rdt
    )


if __name__ == '__main__':
    main()
