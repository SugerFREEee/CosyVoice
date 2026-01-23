#!/usr/bin/env python3
"""
CosyVoice3 SeedTTS Inference with Pre-saved Tokens
跳过 LLM 推理，直接使用预保存的 token 进行 Flow + HiFT 推理
"""

import sys
import os
# 添加项目根目录到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/wjs/workspace/CosyVoice/third_party/Matcha-TTS')
import argparse
import torch
import torchaudio
import numpy as np
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import uuid as uuid_lib


def load_tokens_from_file(token_file):
    """从文件加载预保存的 tokens

    Args:
        token_file: token 文件路径

    Returns:
        torch.Tensor: shape (1, seq_len) 的 token tensor
    """
    with open(token_file, 'r') as f:
        tokens_str = f.read().strip()

    # 解析 tokens (空格分隔的整数)
    tokens = [int(t) for t in tokens_str.split()]

    # 转换为 tensor (1, seq_len)
    token_tensor = torch.tensor([tokens], dtype=torch.int32)

    return token_tensor


def inference_with_preloaded_tokens(model, token_tensor, prompt_wav, prompt_text,
                                   output_file, sample_rate, stream=False, speed=1.0):
    """使用预加载的 token 进行推理（跳过 LLM）

    Args:
        model: CosyVoice3 模型实例
        token_tensor: 预加载的 token tensor (1, seq_len)
        prompt_wav: prompt 音频路径
        prompt_text: prompt 文本（包含 instruction）
        output_file: 输出音频文件路径
        sample_rate: 采样率
        stream: 是否使用流式推理
        speed: 语速
    """
    # 使用 frontend 提取 prompt 相关特征
    frontend = model.frontend

    # 提取 prompt_text token
    if prompt_text:
        prompt_text_token, prompt_text_token_len = frontend._extract_text_token(prompt_text)
    else:
        # 如果没有 prompt_text，使用空 token
        prompt_text_token = torch.zeros((1, 0), dtype=torch.int32)
        prompt_text_token_len = torch.tensor([0], dtype=torch.int32)

    # 提取 speech token, speech feat 和 embedding
    speech_feat, speech_feat_len = frontend._extract_speech_feat(prompt_wav)
    speech_token, speech_token_len = frontend._extract_speech_token(prompt_wav)
    embedding = frontend._extract_spk_embedding(prompt_wav)

    # 准备推理
    device = model.model.device
    uuid_str = str(uuid_lib.uuid4())

    # 初始化缓存
    model.model.hift_cache_dict[uuid_str] = None

    if stream:
        # 流式推理
        audio_chunks = []
        token_hop_len = model.model.token_max_hop_len

        for i in range(0, token_tensor.shape[1], token_hop_len):
            token_chunk = token_tensor[:, i:i + token_hop_len]
            finalize = (i + token_hop_len >= token_tensor.shape[1])

            # 调用 token2wav 进行 Flow + HiFT 推理
            tts_speech = model.model.token2wav(
                token=token_chunk,
                prompt_token=speech_token,
                prompt_feat=speech_feat,
                embedding=embedding,
                token_offset=0,
                uuid=uuid_str,
                stream=True,
                finalize=finalize,
                speed=speed
            )

            audio_chunks.append(tts_speech.cpu())

        # 拼接所有音频块
        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=1)
            torchaudio.save(output_file, full_audio, sample_rate)
    else:
        # 非流式推理
        tts_speech = model.model.token2wav(
            token=token_tensor,
            prompt_token=speech_token,
            prompt_feat=speech_feat,
            embedding=embedding,
            token_offset=0,
            uuid=uuid_str,
            stream=False,
            finalize=True,
            speed=speed
        )

        # torchaudio.save(output_file, tts_speech.cpu(), sample_rate)

    # 清理缓存
    if uuid_str in model.model.hift_cache_dict:
        del model.model.hift_cache_dict[uuid_str]


def main():
    parser = argparse.ArgumentParser(description='CosyVoice3 SeedTTS Inference with Pre-saved Tokens')
    parser.add_argument(
        '-i', '--input_lst',
        type=str,
        default="/home/wjs/workspace/data/CowboyZ/seed-tts-eval/seedtts_testset/zh/meta.lst",
        help='Input meta list file path (format: utt|prompt_text|prompt_wav|infer_text)')
    parser.add_argument(
        '-t', '--tokens_dir',
        type=str,
        default='/home/wjs/workspace/data/seedtts_tokens',
        help='Directory containing pre-saved token files')
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='/home/wjs/workspace/data/new_seedtts_output_with_tokens',
        help='Output directory for generated audio files')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
        help='Path to CosyVoice3 model directory')
    parser.add_argument(
        '--instruction',
        type=str,
        default='You are a helpful assistant.',
        help='Instruction prefix for prompt content')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=22050,
        help='Sample rate for output audio (default: 22050 for CosyVoice3)')
    parser.add_argument(
        '--stream',
        type=int,
        default=0,
        help='Use streaming inference (1) or non-streaming (0), default: 0')
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speech speed (default: 1.0)')
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)')

    args = parser.parse_args()

    # 设置 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {args.gpu}")
    else:
        print("Warning: CUDA not available, using CPU")

    # 读取 meta list
    if not os.path.exists(args.input_lst):
        print(f"Error: Input list file not found: {args.input_lst}")
        return

    with open(args.input_lst, 'r') as f:
        lines = f.readlines()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # 加载模型
    print(f"Loading CosyVoice3 model from: {args.model_dir}")
    cosyvoice = AutoModel(model_dir=args.model_dir)
    print("Model loaded successfully!")

    # 解析数据
    print(f"\nParsing {len(lines)} samples...")
    data_items = []
    missing_tokens = []

    for index, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        # 解析不同格式
        parts = line.split('|')

        if len(parts) == 5:
            utt, prompt_text, prompt_wav, infer_text, infer_wav = parts
        elif len(parts) == 4:
            utt, prompt_text, prompt_wav, infer_text = parts
        elif len(parts) == 3:
            utt, infer_text, prompt_wav = parts
            prompt_text = ""
        elif len(parts) == 2:
            utt, infer_text = parts
            prompt_text = ""
            prompt_wav = ""
        else:
            print(f"Warning: Skipping line {index} with unexpected format")
            continue

        # 处理 utt（去掉 .wav 后缀）
        if utt.endswith('.wav'):
            utt = utt[:-4]

        # 检查 token 文件是否存在
        token_file = os.path.join(args.tokens_dir, f"{utt}.txt")
        if not os.path.exists(token_file):
            missing_tokens.append(utt)
            continue

        # 处理 prompt_wav 路径
        if prompt_wav and not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(args.input_lst), prompt_wav)

        if not prompt_wav or not os.path.exists(prompt_wav):
            print(f"Warning: Skipping {utt}, prompt_wav not found: {prompt_wav}")
            continue

        data_items.append({
            'utt': utt,
            'prompt_text': prompt_text,
            'prompt_wav': prompt_wav,
            'infer_text': infer_text,
            'token_file': token_file
        })

    print(f"\nTotal valid samples: {len(data_items)}")

    if missing_tokens:
        print(f"\nWarning: {len(missing_tokens)} samples have missing token files:")
        for utt in missing_tokens[:10]:  # 只显示前 10 个
            print(f"  - {utt}")
        if len(missing_tokens) > 10:
            print(f"  ... and {len(missing_tokens) - 10} more")

    if len(data_items) == 0:
        print("\nNo valid samples to process!")
        return

    # 开始推理
    print("\n" + "="*80)
    print("Starting inference with pre-saved tokens...")
    print("="*80)
    start_time = time.time()
    start_datetime = datetime.now()

    success_count = 0
    error_count = 0

    for idx, item in enumerate(tqdm(data_items, desc="Processing"), 1):
        try:
            # 加载 tokens
            token_tensor = load_tokens_from_file(item['token_file'])

            # 构建 prompt_content（带 instruction）
            if item['prompt_text']:
                prompt_content = f"{args.instruction}<|endofprompt|>{item['prompt_text']}"
            else:
                prompt_content = f"{args.instruction}<|endofprompt|>"

            # 输出文件路径
            output_file = os.path.join(args.output_dir, f"{item['utt']}.wav")

            # 使用预加载的 token 进行推理
            inference_with_preloaded_tokens(
                model=cosyvoice,
                token_tensor=token_tensor,
                prompt_wav=item['prompt_wav'],
                prompt_text=prompt_content,
                output_file=output_file,
                sample_rate=args.sample_rate,
                stream=bool(args.stream),
                speed=args.speed
            )

            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError processing {item['utt']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 统计信息
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time

    print("\n" + "="*80)
    print("INFERENCE COMPLETED!")
    print("="*80)
    print(f"Start time:              {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:                {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration:          {timedelta(seconds=int(total_duration))}")
    print(f"Total samples:           {len(data_items)}")
    print(f"Successful:              {success_count}")
    print(f"Failed:                  {error_count}")
    print(f"Success rate:            {success_count/len(data_items)*100:.1f}%" if len(data_items) > 0 else "N/A")
    if success_count > 0:
        print(f"Avg time per sample:     {total_duration/success_count:.2f}s")
        print(f"Throughput:              {success_count/total_duration:.2f} samples/s")
    print(f"Output directory:        {args.output_dir}")
    print(f"Using pre-saved tokens:  {args.tokens_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
