"""
python cosy_server.py --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

使用批量推理模式
/home/wjs/workspace/miniconda3/envs/cosyvoice/bin/python myscripts/cosy_server.py       \
    --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512       \
    --batch-infer \
    --input-lst /home/wjs/workspace/data/CowboyZ/seed-tts-eval/seedtts_testset/zh/meta.lst       \
    --tokens-dir /home/wjs/workspace/data/seedtts_tokens       \
    --output-dir /cfs-czb184s7/jasonjswang/fn1bn0warmup0diff0.4       


/home/wjs/workspace/miniconda3/envs/cosyvoice/bin/python myscripts/cosy_server.py \
    --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --batch-infer \
    --input-lst /home/wjs/workspace/data/CowboyZ/seed-tts-eval/seedtts_testset/zh/meta.lst \
    --tokens-dir /home/wjs/workspace/data/seedtts_tokens \
    --num-workers 10

"""


from __future__ import annotations

import argparse
import os
import sys
import json
import msgpack
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
import time
from tqdm import tqdm
from datetime import timedelta

# 添加ZeroMQ支持
try:
    import zmq
except ImportError:
    print("[ERROR] ZeroMQ not installed. Please run 'pip install pyzmq' to enable ZeroMQ support.")
    sys.exit(1)

# 确保zmq可用
if zmq is None:
    print("[ERROR] ZeroMQ not available.")
    sys.exit(1)

import numpy as np
import torch

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "third_party", "Matcha-TTS"))
sys.path.append(current_dir)

from token2wav_dit import CosyVoice2_Token2Wav, load_tokens_from_file

class CosyVoiceRuntime:
    def __init__(self, model_dir: str, device_id: int, dtype: torch.dtype):
        self.model = CosyVoice2_Token2Wav(model_dir=model_dir, device_id=device_id, dtype=dtype)
        self.device = self.model.device
        self.dtype = dtype
        self.sample_rate = 24000

    def _autocast(self):
        if self.device.startswith("cuda"):
            return torch.amp.autocast("cuda", dtype=self.dtype)
        return nullcontext()

    def prepare_dit_inputs(self, tokens: List[int], prompt_wav_path: str) -> Dict[str, Any]:
        """完整预处理，输出 DiT 所需的 condition_vector, speaker_embedding 和 cond"""
        if not os.path.isfile(prompt_wav_path):
            raise FileNotFoundError(f"Prompt audio not found: {prompt_wav_path}")

        token_tensor = torch.tensor([tokens], dtype=torch.int32).to(self.device)

        with torch.inference_mode():
            with self._autocast():
                # Step 1: 提取 prompt 特征
                prompt_token, prompt_feat, embedding = self.model.extract_prompt_features(prompt_wav_path)

                # 确保prompt_token是二维张量，并移到正确设备
                if prompt_token.ndim == 1:
                    prompt_token = prompt_token.unsqueeze(0)
                prompt_token = prompt_token.to(self.device)
                prompt_feat = prompt_feat.to(self.device)
                embedding = embedding.to(self.device)

                # Step 2: 完成 flow 的预处理
                from torch.nn import functional as F
                from cosyvoice.utils.mask import make_pad_mask

                flow = self.model.flow

                # Embedding projection
                embedding_proj = F.normalize(embedding, dim=1)
                embedding_proj = flow.spk_embed_affine_layer(embedding_proj)

                # Concat speech token and prompt speech token
                token_len1, token_len2 = prompt_token.shape[1], token_tensor.shape[1]
                token_concat = torch.concat([prompt_token, token_tensor], dim=1)
                token_len_total = torch.tensor([prompt_token.shape[1] + token_tensor.shape[1]], dtype=torch.int32).to(self.device)

                mask = (~make_pad_mask(token_len_total)).unsqueeze(-1).to(self.device)
                token_embed = flow.input_embedding(torch.clamp(token_concat, min=0)) * mask

                # 根据 flow 类型选择处理方式
                if hasattr(flow, 'pre_lookahead_layer'):
                    # CausalMaskedDiffWithDiT: use pre_lookahead_layer
                    h = flow.pre_lookahead_layer(token_embed)
                    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
                    mel_len1 = prompt_feat.shape[1]
                    mel_len2 = h.shape[1] - prompt_feat.shape[1]
                else:
                    # MaskedDiffWithXvec or CausalMaskedDiffWithXvec: use encoder + length_regulator
                    h, _ = flow.encoder(token_embed, token_len_total)
                    h = flow.encoder_proj(h)
                    mel_len1 = prompt_feat.shape[1]
                    mel_len2 = int(token_len2 / flow.input_frame_rate * 22050 / 256)
                    h, _ = flow.length_regulator.inference(
                        h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, flow.input_frame_rate
                    )

                # h 就是 condition_vector (mu)
                condition_vector = h  # [1, mel_len1 + mel_len2, 80]

                # 准备 cond：前面 mel_len1 帧是 prompt_feat，后面是 0
                output_size = flow.output_size  # 通常是 80
                conds = torch.zeros([1, mel_len1 + mel_len2, output_size], device=self.device).to(h.dtype)
                conds[:, :mel_len1] = prompt_feat
                # transpose 为 [batch, dim, seq] 格式
                conds_transposed = conds.transpose(1, 2)

                # Speaker embedding - 保持为 [batch, spk_dim]，不要扩展!
                speaker_embedding = embedding_proj  # [1, 192]

        return {
            "condition_vector": condition_vector.transpose(1, 2).to(dtype=torch.float32).cpu().tolist(),  # [batch, dim, seq]
            "speaker_embedding": speaker_embedding.to(dtype=torch.float32).cpu().tolist(),  # [batch, spk_dim]
            "cond": conds_transposed.to(dtype=torch.float32).cpu().tolist(),  # [batch, dim, seq]
            "seq_len": condition_vector.shape[1],
            "mel_len1": mel_len1,  # 用于后处理时切分
            "mel_len2": mel_len2,
            "batch_size": 1,
        }

    def dit_output_to_wav(self, mel: List[List[List[float]]], mel_dtype: Optional[str] = None) -> Dict[str, Any]:
        dtype = _parse_dtype(mel_dtype, torch.float32)
        
        # 检查并处理 mel 中的 None 值
        # 将 None 替换为 0.0，避免 torch.tensor() 转换失败
        mel_clean = [[[0.0 if x is None else x for x in subsublist] for subsublist in sublist] for sublist in mel]
        
        tts_mel = torch.tensor(mel_clean, dtype=dtype)

        with torch.inference_mode():
            tts_mel = tts_mel.to(self.device)
            with self._autocast():
                wav = self.model.forward_hift(tts_mel)

        if wav.ndim == 2:
            audio = wav[0].detach().cpu().tolist()
        else:
            audio = wav.detach().cpu().flatten().tolist()

        return {"audio": audio, "sample_rate": self.sample_rate}


runtime: Optional[CosyVoiceRuntime] = None


def _parse_dtype(name: Optional[str], default: torch.dtype) -> torch.dtype:
    if not name:
        return default
    lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return lookup.get(name.lower(), default)


def load_tokens_from_file(token_file):
    """从文件加载预保存的 tokens

    Args:
        token_file: token 文件路径

    Returns:
        list[int]: token 列表
    """
    with open(token_file, 'r') as f:
        tokens_str = f.read().strip()

    # 解析 tokens (空格分隔的整数)
    tokens = [int(t) for t in tokens_str.split()]

    return tokens


class LocalToken2WavDataset(Dataset):
    """本地文件的Token2Wav数据集类"""
    def __init__(self, input_lst, tokens_dir, sample_rate=16000):
        self.tokens_dir = tokens_dir
        self.sample_rate = sample_rate
        self.data_items = []
        
        # 读取并解析输入列表
        with open(input_lst, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
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
                print(f"Warning: Skipping line with unexpected format: {line}")
                continue
            
            # 处理 utt（去掉 .wav 后缀）
            if utt.endswith('.wav'):
                utt = utt[:-4]
            
            
            # 处理 tokens_dir 路径（加入路径前缀）
            token_file = os.path.join(tokens_dir, f"{utt}.txt")

            # 检查 token 文件是否存在
            if not os.path.exists(token_file):
                print(f"Warning: Token file not found for {utt}: {token_file}")
                continue
            
            # 处理 prompt_wav 路径（加入路径前缀）
            if prompt_wav and not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(os.path.dirname(input_lst), prompt_wav)
            
            # 检查 prompt_wav 是否存在
            if not prompt_wav or not os.path.exists(prompt_wav):
                print(f"Warning: Prompt audio not found for {utt}: {prompt_wav}")
                continue
            
            self.data_items.append({
                'utt': utt,
                'prompt_wav': prompt_wav,
                'token_file': token_file
            })
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # 加载 tokens
        target_audio_cosy2_tokens = load_tokens_from_file(item['token_file'])
        
        return {
            'id': item['utt'],
            'target_audio_cosy2_tokens': target_audio_cosy2_tokens,
            'prompt_wav': item['prompt_wav']
        }


def _require_runtime() -> CosyVoiceRuntime:
    if runtime is None:
        raise RuntimeError("Runtime is not initialized")
    return runtime


def _setup_runtime(model_dir: str, device_id: int, dtype_name: str):
    global runtime  # noqa: PLW0603
    dtype = _parse_dtype(dtype_name, torch.float16)
    runtime = CosyVoiceRuntime(model_dir=model_dir, device_id=device_id, dtype=dtype)
    print(f"[cosy_server] Loaded CosyVoice model from {model_dir} on device {runtime.device}")


def _bootstrap_from_env():
    model_dir = os.getenv("COSY_SERVER_MODEL_DIR")
    if not model_dir:
        return
    device_id = int(os.getenv("COSY_SERVER_DEVICE_ID", "0"))
    dtype_name = os.getenv("COSY_SERVER_DTYPE", "float16")
    if runtime is None:
        _setup_runtime(model_dir, device_id, dtype_name)


def _run_batch_inference(
    model_dir: str,
    device_id: int,
    dtype_name: str,
    input_lst: str,
    tokens_dir: str,
    output_dir: str,
    warmup: int,
    total_size: Optional[int] = None,
    sample_rate: int = 16000,
    zmq_address: str = "ipc:///tmp/vllm.sock",
    num_workers: int = 1,
):
    """执行批量推理
    
    Args:
        model_dir: CosyVoice模型目录
        device_id: CUDA设备ID
        dtype_name: 计算数据类型
        input_lst: 输入列表文件路径
        tokens_dir: token文件目录
        output_dir: 输出目录
        warmup: 预热轮数
        total_size: 要处理的样本总数（默认处理所有样本）
        sample_rate: 输入提示音频的采样率
        zmq_address: ZeroMQ服务端地址
        num_workers: 并发线程数（每线程维护自己的ZeroMQ socket）
    """
    # 设置运行时
    _setup_runtime(model_dir, device_id, dtype_name)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载数据集
    print("\nLoading dataset...")
    dataset = LocalToken2WavDataset(
        input_lst=input_lst,
        tokens_dir=tokens_dir,
        sample_rate=sample_rate
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # 如果指定了total_size，则裁剪数据集
    if total_size is not None and total_size > 0:
        if total_size < len(dataset):
            dataset.data_items = dataset.data_items[:total_size]
            print(f"Dataset cropped to {len(dataset)} samples (requested: {total_size})")
        else:
            print(f"Total size {total_size} is larger than dataset size {len(dataset)}, processing all samples")
    
    items = dataset.data_items
    print(f"\nStarting batch inference with {warmup} warmup epoch(s)... (workers={num_workers})")
    print("="*80)

    def _process_single(item):
        utt_id = item["utt"]
        sock = None
        try:
            tokens = load_tokens_from_file(item["token_file"])
            dit_input = runtime.prepare_dit_inputs(tokens, item["prompt_wav"])

            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(zmq_address)

            serialized_data = msgpack.packb(dit_input, use_bin_type=True)
            sock.send(serialized_data)

            response = sock.recv()
            vllm_output = msgpack.unpackb(response, raw=False)

            if "error" in vllm_output:
                raise RuntimeError(f"vLLM error: {vllm_output['error']}")

            mel = np.frombuffer(vllm_output["data"], dtype=vllm_output["dtype"]).reshape(vllm_output["shape"])
            runtime.dit_output_to_wav(mel, str(mel.dtype))
            return True
        except Exception as e:
            print(f"\n[ERROR] Failed to process {utt_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if sock is not None:
                sock.close()

    for epoch in range(warmup):
        print(f"\nEpoch {epoch + 1}/{warmup}")
        start_time = time.time()

        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_single, item) for item in items]
            with tqdm(total=len(futures), desc=f"Epoch {epoch + 1}", unit="sample") as pbar:
                for fut in as_completed(futures):
                    ok = fut.result()
                    if ok:
                        success_count += 1
                    else:
                        error_count += 1
                    pbar.update(1)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"\nEpoch {epoch + 1} completed!")
        print(f"  Duration:          {timedelta(seconds=int(epoch_time))}")
        print(f"  Successful:        {success_count}/{success_count + error_count}")
        if success_count > 0:
            print(f"  Avg time/sample:   {epoch_time/success_count:.3f}s")
            print(f"  Throughput:        {success_count/epoch_time:.2f} samples/s")
    
    print("\n" + "="*80)
    print("BATCH INFERENCE COMPLETED!")
    print("="*80)
    print(f"Output directory:    {output_dir}")
    print(f"Total samples:       {len(dataset)}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="CosyVoice batch inference client")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to CosyVoice checkpoint directory")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--dtype", type=str, default="float16", help="Computation dtype (float16, bfloat16, float32)")
    
    # 批量推理参数
    parser.add_argument("--batch-infer", action="store_true", help="Run batch inference")
    parser.add_argument("--input-lst", type=str, help="Input meta list file path")
    parser.add_argument("--tokens-dir", type=str, help="Directory containing pre-saved token files")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--total-size", type=int, help="Total number of samples to process (default: process all)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for input prompt audio (default: 16000)")
    parser.add_argument("--num-workers", type=int, default=1, help="并发线程数（客户端侧）")
    # ZeroMQ参数
    parser.add_argument("--zmq-address", type=str, default="ipc:///tmp/vllm.sock", help="ZeroMQ服务端地址")
    
    args = parser.parse_args()
    
    if args.batch_infer:
        # 检查批量推理所需参数
        if not args.input_lst or not args.tokens_dir:
            parser.error("--input-lst and --tokens-dir are required for batch inference")
        
        # 执行批量推理
        _run_batch_inference(
            model_dir=args.model_dir,
            device_id=args.device_id,
            dtype_name=args.dtype,
            input_lst=args.input_lst,
            tokens_dir=args.tokens_dir,
            output_dir=args.output_dir,
            warmup=args.warmup,
            total_size=args.total_size,
            sample_rate=args.sample_rate,
            zmq_address=args.zmq_address,
            num_workers=args.num_workers,
        )
    else:
        # 显示帮助信息
        parser.print_help()
        print("\nNote: This script now only supports batch inference mode with ZeroMQ communication.")
        print("Use --batch-infer to run batch inference.")


_bootstrap_from_env()

if __name__ == "__main__":
    main()
