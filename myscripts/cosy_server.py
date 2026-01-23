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
    --tokens-dir /home/wjs/workspace/data/seedtts_tokens
"""


from __future__ import annotations

import argparse
import os
import sys
import json
import msgpack
from contextlib import nullcontext
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader, Dataset
import torchaudio
import time
from tqdm import tqdm
from datetime import datetime, timedelta

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


def collate_fn(batch):
    ids, tokens_list, prompt_wavs_list = [], [], []
    for item in batch:
        tokens_list.append(item['target_audio_cosy2_tokens'])
        prompt_wavs_list.append(item['prompt_wav'])
        ids.append(item['id'])

    return ids, tokens_list, prompt_wavs_list


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
    batch_size: int,
    warmup: int,
    total_size: Optional[int] = None,
    sample_rate: int = 16000,
    zmq_address: str = "ipc:///tmp/vllm.sock"
):
    """执行批量推理
    
    Args:
        model_dir: CosyVoice模型目录
        device_id: CUDA设备ID
        dtype_name: 计算数据类型
        input_lst: 输入列表文件路径
        tokens_dir: token文件目录
        output_dir: 输出目录
        batch_size: 批次大小
        warmup: 预热轮数
        total_size: 要处理的样本总数（默认处理所有样本）
        sample_rate: 输入提示音频的采样率
        zmq_address: ZeroMQ服务端地址
    """
    # 初始化ZeroMQ客户端
    zmq_socket = None
    try:
        context = zmq.Context()
        zmq_socket = context.socket(zmq.DEALER)  # DEALER模式支持多客户端
        zmq_socket.connect(zmq_address)
        print(f"[cosy_server] ZeroMQ client connected to {zmq_address}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to ZeroMQ server: {e}")
        return
    
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
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\nStarting batch inference with {warmup} warmup epoch(s)...")
    print("="*80)
    
    for epoch in range(warmup):
        print(f"\nEpoch {epoch + 1}/{warmup}")
        start_time = time.time()
        start_datetime = datetime.now()
        
        success_count = 0
        error_count = 0
        
        # 用于统计通信开销
        total_comm_time = 0.0
        comm_count = 0
        
        # 使用 tqdm 显示进度条
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}", unit="batch")):
            batch_start_time = time.time()
            batch_success = 0
            batch_error = 0
            
            try:
                ids, tokens_list, prompt_wavs_list = batch
                
                for utt_id, tokens, prompt_wav in zip(ids, tokens_list, prompt_wavs_list):
                    # 检查是否达到了指定的总处理数量
                    if total_size is not None and (success_count + error_count) >= total_size:
                        print(f"\nReached total size limit: {total_size}. Stopping inference.")
                        # 跳出内层循环
                        break
                    
                    try:
                        # Step 1: 获取 DiT 输入特征
                        # 直接调用 runtime 方法，不通过 HTTP API
                        # print(f"[DEBUG] Processing {utt_id}...")
                        dit_input = runtime.prepare_dit_inputs(tokens, prompt_wav)
                        # print(f"[DEBUG]   DiT input prepared, seq_len={dit_input['seq_len']}")

                        # Step 2: 调用 vllm_server 生成 mel
                        vllm_output = None
                        
                      
                        serialized_data = msgpack.packb(dit_input, use_bin_type=True)
                       
                        zmq_socket.send(serialized_data)

                        response = zmq_socket.recv()
                        
                        import numpy as np
                        vllm_output = msgpack.unpackb(response, raw=False)
                        mel = np.frombuffer(vllm_output["data"], dtype=vllm_output["dtype"]).reshape(vllm_output["shape"])
                        
                        # 检查是否有错误
                        if "error" in vllm_output:
                            raise Exception(f"vLLM error: {vllm_output['error']}")
                        

                        # Step 3: 将 mel 转换为音频
                        # 直接调用 runtime 方法，不通过 HTTP API
                        wav_output = runtime.dit_output_to_wav(mel, str(mel.dtype))

                        # 保存音频文件
                        # audio_tensor = torch.tensor(wav_output["audio"])
                        # output_path = os.path.join(output_dir, f"{utt_id}.wav")
                        # torchaudio.save(output_path, audio_tensor.unsqueeze(0), wav_output["sample_rate"])
                        # print(f"[DEBUG]   Saved to: {output_path}")

                        success_count += 1
                        batch_success += 1
                    except Exception as e:
                        error_count += 1
                        batch_error += 1
                        print(f"\n[ERROR] Failed to process {utt_id}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
            except Exception as e:
                error_count += 1
                batch_error += 1
                print(f"\nError processing batch: {str(e)}")
                continue
            
            # 计算并输出 batch 吞吐量
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_size_actual = batch_success + batch_error
            if batch_time > 0 and batch_success > 0:
                batch_throughput = batch_success / batch_time
                print(f"\nBatch {batch_idx + 1} completed!")
                print(f"  Batch size:        {batch_size_actual}")
                print(f"  Successful:        {batch_success}/{batch_size_actual}")
                print(f"  Batch time:        {batch_time:.3f}s")
                print(f"  Batch throughput:  {batch_throughput:.2f} samples/s")
                
                # 检查是否达到了指定的总处理数量
                if total_size is not None and (success_count + error_count) >= total_size:
                    print(f"\nReached total size limit: {total_size}. Stopping inference.")
                    # 跳出外层循环
                    break

        
        end_time = time.time()
        end_datetime = datetime.now()
        epoch_time = end_time - start_time
        
        print(f"\nEpoch {epoch + 1} completed!")
        print(f"  Duration:          {timedelta(seconds=int(epoch_time))}")
        print(f"  Successful:        {success_count}/{success_count + error_count}")
        if success_count > 0:
            print(f"  Avg time/sample:   {epoch_time/success_count:.3f}s")
            print(f"  Throughput:        {success_count/epoch_time:.2f} samples/s")
        # 输出通信开销统计
        # if comm_count > 0:
        #     avg_comm_time = total_comm_time / comm_count
        #     print(f"  Avg communication time: {avg_comm_time:.3f}s")
    
    # 关闭ZeroMQ连接
    if zmq_socket:
        zmq_socket.close()
    
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
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--total-size", type=int, help="Total number of samples to process (default: process all)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for input prompt audio (default: 16000)")
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
            batch_size=args.batch_size,
            warmup=args.warmup,
            total_size=args.total_size,
            sample_rate=args.sample_rate,
            zmq_address=args.zmq_address
        )
    else:
        # 显示帮助信息
        parser.print_help()
        print("\nNote: This script now only supports batch inference mode with ZeroMQ communication.")
        print("Use --batch-infer to run batch inference.")


_bootstrap_from_env()

if __name__ == "__main__":
    main()
