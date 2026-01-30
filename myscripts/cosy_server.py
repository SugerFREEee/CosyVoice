"""
在线 CosyVoice worker：接收 token/prompt，生成 DiT 输入，调用 vllm_server(pool)，将 mel 转为音频返回。

请求格式（msgpack）：
{
  "tokens": [int, ...] | 可选，若缺失则提供 "token_file"
  "token_file": "/path/to/tokens.txt" | 可选
  "prompt_wav": "/path/to/prompt.wav",  # 必填
  "num_inference_steps": 10             # 可选，覆盖默认步数
}

响应格式（msgpack）：
{ "audio": [...], "sample_rate": 24000 } 或 { "error": "..." }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
import torch
import zmq
import zmq.asyncio as zmq_async

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "third_party", "Matcha-TTS"))
sys.path.append(current_dir)

from token2wav_dit import CosyVoice2_Token2Wav, load_tokens_from_file  # noqa: E402


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
        if not os.path.isfile(prompt_wav_path):
            raise FileNotFoundError(f"Prompt audio not found: {prompt_wav_path}")

        token_tensor = torch.tensor([tokens], dtype=torch.int32).to(self.device)

        with torch.inference_mode():
            with self._autocast():
                prompt_token, prompt_feat, embedding = self.model.extract_prompt_features(prompt_wav_path)
                if prompt_token.ndim == 1:
                    prompt_token = prompt_token.unsqueeze(0)
                prompt_token = prompt_token.to(self.device)
                prompt_feat = prompt_feat.to(self.device)
                embedding = embedding.to(self.device)

                from torch.nn import functional as F
                from cosyvoice.utils.mask import make_pad_mask

                flow = self.model.flow
                embedding_proj = F.normalize(embedding, dim=1)
                embedding_proj = flow.spk_embed_affine_layer(embedding_proj)

                token_len1, token_len2 = prompt_token.shape[1], token_tensor.shape[1]
                token_concat = torch.concat([prompt_token, token_tensor], dim=1)
                token_len_total = torch.tensor([prompt_token.shape[1] + token_tensor.shape[1]], dtype=torch.int32).to(self.device)

                mask = (~make_pad_mask(token_len_total)).unsqueeze(-1).to(self.device)
                token_embed = flow.input_embedding(torch.clamp(token_concat, min=0)) * mask

                if hasattr(flow, "pre_lookahead_layer"):
                    h = flow.pre_lookahead_layer(token_embed)
                    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
                    mel_len1 = prompt_feat.shape[1]
                    mel_len2 = h.shape[1] - prompt_feat.shape[1]
                else:
                    h, _ = flow.encoder(token_embed, token_len_total)
                    h = flow.encoder_proj(h)
                    mel_len1 = prompt_feat.shape[1]
                    mel_len2 = int(token_len2 / flow.input_frame_rate * 22050 / 256)
                    h, _ = flow.length_regulator.inference(
                        h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, flow.input_frame_rate
                    )

                condition_vector = h
                output_size = flow.output_size
                conds = torch.zeros([1, mel_len1 + mel_len2, output_size], device=self.device).to(h.dtype)
                conds[:, :mel_len1] = prompt_feat
                conds_transposed = conds.transpose(1, 2)
                speaker_embedding = embedding_proj

        return {
            "condition_vector": condition_vector.transpose(1, 2).to(dtype=torch.float32).cpu().tolist(),
            "speaker_embedding": speaker_embedding.to(dtype=torch.float32).cpu().tolist(),
            "cond": conds_transposed.to(dtype=torch.float32).cpu().tolist(),
            "seq_len": condition_vector.shape[1],
            "mel_len1": mel_len1,
            "mel_len2": mel_len2,
            "batch_size": 1,
        }

    def dit_output_to_wav(self, mel: List[List[List[float]]], mel_dtype: Optional[str] = None) -> Dict[str, Any]:
        dtype = _parse_dtype(mel_dtype, torch.float32)
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


def _setup_runtime(model_dir: str, device_id: int, dtype_name: str):
    global runtime  # noqa: PLW0603
    dtype = _parse_dtype(dtype_name, torch.float16)
    runtime = CosyVoiceRuntime(model_dir=model_dir, device_id=device_id, dtype=dtype)
    print(f"[cosy_server] Loaded CosyVoice model from {model_dir} on device {runtime.device}")


def _require_runtime() -> CosyVoiceRuntime:
    if runtime is None:
        raise RuntimeError("Runtime is not initialized")
    return runtime


def _process_request(payload: dict[str, Any], vllm_address: str) -> bytes:
    rt = _require_runtime()
    tokens = payload.get("tokens")
    token_file = payload.get("token_file")
    if tokens is None:
        if not token_file:
            raise ValueError("tokens or token_file must be provided")
        tokens = load_tokens_from_file(token_file)
    prompt_wav = payload.get("prompt_wav")
    if not prompt_wav:
        raise ValueError("prompt_wav is required")

    dit_input = rt.prepare_dit_inputs(tokens, prompt_wav)
    num_steps = payload.get("num_inference_steps")
    if num_steps is not None:
        dit_input["num_inference_steps"] = num_steps

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(vllm_address)
    try:
        sock.send(msgpack.packb(dit_input, use_bin_type=True))
        resp = sock.recv()
    finally:
        sock.close()

    vllm_output = msgpack.unpackb(resp, raw=False)
    if "error" in vllm_output:
        raise RuntimeError(f"vLLM error: {vllm_output['error']}")

    mel = np.frombuffer(vllm_output["data"], dtype=vllm_output["dtype"]).reshape(vllm_output["shape"])
    audio_dict = rt.dit_output_to_wav(mel, str(mel.dtype))
    return msgpack.packb(audio_dict, use_bin_type=True)


async def _handle_request(socket, envelope: list[bytes], message: bytes, sem: asyncio.Semaphore, vllm_address: str):
    async with sem:
        try:
            payload = msgpack.unpackb(message, raw=False)
            loop = asyncio.get_running_loop()
            packed = await loop.run_in_executor(None, _process_request, payload, vllm_address)
            await socket.send_multipart(envelope + [packed])
        except Exception as e:
            print(f"[ERROR] cosy_server request failed: {e}")
            try:
                await socket.send_multipart(envelope + [json.dumps({"error": str(e)}).encode()])
            except Exception:
                pass


async def run_zmq_server(socket_address: str, max_workers: int, vllm_address: str):
    ctx = zmq_async.Context()
    socket = ctx.socket(zmq.ROUTER)
    socket.bind(socket_address)
    print(f"[cosy_server] ZeroMQ server listening on {socket_address}, max_workers={max_workers}, vllm={vllm_address}")
    sem = asyncio.Semaphore(max_workers)
    while True:
        frames = await socket.recv_multipart()
        if len(frames) < 1:
            continue
        envelope, message = frames[:-1], frames[-1]
        asyncio.create_task(_handle_request(socket, envelope, message, sem, vllm_address))


def main():
    parser = argparse.ArgumentParser(description="CosyVoice online server (token->DiT->wav)")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to CosyVoice checkpoint directory")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--dtype", type=str, default="float16", help="Computation dtype")
    parser.add_argument("--zmq-address", type=str, default="ipc:///tmp/cosy_worker.sock", help="ZeroMQ server address")
    parser.add_argument("--vllm-address", type=str, default="ipc:///tmp/vllm.sock", help="ZeroMQ vllm_server(pool) address")
    parser.add_argument("--max-workers", type=int, default=4, help="并发信号量限制")
    args = parser.parse_args()

    _setup_runtime(args.model_dir, args.device_id, args.dtype)
    asyncio.run(run_zmq_server(args.zmq_address, args.max_workers, args.vllm_address))


if __name__ == "__main__":
    main()
