"""
对比测试：验证 vllm-omni 替换的正确性
python myscripts/compare_test.py
对比原始 cosyvoice flow 和 vllm-omni DiT 的中间结果
"""

import os
import sys
import torch
import requests
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from myscripts.token2wav_dit import CosyVoice2_Token2Wav


def load_tokens_from_file(token_file):
    with open(token_file, 'r') as f:
        tokens_str = f.read().strip()
    return [int(t) for t in tokens_str.split()]


def compare_tensors(name, tensor1, tensor2, rtol=1e-3, atol=1e-3):
    """比较两个张量是否相近"""
    if isinstance(tensor1, list):
        tensor1 = torch.tensor(tensor1)
    if isinstance(tensor2, list):
        tensor2 = torch.tensor(tensor2)

    print(f"\n{name}:")
    print(f"  Shape: {tensor1.shape} vs {tensor2.shape}")
    print(f"  Dtype: {tensor1.dtype} vs {tensor2.dtype}")
    print(f"  Device: {tensor1.device} vs {tensor2.device}")

    # 确保设备一致
    if tensor1.device != tensor2.device:
        print(f"  🔄 将张量移至同一设备...")
        tensor2 = tensor2.to(tensor1.device)

    # 确保数据类型一致
    if tensor1.dtype != tensor2.dtype:
        print(f"  🔄 将张量转换为同一数据类型...")
        tensor2 = tensor2.to(tensor1.dtype)

    # 对于整数类型，转换为浮点数再计算统计量
    t1_float = tensor1.float()
    t2_float = tensor2.float()

    print(f"  Mean: {t1_float.mean():.6f} vs {t2_float.mean():.6f}")
    print(f"  Std: {t1_float.std():.6f} vs {t2_float.std():.6f}")
    print(f"  Min: {t1_float.min():.6f} vs {t2_float.min():.6f}")
    print(f"  Max: {t1_float.max():.6f} vs {t2_float.max():.6f}")

    if tensor1.shape != tensor2.shape:
        print(f"  ❌ Shape mismatch!")
        return False

    # 检查数值是否相差过大
    if (t1_float.mean().abs() > 1.0 or t2_float.mean().abs() > 1.0) and \
       (t1_float.mean() / t2_float.mean()).abs() > 10.0:
        print(f"  ⚠️  警告：两个张量的平均值相差超过10倍，可能存在严重问题！")

    try:
        is_close = torch.allclose(t1_float, t2_float, rtol=rtol, atol=atol)
        max_diff = (t1_float - t2_float).abs().max().item()
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Close (rtol={rtol}, atol={atol}): {'✅' if is_close else '❌'}")
    except Exception as e:
        print(f"  ❌ 比较时发生错误: {e}")
        return False

    return is_close


def main():
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='对比测试：验证 vllm-omni 替换的正确性')
    parser.add_argument('--model-dir', type=str, default='/home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                        help='模型目录路径')
    parser.add_argument('--prompt-wav', type=str, default='/home/wjs/workspace/data/CowboyZ/seed-tts-eval/seedtts_testset/zh/prompt-wavs/10002287-00000094.wav',
                        help='prompt 音频文件路径')
    parser.add_argument('--token-file', type=str, default='/home/wjs/workspace/data/seedtts_tokens/10002287-00000095.txt',
                        help='token 文件路径')
    parser.add_argument('--vllm-host', type=str, default='localhost',
                        help='vLLM 服务器主机地址')
    parser.add_argument('--vllm-port', type=int, default=8001,
                        help='vLLM 服务器端口')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float16', 'float32'],
                        help='模型计算数据类型')
    parser.add_argument('--vllm-dtype', type=str, default='float32', choices=['float16', 'bfloat16', 'float32'],
                        help='vllm_server 使用的 dtype')
    
    args = parser.parse_args()
    
    # 配置
    model_dir = args.model_dir
    prompt_wav = args.prompt_wav
    token_file = args.token_file
    vllm_host = args.vllm_host
    vllm_port = args.vllm_port
    if args.dtype != "float32" or args.vllm_dtype != "float32":
        print("[WARN] 本次对比强制使用 float32；忽略 --dtype/--vllm-dtype 的非 float32 取值")
    dtype = torch.float32
    vllm_dtype = "float32"
    
    print("="*80)
    print("对比测试：原始 CosyVoice vs vLLM-Omni")
    print(f"vLLM-Omni dtype: {vllm_dtype}")
    print("="*80)

    # 加载 tokens
    tokens = load_tokens_from_file(token_file)
    print(f"\nTokens: {len(tokens)} tokens")
    print(f"Prompt wav: {prompt_wav}")

    # ============================================================
    # 方法1：使用原始 CosyVoice (token2wav_dit.py)
    # ============================================================
    print("\n" + "="*80)
    print("方法1：原始 CosyVoice (完全使用 cosyvoice 组件)")
    print("="*80)

    try:
        model_original = CosyVoice2_Token2Wav(model_dir=model_dir, device_id=0, dtype=dtype)
        print(f"  模型加载成功，使用数据类型: {dtype}")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 1: 提取 prompt 特征
    print("\n[Step 1] 提取 prompt 特征...")
    prompt_token_orig, prompt_feat_orig, embedding_orig = model_original.extract_prompt_features(prompt_wav)
    print(f"  prompt_token shape: {prompt_token_orig.shape}")
    print(f"  prompt_feat shape: {prompt_feat_orig.shape}")
    print(f"  embedding shape: {embedding_orig.shape}")

    # Step 2: Flow 推理（原始 DiT）
    print("\n[Step 2] Flow 推理 (原始 DiT)...")
    token_tensor = torch.tensor([tokens], dtype=torch.int32)
    with torch.inference_mode():
        tts_mel_orig = model_original.forward_flow(
            token=token_tensor,
            prompt_token=prompt_token_orig,
            prompt_feat=prompt_feat_orig,
            embedding=embedding_orig
        )
    print(f"  tts_mel shape: {tts_mel_orig.shape}")
    print(f"  tts_mel stats: mean={tts_mel_orig.mean():.6f}, std={tts_mel_orig.std():.6f}")

    # Step 3: HiFT 生成音频
    print("\n[Step 3] HiFT 生成音频...")
    with torch.inference_mode():
        audio_orig = model_original.forward_hift(tts_mel_orig)
    print(f"  audio shape: {audio_orig.shape}")
    print(f"  audio stats: mean={audio_orig.mean():.6f}, std={audio_orig.std():.6f}")

    # ============================================================
    # 方法2：使用 vLLM-Omni (cosy_server + vllm_server)
    # ============================================================
    print("\n" + "="*80)
    print("方法2：vLLM-Omni (cosy_server 预处理 + vllm_server DiT)")
    print("="*80)

    # 重用同一个模型的前端和后端
    print("\n[Step 1] cosy_server 预处理...")
    # 模拟 cosy_server 的 prepare_dit_inputs
    from torch.nn import functional as F
    from cosyvoice.utils.mask import make_pad_mask

    flow = model_original.flow
    token_tensor_new = torch.tensor([tokens], dtype=torch.int32).to(model_original.device)

    with torch.inference_mode():
        # 提取 prompt 特征
        prompt_token_new, prompt_feat_new, embedding_new = model_original.extract_prompt_features(prompt_wav)

        if prompt_token_new.ndim == 1:
            prompt_token_new = prompt_token_new.unsqueeze(0)
        prompt_token_new = prompt_token_new.to(model_original.device)
        prompt_feat_new = prompt_feat_new.to(model_original.device)
        embedding_new = embedding_new.to(model_original.device)

        # Embedding projection
        embedding_proj = F.normalize(embedding_new, dim=1)
        embedding_proj = flow.spk_embed_affine_layer(embedding_proj)

        # Concat tokens
        token_len1, token_len2 = prompt_token_new.shape[1], token_tensor_new.shape[1]
        token_concat = torch.concat([prompt_token_new, token_tensor_new], dim=1)
        token_len_total = torch.tensor([prompt_token_new.shape[1] + token_tensor_new.shape[1]], dtype=torch.int32).to(model_original.device)

        mask = (~make_pad_mask(token_len_total)).unsqueeze(-1).to(model_original.device)
        token_embed = flow.input_embedding(torch.clamp(token_concat, min=0)) * mask

        # 根据 flow 类型选择处理方式
        if hasattr(flow, 'pre_lookahead_layer'):
            h = flow.pre_lookahead_layer(token_embed)
            h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
            mel_len1 = prompt_feat_new.shape[1]
            mel_len2 = h.shape[1] - prompt_feat_new.shape[1]
        else:
            h, _ = flow.encoder(token_embed, token_len_total)
            h = flow.encoder_proj(h)
            mel_len1 = prompt_feat_new.shape[1]
            mel_len2 = int(token_len2 / flow.input_frame_rate * 22050 / 256)
            h, _ = flow.length_regulator.inference(
                h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, flow.input_frame_rate
            )

        condition_vector = h
        speaker_embedding_proj = embedding_proj  # Keep as [1, 192], don't expand

    print(f"  condition_vector shape: {condition_vector.shape}")
    print(f"  speaker_embedding shape: {speaker_embedding_proj.shape}")
    print(f"  mel_len1={mel_len1}, mel_len2={mel_len2}")

    # Prepare cond: prompt features + zeros
    output_size = flow.output_size  # 80
    conds = torch.zeros([1, mel_len1 + mel_len2, output_size], device=model_original.device).to(h.dtype)
    conds[:, :mel_len1] = prompt_feat_new
    conds_transposed = conds.transpose(1, 2)  # [batch, dim, seq]

    # Step 2: vllm_server DiT 推理
    print("\n[Step 2] vLLM-Omni DiT 推理...")
    dit_input = {
        "condition_vector": condition_vector.transpose(1, 2).cpu().tolist(),  # [batch, dim, seq]
        "speaker_embedding": speaker_embedding_proj.cpu().tolist(),  # [batch, spk_dim]
        "cond": conds_transposed.cpu().tolist(),  # [batch, dim, seq]
        "seq_len": condition_vector.shape[1],
        "mel_len1": mel_len1,
        "mel_len2": mel_len2,
        "batch_size": 1,
        "num_inference_steps": 10
    }
    
    # 保存dit_input到文件，用于后续不同dtype的测试
    import json
    import os
    save_dir = "/home/wjs/workspace/data/dit_input_test"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dit_input.json")
    with open(save_path, "w") as f:
        json.dump(dit_input, f)
    print(f"  📁 dit_input已保存到: {save_path}")

    # 先进行echo测试，验证数据传输是否正确
    print("\n[测试] vLLM-Omni 数据传输测试 (echo)...")
    echo_response = requests.post(f"http://{vllm_host}:{vllm_port}/echo", json=dit_input, timeout=30)
    echo_response.raise_for_status()
    echo_output = echo_response.json()
    
    # 验证echo返回的数据是否与发送的数据一致
    echo_success = True
    
    # 验证condition_vector
    if echo_output["condition_vector"] != dit_input["condition_vector"]:
        print("  ❌ condition_vector 数据不一致")
        echo_success = False
    else:
        print("  ✅ condition_vector 数据一致")
    
    # 验证speaker_embedding
    if echo_output["speaker_embedding"] != dit_input["speaker_embedding"]:
        print("  ❌ speaker_embedding 数据不一致")
        echo_success = False
    else:
        print("  ✅ speaker_embedding 数据一致")
    
    # 验证cond
    if echo_output["cond"] != dit_input["cond"]:
        print("  ❌ cond 数据不一致")
        echo_success = False
    else:
        print("  ✅ cond 数据一致")
    
    # 验证其他参数
    if echo_output["seq_len"] != dit_input["seq_len"] or \
       echo_output["mel_len1"] != dit_input["mel_len1"]:
        print("  ❌ 参数不一致")
        echo_success = False
    else:
        print("  ✅ 参数一致")
    
    if echo_success:
        print("  🎉 数据传输测试通过！")
    else:
        print("  ⚠️  数据传输测试失败，可能会影响后续推理结果")

    # 执行实际的推理请求
    print("\n[推理] 执行 vLLM-Omni DiT 推理...")
    response = requests.post(f"http://{vllm_host}:{vllm_port}/infer", json=dit_input, timeout=120)
    response.raise_for_status()
    vllm_output = response.json()

    # 处理 vLLM 返回的 tts_mel 中可能包含的 None 值
    mel_data = vllm_output["tts_mel"]
    # 将 None 替换为 0.0
    mel_clean = [[[0.0 if x is None else x for x in subsublist] for subsublist in sublist] for sublist in mel_data]
    # 使用与模型一致的数据类型和设备
    tts_mel_new = torch.tensor(mel_clean, dtype=dtype, device=model_original.device)
    print(f"  tts_mel shape: {tts_mel_new.shape}")
    print(f"  tts_mel stats: mean={tts_mel_new.mean():.6f}, std={tts_mel_new.std():.6f}")

    # Step 3: HiFT 生成音频
    print("\n[Step 3] HiFT 生成音频...")
    with torch.inference_mode():
        # HiFT 模型期望输入为 float32 类型，与模型参数保持一致
        tts_mel_new_device = tts_mel_new.to(model_original.device, dtype=torch.float32)
        audio_new = model_original.forward_hift(tts_mel_new_device)
    print(f"  audio shape: {audio_new.shape}")
    print(f"  audio stats: mean={audio_new.mean():.6f}, std={audio_new.std():.6f}")

    # ============================================================
    # 对比结果
    # ============================================================
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)

    # 对比 prompt 特征
    compare_tensors("Prompt Token", prompt_token_orig, prompt_token_new)
    compare_tensors("Prompt Feat", prompt_feat_orig, prompt_feat_new)
    compare_tensors("Embedding", embedding_orig, embedding_new)

    # 对比 mel（不再去除 prompt 部分，直接对比完整的 mel）
    print("\n" + "-"*80)
    print("关键对比：生成的 mel spectrogram")
    print("-"*80)
    
    print(f"  原始输出形状: {tts_mel_orig.shape}")
    print(f"  vLLM 输出形状: {tts_mel_new.shape}")
    
    # 确保数据类型一致
    tts_mel_orig_aligned = tts_mel_orig.to(dtype=tts_mel_new.dtype)
    compare_tensors("TTS Mel", tts_mel_orig_aligned, tts_mel_new, rtol=0.1, atol=0.1)

    # 对比音频
    compare_tensors("Audio", audio_orig, audio_new, rtol=0.1, atol=0.1)

    # 保存音频用于听觉对比
    import torchaudio
    output_dir = "//home/wjs/workspace/data/compare_output"
    os.makedirs(output_dir, exist_ok=True)

    torchaudio.save(f"{output_dir}/original.wav", audio_orig.cpu(), 22050)
    torchaudio.save(f"{output_dir}/vllm_omni.wav", audio_new.cpu(), 22050)

    print(f"\n音频已保存到: {output_dir}")
    print(f"  - original.wav: 原始 CosyVoice")
    print(f"  - vllm_omni.wav: vLLM-Omni")

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
