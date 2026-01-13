# CosyVoice 服务指南

以下内容概述了 `myscripts` 目录中新加入的两个 FastAPI 服务，并说明了它们在「CosyVoice ↔ vLLM-Omni」双环境部署中的接口与使用方式。

## cosy_server.py（CosyVoice 环境）

- **目标**：在 CosyVoice 环境内复用前端与 HiFT，实现「Prompt 音频 → DiT 输入特征」和「DiT mel → 波形」两项子任务。
- **启动方式**：
  - CLI：`python cosy_server.py --model-dir <CosyVoice模型目录> [--device-id 0 --dtype float16 --host 0.0.0.0 --port 8000]`
  - 环境变量：`COSY_SERVER_MODEL_DIR`、`COSY_SERVER_DEVICE_ID`、`COSY_SERVER_DTYPE` 可在导入时自动加载模型。

### 关键组件
1. `CosyVoiceRuntime`
   - 内部持有 `CosyVoice2_Token2Wav`，并记录 `device`、`dtype`、默认 `sample_rate=24000`。
   - `prepare_dit_inputs()`：调用 `extract_prompt_features()` 生成 `token / prompt_token / prompt_feat / embedding`，并序列化为 JSON 友好的列表。
   - `dit_output_to_wav()`：接收 mel（默认 float32），走 HiFT 推理得到波形。
2. FastAPI 路由
   - `GET /health`：返回服务状态。
   - `POST /token2dit-input`：
     - `prompt_wav_path` 必填；`tokens` 直接传数组或从 `token_file` 载入。
     - 返回字段包含 `token`, `prompt_token`, `prompt_feat`, `embedding`, `seq_len`, `batch_size`，用于交给 DiT 引擎。
   - `POST /dit-output2wav`：
     - `tts_mel`（三维数组）和可选 `mel_dtype`。
     - 返回 `audio`（单通道波形）与 `sample_rate`。

## vllm_server.py（vLLM-Omni 环境）

- **目标**：在 vLLM-Omni 环境内装配 CosyVoice3Pipeline，通过 REST 接口执行 DiT 推理并返回 mel。
- **启动方式**：
  - CLI：`python vllm_server.py --model-dir <CosyVoice模型目录> [--cache-backend cache_dit --cache-config '{}' --num-steps 10 --dtype float16 --host 0.0.0.0 --port 8001]`
  - 环境变量：`VLLM_SERVER_MODEL_DIR`、`VLLM_SERVER_CACHE_BACKEND`、`VLLM_SERVER_CACHE_CONFIG`、`VLLM_SERVER_NUM_STEPS`、`VLLM_SERVER_DTYPE`。

### 关键组件
1. `VLLMRuntime`
   - 保存默认的 cache backend、cache 配置、步数、dtype。
   - `_build_config()`：根据请求的 `seq_len`、`batch_size` 构造 `OmniDiffusionConfig` + `TransformerConfig`。
   - `infer()`：把 CosyVoice 端传来的四类张量转换为 Torch 张量，调用 `OmniDiffusion.generate()`，并对输出进行解析。
   - `_extract_mel()`：兼容 `OmniRequestOutput` / list / dict 等情况，最终返回 `torch.Tensor`。
2. FastAPI 路由
   - `GET /health`：显示默认 backend、步数等。
   - `POST /infer`：
     - 请求体为 `OmniInferRequest`：包含 CosyVoice 端的 `token/prompt_token/prompt_feat/embedding`，可覆盖 `seq_len`、`batch_size`、`num_inference_steps`、`cache_backend`。
     - 回包提供 `tts_mel`（float32 列表）与 `mel_dtype`，供 CosyVoice 端的 `/dit-output2wav` 继续处理。

## 调用流程示例
1. **CosyVoice 环境** `POST /token2dit-input` → 得到 DiT 输入特征。
2. **vLLM 环境** `POST /infer` → 获得 DiT 输出 mel。
3. **CosyVoice 环境** `POST /dit-output2wav` → 生成最终音频。

通过这两个服务，可以无缝地把「前端 + HiFT」与「DiT (vLLM-Omni)」拆分到两个 Conda 环境中运行，并用 HTTP 接口连接整条 token→wav 推理链路。
