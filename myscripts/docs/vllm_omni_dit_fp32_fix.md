# vLLM-Omni DiT（CosyVoice3）问题排查与 fp32 修复记录

本文记录在「用 vllm-omni 替换 CosyVoice token→wav 链路中的 DiT」过程中发现的问题、定位手段，以及当前采取的 **全流程 fp32** 修复方案与改动点。

## 背景与现象

- 目标：保持 `CosyVoice/myscripts/token2wav_dit.py`（原生、可跑通）不变，只替换 DiT 部分为 vllm-omni 引擎。
- 现象：
  - fp16：vllm-omni DiT 输出在第 2 步开始出现数值崩坏（NaN），或在上层表现为全 0 mel。
  - fp32：原始 vllm-omni 的 “Real inference” 实现与 CosyVoice 的 CFM 迭代不一致，导致 mel 与原生差距大。

## 根因（代码层面）

1. **fp16 不稳定（NaN）**
   - 通过对 `DiTBlock`、`attention`、`feedforward` 加 hook（见下方脚本）确认：在 fp16 下第 1 个 diffusion step 后，几乎所有 block 输出直接变为 NaN，导致整段 mel 崩坏。
   - 结论：当前实现/环境下该 DiT 不适合 fp16 端到端推理（至少需要把 LN/softmax/激活等关键算子强制留在 fp32）。

2. **vllm-omni 的 Real inference 逻辑与 CosyVoice 不一致**
   - CosyVoice3 的 flow decoder 使用 `CausalConditionalCFM.solve_euler`（`CosyVoice/cosyvoice/flow/flow_matching.py`），核心特征：
     - 从固定随机噪声 `z` 开始（`CausalConditionalCFM.rand_noise`），不是从 `cond` 开始；
     - `t_span` 使用 cosine scheduler（来自 `cfm_params.content.t_scheduler`）；
     - 使用 CFG：拼成 batch=2（有条件/无条件），然后按 `inference_cfg_rate` 合成速度场；
     - 传入 estimator 的 `t` 范围是 `[0,1]`（不会乘 1000）。
   - 修复后 vllm-omni pipeline 已按上述逻辑实现（详见“修改点”）。

## 修改点（已落盘）

### 1) vllm-omni：按 CosyVoice CFM 方式实现 DiT 推理，并强制 fp32

文件：`CosyVoice/third_party/vllm-omni/vllm_omni/diffusion/models/cosyvoice3/pipeline_cosyvoice3.py`

- `CosyVoice3Pipeline.__init__`：
  - 强制 `self.model` 以 `torch.float32` 放到 GPU 上（避免 fp16 NaN）。
  - 新增 `_rand_noise` buffer，复刻 CosyVoice3 的 `CausalConditionalCFM.rand_noise`（seed=0，shape `[1,80,15000]`）。
  - `_rand_noise` 的 generator 与张量必须使用 **CUDA device** 创建，否则在 worker 里会报错：
    - 错误：`Expected a 'cuda' device type for generator but found 'cpu'`
    - 修复：generator / tensor 显式 `device=self.device`。

- `_forward_inference`：
  - 输入：`condition_vector` → `mu`，`speaker_embedding` → `spks`，`cond` 保持原样（均为 `[B, 80, T]`）。
  - 生成 `t_span`（cosine scheduler）。
  - 从 `z`（固定噪声）开始，构造 CFG batch=2 的输入并调用 estimator：
    - 通过 `self.model.dit(x, mask, mu, t, spks, cond, streaming=False)` 直接复刻 CosyVoice estimator 的调用方式与张量布局。
  - Euler 更新：`x = x + dt * guided_dphi_dt`。
  - 输出：直接返回 `[B, 80, T]` 的 mel（float32）。

### 2) vllm_server：强制 float32（忽略外部 dtype 参数）

文件：`CosyVoice/third_party/vllm-omni/myscripts/vllm_server.py`

- `_create_omni_engine` 内：
  - 无论 CLI 传什么 `--dtype`，都强制使用 `torch.float32` 初始化 `OmniDiffusionConfig`。
  - 运行时会提示忽略非 float32 参数。

### 3) compare_test：强制 float32

文件：`CosyVoice/myscripts/compare_test.py`

- 默认 `--dtype` / `--vllm-dtype` 调整为 `float32`。
- 即使传入非 float32，也会打印 warning 并强制用 `float32` 进行比较（避免误测 fp16 走进 NaN 路径）。

## 定位用调试脚本

以下脚本用于确认 fp16 崩坏发生在 DiT 的哪个位置：

- `CosyVoice/third_party/vllm-omni/myscripts/debug_dit_fp16.py`
  - 重放 `dit_input.json`，输出每个 diffusion step 的 `v/x` 统计信息（快速判断是否 NaN）。
- `CosyVoice/third_party/vllm-omni/myscripts/debug_dit_block_fp16.py`
  - 对每个 `DiTBlock`、其 `attn` 输出、其 `ff` 输出加 hook，打印每步的统计信息，用于定位 NaN 发生在 attention 还是 MLP 路径。

## 运行方式（推荐）

1) 启动 vllm-omni 服务（float32）：

```bash
cd /home/wjs/workspace/CosyVoice/third_party/vllm-omni
conda activate dit
python myscripts/vllm_server.py --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --dtype float32
```

2) 跑 CosyVoice 侧对比（float32）：

```bash
cd /home/wjs/workspace/CosyVoice/myscripts
conda activate cosyvoice
python myscripts/compare_test.py --vllm-dtype float32
```

3) 若需继续定位 fp16（不建议用于推理，但可用于分析）：

```bash
cd /home/wjs/workspace/CosyVoice/third_party/vllm-omni
conda activate dit
python myscripts/debug_dit_block_fp16.py --dtype float16 --log-steps 3
```

## 备注

- 本次修复优先保证正确性（对齐 CosyVoice CFM solver）与稳定性（全 fp32）。如后续仍需 fp16，需要对 DiT 内部的 LN/softmax/激活等算子进行更细粒度的 fp32 保留（混合精度），否则容易出现 NaN。
