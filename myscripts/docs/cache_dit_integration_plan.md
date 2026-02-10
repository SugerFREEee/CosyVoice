# CosyVoice 集成 cache-dit (DBCache) 修改点与方式

> 目标：在 flow 阶段的 DiT 推理中启用 DBCache。当前链路为 `CausalMaskedDiffWithDiT.inference` → `CausalConditionalCFM.solve_euler` → `DiT.forward`。

## 1. DiT 作为缓存目标模块

- 位置：`/home/wjs/workspace/CosyVoice/cosyvoice/flow/DiT/dit.py`
- 现状：`DiT` 内部 `self.transformer_blocks` 为 `ModuleList`，循环调用 `DiTBlock.forward(x, t, mask, rope)`，仅传入/返回 `hidden_states`。
- 结论：对 cache-dit 而言匹配 `ForwardPattern.Pattern_3`（输入输出只有 hidden_states）。
- 修改方式：不改该文件结构，直接通过 BlockAdapter 包装 `DiT` 和 `transformer_blocks`。

## 2. 缓存启用入口（模型加载阶段）

- 位置：`/home/wjs/workspace/CosyVoice/cosyvoice/cli/model.py`
- 现状：`CosyVoiceModel.load()` 完成 `self.flow` 和 `self.flow.decoder.estimator` 的加载。
- 修改方式：
  - 在 `load()` 内部或之后检测 `self.flow.decoder.estimator` 是否为 `torch.nn.Module`。
  - 若是 Module，则调用 `cache_dit.enable_cache(BlockAdapter(...), cache_config=DBCacheConfig(...))`。
  - 若 estimator 是 TRT 包装（`TrtContextWrapper`），则跳过缓存启用。
- 关键点：
  - `num_inference_steps=10`（来自 `CausalMaskedDiffWithDiT.inference` 的 `n_timesteps=10`）。
  - `enable_separate_cfg=False`（CFG 在 solver 外部以 batch=2 方式完成，非 block 内 CFG）。

## 3. cache-dit 适配器配置

- 位置：在 `cosyvoice/cli/model.py` 中新增 cache-dit 引用与初始化逻辑。
- 适配器示例（将用于实际修改）：

```python
from cache_dit import BlockAdapter, ForwardPattern, DBCacheConfig
import cache_dit

estimator = self.flow.decoder.estimator
adapter = BlockAdapter(
    transformer=estimator,
    blocks=estimator.transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_3,
)

cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
        num_inference_steps=10,
        # 其余参数按性能/质量要求调整
        # max_warmup_steps=..., Fn_compute_blocks=..., Bn_compute_blocks=..., residual_diff_threshold=...
        enable_separate_cfg=False,
    ),
)
```


## 7. 风险提示

- TRT 模式下 `self.flow.decoder.estimator` 不是 PyTorch Module，需跳过缓存。
- streaming 模式要确保 `num_inference_steps` 与实际 `n_timesteps` 一致。
- 若后续增加多 transformer（非当前结构），需升级为多 blocks/params_modifiers 方案。

