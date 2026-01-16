# CosyVoice 3 token2wav 流程

## token2wav

```python
# /home/wjs/workspace/CosyVoice/cosyvoice/cli/model.py

# flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithDiT
class CosyVoice3Model(CosyVoice2Model):
    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(...)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            # ...
            tts_speech, _ = self.hift.inference(speech_feat=tts_mel, finalize=finalize)
            tts_speech = tts_speech[:, self.hift_cache_dict[uuid]['speech_offset']:]
            self.hift_cache_dict[uuid]['speech_offset'] += tts_speech.shape[1]
        return tts_speech
```

```python
class CausalMaskedDiffWithDiT(torch.nn.Module):
    @torch.inference_mode()
    def inference(...):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)

        # decoder: !new:cosyvoice.flow.flow_matching.CausalConditionalCFM
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


```


## CausalConditionalCFM


1. 在 `CausalConditionalCFM.forward` 方法
```python
# CosyVoice/cosyvoice/flow/flow_matching.py
@torch.inference_mode()
def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
    # 生成初始噪声 z
    z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
    # 时间步设置
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    if self.t_scheduler == 'cosine':
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    # 调用 `solve_euler` 方法执行实际的扩散步骤
    return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None
```

2. **每一步噪声处理**：在 `solve_euler` 方法中，通过循环处理每一个时间步
```python
# CosyVoice/cosyvoice/flow/flow_matching.py
def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):

    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(dim=0)

    # I am storing this because I can later plot it by putting a debugger here and saving it to a file
    # Or in future might add like a return_all_steps flag
    sol = []

    # 在扩散步骤循环之前预先分配内存，避免在每个步骤中重复创建张量
    # 双倍批次大小，因为使用CFG
    x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
    mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
    mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
    t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
    spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
    cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)

    for step in range(1, len(t_span)):
        # Classifier-Free Guidance inference introduced in VoiceBox
        x_in[:] = x.repeat(2, 1, 1)        # 扩散状态复制到两个批次
        mask_in[:] = mask.repeat(2, 1, 1)  # 掩码复制到两个批次
        mu_in[:batch_size] = mu            # 只填充前半部分的条件向量
        spks_in[:batch_size] = spks        # 只填充前半部分的说话人嵌入
        cond_in[:batch_size] = cond        # 只填充前半部分的条件输入

        # 调用模型获取预测的噪声
        
        dphi_dt = self.forward_estimator(
            x_in, mask_in,
            mu_in, t_in,
            spks_in,
            cond_in,
            streaming
        )
        # 应用分类器自由引导（CFG）
        # 分割模型输出，将模型输出 dphi_dt 沿批次维度分割为两部分
        # - guided ：前 batch_size 个样本，包含完整条件信息的模型输出
        # - cfg ：后 batch_size 个样本，条件信息被置零的模型输出
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)

        # 引导输出 = 有条件输出 + CFG强度 × (有条件输出 - 无条件输出)
        dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)

        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    return sol[-1].float()

```

```python
def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            #  estimator: !new:cosyvoice.flow.DiT.dit.DiT
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            # ...
```

- CosyVoice3 使用`CFM（Conditional Flow Matching）`
    - 从初始噪声开始
    - 按照预定的时间步序列
    - 每一步都调用模型预测噪声
    - 应用 `CFG（Classifier-Free Guidance 分类器自由引导` 增强条件控制
    - 使用欧拉方法更新状态
    - 最终生成高质量的梅尔频谱

## DiT Block

### 组件来源

| 组件名称 | 类型 | 来源 | 
|---------|------|------|
| **self.time_embed** | TimestepEmbedding | modules.py | 
| **self.input_embed** | InputEmbedding | 当前文件 | 
| **self.rotary_embed** | RotaryEmbedding | x_transformers库 | 
| **self.transformer_blocks** | nn.ModuleList[DiTBlock] | PyTorch内置 + modules.py |
| **self.long_skip_connection** | nn.Linear | PyTorch内置 | 
| **self.norm_out** | AdaLayerNormZero_Final | modules.py | 
| **self.proj_out** | nn.Linear | PyTorch内置 |

<summary> DiTBlock 组件</summary><table><tr><th>组件名称</th><th>类型</th><th>来源</th></tr><tr><td>self.attn_norm</td><td>AdaLayerNormZero</td><td>modules.py</td></tr><tr><td>self.attn</td><td>Attention</td><td>modules.py</td></tr><tr><td>processor</td><td>AttnProcessor</td><td>modules.py</td></tr><tr><td>self.ff_norm</td><td>nn.LayerNorm</td><td>PyTorch内置</td></tr><tr><td>self.ff</td><td>FeedForward</td><td>modules.py</td></tr></table>

### forward

```python
    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        # 维度转置
        x = x.transpose(1, 2)        # (batch, mel_dim, seq_len) → (batch, seq_len, mel_dim)
        mu = mu.transpose(1, 2)      # (batch, mu_dim, seq_len) → (batch, seq_len, mu_dim)
        cond = cond.transpose(1, 2)  # (batch, mel_dim, seq_len) → (batch, seq_len, mel_dim)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)


        # 时间步嵌入
        t = self.time_embed(t)
        # 输入特征嵌入
        x = self.input_embed(x, cond, mu, spks.squeeze(1))
        # 旋转位置编码
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        # 长跳跃连接
        if self.long_skip_connection is not None:
            residual = x

        # 注意力掩码生成
        if streaming is True:
            # 流式推理的注意力掩码
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, self.static_chunk_size, -1).unsqueeze(dim=1)
        else:
            # 非流式推理的注意力掩码
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1).unsqueeze(dim=1)

        # nn.ModuleList[DiTBlock]
        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output
```