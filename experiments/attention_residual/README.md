# Attention Residual 机制

## 背景

来自 [wyf3/llm_related](https://github.com/wyf3/llm_related/tree/main/kimi_attnres) 的 Attention Residual 机制，通过动态残差连接增强模型表达能力。

## 核心思想

**来源**: Kimi-Attention-Residual (wyf3/llm_related)

传统残差连接：`output = x + F(x)`

Attention Residual：`output = attn_weight * x + (1 - attn_weight) * F(x)`

其中 `attn_weight` 是通过注意力机制动态学习的。

## 原理

```python
class AttentionResidualBlock(nn.Module):
    """
    Attention Residual 块

    用注意力权重动态决定保留多少原始输入 vs 变换后的输出
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size

        # 原始变换（如MLP或Attention）
        self.transform = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads
        )

        # 注意力残差权重计算
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入

        Returns:
            output: (B, L, D) 输出
        """
        # 变换分支
        transformed = self.transform(x)

        # 计算残差门控权重
        # 基于输入和变换后状态的联合表示
        gate_input = torch.cat([x, transformed], dim=-1)
        residual_weight = self.residual_gate(gate_input)  # (B, L, 1)

        # 动态残差连接
        output = residual_weight * x + (1 - residual_weight) * transformed

        return output
```

## 应用到 Qwen3-4B 的挑战

### 挑战1：架构修改

Qwen3 使用标准 Transformer，需要修改每个层的残差连接：

```python
# 原始 Qwen3 Decoder Layer
class Qwen3DecoderLayer(nn.Module):
    def forward(self, hidden_states, ...):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states  # 标准残差

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # 标准残差

        return hidden_states

# Attention Residual 版本
class Qwen3DecoderLayerWithAttnRes(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # ... 原始初始化 ...

        # 添加残差门控
        self.attn_res_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.mlp_res_gate = nn.Sequential(...)  # 类似

    def forward(self, hidden_states, ...):
        # Self Attention with Attention Residual
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, ...)

        # 动态权重
        gate_input = torch.cat([residual, attn_out], dim=-1)
        attn_weight = self.attn_res_gate(gate_input)
        hidden_states = attn_weight * residual + (1 - attn_weight) * attn_out

        # MLP with Attention Residual (similar)
        ...

        return hidden_states
```

### 挑战2：训练稳定性

动态残差可能导致训练不稳定，需要：
- 门控初始化偏向恒等映射（权重初始化为1）
- 使用层归一化稳定
- 可能需要预热阶段

```python
def init_gate_for_identity(gate_module):
    """
    初始化门控偏向恒等映射
    """
    # 找到最后的线性层
    for module in reversed(list(gate_module.modules())):
        if isinstance(module, nn.Linear):
            # 初始化输出为较大的正值，使sigmoid接近1
            nn.init.constant_(module.bias, 2.0)
            break
```

### 挑战3：与 LoRA 的兼容性

需要确保 Attention Residual 的额外参数也能被 LoRA 训练或单独微调：

```python
class AttnResWithLoRA(nn.Module):
    """
    Attention Residual + LoRA 联合适配
    """
    def __init__(self, base_model, r=16):
        super().__init__()
        self.base_model = base_model

        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 只训练 Attention Residual 门控
        for layer in self.base_model.model.layers:
            for param in layer.attn_res_gate.parameters():
                param.requires_grad = True
            for param in layer.mlp_res_gate.parameters():
                param.requires_grad = True

        # 添加 LoRA（标准做法）
        self.lora_config = LoraConfig(r=r, ...)
        self.model = get_peft_model(self.base_model, self.lora_config)
```

## 简化实验方案

由于完整修改 Qwen3 架构复杂度高，建议分阶段探索：

### 阶段1：概念验证（在小型模型上）

```python
# 在 1B 级别小模型上验证有效性
class TinyModelWithAttnRes(nn.Module):
    """用于验证的小型模型"""
    def __init__(self, vocab_size, d_model=512, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            AttentionResidualBlock(d_model) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)
```

### 阶段2：适配器方式（Adapter-style）

不修改原模型，在特定层插入 Attention Residual 模块：

```python
class AttnResAdapter(nn.Module):
    """
    可插入的 Attention Residual 适配器
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, residual, transformed):
        gate_input = torch.cat([residual, transformed], dim=-1)
        weight = self.gate(gate_input)
        return weight * residual + (1 - weight) * transformed

# 使用 Hook 插入到 Qwen3
class Qwen3WithAttnResAdapter:
    def __init__(self, model):
        self.model = model
        self.adapters = nn.ModuleList([
            AttnResAdapter(model.config.hidden_size)
            for _ in range(len(model.model.layers))
        ])

    def register_hooks(self):
        """注册前向钩子插入适配器"""
        def make_hook(adapter):
            def hook(module, input, output):
                residual = input[0]
                return adapter(residual, output[0])
            return hook

        for i, layer in enumerate(self.model.model.layers):
            layer.register_forward_hook(make_hook(self.adapters[i]))
```

## 实验设计

| 阶段 | 目标 | 复杂度 | 预期收益 |
|------|------|--------|----------|
| 1 | 小模型验证 | 低 | 概念验证 |
| 2 | 适配器插入 | 中 | +0.5-1% |
| 3 | 完整修改 | 高 | +1-2% |

## 文件结构

```
attention_residual/
├── README.md                   # 本文件
├── attn_res_block.py          # Attention Residual 块实现
├── tiny_model_experiment.py   # 小模型验证
├── adapter_approach.py        # 适配器方案
├── modify_qwen3.py           # 修改Qwen3架构（高风险）
└── results/
    ├── tiny_model_results.json
    └── adapter_results.json
```

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 训练不稳定 | 高 | 高 | 谨慎初始化，小学习率 |
| 显存增加 | 中 | 中 | 门控参数量小，影响有限 |
| 与LoRA冲突 | 中 | 高 | 先在小模型验证 |
| 收益不明显 | 中 | 高 | 先做阶段1验证 |

## 建议

⚠️ **低优先级**：由于复杂度高且收益不确定，建议在完成其他高优先级实验后再尝试。

如果决定进行，严格按阶段1→2→3顺序，每阶段验证有效后再进入下一阶段。
