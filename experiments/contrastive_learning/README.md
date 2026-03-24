# 对比学习增强

## 背景

当前损失函数仅包含KL散度（软标签）和交叉熵（硬标签）。引入对比学习可以帮助模型更好地学习类间关系。

## 核心思想

**来源**: arXiv 2024 "A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation"

对比学习目标：
- 拉近（正样本对）：相似情感的样本在嵌入空间中靠近
- 推远（负样本对）：不同情感的样本在嵌入空间中远离

## 实验方案

### 方案1：批内对比学习（In-batch Contrastive）

```python
class InBatchContrastiveLoss(nn.Module):
    """
    批内对比学习损失

    在同一批次内，相同标签的样本为正对，不同标签为负对
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                embeddings: torch.Tensor,    # (B, D) 句子嵌入
                logits: torch.Tensor,        # (B, C) 分类logits
                labels: torch.Tensor,        # (B,) 硬标签
                teacher_probs: torch.Tensor  # (B, C) 软标签
                ) -> torch.Tensor:
        """
        总损失 = CE + KL + Contrastive
        """
        batch_size = embeddings.size(0)

        # 1. 硬标签交叉熵
        ce = self.ce_loss(logits, labels)

        # 2. KL散度（软标签蒸馏）
        student_probs = F.softmax(logits / 2.0, dim=-1)
        kl = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')

        # 3. 对比学习损失
        # 归一化嵌入
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算相似度矩阵 (B, B)
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 构造正负样本掩码
        labels_expanded = labels.unsqueeze(0).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        # 对角线为自身，排除
        positive_mask.fill_diagonal_(0)

        # InfoNCE损失
        # 对于每个样本，正样本是同类别的其他样本
        contrastive_loss = 0.0
        for i in range(batch_size):
            pos_sim = similarity[i, positive_mask[i].bool()]
            if len(pos_sim) > 0:
                # 正样本对的logits
                pos_logits = pos_sim
                # 所有样本的logits（负样本）
                neg_logits = similarity[i, torch.arange(batch_size) != i]

                # InfoNCE: -log(exp(pos) / sum(exp(all)))
                logits_i = torch.cat([pos_logits, neg_logits])
                labels_i = torch.zeros(len(pos_logits), dtype=torch.long,
                                       device=logits.device)

                contrastive_loss += F.cross_entropy(
                    logits_i.unsqueeze(0),
                    labels_i.unsqueeze(0)
                )

        contrastive_loss = contrastive_loss / batch_size

        # 总损失
        total_loss = ce + kl + 0.5 * contrastive_loss

        return total_loss, {
            'ce': ce.item(),
            'kl': kl.item(),
            'contrastive': contrastive_loss.item()
        }
```

### 方案2：监督对比学习（Supervised Contrastive）

```python
class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比学习损失 (SupCon)

    参考: "Supervised Contrastive Learning" (NeurIPS 2020)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) 归一化后的嵌入
            labels: (B,) 标签
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算相似度
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 构造正样本掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # 排除自身
        mask.fill_diagonal_(0)

        # 计算对比损失
        # exp(similarity)
        exp_sim = torch.exp(similarity_matrix)

        # 对于每个锚点，正样本的和
        pos_sum = (exp_sim * mask).sum(dim=1)

        # 所有样本的和（不包括自身）
        neg_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        all_sum = (exp_sim * neg_mask).sum(dim=1)

        # 只计算有正样本的锚点
        pos_mask = mask.sum(dim=1) > 0
        loss = -torch.log(pos_sum[pos_mask] / all_sum[pos_mask] + 1e-8)

        return loss.mean()
```

### 方案3：知识蒸馏 + 对比学习联合训练

```python
class ContrastiveDistillationTrainer:
    """
    对比学习与知识蒸馏联合训练器
    """
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 128)  # 投影到128维
        )

        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.07)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_loss(self, batch):
        # 前向传播
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True
        )

        # 分类logits
        logits = outputs.logits[:, -1, :]  # 最后一个token

        # 获取句子嵌入（使用[CLS]或平均池化）
        hidden_states = outputs.hidden_states[-1]
        sentence_emb = hidden_states.mean(dim=1)  # (B, D)

        # 投影
        projected = self.projection_head(sentence_emb)

        # 损失计算
        ce = self.ce_loss(logits, batch['labels'])

        kl = self.kl_loss(
            F.softmax(logits / 2.0, dim=-1).log(),
            batch['teacher_probs']
        )

        supcon = self.supcon_loss(projected, batch['labels'])

        # 加权
        total = (
            self.config.alpha_ce * ce +
            self.config.alpha_kl * kl +
            self.config.alpha_contrastive * supcon
        )

        return total, {
            'ce': ce.item(),
            'kl': kl.item(),
            'supcon': supcon.item()
        }
```

## 实验设计

| 实验 | 损失组合 | 对比权重 | 预期提升 |
|------|----------|----------|----------|
| baseline | CE + KL | 0 | 86.50% |
| exp1 | CE + KL + In-batch | 0.5 | +0.5-1% |
| exp2 | CE + KL + SupCon | 0.3 | +1-1.5% |
| exp3 | CE + KL + SupCon | 0.5 | +1-2% |

## 文件结构

```
contrastive_learning/
├── README.md                   # 本文件
├── inbatch_contrastive.py     # 批内对比损失
├── supervised_contrastive.py  # 监督对比损失
├── contrastive_trainer.py     # 联合训练器
├── embedding_visualizer.py    # 嵌入可视化
└── results/
    ├── tSNE_embeddings.png    # t-SNE可视化
    └── accuracy_comparison.png
```

## 快速开始

```bash
# 基础实验（仅In-batch）
python contrastive_trainer.py \
    --strategy inbatch \
    --contrastive_weight 0.5 \
    --output ./results/inbatch.json

# 监督对比学习
python contrastive_trainer.py \
    --strategy supcon \
    --contrastive_weight 0.3 \
    --projection_dim 128 \
    --output ./results/supcon.json

# 可视化嵌入空间
python embedding_visualizer.py \
    --model ./results/supcon_model \
    --data ../../data/curriculum/val_fixed.json \
    --output ./results/tSNE_embeddings.png
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| contrastive_weight | 0.3-0.5 | 对比损失权重 |
| temperature | 0.07 | 温度系数 |
| projection_dim | 128 | 投影头输出维度 |

## 注意事项

1. **显存开销**：对比学习需要存储所有样本的嵌入，可能增加显存占用
2. ** batch_size**：较大的batch_size有助于找到更多正负样本对
3. **训练时间**：对比损失计算较复杂，训练时间可能增加20-30%
