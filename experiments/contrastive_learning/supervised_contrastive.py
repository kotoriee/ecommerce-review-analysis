"""
监督对比学习 (Supervised Contrastive Learning) 实现

结合软标签蒸馏和对比学习的训练脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比学习损失 (SupCon)

    将同类样本在嵌入空间中拉近，不同类样本推远
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) 归一化后的句子嵌入
            labels: (B,) 样本标签

        Returns:
            loss: 对比损失
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算相似度矩阵 (B, B)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 构造正负样本掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # 排除自身
        mask.fill_diagonal_(0)

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)

        # 对于每个锚点，计算正样本和
        pos_sum = (exp_sim * mask).sum(dim=1, keepdim=True)

        # 所有样本的exp和（不包括自身）
        neg_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        all_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)

        # 只计算有正样本的锚点
        valid_mask = (mask.sum(dim=1) > 0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        loss = loss[valid_mask].mean()

        return loss


class ProjectionHead(nn.Module):
    """
    投影头：将模型隐藏状态投影到低维空间用于对比学习
    """

    def __init__(self, hidden_size: int, projection_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, D) 或 (B, D)

        Returns:
            projected: (B, projection_dim)
        """
        # 如果输入是序列，做平均池化
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)

        return self.projection(hidden_states)


class ContrastiveDistillationTrainer:
    """
    对比学习 + 知识蒸馏 联合训练器
    """

    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # 投影头
        hidden_size = model.config.hidden_size
        self.projection_head = ProjectionHead(
            hidden_size,
            projection_dim=config.get('projection_dim', 128)
        ).cuda()

        # 损失函数
        self.supcon_loss = SupervisedContrastiveLoss(
            temperature=config.get('temperature', 0.07)
        )
        self.ce_loss = nn.CrossEntropyLoss()

        # 优化器（同时优化模型和投影头）
        trainable_params = [
            p for p in model.parameters() if p.requires_grad
        ] + list(self.projection_head.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get('lr', 2e-4),
            weight_decay=0.01
        )

        self.alpha_ce = config.get('alpha_ce', 0.5)
        self.alpha_kl = config.get('alpha_kl', 0.5)
        self.alpha_contrastive = config.get('alpha_contrastive', 0.3)

    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        计算联合损失

        Returns:
            total_loss: 总损失
            metrics: 各损失分量的字典
        """
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        soft_labels = batch['soft_label'].cuda()

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 1. 分类logits（最后一个token）
        logits = outputs.logits[:, -1, :3]

        # 2. 句子嵌入（倒数第二层隐藏状态的平均池化）
        hidden_states = outputs.hidden_states[-2]  # 倒数第二层
        sentence_emb = hidden_states.mean(dim=1)  # (B, D)

        # 3. 投影到低维空间
        projected = self.projection_head(sentence_emb)  # (B, P)

        # 损失计算
        # CE损失
        ce = self.ce_loss(logits, labels)

        # KL损失（软标签蒸馏）
        student_probs = F.log_softmax(logits / 2.0, dim=-1)
        kl = F.kl_div(student_probs, soft_labels, reduction='batchmean')

        # 对比损失
        supcon = self.supcon_loss(projected, labels)

        # 加权总损失
        total = (
            self.alpha_ce * ce +
            self.alpha_kl * kl +
            self.alpha_contrastive * supcon
        )

        metrics = {
            'ce': ce.item(),
            'kl': kl.item(),
            'supcon': supcon.item(),
            'total': total.item()
        }

        return total, metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        self.projection_head.train()

        total_metrics = {'ce': 0, 'kl': 0, 'supcon': 0, 'total': 0}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss, metrics = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for k, v in metrics.items():
                total_metrics[k] += v

            pbar.set_postfix({
                'loss': f"{metrics['total']:.4f}",
                'supcon': f"{metrics['supcon']:.4f}"
            })

        # 平均
        n = len(dataloader)
        return {k: v / n for k, v in total_metrics.items()}


def visualize_embeddings(model, projection_head, dataloader, output_path: str):
    """
    使用t-SNE可视化嵌入空间

    Args:
        model: 训练好的模型
        projection_head: 投影头
        dataloader: 数据加载器
        output_path: 输出图片路径
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    model.eval()
    projection_head.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="收集嵌入"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)

            hidden_states = outputs.hidden_states[-2].mean(dim=1)
            projected = projection_head(hidden_states)

            all_embeddings.append(projected.cpu())
            all_labels.append(batch['labels'])

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # t-SNE降维
    print("运行t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘图
    plt.figure(figsize=(10, 8))
    colors = ['red', 'gray', 'green']
    label_names = ['Negative', 'Neutral', 'Positive']

    for i in range(3):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=colors[i], label=label_names[i], alpha=0.6)

    plt.legend()
    plt.title('Sentence Embeddings (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='对比学习增强训练')
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--val_data', required=True)
    parser.add_argument('--base_model', default='unsloth/Qwen3-4B-unsloth-bnb-4bit')
    parser.add_argument('--output_dir', default='./results/contrastive_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)  # 对比学习需要较大batch
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--alpha_contrastive', type=float, default=0.3)
    args = parser.parse_args()

    print("="*60)
    print("对比学习 + 知识蒸馏 联合训练")
    print("="*60)
    print(f"对比损失权重: {args.alpha_contrastive}")
    print(f"投影维度: {args.projection_dim}")
    print(f"温度系数: {args.temperature}")
    print("="*60)

    # TODO: 加载模型和数据
    # model = ...
    # tokenizer = ...
    # train_loader, val_loader = ...

    # 配置
    config = {
        'lr': 2e-4,
        'alpha_ce': 0.5,
        'alpha_kl': 0.5,
        'alpha_contrastive': args.alpha_contrastive,
        'projection_dim': args.projection_dim,
        'temperature': args.temperature
    }

    # trainer = ContrastiveDistillationTrainer(model, tokenizer, config)

    # 训练循环
    # for epoch in range(args.epochs):
    #     metrics = trainer.train_epoch(train_loader, epoch)
    #     print(f"Epoch {epoch}: {metrics}")

    print("\n训练完成!")


if __name__ == '__main__':
    main()
