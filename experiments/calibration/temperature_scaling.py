"""
温度缩放校准脚本

在验证集上学习最优温度参数，用于校准模型置信度
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm


class TemperatureScaler(nn.Module):
    """
    温度缩放校准器

    学习单一温度参数T，使得 softmax(logits/T) 更接近真实概率分布
    """

    def __init__(self):
        super().__init__()
        # 初始化温度参数（取log是为了保证正数）
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用温度缩放

        Args:
            logits: 模型原始输出 (N, C)

        Returns:
            scaled_logits: 温度缩放后的logits
        """
        return logits / self.temperature

    def fit(self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            lr: float = 0.01,
            max_iter: int = 50,
            verbose: bool = True) -> float:
        """
        在验证集上优化温度参数

        目标：最小化 NLL（负对数似然）

        Args:
            logits: 验证集模型输出 (N, C)
            labels: 验证集真实标签 (N,)
            lr: 学习率
            max_iter: 最大迭代次数

        Returns:
            best_temperature: 最优温度值
        """
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn='strong_wolfe'
        )

        losses = []

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            losses.append(loss.item())
            return loss

        if verbose:
            print(f"优化温度参数...")

        optimizer.step(eval_loss)

        optimal_temp = self.temperature.item()

        if verbose:
            print(f"  初始温度: 1.0")
            print(f"  最优温度: {optimal_temp:.4f}")
            print(f"  初始NLL: {losses[0]:.4f}")
            print(f"  最终NLL: {losses[-1]:.4f}")

        return optimal_temp

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        校准logits并返回概率

        Args:
            logits: 原始logits (N, C)

        Returns:
            calibrated_probs: 校准后的概率 (N, C)
        """
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            probs = F.softmax(scaled_logits, dim=-1)
        return probs


def compute_ece(confidences: np.ndarray,
                accuracies: np.ndarray,
                n_bins: int = 10) -> float:
    """
    计算期望校准误差 (Expected Calibration Error)

    ECE = sum(bin_weight * |bin_accuracy - bin_confidence|)

    Args:
        confidences: 模型预测置信度 (N,) - 预测概率的最大值
        accuracies: 二元正确数组 (N,) - 1表示预测正确，0表示错误
        n_bins: 分箱数量

    Returns:
        ece: 期望校准误差
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        # 当前bin的掩码
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:  # 最后一个bin包含边界
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_weight = mask.sum() / len(confidences)

            ece += bin_weight * abs(bin_acc - bin_conf)

            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(mask.sum())

    return ece, bin_accs, bin_confs, bin_counts


def evaluate_calibration(logits: torch.Tensor,
                         labels: torch.Tensor,
                         n_bins: int = 10) -> Dict:
    """
    评估模型校准度

    Args:
        logits: 模型输出 (N, C)
        labels: 真实标签 (N,)
        n_bins: 分箱数量

    Returns:
        metrics: 包含ECE等指标的字典
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)

    confidences = confidences.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracies = (predictions == labels_np).astype(float)
    accuracy = accuracies.mean()
    avg_confidence = confidences.mean()

    ece, bin_accs, bin_confs, bin_counts = compute_ece(
        confidences, accuracies, n_bins
    )

    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'confidence_gap': abs(avg_confidence - accuracy),
        'ece': ece,
        'bin_accuracies': bin_accs,
        'bin_confidences': bin_confs,
        'bin_counts': bin_counts
    }


def plot_reliability_diagram(metrics_before: Dict,
                              metrics_after: Dict,
                              output_path: str):
    """
    绘制可靠性图对比校准前后效果

    Args:
        metrics_before: 校准前指标
        metrics_after: 校准后指标
        output_path: 输出图片路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 校准前
    bin_confs_before = metrics_before['bin_confidences']
    bin_accs_before = metrics_before['bin_accuracies']

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.plot(bin_confs_before, bin_accs_before, 'o-', label='Model')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Before Calibration\nECE: {metrics_before["ece"]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 校准后
    bin_confs_after = metrics_after['bin_confidences']
    bin_accs_after = metrics_after['bin_accuracies']

    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax2.plot(bin_confs_after, bin_accs_after, 'o-', color='green', label='Calibrated')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'After Temperature Scaling\nECE: {metrics_after["ece"]:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可靠性图已保存: {output_path}")


def collect_logits_and_labels(model, dataloader, device='cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    收集模型在数据集上的logits和标签

    Args:
        model: 待评估模型
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        all_logits: (N, C)
        all_labels: (N,)
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="收集logits"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 取最后一个token的logits
            logits = outputs.logits[:, -1, :3].cpu()  # 假设3分类

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='温度缩放校准')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--val_data', required=True, help='验证数据路径')
    parser.add_argument('--test_data', help='测试数据路径（可选）')
    parser.add_argument('--output_dir', default='./results', help='输出目录')
    parser.add_argument('--n_bins', type=int, default=10, help='ECE分箱数')
    parser.add_argument('--max_iter', type=int, default=50, help='优化迭代次数')
    args = parser.parse_args()

    print("="*60)
    print("温度缩放校准")
    print("="*60)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # TODO: 加载你的模型和数据
    # model = ...
    # val_loader = ...

    print("\n1. 收集验证集logits...")
    # val_logits, val_labels = collect_logits_and_labels(model, val_loader)

    # 示例数据（实际使用时删除）
    val_logits = torch.randn(1000, 3)
    val_labels = torch.randint(0, 3, (1000,))

    print(f"   样本数: {len(val_labels)}")

    print("\n2. 评估校准前状态...")
    metrics_before = evaluate_calibration(val_logits, val_labels, args.n_bins)
    print(f"   准确率: {metrics_before['accuracy']:.4f}")
    print(f"   平均置信度: {metrics_before['avg_confidence']:.4f}")
    print(f"   ECE: {metrics_before['ece']:.4f}")

    print("\n3. 学习最优温度...")
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(val_logits, val_labels, max_iter=args.max_iter)

    print("\n4. 评估校准后状态...")
    calibrated_logits = scaler(val_logits)
    metrics_after = evaluate_calibration(calibrated_logits, val_labels, args.n_bins)
    print(f"   准确率: {metrics_after['accuracy']:.4f}")
    print(f"   平均置信度: {metrics_after['avg_confidence']:.4f}")
    print(f"   ECE: {metrics_after['ece']:.4f}")
    print(f"   ECE降低: {metrics_before['ece'] - metrics_after['ece']:.4f}")

    print("\n5. 生成可视化...")
    plot_path = Path(args.output_dir) / 'reliability_diagram.png'
    plot_reliability_diagram(metrics_before, metrics_after, str(plot_path))

    # 保存校准器
    scaler_path = Path(args.output_dir) / 'temperature_scaler.pt'
    torch.save({'temperature': scaler.temperature}, scaler_path)
    print(f"\n校准器已保存: {scaler_path}")

    # 如果有测试集，评估测试集
    if args.test_data:
        print("\n6. 在测试集上评估...")
        # test_logits, test_labels = collect_logits_and_labels(model, test_loader)
        # ... 类似流程

    print("\n校准完成!")


if __name__ == '__main__':
    main()
