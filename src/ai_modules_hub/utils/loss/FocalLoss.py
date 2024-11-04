import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    FocalLoss 是一种用于处理类别不平衡问题的损失函数。

    论文地址:
        https://arxiv.org/abs/1708.02002

    优点:
        - FocalLoss 可以有效地处理类别不平衡问题，通过调整易分类样本的权重，减少其对损失的影响。
        - 通过引入调节因子 γ，FocalLoss 可以更好地聚焦难分类样本，提高模型的泛化能力。
        - FocalLoss 可以与现有的目标检测模型（如 SSD、RetinaNet 等）无缝集成，提升检测性能。

    示例:
        >>> import torch
        >>> from ai_modules_hub.utils.loss import FocalLoss
        >>> criterion = FocalLoss(gamma=2, alpha=[0.25, 0.75])
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> loss = criterion(input, target)
        >>> loss.backward()

    参数:
        alpha (float or list of float, optional): 类别权重。当 alpha 是列表时，为各类别权重；
            当 alpha 为常数时，类别权重为 [α, 1-α, 1-α, ...]。常用于目标检测算法中抑制背景类。
            默认值为 None。
        gamma (float, optional): 难易样本调节参数。默认值为 2。
        num_classes (int, optional): 类别数量。默认值为 2。
        size_average (bool, optional): 损失计算方式，是否取均值。默认值为 True。
    """
    def __init__(self, alpha=None, gamma=2, num_classes=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入, size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        计算 FocalLoss 损失。

        参数:
            preds (torch.Tensor): 预测类别。尺寸为 [B, N, C] 或 [B, C]，分别对应检测和分类任务。
                B 为批次大小，N 为检测框数，C 为类别数。
            labels (torch.Tensor): 实际类别。尺寸为 [B, N] 或 [B]。

        返回:
            torch.Tensor: 计算得到的损失值。
        """
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现 nll_loss (cross entropy = log_softmax + nll)
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1 - preds_softmax), self.gamma) 为 focal loss 中 (1 - pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
