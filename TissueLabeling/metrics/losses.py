"""
File: losses.py
Author: Sabeen Lohawala
Date: 2024-03-21
Description: This file contains the functional and class implementations of the softmax focal loss.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import NLLLoss


class SoftmaxFocalLoss(nn.Module):
    """
    Multi-class version of sigmoid_focal_loss from:
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """

    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = "mean"):
        """
        Constructor.

        Args:
            alpha (float): (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            reduction (str): 'none' | 'mean' | 'sum'
                       'none': No reduction will be applied to the output.
                       'mean': The output will be averaged.
                       'sum': The output will be summed.
        """
        super(SoftmaxFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, mask, probs):
        """
        Computes the softmax focal loss.
        Args
            mask: A tensor of shape (batch_size, nr_of_classes, height, width) containing
                  integer class numbers for each pixel.
            probs: A float tensor with the same shape as inputs. Stores the softmax output
                   probabilities for each class.

        Returns:
            Loss tensor with the reduction option applied.
        """
        return softmax_focal_loss(
            mask, probs, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )


def softmax_focal_loss(
    mask: torch.Tensor,
    probs: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Functional multi-class version of sigmoid_focal_loss from:
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        mask (torch.Tensor): A tensor of shape (batch_size, nr_of_classes, height, width) containing
              integer class numbers for each pixel.
        probs (torch.Tensor): A float tensor with the same shape as inputs. Stores the softmax output
               probabilities for each class.
        alpha (float): (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma (float): Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction (str): 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """

    # Reshape image tensor
    targets = mask.view(
        -1
    )  # Shape: (batch_size, 1, height, width) -> (batch_size * height * width,)

    # Reshape softmax output
    p = probs.permute(0, 2, 3, 1).contiguous()  # Move the channel dimension to the end
    p = p.view(
        -1, probs.shape[1]
    )  # Shape: (batch_size, nr_of_classes, height, width) -> (batch_size * height * width, nr_of_classes)

    loss_fn = NLLLoss(reduction="none")
    ce_loss = loss_fn(p.log(), targets)
    p_t = p[
        torch.arange(targets.size(0)), targets
    ]  # get the probabilities corresponding to the true label
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
