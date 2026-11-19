# -*- coding: utf-8 -*-
"""
TMC adapted to RGB-skeleton + RFID for the NRAGF project.

Reference:
Han et al., "Trusted Multi-View Classification", ICLR 2021.

This is a task adaptation rather than a verbatim reproduction:
- The original tabular/multi-view classifiers are replaced by the same AG-STGCN
  backbones used by the NRAGF comparison pipeline.
- TMC's core mechanism is preserved:
    1) each modality outputs non-negative class evidence;
    2) evidence + 1 parameterizes a Dirichlet distribution;
    3) modality opinions are fused by Dempster-Shafer combination;
    4) each modality and the fused opinion are optimized by the original
       evidential classification objective with KL annealing.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import AG_STGCN
from utils.graph import Graph_RGB, Graph_RFID


class EvidenceHead(nn.Module):
    """
    Official TMC classifier terminates in Softplus so that class evidence >= 0.
    With a 512-D shared backbone feature, the task-adapted head is simply:
        Linear(512 -> num_class) + Softplus
    """
    def __init__(self, in_dim: int, num_class: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_class)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.fc(x))


def ds_combine_two(
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    num_class: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Dempster-Shafer combination used by TMC.

    alpha = evidence + 1
    b = evidence / sum(alpha)
    u = K / sum(alpha)

    The small clamp on (1-C) is only a numerical-safety guard.
    """
    s1 = alpha1.sum(dim=1, keepdim=True)
    s2 = alpha2.sum(dim=1, keepdim=True)

    e1 = alpha1 - 1.0
    e2 = alpha2 - 1.0

    b1 = e1 / s1
    b2 = e2 / s2

    u1 = float(num_class) / s1
    u2 = float(num_class) / s2

    bb = torch.bmm(
        b1.unsqueeze(2),
        b2.unsqueeze(1),
    )

    conflict = (
        bb.sum(dim=(1, 2))
        - torch.diagonal(
            bb,
            dim1=-2,
            dim2=-1,
        ).sum(dim=-1)
    )

    denominator = (1.0 - conflict).clamp_min(eps).unsqueeze(1)

    combined_b = (
        b1 * b2
        + b1 * u2
        + b2 * u1
    ) / denominator

    combined_u = (
        u1 * u2
    ) / denominator

    combined_s = float(num_class) / combined_u.clamp_min(eps)
    combined_e = combined_b * combined_s
    combined_alpha = combined_e + 1.0

    return combined_alpha


def tmc_kl_divergence(
    alpha: torch.Tensor,
    num_class: int,
) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ), matching the official TMC implementation.
    Returns [B, 1].
    """
    alpha = alpha.float()
    beta = torch.ones(
        (1, num_class),
        device=alpha.device,
        dtype=alpha.dtype,
    )

    s_alpha = alpha.sum(dim=1, keepdim=True)
    s_beta = beta.sum(dim=1, keepdim=True)

    ln_b = (
        torch.lgamma(s_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    )
    ln_b_uni = (
        torch.lgamma(beta).sum(dim=1, keepdim=True)
        - torch.lgamma(s_beta)
    )

    dg0 = torch.digamma(s_alpha)
    dg1 = torch.digamma(alpha)

    kl = (
        ((alpha - beta) * (dg1 - dg0)).sum(
            dim=1,
            keepdim=True,
        )
        + ln_b
        + ln_b_uni
    )
    return kl


def tmc_ce_loss(
    labels: torch.Tensor,
    alpha: torch.Tensor,
    num_class: int,
    global_step: int,
    annealing_step: int,
) -> torch.Tensor:
    """
    Original TMC evidential classification loss:
        A = E_{Dir(alpha)}[-log p_y]
        B = annealed KL regularization

    Returns [B, 1].
    """
    alpha = alpha.float()
    labels = labels.long()

    s = alpha.sum(dim=1, keepdim=True)
    evidence = alpha - 1.0

    one_hot = F.one_hot(
        labels,
        num_classes=num_class,
    ).to(dtype=alpha.dtype)

    a_term = (
        one_hot
        * (
            torch.digamma(s)
            - torch.digamma(alpha)
        )
    ).sum(dim=1, keepdim=True)

    if annealing_step <= 0:
        annealing_coef = 1.0
    else:
        annealing_coef = min(
            1.0,
            float(global_step) / float(annealing_step),
        )

    # Remove evidence for the target class before KL regularization,
    # exactly following the original TMC loss construction.
    adjusted_alpha = (
        evidence * (1.0 - one_hot)
        + 1.0
    )

    b_term = (
        annealing_coef
        * tmc_kl_divergence(
            adjusted_alpha,
            num_class,
        )
    )

    return a_term + b_term


def tmc_total_loss(
    labels: torch.Tensor,
    alpha_rgb: torch.Tensor,
    alpha_rfid: torch.Tensor,
    alpha_fused: torch.Tensor,
    num_class: int,
    global_step: int,
    annealing_step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Official TMC multitask structure:
        L = L_rgb + L_rfid + L_fused
    """
    loss_rgb = tmc_ce_loss(
        labels,
        alpha_rgb,
        num_class,
        global_step,
        annealing_step,
    )
    loss_rfid = tmc_ce_loss(
        labels,
        alpha_rfid,
        num_class,
        global_step,
        annealing_step,
    )
    loss_fused = tmc_ce_loss(
        labels,
        alpha_fused,
        num_class,
        global_step,
        annealing_step,
    )

    total = (
        loss_rgb
        + loss_rfid
        + loss_fused
    ).mean()

    details = {
        "loss_rgb": float(loss_rgb.mean().detach().item()),
        "loss_rfid": float(loss_rfid.mean().detach().item()),
        "loss_fused": float(loss_fused.mean().detach().item()),
    }
    return total, details


class TMCRGBRFID(nn.Module):
    """
    TMC adapted to two modalities:
        view 1 = RGB skeleton
        view 2 = RFID

    Evaluation convention:
        forward(...)[0] is fused class evidence, so the existing unified
        evaluation script can use argmax exactly as it does for other methods.
    """

    def __init__(
        self,
        graph_args,
        num_class: int = 21,
        dropout: float = 0.50,
        edge_importance_weighting: bool = True,
        ds_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_class = int(num_class)
        self.ds_eps = float(ds_eps)

        # Same modality backbones as NRAGF/MF/LF for a controlled comparison.
        self.rgb_backbone = AG_STGCN(
            in_channels=2,
            graph=Graph_RGB(**graph_args),
            num_class=None,
            edge_importance_weighting=edge_importance_weighting,
            dropout=dropout,
        )
        self.rfid_backbone = AG_STGCN(
            in_channels=3,
            graph=Graph_RFID(**graph_args),
            num_class=None,
            edge_importance_weighting=edge_importance_weighting,
            dropout=dropout,
        )

        self.rgb_evidence_head = EvidenceHead(
            in_dim=512,
            num_class=self.num_class,
        )
        self.rfid_evidence_head = EvidenceHead(
            in_dim=512,
            num_class=self.num_class,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        rfid: torch.Tensor,
    ):
        rgb_feature = self.rgb_backbone(rgb)
        rfid_feature = self.rfid_backbone(rfid)

        evidence_rgb = self.rgb_evidence_head(
            rgb_feature
        )
        evidence_rfid = self.rfid_evidence_head(
            rfid_feature
        )

        alpha_rgb = evidence_rgb + 1.0
        alpha_rfid = evidence_rfid + 1.0

        alpha_fused = ds_combine_two(
            alpha_rgb,
            alpha_rfid,
            num_class=self.num_class,
            eps=self.ds_eps,
        )

        evidence_fused = alpha_fused - 1.0

        uncertainty_rgb = (
            float(self.num_class)
            / alpha_rgb.sum(
                dim=1,
                keepdim=True,
            )
        )
        uncertainty_rfid = (
            float(self.num_class)
            / alpha_rfid.sum(
                dim=1,
                keepdim=True,
            )
        )
        uncertainty_fused = (
            float(self.num_class)
            / alpha_fused.sum(
                dim=1,
                keepdim=True,
            )
        )

        return (
            evidence_fused,
            evidence_rgb,
            evidence_rfid,
            alpha_fused,
            alpha_rgb,
            alpha_rfid,
            uncertainty_fused,
            uncertainty_rgb,
            uncertainty_rfid,
        )
