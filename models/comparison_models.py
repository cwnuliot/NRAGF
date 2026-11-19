# -*- coding: utf-8 -*-
"""
统一对比模型
保存位置：NRAGF/models/comparison_models.py
"""

from typing import Dict

import torch
import torch.nn as nn

from models.baseline import AG_STGCN, CMSA, MGATE, TriPool
from utils.graph import Graph_RGB, Graph_RFID


class DualStreamBackbone(nn.Module):
    def __init__(
        self,
        graph_args: Dict,
        num_class: int = 21,
        feature_dim: int = 512,
        edge_importance_weighting: bool = True,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        self.rgb_backbone = AG_STGCN(
            2,
            Graph_RGB(**graph_args),
            None,
            edge_importance_weighting,
            **backbone_kwargs,
        )
        self.rfid_backbone = AG_STGCN(
            3,
            Graph_RFID(**graph_args),
            None,
            edge_importance_weighting,
            **backbone_kwargs,
        )

        self.aux_cls_rgb = nn.Linear(feature_dim, num_class)
        self.aux_cls_rfid = nn.Linear(feature_dim, num_class)

    def extract_features(
        self,
        rgb: torch.Tensor,
        rfid: torch.Tensor,
    ):
        feat_rgb = self.rgb_backbone(rgb)
        feat_rfid = self.rfid_backbone(rfid)

        logits_rgb = self.aux_cls_rgb(feat_rgb)
        logits_rfid = self.aux_cls_rfid(feat_rfid)

        return feat_rgb, feat_rfid, logits_rgb, logits_rfid


class MFMeanNet(DualStreamBackbone):
    """
    MF-Mean:
    双流 AG-STGCN -> 特征均值 -> TriPool
    """

    def __init__(
        self,
        graph_args: Dict,
        num_class: int = 21,
        feature_dim: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(
            graph_args=graph_args,
            num_class=num_class,
            feature_dim=feature_dim,
            **kwargs,
        )
        self.tripool = TriPool(feature_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        rfid: torch.Tensor,
    ):
        feat_rgb, feat_rfid, logits_rgb, logits_rfid = (
            self.extract_features(rgb, rfid)
        )

        fused = 0.5 * (feat_rgb + feat_rfid)
        output = self.tripool(fused)

        return output, logits_rgb, logits_rfid


class LFCMSANet(DualStreamBackbone):
    """
    LF-CMSA:
    双流 AG-STGCN -> CMSA -> MGATE -> TriPool
    """

    def __init__(
        self,
        graph_args: Dict,
        num_class: int = 21,
        feature_dim: int = 512,
        num_heads: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            graph_args=graph_args,
            num_class=num_class,
            feature_dim=feature_dim,
            **kwargs,
        )

        self.cmsa = CMSA(dim=feature_dim, heads=num_heads)
        self.mgate = MGATE(dim=feature_dim)
        self.tripool = TriPool(feature_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        rfid: torch.Tensor,
    ):
        feat_rgb, feat_rfid, logits_rgb, logits_rfid = (
            self.extract_features(rgb, rfid)
        )

        feat_rgb, feat_rfid = self.cmsa(
            feat_rgb,
            feat_rfid,
        )
        fused = self.mgate(
            feat_rgb,
            feat_rfid,
        )
        output = self.tripool(fused)

        return output, logits_rgb, logits_rfid
