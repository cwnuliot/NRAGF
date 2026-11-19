# -*- coding: utf-8 -*-
"""
NRAGF-Net
Node-wise Reliability-Aware Gated Fusion Network

保存位置：
    NRAGF/models/nragf_optimized.py

设计原则：
1. 不使用 Generator；
2. 不修改原 baseline.py；
3. 保留原 AG-STGCN、CMSA、MGATE、TriPool 主路径；
4. 新增节点级 TCP 监督、平滑可靠性调制和可靠性引导融合；
5. 通过残差方式接入新模块，干净输入时尽量接近原 baseline。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import AG_STGCN, CMSA, MGATE, TriPool
from utils.graph import Graph_RGB, Graph_RFID


class PartAwareBackbone(AG_STGCN):
    """复用原 AG-STGCN，只池化时间维度，保留节点维度。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, channels = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, num_nodes * channels, time_steps)
        x = self.data_bn(x)

        x = x.view(
            batch_size,
            num_nodes,
            channels,
            time_steps,
        )
        x = x.permute(0, 2, 3, 1).contiguous()

        jk_features = []

        for index, (gcn, bn, importance) in enumerate(
            zip(
                self.st_gcn_networks,
                self.bn_layers,
                self.edge_importance,
            )
        ):
            x = gcn(x, self.A * importance)
            x = F.relu(bn(x), inplace=True)

            if index in (7, 8, 9):
                jk_features.append(x)

        x = self.jk_layer(
            torch.cat(jk_features, dim=1)
        )

        return x.mean(dim=2)


class NodeTCP(nn.Module):
    """节点级 TCP 预测器。"""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                feature_dim,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_dim,
                hidden_dim // 2,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_dim // 2,
                1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)


class QualityGuidedGate(nn.Module):
    """
    学习门控 + 显式可靠性先验。

    learned_logits：
        从两模态特征中学习融合偏好。

    quality_logits：
        由节点 TCP 的模态平均值得到显式可靠性先验。
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        quality_prior_strength: float = 1.0,
    ) -> None:
        super().__init__()

        self.quality_prior_strength = float(
            quality_prior_strength
        )

        self.learned_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        nn.init.zeros_(self.learned_gate[-1].weight)
        nn.init.zeros_(self.learned_gate[-1].bias)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        rfid_feature: torch.Tensor,
        rgb_quality: torch.Tensor,
        rfid_quality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        learned_logits = self.learned_gate(
            torch.cat(
                [rgb_feature, rfid_feature],
                dim=1,
            )
        )

        quality = torch.cat(
            [rgb_quality, rfid_quality],
            dim=1,
        ).clamp_min(1e-4)

        quality_logits = torch.log(quality)

        weights = F.softmax(
            learned_logits
            + self.quality_prior_strength * quality_logits,
            dim=1,
        )

        fused = (
            weights[:, 0:1] * rgb_feature
            + weights[:, 1:2] * rfid_feature
        )
        return fused, weights


class NRAGFNet(nn.Module):
    """
    主路径：
        Part-Aware AG-STGCN
        -> 原 CMSA
        -> 原 MGATE
        -> 原 TriPool

    可靠性增强：
        节点 TCP
        -> 平滑节点调制
        -> 可靠性引导融合
        -> 残差接回原 MGATE 输出
    """

    def __init__(
        self,
        graph_args: Dict,
        num_class: int = 21,
        feature_dim: int = 512,
        edge_importance_weighting: bool = True,
        tcp_hidden_dim: int = 128,
        tcp_dropout: float = 0.20,
        suppress_strength: float = 0.70,
        suppress_power: float = 2.0,
        quality_prior_strength: float = 1.0,
        fusion_residual_strength: float = 1.0,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        if num_class != 21:
            raise ValueError(
                "Current baseline.TriPool output is fixed to 21 classes."
            )

        self.suppress_strength = float(suppress_strength)
        self.suppress_power = float(suppress_power)
        self.fusion_residual_strength = float(
            fusion_residual_strength
        )

        self.backbone_rgb = PartAwareBackbone(
            in_channels=2,
            graph=Graph_RGB(**graph_args),
            num_class=None,
            edge_importance_weighting=edge_importance_weighting,
            **backbone_kwargs,
        )

        self.backbone_rfid = PartAwareBackbone(
            in_channels=3,
            graph=Graph_RFID(**graph_args),
            num_class=None,
            edge_importance_weighting=edge_importance_weighting,
            **backbone_kwargs,
        )

        self.tcp_rgb = NodeTCP(
            feature_dim=feature_dim,
            hidden_dim=tcp_hidden_dim,
            dropout=tcp_dropout,
        )
        self.tcp_rfid = NodeTCP(
            feature_dim=feature_dim,
            hidden_dim=tcp_hidden_dim,
            dropout=tcp_dropout,
        )

        self.aux_cls_rgb = nn.Linear(
            feature_dim,
            num_class,
        )
        self.aux_cls_rfid = nn.Linear(
            feature_dim,
            num_class,
        )

        self.cmsa = CMSA(
            dim=feature_dim,
            heads=8,
        )
        self.mgate = MGATE(
            dim=feature_dim,
        )
        self.tripool = TriPool(
            feature_dim,
        )

        self.quality_gate = QualityGuidedGate(
            feature_dim=feature_dim,
            quality_prior_strength=quality_prior_strength,
        )

    @staticmethod
    def _physical_activity(
        raw_input: torch.Tensor,
        threshold: float = 1e-5,
    ) -> torch.Tensor:
        """只检测整个模态是否全盲。"""
        energy = raw_input.abs().reshape(
            raw_input.size(0),
            -1,
        ).mean(dim=1, keepdim=True)

        return (
            energy > threshold
        ).to(raw_input.dtype).unsqueeze(2)

    def _smooth_modulation(
        self,
        feature: torch.Tensor,
        confidence: torch.Tensor,
        reliability_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        gate = 1 - s * (1 - confidence)^p
        """
        robust_gate = (
            1.0
            - self.suppress_strength
            * torch.pow(
                1.0 - confidence,
                self.suppress_power,
            )
        )

        modulated = robust_gate * feature

        scale = float(
            max(0.0, min(1.0, reliability_scale))
        )

        clean_feature = (
            feature
            + scale * (modulated - feature)
        )

        return clean_feature, robust_gate

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_rfid: torch.Tensor,
        reliability_scale: float = 1.0,
        use_tcp: bool = True,
        use_quality_fusion: bool = True,
        return_details: bool = False,
    ):
        feat_rgb = self.backbone_rgb(x_rgb)
        feat_rfid = self.backbone_rfid(x_rfid)

        vec_rgb_original = feat_rgb.mean(dim=2)
        vec_rfid_original = feat_rfid.mean(dim=2)

        logits_rgb = self.aux_cls_rgb(
            vec_rgb_original
        )
        logits_rfid = self.aux_cls_rfid(
            vec_rfid_original
        )

        if use_tcp:
            conf_rgb = (
                self.tcp_rgb(feat_rgb)
                * self._physical_activity(x_rgb)
            )
            conf_rfid = (
                self.tcp_rfid(feat_rfid)
                * self._physical_activity(x_rfid)
            )
        else:
            conf_rgb = torch.ones_like(
                feat_rgb[:, 0:1, :]
            )
            conf_rfid = torch.ones_like(
                feat_rfid[:, 0:1, :]
            )

        feat_rgb_clean, gate_rgb = self._smooth_modulation(
            feat_rgb,
            conf_rgb,
            reliability_scale,
        )
        feat_rfid_clean, gate_rfid = self._smooth_modulation(
            feat_rfid,
            conf_rfid,
            reliability_scale,
        )

        vec_rgb = feat_rgb_clean.mean(dim=2)
        vec_rfid = feat_rfid_clean.mean(dim=2)

        vec_rgb, vec_rfid = self.cmsa(
            vec_rgb,
            vec_rfid,
        )

        fused_baseline = self.mgate(
            vec_rgb,
            vec_rfid,
        )

        quality_rgb = conf_rgb.mean(dim=2)
        quality_rfid = conf_rfid.mean(dim=2)

        if use_quality_fusion:
            fused_quality, fusion_weights = (
                self.quality_gate(
                    rgb_feature=vec_rgb,
                    rfid_feature=vec_rfid,
                    rgb_quality=quality_rgb,
                    rfid_quality=quality_rfid,
                )
            )

            imbalance = torch.abs(
                quality_rgb - quality_rfid
            )

            scale = float(
                max(0.0, min(1.0, reliability_scale))
            )

            fused = (
                fused_baseline
                + scale
                * self.fusion_residual_strength
                * imbalance
                * (fused_quality - fused_baseline)
            )
        else:
            fused = fused_baseline
            fusion_weights = torch.full(
                (x_rgb.size(0), 2),
                0.5,
                device=x_rgb.device,
                dtype=x_rgb.dtype,
            )

        output = self.tripool(fused)

        result = (
            output,
            conf_rgb,
            conf_rfid,
            logits_rgb,
            logits_rfid,
        )

        if not return_details:
            return result

        details = {
            "feat_rgb": feat_rgb,
            "feat_rfid": feat_rfid,
            "feat_rgb_clean": feat_rgb_clean,
            "feat_rfid_clean": feat_rfid_clean,
            "gate_rgb": gate_rgb,
            "gate_rfid": gate_rfid,
            "quality_rgb": quality_rgb,
            "quality_rfid": quality_rfid,
            "fusion_weights": fusion_weights,
            "fused_baseline": fused_baseline,
            "fused_final": fused,
        }

        return result + (details,)


NRAGF_Net = NRAGFNet
