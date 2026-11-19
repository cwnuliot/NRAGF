# -*- coding: utf-8 -*-
"""
NRAGF 正式消融网络

保存位置：
    NRAGF/models/nragf_ablation.py

四种模式：
1. BASELINE
   原始节点平均 -> CMSA -> MGATE -> TriPool

2. MNRL
   掩码监督节点可靠性 + 非归一化可靠性聚合

3. MNRL_RNA
   MNRL + 可靠性归一化节点聚合

4. FULL
   MNRL_RNA + 双退化感知残差融合

注意：
- 四种模式共用同一骨干、辅助分类头、CMSA、MGATE 和 TriPool；
- 训练协议由统一训练脚本控制；
- FULL 与当前 NRAGF-v2 的方法逻辑一致。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import AG_STGCN, CMSA, MGATE, TriPool
from utils.graph import Graph_RGB, Graph_RFID


VALID_ABLATION_MODES = (
    "BASELINE",
    "MNRL",
    "MNRL_RNA",
    "FULL",
)


def normalize_ablation_mode(mode: str) -> str:
    value = str(mode).strip().upper()

    aliases = {
        "A": "MNRL",
        "B": "MNRL_RNA",
        "NRAGF": "FULL",
        "NRAGF_V2": "FULL",
    }
    value = aliases.get(value, value)

    if value not in VALID_ABLATION_MODES:
        raise ValueError(
            f"Invalid ablation mode: {mode}. "
            f"Choose from {VALID_ABLATION_MODES}."
        )

    return value


class PartAwareBackbone(AG_STGCN):
    """只池化时间维度，保留节点特征。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,T,V,C], got {tuple(x.shape)}"
            )

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

        if len(jk_features) != 3:
            raise RuntimeError(
                f"Expected 3 JK features, got {len(jk_features)}"
            )

        x = self.jk_layer(
            torch.cat(jk_features, dim=1)
        )

        return x.mean(dim=2)


class NodeTCP(nn.Module):
    """节点级可靠性预测器。"""

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
    """学习门控与模态可靠性先验相结合。"""

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

        weights = F.softmax(
            learned_logits
            + self.quality_prior_strength * torch.log(quality),
            dim=1,
        )

        fused = (
            weights[:, 0:1] * rgb_feature
            + weights[:, 1:2] * rfid_feature
        )

        return fused, weights


class NRAGFAblationNet(nn.Module):
    """统一的 NRAGF 消融模型。"""

    def __init__(
        self,
        graph_args: Dict,
        ablation_mode: str,
        num_class: int = 21,
        feature_dim: int = 512,
        edge_importance_weighting: bool = True,
        tcp_hidden_dim: int = 128,
        tcp_dropout: float = 0.20,
        pool_weight_floor: float = 0.05,
        pool_weight_power: float = 2.0,
        quality_prior_strength: float = 1.0,
        asymmetric_fusion_strength: float = 1.0,
        joint_degradation_strength: float = 1.0,
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        if num_class != 21:
            raise ValueError(
                "Current baseline.TriPool output is fixed to 21 classes."
            )

        self.ablation_mode = normalize_ablation_mode(
            ablation_mode
        )

        self.pool_weight_floor = float(pool_weight_floor)
        self.pool_weight_power = float(pool_weight_power)

        self.asymmetric_fusion_strength = float(
            asymmetric_fusion_strength
        )
        self.joint_degradation_strength = float(
            joint_degradation_strength
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

    @property
    def uses_tcp(self) -> bool:
        return self.ablation_mode != "BASELINE"

    @staticmethod
    def _physical_activity(
        raw_input: torch.Tensor,
        threshold: float = 1e-5,
    ) -> torch.Tensor:
        energy = raw_input.abs().reshape(
            raw_input.size(0),
            -1,
        ).mean(dim=1, keepdim=True)

        return (
            energy > threshold
        ).to(raw_input.dtype).unsqueeze(2)

    def _node_weights(
        self,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        confidence = confidence.clamp(0.0, 1.0)

        return (
            self.pool_weight_floor
            + (1.0 - self.pool_weight_floor)
            * confidence.pow(self.pool_weight_power)
        )

    @staticmethod
    def _non_normalized_pool(
        feature: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        非归一化可靠性聚合：
            mean(weight * feature)

        该形式会在大量节点低权重时缩小特征幅值，
        用于验证 RNA 重新归一化是否必要。
        """
        return (
            feature * weights
        ).mean(dim=2)

    @staticmethod
    def _normalized_pool(
        feature: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        可靠性归一化节点聚合：
            sum(weight * feature) / sum(weight)
        """
        numerator = (
            feature * weights
        ).sum(dim=2)

        denominator = weights.sum(
            dim=2
        ).clamp_min(1e-6)

        return numerator / denominator

    def _shared_fusion(
        self,
        rgb_feature: torch.Tensor,
        rfid_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_aligned, rfid_aligned = self.cmsa(
            rgb_feature,
            rfid_feature,
        )

        fused = self.mgate(
            rgb_aligned,
            rfid_aligned,
        )

        return fused, rgb_aligned, rfid_aligned

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_rfid: torch.Tensor,
        reliability_scale: float = 1.0,
        return_details: bool = False,
    ):
        scale = float(
            max(0.0, min(1.0, reliability_scale))
        )

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

        if self.uses_tcp:
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

        quality_rgb = conf_rgb.mean(dim=2)
        quality_rfid = conf_rfid.mean(dim=2)

        damage_rgb = 1.0 - quality_rgb
        damage_rfid = 1.0 - quality_rfid

        node_weights_rgb = self._node_weights(conf_rgb)
        node_weights_rfid = self._node_weights(conf_rfid)

        fused_original, _, _ = self._shared_fusion(
            vec_rgb_original,
            vec_rfid_original,
        )

        # ------------------------------------------------------------
        # BASELINE：无节点可靠性学习、无可靠性聚合、无退化融合。
        # ------------------------------------------------------------
        if self.ablation_mode == "BASELINE":
            fused = fused_original

            vec_rgb_selected = vec_rgb_original
            vec_rfid_selected = vec_rfid_original

            asymmetric_trigger = torch.zeros_like(
                quality_rgb
            )
            joint_trigger = torch.zeros_like(
                quality_rgb
            )
            fusion_weights = torch.full(
                (x_rgb.size(0), 2),
                0.5,
                device=x_rgb.device,
                dtype=x_rgb.dtype,
            )

        else:
            # --------------------------------------------------------
            # MNRL：非归一化节点可靠性聚合。
            # MNRL_RNA / FULL：可靠性归一化节点聚合。
            # --------------------------------------------------------
            if self.ablation_mode == "MNRL":
                vec_rgb_reliable = self._non_normalized_pool(
                    feat_rgb,
                    node_weights_rgb,
                )
                vec_rfid_reliable = self._non_normalized_pool(
                    feat_rfid,
                    node_weights_rfid,
                )
            else:
                vec_rgb_reliable = self._normalized_pool(
                    feat_rgb,
                    node_weights_rgb,
                )
                vec_rfid_reliable = self._normalized_pool(
                    feat_rfid,
                    node_weights_rfid,
                )

            # 保留原始特征旁路；可靠性越低，可靠节点路径参与越多。
            vec_rgb_selected = (
                vec_rgb_original
                + scale
                * damage_rgb
                * (
                    vec_rgb_reliable
                    - vec_rgb_original
                )
            )
            vec_rfid_selected = (
                vec_rfid_original
                + scale
                * damage_rfid
                * (
                    vec_rfid_reliable
                    - vec_rfid_original
                )
            )

            fused_reliable, rgb_aligned, rfid_aligned = (
                self._shared_fusion(
                    vec_rgb_selected,
                    vec_rfid_selected,
                )
            )

            # MNRL 和 MNRL_RNA 到此为止，不使用 DDRF。
            if self.ablation_mode in (
                "MNRL",
                "MNRL_RNA",
            ):
                fused = fused_reliable

                asymmetric_trigger = torch.zeros_like(
                    quality_rgb
                )
                joint_trigger = torch.zeros_like(
                    quality_rgb
                )
                fusion_weights = torch.full(
                    (x_rgb.size(0), 2),
                    0.5,
                    device=x_rgb.device,
                    dtype=x_rgb.dtype,
                )

            # --------------------------------------------------------
            # FULL：增加双退化感知残差融合 DDRF。
            # --------------------------------------------------------
            else:
                fused_quality, fusion_weights = (
                    self.quality_gate(
                        rgb_feature=rgb_aligned,
                        rfid_feature=rfid_aligned,
                        rgb_quality=quality_rgb,
                        rfid_quality=quality_rfid,
                    )
                )

                asymmetric_trigger = torch.abs(
                    quality_rgb - quality_rfid
                ).clamp(0.0, 1.0)

                joint_trigger = (
                    damage_rgb * damage_rfid
                ).clamp(0.0, 1.0)

                fused_asymmetric = (
                    fused_original
                    + scale
                    * self.asymmetric_fusion_strength
                    * asymmetric_trigger
                    * (
                        fused_quality
                        - fused_original
                    )
                )

                fused = (
                    fused_asymmetric
                    + scale
                    * self.joint_degradation_strength
                    * joint_trigger
                    * (
                        fused_reliable
                        - fused_asymmetric
                    )
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
            "ablation_mode": self.ablation_mode,
            "feat_rgb": feat_rgb,
            "feat_rfid": feat_rfid,
            "quality_rgb": quality_rgb,
            "quality_rfid": quality_rfid,
            "damage_rgb": damage_rgb,
            "damage_rfid": damage_rfid,
            "node_weights_rgb": node_weights_rgb,
            "node_weights_rfid": node_weights_rfid,
            "vec_rgb_original": vec_rgb_original,
            "vec_rfid_original": vec_rfid_original,
            "vec_rgb_selected": vec_rgb_selected,
            "vec_rfid_selected": vec_rfid_selected,
            "asymmetric_trigger": asymmetric_trigger,
            "joint_trigger": joint_trigger,
            "fusion_weights": fusion_weights,
            "fused_original": fused_original,
            "fused_final": fused,
        }

        return result + (details,)
