# -*- coding: utf-8 -*-
"""
NRAGF-v2：面向高 K 严重缺失的可靠性归一化双退化融合网络

保存位置：
    NRAGF/models/nragf_highk_v2.py

相较于 NRAGF-v1 的两个关键修改：
1. 可靠性归一化节点聚合：
   不再对调制后的节点特征直接求平均，避免 K 增大时特征幅值持续缩小。

2. 双退化感知残差融合：
   分别处理“单模态质量失衡”和“双模态共同退化”，解决两种模态都损坏时
   abs(q_rgb - q_rfid) 反而变小、可靠性分支被错误关闭的问题。

重要原则：
- 不修改 baseline.py；
- 不使用 Generator；
- 保留原 AG-STGCN、CMSA、MGATE、TriPool；
- 原始全局融合路径始终保留，干净样本下尽量接近原 baseline。
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import AG_STGCN, CMSA, MGATE, TriPool
from utils.graph import Graph_RGB, Graph_RFID


class PartAwareBackbone(AG_STGCN):
    """复用 AG-STGCN，仅池化时间维度，保留节点维度。"""

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

        # [B, 512, T, V] -> [B, 512, V]
        return x.mean(dim=2)


class NodeTCP(nn.Module):
    """节点级真实类别概率/可靠性预测器。"""

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
    """学习门控与显式模态可靠性先验相结合。"""

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

        # 初始为等权，避免训练初期随机偏向某一模态。
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


class NRAGFHighKNet(nn.Module):
    """
    NRAGF-v2。

    原始稳定路径：
        节点特征 -> 普通平均池化 -> CMSA -> MGATE

    可靠性路径：
        节点特征 -> 可靠性归一化聚合 -> CMSA -> MGATE

    融合校正：
        1. 模态质量不均衡时，使用质量门控校正；
        2. 两种模态共同退化时，使用可靠节点聚合路径校正。
    """

    def __init__(
        self,
        graph_args: Dict,
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

        if not 0.0 <= pool_weight_floor < 1.0:
            raise ValueError(
                "pool_weight_floor must be in [0, 1)."
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

        # 两条路径共享同一组融合参数，避免额外引入两套融合网络。
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
        """
        仅在整个模态近乎全零时执行硬阻断。
        不把原数据中的自然零节点直接判为失效。
        """
        energy = raw_input.abs().reshape(
            raw_input.size(0),
            -1,
        ).mean(dim=1, keepdim=True)

        return (
            energy > threshold
        ).to(raw_input.dtype).unsqueeze(2)

    def _reliability_normalized_pool(
        self,
        feature: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        可靠性归一化节点聚合。

        旧方式：
            mean(gate * feature)
        受损节点越多，整体特征幅值越小。

        新方式：
            sum(weight * feature) / sum(weight)
        即使 K 很大、只剩少量可靠节点，也能保持特征尺度稳定。
        """
        confidence = confidence.clamp(0.0, 1.0)

        weights = (
            self.pool_weight_floor
            + (1.0 - self.pool_weight_floor)
            * confidence.pow(self.pool_weight_power)
        )

        numerator = (
            feature * weights
        ).sum(dim=2)

        denominator = weights.sum(
            dim=2
        ).clamp_min(1e-6)

        pooled = numerator / denominator

        return pooled, weights

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
        use_tcp: bool = True,
        use_quality_fusion: bool = True,
        return_details: bool = False,
    ):
        scale = float(
            max(0.0, min(1.0, reliability_scale))
        )

        feat_rgb = self.backbone_rgb(x_rgb)
        feat_rfid = self.backbone_rfid(x_rfid)

        # 原始全局向量：作为稳定锚点与辅助分类输入。
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

        quality_rgb = conf_rgb.mean(dim=2)
        quality_rfid = conf_rfid.mean(dim=2)

        damage_rgb = 1.0 - quality_rgb
        damage_rfid = 1.0 - quality_rfid

        # 可靠节点归一化聚合，不会因 K 增大而持续缩小特征幅值。
        vec_rgb_reliable, node_weights_rgb = (
            self._reliability_normalized_pool(
                feat_rgb,
                conf_rgb,
            )
        )
        vec_rfid_reliable, node_weights_rfid = (
            self._reliability_normalized_pool(
                feat_rfid,
                conf_rfid,
            )
        )

        # 干净时接近原始均值；退化越重，越依赖可靠节点聚合。
        vec_rgb_robust = (
            vec_rgb_original
            + scale
            * damage_rgb
            * (
                vec_rgb_reliable
                - vec_rgb_original
            )
        )
        vec_rfid_robust = (
            vec_rfid_original
            + scale
            * damage_rfid
            * (
                vec_rfid_reliable
                - vec_rfid_original
            )
        )

        # 原始稳定融合路径。
        (
            fused_original,
            rgb_original_aligned,
            rfid_original_aligned,
        ) = self._shared_fusion(
            vec_rgb_original,
            vec_rfid_original,
        )

        # 可靠节点融合路径。
        (
            fused_robust,
            rgb_robust_aligned,
            rfid_robust_aligned,
        ) = self._shared_fusion(
            vec_rgb_robust,
            vec_rfid_robust,
        )

        if use_quality_fusion:
            fused_quality, fusion_weights = (
                self.quality_gate(
                    rgb_feature=rgb_robust_aligned,
                    rfid_feature=rfid_robust_aligned,
                    rgb_quality=quality_rgb,
                    rfid_quality=quality_rfid,
                )
            )

            # 单模态退化：两个模态质量差距大。
            asymmetric_trigger = torch.abs(
                quality_rgb - quality_rfid
            ).clamp(0.0, 1.0)

            # 双模态共同退化：两个模态同时低质量。
            joint_trigger = (
                damage_rgb * damage_rfid
            ).clamp(0.0, 1.0)

            # 第一步：单模态失衡时，转向较可靠模态。
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

            # 第二步：两模态共同退化时，转向可靠节点归一化路径。
            fused = (
                fused_asymmetric
                + scale
                * self.joint_degradation_strength
                * joint_trigger
                * (
                    fused_robust
                    - fused_asymmetric
                )
            )
        else:
            fused = fused_original

            fusion_weights = torch.full(
                (x_rgb.size(0), 2),
                0.5,
                device=x_rgb.device,
                dtype=x_rgb.dtype,
            )

            asymmetric_trigger = torch.zeros_like(
                quality_rgb
            )
            joint_trigger = torch.zeros_like(
                quality_rgb
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
            "quality_rgb": quality_rgb,
            "quality_rfid": quality_rfid,
            "damage_rgb": damage_rgb,
            "damage_rfid": damage_rfid,
            "node_weights_rgb": node_weights_rgb,
            "node_weights_rfid": node_weights_rfid,
            "vec_rgb_original": vec_rgb_original,
            "vec_rfid_original": vec_rfid_original,
            "vec_rgb_reliable": vec_rgb_reliable,
            "vec_rfid_reliable": vec_rfid_reliable,
            "vec_rgb_robust": vec_rgb_robust,
            "vec_rfid_robust": vec_rfid_robust,
            "fusion_weights": fusion_weights,
            "asymmetric_trigger": asymmetric_trigger,
            "joint_trigger": joint_trigger,
            "fused_original": fused_original,
            "fused_robust": fused_robust,
            "fused_final": fused,
        }

        return result + (details,)


NRAGF_Net = NRAGFHighKNet
