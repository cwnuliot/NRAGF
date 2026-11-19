# -*- coding: utf-8 -*-
"""
QMF adapted to RGB-skeleton + RFID for the NRAGF project.

Reference idea:
Zhang et al., "Provable Dynamic Fusion for Low-Quality Multimodal Data", ICML 2023.

Important:
- This is an RGB-RFID adaptation, NOT a verbatim copy of the authors' RGB-D code.
- The modality backbones are deliberately replaced by the same AG-STGCN backbone
  used by the NRAGF project, so the comparison focuses on fusion/reliability.
- QMF core is preserved: modality classifiers -> energy confidence -> dynamic late fusion.
"""

import torch
import torch.nn as nn

from models.baseline import AG_STGCN
from utils.graph import Graph_RGB, Graph_RFID


class QMFRGBRFID(nn.Module):
    def __init__(
        self,
        graph_args,
        num_class=21,
        dropout=0.50,
        edge_importance_weighting=True,
        confidence_scale=0.10,
    ):
        super().__init__()
        self.confidence_scale = float(confidence_scale)

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

        self.rgb_classifier = nn.Linear(512, num_class)
        self.rfid_classifier = nn.Linear(512, num_class)

    def forward(self, rgb, rfid):
        rgb_feature = self.rgb_backbone(rgb)
        rfid_feature = self.rfid_backbone(rfid)

        rgb_logits = self.rgb_classifier(rgb_feature)
        rfid_logits = self.rfid_classifier(rfid_feature)

        # Official QMF RGB-D implementation:
        # energy = -logsumexp(logits), confidence = -0.1 * energy.
        rgb_conf = self.confidence_scale * torch.logsumexp(
            rgb_logits, dim=1, keepdim=True
        )
        rfid_conf = self.confidence_scale * torch.logsumexp(
            rfid_logits, dim=1, keepdim=True
        )

        # QMF dynamic late fusion.
        fused_logits = (
            rgb_logits * rgb_conf
            + rfid_logits * rfid_conf
        )

        return fused_logits, rgb_logits, rfid_logits, rgb_conf, rfid_conf
