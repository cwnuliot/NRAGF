# -*- coding: utf-8 -*-
"""
NRAGF 训练数据集

保存位置：
    NRAGF/data/dataset_nragf.py

说明：
1. 继续复用原 dataset2.py 的数据读取与 RGB/RFID 配对逻辑；
2. 额外返回节点级 clean mask，供 TCP 做真正的细粒度监督；
3. 支持两种训练干扰：
   - random_nodes：随机节点干扰，默认与原训练思路一致；
   - five_parts：按五个人体部位干扰，方便后续鲁棒性实验。
"""

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset2 import Get_RGB_datas, Get_RFID_datas


RGB_PARTS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (5, 7, 9),
    (6, 8, 10),
    (11, 13, 15),
    (12, 14, 16),
)

RFID_PARTS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3, 8),
    (4, 6),
    (5, 7),
    (9, 10, 12, 14),
    (11, 13, 15),
)


class NRAGFDataset(Dataset):
    """
    返回：
        rgb:        [165, 17, 2]
        rfid:       [50, 16, 3]
        label:      scalar
        rgb_mask:   [17]，正常节点=1，已知受损节点=0
        rfid_mask:  [16]，正常节点=1，已知受损节点=0
    """

    def __init__(
        self,
        file_list: Sequence[Sequence[str]],
        mode: str = "train",
        corruption_mode: str = "random_nodes",
        rgb_ratio: float = 0.30,
        rfid_ratio: float = 0.30,
        corruption_probability: float = 0.50,
        rgb_noise_std: float = 1.0,
        rfid_noise_std: float = 1.0,
        modality_dropout_probability: float = 0.15,
        part_k_values: Tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()

        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'.")

        if corruption_mode not in ("random_nodes", "five_parts"):
            raise ValueError(
                "corruption_mode must be 'random_nodes' or 'five_parts'."
            )

        self.file_list = list(file_list)
        self.mode = mode
        self.corruption_mode = corruption_mode

        self.rgb_ratio = float(rgb_ratio)
        self.rfid_ratio = float(rfid_ratio)
        self.corruption_probability = float(corruption_probability)

        self.rgb_noise_std = float(rgb_noise_std)
        self.rfid_noise_std = float(rfid_noise_std)

        self.modality_dropout_probability = float(
            modality_dropout_probability
        )
        self.part_k_values = tuple(int(value) for value in part_k_values)

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def _replace_nodes_with_noise(
        data: np.ndarray,
        node_indices: np.ndarray,
        noise_std: float,
    ) -> None:
        if node_indices.size == 0:
            return

        noise = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=(
                data.shape[0],
                node_indices.size,
                data.shape[2],
            ),
        ).astype(np.float32)

        data[:, node_indices, :] = noise

    @staticmethod
    def _random_node_indices(
        num_nodes: int,
        ratio: float,
    ) -> np.ndarray:
        number = int(round(num_nodes * ratio))
        number = min(max(number, 0), num_nodes)

        if number == 0:
            return np.empty((0,), dtype=np.int64)

        return np.asarray(
            np.random.choice(
                num_nodes,
                size=number,
                replace=False,
            ),
            dtype=np.int64,
        )

    @staticmethod
    def _part_node_indices(
        parts: Tuple[Tuple[int, ...], ...],
        k: int,
    ) -> np.ndarray:
        k = min(max(int(k), 1), len(parts))

        selected_parts = np.random.choice(
            len(parts),
            size=k,
            replace=False,
        )

        indices = []
        for part_id in selected_parts:
            indices.extend(parts[int(part_id)])

        return np.asarray(
            sorted(set(indices)),
            dtype=np.int64,
        )

    def _corrupt_rgb(
        self,
        rgb: np.ndarray,
        rgb_mask: np.ndarray,
    ) -> None:
        if self.corruption_mode == "random_nodes":
            indices = self._random_node_indices(
                num_nodes=17,
                ratio=self.rgb_ratio,
            )
        else:
            k = int(np.random.choice(self.part_k_values))
            indices = self._part_node_indices(
                RGB_PARTS,
                k,
            )

        self._replace_nodes_with_noise(
            rgb,
            indices,
            self.rgb_noise_std,
        )
        rgb_mask[indices] = 0.0

    def _corrupt_rfid(
        self,
        rfid: np.ndarray,
        rfid_mask: np.ndarray,
    ) -> None:
        if self.corruption_mode == "random_nodes":
            indices = self._random_node_indices(
                num_nodes=16,
                ratio=self.rfid_ratio,
            )
        else:
            k = int(np.random.choice(self.part_k_values))
            indices = self._part_node_indices(
                RFID_PARTS,
                k,
            )

        self._replace_nodes_with_noise(
            rfid,
            indices,
            self.rfid_noise_std,
        )
        rfid_mask[indices] = 0.0

    def __getitem__(self, index: int):
        rgb_path, rfid_path, label = self.file_list[index]

        rgb = np.asarray(
            Get_RGB_datas(rgb_path),
            dtype=np.float32,
        ).copy()

        rfid = np.asarray(
            Get_RFID_datas(rfid_path),
            dtype=np.float32,
        ).copy()

        if rgb.shape != (165, 17, 2):
            raise ValueError(
                f"RGB shape error: {rgb.shape}, path={rgb_path}"
            )

        if rfid.shape != (50, 16, 3):
            raise ValueError(
                f"RFID shape error: {rfid.shape}, path={rfid_path}"
            )

        rgb_mask = np.ones(17, dtype=np.float32)
        rfid_mask = np.ones(16, dtype=np.float32)

        if self.mode == "train":
            if (
                self.rgb_ratio > 0
                and np.random.random()
                < self.corruption_probability
            ):
                self._corrupt_rgb(rgb, rgb_mask)

            if (
                self.rfid_ratio > 0
                and np.random.random()
                < self.corruption_probability
            ):
                self._corrupt_rfid(rfid, rfid_mask)

            if (
                self.modality_dropout_probability > 0
                and np.random.random()
                < self.modality_dropout_probability
            ):
                if np.random.random() < 0.5:
                    rgb.fill(0.0)
                    rgb_mask.fill(0.0)
                else:
                    rfid.fill(0.0)
                    rfid_mask.fill(0.0)

        return (
            torch.from_numpy(rgb),
            torch.from_numpy(rfid),
            torch.tensor(int(label), dtype=torch.long),
            torch.from_numpy(rgb_mask),
            torch.from_numpy(rfid_mask),
        )
