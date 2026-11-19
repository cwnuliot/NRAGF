import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph import Graph_RGB, Graph_RFID


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()


class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class AG_STGCN(nn.Module):
    def __init__(self, in_channels, graph, num_class=None, edge_importance_weighting=True, **kwargs):
        super().__init__()
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # 定义GNN层
        self.st_gcn_networks = nn.ModuleList([
            st_gcn(in_channels, 64, kernel_size, 1, residual=True, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, residual=True, **kwargs),
            st_gcn(64, 64, kernel_size, 1, residual=True, **kwargs),
            st_gcn(64, 64, kernel_size, 1, residual=True, **kwargs),
            st_gcn(64, 128, kernel_size, 2, residual=True, **kwargs),
            st_gcn(128, 128, kernel_size, 1, residual=True, **kwargs),
            st_gcn(128, 128, kernel_size, 1, residual=True, **kwargs),
            st_gcn(128, 256, kernel_size, 2, residual=True, **kwargs),
            st_gcn(256, 256, kernel_size, 1, residual=True, **kwargs),
            st_gcn(256, 512, kernel_size, 1, residual=True, **kwargs)
        ])

        # 额外的BatchNorm层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(64), nn.BatchNorm2d(64), nn.BatchNorm2d(64), nn.BatchNorm2d(64),
            nn.BatchNorm2d(128), nn.BatchNorm2d(128), nn.BatchNorm2d(128),
            nn.BatchNorm2d(256), nn.BatchNorm2d(256), nn.BatchNorm2d(512)
        ])

        # 跳跃连接（Jumping Knowledge）
        self.jk_layer = nn.Conv2d(512 + 256 + 256, 512, kernel_size=1)

        # 计算边权重
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size())) for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 全局平均池化 + 分类层
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Conv2d(512, num_class, kernel_size=1) if num_class is not None else lambda x: x

    def forward(self, x):
        N, T, V, C = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()

        # 跳跃连接存储层
        jk_features = []

        for idx, (gcn, bn, importance) in enumerate(zip(self.st_gcn_networks, self.bn_layers, self.edge_importance)):

            x = gcn(x, self.A * importance)
            x = bn(x)  # 额外的BN层
            x = F.relu(x)
            # print(f"{idx}:{x.shape}")

            if idx in [7, 8, 9]:  # 仅存储最后3层用于跳跃连接
                jk_features.append(x)

        # 跳跃连接融合
        x = torch.cat(jk_features, dim=1)  # 512 + 256 + 256 维度
        # print(f"cat:{x.shape}")
        x = self.jk_layer(x)  # 降维到512通道
        # print(f"jk:{x.shape}")
        # 全局平均池化
        x = self.gap(x)
        x = self.cls(x)
        x = x.view(x.size(0), -1)

        return x


class CosLayer(nn.Module):
    def __init__(self, in_size, out_size, s=4.6):
        super(CosLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.W = nn.Parameter(torch.randn(out_size, in_size))
        self.W.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = nn.Parameter(torch.randn(1, )) if s is None else s

    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.W))
        output = cosine * self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_size={}, out_size={}, s={})'.format(
            self.in_size, self.out_size,
            'learn' if isinstance(self.s, nn.Parameter) else self.s)


class TriPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Linear1 = nn.Linear(hidden_dim, 512)
        self.Linear2 = CosLayer(512, 256)
        self.Linear3 = nn.Linear(256, 256)
        self.Linear4 = nn.Linear(256, 256)

        self.Linear5 = nn.Linear(512, 21)

        # 卷积池化操作
        self.conv = nn.Conv1d(3, 3, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.norm_bn4 = nn.BatchNorm1d(3)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.mean_pool = nn.AvgPool2d(kernel_size=(3, 1), stride=(3, 1))

    def forward(self, x):
        FC1 = self.Linear1(x)
        FC2 = self.Linear2(FC1)
        FC3 = self.Linear3(FC2)
        FC4 = self.Linear4(FC3)

        # 将经过线性层的张亮改为三通道
        Fm = torch.cat((FC2.unsqueeze(1), FC3.unsqueeze(1), FC4.unsqueeze(1)), dim=1)
        # print(self.conv(Fm).shape)  # [128, 3, 21]

        r_conv = F.relu(self.norm_bn4(self.conv(Fm)))
        # 最大池化
        pool_max = self.max_pool(r_conv.unsqueeze(1)).squeeze(1).squeeze(1)
        pool_mean = self.mean_pool(r_conv.unsqueeze(1)).squeeze(1).squeeze(1)
        output = torch.cat((pool_mean, pool_max), dim=1)
        output = self.Linear5(output)
        return output


class CMSA(nn.Module):  # Cross-Modal Self-Attention
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, rgb_feat, rfid_feat):
        # (B, C) → (B, 1, C)
        # print(rgb_feat.shape, rfid_feat.shape)
        rgb = rgb_feat.unsqueeze(1)
        rfid = rfid_feat.unsqueeze(1)
        # Cross-Attention
        rgb2, _ = self.attn(rgb, rfid, rfid)  # RGB attends to RFID
        rfid2, _ = self.attn(rfid, rgb, rgb)  # RFID attends to RGB

        rgb = self.norm1(rgb + rgb2)
        rfid = self.norm1(rfid + rfid2)

        # FFN
        rgb = self.norm2(rgb + self.ff(rgb))
        rfid = self.norm2(rfid + self.ff(rfid))

        return rgb.squeeze(1), rfid.squeeze(1)


class MGATE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),  # 输出两个模态的权重
            nn.Softmax(dim=-1)
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        weights = self.gate_fc(combined)
        x = weights[:, 0:1] * x1 + weights[:, 1:2] * x2
        return x


class DualStreamSpatialTemporalGraph(nn.Module):
    def __init__(self, graph_args, num_class, RGB_quality_score, RFID_quality_score, edge_importance_weighting=True,
                 num_heads=8, embed_dim=512, **kwargs):
        super().__init__()
        self.stream1 = AG_STGCN(2, Graph_RGB(**graph_args), None, edge_importance_weighting,
                                **kwargs)
        self.stream2 = AG_STGCN(3, Graph_RFID(**graph_args), None, edge_importance_weighting,
                                **kwargs)
        self.TriPool = TriPool(512)
        self.fcn = nn.Linear(512, num_class)

        self.cmsa = CMSA(dim=512, heads=num_heads)
        self.mgate = MGATE(dim=512)

    def forward(self, x1, x2):
        x1 = self.stream1(x1)  # torch.Size([128, 512])
        x2 = self.stream2(x2)  # torch.Size([128, 512])
        result1 = self.fcn(x1)
        result2 = self.fcn(x2)

        # 5.4
        # x = torch.cat((x1, x2), dim=1)  # torch.Size([128, 1024])
        # x = self.TriPool(x)
        # 5.4

        # 5.5
        # 1.CMSA
        x1, x2 = self.cmsa(x1, x2)
        # 2.MGATE
        fused = self.mgate(x1, x2)
        x = self.TriPool(fused)
        # 5.5

        return x, result1, result2
