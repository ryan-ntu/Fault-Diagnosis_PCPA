import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, LeakyReLU
from torch_geometric.nn import GATv2Conv


class gatcase30(torch.nn.Module):
    """
    来自 Model.py 的 GAT baseline：多层 GAT 后按边拼接节点特征，逐边 MLP 分类。
    """

    def __init__(self, in_features,
                 gat_channels=(256, 256, 256),
                 gat_heads=(3, 2, 1),
                 mlp_dims=(256, 128, 128),
                 dropout=0.2,
                 num_nodes=30,
                 e_h=None,
                 edge_index=None,
                 mlp_out_features=None):
        super(gatcase30, self).__init__()

        if isinstance(gat_channels, int):
            gat_channels = (gat_channels,)
        else:
            gat_channels = tuple(gat_channels)
        gat_heads = (gat_heads,) * len(gat_channels) if isinstance(gat_heads, int) else tuple(gat_heads)
        assert len(gat_channels) == len(gat_heads) >= 1
        # 固定输出维度为首层通道数，后续层保持同一 hidden_dim 以启用残差
        hidden_dim = gat_channels[0]

        self.num_nodes = num_nodes
        # 将输入特征投影到 hidden_dim，保证后续层残差始终可用
        self.input_proj = Linear(in_features, hidden_dim)

        # GAT 堆叠
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_c = hidden_dim
        for hd in gat_heads:
            # 预归一化，维度等于输入特征维
            self.norms.append(nn.LayerNorm(in_c))
            conv = GATv2Conv(in_c, hidden_dim, heads=hd, concat=False)
            self.layers.append(conv)
            in_c = hidden_dim
        self.hidden_dim = hidden_dim

        # 边索引
        assert e_h is not None and edge_index is not None, "e_h 和 edge_index 必须提供以构造逐边特征"
        edge_index_cpu = edge_index.cpu() if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long)
        e_h_node_pairs = []
        for e_idx in e_h:
            src = edge_index_cpu[0, e_idx].item()
            dst = edge_index_cpu[1, e_idx].item()
            e_h_node_pairs.append([src, dst])
        self.register_buffer('e_h_node_pairs', torch.tensor(e_h_node_pairs, dtype=torch.long), persistent=True)

        # MLP
        if mlp_out_features is None:
            mlp_out_features = len(e_h)
        assert len(mlp_dims) == 3
        edge_feature_dim = 2 * self.hidden_dim

        self.gat_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.mlp_fc1 = nn.Linear(edge_feature_dim, mlp_dims[0])
        self.mlp_fc2 = nn.Linear(mlp_dims[0], mlp_dims[1])
        self.mlp_fc3 = nn.Linear(mlp_dims[1], mlp_dims[2])
        self.mlp_fc4 = nn.Linear(mlp_dims[2], 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        B = x.size(0) // self.num_nodes

        h = self.input_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            # 先归一化再卷积，减小分布漂移；若维度一致则做残差
            h_res = h
            h = norm(h)
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.gat_dropout(h)
            h = h + h_res

        h = h.view(B, self.num_nodes, self.hidden_dim)

        src_indices = self.e_h_node_pairs[:, 0]
        dst_indices = self.e_h_node_pairs[:, 1]
        edge_features = torch.cat([h[:, src_indices, :], h[:, dst_indices, :]], dim=-1)

        edge_flat = edge_features.view(-1, edge_features.size(-1))
        out = self.mlp_dropout(self.activation(self.mlp_fc1(edge_flat)))
        out = self.activation(self.mlp_fc2(out))
        out = self.activation(self.mlp_fc3(out))
        out = self.mlp_fc4(out)  # logits
        return out.view(-1)

