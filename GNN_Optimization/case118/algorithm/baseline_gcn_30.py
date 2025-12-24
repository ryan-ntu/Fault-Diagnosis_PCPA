import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch_geometric.nn import ChebConv


class gcncase30(torch.nn.Module):
    """
    来自 Model.py 的 gcncase30，按 V_H 节点构造边特征并逐边 MLP 分类。
    """

    def __init__(self, in_features,
                 gcn_channels=(256, 256, 256),
                 cheb_K=(3, 4, 5),
                 mlp_dims=(256, 128, 128),
                 dropout=0.3,
                 num_nodes=30,
                 v_h=None,
                 e_h=None,
                 edge_index=None):
        super(gcncase30, self).__init__()

        gcn_channels = (gcn_channels,) if isinstance(gcn_channels, int) else tuple(gcn_channels)
        cheb_K = (cheb_K,) if isinstance(cheb_K, int) else tuple(cheb_K)
        assert len(gcn_channels) == 3, "需要 3 层 ChebConv"
        assert len(cheb_K) == 3, "cheb_K 应提供 3 个 K 值"

        self.num_nodes = num_nodes

        # 构建多层 GCN
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_c = in_features
        for ch, k in zip(gcn_channels, cheb_K):
            self.layers.append(ChebConv(in_c, ch, K=k))
            in_c = ch
            self.norms.append(nn.LayerNorm(in_c))

        self.hidden_dim = in_c

        # 注册 V_H 与 E_H
        assert e_h is not None and edge_index is not None and v_h is not None, "v_h/e_h/edge_index 必须提供"
        self.register_buffer('vh_idx', torch.as_tensor(v_h, dtype=torch.long), persistent=True)

        edge_index_cpu = edge_index.cpu() if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long)
        v_h_to_idx = {node: idx for idx, node in enumerate(v_h)}
        e_h_node_pairs = []
        for e_idx in e_h:
            src_node = edge_index_cpu[0, e_idx].item()
            dst_node = edge_index_cpu[1, e_idx].item()
            e_h_node_pairs.append([v_h_to_idx.get(src_node, -1), v_h_to_idx.get(dst_node, -1)])
        self.register_buffer('e_h_node_pairs', torch.tensor(e_h_node_pairs, dtype=torch.long), persistent=True)

        # MLP
        assert len(mlp_dims) == 3
        edge_feature_dim = 2 * self.hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.mlp_fc1 = nn.Linear(edge_feature_dim, mlp_dims[0])
        self.mlp_fc2 = nn.Linear(mlp_dims[0], mlp_dims[1])
        self.mlp_fc3 = nn.Linear(mlp_dims[1], mlp_dims[2])
        self.mlp_fc4 = nn.Linear(mlp_dims[2], 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        B = x.size(0) // self.num_nodes

        h = x
        for conv, norm in zip(self.layers, self.norms):
            h = self.dropout(self.activation(conv(h, edge_index)))
            h = norm(h)

        h = h.view(B, self.num_nodes, self.hidden_dim)
        h_vh = h[:, self.vh_idx, :]

        src_indices = self.e_h_node_pairs[:, 0]
        dst_indices = self.e_h_node_pairs[:, 1]
        edge_features = torch.cat([h_vh[:, src_indices, :], h_vh[:, dst_indices, :]], dim=-1)

        edge_flat = edge_features.view(-1, edge_features.size(-1))
        out = self.dropout(self.activation(self.mlp_fc1(edge_flat)))
        out = self.dropout(self.activation(self.mlp_fc2(out)))
        out = self.dropout(self.activation(self.mlp_fc3(out)))
        out = self.mlp_fc4(out)  # logits
        return out.view(-1)