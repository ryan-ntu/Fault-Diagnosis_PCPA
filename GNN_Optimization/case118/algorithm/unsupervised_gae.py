import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, GELU
from torch_geometric.nn import ChebConv


class baseline_gae_30(nn.Module):
    """
    GAE 编码器 + 自注意力，输出对齐 GCN/GAT 的逐边 MLP：按 E_H 拼接两端节点特征后分类。
    """

    def __init__(self, in_features: int, mlp_out_features: int = None,
                 num_nodes: int = 30, e_h=None, edge_index=None,
                 K: int = None, dropout: float = 0.2):
        super().__init__()
        assert e_h is not None and edge_index is not None, "e_h / edge_index 必须提供"
        self.num_nodes = num_nodes
        self.num_layers = 3
        self.hidden_dim = 256
        self.K = 5 if K is None else K
        self.dropout_p = dropout

        # Encoder: 3 × ChebConv(K=5)
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        self.gcn_layers.append(ChebConv(in_features, self.hidden_dim, K=self.K))
        self.gcn_norms.append(nn.LayerNorm(self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.gcn_layers.append(ChebConv(self.hidden_dim, self.hidden_dim, K=self.K))
            self.gcn_norms.append(nn.LayerNorm(self.hidden_dim))

        # Spatial self-attention + FFN
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=4, dropout=self.dropout_p, batch_first=True
        )
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            Linear(self.hidden_dim, self.hidden_dim * 4),
            GELU(),
            Dropout(self.dropout_p),
            Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.drop = Dropout(self.dropout_p)
        self.relu = ReLU()

        # 边索引映射
        edge_index_cpu = edge_index.cpu() if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long)
        e_h_node_pairs = []
        for e_idx in e_h:
            src = edge_index_cpu[0, e_idx].item()
            dst = edge_index_cpu[1, e_idx].item()
            e_h_node_pairs.append([src, dst])
        self.register_buffer('e_h_node_pairs', torch.tensor(e_h_node_pairs, dtype=torch.long), persistent=True)

        # MLP head 逐边分类
        if mlp_out_features is None:
            mlp_out_features = len(e_h)
        edge_feature_dim = 2 * self.hidden_dim
        self.mlp_fc1 = Linear(edge_feature_dim, 256)
        self.mlp_fc2 = Linear(256, 128)
        self.mlp_fc3 = Linear(128, 128)
        self.mlp_fc4 = Linear(128, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Graph encoding
        for conv, norm in zip(self.gcn_layers, self.gcn_norms):
            x = self.relu(conv(x, edge_index))
            x = norm(x)
            x = self.drop(x)

        # reshape to [B, N, C]
        if hasattr(data, 'batch') and data.batch is not None:
            total_nodes = x.size(0)
            if self.num_nodes > 0 and total_nodes % self.num_nodes == 0:
                B = total_nodes // self.num_nodes
                x = x.view(B, self.num_nodes, -1)
            else:
                x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(0)

        # Spatial attention
        res = x
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, need_weights=False)
        x = res + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))

        # 边特征构造
        src_indices = self.e_h_node_pairs[:, 0]
        dst_indices = self.e_h_node_pairs[:, 1]
        edge_features = torch.cat([x[:, src_indices, :], x[:, dst_indices, :]], dim=-1)  # [B, |E_H|, 2*hidden]

        edge_flat = edge_features.view(-1, edge_features.size(-1))
        out = self.drop(self.relu(self.mlp_fc1(edge_flat)))
        out = self.drop(self.relu(self.mlp_fc2(out)))
        out = self.drop(self.relu(self.mlp_fc3(out)))
        out = self.mlp_fc4(out)  # logits
        return out.view(-1)



