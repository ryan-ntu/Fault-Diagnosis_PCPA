import torch
import torch.nn as nn
import torch.nn.functional as F

class PerLineTokenizer(nn.Module):
    """
    将序列特征 F_in -> 潜在维 d_model 的“1x1卷积/MLP”
    通用序列打标器：既可用于按线路，也可用于按节点。
    输入:  X [B, S, F_in]
    输出:  T [B, S, d_model]
    """
    def __init__(self, F_in, d_model, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, X):
        return self.net(X)  # [B, S, d_model]


class TopologyPositionalEncoding(nn.Module):
    """
    位置/拓扑编码：可学习的 ID embedding；可选加入先验特征
    支持线路或节点两种序列。
    """
    def __init__(self, num_tokens, d_model, topo_feat_dim=0):
        super().__init__()
        self.id_emb = nn.Embedding(num_tokens, d_model)
        self.proj = None
        if topo_feat_dim > 0:
            self.proj = nn.Linear(topo_feat_dim, d_model, bias=False)

    def forward(self, tokens, ids, topo_feats=None):
        """
        tokens:    [B, S, d_model]
        ids:       [S] 或 [B, S] 的长整型索引（0..S-1）
        topo_feats:[B, S, topo_feat_dim] 可选
        """
        B, S, D = tokens.shape
        pe = self.id_emb(ids if ids.dim() == 1 else ids)  # [S, D]或[B,S,D]
        if pe.dim() == 2:
            pe = pe.unsqueeze(0).expand(B, -1, -1)        # -> [B,S,D]

        if self.proj is not None and topo_feats is not None:
            pe = pe + self.proj(topo_feats)               # 加性注入先验

        return tokens + pe


class cnn_transformer_locator(nn.Module):
    """
    改造为与 GCN/GAT 一致的“逐边 MLP”流程：
    Transformer 编码节点 -> 按 E_H 拼接两端节点特征 -> MLP 逐边输出概率。
    """
    def __init__(self, in_features, mlp_out_features=None, num_nodes=30,
                 d_model=256, nhead=4, nlayers=2, dim_ff=256, dropout=0.1,
                 e_h=None, edge_index=None):
        super().__init__()

        assert e_h is not None and edge_index is not None, "e_h / edge_index 必须提供以构造边特征"
        self.num_nodes = num_nodes

        # 节点级 tokenizer 与位置编码（节点ID）
        self.node_tokenizer = PerLineTokenizer(in_features, d_model, hidden=dim_ff // 2, dropout=dropout)
        self.node_posenc = TopologyPositionalEncoding(num_nodes, d_model, topo_feat_dim=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # 边索引映射
        edge_index_cpu = edge_index.cpu() if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long)
        e_h_node_pairs = []
        for e_idx in e_h:
            src = edge_index_cpu[0, e_idx].item()
            dst = edge_index_cpu[1, e_idx].item()
            e_h_node_pairs.append([src, dst])
        self.register_buffer('e_h_node_pairs', torch.tensor(e_h_node_pairs, dtype=torch.long), persistent=True)

        # MLP 逐边分类
        if mlp_out_features is None:
            mlp_out_features = len(e_h)
        edge_feature_dim = 2 * d_model
        self.mlp_fc1 = nn.Linear(edge_feature_dim, 256)
        self.mlp_fc2 = nn.Linear(256, 128)
        self.mlp_fc3 = nn.Linear(128, 128)
        self.mlp_fc4 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, data):
        X = data.x  # [N_total, F_in] 或单图 [N, F_in]
        if hasattr(data, 'batch') and data.batch is not None and X.dim() == 2:
            total_nodes = X.size(0)
            if total_nodes % self.num_nodes == 0:
                B = total_nodes // self.num_nodes
                X = X.view(B, self.num_nodes, -1)
            else:
                X = X.unsqueeze(0)
        elif X.dim() == 2:
            X = X.unsqueeze(0)  # 单图 -> [1, N, F_in]

        B, N, _ = X.shape
        tok = self.node_tokenizer(X)  # [B, N, d_model]

        device = tok.device
        node_ids = torch.arange(N, dtype=torch.long, device=device)
        if N > self.num_nodes:
            node_ids = node_ids % self.num_nodes
        tok = self.node_posenc(tok, node_ids)  # [B, N, d_model]

        enc = self.encoder(tok)  # [B, N, d_model]

        # 边特征拼接
        src_indices = self.e_h_node_pairs[:, 0]
        dst_indices = self.e_h_node_pairs[:, 1]
        edge_features = torch.cat([enc[:, src_indices, :], enc[:, dst_indices, :]], dim=-1)  # [B, |E_H|, 2*d_model]

        edge_flat = edge_features.view(-1, edge_features.size(-1))  # [B*|E_H|, 2*d_model]
        out = self.dropout(self.activation(self.mlp_fc1(edge_flat)))
        out = self.dropout(self.activation(self.mlp_fc2(out)))
        out = self.dropout(self.activation(self.mlp_fc3(out)))
        out = self.mlp_fc4(out)  # logits [B*|E_H|,1]
        return out.view(-1)