from time import sleep
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class MultiHeadCrossAttention(nn.Module):
    """
    通用多头 Cross-Attention：
    query:         [B, N_q, d_model]
    key_context:   [B, N_k, d_model]
    value_context: [B, N_v, d_model]（可选，默认与 key 相同）
    context_mask:  [B, N_k] (1=valid, 0=masked) 可选
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_context, value_context=None, context_mask=None):
        B, Nq, _ = query.shape
        B, Nk, _ = key_context.shape
        value_context = key_context if value_context is None else value_context
        assert value_context.shape[:2] == (B, Nk), "value_context 需与 key_context 对齐"

        # 线性投影
        Q = self.q_proj(query)            # [B, N_q, d_model]
        K = self.k_proj(key_context)      # [B, N_k, d_model]
        V = self.v_proj(value_context)    # [B, N_k, d_model]

        # 拆成多头: [B, heads, N, head_dim]
        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数 [B, heads, N_q, N_kv]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 如果有 mask，对缺失/不可见位置打 -inf
        if context_mask is not None:
            mask = context_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 汇总 V
        out = torch.matmul(attn, V)  # [B, heads, N_q, head_dim]

        # 拼回 d_model
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        out = self.out_proj(out)    # [B, N_q, d_model]
        return out


class GAT_CNN_Layer(nn.Module):
    """
    GAT + CNN 并行层：GATv2 捕获图结构，CNN 以节点序列卷积补充模式信息；融合后残差+归一化。
    """
    def __init__(self, in_features, out_features, heads, num_nodes, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_features

        # GATv2 分支
        self.gat = GATv2Conv(in_features, out_features, heads=heads, concat=False, dropout=dropout)
        self.gat_activation = nn.LeakyReLU()
        self.gat_norm = nn.LayerNorm(out_features)

        # CNN 分支（大核）
        self.cnn_in_proj = nn.Linear(in_features, out_features)
        self.cnn_conv = nn.Conv1d(out_features, out_features, kernel_size=7, padding=3)
        self.cnn_activation = nn.ReLU()
        self.cnn_norm = nn.LayerNorm(out_features)

        # 融合与残差
        self.fuse_norm = nn.LayerNorm(out_features)
        self.res_proj = nn.Linear(in_features, out_features) if in_features != out_features else None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        N_total = x.size(0)
        B = N_total // self.num_nodes

        # GAT 分支
        h_gat = self.gat(x, edge_index)
        h_gat = self.gat_activation(h_gat)
        h_gat = self.gat_norm(h_gat)

        # CNN 分支
        x_cnn = self.cnn_in_proj(x)
        x_cnn = x_cnn.view(B, self.num_nodes, -1).transpose(1, 2)  # [B, C, N]
        h_cnn = self.cnn_conv(x_cnn)
        h_cnn = h_cnn.transpose(1, 2).contiguous().view(B * self.num_nodes, -1)
        h_cnn = self.cnn_activation(h_cnn)
        h_cnn = self.cnn_norm(h_cnn)

        # 融合 + 残差
        h = h_gat + h_cnn
        residual = x if self.res_proj is None else self.res_proj(x)
        h = h + residual
        h = self.fuse_norm(h)

        h = self.dropout(h)
        return h


class gatcase30(nn.Module):
    """
    GAT + CNN 并行堆叠 → 跨节点注意力 → MLP 逐边分类
    """
    def __init__(self, in_features,
                 gat_channels=(256, 256, 256), gat_heads=(3, 3, 3),
                 mlp_dims=(256, 128, 128), dropout=0.3, num_nodes=30, 
                 v_h=None, e_h=None, edge_index=None, attn_heads=None):
        super().__init__()
        
        # 规范化输入参数
        if isinstance(gat_channels, int):
            if isinstance(gat_heads, (tuple, list)):
                gat_channels = tuple([gat_channels] * len(gat_heads))
            else:
                gat_channels = (gat_channels,)
                gat_heads = (gat_heads,)
        elif isinstance(gat_heads, int):
            gat_heads = tuple([gat_heads] * len(gat_channels))
        else:
            gat_channels = tuple(gat_channels)
            gat_heads = tuple(gat_heads)
        
        assert len(gat_channels) == len(gat_heads) >= 1
        
        self.num_nodes = num_nodes
        self.v_h = v_h
        self.e_h = e_h
        vh_idx_tensor = torch.as_tensor(v_h if v_h else [], dtype=torch.long)
        
        # 并行 GAT+CNN 层堆叠
        self.layers = nn.ModuleList()
        
        in_c = in_features
        for ch, hd in zip(gat_channels, gat_heads):
            layer = GAT_CNN_Layer(
                in_features=in_c,
                out_features=ch,
                heads=hd,
                num_nodes=num_nodes,
                dropout=dropout
            )
            self.layers.append(layer)
            in_c = layer.out_dim
        
        self.hidden_dim = in_c

        # 注册V_H节点索引
        self.register_buffer('vh_idx', vh_idx_tensor, persistent=True)
        
        # 构建E_H边对应的节点对索引
        assert e_h is not None and edge_index is not None, "e_h and edge_index must be provided"
        edge_index_cpu = edge_index.cpu() if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index, dtype=torch.long)
        v_h_to_idx = {node: idx for idx, node in enumerate(v_h)} if v_h else {}
        
        e_h_node_pairs = []
        for e_idx in e_h:
            src_node = edge_index_cpu[0, e_idx].item()
            dst_node = edge_index_cpu[1, e_idx].item()
            e_h_node_pairs.append([v_h_to_idx.get(src_node, -1), v_h_to_idx.get(dst_node, -1)])
        
        self.register_buffer('e_h_node_pairs', torch.tensor(e_h_node_pairs, dtype=torch.long), persistent=True)

        # 跨注意力（在堆叠后、MLP 前）
        attn_heads = attn_heads if attn_heads is not None else (gat_heads[-1] if len(gat_heads) > 0 else 4)
        self.cross_ln = nn.LayerNorm(self.hidden_dim)
        self.k_proj = nn.Linear(in_features, self.hidden_dim)
        self.k_ln = nn.LayerNorm(self.hidden_dim)
        self.v_ln = nn.LayerNorm(self.hidden_dim)
        self.cross_attn = MultiHeadCrossAttention(d_model=self.hidden_dim, num_heads=attn_heads, dropout=dropout)
        self.cross_ffn_ln = nn.LayerNorm(self.hidden_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_dropout = nn.Dropout(dropout)

        # 构建MLP（直接使用边特征）
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

        # 多层并行 GAT + CNN 组合层
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)

        # reshape
        h = h.view(B, self.num_nodes, self.hidden_dim)  # [B, N, hidden_dim]

        # Cross-Attention：在并行堆叠后、MLP 前
        if self.vh_idx.numel() > 0:
            # K 来自原始输入，V 来自堆叠后的特征
            k_ctx = self.k_ln(self.k_proj(x.view(B, self.num_nodes, -1)))
            v_ctx = self.v_ln(h)

            context_mask = torch.ones(B, self.num_nodes, device=h.device, dtype=torch.float32)
            context_mask[:, self.vh_idx] = 0.0  # 遮盖 V_H 作为 K/V

            q = v_ctx[:, self.vh_idx, :]  # [B, |V_H|, d]
            attn_out = self.cross_attn(query=q, key_context=k_ctx, value_context=v_ctx, context_mask=context_mask)
            h = h.clone()
            h[:, self.vh_idx, :] = h[:, self.vh_idx, :] + self.attn_dropout(attn_out)
            # FFN on V_H
            vh_slice = h[:, self.vh_idx, :]
            h[:, self.vh_idx, :] = vh_slice + self.cross_ffn(self.cross_ffn_ln(vh_slice))

        h_vh = h[:, self.vh_idx, :]  # [B, |V_H|, hidden_dim]

        # 构建边特征：[B, |E_H|, 2*hidden_dim]
        src_indices = self.e_h_node_pairs[:, 0]
        dst_indices = self.e_h_node_pairs[:, 1]
        edge_features = torch.cat([h_vh[:, src_indices, :], h_vh[:, dst_indices, :]], dim=-1)

        # MLP 按批处理边特征，输出 [B, |E_H|, 1]
        edge_flat = edge_features.view(-1, edge_features.size(-1))  # [B*|E_H|, 2*hidden_dim]
        out = self.dropout(self.activation(self.mlp_fc1(edge_flat)))
        out = self.activation(self.mlp_fc2(out))
        out = self.activation(self.mlp_fc3(out))
        out = self.mlp_fc4(out)  # logits [B*|E_H|, 1]
        return out.view(-1)
