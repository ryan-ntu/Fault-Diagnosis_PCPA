import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_CNN_Layer(nn.Module):
    """
    GAT + CNN 组合层
    每一层先通过GAT学习局部图结构特征，然后通过CNN学习全局空间模式
    """
    def __init__(self, in_features, out_features, heads, num_nodes, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_features * heads
        
        # GAT层：学习局部图结构特征（concat=True）
        self.gat = GATConv(in_features, out_features, heads=heads, concat=True, dropout=dropout)
        self.gat_norm = nn.LayerNorm(self.out_dim)
        self.gat_activation = nn.LeakyReLU()
        
        # CNN层：学习全局空间模式（单层CNN）
        self.cnn_conv = nn.Conv1d(self.out_dim, self.out_dim, kernel_size=3, padding=1)
        self.cnn_norm = nn.LayerNorm(self.out_dim)
        self.cnn_activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        x: [B*N, in_features] 或 [N, in_features]
        edge_index: [2, E]
        返回: [B*N, out_features] 或 [N, out_features]
        """
        # GAT：学习局部图结构特征
        h_gat = self.gat(x, edge_index)  # [B*N, out_dim]
        h_gat = self.gat_norm(h_gat)
        h_gat = self.gat_activation(h_gat)
        
        # 获取批次大小
        N_total = h_gat.size(0)
        B = N_total // self.num_nodes
        
        # Reshape for CNN: [B*N, out_features] -> [B, out_features, N]
        h_gat_reshaped = h_gat.view(B, self.num_nodes, -1)  # [B, N, out_dim]
        h_gat_cnn = h_gat_reshaped.transpose(1, 2)  # [B, out_dim, N]
        
        # CNN：学习全局空间模式（单层）
        h_cnn = self.cnn_conv(h_gat_cnn)  # [B, out_dim, N]
        
        # Reshape back: [B, out_features, N] -> [B*N, out_features]
        h_cnn_reshaped = h_cnn.transpose(1, 2).contiguous()  # [B, N, out_dim]
        h_cnn_reshaped = h_cnn_reshaped.view(B * self.num_nodes, -1)  # [B*N, out_features]
        
        # LayerNorm和激活
        h_cnn_reshaped = self.cnn_norm(h_cnn_reshaped)
        h_cnn_reshaped = self.cnn_activation(h_cnn_reshaped)
        
        # 残差连接（如果维度匹配）或融合GAT和CNN特征
        h = h_gat + h_cnn_reshaped  # 残差连接
        
        h = self.dropout(h)
        
        return h


class gatcase30(nn.Module):
    """
    GAT + CNN 混合模型
    使用多层GAT提取节点特征，然后用CNN提取全局特征，最后预测每条边的分数
    """
    def __init__(self, in_features,
                 gat_channels=(128, 128, 128), gat_heads=(2, 2, 2),
                 mlp_dims=(1024, 256, 256), dropout=0.3, num_nodes=30, 
                 v_h=None, e_h=None, edge_index=None):
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
        
        # 构建多层GAT_CNN组合层
        # 每一层都同时学习局部图结构（GAT）和全局空间模式（CNN）
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
        self.register_buffer('vh_idx', torch.as_tensor(v_h if v_h else [], dtype=torch.long), persistent=True)
        
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

        # 构建MLP（直接使用边特征）
        # 输入：边特征 (2 * hidden_dim)
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

        # 多层GAT_CNN组合层
        # 每一层都同时学习局部图结构（GAT）和全局空间模式（CNN）
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)

        # 提取V_H节点特征并构建边特征
        h = h.view(B, self.num_nodes, self.hidden_dim)  # [B, N, hidden_dim]
        h_vh = h[:, self.vh_idx, :]  # [B, |V_H|, hidden_dim]
        src_indices = self.e_h_node_pairs[:, 0]
        dst_indices = self.e_h_node_pairs[:, 1]
        edge_features = torch.cat([h_vh[:, src_indices, :], h_vh[:, dst_indices, :]], dim=-1)  # [B, |E_H|, 2*hidden_dim]
        
        # MLP处理（直接使用边特征）
        edge_features_flat = edge_features.view(B * len(self.e_h), -1)
        out = self.dropout(self.activation(self.mlp_fc1(edge_features_flat)))
        out = self.dropout(self.activation(self.mlp_fc2(out)))
        out = self.dropout(self.activation(self.mlp_fc3(out)))
        out = torch.sigmoid(self.mlp_fc4(out)).squeeze(-1)
        
        return out
