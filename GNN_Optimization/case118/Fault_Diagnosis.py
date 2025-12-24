import os
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from collections import defaultdict

from pypower.api import rundcpf, ppoption, case118
from pypower.idx_bus import PD
from copy import deepcopy

from algorithm import (
    gatcase30,
    gat_cnn_case30,
    gcncase30,
    cnn_transformer_locator,
    baseline_gae_30,
    FLVLSIAlgorithm,
    FLDAlgorithm,
)
from utils import (
    observe_state,
    construct_incidence_matrix,
    get_admittance_matrix,
    BFS_algorithm,
    find_edge_indices_within_nodes,
)


def generate_attack(mpc, random_index, attack_types=['cutting', 'weak attack']):
    """生成攻击向量"""
    attacks = np.zeros_like(mpc['branch'][:, 3])
    for idx in random_index:
        attack_type = np.random.choice(attack_types)
        if attack_type == 'cutting':
            attacks[idx] = np.random.randint(10000, 100000) / 100
        else:
            attacks[idx] = np.random.randint(150, 500) / 100
    return attacks


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 初始化配置 ==========
mpc = deepcopy(case118())
B = get_admittance_matrix(mpc)
D, Gamma = construct_incidence_matrix(mpc)
start_node = 20
length_V_H = 20
V_H = BFS_algorithm(mpc, start_node, length_V_H)
E_H = find_edge_indices_within_nodes(mpc['branch'], V_H)
# BFS_algorithm返回的是1-based节点编号，需要转换为0-based索引
V_H_zero = [int(v) - 1 for v in V_H]
V_H = [int(v) for v in V_H]

N = mpc['bus'].shape[0]
mean_load = 8
std_load = 10

# 构建图结构
edges = mpc["branch"][:, :2].astype(int) - 1
edge_index = torch.tensor(edges.T, dtype=torch.long).to(device)
simulation_options = ppoption(PF_DC=True, VERBOSE=0, OUT_ALL=0)

print(f"设备: {device}")
print(f"隐藏节点数: {len(V_H)}, 隐藏边数: {len(E_H)}")
print(f"V_H (1-based): {V_H}")
print(f"V_H_zero (0-based): {V_H_zero}")
print(f"E_H: {E_H}")

# ========== 初始化优化算法（使用 0-based V_H_zero）==========
solver_flvlsi = FLVLSIAlgorithm(case118(), V_H_zero, E_H, B)
fld_baseline = FLDAlgorithm(case118(), V_H_zero, E_H, B)
print("✓ 成功初始化优化算法")

# ========== 加载模型（对齐 main.py 配置）==========
models = {}
model_configs = [
    (
        'GAT',
        gatcase30,
        'model/case118_gat_baseline.pth',
        {
            'e_h': E_H,
            'edge_index': edge_index,
            'gat_channels': (256, 256, 256),
            'gat_heads': (4, 4, 4),
            'mlp_dims': (256, 128, 128),
            'dropout': 0.2,
            'num_nodes': N,
        },
    ),
    (
        'GCN Baseline',
        gcncase30,
        'model/case118_gcn_baseline.pth',
        {
            'v_h': V_H_zero,
            'e_h': E_H,
            'edge_index': edge_index,
            'gcn_channels': (256, 256, 256),
            'mlp_dims': (256, 128, 128),
            'dropout': 0.2,
            'num_nodes': N,
        },
    ),
    (
        'GAT-CNN',
        gat_cnn_case30,
        'model/case118_gat_cnn.pth',
        {
            'v_h': V_H_zero,
            'e_h': E_H,
            'edge_index': edge_index,
            'gat_channels': (256, 256, 256),
            'gat_heads': (4, 4, 4),
            'dropout': 0.2,
            'num_nodes': N,
        },
    ),
    (
        'CNN Transformer',
        cnn_transformer_locator,
        'model/case118_cnn_transformer.pth',
        {
            'mlp_out_features': len(E_H),
            'num_nodes': N,
            'e_h': E_H,
            'edge_index': edge_index,
        },
    ),
    (
        'GAE',
        baseline_gae_30,
        'model/case118_gae.pth',
        {
            'mlp_out_features': len(E_H),
            'num_nodes': N,
            'e_h': E_H,
            'edge_index': edge_index,
            'dropout': 0.2,
        },
    ),
]

print("\n开始加载模型...")
for name, model_class, model_path, kwargs in model_configs:
    try:
        # 所有模型使用 3 维特征 (Va, Pbus, Load)
        model = model_class(in_features=3, **kwargs).to(device)

        possible_paths = [
            model_path,
            os.path.join('model', os.path.basename(model_path)),
            os.path.basename(model_path),
        ]
        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=device)
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                models[name] = model
                print(f"✓ 成功加载模型: {name} (从 {path})")
                loaded = True
                break
        if not loaded:
            print(f"✗ 加载模型 {name} 失败: 找不到模型文件")
    except Exception as e:
        print(f"✗ 加载模型 {name} 失败: {e}")

print(f"\n成功加载 {len(models)} 个模型: {list(models.keys())}")

# ========== 测试配置 ==========
np.random.seed(23)
num_load_samples = 10
num_trials_per_load = 20
Load = pd.DataFrame(np.round(np.clip(np.random.normal(mean_load, std_load, (num_load_samples, N)), 0, 25), 2))

# ========== 存储所有算法的误差结果 ==========
# 结构: {算法名: {故障数: [误差列表]}}
error_results = defaultdict(lambda: defaultdict(list))

# ========== 故障诊断测试 ==========
for rand_num in range(1, len(E_H) + 1):
    print(f"\n{'='*60}")
    print(f"测试 {rand_num} 条故障线路")
    print(f"{'='*60}")
    
    for load_idx, load in enumerate(Load.values):
        for trial in range(num_trials_per_load):
            # 随机选择故障线路
            random_index = np.random.choice(E_H, rand_num, replace=False)
            
            # 生成攻击
            attacks = generate_attack(mpc, random_index)
            
            # 运行原始潮流
            mpc_test = deepcopy(case118())
            mpc_test['bus'][:, PD] = load
            results_origin, _ = rundcpf(mpc_test, simulation_options)
            
            # 计算原始导纳矩阵 Gamma
            D_orig, Gamma_orig = construct_incidence_matrix(mpc_test)
            
            # 应用攻击并运行故障后潮流
            mpc_test['branch'][random_index, 3] *= attacks[random_index]
            B_post = get_admittance_matrix(mpc_test)
            D_post, Gamma_post = construct_incidence_matrix(mpc_test)
            results_post, _ = rundcpf(mpc_test, simulation_options)
            Va_post, Pbus_post = observe_state(results_post)
            
            # 准备输入数据（对齐训练/推理：Va, Pbus, Load）
            load_norm = (mpc_test['bus'][:, PD].reshape(-1, 1) / results_origin['baseMVA'])
            input_array = np.concatenate((Va_post, Pbus_post, load_norm), axis=1)
            input_array[V_H_zero, 1] = 0.0  # 隐藏V_H节点的功率注入
            input_array[V_H_zero, 2] = 0.0  # 隐藏V_H节点的负荷特征
            
            # 转换为PyTorch张量
            input_tensor = Data(
                x=torch.tensor(input_array, dtype=torch.float).to(device),
                edge_index=edge_index
            )
            
            # 准备用于优化算法的数据
            Va_post_flat = Va_post.reshape(-1)
            Pbus_post_flat = Pbus_post.reshape(-1)
            
            # 构建真实故障向量 x_ground（使用导纳矩阵的变化）
            gamma_diag = np.diag(Gamma_orig)
            gamma_post_diag = np.diag(Gamma_post)
            # 计算全部分支的故障向量
            x_ground_full = 1 - np.divide(gamma_post_diag, gamma_diag, 
                                          out=np.zeros_like(gamma_post_diag), 
                                          where=gamma_diag != 0)
            # 提取E_H对应的部分
            x_ground = x_ground_full[E_H]
            
            # 计算真实向量的范数（用于归一化）
            x_ground_norm = np.linalg.norm(x_ground)
            
            
            # ========== 评估所有模型 + 优化算法 ==========
            for model_name, model in models.items():
                try:
                    with torch.no_grad():
                        # 模型预测（logits）
                        model_output = model(input_tensor)
                        # 转换为概率
                        model_pred = torch.sigmoid(model_output).cpu().numpy()
                    
                    # 使用FLVLSI算法优化
                    try:
                        x_flvlsi_full = solver_flvlsi.eval(Va_post_flat, Pbus_post_flat, results_origin, model_pred)
                        # 提取E_H对应的部分
                        x_flvlsi_eh = x_flvlsi_full[E_H]
                        # 计算归一化误差
                        error_flvlsi = np.linalg.norm(x_flvlsi_eh - x_ground) / x_ground_norm
                        error_results[f"{model_name}+FLVLSI"][rand_num].append(error_flvlsi)
                    except Exception as e:
                        print(f"  {model_name}+FLVLSI 优化失败: {e}")
                        
                except Exception as e:
                    print(f"  模型 {model_name} 预测失败: {e}")
            
            # ========== 评估FLD基线算法 ==========
            try:
                x_fld_full = fld_baseline.eval(Va_post_flat, Pbus_post_flat, results_origin)
                # 提取E_H对应的部分
                x_fld_eh = x_fld_full[E_H]
                # 计算归一化误差
                error_fld = np.linalg.norm(x_fld_eh - x_ground) / x_ground_norm
                error_results["FLD"][rand_num].append(error_fld)
            except Exception as e:
                print(f"  FLD 算法失败: {e}")

# ========== 统计和打印结果 ==========
print(f"\n{'='*100}")
print("所有算法的误差统计（均值 ± 标准差）")
print(f"{'='*100}")

# 汇总所有结果
summary_data = []

for algo_name in sorted(error_results.keys()):
    for fault_num in sorted(error_results[algo_name].keys()):
        errors = error_results[algo_name][fault_num]
        if len(errors) > 0:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            summary_data.append({
                'algorithm': algo_name,
                'num_faults': fault_num,
                'mean_error': mean_error,
                'std_error': std_error,
                'num_samples': len(errors)
            })

# 打印汇总表格
if summary_data:
    print(f"\n{'算法':<25} {'故障数':<10} {'平均误差':<15} {'标准差':<15} {'样本数':<10}")
    print("-" * 100)
    for row in summary_data:
        print(f"{row['algorithm']:<25} {row['num_faults']:<10} "
              f"{row['mean_error']:<15.6f} {row['std_error']:<15.6f} {row['num_samples']:<10}")
    
    # 按算法分组统计
    print(f"\n{'='*100}")
    print("按算法汇总（所有故障数）")
    print(f"{'='*100}")
    algo_summary = defaultdict(lambda: {'errors': [], 'fault_nums': []})
    for row in summary_data:
        algo_summary[row['algorithm']]['errors'].extend(
            error_results[row['algorithm']][row['num_faults']]
        )
        algo_summary[row['algorithm']]['fault_nums'].append(row['num_faults'])
    
    print(f"\n{'算法':<30} {'平均误差':<15} {'标准差':<15} {'测试故障数范围':<20}")
    print("-" * 100)
    for algo_name in sorted(algo_summary.keys()):
        errors = algo_summary[algo_name]['errors']
        fault_nums = algo_summary[algo_name]['fault_nums']
        if len(errors) > 0:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            fault_range = f"{min(fault_nums)}-{max(fault_nums)}"
            print(f"{algo_name:<30} {mean_error:<15.6f} {std_error:<15.6f} {fault_range:<20}")

# 保存结果到CSV
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('fault_diagnosis_summary.csv', index=False)
print(f"\n结果已保存到: fault_diagnosis_summary.csv")

# 保存详细结果
detailed_results = []
for algo_name in sorted(error_results.keys()):
    for fault_num in sorted(error_results[algo_name].keys()):
        for error in error_results[algo_name][fault_num]:
            detailed_results.append({
                'algorithm': algo_name,
                'num_faults': fault_num,
                'error': error
            })
df_detailed = pd.DataFrame(detailed_results)
df_detailed.to_csv('fault_diagnosis_detailed.csv', index=False)
print(f"详细结果已保存到: fault_diagnosis_detailed.csv")
