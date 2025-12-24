import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import random
from scipy import stats  # 用于方差、置信区间和显著性检验

from pypower.api import rundcpf, ppoption, case30
from pypower.idx_bus import PD
from copy import deepcopy

from algorithm import (
    gatcase30, gat_cnn_case30, cnn_transformer_locator, baseline_gae_30,
    gcncase30, FLVLSIAlgorithm, FLDAlgorithm, NormFLVLSIAlgorithm
)
from utils import find_edge_indices_within_nodes, observe_state, construct_incidence_matrix, \
    get_admittance_matrix, BFS_algorithm

# 设置随机种子以确保结果可重复
SEED = 23
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 确保 CUDA 操作的确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mpc = deepcopy(case30())
B = get_admittance_matrix(mpc)
D, Gamma = construct_incidence_matrix(mpc)
start_node = 6
length_V_H = 8
V_H = BFS_algorithm(mpc, start_node, length_V_H)
V_H = [int(v) for v in V_H]
# 转换为0-based索引（用于模型）
V_H_zero = [v - 1 for v in V_H]
E_H = find_edge_indices_within_nodes(mpc['branch'], V_H)
V_bar = np.setdiff1d(np.arange(30), V_H)
B_sub = B[V_bar, :]

N = mpc['bus'].shape[0]
mean_load = 8
std_load = 10

edges = mpc["branch"][:, :2].astype(int)
edges = edges - 1
edge_index = torch.tensor(edges.T, dtype=torch.long)

# 初始化 GAT baseline 模型（参数与 main.py 保持一致）
gat_fl = gatcase30(
    in_features=3,  # Va, Pbus, Load
    gat_channels=(256, 256, 256),
    gat_heads=(4, 4, 4),  # main.py 中使用 (4, 4, 4)
    mlp_dims=(256, 128, 128),
    dropout=0.2,
    num_nodes=30,
    e_h=E_H,
    edge_index=edge_index
).to(device)
gat_fl.load_state_dict(torch.load('model/case30_gat_baseline.pth', map_location=device))
gat_fl.eval()

# 初始化 GAT-CNN 模型（参数与 main.py 保持一致）
gat_cnn = gat_cnn_case30(
    in_features=3,  # Va, Pbus, Load
    gat_channels=(256, 256, 256),  # main.py 中使用 (256, 256, 256)
    gat_heads=(4, 4, 4),  # main.py 中使用 (4, 4, 4)
    dropout=0.2,
    num_nodes=30,
    v_h=V_H_zero,
    e_h=E_H,
    edge_index=edge_index
).to(device)
gat_cnn.load_state_dict(torch.load('model/case30_gat_cnn.pth', map_location=device))
gat_cnn.eval()

# 初始化 GCN Baseline 模型（参数与 main.py 保持一致）
gcn_baseline = gcncase30(
    in_features=3,  # Va, Pbus, Load
    gcn_channels=(256, 256, 256),  # main.py 中使用 (256, 256, 256)
    mlp_dims=(256, 128, 128),
    dropout=0.2,
    num_nodes=30,
    v_h=V_H_zero,
    e_h=E_H,
    edge_index=edge_index
).to(device)
gcn_baseline.load_state_dict(torch.load('model/case30_gcn_baseline.pth', map_location=device))
gcn_baseline.eval()

# 初始化 CNN-Transformer 模型（参数与 main.py 保持一致）
cnn_transformer = cnn_transformer_locator(
    in_features=3,  # Va, Pbus, Load
    mlp_out_features=len(E_H),
    num_nodes=30,
    e_h=E_H,
    edge_index=edge_index
).to(device)
cnn_transformer.load_state_dict(torch.load('model/case30_cnn_transformer.pth', map_location=device))
cnn_transformer.eval()

# 初始化 GAE 模型（参数与 main.py 保持一致）
gae_model = baseline_gae_30(
    in_features=3,  # Va, Pbus, Load
    mlp_out_features=len(E_H),
    num_nodes=30,
    e_h=E_H,
    edge_index=edge_index,
    dropout=0.2
).to(device)
gae_model.load_state_dict(torch.load('model/case30_gae.pth', map_location=device))
gae_model.eval()

solver = FLVLSIAlgorithm(case30(), V_H_zero, E_H, B)
fld = FLDAlgorithm(case30(), V_H_zero, E_H, B)

# Attacks Construction
number = 10
# 随机种子已在文件开头设置，确保结果可重复
# GAT baseline 结果 (strong baseline)
x_results = []
error_values_proposed = []
# GAT-CNN 结果 (strong baseline)
x_gat_cnn_results = []
error_values_gat_cnn = []
# GCN baseline 结果 (strong baseline)
x_gcn_results = []
error_values_gcn = []
# CNN-Transformer 结果 (strong baseline)
x_cnn_transformer_results = []
error_values_cnn_transformer = []
# GAE 结果 (strong baseline)
x_gae_results = []
error_values_gae = []
# FLD 结果 (weak baseline - 不使用学习算法)
x_fld_results = []
error_values_fld = []
# 通用
attack_values = []


for rand_num in range(1, 9):
    counter_proposed = 0
    counter_fld = 0
    mse_gat = 0
    mse_gat_cnn = 0
    mse_gcn = 0
    mse_cnn_transformer = 0
    mse_gae = 0
    mse_fld = 0

    # 记录当前 fault 数下各算法的误差，用于方差、置信区间和显著性检验
    rand_errors_gat = []
    rand_errors_gat_cnn = []
    rand_errors_gcn = []
    rand_errors_cnn = []
    rand_errors_gae = []
    rand_errors_fld = []

    Load = pd.DataFrame(np.round(np.clip(np.random.normal(mean_load, std_load, (50, N)), 0.1, 20), 2))
    for load in Load.values:
        for i in range(number):
            random_index = np.random.choice(E_H, rand_num, replace=False)
            attacks = np.zeros_like(mpc['branch'][:, 3])
            for idx in random_index:
                attack_type = np.random.choice(['cutting', 'weak attack'])
                if attack_type == 'cutting':
                    attacks[idx] = np.random.randint(10000, 100000) / 100
                else:
                    attacks[idx] = np.random.randint(150, 500) / 100

            """ OPTIMIZATION """
            mpc = deepcopy(case30())
            simulation_options = ppoption(PF_DC=True, VERBOSE=0, OUT_ALL=0)
            mpc['bus'][:, PD] = load
            results_origin, _ = rundcpf(mpc, simulation_options)
            Va, Pbus = observe_state(results_origin)

            mpc['branch'][random_index, 3] *= attacks[random_index]
            B_post = get_admittance_matrix(mpc)
            D_post, Gamma_post = construct_incidence_matrix(mpc)
            gamma_diag = np.diag(Gamma)
            gamma_post_diag = np.diag(Gamma_post)
            x_ground = 1-np.divide(gamma_post_diag, gamma_diag, out=np.zeros_like(gamma_post_diag), where=gamma_diag != 0)

            results, _ = rundcpf(mpc, simulation_options)
            Va_post, Pbus_post = observe_state(results)
            
            # 归一化 Load 特征（与训练时保持一致）
            load_normalized = (mpc['bus'][:, PD].reshape(-1, 1) / results['baseMVA'])
            
            # 拼接 Va, Pbus, Load -> [N, 3]
            input_array = np.concatenate((Va_post, Pbus_post, load_normalized), axis=1)
            Va_post = Va_post.reshape(-1, )
            Pbus_post = Pbus_post.reshape(-1, )

            input_array[V_H_zero, 1] = 0  # 使用0-based索引隐藏V_H节点的功率注入
            input_array[V_H_zero, 2] = 0  # 隐藏V_H节点的Load特征
            input_array = torch.tensor(input_array, dtype=torch.float).to(device)
            input_tensor = Data(x=input_array, edge_index=edge_index).to(device)

            y = (attacks[E_H] != 0).astype(int)

            with torch.no_grad():
                # GAT baseline
                output_gat = gat_fl(input_tensor)
                output_gat = torch.sigmoid(output_gat)  # 将 logits 转换为概率
                output_gat = output_gat.cpu().detach().numpy().reshape(-1, )
                
                # GAT-CNN
                output_gat_cnn = gat_cnn(input_tensor)
                output_gat_cnn = torch.sigmoid(output_gat_cnn)  # 将 logits 转换为概率
                output_gat_cnn = output_gat_cnn.cpu().detach().numpy().reshape(-1, )
                
                # GCN baseline
                output_gcn = gcn_baseline(input_tensor)
                output_gcn = torch.sigmoid(output_gcn)  # 将 logits 转换为概率
                output_gcn = output_gcn.cpu().detach().numpy().reshape(-1, )
                
                # CNN-Transformer
                output_cnn_transformer = cnn_transformer(input_tensor)
                output_cnn_transformer = torch.sigmoid(output_cnn_transformer)  # 将 logits 转换为概率
                output_cnn_transformer = output_cnn_transformer.cpu().detach().numpy().reshape(-1, )
                
                # GAE
                output_gae = gae_model(input_tensor)
                output_gae = torch.sigmoid(output_gae)  # 将 logits 转换为概率
                output_gae = output_gae.cpu().detach().numpy().reshape(-1, )

            # 使用 LSI 算法计算故障定位结果 (strong baselines - 结合学习算法输出)
            x_gat = solver.eval(Va_post, Pbus_post, results_origin, output_gat)
            x_gat_cnn = solver.eval(Va_post, Pbus_post, results_origin, output_gat_cnn)
            x_gcn = solver.eval(Va_post, Pbus_post, results_origin, output_gcn)
            x_cnn_transformer = solver.eval(Va_post, Pbus_post, results_origin, output_cnn_transformer)
            x_gae = solver.eval(Va_post, Pbus_post, results_origin, output_gae)
            
            # FLD 算法 (weak baseline - 不使用学习算法输出)
            x_fld = fld.eval(Va_post, Pbus_post, results_origin)

            # 计算误差
            error_gat = np.linalg.norm(x_gat - x_ground) / np.linalg.norm(x_ground)
            error_gat_cnn = np.linalg.norm(x_gat_cnn - x_ground) / np.linalg.norm(x_ground)
            error_gcn = np.linalg.norm(x_gcn - x_ground) / np.linalg.norm(x_ground)
            error_cnn_transformer = np.linalg.norm(x_cnn_transformer - x_ground) / np.linalg.norm(x_ground)
            error_gae = np.linalg.norm(x_gae - x_ground) / np.linalg.norm(x_ground)
            error_fld = np.linalg.norm(x_fld - x_ground) / np.linalg.norm(x_ground)

            # 保存结果（全局）
            x_results.append(np.linalg.norm(x_gat - x_ground))
            x_gat_cnn_results.append(np.linalg.norm(x_gat_cnn - x_ground))
            x_gcn_results.append(np.linalg.norm(x_gcn - x_ground))
            x_cnn_transformer_results.append(np.linalg.norm(x_cnn_transformer - x_ground))
            x_gae_results.append(np.linalg.norm(x_gae - x_ground))
            x_fld_results.append(np.linalg.norm(x_fld - x_ground))
            
            error_values_proposed.append(error_gat)
            error_values_gat_cnn.append(error_gat_cnn)
            error_values_gcn.append(error_gcn)
            error_values_cnn_transformer.append(error_cnn_transformer)
            error_values_gae.append(error_gae)
            error_values_fld.append(error_fld)

            # 保存当前 fault 数下的误差（局部）
            rand_errors_gat.append(error_gat)
            rand_errors_gat_cnn.append(error_gat_cnn)
            rand_errors_gcn.append(error_gcn)
            rand_errors_cnn.append(error_cnn_transformer)
            rand_errors_gae.append(error_gae)
            rand_errors_fld.append(error_fld)
            
            attack_values.append(attacks[E_H])
            
            if error_fld > 5:
                counter_fld += 1
            if error_gat > 5:
                counter_proposed += 1

            mse_gat += error_gat
            mse_gat_cnn += error_gat_cnn
            mse_gcn += error_gcn
            mse_cnn_transformer += error_cnn_transformer
            mse_gae += error_gae
            mse_fld += error_fld

    # 计算当前 fault 数下的均值、方差和 95% 置信区间
    n_samples = number * Load.shape[0]

    def summarize(name, arr):
        arr = np.array(arr)
        mean = arr.mean()
        var = arr.var(ddof=1)
        std = arr.std(ddof=1)
        ci_half = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        print(f'{name}: mean={mean:.4f}, var={var:.4f}, 95% CI=({mean-ci_half:.4f}, {mean+ci_half:.4f})')
        return arr

    print(f'\n=== Statistics for {rand_num} fault lines ===')
    # 以提出的 GAT-CNN 算法作为主要参考
    arr_gat_cnn = summarize('GAT-CNN (proposed, strong)', rand_errors_gat_cnn)
    arr_gat = summarize('GAT baseline (strong)', rand_errors_gat)
    arr_gcn = summarize('GCN baseline (strong)', rand_errors_gcn)
    arr_cnn = summarize('CNN-Transformer (strong)', rand_errors_cnn)
    arr_gae = summarize('GAE (strong)', rand_errors_gae)
    arr_fld = summarize('FLD (weak)', rand_errors_fld)

    # 与提出的 GAT-CNN 算法的配对 t 检验，评估显著性差异
    def paired_t(name, ref_arr, other_arr):
        if len(ref_arr) > 1 and len(ref_arr) == len(other_arr):
            t_stat, p_val = stats.ttest_rel(ref_arr, other_arr)
            print(f'Paired t-test GAT-CNN (proposed) vs {name}: t={t_stat:.4f}, p={p_val:.4e}')

    paired_t('GAT baseline', arr_gat_cnn, arr_gat)
    paired_t('GCN baseline', arr_gat_cnn, arr_gcn)
    paired_t('CNN-Transformer', arr_gat_cnn, arr_cnn)
    paired_t('GAE', arr_gat_cnn, arr_gae)
    paired_t('FLD', arr_gat_cnn, arr_fld)

    # 仍保留原始 MSE 指标输出，便于与论文中结果对应（第一行是提出的 GAT-CNN 算法）
    print(f'\nFor {rand_num} fault lines, the MSE of the GAT-CNN (proposed, strong) is: '
          f'{mse_gat_cnn / n_samples}')

    print(f'\nFor {rand_num} fault lines, the MSE of the GAT baseline (strong) is: '
          f'{mse_gat / n_samples}')
    
    print(f'For {rand_num} fault lines, the MSE of the GCN baseline (strong) is: '
          f'{mse_gcn / n_samples}')
    
    print(f'For {rand_num} fault lines, the MSE of the CNN-Transformer (strong) is: '
          f'{mse_cnn_transformer / n_samples}')
    
    print(f'For {rand_num} fault lines, the MSE of the GAE (strong) is: '
          f'{mse_gae / n_samples}')
    
    print(f'For {rand_num} fault lines, the MSE of the FLD (weak baseline) is: '
          f'{mse_fld / n_samples}')


# Convert lists to numpy arrays or pandas DataFrames
x_results_array = np.array(x_results)
x_gat_cnn_array = np.array(x_gat_cnn_results)
x_gcn_array = np.array(x_gcn_results)
x_cnn_transformer_array = np.array(x_cnn_transformer_results)
x_gae_array = np.array(x_gae_results)
x_fld_array = np.array(x_fld_results)

error_values_proposed_array = np.array(error_values_proposed)
error_values_gat_cnn_array = np.array(error_values_gat_cnn)
error_values_gcn_array = np.array(error_values_gcn)
error_values_cnn_transformer_array = np.array(error_values_cnn_transformer)
error_values_gae_array = np.array(error_values_gae)
error_values_fld_array = np.array(error_values_fld)
attack_values_array = np.array(attack_values)

# Create a pandas DataFrame to store all results
df_results = pd.DataFrame({
    'error_gat_baseline': error_values_proposed_array,
    'error_gat_cnn': error_values_gat_cnn_array,
    'error_gcn_baseline': error_values_gcn_array,
    'error_cnn_transformer': error_values_cnn_transformer_array,
    'error_gae': error_values_gae_array,
    'error_fld': error_values_fld_array,
    'x_gat_baseline': x_results_array,
    'x_gat_cnn': x_gat_cnn_array,
    'x_gcn_baseline': x_gcn_array,
    'x_cnn_transformer': x_cnn_transformer_array,
    'x_gae': x_gae_array,
    'x_fld': x_fld_array,
})

# Save the DataFrame to a CSV file (optional)
df_results.to_csv('fault_diagnosis_results.csv', index=False)
