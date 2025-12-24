import numpy as np
import pandas as pd
from attack_mode import dgba
from aclsi import lp_estimator
from pypower.api import case30, runpf, ppoption
from utilies import get_admittance, construct_incidence_matrix
from scipy.sparse import csr_matrix


# 1) Load the IEEE 30-bus test case
ppc = case30()

V_H, E_H = dgba(ppc, 8, 6)

Ybus, Yf, Yt = get_admittance(ppc)
Df, Dt = construct_incidence_matrix(ppc)

# 2) Silence the solver output
opts = ppoption(VERBOSE=0, OUT_ALL=0)


estimator = lp_estimator(ppc, V_H, E_H)

# 3) Run the AC power flow
results, success = runpf(ppc, opts)
if not success:
    raise RuntimeError("Power flow did not converge")

# 4) Define column names for clarity
bus_cols = [
    'BUS_I','TYPE','Pd','Qd','Gs','Bs','BUS_AREA',
    'Vm','Va','BASE_KV','ZONE','Vmax','Vmin'
]
branch_cols = [
    'F_BUS','T_BUS','BR_R','BR_X','BR_B','RATE_A','RATE_B',
    'RATE_C','TAP','SHIFT','BR_STATUS','ANGMIN','ANGMAX',
    'Pf','Qf','Pt','Qt'   # Note: MATPOWER places flow results in the last 4 columns
]
gen_cols = [
    'GEN_BUS','Pg','Qg','Qmax','Qmin','Vg','mBase',
    'GEN_STATUS','Pmax','Pmin','Pc1','Pc2','Qc1min','Qc1max',
    'Qc2min','Qc2max','Ramp_agc','Ramp_10','Ramp_30','Ramp_q','Apf'
]


bus_df    = pd.DataFrame(results['bus'],    columns=bus_cols)
branch_df = pd.DataFrame(results['branch'], columns=branch_cols)
gen_df    = pd.DataFrame(results['gen'],    columns=gen_cols)

voltage_data = bus_df[['BUS_I', 'Vm', 'Va']]


gen_agg = gen_df.groupby('GEN_BUS')[['Pg', 'Qg']].sum().reset_index()

bus_data = pd.merge(bus_df, gen_agg, left_on='BUS_I', right_on='GEN_BUS', how='left')
bus_data['Pg'] = bus_data['Pg'].fillna(0)
bus_data['Qg'] = bus_data['Qg'].fillna(0)

bus_data['P_inj'] = bus_data['Pg'] - bus_data['Pd']
bus_data['Q_inj'] = bus_data['Qg'] - bus_data['Qd']

# calculate the voltage in expression of magnititude and angle
voltage_data['Voltage_complex'] = voltage_data['Vm'] * np.exp(1j * np.deg2rad(voltage_data['Va']))

attack_position = np.random.choice(E_H)
attack_index = 1- (np.array(E_H) == attack_position).astype(int)

print("攻击指标向量:", attack_index)

ppc['branch'][attack_position, 3] *= np.random.randint(2,5)
Ybus1, Yf1, Yt1 = get_admittance(ppc)
results1, _ = runpf(ppc, opts)
[fault_node1, fault_node2] = map(int, ppc['branch'][attack_position, :2])

denom = Ybus[fault_node1-1, fault_node2-1]
if np.isclose(denom, 0):
    raise ZeroDivisionError("计算 x 时检测到分母为 0，请检查Ybus数据是否有误")
x =  (Ybus1[fault_node1-1, fault_node2-1]) / denom
x_H = np.zeros((41,), dtype=complex)
x_H[attack_position, ] = 1-x
print(x_H[E_H])

print(np.linalg.norm(Ybus1- Ybus +Df.dot(np.diag(x_H)).dot(Yf) + Dt.dot(np.diag(x_H)).dot(Yt)))

bus_df_new = pd.DataFrame(results1['bus'], columns=bus_cols)
gen_df_new = pd.DataFrame(results1['gen'], columns=gen_cols)

voltage_data_new = bus_df_new[['BUS_I', 'Vm', 'Va']]

voltage_complex = voltage_data_new['Vm'] * np.exp(1j * np.deg2rad(voltage_data_new['Va']))

gen_agg_new = gen_df_new.groupby('GEN_BUS')[['Pg', 'Qg']].sum().reset_index()


bus_data_new = pd.merge(bus_df_new, gen_agg_new, left_on='BUS_I', right_on='GEN_BUS', how='left')
bus_data_new['Pg'] = bus_data_new['Pg'].fillna(0)
bus_data_new['Qg'] = bus_data_new['Qg'].fillna(0)

bus_data_new['P_inj'] = bus_data_new['Pg'] - bus_data_new['Pd']
bus_data_new['Q_inj'] = bus_data_new['Qg'] - bus_data_new['Qd']
print(np.sum(bus_data['P_inj']+ bus_data['Q_inj'] - bus_data_new['P_inj']- bus_data_new['Q_inj']))
esti_x, esti_P_inj, esti_Q_inj = estimator.estimation(bus_data['P_inj'], bus_data['Q_inj'], voltage_complex, attack_index)
print(esti_x)
print(esti_P_inj - bus_data_new['P_inj'].iloc[V_H])
print(esti_Q_inj - bus_data_new['Q_inj'].iloc[V_H])






# # 5) 根据复功率公式 S = V * I*，计算节点注入电流的幅值和相角
# #    公式推导：I = conj(S)/conj(V)  => |I| = |S|/Vm, 角度 I_angle = Va - arctan2(Q_inj, P_inj)
# bus_data['I_inj'] = bus_data.apply(
#     lambda row: (__import__('math').sqrt(row['P_inj']**2 + row['Q_inj']**2)) / row['Vm']
#                 if row['Vm'] != 0 else None,
#     axis=1
# )
# bus_data['I_inj_angle'] = bus_data.apply(
#     lambda row: np.deg2rad(row['Va'] - (180/np.pi) * __import__('math').atan2(row['Q_inj'], row['P_inj'])),
#     axis=1
# )

# # 6) 输出节点的电压数据以及对应的注入电流信息
# nodal_info = bus_data[['BUS_I', 'Vm', 'Va', 'I_inj', 'I_inj_angle']]
# # print("=== 节点电压（幅值、相角）及注入电流（幅值、相角）信息 ===")
# # print(nodal_info)



