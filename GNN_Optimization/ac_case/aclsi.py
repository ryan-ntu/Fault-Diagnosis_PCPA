import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from utilies import get_admittance, construct_incidence_matrix

class lp_estimator:
    def __init__(self, case, v_set, e_set):
        self.case = case
        self.v_set = [int(x) for x in v_set]
        self.e_set = e_set
        self.Ybus, self.Yf, self.Yt = get_admittance(self.case)
        self.Df, self.Dt = construct_incidence_matrix(self.case)
        self.Ysh = self.Ybus - self.Df.dot(self.Yf) - self.Dt.dot(self.Yt)
        if np.linalg.norm(self.Ysh) <= 1e-5:
            print("No parallel conduction!")
        else:
            print("Parallel conduction exists, please conduct estimation carefully!")
        
    def estimation(self, acpower_before, repower_before, v, index):
        acpower_before = acpower_before.to_numpy()
        repower_before = repower_before.to_numpy()

        v = v.to_numpy()
        Yv = np.diag(v[self.v_set]).dot(self.Ybus[self.v_set, :].dot(v))

        D = np.diag(v[self.v_set]).dot(self.Df[np.ix_(self.v_set, self.e_set)].dot(np.diag(self.Yf[self.e_set, :].dot(v)).conj()) + self.Dt[np.ix_(self.v_set, self.e_set)].dot(np.diag(self.Yt[self.e_set, :].dot(v)).conj()))
        
        # 获取分支数量和节点数量
        n_branch = self.case['branch'][self.e_set].shape[0]
        n_bus = len(self.v_set)
    
        # 初始化变量：x_H在[0, 1]之间，delta_pH和delta_qH初始为0
        initial_x0_r = np.random.uniform(0, 1, n_branch)
        initial_x0_i = np.random.uniform(0, 1, n_branch)
        delta_pH = np.zeros(n_bus)
        delta_qH = np.zeros(n_bus)
    
        # 将所有决策变量合并为一个向量，变量顺序为 [x_H, delta_pH, delta_qH]
        x0 = np.concatenate([initial_x0_r, initial_x0_i, delta_pH, delta_qH])
    
        # 设置各变量边界：x_H限制在[0, 1]，而delta_pH和delta_qH不作限制（使用 -∞ 到 +∞）
        lower_bounds = np.concatenate([-1*np.ones(n_branch), 
                                       -1*np.zeros(n_branch), 
                                       -np.inf * np.ones(n_bus), 
                                       -np.inf * np.ones(n_bus)])
        
        upper_bounds = np.concatenate([np.ones(n_branch), 
                                       np.ones(n_branch),
                                       np.inf * np.ones(n_bus), 
                                       np.inf * np.ones(n_bus)])
        bounds = Bounds(lower_bounds, upper_bounds)
    
        # 定义约束函数，将决策变量拆分为 x_H, delta_pH 和 delta_qH
        def constraint1(x):
            x_H_r = x[0:n_branch]
            x_H_i = x[n_branch:2*n_branch]
            x_H = x_H_r - 1j* x_H_i
            delta_pH_var = x[2*n_branch:2*n_branch+n_bus]
            return acpower_before[self.v_set] - np.real(Yv) + np.real(D.dot(x_H)) - delta_pH_var
    
        def constraint2(x):
            x_H_r = x[0:n_branch]
            x_H_i = x[n_branch:2*n_branch]
            x_H = x_H_r - 1j* x_H_i
            delta_qH_var = x[2*n_branch+n_bus:]
            return repower_before[self.v_set] - np.imag(Yv) + np.imag(D.dot(x_H)) - delta_qH_var
        
        # def constraint3(x, index):
        #     attacked_indices = np.where(index == 0)[0]
        #     attack_position_obtained = np.array(self.e_set)[attacked_indices[0]]
        #     z = self.Ybus[int(attack_position_obtained)]

        #     x_H_r = x[0:n_branch]
        #     x_H_i = x[n_branch:2*n_branch]
        #     x_H = x_H_r - 1j*x_H_i

        ymask = index
        constraints = [{'type': 'eq', 'fun': constraint1},
                       {'type': 'eq', 'fun': constraint2}]
    
        # 目标函数仅针对 x_H 部分，此处目标为 x_H 分量的平方和
        def objective(x, mask):
            x_H_r = x[:n_branch]
            x_H_i = x[n_branch:2*n_branch]
            x_H = x_H_r - 1j * x_H_i
            return np.mean((2*mask * np.real(x_H))**2 + (mask * np.imag(x_H))**2)
        
        result = minimize(objective, x0,
                          args=(ymask,),
                          constraints=constraints, bounds=bounds,
                          method='SLSQP')
        x_r = result.x[:n_branch]
        x_i = result.x[n_branch:2*n_branch]
        x = x_r + 1j*x_i

        acpower_after = result.x[2*n_branch: 2*n_branch+n_bus] - acpower_before[self.v_set]
        repower_after = result.x[2*n_branch+n_bus:] - repower_before[self.v_set]

        return x,  acpower_after, repower_after
        
