"""
优化算法模块 - 提供故障定位的优化算法
"""
import numpy as np
from scipy.optimize import minimize, Bounds
from utils import observe_state, construct_incidence_matrix


class FLVLSIAlgorithm:
    def __init__(self, case, v_set, e_set, B):
        self.case = case
        self.v_set = v_set
        self.v_bar = np.setdiff1d(np.arange(case['bus'].shape[0]), v_set)
        self.e_set = e_set
        self.B = B
        self.D, self.Gamma = construct_incidence_matrix(case)

    def eval(self, va_post, pbus_post, results_original, reference_signal):
        t_Va_original, t_Pbus_original = observe_state(results_original)
        t_Va_original, t_Pbus_original = t_Va_original.reshape(-1, ), t_Pbus_original.reshape(-1, )
        t_Va_post, t_Pbus_post = va_post, pbus_post
        t_Delta = (t_Pbus_original - t_Pbus_post)

        test_A = self.D.dot(self.Gamma).dot(np.diag(self.D.T.dot(t_Va_post).flatten()))

        test_x0 = np.zeros(self.case['branch'][self.e_set].shape[0])
        lower_bounds = [0] * self.case['branch'][self.v_set].shape[0]
        upper_bounds = [1] * self.case['branch'][self.v_set].shape[0]
        test_bounds = Bounds(lower_bounds, upper_bounds)

        def test_objective(x, y_mask):
            x_branch = x
            penalty = (1 - y_mask) @ x_branch
            return penalty

        def test_constraint(x):
            x_branch = np.zeros(self.case['branch'].shape[0])
            x_branch[self.e_set] = x
            delta = t_Delta

            return np.linalg.norm(test_A @ x_branch + self.B @ (t_Va_original - t_Va_post) - delta)

        test_constraints = {'type': 'eq', 'fun': test_constraint}
        y_mask = reference_signal

        res_test = minimize(test_objective, test_x0, args=(y_mask,),
                            constraints=test_constraints, bounds=test_bounds,
                            method='SLSQP')
        x = np.zeros(self.case['branch'].shape[0])
        x[self.e_set] = res_test.x[:self.case['branch'][self.e_set].shape[0]]
        return x


class FLDAlgorithm:
    def __init__(self, case, v_set, e_set, B):
        self.case = case
        self.v_set = v_set
        self.v_bar = np.setdiff1d(np.arange(case['bus'].shape[0]), v_set)
        self.e_set = e_set
        self.B = B
        self.D, self.Gamma = construct_incidence_matrix(case)

    def eval(self, va_post, pbus_post, results_original):
        t_Va_original, t_Pbus_original = observe_state(results_original)
        t_Va_original, t_Pbus_original = t_Va_original.reshape(-1, ), t_Pbus_original.reshape(-1, )
        t_Va_post, t_Pbus_post = va_post, pbus_post
        t_Delta = (t_Pbus_original - t_Pbus_post)[self.v_bar]

        test_A = self.D.dot(self.Gamma).dot(np.diag(self.D.T.dot(t_Va_post).flatten()))

        test_x0 = np.zeros(len(self.e_set))
        # 优化变量结构：前 len(e_set) 个是边的故障变量，后 len(v_set) 个是隐藏节点的功率注入变量
        
        # 为隐藏节点的功率注入变量设置边界约束
        # 约束：p_original >= delta >= 0 或 p_original <= delta <= 0
        # 即：如果 p_original[i] >= 0，则 0 <= delta[i] <= p_original[i]
        #     如果 p_original[i] <= 0，则 p_original[i] <= delta[i] <= 0
        delta_lower_bounds = []
        delta_upper_bounds = []
        for node_idx in self.v_set:
            p_orig = t_Pbus_original[node_idx]
            if p_orig >= 0:
                delta_lower_bounds.append(0.0)
                delta_upper_bounds.append(p_orig)
            else:
                delta_lower_bounds.append(p_orig)
                delta_upper_bounds.append(0.0)
        
        lower_bounds = [0] * len(self.e_set) + delta_lower_bounds
        upper_bounds = [1] * len(self.e_set) + delta_upper_bounds
        test_bounds = Bounds(lower_bounds, upper_bounds)
        x0_extended = np.concatenate((test_x0, np.zeros(len(self.v_set))))

        def test_objective(x):
            x_branch = x[:len(self.e_set)]
            penalty = np.sum(x_branch)
            return penalty

        def test_constraint(x):
            x_branch = np.zeros(self.case['branch'].shape[0])
            x_branch[self.e_set] = x[:len(self.e_set)]
            delta = np.zeros_like(t_Pbus_original)
            delta[self.v_set] = x[len(self.e_set):]
            delta[self.v_bar] = t_Delta

            return np.linalg.norm(test_A @ x_branch + self.B @ (t_Va_original - t_Va_post) - delta)

        def delta_sum_constraint(x):
            """约束：delta 的和为 0"""
            # delta = delta[v_set] + delta[v_bar]
            # delta[v_set] = x[len(self.e_set):] (优化变量)
            # delta[v_bar] = t_Delta (已知值)
            delta_v_set_sum = np.sum(x[len(self.e_set):])
            delta_v_bar_sum = np.sum(t_Delta)
            return delta_v_set_sum + delta_v_bar_sum

        test_constraints = [
            {'type': 'eq', 'fun': test_constraint},
            {'type': 'eq', 'fun': delta_sum_constraint}
        ]

        res_test = minimize(test_objective, x0_extended,
                            constraints=test_constraints, bounds=test_bounds,
                            method='SLSQP')
        estimated_x = np.zeros(self.case['branch'].shape[0])
        estimated_x[self.e_set] = res_test.x[:len(self.e_set)]
                                            
        return estimated_x


class NormFLVLSIAlgorithm:
    def __init__(self, case, v_set, e_set, B):
        self.case = case
        self.v_set = v_set
        self.v_bar = np.setdiff1d(np.arange(case['bus'].shape[0]), v_set)
        self.e_set = e_set
        self.B = B
        self.D, self.Gamma = construct_incidence_matrix(case)

    def eval(self, va_post, pbus_post, results_original, reference_signal):
        t_Va_original, t_Pbus_original = observe_state(results_original)
        t_Va_original, t_Pbus_original = t_Va_original.reshape(-1, ), t_Pbus_original.reshape(-1, )
        t_Va_post, t_Pbus_post = va_post, pbus_post
        t_Delta = (t_Pbus_original - t_Pbus_post)

        test_A = self.D.dot(self.Gamma).dot(np.diag(self.D.T.dot(t_Va_post).flatten()))

        test_x0 = np.random.uniform(0, 1, self.case['branch'][self.e_set].shape[0])
        lower_bounds = [0] * self.case['branch'][self.v_set].shape[0]
        upper_bounds = [1] * self.case['branch'][self.v_set].shape[0]
        test_bounds = Bounds(lower_bounds, upper_bounds)

        def test_objective(x, y_mask):
            x_branch = x
            penalty = np.linalg.norm((1 - y_mask) * x_branch)
            return penalty

        def test_constraint(x):
            x_branch = np.zeros(self.case['branch'].shape[0])
            x_branch[self.e_set] = x
            delta = t_Delta

            return np.linalg.norm(test_A @ x_branch + self.B @ (t_Va_original - t_Va_post) - delta)

        test_constraints = {'type': 'eq', 'fun': test_constraint}
        y_mask = reference_signal

        res_test = minimize(test_objective, test_x0, args=(y_mask,),
                            constraints=test_constraints, bounds=test_bounds,
                            method='SLSQP')
        x = np.zeros(self.case['branch'].shape[0])
        x[self.e_set] = res_test.x[:self.case['branch'][self.e_set].shape[0]]
        return x

