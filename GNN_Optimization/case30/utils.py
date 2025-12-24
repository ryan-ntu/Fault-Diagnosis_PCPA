import pandas as pd
import torch
import numpy as np
from pypower.runpf import ext2int
from pypower.api import rundcpf, case30
from pypower.bustypes import bustypes
from pypower.makeBdc import makeBdc
from torch_geometric.data import DataLoader
from itertools import combinations
from torch.utils.data import random_split
from numpy import transpose, r_, matrix
from pypower.idx_bus import GS, PD
from copy import deepcopy

from torch_geometric.data import Data, InMemoryDataset
import scipy.io

def get_admittance_matrix(mpc):
    """
    Calculate the admittance matrix of a power system.

    Parameters:
    mpc (dict): The power system data.

    Returns:
    B (numpy.ndarray): The admittance matrix.
    """

    mpc_trans = ext2int(mpc)
    baseMVA, bus, gen, branch = \
        mpc_trans["baseMVA"], mpc_trans["bus"], mpc_trans["gen"], \
            mpc_trans["branch"]

    ref, pv, pq = bustypes(bus, gen)
    pvpq = matrix(r_[pv, pq])

    B, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)
    B = B.toarray()
    return B


def run_simulation(case, Length, set, r, H_lines, simulation_options):
    """Run the simulation with the given load data and return the results"""
    # Need change the copy case if the case is not same with the original case. Now copy case is case 118.

    results_list = []
    mean_load = 8
    std_load = 10
    for l in range(Length):
        load_data = pd.DataFrame(np.round(np.clip(np.random.normal(mean_load, std_load, (1,case['bus'].shape[0])),
                                                  0, 25), 2))
        label = ['0'] * len(H_lines)
        if r:
            for F_index in set:
                fault_type = np.random.choice(['Cutting', 'WeakAttack'])
                if fault_type == 'Cutting':
                    case['branch'][F_index, 3] *= np.random.randint(10000, 100000) / 100

                else:
                    case['branch'][F_index, 3] *= np.random.randint(150, 500) / 100

                label_index = H_lines.index(F_index)
                label[label_index] = '1'

        label_str = ''.join(label)
        case['bus'][:, PD] = load_data

        # Run the power flow
        results, success = rundcpf(case, ppopt=simulation_options)
        if not success:
            print('Power flow did not converge.')
        Va, Pbus = observe_state(results)
        B = get_admittance_matrix(case)
        results_list.append((Va, Pbus, label_str))
        case = deepcopy(case30())

    return case, results_list


def find_edge_indices_within_nodes(branch, nodes):
    """
    Find the indices of edges within a set of nodes.

    Parameters:
    branch_data (numpy.ndarray): The branch data.
    nodes (set): The set of nodes.

    Returns:
    edge_indices (list): The indices of edges within the node set.
    """
    edges = []
    for i in range(branch.shape[0]):
        if branch[i, 0] in nodes and branch[i, 1] in nodes:
            edges.append(i)
    return edges


def data_collection(mpc, length, e, args):
    """
    Collect data for training a model.

    Parameters:
    mpc (dict): The power system data.
    v_nodes (list): The list of nodes.
    e (list): The list of edges
    args (dict): Additional arguments.

    Returns:
    Va_list (list): The list of voltage angles.
    Pbus_list (list): The list of bus power injections.
    admittances (list): The list of admittance matrices.
    labels (list): The list of labels.
    """

    Va_list = []
    Pbus_list = []
    V_list = []
    labels = []

    for r in range(0, 8):
        for subset in combinations(e, r):
            mpc, simulation_results = run_simulation(mpc, length, subset, r, e, args)
            for Va, Pbus, label in simulation_results:
                Va_list.append(Va)
                Pbus_list.append(Pbus)
                labels.append(label)
    return Va_list, Pbus_list, labels


def split_dataset_randomly(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Randomly split the dataset into train, validation, and test sets.

    Parameters:
    dataset (PyTorch Geometric Dataset): The dataset to split.
    train_ratio (float): The proportion of the dataset to include in the train split.
    val_ratio (float): The proportion of the dataset to include in the validation split.
    test_ratio (float): The proportion of the dataset to include in the test split.

    Returns:
    train_data, val_data, test_data: Random splits of the dataset.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6, "Ratios must sum to 1"

    # Shuffle dataset indices
    total_size = len(dataset)
    indices = torch.randperm(total_size)

    # Calculate split sizes
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Ensure that sum of splits equals total_size

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subsets
    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    return train_data, val_data, test_data


def observe_state(results):
    """
    This function is used to observe the state of the power system. It extracts the voltage phase angles and power.
    Results: the information of the power system after running the power flow simulation. It is a dictionary with keys
    """
    Va = results['bus'][:, 8].reshape(-1, 1) * (np.pi / 180)
    P_inj = np.zeros((results['bus'].shape[0], 1))
    bus_indices = {int(bus): idx for idx, bus in enumerate(results['bus'][:, 0])}
    for i in range(results['gen'].shape[0]):
        bus_id = int(results['gen'][i, 0])
        if bus_id in bus_indices:
            P_inj[bus_indices[bus_id]] = results['gen'][i, 1]

    P_inj = (P_inj - results['bus'][:, 2].reshape(-1, 1)) / results['baseMVA']

    return Va, P_inj


def construct_incidence_matrix(mpc):
    """
    Construct the incidence matrix of a power system.

    Parameters:
    mpc (dict): The power system data.

    Returns:
    D (numpy.ndarray): The incidence matrix
    """
    n_nodes = mpc['bus'].shape[0]
    n_branches = mpc['branch'].shape[0]

    D = np.zeros((n_nodes, n_branches))
    Gamma = np.zeros((n_branches, n_branches))

    for i in range(n_branches):
        from_node = int(mpc['branch'][i, 0])
        to_node = int(mpc['branch'][i, 1])
        D[from_node - 1, i] = 1
        D[to_node - 1, i] = -1
        reactance = mpc['branch'][i, 3]
        Gamma[i, i] = 1 / reactance if reactance != 0 else 0

    return D, Gamma


class GNN_dataset(InMemoryDataset):
    def __init__(self, root, edge_index, transform=None, pre_transform=None):
        self.edge_index = edge_index
        super(GNN_dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data_case30.mat']

    @property
    def processed_file_names(self):
        return ['data_case30.pt']

    def download(self):
        # Not applicable
        pass

    def process(self):
        data_list = []

        # 遍历所有的原始文件路径
        for raw_path in self.raw_paths:
            # 加载每一个 .mat 文件
            mat = scipy.io.loadmat(raw_path)
            Vas = mat['Va']
            Pbusinj = mat['Pbus']
            labels = mat['labels']

            # 将标签转换为二进制格式
            labels_binary = np.array([list(map(int, list(label))) for label in labels])

            # 为每个样本创建图数据对象
            for i in range(Vas.shape[0]):
                x = torch.tensor(np.hstack([Vas[i], Pbusinj[i]]), dtype=torch.float)
                y = torch.tensor(labels_binary[i], dtype=torch.float)
                data = Data(x=x, edge_index=self.edge_index, y=y)
                data_list.append(data)

        # 整合所有样本数据
        data, slices = self.collate(data_list)

        # 保存到单一的处理文件中
        torch.save((data, slices), self.processed_paths[0])


def plot_curve(data_list_training, data_list_valid):
    """
    Plot the training and validation curves.

    Parameters:
    data_list_training (list): The training data.
    data_list_valid (list): The validation data.
    """
    import matplotlib.pyplot as plt

    plt.plot(data_list_training, label='Training')
    plt.plot(data_list_valid, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')


def flatten_graph_data(graph_data, num_nodes, feature_dim):
    flattened_data = []
    for data in graph_data:
        node_features = data.x
        if node_features.size(0) != num_nodes:
            raise ValueError(f"Expected {num_nodes} nodes, but got {node_features.size(0)} nodes.")

        flat_features = node_features.view(-1)

        if data.y is not None:
            flat_features = torch.cat([flat_features, data.y])

        flattened_data.append(flat_features)

    flattened_data = torch.stack(flattened_data)
    return flattened_data


def BFS_algorithm(mpc, start_node, length_H):
    V_H = [start_node]
    neighbors = []
    # choose the neighbors of the start node based on the degree of the node
    while len(V_H) < length_H:
        # find the neighbors of the last element within the V_H
        node = V_H[-1]
        for edge in mpc['branch']:
            if edge[0] == node and edge[1] not in V_H:
                neighbors.append(edge[1])
            if edge[1] == node and edge[0] not in V_H:
                neighbors.append(edge[0])
        neighbors = list(set(neighbors))
        # calculate the degree of each neighbor except the degree related to the node within V_H
        degrees = []
        for neighbor in neighbors:
            degree = 0
            for edge in mpc['branch']:
                if (edge[0] == neighbor and edge[1] not in V_H) or (edge[1] == neighbor and edge[0] not in V_H):
                    degree += 1
            degrees.append(degree)
        # choose the neighbor with the highest degree
        if degrees:
            max_degree = max(degrees)
            max_index = degrees.index(max_degree)
            if max_degree > 0:
                V_H.append(neighbors[max_index])
                neighbors.pop(max_index)
                degrees.pop(max_index)
            else:
                break
        else:
            break

    return V_H
