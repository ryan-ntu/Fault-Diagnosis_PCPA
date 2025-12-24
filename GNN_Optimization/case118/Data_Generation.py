""""""""""""""""""""""""""""""""""""""""""""""Import necessary libraries"""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from pypower.api import ppoption, case30, case118
import pandas as pd
import scipy.io
import copy
from utils import find_edge_indices_within_nodes, data_collection, BFS_algorithm


"""""""""Parameters setting & data collection"""""""""
mpc = copy.deepcopy(case118())
baseMVA = mpc['baseMVA']
N = mpc['bus'].shape[0]
length_per_fault = 500

start_node = 20
length_V_H = 20

V_H = BFS_algorithm(mpc, start_node, length_V_H)
E_H = find_edge_indices_within_nodes(mpc['branch'], V_H)
print(f'The number of nodes in H is {len(V_H)}, and the number of edges is {len(E_H)} \n')
simulation_options = ppoption(PF_DC=True, VERBOSE=0, OUT_ALL=0)
Vas, Power_inj, Loads, labels = data_collection(mpc, length_per_fault, E_H, simulation_options)

data = {
    'Va': np.array(Vas),
    'Pbus': np.array(Power_inj),
    'Load': np.array(Loads),
    'labels': np.array(labels)
}
scipy.io.savemat('./raw/data_case118.mat', data)
print("Data collection is done!")
print("The data contains the following keys: ", data.keys(), "The length of the data is: ", len(data['Va']))
print("---------------------------------------------------------------------------------------------------------------")

