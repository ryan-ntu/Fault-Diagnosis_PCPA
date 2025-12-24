from pypower.makeYbus import makeYbus
from pypower.runpf import ext2int
import numpy as np

def get_admittance(mpc):
    mpc_trans = ext2int(mpc)
    baseMVA, bus, _, branch = \
        mpc_trans["baseMVA"], mpc_trans["bus"], mpc_trans["gen"], \
            mpc_trans["branch"]
    Ybus, Yf, Yt  = makeYbus(baseMVA, bus, branch)

    return Ybus.toarray(), Yf.toarray(), Yt.toarray()

def construct_incidence_matrix(mpc):
    nb     = mpc['bus'].shape[0]
    nl     = mpc['branch'].shape[0]
    F_BUS, T_BUS = 0, 1   # 在 branch 数组里 from/to 的列索引

    Df = np.zeros((nb, nl), dtype=int)
    Dt = np.zeros((nb, nl), dtype=int)

    for e in range(nl):
        f = int(mpc['branch'][e, F_BUS] - 1)  # 转成 0-based
        t = int(mpc['branch'][e, T_BUS] - 1)
        Df[f, e] = 1
        Dt[t, e] = 1

    return Df, Dt