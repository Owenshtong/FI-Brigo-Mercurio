#### part 2 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import direct
from scipy.optimize import differential_evolution

import part1
import part2

# (1)
nss_coeff = pd.read_excel("TP1 data 60201 W2025.xlsx", index_col=0)/100
tau = [0.25, 0.5, 1,3,5,10,30]
nss_vect = np.vectorize(part1.nss, excluded=["t"])
vault = []
for i in nss_coeff.index:
    nss_pack = nss_coeff.loc[i]
    yi = nss_vect(tau, *nss_pack)
    vault.extend(list(yi))
y_observed = pd.DataFrame(np.reshape(vault, (len(nss_coeff),7)), index=nss_coeff.index, columns=tau)
summ_stat = round(y_observed.describe().iloc[[1,2,4,5,6],] ,4)

# (2)
def neg_L_sum(pack):
    LLh, _ =  part2.L_sum(
        *pack[0:4], pack[4:11], y_observed, tau
    )
    nLLh = -1 * LLh
    print(nLLh)
    return nLLh

# bond_kts = [(.0001,0.5)] * 3
k_bond = [(0,.2)]
theta_bond = [(0,0.1)]
sigma_bond = [(0, 2)]
lambda_bond = [(-1, -0.001)]
h_bond = [(1e-5, 1e-2)]
bond = k_bond+theta_bond+sigma_bond+lambda_bond+h_bond*7


opt_pack = minimize(neg_L_sum,
                    x0 =opt_x,
                    bounds = bond,
                    method="Nelder-Mead",
                    # options={"maxiter": 2000}
                   )
opt_pack = direct(neg_L_sum, bounds=bond)
opt_x = opt_pack["x"]
opt_x =  [ 1.982e-01 , 2.713e-02,  4.996e-02, -1.042e-01 , 1.804e-05,
                  1.000e-05  ,6.405e-05 , 2.072e-04  ,2.266e-04 , 3.093e-04
                  ,2.451e-04]

# (3)
# Backout rt path
_, rt_path = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:11], y_observed, tau)


r_obs = []
for i in y_observed.index:
    yt = nss_coeff.loc[i]
    r_obs.append(
        part1.nss(10e-4, *yt)
    )
plt.scatter(nss_coeff.index, r_obs, s = 5)
plt.scatter(nss_coeff.index, rt_path, s = 5)
plt.show()






#
# def yield_plt(lb, ub, prarm):
#     x = np.linspace(lb, ub, 500)
#     nss_vect = np.vectorize(part1.nss, excluded=["a",'b','c',"d", "tau", "theta"])
#     y = nss_vect(x, *prarm)
#     plt.scatter(x,y, s=5)
#     plt.show()
#






