#### part 2 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import direct
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
    # print(pack)
    print(nLLh)
    print(pack)
    return nLLh

bond_kts = [(10e-8,0.5)] * 3
bond_lambda = [(-1, -0.01)]
bond_h = [(10e-8, 0.05)] * 7

bond = bond_kts+bond_lambda +bond_h

opt_pack = minimize(neg_L_sum,
                    x0 = [.1349, .0061, .0506, -.0425] + [.0001] * 7,
                    bounds = bond,
                    method="Nelder-Mead",
                    # options={"maxiter": 2000}
                   )
opt_pack = direct(neg_L_sum, bounds=bond)
opt_x = opt_pack["x"]
# Backout rt path
_, rt_path = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:11], y_observed, tau)




# (3)
r_obs = []
for i in y_observed.index:
    yt = nss_coeff.loc[i]
    r_obs.append(
        part1.nss(10e-10, *yt)
    )
# plt.scatter(nss_coeff.index, r_obs, s = 5)
plt.scatter(nss_coeff.index, rt_path, s = 5)
plt.show()




def yield_plt(lb, ub, prarm):
    x = np.linspace(lb, ub, 500)
    nss_vect = np.vectorize(part1.nss, excluded=["a",'b','c',"d", "tau", "theta"])
    y = nss_vect(x, *prarm)
    plt.scatter(x,y, s=5)
    plt.show()

