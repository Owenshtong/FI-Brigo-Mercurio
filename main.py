#### part 1 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy import optimize
from scipy.optimize import direct
from scipy.optimize import differential_evolution
import part1

# (1)
info = pd.DataFrame({
    "maturity": [2, 3, 5, 7, 10, 30],
    "yield_quote": [.0429, .0434, .0442, .0451, .0460, .0482],
})


T_bill = pd.DataFrame({
    "maturity": [1/12, 3/12, 6/12, 1],
    "yield_quote": [.0433, .0429, .0422, .0414]})
T_bill.set_index('maturity', inplace=True)
T_bill["P"] = 1 - T_bill.yield_quote * T_bill.index

# Solve for discount factor
p = T_bill.P
for t in range(6):
    fun = part1._root_equation(p)
    T = info.maturity[t]
    c = info.yield_quote[t]
    pT = fsolve(fun, args = (c,T), x0 = 0.9)[0]
    p[T] = pT

# Turn into zcy
y = (-1) * np.log(p) / p.index


#(2)
nss_vect = np.vectorize(part1.nss)
opt_result = minimize(lambda x: sum((nss_vect(y.index,*x) - y)**2),
                      # x0 =[0.057, -0.013, -0.014, -0.028, 1.175, 5.15],
                      x0=[0.05, -0.01, -0.01, -0.03, 1.15, 5.1],
                      method='L-BFGS-B')
x_op = opt_result["x"]

t = np.linspace(0.001, 35, 100)
nss_fit = nss_vect(t ,*x_op)

fig, ax = plt.subplots(figsize=(9, 5), dpi = 100)
ax.scatter(y.index, y * 100, color = "dodgerblue", marker="D",label = "Observed",s = 20)
ax.plot(t, nss_fit * 100, color = "red", label = "NSS Fitted")
plt.title("Smoothed NSS Curve", fontname="Times New Roman")
plt.xlabel("Maturity (t)")
plt.ylabel("y(t)%")
plt.legend(facecolor='none', edgecolor='none')
plt.gcf()
plt.savefig("nss.pdf")
plt.show()

# (3)
# market quoted vol
cap_quoted_vol = pd.DataFrame({
    0.95: [0.157, 0.176, 0.175, 0.168, 0.155, 0.142],
    1: [0.152, 0.164, 0.161, 0.156, 0.148, 0.126],
    1.05: [0.146, 0.157, 0.158, 0.152, 0.140, 0.121],
    "Maturity": [1,3,5,7,10,20]
})


# Solve the K_f for each maturity
K_atm_vect = np.vectorize(part1.K_atm, excluded=["nss_para", "tau"])
K_atm = K_atm_vect(cap_quoted_vol.Maturity, nss_para = x_op)


# strike with maturity
strike = pd.DataFrame(
    {
        "Maturity": [1,3,5,7,10,20],
        "K_atm": K_atm,
    }
)
cap_quoted_vol = cap_quoted_vol.melt(id_vars=['Maturity'], value_vars=[0.95, 1, 1.05],
                    var_name='Strike_times', value_name='IV_market')

cap_quoted_vol = pd.merge(cap_quoted_vol, strike,  how='left', on = "Maturity")
cap_quoted_vol.K_atm = cap_quoted_vol.K_atm * cap_quoted_vol.Strike_times
cap_quoted_vol["K_atm_zcb"]  = 1 / (1 + 0.25 * cap_quoted_vol.K_atm)

bms_cap_vect = np.vectorize(part1.bms_cap, excluded=["pack", "tau", "M"])
cap_quoted_vol["BlackPrice"] = bms_cap_vect(cap_quoted_vol.K_atm, cap_quoted_vol.Maturity, cap_quoted_vol.IV_market, pack = x_op)


# # Target function
BM_cap_vect = np.vectorize(part1.BrigoMercurio_cap, excluded=["t", "kappa", "theta", "sigma", "rt", "gamma", "x0","pack_nss" ,"tau"])
def sse(BM_para):
    cap_BM = BM_cap_vect(
        t = 0.003,
        K = cap_quoted_vol.K_atm_zcb,
        T = cap_quoted_vol.Maturity,
        kappa = BM_para[0],
        theta = BM_para[1],
        sigma = BM_para[2],
        rt = BM_para[3],
        gamma = BM_para[4],
        x0 = 0,
        pack_nss = x_op
    )
    market_premium = cap_quoted_vol.BlackPrice
    loss = sum(((cap_BM - market_premium)/(market_premium))**2)
    # loss = sum((cap_BM - market_premium) ** 2)
    print([BM_para, loss])
    return loss

# "kappa", "theta", "sigma", "rt", "gamma",
bunds = [(0.01,1),(0.001,0.2),(0.01,0.4),(0.01, 0.1),(0.01,0.5)]
opt_result = minimize(sse,
                      x0= [0.05      , 0.13299393, 0.04761863, 0.04746266, 0.09265682],
                      method='Nelder-Mead',
                      bounds=bunds,
                      options={'maxiter': 10000}
                      )
pack_BM = opt_result["x"]

pack_BM = [0.04640003, 0.1386506 , 0.04768863, 0.04745836, 0.09139257]
# [0.04640003, 0.1386506 , 0.04768863, 0.04745836, 0.09139257]  best on sse%
# [0.01323959, 0.09478633, 0.0479164 , 0.04820152, 0.06845098]  converges well on sse%
p_bm = BM_cap_vect(
    kappa=pack_BM[0],
    theta=pack_BM[1],
    sigma=pack_BM[2],
    rt=pack_BM[3],
    gamma=pack_BM[4],
    x0=0, K=cap_quoted_vol.K_atm_zcb, T=cap_quoted_vol.Maturity, t=0.003, pack_nss=x_op
)
sse_percent = sse(pack_BM)
cap_quoted_vol["BMPrice"] = p_bm

# Back out the IV
def f(sigma, K ,T ,pack, price):
    return part1.bms_cap(K ,T ,sigma, pack) - price
def root(K, T, price, pack):
   return fsolve(f, args=(K, T, pack, price), x0=0.1)
root_vect = np.vectorize(root, excluded=["pack"])
cap_quoted_vol["BM_IV"] = root_vect(
    K = cap_quoted_vol.K_atm,
    T = cap_quoted_vol.Maturity,
    price = cap_quoted_vol.BMPrice,
    pack = x_op
)



# Plot
M = [0.95, 1, 1.05]
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 15), dpi = 200)
for i in range(3):
    atm = cap_quoted_vol[cap_quoted_vol['Strike_times'] == M[i]]
    axj1 = ax[i,0]
    axj2 = ax[i,1]
    axj1.plot(atm.Maturity, atm.BlackPrice, color = "red", marker  = "D", markersize=4, linestyle = "-.", label = "Observed")
    axj1.plot(atm.Maturity, atm.BMPrice, color = "black",marker  = "D",markersize=4,label="Modeled")
    axj1.set_title(r"Brigo-Mercurio Price v.s. Black Price, $K="+ str(M[i]) + "K_f$")

    axj2.plot(atm.Maturity, atm.IV_market,"red", marker  = "D", linestyle = "-.", markersize=4,label="Observed")
    axj2.plot(atm.Maturity, atm.BM_IV, color = "black", marker  = "D", markersize=4,label="Modeled")
    axj2.set_title("Brigo-Mercurio IV v.s. Quoted IV, $K="+ str(M[i]) + "K_f$")


plt.legend(facecolor='none', edgecolor='none')
plt.tight_layout()
plt.gcf()
plt.savefig("cap_fit.pdf")
plt.show()