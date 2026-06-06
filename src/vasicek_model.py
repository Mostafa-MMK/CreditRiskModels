
# Vasicek Model Paths
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt

def vasicek_model_func(r0, kappa, theta, sigma, T, dt, n_paths):
    rng = np.random.default_rng(seed=12)

    n_steps = int(T/dt)
    r = np.zeros((n_steps + 1, n_paths))
    r[0,:] = r0

    for i in range(n_steps):
        dw = rng.normal(size=n_paths)
        dr = kappa * (theta - r[i]) * dt + sigma * np.sqrt(dt) * dw
        r[i+1,:] = r[i,:] + dr

    return r


# Calibrating Vasicek Model Parameters using GSPC
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def vasicek_model_params_calc(tickers, ticker_name):
    snp500 = yf.download(tickers=tickers, start='2020-01-01', auto_adjust=True)['Close']
    snp500.columns = [ticker_name]
    df_returns = np.log(snp500/snp500.shift(1)).dropna()
    df_returns = df_returns.squeeze()

    systematic_factor = df_returns.rolling(window = 60).mean().dropna()
    x = systematic_factor[:-1].values.reshape(-1,1)
    y = systematic_factor[1:].values

    reg = LinearRegression()
    reg.fit(x,y)

    a = reg.intercept_
    b = reg.coef_[0]
    dt = 1/252

    # Solutions to Vasicek SDE (r_{t+Δt} = r_t e^{-κΔt} + θ(1 - e^{-κΔt}) + σ √((1 - e^{-2κΔt})/(2κ)) * ε_t)
    # κ = -ln(β) / Δt
    # θ = α / (1 - β)
    # σ = σ_η * √(2κ / (1 - e^{-2κΔt}))

    kappa = -np.log(b) / dt
    theta = a / (1 - b)
    r0 = systematic_factor.iloc[-1]
    
    residuals = y - reg.predict(x)
    sigma_eps = residuals.std(ddof=1)
    sigma = sigma_eps * np.sqrt(2 * kappa / 1 - np.exp(-2 * kappa * dt))

    print("\n ====== Calibrated Params =====")
    print(f'Kappa: {kappa:.4f}')
    print(f'Theta: {theta:.4f}')
    print(f'Sigma: {sigma:.4f}')
    print(f'r0: {r0:.4f}')

    return kappa, theta, sigma, r0


# Calculating Portolio Losses
def calc_portolio_loss(pd_unconditional, lgd, ead, rho, r_paths):
    default_threshold = norm.ppf(pd_unconditional)
    pd_conditional = norm.cdf((default_threshold - np.sqrt(rho)* r_paths) / np.sqrt( 1 - rho))
    portfolio_loss = pd_conditional * lgd * ead

    return portfolio_loss

# Basel IRB rho Calculation 
def basel_rho_calc(pd):
    term = (1 - np.exp(-50 * pd)) / (1- np.exp(-50))
    rho = 0.12 * term + 0.24 * (1 - term)
    
    return rho
