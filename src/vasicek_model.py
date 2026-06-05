
# Vasicek Model

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