# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:36:00 2025

@author: sander
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.dates as mdates
from scipy.optimize import differential_evolution


# === Load data from Excel ===
data = pd.read_excel("data.xlsx")
vix = data['VIX close']  # VIX index values
rv5 = data['realised variance']  # Realized variance values
dates_num = data['date']  # Dates in numeric YYYYMMDD format
returns = data['close-to-close log  return']  # Log returns

# === Convert numeric dates to datetime ===
dates_str = [str(int(d)) for d in dates_num]
dates = pd.to_datetime(dates_str, format='%Y%m%d')

# === Plot log returns over time ===
plt.figure(figsize=(25, 15))
plt.plot(dates, returns, 'k')  # Black line
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Show years only on x-axis
plt.ylabel(r'$r_t$', fontsize=20)
plt.xticks(pd.date_range('2000-01-01', '2024-01-01', freq='YS').to_pydatetime(), rotation=0)
plt.xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2025-01-01')])
plt.ylim([-15, 15])
plt.gca().tick_params(direction='out', labelsize=20)
plt.show()

# === Plot VIX over time ===
plt.figure(figsize=(25, 15))
plt.plot(dates, vix, 'k')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.title('VIX', fontsize=20)
plt.xticks(pd.date_range('2000-01-01', '2024-01-01', freq='YS').to_pydatetime(), rotation=0)
plt.xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2025-01-01')])
plt.ylim([0, 100])
plt.gca().tick_params(direction='out', labelsize=20)
plt.show()

# === Scatter plot of daily change in VIX vs returns ===
delta_vix = np.diff(vix)  # First difference of VIX
plt.figure(figsize=(14, 8))
plt.scatter(delta_vix, returns[1:], edgecolor='k', facecolor='none')  # Hollow circles
plt.xlabel(r'$\Delta \mathrm{VIX}_t$', fontsize=20)
plt.ylabel(r'$r_t$', fontsize=20)
plt.gca().tick_params(direction='out', labelsize=20)
plt.show()

# === Define negative log-likelihood function for GARCH(1,1) ===
def negative_log_likelihood(params, returns):
    mu, omega, alpha, beta = params
    T = len(returns)
    sigma2 = np.zeros(T)
    eps = returns - mu  # Residuals
    sigma2[0] = np.var(returns)  # Initialize with sample variance
    
    punishment = 0
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]  # GARCH recursion
        if sigma2[t] < 0:
            punishment += 1e6
        if np.isnan(sigma2[t]):
            punishment += 1e6

    loglikelihood = 0.5 * (np.log(2*np.pi) + np.log(sigma2) + eps**2 / sigma2)
    
    return np.sum(loglikelihood) + punishment

# === Starting parameter values for optimization ===
starting_values = [np.mean(returns), 0.02, 0.10, 0.85]

# === Evaluate initial negative log-likelihood ===
print('NLL at start:', negative_log_likelihood(starting_values, returns))

# === Parameter bounds ===
bounds = [(-1, 1), (0, 10), (0, 1), (0, 1)]

# === Optimize negative log-likelihood ===
result = differential_evolution(negative_log_likelihood, bounds=bounds, args=(returns,))

ML_parameters = result.x  # Estimated parameters
print('Estimated parameters:', ML_parameters)
print('Loglikelihood:', negative_log_likelihood(ML_parameters, returns))

# === Filter volatility path with estimated parameters ===
def filter_garch(params, returns):
    mu, omega, alpha, beta = params
    T = len(returns)
    sigma2 = np.zeros(T)
    eps = returns - mu
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    return sigma2

sigma2_filtered = filter_garch(ML_parameters, returns)

# === Plot returns and estimated volatility over time ===
plt.figure(figsize=(25, 15))
plt.plot(dates, returns, label='Data')
plt.plot(dates, np.sqrt(sigma2_filtered), 'r', label=r'Estimated $\sigma_{t|t-1}$ from GARCH')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(pd.date_range('2000-01-01', '2024-01-01', freq='YS').to_pydatetime(), rotation=0)
plt.xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2025-01-01')])
plt.ylim([-15, 15])
plt.gca().tick_params(direction='out', labelsize=20)
plt.legend(prop={'size': 15})
plt.show()

# === Calculate implied shocks (standardized residuals) ===
mu_ml = ML_parameters[0]
implied_epsilon = (returns - mu_ml) / np.sqrt(sigma2_filtered)
print('Mean, Var of implied epsilon:', np.mean(implied_epsilon), np.var(implied_epsilon))

# === Plot different variance measures over time ===
plt.figure(figsize=(25, 15))
plt.plot(dates, returns**2, 'ok', label='Squared returns')
plt.plot(dates, rv5, 'b', linewidth=3, label='Realized variance')
plt.plot(dates, (1/250) * vix**2, 'y', linewidth=3, label=r'Rescaled $VIX^2$')
plt.plot(dates, sigma2_filtered, 'r:', linewidth=2, label=r'Estimated $\sigma^2_{t|t-1}$ from GARCH')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(pd.date_range('2000-01-01', '2024-01-01', freq='YS').to_pydatetime(), rotation=0)
plt.xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2025-01-01')])
plt.ylim([0, 100])
plt.gca().tick_params(direction='out', labelsize=20)
plt.legend(prop={'size': 15})
plt.title('Variance of S&P500 returns by different measures', fontsize=20)
plt.show()

# === Check mean squared returns vs 1.4 * mean realized variance ===
print('Mean squared returns:', np.mean(returns**2))
print('1.4 * mean RV5:', 1.4 * np.mean(rv5))
