import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

csv_file_path = "results/data/Chi2_vs_AC_voltage_iron.csv"

if not os.path.exists(csv_file_path):
    csv_file_path = "Chi2_vs_AC_voltage_tungsten.csv"  # Fallback to root path

df = pd.read_csv(csv_file_path)

x_col = df.columns[0]
y_col = df.columns[1]

df = df[(df[x_col] > 0) & (df[y_col] > 0)]

x_data = df.iloc[:, 0].values
y_data = df.iloc[:, 1].values


def exp_growth_model(x, A, a, C):
    """Exponential growth model: A * exp(a * x) + C"""
    return A * np.exp(a * x) + C


p0 = [min(y_data), 0.01, min(y_data)]
bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

# Fit every point with an exponential increase function
popt, _ = curve_fit(exp_growth_model, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)

A_fit, a_fit, C_fit = popt

y_pred = exp_growth_model(x_data, *popt)
ss_res = np.sum((y_data - y_pred) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"--- FIT RESULTS ---")
print(f"Amplitude (A)       : {A_fit:.4e}")
print(f"Exponent coeff (a)  : {a_fit:.4f}")
print(f"Offset (C)          : {C_fit:.4e}")
print(f"R^2 Score (Linear)  : {r_squared:.6f}")

x_smooth = np.linspace(min(x_data), max(x_data), 500)
y_smooth = exp_growth_model(x_smooth, *popt)

plt.figure(figsize=(8, 5))

plt.plot(
    x_data,
    y_data,
    "o",
    color="tab:blue",
    markersize=6,
    linewidth=1.5,
    label="Experimental Data",
)

plt.plot(
    x_smooth,
    y_smooth,
    linestyle="--",
    color="tab:red",
    linewidth=2,
    label=f"Exponential Fit ($R^2 = {r_squared:.4f}$)",
)

plt.xlabel("AC Voltage (V)", fontsize=11)
plt.ylabel("Max Chi(2)", fontsize=11)
plt.title(
    "Max Chi(2) vs. AC Voltage (Tungsten)",
    fontsize=13,
)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.show()