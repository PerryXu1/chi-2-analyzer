import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

csv_file_path = "results/data/Chi2_vs_AC_frequency_pure_silica_2.csv"

df = pd.read_csv(csv_file_path)

df = df[(df.iloc[:, 0] > 0) & (df.iloc[:, 1] > 0)]

x_data = df.iloc[:, 0].values  # AC Frequency
y_data = df.iloc[:, 1].values  # Max Chi(2)


# Exponential Decay Model: y = A * exp(-k * x) + C
def exp_decay_model(x, A, k, C):
    return A * np.exp(-k * x) + C


# Initial guesses: A ~ amplitude span, k ~ decay rate inverse scale, C ~ offset
A_guess = max(y_data) - min(y_data)
k_guess = 1.0 / np.median(x_data)
C_guess = min(y_data)

p0 = [A_guess, k_guess, C_guess]
bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

# Fit exponential decay model
popt, _ = curve_fit(exp_decay_model, x_data, y_data, p0=p0, bounds=bounds)
A_fit, k_fit, C_fit = popt

# Calculate R^2 score
y_pred = exp_decay_model(x_data, *popt)
residuals = y_data - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("--- EXPONENTIAL DECAY FIT RESULTS ---")
print(f"Amplitude (A) : {A_fit:.4e}")
print(f"Decay Rate (k): {k_fit:.4e}")
print(f"Offset (C)    : {C_fit:.4e}")
print(f"R^2 Score     : {r_squared:.6f}")

# Generate smooth grid
x_smooth = np.linspace(min(x_data), max(x_data), 500)
y_smooth = exp_decay_model(x_smooth, *popt)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(
    x_data,
    y_data,
    "o",
    color="tab:blue",
    markersize=6,
    label="Experimental Data",
)
plt.plot(
    x_smooth,
    y_smooth,
    linestyle="-",
    color="tab:red",
    linewidth=2,
    label=f"Exp Decay Fit ($R^2 = {r_squared:.4f}$)",
)

# Label DC Asymptote (A + C) as x -> 0
dc_asymptote = A_fit + C_fit
plt.axhline(
    y=dc_asymptote,
    color="purple",
    linestyle=":",
    linewidth=1.5,
    label=f"DC Asymptote ({dc_asymptote:.3e})",
)

plt.xlabel("AC Frequency (Hz)", fontsize=11)
plt.ylabel("Max Chi(2)", fontsize=11)
plt.title(
    "Max Chi(2) vs. AC Frequency, Pure Silica (Linear Scale)",
    fontsize=13,
)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()