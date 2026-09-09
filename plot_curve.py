import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

csv_file_path = "results/data/AC_voltage_2_fit_data.csv"

df = pd.read_csv(csv_file_path)

df = df[(df.iloc[:, 0] > 0) & (df.iloc[:, 1] > 0)]

x_data = df.iloc[:, 0].values
raw_y_data = df.iloc[:, 1].values

# --- NORMALIZE Y VALUES TO MAXIMUM ---
max_y = np.max(raw_y_data)
y_data = raw_y_data / max_y
log_y_data = np.log10(y_data)


def log10_freq_response(f, A, fc, C):
    val = A / ((np.sqrt(1 + (f / fc) ** 2)) ** 3) + C
    return np.log10(np.maximum(val, 1e-12))


p0 = [max(y_data), 9.0, min(y_data)]

popt, _ = curve_fit(log10_freq_response, x_data, log_y_data, p0=p0)
A_fit, fc_fit, C_fit = popt

# Calculate R^2 in log-space
log_y_pred = log10_freq_response(x_data, *popt)
residuals = log_y_data - log_y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_y_data - np.mean(log_y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Normalized DC Amplitude (A) : {A_fit:.4f}")
print(f"Cutoff Frequency (fc)       : {fc_fit:.2f} Hz")
print(f"Normalized Offset (C)       : {C_fit:.4f}")
print(f"R^2 Score (Log-Space)       : {r_squared:.6f}")

# Smooth x values for log-log plot
x_smooth = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 500)
y_smooth = A_fit / ((np.sqrt(1 + (x_smooth / fc_fit) ** 2)) ** 3) + C_fit

# Log-Log Plot
plt.figure(figsize=(8, 5))
plt.loglog(
    x_data,
    y_data,
    "o",
    color="tab:blue",
    markersize=6,
    label="Normalized Data",
)
plt.loglog(
    x_smooth,
    y_smooth,
    linestyle="-",
    color="tab:red",
    linewidth=2,
    label=f"Normalized Fit ($R^2 = {r_squared:.4f}$)",
)

# Mark cutoff frequency
plt.axvline(
    x=fc_fit,
    color="green",
    linestyle="--",
    linewidth=1.5,
    label=f"Cutoff Frequency: {fc_fit:.1f} Hz",
)

plt.xlabel("Frequency (Hz)", fontsize=11)
plt.ylabel("Normalized Amplitude", fontsize=11)
plt.title(
    "Normalized Amplitude vs. Frequency (3rd-Order Response Fit)",
    fontsize=13,
)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()