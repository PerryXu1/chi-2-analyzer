import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

csv_file_path = "results/data/Chi2_vs_AC_frequency_pure_silica.csv"

df = pd.read_csv(csv_file_path)

df = df[(df.iloc[:, 0] > 0) & (df.iloc[:, 1] > 0)]

x_data = df.iloc[:, 0].values  # AC Voltage or Frequency
y_data = df.iloc[:, 1].values  # Amplitude / Chi(2)


# Linear Model: y = m*x + b
def linear_model(x, m, b):
    return m * x + b


# Fit linear model directly using curve_fit (or np.polyfit)
popt, _ = curve_fit(linear_model, x_data, y_data)
m_fit, b_fit = popt

# Calculate R^2 score in linear space
y_pred = linear_model(x_data, *popt)
residuals = y_data - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("--- LINEAR FIT RESULTS ---")
print(f"Slope (m)     : {m_fit:.4e}")
print(f"Y-Intercept (b): {b_fit:.4e}")
print(f"R^2 Score     : {r_squared:.6f}")

# Generate linearly spaced points for smooth line rendering
x_smooth = np.linspace(min(x_data), max(x_data), 500)
y_smooth = linear_model(x_smooth, *popt)

# Linear Plot
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
    label=f"Linear Fit ($R^2 = {r_squared:.4f}$)",
)

plt.xlabel("AC Voltage (V)", fontsize=11)
plt.ylabel("Amplitude", fontsize=11)
plt.title(
    "Amplitude vs. AC Voltage, Tungsten (Linear Fit)",
    fontsize=13,
)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()