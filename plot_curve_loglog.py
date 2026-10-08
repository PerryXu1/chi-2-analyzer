import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

LOWER_FREQ = 200
UPPER_FREQ = 6000

csv_file_path = "results/data/Chi2_vs_AC_frequency_pure_silica_3_1.csv"

df = pd.read_csv(csv_file_path)

freq_col = df.columns[0]
chi2_col = df.columns[1]

df = df[(df[freq_col] > 0) & (df[chi2_col] > 0)]
df = df[df[freq_col].between(LOWER_FREQ, UPPER_FREQ)]

x_data = df.iloc[:, 0].values
raw_y_data = df.iloc[:, 1].values

y_data = raw_y_data
log_y_data = np.log10(y_data)


def log10_lowpass(v, log_A, log_Vc, log_C):
    A = 10**log_A
    Vc = 10**log_Vc
    C = 10**log_C
    val = A / np.sqrt(1 + (v / Vc) ** 2) + C
    return np.log10(val)


p0_log = [
    np.log10(max(y_data) - min(y_data)),
    np.log10(np.median(x_data)),
    np.log10(min(y_data)),
]

bounds = ([-6, np.log10(min(x_data)), -6], [2, np.log10(max(x_data) * 10), 2])

popt_log, _ = curve_fit(
    log10_lowpass, x_data, log_y_data, p0=p0_log, bounds=bounds
)

A_fit = 10 ** popt_log[0]
Vc_fit = 10 ** popt_log[1]
C_fit = 10 ** popt_log[2]

dc_asymptote = A_fit + C_fit

log_y_pred = log10_lowpass(x_data, *popt_log)
residuals = log_y_data - log_y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_y_data - np.mean(log_y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"--- FIT RESULTS ({LOWER_FREQ} Hz - {UPPER_FREQ} Hz) ---")
print(f"Max Chi(2) (A)             : {A_fit:.4e}")
print(f"Corner AC Frequency (fc)  : {Vc_fit:.2f} Hz")
print(f"Normalized Offset (C)     : {C_fit:.4e}")
print(f"DC Asymptote (A + C)      : {dc_asymptote:.4e}")
print(f"R^2 Score (Log-Space)     : {r_squared:.6f}")

f_min_plot = 10.0
f_max_plot = 1e5

x_smooth = np.logspace(np.log10(f_min_plot), np.log10(f_max_plot), 500)
y_smooth = A_fit / np.sqrt(1 + (x_smooth / Vc_fit) ** 2) + C_fit

plt.figure(figsize=(8, 5))

plt.loglog(
    x_data,
    y_data,
    "o",
    color="tab:blue",
    markersize=6,
    linewidth=1.5,
    label="Experimental Data",
)

plt.loglog(
    x_smooth,
    y_smooth,
    linestyle="--",
    color="tab:red",
    linewidth=2,
    label=f"1st-Order Low-Pass Fit ($R^2 = {r_squared:.4f}$)",
)

plt.axhline(
    y=dc_asymptote,
    color="purple",
    linestyle=":",
    linewidth=1.8,
    label=f"Fitted Max $\\chi^{(2)}$ Asymptote ({dc_asymptote:.3e})",
)

if f_min_plot <= Vc_fit <= f_max_plot:
    plt.axvline(
        x=Vc_fit,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Corner Frequency ($f_c = {Vc_fit:.1f}$ Hz)",
    )

plt.xlim(f_min_plot, f_max_plot)

plt.xlabel("AC Frequency (Hz)", fontsize=11)
plt.ylabel("Max Chi(2)", fontsize=11)
plt.title(
    f"Max Chi(2) vs. AC Frequency ({LOWER_FREQ}–{UPPER_FREQ} Hz, Log-Log Scale)",
    fontsize=13,
)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()