import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CSV_FILE = "results/chi2/chi2_frequency_response.csv"
df = pd.read_csv(CSV_FILE)

frequency = df["Frequency"].values
chi2 = df["Max Chi(2)"].values

def log10_low_pass(f, log_chi0, log_fc):
    fc = 10**log_fc
    return log_chi0 - 0.5 * np.log10(1 + (f / fc) ** 2)

p0_log = [np.log10(np.max(chi2)), np.log10(1000.0)]
log_chi2_data = np.log10(chi2)

popt_log, _ = curve_fit(log10_low_pass, frequency, log_chi2_data, p0=p0_log)

chi0_fit = 10 ** popt_log[0]
fc_fit = 10 ** popt_log[1]

residuals = log_chi2_data - log10_low_pass(
    frequency, popt_log[0], popt_log[1]
)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_chi2_data - np.mean(log_chi2_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("--- FIT RESULTS ---")
print(f"DC Response (chi0)   : {chi0_fit:.4f}")
print(f"Cutoff Frequency (fc): {fc_fit:.2f} Hz")
print(f"R^2 Score            : {r_squared:.6f}")

# 5. Plotting on Log-Log Scale
f_smooth = np.logspace(
    np.log10(min(frequency) * 0.1), np.log10(max(frequency) * 10), 1000
)
chi2_smooth = chi0_fit / np.sqrt(1 + (f_smooth / fc_fit) ** 2)

plt.figure(figsize=(8, 5.5))

plt.loglog(
    f_smooth,
    chi2_smooth,
    color="tab:blue",
    linewidth=2,
    label="Low-Pass Fit",
)
plt.loglog(
    frequency,
    chi2,
    "o",
    color="tab:blue",
    markersize=6,
    alpha=0.8,
    label="Data Points",
)

plt.axvline(
    x=fc_fit,
    color="red",
    linestyle="--",
    linewidth=1.8,
    label=f"Cutoff Frequency: {fc_fit:.1f} Hz",
)

plt.xlabel("Frequency (Hz)", fontsize=11)
plt.ylabel(r"Max $\chi^{(2)}$", fontsize=11)
plt.title(r"Frequency Response of $\chi^{(2)}$", fontsize=13)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()