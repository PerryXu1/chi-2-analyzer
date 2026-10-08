import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

LOWER_FREQ = 200
UPPER_FREQ = 500
MIN_PROMINENCE = 0.05
MIN_DISTANCE = 3

csv_file_path = "results/data/Chi2_vs_AC_frequency_pure_silica_3_1.csv"

df = pd.read_csv(csv_file_path)

freq_col = df.columns[0]
chi2_col = df.columns[1]

df = df[(df[freq_col] > 0) & (df[chi2_col] > 0)]
df = df[df[freq_col].between(LOWER_FREQ, UPPER_FREQ)]

x_data = df.iloc[:, 0].values
y_data = df.iloc[:, 1].values


def exp_decay_model(x, A, k, C):
    return A * np.exp(-k * x) + C

A_guess = max(y_data) - min(y_data)
k_guess = 1.0 / np.median(x_data)
C_guess = min(y_data)

p0 = [A_guess, k_guess, C_guess]
bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

popt, _ = curve_fit(exp_decay_model, x_data, y_data, p0=p0, bounds=bounds)
A_fit, k_fit, C_fit = popt

y_pred = exp_decay_model(x_data, *popt)
residuals = y_data - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"--- EXPONENTIAL DECAY FIT RESULTS ({LOWER_FREQ} Hz - {UPPER_FREQ} Hz) ---")
print(f"Amplitude (A) : {A_fit:.4e}")
print(f"Decay Rate (k): {k_fit:.4e}")
print(f"Offset (C)    : {C_fit:.4e}")
print(f"R^2 Score     : {r_squared:.6f}")

peak_indices, properties = find_peaks(
    y_data, prominence=MIN_PROMINENCE, distance=MIN_DISTANCE
)

peak_indices = np.insert(peak_indices, 0, 0)

peak_frequencies = x_data[peak_indices]
peak_amplitudes = y_data[peak_indices]
prominences = properties["prominences"]
peak_spacings = np.diff(peak_frequencies)

print("\n--- DETECTED PEAKS ---")
for i, (freq, amp, prom) in enumerate(
    zip(peak_frequencies, peak_amplitudes, prominences)
):
    print(
        f"Peak {i+1}: {freq:.2f} Hz (Max Chi2: {amp:.4e}, Prominence: {prom:.4e})"
    )

print("\n--- PEAK SPACINGS ---")
if len(peak_spacings) > 0:
    for i, spacing in enumerate(peak_spacings):
        print(f"Spacing between Peak {i+1} and Peak {i+2}: {spacing:.2f} Hz")
    print(f"Mean Peak Spacing: {np.mean(peak_spacings):.2f} Hz")
else:
    print("Fewer than 2 peaks detected; cannot compute spacings.")

x_smooth = np.linspace(min(x_data), max(x_data), 500)
y_smooth = exp_decay_model(x_smooth, *popt)

plt.figure(figsize=(8, 5))
plt.plot(
    x_data,
    y_data,
    marker="o",
    linestyle="-",
    color="tab:blue",
    markersize=6,
    label="Experimental Data",
)
plt.xlim(200, 450)
# plt.plot(
#     x_smooth,
#     y_smooth,
#     linestyle="-",
#     color="tab:red",
#     linewidth=2,
#     label=f"Exp Decay Fit ($R^2 = {r_squared:.4f}$)",
# )

for i, freq in enumerate(peak_frequencies):
    plt.axvline(
        x=freq,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label="Detected Peaks" if i == 0 else "",
    )

plt.xlabel("AC Frequency (Hz)", fontsize=11)
plt.ylabel("Max Chi(2)", fontsize=11)
plt.title(
    f"Max Chi(2) vs. AC Frequency ({LOWER_FREQ}–{UPPER_FREQ} Hz)",
    fontsize=13,
)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()