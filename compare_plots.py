import os
import matplotlib.pyplot as plt
import pandas as pd

file1_path = "results/data/AC_voltage_iron_1_fit_data.csv"
file2_path = "results/data/AC_voltage_2_fit_data.csv"

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

x1 = df1.iloc[:, 0].values
y1 = df1.iloc[:, 1].values
y1_norm = y1 / y1.max()

print(y1.max())

x2 = df2.iloc[:, 0].values
y2 = df2.iloc[:, 1].values
y2_norm = y2 / y2.max()

print(y2.max())

plt.figure(figsize=(8, 5))

plt.plot(
    x1,
    y1_norm,
    "o-",
    color="tab:blue",
    linewidth=1.5,
    markersize=5,
    label="Iron 1 (Normalized)",
)
plt.plot(
    x2,
    y2_norm,
    "s-",
    color="tab:orange",
    linewidth=1.5,
    markersize=5,
    label="AC Voltage 2 (Normalized)",
)

plt.xlabel("AC Voltage (V)", fontsize=11)
plt.ylabel("Normalized Amplitude", fontsize=11)
plt.title("Normalized Amplitude vs. AC Voltage Overlay", fontsize=13)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()