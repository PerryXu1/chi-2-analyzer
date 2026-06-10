import numpy as np

chi2_values = np.loadtxt("results.txt")

mean_val = np.median(chi2_values)
std_val = np.std(chi2_values, ddof=1)

print(f"median: {mean_val:.6f}")
print(f"stddev: {std_val:.6f}")