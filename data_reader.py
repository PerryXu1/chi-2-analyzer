import numpy as np

chi2_values = np.loadtxt("results.txt")

mean_val = np.mean(chi2_values)
std_val = np.std(chi2_values, ddof=1)

print(f"mean: {mean_val:.6f}")
print(f"stddev: {std_val:.6f}")