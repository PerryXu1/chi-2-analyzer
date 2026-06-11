import matplotlib.pyplot as plt
import numpy as np

chi2 = 1e12 * np.loadtxt("results.txt")

median = np.median(chi2)
std = np.std(chi2, ddof=1)

# Print the values to the console
print(f"Median: {median:.6e}")
print(f"Standard Deviation: {std:.6e}")

# Create the histogram plot
plt.figure(figsize=(8, 5))
plt.hist(
    chi2,
    bins=30,
    edgecolor="black",
    color="skyblue",
    alpha=0.7,
)

# Add a vertical dashed line to mark the median
plt.axvline(
    median,
    color="red",
    linestyle="dashed",
    linewidth=2,
    label=f"Median: {median:.4e}",
)

# Label the chart
plt.title(r"Distribution of $\chi^2$ Values", fontsize=14)
plt.xlabel(r"$\chi^2$ Value (pm/V)", fontsize=12)
plt.ylabel("Frequency (Counts)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.show()