import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Frequency values from 250 to 6000 Hz in steps of 250 (inclusive)
frequencies = np.arange(250, 6250, 250)
mean_amplitudes = []

for freq in frequencies:
    file_format = f"results/waveforms/AC_frequency_dependence_3/frequency_dependence_tungsten_{freq}_max"

    amplitudes = []
    # Loop over 001 to 009
    for i in range(1, 10):
        filename = f"{file_format}{i:03d}.txt"

        data = np.loadtxt(filename, skiprows=2, delimiter=",")

        voltage_array = data[:, 1]
        voltage_array = voltage_array[voltage_array > 0]

        amp = float(np.max(voltage_array) - np.min(voltage_array))
        amplitudes.append(amp)

    # Average amplitude for the current frequency
    mean_amplitudes.append(np.mean(amplitudes))

# Convert to numpy array and normalize to maximum
mean_amplitudes = np.array(mean_amplitudes)
normalized_amplitudes = mean_amplitudes

# Save results to CSV (includes both raw and normalized values)
output_df = pd.DataFrame(
    {
        "Frequency": frequencies,
        "Normalized Mean Amplitude": normalized_amplitudes,
    }
)
output_df.to_csv("frequency_dependence_results.csv", index=False)

# Linear Scale Plot (Normalized)
plt.figure(figsize=(8, 5))
plt.plot(
    frequencies,
    normalized_amplitudes,
    "o-",
    color="tab:blue",
    linewidth=1.5,
    markersize=6,
    label="Normalized Experimental Data",
)

plt.xlabel("Frequency (Hz)", fontsize=11)
plt.ylabel("Normalized Amplitude", fontsize=11)
plt.title("Normalized Amplitude vs. Frequency (Linear Scale)", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()