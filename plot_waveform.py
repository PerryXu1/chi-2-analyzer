import os
import matplotlib.pyplot as plt
import numpy as np

"""Script to plot a single waveform file and dynamically display its header parameters as the title."""

FILENAME = "results/waveforms/waveforms_polarimeter_1/waveform_polarimeter_0_22.5_05.txt"

if not os.path.exists(FILENAME):
    print(f"Error: '{FILENAME}' not found at {FILENAME}")
else:
    print(f"Displaying {FILENAME}...")

    with open(FILENAME, "r") as f:
        raw_header = f.readline().strip()

    params = raw_header.split(", ")
    
    lines = [", ".join(params[i:i + 3]) for i in range(0, len(params), 3)]
    formatted_title = "\n".join(lines)

    data = np.loadtxt(FILENAME, skiprows=2, delimiter=",")

    time_data = data[:, 0]
    voltage_data = data[:, 1]

    plt.figure(figsize=(9, 5.5))

    plt.plot(time_data, voltage_data, color="royalblue", linewidth=1.5)

    plt.title(formatted_title, fontsize=9, pad=12)
    plt.xlabel("Time (s)", fontsize=10)
    plt.ylabel("Voltage (V)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()