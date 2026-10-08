import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np

file_pattern = "results/waveforms/AC_frequency_dependence_3/frequency_dependence_tungsten_1000_max*.txt"
filenames = glob.glob(file_pattern)


def extract_index(filepath):
    filename_only = os.path.basename(filepath)
    match = re.search(r"_(\d+)\.txt$", filename_only)
    return int(match.group(1)) if match else 0


filenames = sorted(filenames, key=extract_index)

if not filenames:
    print(f"Error: No files matching '{file_pattern}' found.")
else:
    print(f"Processing {len(filenames)} files for FFT spectrum overlay...")

    plt.figure(figsize=(10, 6))

    colors = plt.cm.inferno(np.linspace(0, 1, len(filenames)))

    for filename, color in zip(filenames, colors):
        file_idx = extract_index(filename)

        data = np.loadtxt(filename, skiprows=2, delimiter=",")

        time_data = data[:, 0]
        voltage_data = data[:, 1]

        N = len(time_data)
        dt = time_data[1] - time_data[0]
        fs = 1.0 / dt

        voltage_ac = voltage_data - np.mean(voltage_data)

        window = np.hanning(N)
        fft_vals = np.fft.rfft(voltage_ac * window)
        freq_data = np.fft.rfftfreq(N, d=dt)

        magnitude = np.abs(fft_vals) * (2.0 / N)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)

        freq_data_khz = freq_data / 1e3

        plt.plot(
            freq_data_khz,
            magnitude_db,
            label=f"File {file_idx:03d}",
            color=color,
            linewidth=1.2,
            alpha=0.85,
        )

    plt.title("FFT Frequency Spectra Comparison Across Fiber Shots", fontsize=12)
    plt.xlabel("Frequency (kHz)", fontsize=10)
    plt.ylabel("Magnitude (dB V)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.xlim(0, 30)
    plt.legend(fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.show()