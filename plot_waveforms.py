import glob
import os
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

"""Animates sequential waveform text files on a single plot over time."""

FOLDER_PATH = "results/waveforms/potential_mode_mismatch"

FILE_PATTERN = os.path.join(FOLDER_PATH, "potential_mode_mismatch_min_nopiezo_10_3000_*.txt")
OUTPUT_GIF = "potential_mode_mismatch_min_nopiezo_10_3000.gif"
INTERVAL_MS = 150

file_list = glob.glob(FILE_PATTERN)

if not file_list:
    print(f"Error: No files found matching pattern '{FILE_PATTERN}'")
    exit()


def extract_shot_number(filename):
    match = re.search(r"_(\d+)\.txt$", filename)
    return int(match.group(1)) if match else 0


file_list.sort(key=extract_shot_number)

all_y_min = []
all_y_max = []

for file in file_list:
    data = np.loadtxt(file, skiprows=2, delimiter=",")
    all_y_min.append(np.min(data[:, 1]))
    all_y_max.append(np.max(data[:, 1]))

global_y_min = min(all_y_min)
global_y_max = max(all_y_max)

y_margin = (global_y_max - global_y_min) * 0.10 if global_y_max != global_y_min else 0.1
fixed_y_lims = (global_y_min - y_margin, global_y_max + y_margin)

first_data = np.loadtxt(file_list[0], skiprows=2, delimiter=",")
fixed_x_lims = (np.min(first_data[:, 0]), np.max(first_data[:, 0]))

fig, ax = plt.subplots(figsize=(9, 6))

(line,) = ax.plot([], [], color="royalblue", linewidth=1.5)

ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("Voltage (V)", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.6)

ax.set_xlim(fixed_x_lims)
ax.set_ylim(fixed_y_lims)

def update(frame_idx):
    current_file = file_list[frame_idx]

    with open(current_file, "r") as f:
        raw_header = f.readline().strip()

    params = raw_header.split(", ")
    lines = [", ".join(params[i : i + 3]) for i in range(0, len(params), 3)]
    formatted_title = f"Frame {frame_idx + 1}/{len(file_list)} — {os.path.basename(current_file)}\n" + "\n".join(lines)

    data = np.loadtxt(current_file, skiprows=2, delimiter=",")
    time_data = data[:, 0]
    voltage_data = data[:, 1]

    line.set_data(time_data, voltage_data)
    ax.set_title(formatted_title, fontsize=8, pad=10)

    return line,


if __name__ == '__main__':
    anim = FuncAnimation(
        fig,
        update,
        frames=len(file_list),
        interval=INTERVAL_MS,
        repeat=True,
    )

    fps = int(1000 / INTERVAL_MS)

    print(f"Saving animation to {OUTPUT_GIF}...")
    anim.save(OUTPUT_GIF, writer=PillowWriter(fps=fps))
    print(f"Successfully saved {OUTPUT_GIF}!")

    plt.tight_layout()
    plt.show()