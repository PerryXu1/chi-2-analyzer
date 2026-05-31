import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

"""
Collection of test signals to test the data analyzer with. Reference signal to check how well-adjusted the analyzer parameters are
"""

FloatArray = npt.NDArray[np.float64]

def visualize_waveform(time: FloatArray, voltage: FloatArray,
    title: str = "Signal Waveform",
    peak_indices: Optional[list[int]] = None, valley_indices: Optional[list[int]] = None,
    ax: Optional[plt.Axes] = None) -> None:
    """Generates a standardized Voltage vs. Time graph for oscilloscope data

    :param time: An array representing the time axis, typically measured in seconds (s)
    :type time: NDArray[float64]
    :param voltage: An array representing the captured voltage values, measured in Volts (V)
    :type voltage: NDArray[float64]
    :param title: The title text displayed at the top of the plot
    :type title: str, optional
    :param peak_indices: A list of array indices corresponding to localized peak maximums tracked by the segmented sweep algorithm
    :type peak_indices: list[int], optional
    :param valley_indices: A list of array indices corresponding to localized valley minimums tracked by the segmented sweep algorithm
    :type valley_indices: list[int], optional
    :param ax: An existing Matplotlib Axes object to plot onto. If None, a new figure and axes context will be created and displayed immediately
    :type ax: plt.Axes, optional

    :return: None
    :rtype: None
    """
    # Creates axes
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 5))
        show_plot = True
    else:
        show_plot = False

    # Plot signal
    ax.plot(time, voltage, label="Raw Signal", color="gray", alpha=0.6, linewidth=1)


    # Plot optional peak/trough markers
    if peak_indices is not None and len(peak_indices) > 0:
        ax.scatter(time[peak_indices], voltage[peak_indices], 
                   color="red", marker="v", s=40, zorder=5, label="Tracked Peaks")
    if valley_indices is not None and len(valley_indices) > 0:
        ax.scatter(time[valley_indices], voltage[valley_indices], 
                   color="blue", marker="^", s=40, zorder=5, label="Tracked Valleys")

    # Chart display settings
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Voltage (V)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    if show_plot:
        plt.tight_layout()
        plt.show()

def add_noise(signal: NDArray[float64], SNR: float) -> NDArray[float64]:
    """Adds Additive White Gaussian Noise (AWGN) to the input signal

    :param signal: An array representing the signal waveform, measured in Volts (V)
    :type signal: NDArray[float64]
    :param SNR: signal-to-noise ratio of the signal. Defines the intensity of Gaussian Noise. Measured in decibels (dB)
    :type SNR: float

    :return: the original inputted signal, with the added AWGN
    :rtype: NDArray[np.float64]
    """

    # Calculating noise distribution
    signal_power = np.mean(signal ** 2)
    SNR_linear = 10 ** (SNR / 10)
    noise_power = signal_power / SNR_linear
    noise_rms = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_rms, len(signal))

    return signal + noise


def noisy_sine(*, Vpp: float, time_array: NDArray[float64], T: float, t_offset: float = 0, SNR: float) -> NDArray[float64]:
    """Sinusoidal signal with Additive White Gaussian Noise (AWGN) at the specified SNR. Default signal is a cosine wave with its troughs at zero

    :param Vpp: peak-to-peak voltage of the signal waveform in Volts (V)
    :type Vpp: float
    :param time_array: Array of time values, measured in seconds (s)
    :type time_array: float
    :param T: period of the signal waveform in seconds (s)
    :type T: float
    :param t_offset: time phase offset of the waveform. Positive offset moves the signal to the right. Negative offset moves the signal to the left.
        Measured in seconds (s). No offset by default
    :type t_offset: float
    :param SNR: signal-to-noise ratio of the signal. Defines the intensity of Gaussian Noise. Measured in decibels (dB)
    :type SNR: float

    :return: an array of voltage values corresponding to the signal waveform
    :rtype: NDArray[np.float64]
    """
    signal = (Vpp / 2) * (1 + np.cos((2 * np.pi / T) * time_array))
    return add_noise(signal, SNR)
