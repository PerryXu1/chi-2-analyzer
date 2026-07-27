import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Optional
import matplotlib.pyplot as plt

class Display():
    """Class to visualize waveforms/plots using matplotlib

    :param title: the title of the plot
    :type title: str
    """

    title: str
    def __init__(self, title: str = "Signal Waveform"):
        self.title = title


    def visualize_waveform(self, time: NDArray[float64], voltage: NDArray[float64], *,
        max_indices: Optional[list[int]] = None, min_indices: Optional[list[int]] = None, mid_indices: Optional[list[int]] = None,
        ideal_max_indices: Optional[list[int]] = None, ideal_min_indices: Optional[list[int]] = None, ideal_mid_indices: Optional[list[int]] = None,
        modulation_optima_indices: Optional[list[int]] = None,
        ax: Optional[plt.Axes] = None) -> None:
        """Generates a standardized Voltage vs. Time graph for oscilloscope data

        :param time: An array representing the time values, measured in Seconds (s)
        :type time: NDArray[float64]
        :param voltage: An array representing the captured voltage values, measured in Volts (V)
        :type voltage: NDArray[float64]
        :param max_indices: A list of array indices corresponding to localized maximums tracked by the segmented sweep algorithm
        :type max_indices: list[int], optional
        :param min_indices: A list of array indices corresponding to localized minimums tracked by the segmented sweep algorithm
        :type min_indices: list[int], optional
        :param mid_indices: A list of array indices corresponding to midway points between the maxes and mins
        :type mid_indices: list[int], optional
        :param ideal_max_indices: A list of array indices corresponding to ideal localized maximums
        :type ideal_max_indices: list[int], optional
        :param ideal_min_indices: A list of array indices corresponding to ideal localized minimums
        :type ideal_min_indices: list[int], optional
        :param ideal_mid_indices: A list of array indices corresponding to ideal midway points between the maxes and mins
        :type ideal_mid_indices: list[int], optional
        :param modulation_optima_indices: A list of array indices corresponding to modulated localized optimums
        :type modulation_optima_indices: list[int], optional
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


        # Plot optional max/min/mid markers
        if max_indices is not None and len(max_indices) > 0:
            ax.scatter(time[max_indices], voltage[max_indices], 
                    color="red", marker="v", s=40, zorder=5, label="Dominant Maxima")
        if min_indices is not None and len(min_indices) > 0:
            ax.scatter(time[min_indices], voltage[min_indices], 
                    color="blue", marker="^", s=40, zorder=5, label="Dominant Minima")
        if mid_indices is not None and len(mid_indices) > 0:
            ax.scatter(time[mid_indices], voltage[mid_indices], 
                    color="green", marker="o", s=40, zorder=5, label="Max Phase Change")
        
        # Plot optional ideal max/min/mid markers
        if ideal_max_indices is not None and len(ideal_max_indices) > 0:
            ax.scatter(time[ideal_max_indices], voltage[ideal_max_indices], 
                    color="darkred", marker="s", s=40, zorder=5, label="Ideal Dominant Maxima")
        if ideal_min_indices is not None and len(ideal_min_indices) > 0:
            ax.scatter(time[ideal_min_indices], voltage[ideal_min_indices], 
                    color="darkblue", marker="s", s=40, zorder=5, label="Ideal Dominant Minima")
        if ideal_mid_indices is not None and len(ideal_mid_indices) > 0:
            ax.scatter(time[ideal_mid_indices], voltage[ideal_mid_indices], 
                    color="darkgreen", marker="s", s=40, zorder=5, label="Ideal Max Phase Change")
        
        #plot modulation minima and maxima
        if modulation_optima_indices is not None and len(modulation_optima_indices) > 0:
            ax.scatter(time[modulation_optima_indices], voltage[modulation_optima_indices], 
                    color="black", marker="o", s=40, zorder=5, label="Modulation Optima")
        
        # Chart display settings
        ax.set_title(self.title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Voltage (V)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_fitted_curve(self, *, time_array: NDArray[float64], A: float, B: float, C: float, D: float, E: float, F: float):

        voltage_array = A * (1 + np.cos(B * (time_array - E) - C * np.cos(D * (time_array - E)))) + F

        plt.figure(figsize=(8, 4))
        plt.plot(time_array, voltage_array, color="royalblue", linewidth=2, label="y = sin(x)")

        # 4. Add labels, grid, and title
        plt.title("Plot of fitted-chi2 signal", fontsize=12)
        plt.xlabel("Time", fontsize=10)
        plt.ylabel("Voltage", fontsize=10)
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()

        # 5. Display the graph
        plt.tight_layout()
        plt.show()

    def compare_fitted_curve(self, *, time_array: NDArray[float64], voltage_array: NDArray[float64], A: float, B: float, C: float, D: float, E: float, F: float):


            fitted_voltage_array = A * (1 + np.cos(B * (time_array - E) - C * np.cos(D * (time_array - E)))) + F
    
            plt.figure(figsize=(8, 4))
            plt.plot(time_array, voltage_array, color="royalblue", linewidth=2, label="Chi(2) Signal")
    
            # 4. Add labels, grid, and title
            plt.title("Plot of fitted chi(2) signal", fontsize=12)
            plt.xlabel("Time", fontsize=10)
            plt.ylabel("Voltage", fontsize=10)
            plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
            plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()
    
            # 5. Display the graph
            plt.tight_layout()
            plt.show()