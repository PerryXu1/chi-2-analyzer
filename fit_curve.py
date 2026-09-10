from classes.simulator import Simulator
from classes.curve_fitter import CurveFitter
import numpy as np
from classes.display import Display
import os

FILENAME = "results/waveforms/AC_frequency_dependence_2/frequency_dependence_0.5k_max001.txt"

if not os.path.exists(FILENAME):
    print(f"Error: '{FILENAME}' not found at {FILENAME}")
else:
    print(f"Displaying {FILENAME}...")

    with open(FILENAME, "r") as f:
        raw_header = f.readline().strip()

    data = np.loadtxt(FILENAME, skiprows=2, delimiter=",")
    
    time_array = data[:, 0]
    voltage_array = data[:, 1]
    
    print(max(voltage_array) - min(voltage_array))
    
    # mask = (time_array >= 0) & (time_array <= 0.025)

    # time_array = time_array[mask]
    # voltage_array = voltage_array[mask]

    curve_fitter = CurveFitter(poled_fiber_length=0.3,
                            core_index=1.45,
                            effective_distance=41.1e-6,
                            field_adjustment_factor=1.77377708245,
                            periods_per_piezo_cycle=2.5,
                            piezo_frequency=10,
                            ac_voltage=1234,
                            ac_frequency=500,
                            wavelength=1550e-9
                            )

    A, B, C, D, E, F, G = curve_fitter.fit_waveform(time_array=time_array,
                                        voltage_array=voltage_array,
                                        estimated_chi2=0.32e-12,
                                        tolerance_s1=0.1,
                                        tolerance_s2=0.1,
                                        min_C=2)
    
    print(F)

    display = Display()

    chi2 = curve_fitter.get_chi2(C) * 1e12
    print(chi2)

    display.visualize_waveform(time=time_array, voltage=voltage_array)

    display.plot_fitted_curve(
                            time_array=time_array,
                            A=A,
                            B=B,
                            C=C,
                            D=D,
                            E=E,
                            F=F,
                            G=G)

    display.compare_fitted_curve(fitted_time_array=time_array,
                                time_array=time_array,
                                voltage_array=voltage_array,
                                A=A,
                                B=B,
                                C=C,
                                D=D,
                                E=E,
                                F=F,
                                G=G,
                                chi2=chi2)