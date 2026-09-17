from classes.simulator import Simulator
from classes.curve_fitter import CurveFitter
import numpy as np
from classes.display import Display
import os
raw_header = ""
FILENAME = "results/waveforms/pure_silica/AC_frequency_dependence_2_1/AC_frequency_dependence_ps_2000_max005.txt"

if not os.path.exists(FILENAME):
    print(f"Error: '{FILENAME}' not found")
else:
    print(f"Displaying {FILENAME}...")

    with open(FILENAME, "r") as f:
        raw_header = f.readline().strip()
    
    params = {}
    for item in raw_header.split(","):
        if "=" in item:
            key, value = item.strip().split("=", 1)
            params[key] = value

    ac_voltage = float(params["AC_VOLTAGE"])
    ac_frequency = float(params["AC_FREQUENCY"])
    piezo_frequency = float(params["PIEZO_FREQUENCY"])
    print(ac_voltage)
    print(ac_frequency)
    print(piezo_frequency)

    data = np.loadtxt(FILENAME, skiprows=2, delimiter=",")
    
    time_array = data[:, 0]
    voltage_array = data[:, 1]
    
    print(max(voltage_array) - min(voltage_array))
    
    mask = (time_array >= 0.005) & (time_array <= 0.015)

    time_array = time_array[mask]
    voltage_array = voltage_array[mask]

    curve_fitter = CurveFitter(poled_fiber_length=0.3,
                            core_index=1.45,
                            effective_distance=31e-6,
                            field_adjustment_factor=2.0486,
                            periods_per_piezo_cycle=2.5,
                            piezo_frequency=piezo_frequency,
                            ac_voltage=ac_voltage,
                            ac_frequency=ac_frequency,
                            wavelength=1550e-9
                            )

    A, B, C, D, E, F, G = curve_fitter.fit_waveform(time_array=time_array,
                                        voltage_array=voltage_array,
                                        estimated_chi2=0.3e-12,
                                        tolerance_s1=0.01,
                                        tolerance_s2=0.01,
                                        min_C=0.3)
    
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