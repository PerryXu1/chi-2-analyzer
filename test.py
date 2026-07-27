from classes.simulator import Simulator
from classes.curve_fitter import CurveFitter
import numpy as np
from classes.display import Display


time_array = np.linspace(0, 10e-3, 2000)
sim = Simulator(time_array)
voltage_array = sim.noisy_modulated_sine(Vpp=1,
                                         phase_mod_cycles=2.5,
                                         phase_mod_frequency=120,
                                         fiber_length=0.3,
                                         wavelength=1550e-9,
                                         chi2=0.15e-12,
                                         core_index=1.45,
                                         ac_voltage=240,
                                         eff_distance=41.1e-6,
                                         field_adjustment_factor=1.7,
                                         ac_frequency=3000,
                                         t_offset=0,
                                         SNR=50
                                         )

curve_fitter = CurveFitter(poled_fiber_length=0.3,
                           core_index=1.45,
                           effective_distance=41.1e-6,
                           field_adjustment_factor=1.7,
                           periods_per_piezo_cycle=2.5,
                           piezo_frequency=120,
                           ac_voltage=240,
                           ac_frequency=3000,
                           wavelength=1550e-9
                           )

A, B, C, D, E, F = curve_fitter.fit_waveform(time_array=time_array,
                                       voltage_array=voltage_array,
                                       estimated_chi2=0.15e-12)

display = Display()

chi2 = curve_fitter.get_chi2(C)
print(chi2)

display.visualize_waveform(time=time_array, voltage=voltage_array)

display.plot_fitted_curve(
                        time_array=time_array,
                        A=A,
                        B=B,
                        C=C,
                        D=D,
                        E=E,
                        F=F)

display.compare_fitted_curve(fitted_time_array=time_array,
                             time_array=time_array,
                             voltage_array=voltage_array,
                             A=A,
                             B=B,
                             C=C,
                             D=D,
                             E=E,
                             F=F)