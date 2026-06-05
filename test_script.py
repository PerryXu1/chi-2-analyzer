from simulator import Simulator
from analyzer import Analyzer
import numpy as np

time = np.linspace(0, 20e-3, 1_000_000)
sim = Simulator(time=time)
signal = sim.noisy_modulated_sine(
    Vpp=0.114,
    phase_mod_cycles=2.5,
    phase_mod_frequency=100,
    fiber_length=0.3,
    wavelength=1550e-9,
    chi2=0.15e-12,
    ac_frequency=3000,
    ac_voltage=240,
    SNR=50)
max_m, min_m, mid_m = sim.get_ideal_markers(phase_mod_cycles=2.5,
                                            phase_mod_frequency=100,
                                            t_offset=0)
analyzer = Analyzer(time=time,
                    voltage=signal,
                    core_index=1.52,
                    wavelength=1550e-9,
                    eff_distance=33e-6,
                    ac_voltage=240,
                    length=0.3,
                    driver_frequency=100,
                    phase_mod_cycles=2.5)
chi2 = analyzer.analyze(window_size=50,
                        dominant_sweep_factor=4,
                        discontinuity_exclusion_factor=0.8,
                        modulation_sweep_factor=4,
                        modulation_overlap_factor=8,
                        prominence=0.01,
                        debug=False)
print(chi2 * 1e12)
print(np.mean(chi2) * 1e12)
print(np.std(chi2) * 1e12)

sim.visualize_waveform(signal,
                       ideal_max_indices=max_m, ideal_mid_indices=mid_m, ideal_min_indices=min_m)
