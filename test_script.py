from simulator import Simulator
from analyzer import Analyzer
import numpy as np

time = np.linspace(0, 10e-3, 1_000_000)
sim = Simulator(time=time)
signal = sim.noisy_modulated_sine(
    Vpp=0.114,
    phase_mod_cycles=2.5,
    phase_mod_frequency=100,
    fiber_length=0.3,
    wavelength=1550e-9,
    chi2=0.0168e-12,
    ac_frequency=5000,
    ac_voltage=1200,
    SNR=50)
max_m, min_m, mid_m = sim.get_ideal_markers(phase_mod_cycles=2.5, phase_mod_frequency=160)

analyzer = Analyzer(time=time, voltage=signal)
chi2 = analyzer.analyze()
print(chi2)
print(np.mean(chi2))
print(np.std(chi2))

sim.visualize_waveform(signal, ideal_max_indices=max_m, ideal_mid_indices=mid_m, ideal_min_indices=min_m)
