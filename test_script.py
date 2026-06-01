from simulator import Simulator
from analyzer import Analyzer
import numpy as np

time = np.linspace(0, 100e-3, 1_000_000)
sim = Simulator(time=time)
signal = sim.noisy_modulated_sine(phase_mod_frequency=160, SNR=40)
max_m, min_m, mid_m = sim.get_ideal_markers(phase_mod_frequency=160)

analyzer = Analyzer(time=time, voltage=signal)
chi2 = analyzer.analyze()
print(chi2)
print(np.mean(chi2))
print(np.std(chi2))

sim.visualize_waveform(signal, ideal_max_indices=max_m, ideal_mid_indices=mid_m, ideal_min_indices=min_m)
