from simulator import Simulator
from analyzer import Analyzer
import numpy as np
from interface import Interface
import time

# time = np.linspace(0, 10e-3, 2000)
# sim = Simulator(time=time)
# signal = sim.noisy_modulated_sine(
#     Vpp=0.114,
#     phase_mod_cycles=2.5,
#     phase_mod_frequency=100,
#     fiber_length=0.3,
#     wavelength=1550e-9,
#     chi2=0.15e-12,
#     ac_frequency=3000,
#     ac_voltage=240,
#     SNR=50)
# max_m, min_m, mid_m = sim.get_ideal_markers(phase_mod_cycles=2.5,
#                                             phase_mod_frequency=100,
#                                             t_offset=0)
# analyzer = Analyzer(time=time,
#                     voltage=signal,
#                     core_index=1.52,
#                     wavelength=1550e-9,
#                     eff_distance=33e-6,
#                     ac_voltage=240,
#                     length=0.3,
#                     driver_frequency=100,
#                     phase_mod_cycles=2.5)

# chi2 = analyzer.analyze(window_size=50,
#                         dominant_sweep_factor=4,
#                         discontinuity_exclusion_factor=0.8,
#                         modulation_sweep_factor=4,
#                         modulation_overlap_factor=8,
#                         prominence=0.01,
#                         debug=False)

interface = Interface(instrument_num=1)
for i in range(3):
    interface.set_screen(channel=2,
                         volts_per_div=50e-3,
                         time_per_div=5e-3,
                         vertical_offset=0,
                         horizontal_offset=0,
                         trigger_level=0,
                         ext_trigger=False)
    time.sleep(3)
    interface.set_screen(channel=2,
                         volts_per_div=300e-3,
                         time_per_div=2e-3,
                         vertical_offset=0,
                         horizontal_offset=0,
                         trigger_level=0,
                         ext_trigger=False)
    time.sleep(3)

interface.close()