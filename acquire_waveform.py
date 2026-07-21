from classes.analyzer import Analyzer
from classes.interface import Interface
import numpy as np
import time as tm

"""Script to capture pure oscilloscope data, transcribing the waveforms into a file.

    Calculates the AC voltage and stores it + other parameters in the file header.
"""

WAVELENGTH = 1550e-9
POLING_LENGTH = 0.3
EFFECTIVE_DISTANCE = 41.1e-6
CORE_INDEX = 1.5

SHOTS = 1

analyzer = Analyzer(core_index=1.5,
                    wavelength=1550e-9,
                    eff_distance=41.1e-6,
                    ac_voltage=240,
                    length=0.3,
                    driver_frequency=100,
                    phase_mod_cycles=2.5)

scope = Interface(instrument_num=1)
scope.reset()

for i in range(SHOTS):
    # scope.set_screen(channel=1,
    #                 volts_per_div=100e-3,
    #                 time_per_div=1e-3,
    #                 vertical_offset=0,
    #                 horizontal_offset=0,
    #                 trigger_level=0,
    #                 ext_trigger=False)
    # AC_VOLTAGE = 1000 * scope.get_amplitude(channel=1)
    AC_VOLTAGE = 1 #preset rn due to bad oscilloscope
    AC_VOLTAGE = 120 * AC_VOLTAGE

    scope.set_screen(channel=2,
                    volts_per_div=10e-3,
                    time_per_div=5e-3,
                    vertical_offset=30e-3,
                    horizontal_offset=0,
                    trigger_level=0,
                    ext_trigger=True)

    time, voltage = scope.acquire_signal(channel=2)
    
    filename = f"high_voltage_240V_100Hz_unpoled_{i+1:05d}.txt"

    # Construct header line with all metadata parameters
    header_info = (
        f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
        f"AC_VOLTAGE={AC_VOLTAGE}, POLING_LENGTH={POLING_LENGTH}, CORE_INDEX={CORE_INDEX}\n"
        f"Time(s),Voltage(V)"
    )

    data_matrix = np.column_stack((time, voltage))

    np.savetxt(
        filename,
        data_matrix,
        delimiter=",",
        header=header_info,
        comments="",
    )
    
scope.close()