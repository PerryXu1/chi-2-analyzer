import time
import pyvisa
import numpy as np
from classes.interface import Interface

ELLIPTICITY = 15
AZIMUTHAL = 0
SHOTS = 100
CHI2 = 0.3230
VOLTS_PER_DIV = 5.3e-3

CHANNEL = 2
VISA_ADDRESS = "GPIB0::7::INSTR"  

WAVELENGTH = 1550e-9
POLED_LENGTH = 0.3
EFFECTIVE_DISTANCE = 41.1e-6
CORE_INDEX = 1.45
AC_VOLTAGE = 1006
ELECTRODE_MATERIAL = "TUNGSTEN"
TIME = 50
POLING_VOLTAGE = 4500
PIEZO_FREQUENCY = 120
AC_FREQUENCY = 3000

try:
    scope = Interface(instrument_num=1)
    scope.reset()

    scope.set_screen(channel=2,
                    volts_per_div=VOLTS_PER_DIV,
                    time_per_div=1e-3,
                    vertical_offset=3.5 * VOLTS_PER_DIV,
                    horizontal_offset=5e-3,
                    trigger_level=0,
                    ext_trigger=True)

    for i in range(SHOTS):
        filename = f"potential_mode_mismatch_min_nopiezo_10_3000_{i + 1:03d}.txt"

        time, voltage = scope.acquire_signal(channel=2)

        header_info = (
            f"ELECTRODE={ELECTRODE_MATERIAL}, TIME={TIME}, POLING_VOLTAGE={POLING_VOLTAGE}, "
            f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
            f"POLED_LENGTH={POLED_LENGTH}, CORE_INDEX={CORE_INDEX}, "
            f"POLE=MIN, AC_VOLTAGE={AC_VOLTAGE}, AC_FREQUENCY={AC_FREQUENCY}\n"
            f"Time(s),Voltage(V)"
        )
        
        data_matrix = np.column_stack((time, voltage))
        np.savetxt(filename, data_matrix, delimiter=",", header=header_info, comments="")
        
    scope.close()

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")