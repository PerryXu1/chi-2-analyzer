import time
import pyvisa
import numpy as np
from classes.interface import HP54600B
import time as tm

ELLIPTICITY = 15
AZIMUTHAL = 0
SHOTS = 20
CHI2 = 0.3230

CHANNEL = 2
VISA_ADDRESS = "GPIB0::7::INSTR"  

WAVELENGTH = 1550e-9
POLED_LENGTH = 0.3
EFFECTIVE_DISTANCE = 31e-6
CORE_INDEX = 1.45

ELECTRODE_MATERIAL = "TUNGSTEN"
TIME = 30
POLING_VOLTAGE = 3700

# CHANGE
AC_FREQUENCY = 6000

PIEZO_FREQUENCY = AC_FREQUENCY / 50

VOLTS_PER_DIV = 3e-3
TIME_PER_DIV = 100e-3 / (PIEZO_FREQUENCY)

# CHANGE
AC_VOLTAGE = 1025

try:
    scope = HP54600B(instrument_num=1)
    scope.reset()

    scope.set_screen(channel=2,
                    volts_per_div=VOLTS_PER_DIV,
                    time_per_div=TIME_PER_DIV,
                    vertical_offset=3.5 * VOLTS_PER_DIV,
                    horizontal_offset=5 * TIME_PER_DIV,
                    trigger_level=0,
                    ext_trigger=True)

    for i in range(SHOTS):
        filename = f"AC_frequency_dependence_ps_{str(AC_FREQUENCY)}_max{i + 1:03d}.txt"

        time, voltage = scope.acquire_signal(channel=2)

        header_info = (
            f"PURE SILICA, ELECTRODE={ELECTRODE_MATERIAL}, TIME={TIME}, POLING_VOLTAGE={POLING_VOLTAGE}, "
            f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
            f"POLED_LENGTH={POLED_LENGTH}, CORE_INDEX={CORE_INDEX}, "
            f"POLE=MAX, AC_VOLTAGE={AC_VOLTAGE}, AC_FREQUENCY={AC_FREQUENCY}, PIEZO_FREQUENCY={PIEZO_FREQUENCY}\n"
            f"Time(s),Voltage(V)"
        )
        
        data_matrix = np.column_stack((time, voltage))
        np.savetxt(filename, data_matrix, delimiter=",", header=header_info, comments="")
        tm.sleep(1)
        
    scope.close()

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")