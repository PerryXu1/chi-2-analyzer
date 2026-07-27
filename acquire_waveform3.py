import time
import pyvisa
import numpy as np
from classes.interface import Interface

ELLIPTICITY = 15
AZIMUTHAL = 0
SHOTS = 10
CHI2 = 0.3230
VOLTS_PER_DIV = 300e-3

CHANNEL = 2
VISA_ADDRESS = "GPIB0::7::INSTR"  

WAVELENGTH = 1550e-9
POLED_LENGTH = 0.3
EFFECTIVE_DISTANCE = 41.1e-6
CORE_INDEX = 1.45
AC_VOLTAGE = 1150
ELECTRODE_MATERIAL = "TUNGSTEN"
TIME = 50
POLING_VOLTAGE = 4500
PIEZO_FREQUENCY = 60
AC_FREQUENCY = 3000

try:
    scope = Interface(instrument_num=1)
    scope.reset()

    scope.set_screen(channel=2,
                    volts_per_div=VOLTS_PER_DIV,
                    time_per_div=1.6e-3,
                    vertical_offset=3 * VOLTS_PER_DIV,
                    horizontal_offset=8e-3,
                    trigger_level=0,
                    ext_trigger=True)

    for i in range(SHOTS):
        filename = f"AC_voltage_dependence_60Hz_3000Hz_10V_{i + 1:03d}.txt"

        time, voltage = scope.acquire_signal(channel=2)

        header_info = (
            f"ELECTRODE={ELECTRODE_MATERIAL}, TIME={TIME}, POLING_VOLTAGE={POLING_VOLTAGE}, "
            f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
            f"POLED_LENGTH={POLED_LENGTH}, CORE_INDEX={CORE_INDEX}, "
            f"POLE=MAX, AC_VOLTAGE={AC_VOLTAGE}, AC_FREQUENCY={AC_FREQUENCY}, PIEZO_FREQUENCY={PIEZO_FREQUENCY}\n"
            f"Time(s),Voltage(V)"
        )
        
        data_matrix = np.column_stack((time, voltage))
        np.savetxt(filename, data_matrix, delimiter=",", header=header_info, comments="")
        
    scope.close()

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")