import time
import pyvisa
import numpy as np
from classes.interface import Interface
import time as tm

ELLIPTICITY = 15
AZIMUTHAL = 0
SHOTS = 10
CHI2 = 0.3230

CHANNEL = 2
VISA_ADDRESS = "GPIB0::7::INSTR"  

WAVELENGTH = 1550e-9
POLED_LENGTH = 0.3
EFFECTIVE_DISTANCE = 41.1e-6
CORE_INDEX = 1.45
AC_VOLTAGE = 475
ELECTRODE_MATERIAL = "IRON"
TIME = 50
POLING_VOLTAGE = 4500
AC_FREQUENCY = 500
PIEZO_FREQUENCY = AC_FREQUENCY / 50

VOLTS_PER_DIV = 500e-3
TIME_PER_DIV = 100e-3 / (PIEZO_FREQUENCY)

try:
    scope = Interface(instrument_num=1)
    scope.reset()

    scope.set_screen(channel=2,
                    volts_per_div=VOLTS_PER_DIV,
                    time_per_div=TIME_PER_DIV,
                    vertical_offset=3 * VOLTS_PER_DIV,
                    horizontal_offset=5 * TIME_PER_DIV,
                    trigger_level=0,
                    ext_trigger=True)

    for i in range(SHOTS):
        filename = f"frequency_dependence_iron_0.5k_max{i + 1:03d}.txt"

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
        tm.sleep(1)
        
    scope.close()

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")