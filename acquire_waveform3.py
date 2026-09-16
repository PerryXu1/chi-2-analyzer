import time
import pyvisa
import numpy as np
from classes.interface import HP54600B, Agilent33120A, Agilent33250A, WaveformShape
import time as tm

# ========== CONSTANTS ==========
WAVELENGTH = 1550e-9
POLED_LENGTH = 0.3
EFFECTIVE_DISTANCE = 31e-6
CORE_INDEX = 1.45

ELECTRODE_MATERIAL = "TUNGSTEN"
TIME = 35
POLING_VOLTAGE = 3700
FREQUENCY_RATIO = 50 # AC_FREQUENCY / PIEZO_FREQUENCY

# ========== CHANGE ==========
SHOTS = 5
VOLTS_PER_DIV = 3e-3

# VOLTAGE FUNCTION
TAU = 1500
INITIAL_VALUE = 3
FINAL_VALUE = 10

def voltage_function(frequency: float) -> int:
    """Function that defines how the AC voltage will change with frequency
    
    :param frequency: The frequency the experiment is run at
    :type frequency: float
    :return: The voltage (integer)
    :rtype: int
    """
    A = FINAL_VALUE - INITIAL_VALUE
    return int(round(A * (1 - np.exp(-frequency / TAU)) + INITIAL_VALUE))

ac_frequencies = range(1000, 6001, 1000)

try:
    awg_piezo = Agilent33250A(instrument_num=12,
                              default_voltage=3.3,
                              default_shape=WaveformShape.RAMP,
                              default_symmetry=100)
    awg_eo = Agilent33120A(instrument_num=10,
                           default_shape=WaveformShape.SINUSOID)

    scope = HP54600B(instrument_num=1)
    scope.reset()

    for ac_frequency in ac_frequencies:
        awg_piezo.reset()
        awg_piezo.output_on()
        awg_eo.reset()
        
        piezo_frequency = ac_frequency / FREQUENCY_RATIO
        
        awg_piezo.change_frequency(piezo_frequency)
        awg_eo.change_frequency(ac_frequency)
        
        ac_voltage = voltage_function(ac_frequency) # set value
        awg_eo.change_voltage(ac_voltage)
        
        scope.set_screen(channel=1,
                    volts_per_div=ac_voltage / 8, # 5 divs so technically should be 10, 8 as buffer
                    time_per_div=100e-3,
                    vertical_offset=0,
                    horizontal_offset=0,
                    trigger_level=0,
                    ext_trigger=False)
        measured_ac_voltage = 1000 * scope.get_amplitude(channel=1) # measured amplified value

        TIME_PER_DIV = 100e-3 / (piezo_frequency)

        scope.set_screen(channel=2,
                    volts_per_div=VOLTS_PER_DIV,
                    time_per_div=TIME_PER_DIV,
                    vertical_offset=3.5 * VOLTS_PER_DIV,
                    horizontal_offset=5 * TIME_PER_DIV,
                    trigger_level=0,
                    ext_trigger=True)
        
        for shot in range(SHOTS):
            # CHANGE
            filename = f"AC_frequency_dependence_ps_{ac_frequency:01d}_max{shot + 1:03d}.txt"

            time, voltage = scope.acquire_signal(channel=2)

            header_info = (
                f"PURE SILICA, ELECTRODE={ELECTRODE_MATERIAL}, TIME={TIME}, POLING_VOLTAGE={POLING_VOLTAGE}, "
                f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
                f"POLED_LENGTH={POLED_LENGTH}, CORE_INDEX={CORE_INDEX}, "
                f"POLE=MAX, AC_VOLTAGE={measured_ac_voltage}, AC_FREQUENCY={ac_frequency}, PIEZO_FREQUENCY={piezo_frequency}\n"
                f"Time(s),Voltage(V)"
            )
            
            data_matrix = np.column_stack((time, voltage))
            np.savetxt(filename, data_matrix, delimiter=",", header=header_info, comments="")
            tm.sleep(1)
            

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")