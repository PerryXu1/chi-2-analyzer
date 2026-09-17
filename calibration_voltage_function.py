import time as tm
import pyvisa
import numpy as np
from classes.interface import HP54600B, Agilent33120A, Agilent33250A, WaveformShape

# ========== CONSTANTS ==========
WAVELENGTH = 1550e-9
POLED_LENGTH = 0.3
EFFECTIVE_DISTANCE = 31e-6
CORE_INDEX = 1.45

ELECTRODE_MATERIAL = "TUNGSTEN"
TIME = 35
POLING_VOLTAGE = 3700
FREQUENCY_RATIO = 50  # AC_FREQUENCY / PIEZO_FREQUENCY

SHOT_DELAY = 0.5
VOLTAGE_MEASUREMENT_DELAY = 3

SHOTS = 5
VOLTS_PER_DIV = 10e-3

ac_frequencies = list(range(250, 6000, 500))
ac_voltages = list(range(5, 11))


try:
    awg_piezo = Agilent33250A(
        instrument_num=12,
        default_voltage=3.3,
        default_shape=WaveformShape.RAMP,
        default_symmetry=100,
    )
    awg_eo = Agilent33120A(
        instrument_num=10, 
        default_shape=WaveformShape.SINUSOID
    )

    scope = HP54600B(instrument_num=1)
    scope.reset()

    for ac_frequency in ac_frequencies:
        piezo_frequency = ac_frequency / FREQUENCY_RATIO

        awg_piezo.reset()
        awg_piezo.output_on()
        awg_piezo.change_frequency(piezo_frequency)

        for ac_voltage in ac_voltages:
            awg_eo.reset()
            awg_eo.change_frequency(ac_frequency)
            awg_eo.change_voltage(ac_voltage)

            estimated_ac_voltage = ac_voltage * (1.2 / 10)

            scope.set_screen(
                channel=1,
                volts_per_div=estimated_ac_voltage / 6,
                time_per_div=100e-3,
                vertical_offset=0,
                horizontal_offset=0,
                trigger_level=0,
                ext_trigger=False,
            )

            tm.sleep(VOLTAGE_MEASUREMENT_DELAY)
            measured_ac_voltage = 1000 * scope.get_amplitude(channel=1)  # measured amplified value (mV)

            time_per_div = 100e-3 / piezo_frequency

            scope.set_screen(
                channel=2,
                volts_per_div=VOLTS_PER_DIV,
                time_per_div=time_per_div,
                vertical_offset=3.5 * VOLTS_PER_DIV,
                horizontal_offset=5 * time_per_div,
                trigger_level=0,
                ext_trigger=True,
            )

            for shot in range(1, SHOTS + 1):
                filename = f"TEST_{ac_frequency}_{ac_voltage}_{shot}.txt"

                time_data, voltage_data = scope.acquire_signal(channel=2)

                header_info = (
                    f"PURE SILICA, ELECTRODE={ELECTRODE_MATERIAL}, TIME={TIME}, POLING_VOLTAGE={POLING_VOLTAGE}, "
                    f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
                    f"POLED_LENGTH={POLED_LENGTH}, CORE_INDEX={CORE_INDEX}, "
                    f"POLE=MAX, AC_VOLTAGE={measured_ac_voltage}, AC_FREQUENCY={ac_frequency}, PIEZO_FREQUENCY={piezo_frequency}\n"
                    f"Time(s),Voltage(V)"
                )

                data_matrix = np.column_stack((time_data, voltage_data))
                np.savetxt(
                    filename,
                    data_matrix,
                    delimiter=",",
                    header=header_info,
                    comments="",
                )
                
                tm.sleep(SHOT_DELAY)

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")