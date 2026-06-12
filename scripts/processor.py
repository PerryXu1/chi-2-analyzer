from classes.analyzer import Analyzer
from classes.interface import Interface
import numpy as np

"""Script that acquires chi2 signal waveforms from the oscilloscope, calculates
    the chi2 from the waveform and other parameters, and saves it to a file.
    
    The script also sets the oscilloscope screen settings properly before getting a measurement,
    so no need for manual oscilloscope calibration. The chi2 calculation algorithm may need
    to be adjusted for different conditions, such as larger or more frequent modulation optima
    per dominant period. This can be done through running processor_calibrator.py and visually
    checking for approximate frequency and checking for errors. More can be found in the documentation
    for that script.
    
    Alternatively, less nuanced calibration done can be done by running this file, and using
    data_reader.py to find check the chi2 distribution. Ideally, with no algorithmic error, the
    distribution of chi2 values should be a bell curve. When there are significant errors, there may
    be another peak located to the left and/or right of the ideal bell curve. The reasoning for these
    errors is found in processor_calibrator.py
"""


WAVELENGTH = 1550e-9
POLING_LENGTH = 0.3
EFFECTIVE_DISTANCE = 33e-6
CORE_INDEX = 1.52
NOMINAL_AC_VOLTAGE = 240

SHOTS = 1000
V_AC_CALIBRATION_PASSES = 100 # Recalculate the V_AC every V_AC_CALIBRATION_PASSES shots

analyzer = Analyzer(core_index=1.52,
                    wavelength=1550e-9,
                    eff_distance=33e-6,
                    ac_voltage=240,
                    length=0.3,
                    driver_frequency=100,
                    phase_mod_cycles=2.5)

scope = Interface(instrument_num=1)
scope.reset()

V_ac = None

chi2_array = []

for i in range(SHOTS):
    if i % V_AC_CALIBRATION_PASSES == 0:
        scope.set_screen(channel=1,
                        volts_per_div=100e-3,
                        time_per_div=1e-3,
                        vertical_offset=0,
                        horizontal_offset=0,
                        trigger_level=0,
                        ext_trigger=False)
        V_ac = 1000 * scope.get_amplitude(channel=1)
        V_ac = 240 # preset for now due to bad oscilloscope
        analyzer.set_ac_voltage(V_ac)

        scope.set_screen(channel=2,
                        volts_per_div=50e-3,
                        time_per_div=1e-3,
                        vertical_offset=100e-3,
                        horizontal_offset=5e-3,
                        trigger_level=0,
                        ext_trigger=True)

    time, voltage = scope.acquire_signal(channel=2)
    chi2 = analyzer.analyze(time=time,
                            voltage=voltage,
                            window_size=50,
                            dominant_sweep_factor=4,
                            discontinuity_exclusion_factor=0.7,
                            discontinuity_exclusion_optima=14,
                            modulation_sweep_factor=4,
                            modulation_overlap_factor=8,
                            prominence=0.01,
                            voltage_ratio_acceptance=0.7,
                            debug=False)
    
    chi2_array.extend(chi2)
    
scope.close()
chi2_array = np.array(chi2_array)

filename = "chi2.txt"

# Construct header line with all metadata parameters
header_info = (
    f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
    f"NOMINAL_AC_VOLTAGE={NOMINAL_AC_VOLTAGE}, POLING_LENGTH={POLING_LENGTH}, CORE_INDEX={CORE_INDEX}, NUM_SHOTS={SHOTS}\n"
    f"Chi-(2) (m/V)"
)

data_matrix = np.column_stack((chi2_array))

np.savetxt(
    filename,
    data_matrix,
    delimiter=",",
    header=header_info,
    comments="",
)