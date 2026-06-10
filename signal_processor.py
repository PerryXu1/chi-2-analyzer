from display import Display
from analyzer import Analyzer
from interface import Interface

SHOTS = 19
V_AC_CALIBRATION_PASSES = 5 # Recalculate the V_AC every V_AC_CALIBRATION_PASSES shots

analyzer = Analyzer(core_index=1.52,
                    wavelength=1550e-9,
                    eff_distance=33e-6,
                    ac_voltage=240,
                    length=0.3,
                    driver_frequency=100,
                    phase_mod_cycles=2.5)

display = Display(title="Chi-2 Waveform")

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

        scope.set_screen(channel=2,
                        volts_per_div=50e-3,
                        time_per_div=1e-3,
                        vertical_offset=100e-3,
                        horizontal_offset=0,
                        trigger_level=2.5,
                        ext_trigger=True)

    time, voltage = scope.acquire_signal(channel=2)
    chi2, max_in, min_in = analyzer.analyze(time=time,
                            voltage=voltage,
                            window_size=50,
                            dominant_sweep_factor=4,
                            discontinuity_exclusion_factor=0.7,
                            discontinuity_exclusion_optima=14,
                            modulation_sweep_factor=5,
                            modulation_overlap_factor=10,
                            prominence=0.01,
                            debug=True)
    
    display.visualize_waveform(time=time, voltage=voltage, modulation_optima_indices=chi2, max_indices=max_in, min_indices=min_in)
    
    # chi2_array.extend(chi2)
    
scope.close()

# if len(chi2_array) != 0:    
#     with open("results.txt", "w") as file:
#         for value in chi2_array:
#             file.write(f"{value}\n")