from classes.display import Display
from classes.analyzer import Analyzer
from classes.interface import Interface

"""Script used to calibrate processor.py to ensure accurate acquisition and recording of chi2 values.

    This script produces a graph of the recorded chi-(2) signal waveform for each shot, along with markers
    of the dominant minima and maxima as found by the algorithm. Also plotted are markers of the
    modulation minima and maxima, used directly in the chi-(2) calculation. The dominant and modulation
    markers can be visually inspected for correctness for each shot, enabling the user to note the
    approximate frequency of errors.
    
    The dominant markers are expected to be on the minima and maxima of the dominant carrier sinusoid,
    excluding any deformed dominant optima due to discontinuities. The modulation markers are expected
    to be on the maxima and minima of a modulation oscillation roughly midway between adjacent dominant
    minima and maxima. At higher modulation frequencies, slight deviations from the exact midpoint may
    be acceptable, as neighbouring oscillations are of similar size.
    
    Two common forms of error are expected with subpar calibration.
    (1) The algorithm mistakenly calculates the modulation amplitude by taking the minima of one
        oscillation and the maxima of a neighbouring oscillation (rather than the same oscillation).
        
        This typically results in a much smaller resulting chi-(2) value, manifesting itself as a
        secondary peak to the left of the main peak in the chi-(2) distribution. Generally, when
        calibrating, this peak is to be minimized.
        
        This error is typically due to the midpoint guess being between oscillations and the sweep
        being too small to find the correct peaks and instead settling for the minima and maxima
        closest to it, even though they are from different oscillations.
        
        Common ways of dealing with this is to lower the modulation_sweep_factor, resulting in the
        algorithm sweeping a wider range of values from its midpoint guess, leading to an increased
        chance of actually finding the correct maxima/minima. From the experimental side, increasing
        the AC voltage frequency helps with this, as it increases the modulation oscillation density,
        allowing the midpoint sweep to capture the proper modulation optima within its bounds
        
    (2) The algorithm is misled by a discontinuity deforming a midpoint region, causing that section
        to have essentially no modulation oscillations. This makes the algorithm calculate the
        modulation amplitude by taking essentially the dominant minima and maxima.
        
        This typically results in a much larger chi-(2) value, manifesting itself as a secondary peak
        to the right to the main peak in the chi-(2) distribution. Generally, when calibrating, this
        peak is to be minimized.
        
        The phase modulator in the interferometer is driven by a sawtooth wave, which ramps up before
        dropping almost-discontinuously. When this discontinuity in the driving signal happens, it
        creates a discontinuity in the chi-(2) signal. When the discontinuity coincides with an optima,
        it generally lowers its amplitude and distorts it. When the discontinuity coincides with a
        midpoint, it generally destroys the modulation oscillations in that section. Since most of the
        modulation oscillations are destroyed, the algorithm sweeps far to get maxima/minima, leading
        it to capture the dominant optima (or close) as the modulation optima.
        
        Common ways of dealing with this is to increase the discontinuity_exclusion_optima. When
        sweeping from the midpoint, the algorithm will count the amount of optima (above a certain
        prominence) found within the sweep. By setting the discontinuity_exclusion_optima appropriately,
        these errors can be mitigated. On the hardware side, an external trigger from the original
        sawtooth signal can be used. By setting the trigger on falling edge at 0V, the discontinuity
        can be synced to t=0 on the oscilloscope. By setting the delay, the discontinuity can be set
        to the edge of the screen (where it causes less problems) or moved off the screen entirely.
    
"""

SHOTS = 20
V_AC_CALIBRATION_PASSES = 10 # Recalculate the V_AC every V_AC_CALIBRATION_PASSES shots

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
                        volts_per_div=200e-3,
                        time_per_div=0.8e-3,
                        vertical_offset=400e-3,
                        horizontal_offset=5e-3,
                        trigger_level=0,
                        ext_trigger=True)

    time, voltage = scope.acquire_signal(channel=2)
    modulation_indices, max_indices, min_indices = analyzer.analyze(time=time,
                            voltage=voltage,
                            window_size=50,
                            dominant_sweep_factor=4,
                            discontinuity_exclusion_factor=0.7,
                            discontinuity_exclusion_optima=14,
                            modulation_sweep_factor=4,
                            modulation_overlap_factor=8,
                            prominence=0.02,
                            voltage_ratio_acceptance=0.7,
                            debug=True)
    
    display.visualize_waveform(time=time,
                               voltage=voltage,
                               modulation_optima_indices=modulation_indices,
                               max_indices=max_indices,
                               min_indices=min_indices)
        
scope.close()