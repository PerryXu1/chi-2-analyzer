import numpy as np
from numpy.typing import NDArray
from numpy import float64
from simulator import Simulator
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from enum import Enum

from enum import Enum

class ModulationMode(Enum):
    """Represents the directional orientation of the high-frequency
    electro-optic (EO) carrier ripple transition near a detected signal midpoint.
    
    This is used to instruct the localized peak-finding window whether it should 
    expect a local maximum (crest) or local minimum (trough) on the immediate left 
    or right side of a macro-wave transition boundary.
    """

    PEAK_TO_TROUGH = 0
    """Indicates a falling dominant slope. The localized search treats the immediate left 
    of the midpoint as a high-frequency carrier maximum, and the immediate right as 
    a high-frequency carrier minimum.
    """

    TROUGH_TO_PEAK = 1
    """Indicates a rising dominant slope. The localized search treats the immediate left 
    of the midpoint as a high-frequency carrier minimum, and the immediate right as 
    a high-frequency carrier maximum.
    """
    
class Optima(Enum):
    """Represents whether an optima is a minima or maxima. Used to properly calculate the midpoints
    between minima and maxima while accounting for discontinuities that may distort a true optima
    """
    
    MINIMA = 0
    """Represents a minima of the lower frequency dominant wave
    """
    
    MAXIMA = 1
    """Represents a maxima of the lower frequency dominant wave
    """

class Analyzer:
    """Class of methods used to analyze an input signal waveform and calculate the chi-2
    
    :param time: array of times from the oscilloscope
    :type time: np ndarray
    :param voltage: array of voltage samples from the oscilloscope
    :type voltage: np ndarray
    :param core_index: refractive index of the fibre core
    :type core_index: float
    :param wavelength: wavelength of light coupled into the PPSF
    :type wavelength: float
    :param eff_distance: shortest distance between the two electrodes threaded into the PPSF
    :type eff_distance: float
    :param ac_voltage: the peak-to-peak AC voltage used to probe the PPSF through the electro-optic effect
    :type ac_voltage: float
    :param length: length of the electrodes overlapping. Approximately the length of the fibre
    :type length: float
    :param driver_frequency: frequency of the signal used to drive the piezo phase modulator. Frequency of larger wave
    :type driver_frequency: float
    :param phase_mod_cycles: The amount of 2pi phase cycles the piezo phase modulator moves through in one input signal cycle
    :type phase_mod_cycles: float
    """
    
    time: NDArray[float64]
    voltage: NDArray[float64]
    core_index: float
    wavelength: float
    eff_distance: float
    ac_voltage: float
    length: float
    driver_frequency: float
    phase_mod_cycles: float
    
    def __init__(self, time: NDArray[float64], voltage: NDArray[float64], *,
                 core_index: float = 1.52, wavelength: float = 1550e-9, eff_distance: float = 33e-6,
                 ac_voltage: float = 1200, length: float = 0.3, driver_frequency: float = 100,
                 phase_mod_cycles = 2.5):
        self.time = time
        self.voltage = voltage
        self.core_index = core_index
        self.wavelength = wavelength
        self.eff_distance = eff_distance
        self.ac_voltage = ac_voltage
        self.length = length
        self.driver_frequency = driver_frequency
        self.phase_mod_cycles = phase_mod_cycles # technically may not be needed
        
    def _get_optima(self, *, voltage: NDArray[float64], array_size: int, period_index_offset: int,
                    max_min_epsilon_factor: float,
                    discontinuity_exclusion_factor) -> tuple[NDArray[float64], NDArray[float64]]:
        """Helper function to get the max/min of the larger dominant sinusoid
        
        :param voltage: array of voltage values. This is typically an altered version of the original signal smoothed with a moving average
        :type voltage: np ndarray
        :param array_size: Size of the time array
        :type array_size: int
        :param period_index_offset: The change in indices of the time array between 1 period
        :type period_index_offset: int
        :param max_min_epsilon_factor: from the guess, the analyzer sweeps period/max_min_epsilon_factor
            on both sides to find the true max/min. Default sweeps period/8
        :type max_min_epsilon_factor: float
        
        :return: a tuple containing an array of the indices of the maxima within the time array
            and another array of the indices of the minima within the time array
        :rtype: tuple[NDArray[float64], NDArray[float64]]
        """
        
        # Get approximate min and max guesses
        global_max_index = np.argmax(self.voltage)
        global_min_index = np.argmin(self.voltage)
        
        max_amplitude = (self.voltage[global_max_index] - self.voltage[global_min_index])/2
        midline = self.voltage[global_min_index] + max_amplitude
        maxima_exclusion_bound = midline + discontinuity_exclusion_factor * max_amplitude
        minima_exclusion_bound = midline - discontinuity_exclusion_factor * max_amplitude
        
        max_guess_index_array = [] # approximate guesses for max indices
        min_guess_index_array = [] # approximate guesses for min indices
        
        first_max_index = global_max_index
        first_min_index = global_min_index
        
        while first_max_index - period_index_offset >= 0:
            first_max_index -= period_index_offset
        while first_min_index - period_index_offset >= 0:
            first_min_index -= period_index_offset
        
        current_index = first_max_index
        while current_index < array_size:
            max_guess_index_array.append(current_index)
            current_index += period_index_offset
        current_index = first_min_index
        while current_index < array_size:
            min_guess_index_array.append(current_index)
            current_index += period_index_offset

        max_index_array = [] # array of true max indices
        min_index_array = [] # array of true min indices
        sweep_radius = int(round(period_index_offset / max_min_epsilon_factor))
        
        num_max = len(max_index_array)
        num_min = len(min_index_array)

        for idx, max_guess_index in enumerate(max_guess_index_array):
            true_max_index = max_guess_index
            true_max = voltage[max_guess_index]

            if max_guess_index == first_max_index and self.voltage[max_guess_index] >= maxima_exclusion_bound:
                max_index_array.append(max_guess_index)

            else:
                if max_guess_index - sweep_radius > 0:
                    start = max_guess_index - sweep_radius
                else:
                    start = 0
                if max_guess_index + sweep_radius > array_size - 1:
                    stop = array_size - 1
                else:
                    stop = max_guess_index + sweep_radius

                for sweep_index in range(start, stop):
                    if voltage[sweep_index] > true_max:
                        true_max = voltage[sweep_index]
                        true_max_index = sweep_index
                
                if self.voltage[true_max_index] >= maxima_exclusion_bound:
                    max_index_array.append(true_max_index)

        for idx, min_guess_index in enumerate(min_guess_index_array):
            true_min_index = min_guess_index
            true_min = voltage[min_guess_index]

            if min_guess_index == first_min_index and self.voltage[min_guess_index] <= minima_exclusion_bound:
                min_index_array.append(min_guess_index)
            
            else:
                if min_guess_index - sweep_radius > 0:
                    start = min_guess_index - sweep_radius
                else:
                    start = 0
                if min_guess_index + sweep_radius > array_size - 1:
                    stop = array_size - 1
                else:
                    stop = min_guess_index + sweep_radius

                for sweep_index in range(start, stop):
                    if voltage[sweep_index] < true_min:
                        true_min = voltage[sweep_index]
                        true_min_index = sweep_index
                
                if self.voltage[true_min_index] <= minima_exclusion_bound:
                    min_index_array.append(true_min_index)

        return max_index_array, min_index_array

    def _smooth_voltage(self, window_size: int) -> NDArray[np.float64]:
        """Smooths the raw voltage trace using an ultra-fast 1D uniform filter

        :param window_size: The number of data points to include in the moving average window
        :type window_size: int

        :return: A 1D array containing the high-speed smoothed voltage trace
        :rtype: NDArray[np.float64]
        """

        smoothed_voltage = ndi.uniform_filter1d(self.voltage, size=window_size)
        
        return smoothed_voltage

    def _get_modulation_extrema(self, mid_index: int, search_radius: int, overlap_radius: int,
                                mode: ModulationMode, prominence: float = 0.01) -> tuple[int, int]:
        """Finds the correct high-frequency carrier modulation extrema (max and min) near a midpoint.
        Handles edge cases where the midpoint falls into the gap between two modulation cycles.

        :param mid_index: the array index of the calculated midpoint.
        :type mid_index: int
        :param search_radius: maximum number of array indices to search left and right.
        :type search_radius: int
        :param search_radius: maximum number of array indices to overlap (e.g. search past the midpoint)
        :type search_radius: int
        :param mode: direction of the carrier ripple modulation transition.
        :type mode: ModulationMode
        :param prominence: required vertical height the local feature must stand out.
        :type prominence: float

        :return: a tuple containing (left_optimum_index, right_optimum_index)
        :rtype: tuple[int, int]
        """
        array_size = len(self.voltage)
        
        start_idx = max(0, mid_index - search_radius)
        end_idx = min(array_size, mid_index + search_radius)
        
        # Slices overlap past the midpoint index using the overlap_radius parameter
        left_slice = self.voltage[start_idx : min(array_size, mid_index + overlap_radius)]
        right_slice = self.voltage[max(0, mid_index - overlap_radius) : end_idx]
        
        full_start = start_idx
        full_end = end_idx
        full_slice = self.voltage[full_start:full_end]
        
        all_peaks, _ = find_peaks(full_slice, prominence=prominence)
        all_troughs, _ = find_peaks(-full_slice, prominence=prominence)
        
        # Convert found features to absolute global indices
        all_peaks_abs = full_start + all_peaks
        all_troughs_abs = full_start + all_troughs
                
        # Filter features into your original left/right pools matching your slice boundaries
        l_peaks = all_peaks_abs[all_peaks_abs < min(array_size, mid_index)]
        l_troughs = all_troughs_abs[all_troughs_abs < min(array_size, mid_index)]
        
        r_peaks = all_peaks_abs[all_peaks_abs >= max(0, mid_index)]
        r_troughs = all_troughs_abs[all_troughs_abs >= max(0, mid_index)]
        
        # Determine the absolute closest features to the midpoint
        if len(l_peaks) > 0:
            max_L = l_peaks[-1]
        else:
            max_L = start_idx + int(np.argmax(left_slice))
            
        if len(l_troughs) > 0:
            min_L = l_troughs[-1]
        else:
            min_L = start_idx + int(np.argmin(left_slice))
            
        if len(r_peaks) > 0:
            max_R = r_peaks[0]
        else:
            right_base_idx = max(0, mid_index - overlap_radius)
            max_R = right_base_idx + int(np.argmax(right_slice))
            
        if len(r_troughs) > 0:
            min_R = r_troughs[0]
        else:
            right_base_idx = max(0, mid_index - overlap_radius)
            min_R = right_base_idx + int(np.argmin(right_slice))
        
        # 3. Calculate absolute distances to midpoint
        dist_max_L = abs(mid_index - max_L)
        dist_min_L = abs(mid_index - min_L)
        dist_max_R = abs(max_R - mid_index)
        dist_min_R = abs(min_R - mid_index)
        
        if mode == ModulationMode.PEAK_TO_TROUGH: # max on left and min on right expected
            if dist_max_L <= dist_min_L and dist_min_R <= dist_max_R:
                # CASE: midpoint perfectly within cycle
                return max_L, min_R
            else:
                # CASE: midpoint between cycles
                if dist_min_L < dist_max_R:
                    # left cycle boundary is closer, return its peak and trough
                    return max_L, min_L
                else:
                    # right cycle boundary is closer, return its peak and trough
                    return max_R, min_R
                    
        elif mode == ModulationMode.TROUGH_TO_PEAK: # min on left and max on right expected
            if dist_min_L <= dist_max_L and dist_max_R <= dist_min_R:
                # CASE: midpoint perfectly within cycle
                return max_R, min_L
            else:
                # CASE: midpoint between cycles
                if dist_max_L < dist_min_R:
                    # left cycle boundary is closer, return its peak and trough
                    return max_L, min_L
                else:
                    # right cycle boundary is closer, return its peak and trough
                    return max_R, min_R

    def analyze(self, *, window_size: int = 50, dominant_sweep_factor: float = 4, discontinuity_exclusion_factor: float = 0.8,
                modulation_sweep_factor: float = 4, modulation_overlap_factor: float = 8, prominence: float = 0.01,
                debug: bool = False) -> NDArray:
        """Gets the chi-2 of PPSF through oscilloscope data and other parameters through
        algorithmically finding specific optima of the data
        
        :param window_size: The number of data points to include in the moving average window. Used to find the dominant optima and midpoints
        :type window_size: int
        :param dominant_sweep_factor: from the guess, the analyzer sweeps period/dominant_sweep_factor
            on both sides to find the true max/min. Default sweeps period/4
        :type dominant_sweep_factor: float
        :param discontinuity_exclusion_factor: When finding optima, the algorithm will ignore any
            optima with an amplitude less than discontinuity_exclusion_factor * max_amplitude,
            where max_amplitude is global_max - global_min, which is a rough amplitude approximation
            Used to filter out discontinuities that can distort optima, thus distorting the chi2.
            Default is 0.8
        :type discontinuity_exclusion_factor: float
        :param modulation_sweep_factor: from the midpoint guess, the analyzer sweeps period/modulation_sweep_factor
            on both sides to find the carrier modulation max/min. Default sweeps period/4
        :type modulation_sweep_factor: float
        :param modulation_overlap_factor: from the midpoint guess, the analyzer sweeps period/modulation_overlap_factor past the midpoint
            on both sides to prevent edge cases when finding the carrier modulation max/min. Default sweeps period/8
        :type modulation_overlap_factor: float
        :param prominence: required vertical height the local feature must stand out. Measured in Volts (V)
        :type prominence: float
        :param debug: if debug is true, the function will return the modulation maxima and minima. Used to
            visualize calibrate the analysis parameters
        :type debug: boolean
        
        :return: array of chi-2 values calculated from the data. One for each full interval between peak and trough.
            If ijn debug mode, an array of indices corresponding to modulation optima will be returned.
        :rtype: np ndarray
        """

        # pre-processing
        array_size = len(self.time)
        period = 1 / (self.phase_mod_cycles * self.driver_frequency)
        time_step = self.time[1] - self.time[0] # can also use XINCR from the scope
        period_index_offset = int(round(period/time_step))

        # smooth out the voltage
        smoothed_voltage = self._smooth_voltage(window_size)
        
        # getting min/max of dominant sinusoid
        max_index_array, min_index_array = self._get_optima(voltage=smoothed_voltage,
                                                            array_size=array_size,
                                                            period_index_offset=period_index_offset,
                                                            max_min_epsilon_factor=dominant_sweep_factor,
                                                            discontinuity_exclusion_factor=0.8)

        # finding positions of max phase change
        num_max = len(max_index_array)
        num_min = len(min_index_array)
        
        max_index_num = 0
        min_index_num = 0
        phase_change_index_array = []
        
        dominant_amplitudes = []
        prev_optima = None
        
        while max_index_num != num_max and min_index_num != num_min:
            if max_index_num == 0 and min_index_array == 0:
                if max_index_array[max_index_num] < min_index_array[min_index_num]:
                    prev_optima = Optima.MAXIMA
                    max_index_num += 1
                else:
                    prev_optima = Optima.MINIMA
                    min_index_num += 1
            else:
                if max_index_array[max_index_num] < min_index_array[min_index_num]:
                    if prev_optima != Optima.MAXIMA:
                        phase_change_index_array.append(int((max_index_array[max_index_num] + min_index_array[min_index_num])/2))
                        dominant_amplitudes.append(self.voltage[max_index_array[max_index_num]]
                                                    - self.voltage[min_index_array[min_index_num]])
                        
                    max_index_num += 1
                elif max_index_array[max_index_num] > min_index_array[min_index_num]:
                    if prev_optima != Optima.MINIMA:
                        phase_change_index_array.append(int((max_index_array[max_index_num] + min_index_array[min_index_num])/2))
                        dominant_amplitudes.append(self.voltage[max_index_array[max_index_num]]
                                                    - self.voltage[min_index_array[min_index_num]])
                            
                    min_index_num += 1
        
        # Finding modulation optima
        modulation_max_indices = []
        modulation_min_indices = []
        sweep_radius = int(round(period_index_offset / modulation_sweep_factor))

        if max_index_array[0] < min_index_array[0]:
            first_slope_is_falling = True
        else:
            first_slope_is_falling = False

        for idx, phase_change_index in enumerate(phase_change_index_array):
            if first_slope_is_falling:
                if idx % 2 == 0:
                    mode = ModulationMode.PEAK_TO_TROUGH
                else:
                    mode = ModulationMode.TROUGH_TO_PEAK
            else:
                if idx % 2 == 0:
                    mode = ModulationMode.TROUGH_TO_PEAK
                else:
                    mode = ModulationMode.PEAK_TO_TROUGH
            
            local_max_idx, local_min_idx = self._get_modulation_extrema(
                mid_index=phase_change_index, 
                search_radius=sweep_radius,
                overlap_radius=int(round(sweep_radius/modulation_overlap_factor)),
                mode=mode, 
                prominence=prominence
            )
            
            modulation_max_indices.append(local_max_idx)
            modulation_min_indices.append(local_min_idx)
            
        modulation_max_indices = np.array(modulation_max_indices)
        modulation_min_indices = np.array(modulation_min_indices)
            
        modulation_amplitudes = self.voltage[modulation_max_indices] - self.voltage[modulation_min_indices]
        voltage_ratio = np.abs(modulation_amplitudes / dominant_amplitudes)
        
        chi2_coeff = (self.core_index * self.wavelength * self.eff_distance) / (np.pi * self.ac_voltage * self.length)
        chi2 = chi2_coeff * np.arcsin(voltage_ratio)
        
        chi2 = chi2[~np.isnan(chi2)]
        
        if not debug:
            return chi2
        else:
            return np.append(modulation_max_indices, modulation_min_indices)