import numpy as np
from numpy import ndarray

def analyzer(*, time: ndarray, voltage: ndarray,
             core_index: float, wavelength: float, eff_distance: float, ac_voltage: float, length: float,
             driver_frequency: float, max_min_epsilon_factor: float = 8) -> ndarray:
    """Gets the chi-2 of PPSF through oscilloscope data and other parameters
    
    
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
    :param max_min_epsilon_factor: from the guess, the analyzer sweeps period/max_min_epsilon_factor
        on both sides to find the true max/min. Default sweeps period/8
    :type max_min_epsilon_factor: float
    
    :return: array of chi-2 values calculated from the data. One for each full interval between peak and trough
    :rtype: np ndarray
    """
    # pre-processing
    array_size = len(time)
    period = 1/driver_frequency
    time_step = time[1] - time[0] # can also use XINCR from the scope
    
    # Get approximate min and max guesses
    global_max_index = np.argmax(voltage)
    global_min_index = np.argmin(voltage)
    
    max_guess_index_array = []
    min_guess_index_array = []
    
    first_max_index = array_size - 1
    first_min_index = array_size - 1
    period_index_offset = int(round(1.0/time_step))
    
    while first_max_index >= 0:
        first_max_index -= period_index_offset
    while first_min_index >= 0:
        first_min_index -= period_index_offset
    
    current_index = first_max_index
    while current_index < array_size:
        max_guess_index_array.append(current_index)
        current_index += period_index_offset
    current_index = first_min_index
    while current_index < array_size:
        min_guess_index_array.append(current_index)
        current_index += period_index_offset
        
    num_max = len(max_guess_index_array)
    num_min = len(min_guess_index_array)
    
    for max
    
    
    