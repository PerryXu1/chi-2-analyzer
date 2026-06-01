import numpy as np
from numpy.typing import NDArray
from numpy import float64
from simulator import Simulator

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
    
    time: float
    voltage: float
    core_index: float
    wavelength: float
    eff_distance: float
    ac_voltage: float
    length: float
    driver_frequency: float
    phase_mod_cycles: float
    
    def __init__(self, time: NDArray[float64], voltage: NDArray[float64], *,
                 core_index: float = 1.52, wavelength: float = 1550e-9, eff_distance: float = 33e-6,
                 ac_voltage: float = 1200, length: float = 0.3, driver_frequency: float = 150,
                 phase_mod_cycles = 2.9):
        self.time = time
        self.voltage = voltage
        self.core_index = core_index
        self.wavelength = wavelength
        self.eff_distance = eff_distance
        self.ac_voltage = ac_voltage
        self.length = length
        self.driver_frequency = driver_frequency
        self.phase_mod_cycles = phase_mod_cycles # technically may not be needed
        
    def _get_optima(self, *, array_size: int, period_index_offset: int,
                    max_min_epsilon_factor: float) -> tuple[NDArray[float64], NDArray[float64]]:
        """Helper function to get the max/min of the larger dominant sinusoid
        
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
            true_max = self.voltage[max_guess_index]

            if max_guess_index == first_max_index:
                max_index_array.append(max_guess_index)

            else:
                if idx == 0:
                    if max_guess_index - sweep_radius > 0:
                        start = max_guess_index - sweep_radius
                    else:
                        start = 0
                        stop = max_guess_index + sweep_radius
                elif idx == num_max - 1:
                    if max_guess_index + sweep_radius > array_size - 1:
                        stop = array_size - 1
                    else:
                        stop = max_guess_index + sweep_radius
                        start = max_guess_index - sweep_radius
                else:
                    start = max_guess_index - sweep_radius
                    stop = max_guess_index + sweep_radius

                for sweep_index in range(start, stop):
                    if self.voltage[sweep_index] > true_max:
                        true_max = self.voltage[sweep_index]
                        true_max_index = sweep_index
                
                max_index_array.append(true_max_index)

        for idx, min_guess_index in enumerate(min_guess_index_array):
            true_min_index = min_guess_index
            true_min = self.voltage[min_guess_index]

            if min_guess_index == first_min_index:
                min_index_array.append(min_guess_index)
            
            else:
                if idx == 0:
                    if min_guess_index - sweep_radius > 0:
                        start = min_guess_index - sweep_radius
                    else:
                        start = 0
                        stop = min_guess_index + sweep_radius
                elif idx == num_min - 1:
                    if min_guess_index + sweep_radius > array_size - 1:
                        stop = array_size - 1
                    else:
                        stop = min_guess_index + sweep_radius
                        start = min_guess_index - sweep_radius
                else:
                    start = min_guess_index - sweep_radius
                    stop = min_guess_index + sweep_radius

                for sweep_index in range(start, stop):
                    if self.voltage[sweep_index] < true_min:
                        true_min = self.voltage[sweep_index]
                        true_min_index = sweep_index
                
                min_index_array.append(true_min_index)

        return max_index_array, min_index_array
        
    

    def analyze(self, *, max_min_epsilon_factor: float = 4): # -> NDArray[float64]:
        """Gets the chi-2 of PPSF through oscilloscope data and other parameters through
        algorithmically finding specific optima of the data
        
        :param max_min_epsilon_factor: from the guess, the analyzer sweeps period/max_min_epsilon_factor
            on both sides to find the true max/min. Default sweeps period/8
        :type max_min_epsilon_factor: float
        
        :return: array of chi-2 values calculated from the data. One for each full interval between peak and trough
        :rtype: np ndarray
        """

        # pre-processing
        array_size = len(self.time)
        period = 1 / (self.phase_mod_cycles * self.driver_frequency)
        time_step = self.time[1] - self.time[0] # can also use XINCR from the scope
        period_index_offset = int(round(period/time_step))
        
        # getting min/max of dominant sinusoid
        max_index_array, min_index_array = self._get_optima(array_size=array_size,
                                                            period_index_offset=period_index_offset,
                                                            max_min_epsilon_factor=max_min_epsilon_factor)

        # finding positions of max phase change
        num_max = len(max_index_array)
        num_min = len(min_index_array)
        
        max_index_num = 0
        min_index_num = 0
        phase_change_index_array = []
        # if max_index_array[0] < min_index_array[0]:
        #     while (max_index_num == num_max or min_index_num == num_min):
                
        # else:
        
        return max_index_array, min_index_array


time = np.linspace(0, 10e-3, 1_000_000)
sim = Simulator(time)
signal = sim.noisy_modulated_sine(SNR=40)
analyzer = Analyzer(time, signal)
max_indices, min_indices = analyzer.analyze(max_min_epsilon_factor=50)
sim.visualize_waveform(signal, max_indices=max_indices, min_indices=min_indices)
        