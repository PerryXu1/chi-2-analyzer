import numpy as np
from scipy.optimize import curve_fit
from numpy import float64
from numpy.typing import NDArray

class CurveFitter:
    """A class of methods to fit a curve to experimental interferometer data, allowing for the
    characterization of the chi(2)
    
    :param periods_per_piezo_cycle: The amount of periods of phase the piezo phase modulator goes through in
    a single cycle of its driving source
    :type periods_per_piezo_cycle: float
    :param piezo_frequency: The frequency at which the piezo phase modulator is driven at in Hertz (Hz)
    :type piezo_frequency: float
    :param poled_fiber_length: Length of the section of the silica fiber that is poled in metres (m)
    :type poled_fiber_length: float
    :param wavelength: Wavelength of light coupled into the PSF in metres (m)
    :type wavelength: float
    :param core_index: Refractive index of the fibre core
    :type core_index: float
    :param field_adjustment_factor: Ratio of the simulated average electric field in the direction spanning both
        both electrodes to the electric field predicted by a parallel plate approximation
    :type field_adjustment_factor: float
    :param ac_voltage: The peak-to-peak AC voltage used to probe the PSF through the electro-optic effect
    :type ac_voltage: float
    :param effective_distance: Shortest distance between the two electrodes threaded into the PPSF
    :type effective_distance: float
    :param ac_frequency: The frequency of the AC source used to probe the PSF through the electro-optic effect
    :type ac_frequency: float
    """
    periods_per_piezo_cycle: float
    piezo_frequency: float
    poled_fiber_length: float
    wavelength: float
    core_index: float
    field_adjustment_factor: float
    ac_voltage: float
    effective_distance: float
    ac_frequency: float
    
    def __init__(self, *, poled_fiber_length: float, core_index: float, effective_distance: float, field_adjustment_factor: float,
                 periods_per_piezo_cycle: float, piezo_frequency: float,
                 ac_voltage: float, ac_frequency: float,
                 wavelength: float):
        self.poled_fiber_length = poled_fiber_length
        self.core_index = core_index
        self.effective_distance = effective_distance
        self.field_adjustment_factor = field_adjustment_factor
        self.periods_per_piezo_cycle = periods_per_piezo_cycle
        self.piezo_frequency = piezo_frequency
        self.ac_voltage = ac_voltage
        self.ac_frequency = ac_frequency
        self.wavelength = wavelength

    def fit_waveform(self, *, time_array: NDArray[float64], voltage_array: NDArray[float64], estimated_chi2: float,
                     tolerance_s1: float = 0.01, tolerance_s2: float = 0.05, min_C: float = 0.2) -> tuple[float]:
        """Fits the equation V = A*(1 + cos(B*(x-E) - C*cos(D*(x-F)))) + G to oscilloscope waveform data
        and returns the optimized C parameter.
        The guesses for the parameters A, E, G are determined through processing the voltage waveform.

        :param time_array: Time array (x-values)
        :type time_array: NDArray[float64]
        :param voltage_array: Voltage array (y-values)
        :type voltage_array: NDArray[float64]
        :param estimated_chi2: The guess of the chi2 of the fiber. Can be inputted manually or paired with the chi2 analyzer
        :type estimated_ch2: float
        :param tolerance_s1: The tolerances used to set the bounds on A and G in the first round of fitting
        :type tolerance_s1: float
        :param tolerance_s2: The tolerances used to set the bounds on A, B, D, G in the second round of fitting
        :type tolerance_s2: float
        :param min_C: the lower bound on the fit for C in both rounds of fitting
        :type min_C: float
        :return: The fit parameters
        :rtype: tuple[float]
        """

        # CALCULATE GUESSES
        A_guess = (np.max(voltage_array) - np.min(voltage_array)) / 2

        B_guess = self.periods_per_piezo_cycle * 2 * np.pi * self.piezo_frequency

        C_guess = ((np.pi * self.poled_fiber_length) / self.wavelength) * (estimated_chi2 / self.core_index) * (self.ac_voltage / (self.effective_distance * self.field_adjustment_factor))

        D_guess = 2 * np.pi * self.ac_frequency

        max_idx = np.argmax(voltage_array)
        E_guess = time_array[max_idx]
        
        F_guess = 0

        G_guess = float(np.min(voltage_array))

        def model_s1(x, A, C, E, G):
            return A * (1 + np.cos(B_guess * (x - E) - C * np.cos(D_guess * (x - F_guess)))) + G
        
        # Initial guesses
        p0_s1 = [A_guess, C_guess, E_guess, G_guess]

        # Set physical parameter boundaries
        lower_bounds_s1 = [A_guess * (1 - tolerance_s1), min_C, -np.inf, G_guess * (1 - tolerance_s1)]
        upper_bounds_s1 = [A_guess * (1 + tolerance_s1), np.inf, np.inf, G_guess * (1 + tolerance_s1)]

        optimized_parameters_s1, _ = curve_fit(
            model_s1,
            time_array,
            voltage_array,
            p0=p0_s1,
            bounds=(lower_bounds_s1, upper_bounds_s1),
            maxfev=100000,
        )

        A_s1, C_s1, E_s1, G_s1 = optimized_parameters_s1

        def model_s2(x, A, B, C, D, E, F, G):
            return A * (1 + np.cos(B * (x - E) - C * np.cos(D * (x - F)))) + G

        p0_s2 = [A_s1, B_guess, C_s1, D_guess, E_s1, F_guess, G_s1]

        lower_bounds_s2 = [A_s1 * (1 - tolerance_s2), B_guess * (1 - tolerance_s2), min_C, D_guess * (1 - tolerance_s2), -np.inf, -np.inf, G_s1 * (1 - tolerance_s2)]
        upper_bounds_s2 = [A_s1 * (1 + tolerance_s2), B_guess * (1 + tolerance_s2), np.inf, D_guess * (1 + tolerance_s2), np.inf, np.inf, G_s1 * (1 + tolerance_s2)]

        optimized_parameters, _ = curve_fit(
            model_s2,
            time_array,
            voltage_array,
            p0=p0_s2,
            bounds=(lower_bounds_s2, upper_bounds_s2),
            maxfev=100000,
        )
        print(optimized_parameters)


        return tuple(optimized_parameters)

    def get_chi2(self, C: float) -> float:
        """Gets the chi(2) from the optimized C parameter

        :param C: The optimized C parameter
        :type C: float
        :return: The chi(2) associated with the C parameter
        :rtype: float
        """

        chi2 = (C * self.wavelength * self.field_adjustment_factor * self.core_index * self.effective_distance) / (np.pi * self.poled_fiber_length * self.ac_voltage)
        return chi2