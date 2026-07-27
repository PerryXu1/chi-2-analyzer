import numpy as np
from scipy.optimize import curve_fit

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
    :param estimated_chi2: A guess of the chi(2) of the fiber in metres per volt (m/V)
    :type estimated_chi2: float
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
    
    def __init__(self, *, poled_fiber_length: float, core_index: float, effective_distance: float, field_adjustment_factor,
                 periods_per_piezo_cycle: float, piezo_frequency: float,
                 ac_voltage: float, ac_frequency: float,
                 wavelength: float,
                 estimated_chi2: float):
        self.poled_fiber_length = poled_fiber_length
        self.core_index = core_index
        self.effective_distance = effective_distance
        self.field_adjustment_factor = field_adjustment_factor
        self.periods_per_piezo_cycle = periods_per_piezo_cycle
        self.piezo_frequency = piezo_frequency
        self.ac_voltage = ac_voltage
        self.ac_frequency = ac_frequency
        self.wavelength = wavelength

    def fit_waveform_get_C(
        time_array: np.ndarray,
        voltage_array: np.ndarray,
        B_guess: float,
        C_guess: float,
        D_guess: float,
    ) -> float:
        """Fits the equation V = A*(1 + cos(B*(x-E) - C*cos(D*(x-E)))) + F to oscilloscope waveform data
        and returns the optimized C parameter.
        The guesses for the parameters A, E, F are determined through processing the voltage waveform.

        :param time_array: Time array
        :param y_data: Voltage array (1D numpy array)
        :param A_guess: Initial guess for Amplitude A
        :param B_guess: Initial guess for Frequency parameter B
        :param C_guess: Initial guess for Phase modulation depth C
        :param D_guess: Initial guess for Modulation frequency D
        :return: Optimized C parameter float
        """

        # 1. Define model function
        def waveform_model(x, A, B, C, D, E, F):
            return A * (1 + np.cos(B * (x - E) - C * np.cos(D * (x - E)))) + F

        # 2. Automatically estimate initial guesses for E and F from data
        F_guess = float(np.min(y_data))

        # Find time offset E near the signal peak
        max_idx = np.argmax(y_data)
        E_guess = float(x_data[max_idx])

        # Assemble full parameter vector: [A, B, C, D, E, F]
        initial_guesses = [A_guess, B_guess, C_guess, D_guess, E_guess, F_guess]

        # 3. Apply physical parameter bounds
        # A > 0, C > 0 (prevents negative scale/phase inversions)
        lower_bounds = [0.0, -np.inf, 0.0, -np.inf, -np.inf, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

        try:
            popt, _ = curve_fit(
                waveform_model,
                x_data,
                y_data,
                p0=initial_guesses,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,  # Ensure sufficient iterations for convergence
            )

            # Unpack fitted parameters
            _, _, C_fitted, _, _, _ = popt

            return float(C_fitted)

        except RuntimeError as e:
            print(f"Error: Fitting algorithm failed to converge: {e}")
            return None