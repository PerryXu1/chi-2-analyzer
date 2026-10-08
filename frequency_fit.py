import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.special import jv

from classes.curve_fitter import CurveFitter
from classes.display import Display

raw_header = ""
    ################################
    #                              #
    #            CHANGE            #
    #                              #
    ################################
FILENAME = "results/waveforms/pure_silica/AC_frequency_dependence_3_1/AC_frequency_dependence_ps_1000_max008.txt"

if not os.path.exists(FILENAME):
    print(f"Error: '{FILENAME}' not found")
else:
    print(f"Loading {FILENAME}...")

    with open(FILENAME, "r") as f:
        raw_header = f.readline().strip()
    
    params = {}
    for item in raw_header.split(","):
        if "=" in item:
            key, value = item.strip().split("=", 1)
            params[key] = value

    ac_voltage = float(params["AC_VOLTAGE"])
    ac_frequency = float(params["AC_FREQUENCY"])
    piezo_frequency = float(params["PIEZO_FREQUENCY"])

    data = np.loadtxt(FILENAME, skiprows=2, delimiter=",")
    
    time_array = data[:, 0]
    voltage_array = data[:, 1]
    
    ################################
    #                              #
    #            CHANGE            #
    #                              #
    ################################
    
    mask = (time_array >= 0.008) & (time_array <= 0.03)

    time_array = time_array[mask]
    voltage_array = voltage_array[mask]
    
    curve_fitter = CurveFitter(
        poled_fiber_length=0.3,
        core_index=1.45,
        effective_distance=31e-6,
        field_adjustment_factor=2.0486,
        periods_per_piezo_cycle=2.5,
        piezo_frequency=piezo_frequency,
        ac_voltage=ac_voltage,
        ac_frequency=ac_frequency,
        wavelength=1550e-9
    )

    ################################
    #                              #
    #            CHANGE            #
    #                              #
    ################################
    max_harmonic = 3
    search_radius_hz = 2 * piezo_frequency

    ################################
    #                              #
    #            CHANGE            #
    #                              #
    ################################
    A_guess, B_guess, C, D_guess, E_guess, F_guess, G_guess = curve_fitter.fit_waveform_frequency(
        time_array=time_array,
        voltage_array=voltage_array,
        estimated_chi2=0.15e-12,
        max_harmonic=max_harmonic,
        search_radius_hz=search_radius_hz,
        min_C=0.3
    )

    chi2 = curve_fitter.get_chi2(C) * 1e12
    print(f"Frequency-Fitted C parameter      : {C:.6f}")

    N = len(time_array)
    dt = time_array[1] - time_array[0]

    window = np.hanning(N)
    windowed_voltage = (voltage_array - np.mean(voltage_array)) * window
    fft_spectrum = np.abs(rfft(windowed_voltage)) * (2.0 / N)
    freq_array = rfftfreq(N, d=dt)

    harmonic_orders = np.arange(1, max_harmonic + 1)
    peak_magnitudes = []
    extracted_freqs = []

    for n in harmonic_orders:
        target_freq = n * ac_frequency
        
        freq_mask = (freq_array >= target_freq - search_radius_hz) & \
                    (freq_array <= target_freq + search_radius_hz)

        if np.any(freq_mask):
            sub_spectrum = fft_spectrum[freq_mask]
            sub_freqs = freq_array[freq_mask]
            
            max_idx = np.argmax(sub_spectrum)
            extracted_freqs.append(sub_freqs[max_idx])

            peak_val = sub_spectrum[max_idx]
            significant_bins = sub_spectrum[sub_spectrum > peak_val * 0.1]
            total_harmonic_amplitude = np.sqrt(np.sum(significant_bins**2))
            
            peak_magnitudes.append(total_harmonic_amplitude)
        else:
            extracted_freqs.append(target_freq)
            peak_magnitudes.append(1e-12)

    peak_magnitudes = np.array(peak_magnitudes)
    extracted_freqs = np.array(extracted_freqs)

    A_scale_plot = np.max(peak_magnitudes) / np.max(np.abs(jv(harmonic_orders, C)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    max_freq_khz = (max_harmonic + 1) * ac_frequency / 1e3
    ax1.plot(freq_array / 1e3, 20 * np.log10(fft_spectrum + 1e-12), label="FFT Spectrum Data", color="navy")
    ax1.plot(extracted_freqs / 1e3, 20 * np.log10(peak_magnitudes + 1e-12), "ro", label="Integrated Harmonics (RSS)")
    ax1.set_xlim(0, max_freq_khz)
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("FFT Spectrum of Signal")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.stem(harmonic_orders, peak_magnitudes, linefmt="r-", markerfmt="ro", basefmt="k-", label="Harmonic Peaks $|Y_n|$")
    
    n_fine = np.linspace(0.5, max_harmonic + 0.5, 200)
    bessel_curve = A_scale_plot * np.abs(jv(n_fine, C))
    ax2.plot(n_fine, bessel_curve, "b--", label=f"Bessel Fit ($C={C:.4f}$)")
    
    ax2.set_xlabel("Harmonic Order ($n$)")
    ax2.set_ylabel("Linear Peak Magnitude")
    ax2.set_title(f"Frequency Domain Fit ($C={C:.3f}$ pm/V)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    def full_time_domain_model(t, A_fit, B_fit, C_fit, D_fit, E_fit, F_fit, G_fit):
        return A_fit * (1.0 + np.cos(B_fit * (t - E_fit) - C_fit * np.cos(D_fit * (t - F_fit)))) + G_fit

    total_time_span = time_array[-1] - time_array[0]
    
    ################################
    #                              #
    #            CHANGE            #
    #                              #
    ################################
    tolerances = {
        "A": 0.05,                  # ±5% of guessed fringe amplitude
        "B": 0.05,                  # ±5% of piezo angular frequency
        "C": 0.20,                  # ±20% of frequency-domain C guess
        "D": 0.05,                  # ±5% of AC angular frequency
        "E": total_time_span / 2.0,  # Absolute time phase shift limit (sec)
        "F": 1.0 / ac_frequency,    # Absolute phase limit within 1 AC cycle (sec)
        "G": 0.25                   # ±25% of DC voltage offset bound
    }

    C_guess = C 

    p0_time = [A_guess, B_guess, C_guess, D_guess, E_guess, F_guess, G_guess]

    lower_bounds = [
        A_guess * (1 - tolerances["A"]),
        B_guess * (1 - tolerances["B"]),
        max(0.0, C_guess * (1 - tolerances["C"])),  # Clamp to 0.0 lower bound to prevent negative C
        D_guess * (1 - tolerances["D"]),
        E_guess - tolerances["E"],
        F_guess - tolerances["F"],
        G_guess - abs(G_guess) * tolerances["G"] if G_guess != 0 else -1.0
    ]

    upper_bounds = [
        A_guess * (1 + tolerances["A"]),
        B_guess * (1 + tolerances["B"]),
        C_guess * (1 + tolerances["C"]),
        D_guess * (1 + tolerances["D"]),
        E_guess + tolerances["E"],
        F_guess + tolerances["F"],
        G_guess + abs(G_guess) * tolerances["G"] if G_guess != 0 else 1.0
    ]

    time_bounds = (lower_bounds, upper_bounds)

    try:
        popt_time, _ = curve_fit(
            full_time_domain_model,
            time_array,
            voltage_array,
            p0=p0_time,
            bounds=time_bounds,
            maxfev=100000
        )
        A, B, C, D, E, F, G = popt_time
        chi2 = curve_fitter.get_chi2(C) * 1e12
        print("\nJoint time domain fit succeeded with relaxed C and bounded parameters:")
        print(f"  A={A:.4f}, B={B:.4f}, C={C:.4f}, D={D:.4f}, E={E:.6f}, F={F:.6f}, G={G:.4f}")
        print(f"Chi(2)     : {chi2:.6f}")

    except Exception as e:
        print(f"\nJoint time domain fit failed: {e}. Falling back to initial guesses.")
        A, B, C, D, E, F, G = A_guess, B_guess, C_guess, D_guess, E_guess, F_guess, G_guess

    display = Display()

    display.visualize_waveform(time=time_array, voltage=voltage_array)

    display.plot_fitted_curve(
        time_array=time_array,
        A=A_guess,
        B=B,
        C=C,
        D=D,
        E=E,
        F=F,
        G=G
    )

    display.compare_fitted_curve(
        fitted_time_array=time_array,
        time_array=time_array,
        voltage_array=voltage_array,
        A=A,
        B=B,
        C=C,
        D=D,
        E=E,
        F=F,
        G=G,
        chi2=chi2
    )