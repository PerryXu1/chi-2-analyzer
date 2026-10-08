from classes.simulator import Simulator
import numpy as np

time = np.linspace(0, 0.02, 2000)
sim = Simulator(time=time)

chi2_values = np.arange(0.0e-12, 0.55e-12, 0.05e-12)

for chi2 in chi2_values:
    chi2_pm_v = chi2 * 1e12
    filename = f"fourier_test_{chi2_pm_v:.2f}.csv"

    waveform = sim.modulated_sine(
        Vpp=0.05,
        phase_mod_cycles=2.5,
        phase_mod_frequency=40,
        fiber_length=0.3,
        wavelength=1550e-9,
        chi2=chi2,
        core_index=1.45,
        ac_voltage=237.5,
        eff_distance=3.1e-5,
        field_adjustment_factor=1.2,
        ac_frequency=2000,
        t_offset=0,
    )

    data = np.column_stack((time, waveform))

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="time,waveform",
        comments="",
        fmt="%.8e",
    )

    print(f"Saved {filename} (chi2 = {chi2:.2e} m/V)")