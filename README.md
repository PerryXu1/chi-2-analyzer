# Electro-Optic Phase Modulator Interferometer Signal Analyzer

A specialized Python-based tool designed for processing high-frequency oscilloscope waveforms from phase-modulated fiber setups. The application isolates the dominant macroscopic wave from high-frequency modulation oscillations, algorithmically tracks locations of maximum phase-change, and extracts localized modulation extrema for computing the second order nonlinear susceptibility $\chi^{(2)}$ with high accuracy and precision.

---

## Signal Profile & Context

The physical dataset consists of high-frequency oscilloscope waveforms captured from an electro-optic phase modulator interferometer setup. The data is structurally shaped as a **dominant sine wave** carrying a high-frequency **phase modulation oscillation** on top of it. 

As the dominant macro-wave undergoes rapid phase transitions, the phase modulation oscillation creates localized sub-cycles. The primary physical goal of this project is to algorithmically isolate these sub-cycle peaks and troughs exactly at the points of maximum phase change (halfway between peaks and troughs) to calculate a precise $\chi^{(2)}$.

---

## How the Code Works

The project has two components, which allow for separate data processing and algorithm validation:

* **`Analyzer` Class:** Does the data processing. It consumes the raw oscilloscope data, smooths the signal to identify macroscopic transition boundaries, calculates exact phase midpoints, and applies localized neighborhood extraction to accurately locate the nearest carrier peaks and troughs.
* **`SignalSimulator` Class:** Provides a reference to check analysis parameters against. It synthesizes phase-modulated sine waves mathematically with configurable Additive Gaussian White Noise (AGWN). This allows you to calibrate and verify your analysis tracking parameters against a known baseline before applying them to experimental laboratory data.

---

## Installation & Dependencies

Ensure your execution environment has Python 3.9+ and the necessary scientific computing libraries installed:

```bash
pip install numpy scipy matplotlib
