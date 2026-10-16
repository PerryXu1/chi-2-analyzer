import os
import glob
import re
import sys
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.special import jv

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QLabel,
    QSlider,
    QScrollArea,
    QMessageBox,
)
import pyqtgraph as pg

from classes.curve_fitter import CurveFitter

# Set white background globally for pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class FiberGUIAppPyQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Fiber Characterization GUI (Full-Width Default)")
        self.resize(1450, 850)

        self.base_dir = "results/waveforms/pure_silica/AC_frequency_dependence_1_3"
        self.csv_record_path = "results/data/Chi2_vs_AC_frequency_pure_silica_1_3.csv"

        self.frequencies, self.shots_map = self.scan_data_files()

        self.freq_idx = 0 if self.frequencies else 0
        self.shot_idx = 0 if self.shots_map else 0
        self.mask_lower = 0.0
        self.mask_upper = 0.01
        self.is_fitted = False
        self._updating_from_ui = False

        # Advanced control default parameters
        self.est_chi2_val = 0.3e-12
        self.min_c_val = 0.3
        self.max_harmonic_val = 3

        # Default tolerances for fit variables
        self.tolerances = {
            "A": 0.05,
            "B": 0.50,
            "C": 0.80,
            "D": 0.05,
            "E": 0.50,
            "F": 1.0,
            "G": 0.25
        }

        self.fit_data = {}

        self.init_ui()
        self.load_current_file()

    def scan_data_files(self):
        pattern = os.path.join(self.base_dir, "AC_frequency_dependence_ps_*_max*.txt")
        files = glob.glob(pattern)

        freq_set = set()
        shots_map = {}

        for f in files:
            basename = os.path.basename(f)
            match = re.search(r"AC_frequency_dependence_ps_(\d+)_max(\d+)\.txt", basename)
            if match:
                freq = int(match.group(1))
                shot = int(match.group(2))
                freq_set.add(freq)
                if freq not in shots_map:
                    shots_map[freq] = set()
                shots_map[freq].add(shot)

        frequencies = sorted(list(freq_set))
        formatted_shots_map = {f: sorted(list(shots)) for f, shots in shots_map.items()}

        if not frequencies:
            frequencies = [200, 500, 1000, 2000, 4000, 6000]
            formatted_shots_map = {f: list(range(1, 31)) for f in frequencies}

        return frequencies, formatted_shots_map

    def get_current_filename(self):
        if not self.frequencies:
            return ""
        freq = self.frequencies[self.freq_idx]
        available_shots = self.shots_map.get(freq, [1])
        if self.shot_idx >= len(available_shots):
            self.shot_idx = 0
        shot = available_shots[self.shot_idx]
        return os.path.join(self.base_dir, f"AC_frequency_dependence_ps_{freq}_max{shot:03d}.txt")

    def snap_to_step(self, val):
        step = 0.0005
        return round(round(val / step) * step, 4)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_widget, stretch=4)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(340)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        scroll_area.setWidget(sidebar_content)
        main_layout.addWidget(scroll_area, stretch=1)

        # --- 1. Frequency Selector ---
        freq_group = QGroupBox("Frequency (Hz)")
        freq_layout = QHBoxLayout(freq_group)
        self.btn_prev_freq = QPushButton("<<")
        self.btn_prev_freq.clicked.connect(self.prev_freq)
        self.freq_input = QLineEdit()
        self.freq_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_input.returnPressed.connect(self.on_freq_entry)
        self.btn_next_freq = QPushButton(">>")
        self.btn_next_freq.clicked.connect(self.next_freq)
        freq_layout.addWidget(self.btn_prev_freq)
        freq_layout.addWidget(self.freq_input)
        freq_layout.addWidget(self.btn_next_freq)
        sidebar_layout.addWidget(freq_group)

        # --- 2. Shot Selector ---
        shot_group = QGroupBox("Shot Number (Wraps 29->30->1)")
        shot_layout = QHBoxLayout(shot_group)
        self.btn_prev_shot = QPushButton("<")
        self.btn_prev_shot.clicked.connect(self.prev_shot)
        self.shot_input = QLineEdit()
        self.shot_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.shot_input.returnPressed.connect(self.on_shot_entry)
        self.btn_next_shot = QPushButton(">")
        self.btn_next_shot.clicked.connect(self.next_shot)
        shot_layout.addWidget(self.btn_prev_shot)
        shot_layout.addWidget(self.shot_input)
        shot_layout.addWidget(self.btn_next_shot)
        sidebar_layout.addWidget(shot_group)

        # --- 3. Mask Bounds Controls ---
        mask_group = QGroupBox("Time Mask Bounds (s)")
        mask_layout = QVBoxLayout(mask_group)

        l_layout = QHBoxLayout()
        l_layout.addWidget(QLabel("Lower:"))
        self.lower_input = QLineEdit(str(self.mask_lower))
        self.lower_input.textChanged.connect(self.on_mask_text_changed)
        l_layout.addWidget(self.lower_input)
        btn_l_minus = QPushButton("-")
        btn_l_minus.setFixedWidth(30)
        btn_l_minus.clicked.connect(lambda: self.adjust_mask('lower', -0.0005))
        btn_l_plus = QPushButton("+")
        btn_l_plus.setFixedWidth(30)
        btn_l_plus.clicked.connect(lambda: self.adjust_mask('lower', 0.0005))
        l_layout.addWidget(btn_l_minus)
        l_layout.addWidget(btn_l_plus)
        mask_layout.addLayout(l_layout)

        u_layout = QHBoxLayout()
        u_layout.addWidget(QLabel("Upper:"))
        self.upper_input = QLineEdit(str(self.mask_upper))
        self.upper_input.textChanged.connect(self.on_mask_text_changed)
        u_layout.addWidget(self.upper_input)
        btn_u_minus = QPushButton("-")
        btn_u_minus.setFixedWidth(30)
        btn_u_minus.clicked.connect(lambda: self.adjust_mask('upper', -0.0005))
        btn_u_plus = QPushButton("+")
        btn_u_plus.setFixedWidth(30)
        btn_u_plus.clicked.connect(lambda: self.adjust_mask('upper', 0.0005))
        u_layout.addWidget(btn_u_minus)
        u_layout.addWidget(btn_u_plus)
        mask_layout.addLayout(u_layout)

        sidebar_layout.addWidget(mask_group)

        # --- 4. Advanced Fitting Hyperparameters ---
        adv_group = QGroupBox("Advanced Fitting Hyperparameters")
        adv_layout = QVBoxLayout(adv_group)

        chi2_box = QHBoxLayout()
        chi2_box.addWidget(QLabel("Est Chi2:"))
        self.chi2_input = QLineEdit(f"{self.est_chi2_val:.1e}")
        self.chi2_input.returnPressed.connect(self.on_advanced_text_changed)
        chi2_box.addWidget(self.chi2_input)
        adv_layout.addLayout(chi2_box)
        self.chi2_slider = QSlider(Qt.Orientation.Horizontal)
        self.chi2_slider.setRange(1, 20)
        self.chi2_slider.setValue(6)
        self.chi2_slider.valueChanged.connect(self.on_chi2_slider_changed)
        adv_layout.addWidget(self.chi2_slider)

        minc_box = QHBoxLayout()
        minc_box.addWidget(QLabel("Min C:"))
        self.minc_input = QLineEdit(str(self.min_c_val))
        self.minc_input.returnPressed.connect(self.on_advanced_text_changed)
        minc_box.addWidget(self.minc_input)
        adv_layout.addLayout(minc_box)
        self.minc_slider = QSlider(Qt.Orientation.Horizontal)
        self.minc_slider.setRange(0, 20)
        self.minc_slider.setValue(3)
        self.minc_slider.valueChanged.connect(self.on_minc_slider_changed)
        adv_layout.addWidget(self.minc_slider)

        harm_box = QHBoxLayout()
        harm_box.addWidget(QLabel("Max Harmonic:"))
        self.harm_input = QLineEdit(str(self.max_harmonic_val))
        self.harm_input.returnPressed.connect(self.on_advanced_text_changed)
        harm_box.addWidget(self.harm_input)
        adv_layout.addLayout(harm_box)
        self.harm_slider = QSlider(Qt.Orientation.Horizontal)
        self.harm_slider.setRange(1, 10)
        self.harm_slider.setValue(self.max_harmonic_val)
        self.harm_slider.valueChanged.connect(self.on_harm_slider_changed)
        adv_layout.addWidget(self.harm_slider)

        adv_layout.addWidget(QLabel("<b>Parameter Tolerances:</b>"))
        
        self.tol_inputs = {}
        self.tol_sliders = {}
        tol_grid = QGridLayout()
        
        row = 0
        for param, val in self.tolerances.items():
            tol_grid.addWidget(QLabel(f"{param}:"), row, 0)
            
            t_input = QLineEdit(str(val))
            t_input.setFixedWidth(65)
            t_input.returnPressed.connect(self.on_tolerance_text_changed)
            tol_grid.addWidget(t_input, row, 1)
            self.tol_inputs[param] = t_input

            t_slider = QSlider(Qt.Orientation.Horizontal)
            t_slider.setRange(0, 100)
            t_slider.setValue(int(val * 100))
            t_slider.valueChanged.connect(lambda v, p=param: self.on_tolerance_slider_changed(p, v))
            tol_grid.addWidget(t_slider, row, 2)
            self.tol_sliders[param] = t_slider

            row += 1

        adv_layout.addLayout(tol_grid)
        sidebar_layout.addWidget(adv_group)

        # --- 5. Fit / Reset Button ---
        self.fit_reset_btn = QPushButton("Fit Curve")
        self.fit_reset_btn.setFixedHeight(40)
        self.fit_reset_btn.clicked.connect(self.toggle_fit_reset)
        sidebar_layout.addWidget(self.fit_reset_btn)

        # --- 6. Record Button ---
        self.record_btn = QPushButton("Record Chi2 to CSV")
        self.record_btn.setFixedHeight(40)
        self.record_btn.clicked.connect(self.record_chi2)
        sidebar_layout.addWidget(self.record_btn)

        # --- Info Label ---
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        sidebar_layout.addWidget(self.info_label)

    def load_current_file(self):
        filename = self.get_current_filename()
        freq = self.frequencies[self.freq_idx]
        shots = self.shots_map.get(freq, [1])
        shot = shots[self.shot_idx]

        self.freq_input.setText(str(freq))
        self.shot_input.setText(str(shot))

        if not os.path.exists(filename):
            self.time_array = np.linspace(0, 0.01, 1000)
            self.voltage_array = 0.05 * np.sin(2 * np.pi * 1000 * self.time_array) + 0.1
            self.ac_voltage = 237.5
            self.ac_frequency = float(freq)
            self.piezo_frequency = 40.0
            self.info_label.setText(f"File not found:\n{os.path.basename(filename)}\n(Using dummy preview)")
        else:
            try:
                with open(filename, "r") as f:
                    raw_header = f.readline().strip()
                params = {}
                for item in raw_header.split(","):
                    if "=" in item:
                        k, v = item.strip().split("=", 1)
                        params[k] = v

                self.ac_voltage = float(params.get("AC_VOLTAGE", 237.5))
                self.ac_frequency = float(params.get("AC_FREQUENCY", freq))
                self.piezo_frequency = float(params.get("PIEZO_FREQUENCY", 40.0))

                data = np.loadtxt(filename, skiprows=2, delimiter=",")
                self.time_array = data[:, 0]
                self.voltage_array = data[:, 1]
                self.info_label.setText(f"Loaded:\nFreq: {freq} Hz\nShot: {shot:03d}\nPoints: {len(self.time_array)}")
            except Exception as e:
                QMessageBox.critical(self, "Loading Error", f"Failed to load file: {e}")
                return

        if self.is_fitted:
            self.toggle_fit_reset()
        else:
            self.redraw_plot()

    def redraw_plot(self):
        self.plot_widget.clear()

        if not self.is_fitted:
            # Full-width span when not fitted (spans across both columns row 0 and row 1)
            p = self.plot_widget.addPlot(row=0, col=0, rowspan=2, colspan=2)
            p.setLabel('bottom', 'Time (s)')
            p.setLabel('left', 'Voltage (V)')
            p.setTitle(f"Time Domain Waveform (Freq: {self.frequencies[self.freq_idx]} Hz)")
            p.addLegend()
            
            p.setMouseEnabled(x=False, y=False)

            mask = (self.time_array >= self.mask_lower) & (self.time_array <= self.mask_upper)
            
            t_out = self.time_array.copy()
            v_out = self.voltage_array.copy()
            t_out[mask] = np.nan
            v_out[mask] = np.nan

            t_in = self.time_array.copy()
            v_in = self.voltage_array.copy()
            t_in[~mask] = np.nan
            v_in[~mask] = np.nan

            self.curve_out = p.plot(t_out, v_out, pen=pg.mkPen('#333333', width=5), name="Unmasked Data")
            self.curve_in = p.plot(t_in, v_in, pen=pg.mkPen('#00008B', width=5), name="Masked Data")

            self.region = pg.LinearRegionItem([self.mask_lower, self.mask_upper], brush=(173, 216, 230, 50), movable=True)
            self.region.setZValue(10)
            p.addItem(self.region)
            self.region.sigRegionChanged.connect(self.on_region_changed)

        else:
            # Set 75:25 layout ratio (Column 0 width weight = 3, Column 1 width weight = 1)
            self.plot_widget.ci.layout.setColumnStretchFactor(0, 3)
            self.plot_widget.ci.layout.setColumnStretchFactor(1, 1)

            p1 = self.plot_widget.addPlot(row=0, col=0, rowspan=2)
            p1.setLabel('bottom', 'Time (s)')
            p1.setLabel('left', 'Voltage (V)')
            p1.setTitle(f"Time Domain Fit (Chi2: {self.fit_data['chi2_val']:.4e})")
            p1.addLegend()
            p1.setMouseEnabled(x=False, y=False)

            t_m = self.fit_data["time_masked"]
            v_m = self.fit_data["voltage_masked"]
            v_fit = self.fit_data["voltage_fit"]

            p1.plot(t_m, v_m, pen=pg.mkPen('#00008B', width=5), name="Data (Masked)")
            p1.plot(t_m, v_fit, pen=pg.mkPen('r', width=5), name="Fit Curve")

            # Top Right: FFT Spectrum (compacted)
            p2 = self.plot_widget.addPlot(row=0, col=1)
            p2.setLabel('bottom', 'Freq (kHz)')
            p2.setLabel('left', 'Mag (dB)')
            p2.setTitle("FFT Spectrum")
            p2.setMouseEnabled(x=False, y=False)
            
            freq_arr = self.fit_data["freq_array"]
            fft_spec = self.fit_data["fft_spectrum"]
            ext_freqs = self.fit_data["extracted_freqs"]
            pk_mags = self.fit_data["peak_magnitudes"]
            max_h_khz = (self.fit_data["max_harmonic"] + 1) * self.ac_frequency / 1e3

            p2.setXRange(0, max_h_khz)
            p2.plot(freq_arr / 1e3, 20 * np.log10(fft_spec + 1e-12), pen=pg.mkPen('navy', width=2))
            p2.plot(ext_freqs / 1e3, 20 * np.log10(pk_mags + 1e-12), pen=None, symbol='o', symbolBrush='r', symbolSize=6)

            # Bottom Right: Bessel Fit (compacted)
            p3 = self.plot_widget.addPlot(row=1, col=1)
            p3.setLabel('bottom', 'Harmonic ($n$)')
            p3.setLabel('left', 'Peak Mag')
            C_val = self.fit_data["C"]
            p3.setTitle(f"Bessel ($C={C_val:.4f}$)")
            p3.setMouseEnabled(x=False, y=False)

            h_orders = self.fit_data["harmonic_orders"]
            n_fine = self.fit_data["n_fine"]
            b_curve = self.fit_data["bessel_curve"]

            p3.plot(h_orders, pk_mags, pen=None, symbol='o', symbolBrush='r', symbolSize=6)
            p3.plot(n_fine, b_curve, pen=pg.mkPen('b', style=Qt.PenStyle.DashLine, width=2))

    def on_region_changed(self):
        if self._updating_from_ui:
            return
        self._updating_from_ui = True
        low, high = self.region.getRegion()
        self.mask_lower = self.snap_to_step(low)
        self.mask_upper = self.snap_to_step(high)
        self.lower_input.setText(str(self.mask_lower))
        self.upper_input.setText(str(self.mask_upper))
        
        if not self.is_fitted and hasattr(self, 'curve_in') and hasattr(self, 'curve_out'):
            mask = (self.time_array >= self.mask_lower) & (self.time_array <= self.mask_upper)
            
            t_out = self.time_array.copy()
            v_out = self.voltage_array.copy()
            t_out[mask] = np.nan
            v_out[mask] = np.nan

            t_in = self.time_array.copy()
            v_in = self.voltage_array.copy()
            t_in[~mask] = np.nan
            v_in[~mask] = np.nan

            self.curve_out.setData(t_out, v_out)
            self.curve_in.setData(t_in, v_in)

        self._updating_from_ui = False

    def on_mask_text_changed(self):
        if self._updating_from_ui:
            return
        try:
            l_val = float(self.lower_input.text())
            u_val = float(self.upper_input.text())
            l_val = self.snap_to_step(l_val)
            u_val = self.snap_to_step(u_val)
            if l_val < u_val:
                self.mask_lower = l_val
                self.mask_upper = u_val
                
                if hasattr(self, 'region'):
                    self._updating_from_ui = True
                    self.region.setRegion([self.mask_lower, self.mask_upper])
                    self._updating_from_ui = False

                if self.is_fitted:
                    self.is_fitted = False
                    self.fit_reset_btn.setText("Fit Curve")
                
                if not self.is_fitted and hasattr(self, 'curve_in') and hasattr(self, 'curve_out'):
                    mask = (self.time_array >= self.mask_lower) & (self.time_array <= self.mask_upper)
                    t_out = self.time_array.copy()
                    v_out = self.voltage_array.copy()
                    t_out[mask] = np.nan
                    v_out[mask] = np.nan

                    t_in = self.time_array.copy()
                    v_in = self.voltage_array.copy()
                    t_in[~mask] = np.nan
                    v_in[~mask] = np.nan

                    self.curve_out.setData(t_out, v_out)
                    self.curve_in.setData(t_in, v_in)
        except ValueError:
            pass

    # --- ADVANCED HYPERPARAMETER UI SYNC ---
    def on_chi2_slider_changed(self, val):
        if self._updating_from_ui:
            return
        self._updating_from_ui = True
        self.est_chi2_val = (val / 10.0) * 1e-12
        self.chi2_input.setText(f"{self.est_chi2_val:.1e}")
        self._updating_from_ui = False

    def on_minc_slider_changed(self, val):
        if self._updating_from_ui:
            return
        self._updating_from_ui = True
        self.min_c_val = val / 10.0
        self.minc_input.setText(str(self.min_c_val))
        self._updating_from_ui = False

    def on_harm_slider_changed(self, val):
        if self._updating_from_ui:
            return
        self._updating_from_ui = True
        self.max_harmonic_val = val
        self.harm_input.setText(str(self.max_harmonic_val))
        self._updating_from_ui = False

    def on_tolerance_slider_changed(self, param, val):
        if self._updating_from_ui:
            return
        self._updating_from_ui = True
        tol_val = val / 100.0
        self.tolerances[param] = tol_val
        self.tol_inputs[param].setText(str(tol_val))
        self._updating_from_ui = False

    def on_advanced_text_changed(self):
        if self._updating_from_ui:
            return
        try:
            self._updating_from_ui = True
            self.est_chi2_val = float(self.chi2_input.text())
            self.min_c_val = float(self.minc_input.text())
            self.max_harmonic_val = int(self.harm_input.text())
            
            self.chi2_slider.setValue(int(self.est_chi2_val * 1e12 * 10))
            self.minc_slider.setValue(int(self.min_c_val * 10))
            self.harm_slider.setValue(self.max_harmonic_val)
            self._updating_from_ui = False
        except ValueError:
            self._updating_from_ui = False

    def on_tolerance_text_changed(self):
        if self._updating_from_ui:
            return
        try:
            self._updating_from_ui = True
            for param, field in self.tol_inputs.items():
                val = float(field.text())
                self.tolerances[param] = val
                if param in self.tol_sliders:
                    self.tol_sliders[param].setValue(int(val * 100))
            self._updating_from_ui = False
        except ValueError:
            self._updating_from_ui = False

    # --- NAVIGATION METHODS ---
    def prev_freq(self):
        if self.freq_idx > 0:
            self.freq_idx -= 1
            self.shot_idx = 0
            self.load_current_file()

    def next_freq(self):
        if self.freq_idx < len(self.frequencies) - 1:
            self.freq_idx += 1
            self.shot_idx = 0
            self.load_current_file()

    def on_freq_entry(self):
        try:
            val = int(self.freq_input.text())
            if val in self.frequencies:
                self.freq_idx = self.frequencies.index(val)
                self.shot_idx = 0
                self.load_current_file()
            else:
                QMessageBox.warning(self, "Invalid Frequency", f"Frequency {val} not found in available dataset.")
        except ValueError:
            pass

    def prev_shot(self):
        freq = self.frequencies[self.freq_idx]
        shots = self.shots_map.get(freq, [1])
        self.shot_idx = (self.shot_idx - 1) % len(shots)
        self.load_current_file()

    def next_shot(self):
        freq = self.frequencies[self.freq_idx]
        shots = self.shots_map.get(freq, [1])
        self.shot_idx = (self.shot_idx + 1) % len(shots)
        self.load_current_file()

    def on_shot_entry(self):
        try:
            val = int(self.shot_input.text())
            freq = self.frequencies[self.freq_idx]
            shots = self.shots_map.get(freq, [1])
            if val in shots:
                self.shot_idx = shots.index(val)
                self.load_current_file()
            else:
                QMessageBox.warning(self, "Invalid Shot", f"Shot {val} not found for frequency {freq}.")
        except ValueError:
            pass

    def adjust_mask(self, bound_type, delta):
        if bound_type == 'lower':
            self.mask_lower = self.snap_to_step(self.mask_lower + delta)
            self.lower_input.setText(str(self.mask_lower))
        elif bound_type == 'upper':
            self.mask_upper = self.snap_to_step(self.mask_upper + delta)
            self.upper_input.setText(str(self.mask_upper))
        
        if hasattr(self, 'region'):
            self._updating_from_ui = True
            self.region.setRegion([self.mask_lower, self.mask_upper])
            self._updating_from_ui = False

        if self.is_fitted:
            self.is_fitted = False
            self.fit_reset_btn.setText("Fit Curve")
        
        if not self.is_fitted and hasattr(self, 'curve_in') and hasattr(self, 'curve_out'):
            mask = (self.time_array >= self.mask_lower) & (self.time_array <= self.mask_upper)
            t_out = self.time_array.copy()
            v_out = self.voltage_array.copy()
            t_out[mask] = np.nan
            v_out[mask] = np.nan

            t_in = self.time_array.copy()
            v_in = self.voltage_array.copy()
            t_in[~mask] = np.nan
            v_in[~mask] = np.nan

            self.curve_out.setData(t_out, v_out)
            self.curve_in.setData(t_in, v_in)

    # --- FITTING & RESET ---
    def toggle_fit_reset(self):
        if not self.is_fitted:
            try:
                mask = (self.time_array >= self.mask_lower) & (self.time_array <= self.mask_upper)
                t_masked = self.time_array[mask]
                v_masked = self.voltage_array[mask]

                if len(t_masked) < 10:
                    QMessageBox.critical(self, "Mask Error", "Mask is too narrow or contains insufficient data points.")
                    return

                curve_fitter = CurveFitter(
                    poled_fiber_length=0.3,
                    core_index=1.45,
                    effective_distance=31e-6,
                    field_adjustment_factor=2.0486,
                    periods_per_piezo_cycle=2.5,
                    piezo_frequency=self.piezo_frequency,
                    ac_voltage=self.ac_voltage,
                    ac_frequency=self.ac_frequency,
                    wavelength=1550e-9
                )

                max_harmonic = self.max_harmonic_val
                search_radius_hz = 2 * self.piezo_frequency

                A_guess, B_guess, C, D_guess, E_guess, F_guess, G_guess = curve_fitter.fit_waveform_frequency(
                    time_array=t_masked,
                    voltage_array=v_masked,
                    estimated_chi2=self.est_chi2_val,
                    max_harmonic=max_harmonic,
                    search_radius_hz=search_radius_hz,
                    min_C=self.min_c_val
                )

                def full_time_domain_model(t, A_f, B_f, C_f, D_f, E_f, F_f, G_f):
                    return A_f * (1.0 + np.cos(B_f * (t - E_f) - C_f * np.cos(D_f * (t - F_f)))) + G_f

                total_time_span = t_masked[-1] - t_masked[0]
                tols = self.tolerances

                p0_time = [A_guess, B_guess, C, D_guess, E_guess, F_guess, G_guess]
                lower_b = [
                    A_guess * (1 - tols["A"]),
                    B_guess * (1 - tols["B"]),
                    max(0.0, C * (1 - tols["C"])),
                    D_guess * (1 - tols["D"]),
                    E_guess - total_time_span * tols["E"],
                    F_guess - (1.0 / self.ac_frequency) * tols["F"],
                    G_guess - abs(G_guess) * tols["G"] if G_guess != 0 else -1.0
                ]
                upper_b = [
                    A_guess * (1 + tols["A"]),
                    B_guess * (1 + tols["B"]),
                    C * (1 + tols["C"]),
                    D_guess * (1 + tols["D"]),
                    E_guess + total_time_span * tols["E"],
                    F_guess + (1.0 / self.ac_frequency) * tols["F"],
                    G_guess + abs(G_guess) * tols["G"] if G_guess != 0 else 1.0
                ]

                popt_time, _ = curve_fit(
                    full_time_domain_model, t_masked, v_masked,
                    p0=p0_time, bounds=(lower_b, upper_b), maxfev=100000
                )
                A_f, B_f, C_f, D_f, E_f, F_f, G_f = popt_time
                chi2_final = curve_fitter.get_chi2(C_f) * 1e12
                v_fit = full_time_domain_model(t_masked, *popt_time)

                N = len(t_masked)
                dt = t_masked[1] - t_masked[0]
                window = np.hanning(N)
                fft_spec = np.abs(rfft((v_masked - np.mean(v_masked)) * window)) * (2.0 / N)
                freq_arr = rfftfreq(N, d=dt)

                harmonic_orders = np.arange(1, max_harmonic + 1)
                pk_mags, ext_freqs = [], []
                for n in harmonic_orders:
                    t_freq = n * self.ac_frequency
                    f_mask = (freq_arr >= t_freq - search_radius_hz) & (freq_arr <= t_freq + search_radius_hz)
                    if np.any(f_mask):
                        sub_spec = fft_spec[f_mask]
                        sub_freqs = freq_arr[f_mask]
                        max_idx = np.argmax(sub_spec)
                        ext_freqs.append(sub_freqs[max_idx])
                        pk_val = sub_spec[max_idx]
                        sig_bins = sub_spec[sub_spec > pk_val * 0.1]
                        pk_mags.append(np.sqrt(np.sum(sig_bins**2)))
                    else:
                        ext_freqs.append(t_freq)
                        pk_mags.append(1e-12)

                pk_mags = np.array(pk_mags)
                ext_freqs = np.array(ext_freqs)
                A_scale_plot = np.max(pk_mags) / np.max(np.abs(jv(harmonic_orders, C_f)))
                n_fine = np.linspace(0.5, max_harmonic + 0.5, 200)
                b_curve = A_scale_plot * np.abs(jv(n_fine, C_f))

                self.fit_data = {
                    "time_masked": t_masked,
                    "voltage_masked": v_masked,
                    "voltage_fit": v_fit,
                    "chi2_val": chi2_final,
                    "C": C_f,
                    "freq_array": freq_arr,
                    "fft_spectrum": fft_spec,
                    "extracted_freqs": ext_freqs,
                    "peak_magnitudes": pk_mags,
                    "harmonic_orders": harmonic_orders,
                    "max_harmonic": max_harmonic,
                    "n_fine": n_fine,
                    "bessel_curve": b_curve
                }

                self.is_fitted = True
                self.fit_reset_btn.setText("Reset")
                self.redraw_plot()
            except Exception as e:
                QMessageBox.critical(self, "Fitting Error", f"Curve fitting failed: {e}")
        else:
            self.is_fitted = False
            self.fit_reset_btn.setText("Fit Curve")
            self.redraw_plot()

    # --- RECORD TO CSV ---
    def record_chi2(self):
        if not self.is_fitted:
            QMessageBox.warning(self, "Not Fitted", "Please fit the curve successfully before recording chi2 data.")
            return

        current_freq = self.frequencies[self.freq_idx]
        chi2_val = self.fit_data["chi2_val"]

        if os.path.exists(self.csv_record_path):
            df = pd.read_csv(self.csv_record_path)
        else:
            df = pd.DataFrame(columns=["AC frequency", "Chi2"])

        freq_col = df.columns[0]
        chi2_col = df.columns[1]

        if current_freq in df[freq_col].values:
            df.loc[df[freq_col] == current_freq, chi2_col] = chi2_val
        else:
            new_row = pd.DataFrame({freq_col: [current_freq], chi2_col: [chi2_val]})
            df = pd.concat([df, new_row], ignore_index=True)
            df = df.sort_values(by=freq_col).reset_index(drop=True)

        df.to_csv(self.csv_record_path, index=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FiberGUIAppPyQt()
    window.show()
    sys.exit(app.exec())