import pyvisa
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum

class HP54600B:
    """A class of methods to acquire data from an oscilloscope through GPIB. Allows for connection checking, and waveform collection.
    Designed for the HP 54600B Oscilloscope

    :param instrument_num: GPIB address of the oscilloscope, as set through the HPIB setting in the Print Utility menu. Default is 1
    :type instrument_num: int. Between 1 and 30, inclusive
    """

    instrument_num: int
    
    def __init__(self, instrument_num: int = 1):
        self.instrument_num = instrument_num

        self.rm = pyvisa.ResourceManager()
        visa_address = f"GPIB0::{self.instrument_num}::INSTR"
        
        self.instrument = self.rm.open_resource(visa_address)
        self.instrument.timeout = 5000 
        self.instrument.read_termination = '\n'
        self.write_termination = '\n'
        
        self.instrument.write("*CLS")

    def init_instrument(self) -> bool:
        """Checks for proper connection to oscilloscope

        :return: True if the scope is connected. False if not
        :rtype: bool
        """
        
        try:      
            self.instrument.write("*CLS")
            
            self.instrument.query("*IDN?") # Test query
            
            return True
            
        except pyvisa.errors.VisaIOError as e:
            print(f"\nNo connection between python and the oscilloscope: {e}")
            return False

    def acquire_signal(self, *, channel: int = 2) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Captures a waveform from the specified channel and returns the time and voltage arrays
        
        :param channel: Channel number to read. Default is 2
        :type channel: int
        :return: An array of times and a corresponding array of voltages
        :rtype: tuple[NDArray[np.float64], NDArray[np.float64]]
        """
        
        try:
            self.instrument.write("*CLS")

            self.instrument.write(":TIMEBASE:MODE NORMAL")

            self.instrument.write(":ACQUIRE:TYPE NORMAL")
            self.instrument.write(":ACQUIRE:COMPLETE 100")

            # configure data transfer to get bytes from selected channel
            self.instrument.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
            self.instrument.write(":WAVEFORM:FORMAT BYTE")
            
            # get preamble parameters
            preamble = self.instrument.query(":WAVEFORM:PREAMBLE?").split(',')

            num_points = float(preamble[2])

            x_increment = float(preamble[4])
            x_origin = float(preamble[5])
            x_reference = float(preamble[6])
            
            y_increment = float(preamble[7])
            y_origin = float(preamble[8])
            y_reference = float(preamble[9])

            # get raw data
            raw_voltage = self.instrument.query_binary_values(":WAVEFORM:DATA?", datatype='B', container=np.array)
            raw_time = np.arange(num_points)

            # convert to time and voltage arrays
            time = ((raw_time - x_reference) * x_increment) + x_origin
            voltage = ((raw_voltage - y_reference) * y_increment) + y_origin
            
            return (time, voltage)

        except pyvisa.errors.VisaIOError as e:
            print(f"Visa IO Error: {e}")
            return None
        except Exception as e:
            print(f"\nSystem Parsing Error: {e}")
            return None

    def get_amplitude(self, *, channel: int = 1) -> float:
        """Gets the amplitude of a waveform from the selected oscilloscope channel

        :param channel: The channel from which the waveform is inputted. Default is 1
        :type channel: int
        :return: The amplitude of the waveform
        :rtype: float
        """

        try:
            self.instrument.write("*CLS")

            self.instrument.write(f":MEASURE:SOURCE CHANNEL{channel}")
            vpp = float(self.instrument.query(":MEASURE:VPP?"))

            return vpp

        except Exception as e:
            print(f"Failed Measurement: {e}")
            return None

    def set_screen(self, *, channel: int,
                   volts_per_div: float, time_per_div: float,
                   vertical_offset: float = 0, horizontal_offset: float = 0,
                   trigger_level: float, ext_trigger: bool = False) -> None:
        """Sets screen to the appropriate settings for proper data acquisition

        :param channel: The channel from which the waveform is inputted. Default is 1
        :type channel: int
        :param volts_per_div: The volts per div setting on the oscilloscope screen
        :type volts_per_div: float
        :param time_per_div: The time per div setting on the oscilloscope screen
        :type time_per_div: float
        :param vertical_offset: The vertical offset setting on the oscilloscope screen. Default is 0.
        :type vertical_offset: float
        :param horizontal_offset: The horizontal offset setting on the oscilloscope screen. Default is 0.
        :type horizontal_offset: float
        :param trigger_level: The trigger level of either the internal or external trigger, based on the ext_trigger argument
        :type trigger_level: float
        :param ext_trigger: Determines whether the external trigger will be used (True) or not (False). Default is False
        :type ext_trigger: bool

        :return: None
        """

        vertical_range = volts_per_div * 8
        horizontal_range = time_per_div * 10

        try:
            self.instrument.write("*CLS")

            if channel == 1:
                self.instrument.write(":BLANK CHANNEL2")
                self.instrument.write(":VIEW CHANNEL1")
            else:
                self.instrument.write(":BLANK CHANNEL1")
                self.instrument.write(":VIEW CHANNEL2")

            # vertical settings
            self.instrument.write(f":CHANNEL{channel}:RANGE {vertical_range:.4f}")
            self.instrument.write(f":CHANNEL{channel}:OFFSET {vertical_offset:.4f}")
            self.instrument.write(f":CHANNEL{channel}:COUPLING DC")

            # horizontal settings
            self.instrument.write(f":TIMEBASE:RANGE {horizontal_range:.4f}")
            self.instrument.write(f":TIMEBASE:DELAY {horizontal_offset:.4f}")

            if ext_trigger == False:
                self.instrument.write(f":TRIGGER:SOURCE CHANNEL{channel}")
            else:
                self.instrument.write(":TRIGGER:SOURCE EXTERNAL")
                self.instrument.write(":TRIGGER:SLOPE NEGATIVE")

            self.instrument.write(":TRIGGER:MODE NORMAL")
            self.instrument.write(f":TRIGGER:LEVEL {trigger_level:.4f}")


            # tm.sleep(0.5) # delay for change time

        except Exception as e:
            print(f"Hardware Error {e}")

    def reset(self) -> None:
        """Resets the oscilloscope settings to default for consistency"""

        self.instrument.write("*RST")

    def close(self) -> None:
        """Clean up the bus link to the oscilloscope and resource manager"""

        try:
            self.instrument.close()
        finally:
            self.rm.close()

class WaveformShape(Enum):
    """Enum for all the waveform types for AWGs.
    
    Note that for the 33250A, RAMP refers to the standard triangle wave and its
    symmetry can be adjusted to create a sawtooth. TRIANGLE is not an option on the
    33250A. For the 33120, TRIANGLE refers to the standard triangle wave, while RAMP
    references to a sawtooth wave. No further symmetry adjustment can be made.
    """
    
    SINUSOID = 0
    SQUARE = 1
    RAMP = 2
    TRIANGLE = 3
    PULSE = 4
    NOISE = 5
    DC = 6
    USER = 7

class AWG(ABC):
    """Abstract Base Class to control AWG instruments via GPIB.

    :param instrument_num: GPIB address of the AWG (1 to 30).
    :type instrument_num: int
    :param default_shape: Default function shape for reset.
    :type default_shape: Optional[WaveformShape]
    :param default_frequency: Default frequency in Hz for reset.
    :type default_frequency: Optional[float]
    :param default_voltage: Default amplitude in Vpp for reset.
    :type default_voltage: Optional[float]
    :param default_symmetry: Default ramp symmetry percentage (0.0 to 100.0) for reset.
    :type default_symmetry: Optional[float]
    """

    def __init__(
        self,
        instrument_num: int = 1,
        default_shape: Optional[WaveformShape] = None,
        default_frequency: Optional[float] = None,
        default_voltage: Optional[float] = None,
        default_symmetry: Optional[float] = None,
    ):
        self.instrument_num = instrument_num
        self.default_shape = default_shape
        self.default_frequency = default_frequency
        self.default_voltage = default_voltage
        self.default_symmetry = default_symmetry

        self.rm = pyvisa.ResourceManager()
        visa_address = f"GPIB0::{self.instrument_num}::INSTR"

        self.instrument = self.rm.open_resource(visa_address)
        self.instrument.timeout = 5000
        self.instrument.read_termination = '\n'
        self.write_termination = '\n'

    @abstractmethod
    def change_frequency(self, frequency: float) -> None:
        """Change the frequency of the AWG output (Hz)
        
        :param frequency: The new frequency outputted by the AWG
        :type frequency: float
        """
        pass

    @abstractmethod
    def change_voltage(self, voltage: float) -> None:
        """Change the voltage of the AWG output (V)
        
        :param voltage: The new voltage outputted by the AWG
        :type voltage: float
        """
        pass

    @abstractmethod
    def change_shape(self, shape: WaveformShape) -> None:
        """Change the shape of the AWG outputted
        
        :param shape: The shape of the waveform outputted by the AWG
        :type shape: WaveformShape
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the instrument parameters to the stored defaults if they were provided."""
        pass


class Agilent33250A(AWG):
    """Class to control an Agilent 33250A AWG over GPIB."""

    def change_frequency(self, frequency: float) -> None:
        """Change the frequency of the AWG output (Hz)
        
        :param frequency: The new frequency outputted by the AWG
        :type frequency: float
        """
        
        self.instrument.write(f"FREQ {frequency}")

    def change_voltage(self, amplitude: float) -> None:
        """Change the voltage of the AWG output (V)
        
        :param voltage: The new voltage outputted by the AWG
        :type voltage: float
        """
        
        self.instrument.write(f"VOLT {amplitude}")

    def change_shape(self, shape: WaveformShape) -> None:
        """Sets waveform function shape
        
        :param shape: The shape of the waveform
        :type shape: WaveformShape
        """
        
        if shape == WaveformShape.SINUSOID:
            self.instrument.write(f"FUNCTION SINUSOID")
        elif shape == WaveformShape.RAMP:
            self.instrument.write(f"FUNCTION RAMP")
            
    def change_ramp_symmetry(self, symmetry: float) -> None:
        """Change the symmetry of ramp
        
        :param symmetry: The symmetry of the ramp. The percent of the period
        where the voltage increases
        :type symmetry: WaveformShape
        """
        
        self.instrument.write(f"FUNCTION:RAMP:SYMMETRY {symmetry}")
    
    def reset(self) -> None:
        """Resets the instrument parameters to the stored defaults if they were provided."""
        
        if self.default_shape is not None:
            self.change_shape(self.default_shape)

        if self.default_frequency is not None:
            self.change_frequency(self.default_frequency)

        if self.default_voltage is not None:
            self.change_voltage(self.default_voltage)

        if self.default_symmetry is not None:
            self.change_ramp_symmetry(self.default_symmetry)
    
    def output_on(self) -> None:
        """Enables the front-panel Output terminal relay."""
        self.instrument.write("OUTPUT ON")

    def output_off(self) -> None:
        """Disables the front-panel Output terminal relay."""
        self.instrument.write("OUTPUT OFF")


class Agilent33120A(AWG):
    """Class to control an Agilent 33120A AWG over GPIB."""

    def change_frequency(self, frequency: float) -> None:
        """Sets output frequency in Hz."""
        self.instrument.write(f"FREQ {frequency}")

    def change_voltage(self, amplitude: float) -> None:
        """Sets peak-to-peak output amplitude in Vpp."""
        self.instrument.write(f"VOLT {amplitude}")

    def change_shape(self, shape: WaveformShape) -> None:
        """Sets waveform function shape
        
        :param shape: The shape of the waveform
        :type shape: WaveformShape
        """
        
        if shape == WaveformShape.SINUSOID:
            self.instrument.write(f"FUNCTION:SHAPE SINUSOID")
        elif shape == WaveformShape.RAMP:
            self.instrument.write(f"FUNCTION:SHAPE RAMP")
            
    def reset(self) -> None:
        """Resets the instrument parameters to the stored defaults if they were provided."""
        
        if self.default_shape is not None:
            self.change_shape(self.default_shape)

        if self.default_frequency is not None:
            self.change_frequency(self.default_frequency)

        if self.default_voltage is not None:
            self.change_voltage(self.default_voltage)