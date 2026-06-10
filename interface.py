import pyvisa
import numpy as np
from numpy.typing import NDArray
import time

class Interface:
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
        
        self.scope = self.rm.open_resource(visa_address)
        self.scope.timeout = 5000 
        self.scope.read_termination = '\n'
        self.write_termination = '\n'
        
        self.scope.write("*CLS")

    def init_instrument(self) -> bool:
        """Checks for proper connection to oscilloscope

        :return: True if the scope is connected. False if not
        :rtype: bool
        """
        
        try:      
            self.scope.write("*CLS")
            
            self.scope.query("*IDN?") # Test query
            
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
            self.scope.write("*CLS")

            self.scope.write(":ACQUIRE:TYPE NORMAL")
            self.scope.write(":ACQUIRE:COMPLETE 100")

            # configure data transfer to get bytes from selected channel
            self.scope.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
            self.scope.write(":WAVEFORM:FORMAT BYTE")
            
            # DIGITIZE to freeze display buffer
            self.scope.write(f":DIGITIZE CHANNEL{channel}")
            
            print("A")
            # get preamble parameters
            preamble = self.scope.query(":WAVEFORM:PREAMBLE?").split(',')

            print("B")
            num_points = float(preamble[2])

            x_increment = float(preamble[4])
            x_origin = float(preamble[5])
            x_reference = float(preamble[6])
            
            y_increment = float(preamble[7])
            y_origin = float(preamble[8])
            y_reference = float(preamble[9])

            print("C")

            # get raw data
            # raw_voltage = self.scope.query(":WAVEFORM:DATA?")
            raw_voltage = self.scope.query_binary_values(":WAVEFORM:DATA?", datatype='B', container=np.array)
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
            self.scope.write("*CLS")

            self.scope.write(f":MEASURE:SOURCE CHANNEL{channel}")
            vpp = float(self.scope.query(":MEASURE:VPP?"))
            print(vpp)

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
            self.scope.write("*CLS")
            # self.scope.write("*RST")

            if channel == 1:
                self.scope.write(":BLANK CHANNEL2")
                self.scope.write(":VIEW CHANNEL1")
            else:
                self.scope.write(":BLANK CHANNEL1")
                self.scope.write(":VIEW CHANNEL2")

            # vertical settings
            self.scope.write(f":CHANNEL{channel}:RANGE {vertical_range:.4f}")
            self.scope.write(f":CHANNEL{channel}:OFFSET {vertical_offset:.4f}")
            self.scope.write(f":CHANNEL{channel}:COUPLING DC")

            # horizontal settings
            self.scope.write(f":TIMEBASE:RANGE {horizontal_range:.4f}")
            self.scope.write(f":TIMEBASE:DELAY {horizontal_offset:.4f}")

            if ext_trigger == False:
                self.scope.write(f":TRIGGER:SOURCE CHANNEL{channel}")
            else:
                self.scope.write(":TRIGGER:SOURCE EXTERNAL")
                self.scope.write(":TRIGGER:SLOPE POSITIVE")

            self.scope.write(":TRIGGER:MODE NORMAL")
            self.scope.write(f":TRIGGER:LEVEL {trigger_level:.4f}")


            time.sleep(1) # delay for change time

        except Exception as e:
            print(f"Hardware Error {e}")

    def reset(self) -> None:
        """Resets the oscilloscope settings to default for consistency"""

        self.scope.write("*RST")

    def close(self) -> None:
        """Clean up the bus link to the oscilloscope and resource manager"""

        try:
            self.scope.close()
        finally:
            self.rm.close()