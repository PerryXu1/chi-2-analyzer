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

    def init_instrument(self) -> bool:
        """Checks for proper connection to oscilloscope

        :return: True if the scope is connected. False if not
        :rtype: bool
        """
        rm = pyvisa.ResourceManager()
        visa_address = f"GPIB0::{self.instrument_num}::INSTR"
        
        try:
            scope = rm.open_resource(visa_address)
            
            scope.timeout = 5000 # 5 second timeout limit
            scope.read_termination = '\n'
            scope.write_termination = '\n'
            
            scope.query("*IDN?") # Test query
            
            scope.close()
            return True
            
        except pyvisa.errors.VisaIOError as e:
            print(f"\nNo connection between python and the oscilloscope: {e}")
            return False
        finally:
            rm.close()

    def acquire_signal(self, *, channel: int = 2) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Captures a waveform from the specified channel and returns scaled voltages as a NumPy ndarray.
        
        :param channel: Channel number to read. Default is 2
        :type channel: int
        :return: An array of times and a corresponding array of voltages
        :rtype: tuple[NDArray[np.float64], NDArray[np.float64]]
        """

        rm = pyvisa.ResourceManager()
        visa_address = f"GPIB0::{self.instrument_num}::INSTR"
        
        try:
            scope = rm.open_resource(visa_address)
            scope.timeout = 5000  # 5 second timeout window
            scope.read_termination = '\n'
            scope.write_termination = '\n'
            
            # DIGITIZE to freeze display buffer
            scope.write(f":DIGITIZE CHANNEL{channel}")
            scope.write(f":DIGITIZE CHANNEL")
            
            # configure data transfer to get bytes from selected channel
            scope.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
            scope.write(":WAVEFORM:FORMAT BYTE")
            
            # get preamble parameters
            preamble = scope.query(":WAVEFORM:PREAMBLE?").split(',')

            num_points = float(preamble[2])

            x_increment = float(preamble[4])
            x_origin = float(preamble[5])
            x_reference = float(preamble[6])
            
            y_increment = float(preamble[7])
            y_origin = float(preamble[8])
            y_reference = float(preamble[9])

            # get raw data
            scope.write(":WAVEFORM:DATA?")
            raw_voltage = scope.read_binary_values(datatype='B', container=np.array)
            raw_time = np.arange(num_points)

            # convert to time and voltage arrays
            time = ((raw_time - x_reference) * x_increment) + x_origin
            voltage = ((raw_voltage - y_reference) * y_increment) + y_origin
            
            scope.close()
            return (time, voltage)

        except pyvisa.errors.VisaIOError as e:
            print(f"\nHardware IO Error: {e}")
            return None
        except Exception as e:
            print(f"\nSystem Parsing Error: {e}")
            return None
        finally:
            rm.close()

    def get_amplitude(self, *, channel: int = 1) -> float:
        """Gets the amplitude of a waveform from the selected oscilloscope channel

        :param channel: The channel from which the waveform is inputted. Default is 1
        :type channel: int
        :return: The amplitude of the waveform
        :rtype: float
        """

        rm = pyvisa.ResourceManager()
        visa_address = f"GPIB0::{self.instrument_num}::INSTR"

        try:
            scope = rm.open_resource(visa_address)
            scope.timeout = 5000 # 5 second timeout limit
            scope.read_termination = '\n'
            scope.write_termination = '\n'

            scope.write(f":MEASURE:SOURCE CHANNEL{channel}")
            vpp = float(scope.query(":MEASURE:VPP?"))

            scope.close()
            return vpp

        except Exception as e:
            print(f"Failed Measurement: {e}")
            return None
        finally:
            rm.close()

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

        rm = pyvisa.ResourceManager()
        visa_address = f"GPIB0::{self.instrument_num}::INSTR"

        try:
            scope = rm.open_resource(visa_address)
            scope.timeout = 5000 # 5 second timeout limit
            scope.read_termination = '\n'
            scope.write_termination = '\n'

            scope.write(f":CHANNEL{channel}:DISPLAY ON")

            # vertical settings
            scope.write(f":CHANNEL{channel}:RANGE {vertical_range}")
            scope.write(f":CHANNEL{channel}:OFFSET {vertical_offset}")
            scope.write(f":CHANNEL{channel}:COUPLING DC")

            # horizontal settings
            scope.write(f":TIMEBASE:RANGE {horizontal_range}")
            scope.write(f":TIMEBASE:DELAY {horizontal_offset}")

            if ext_trigger == False:
                scope.write(f":TRIGGER:SOURCE CHANNEL{channel}")
                scope.write(f":TRIGGER:LEVEL {trigger_level}")
            else:
                scope.write(":TRIGGER:SOURCE EXTERNAL")
                scope.write(":TRIGGER:EXTERNAL:RANGE 1.0")
                scope.write(f":TRIGGER:LEVEL {trigger_level}")
                scope.write(":TRIGGER:SLOPE POSITIVE")

            time.sleep(0.1) # delay for change time
            scope.close()

        except Exception as e:
            print(f"❌ Failed to assign external trigger criteria: {e}")
        finally:
            rm.close()