import pyvisa
import numpy as np
from numpy.typing import NDArray

class Interface:
    """A class of methods to acquire data from an oscilloscope through GPIB. Allows for connection checking, and waveform collection.
    Designed for the HP 54600B Oscilloscope

    :param instrument_num: GPIB address of the oscilloscope, as set through the HPIB setting in the Print Utility menu. Default is 1
    :type instrument_num: int. Between 1 and 30, inclusive
    """

    instrument_num: int

    def __init__(self, instrument_num: int = 1):
        self.instrument_num = instrument_num

    def init_instrument(instrument_num: int) -> bool:
        """Checks for proper connection to oscilloscope

        :param instrument_num: The instrument number of the oscilloscope, set through the scope
        :type instrument num: int, between 1 and 30 inclusive
        :return: True if the scope is connected. False if not
        :rtype: bool
        """
        rm = pyvisa.ResourceManager()
        visa_address = "GPIB0::" + str(instrument_num) + "::INSTR"
        
        try:
            scope = rm.open_resource(visa_address)
            
            scope.timeout = 3000 # 3 second timeout limit
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

    def acquire_signal(self, channel: int = 2) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
