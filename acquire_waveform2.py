import time
import pyvisa
import numpy as np

CHANNEL = 2
VISA_ADDRESS = "GPIB0::7::INSTR"  
FILENAME = "waveform.txt"

WAVELENGTH = 1550e-9
POLING_LENGTH = 0.3
EFFECTIVE_DISTANCE = 41.1e-6
CORE_INDEX = 1.52
AC_VOLTAGE = 240  

rm = pyvisa.ResourceManager()

try:
    scope = rm.open_resource(VISA_ADDRESS)
    
    # Standard connection parameters
    scope.read_termination = '\n'
    scope.write_termination = '\n'
    
    # 1. Point the waveform processor to the channel you want to read
    scope.write(f":WAVEFORM:SOURCE CHANNEL{CHANNEL}")
    scope.write(":WAVEFORM:FORMAT BYTE")
    time.sleep(0.1) 
    
    # 2. Get the preamble for the frozen screen data
    preamble_raw = scope.query(":WAVEFORM:PREAMBLE?")
    preamble = preamble_raw.strip().split(',')

    num_points = float(preamble[2])
    x_increment = float(preamble[4])
    x_origin = float(preamble[5])
    x_reference = float(preamble[6])
    
    y_increment = float(preamble[7])
    y_origin = float(preamble[8])
    y_reference = float(preamble[9])

    time.sleep(0.2)

    raw_voltage = scope.query_binary_values(
        ":WAVEFORM:DATA?", 
        datatype='B', 
        container=np.array,
        header_fmt='hp',
        expect_termination=False
    )
    
    raw_time = np.arange(num_points)

    time_array = ((raw_time - x_reference) * x_increment) + x_origin
    voltage_array = ((raw_voltage - y_reference) * y_increment) + y_origin
    
    scope.close()

    header_info = (
        f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
        f"AC_VOLTAGE={AC_VOLTAGE}, POLING_LENGTH={POLING_LENGTH}, CORE_INDEX={CORE_INDEX}\n"
        f"Time(s),Voltage(V)"
    )
    data_matrix = np.column_stack((time_array, voltage_array))
    np.savetxt(FILENAME, data_matrix, delimiter=",", header=header_info, comments="")
    print(f"Successfully captured static screen data to {FILENAME}")

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")