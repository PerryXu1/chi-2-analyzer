import time
import pyvisa
import numpy as np

ELLIPTICITY = 0.5
AZIMUTHAL = 45.0
SHOTS = 10

CHANNEL = 2
VISA_ADDRESS = "GPIB0::7::INSTR"  

WAVELENGTH = 1550e-9
POLED_LENGTH = 0.3
EFFECTIVE_DISTANCE = 41.1e-6
CORE_INDEX = 1.52
AC_VOLTAGE = 240
ELECTRODE_MATERIAL = "IRON"
TIME = 50
POLING_VOLTAGE = 4500
PIEZO_FREQUENCY = 70
AC_FREQUENCY = 3000
CHI2 = 0.1

rm = pyvisa.ResourceManager()

try:
    scope = rm.open_resource(VISA_ADDRESS)
    
    scope.read_termination = '\n'
    scope.write_termination = '\n'
    
    scope.write(f":WAVEFORM:SOURCE CHANNEL{CHANNEL}")
    scope.write(":WAVEFORM:FORMAT BYTE")
    time.sleep(0.1) 


    for i in range(SHOTS):
        filename = f"waveform_polarimeter_{ELLIPTICITY}_{AZIMUTHAL}_{i + 1:02d}.txt"

        preamble_raw = scope.query(":WAVEFORM:PREAMBLE?")
        preamble = preamble_raw.strip().split(',')

        num_points = float(preamble[2])
        x_increment = float(preamble[4])
        x_origin = float(preamble[5])
        x_reference = float(preamble[6])
        
        y_increment = float(preamble[7])
        y_origin = float(preamble[8])
        y_reference = float(preamble[9])

        time.sleep(0.1)

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

        header_info = (
            f"ELECTRODE={ELECTRODE_MATERIAL}, TIME={TIME}, POLING_VOLTAGE={POLING_VOLTAGE}, "
            f"WAVELENGTH={WAVELENGTH}, EFFECTIVE_DISTANCE={EFFECTIVE_DISTANCE}, "
            f"AC_VOLTAGE={AC_VOLTAGE}, POLED_LENGTH={POLED_LENGTH}, CORE_INDEX={CORE_INDEX}, CHI2={CHI2}"
            f"ELLIPTICITY={ELLIPTICITY}, AZIMUTHAL={AZIMUTHAL}, PIEZO_FREQUENCY={PIEZO_FREQUENCY}, AC_FREQUENCY={AC_FREQUENCY}\n"
            f"Time(s),Voltage(V)"
        )
        
        data_matrix = np.column_stack((time_array, voltage_array))
        np.savetxt(filename, data_matrix, delimiter=",", header=header_info, comments="")
        
        time.sleep(0.1)

    scope.close()

except pyvisa.errors.VisaIOError as e:
    print(f"\nVisa IO Error: {e}")
except Exception as e:
    print(f"\nSystem Error: {e}")