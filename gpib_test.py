import pyvisa

# Initialize the Resource Manager
rm = pyvisa.ResourceManager()

# List available resources to find the exact address if unsure
print(rm.list_resources())

# Open connection to the TDS460A
# (e.g., 'GPIB0::1::INSTR' for NI hardware or 'ASRL1::INSTR' if using Prologix)
scope = rm.open_resource('GPIB0::1::INSTR')

# Send the standard Identification query
response = scope.query('*IDN?')
print("Instrument ID:", response)

# Reset scope to factory defaults and close connection
scope.write('*RST')
scope.close()
