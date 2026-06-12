from classes.interface import Interface
import time

"""Script for testing communication between the computer and oscilloscope,
    as well as the screen-setting capability of the code.
    
    Switches back and forth between two oscilloscope screen settings, as can be seen
    in the code below. If the screen periodically switches between those two settings,
    the oscilloscope and computer are properly communicating and the code is working.
"""

interface = Interface(instrument_num=1)
interface.reset()
for i in range(3):
    interface.set_screen(channel=2,
                         volts_per_div=2,
                         time_per_div=2e-3,
                         vertical_offset=-1,
                         horizontal_offset=-1e-3,
                         trigger_level=1,
                         ext_trigger=False)
    time.sleep(2)
    interface.set_screen(channel=2,
                         volts_per_div=2,
                         time_per_div=2e-3,
                         vertical_offset=0,
                         horizontal_offset=0,
                         trigger_level=2,
                         ext_trigger=False)
    time.sleep(2)

interface.close()