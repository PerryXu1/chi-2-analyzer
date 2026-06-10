from interface import Interface
import time

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
                         ext_trigger=True)
    time.sleep(2)

interface.close()