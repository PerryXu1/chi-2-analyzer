from simulator import Simulator
from analyzer import Analyzer
import numpy as np
from interface import Interface

SHOTS = 20
V_AC_CALIBRATION_PASSES = 5 # Recalculate the V_AC every V_AC_CALIBRATION_PASSES shots

analyzer = Analyzer()


scope = Interface(instrument_num=1)
scope.reset()

V_ac = None

for i in range(SHOTS):
    if i % V_AC_CALIBRATION_PASSES == 0:
        scope.set_screen(channel=1,
                        volts_per_div=100e-3,
                        time_per_div=1e-3,
                        vertical_offset=0,
                        horizontal_offset=0,
                        trigger_level=0,
                        ext_trigger=False)
        V_ac = 1000 * scope.get_amplitude(channel=1)

    
    
