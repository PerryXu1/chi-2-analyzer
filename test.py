from classes.interface import HP54600B, Agilent33120A, Agilent33250A, WaveformShape

awg1 = Agilent33250A(instrument_num=12,
                     default_frequency=10,
                     default_voltage=3.3,
                     default_shape=WaveformShape.RAMP,
                     default_symmetry=100)

target_frequency = 120.0
awg1.change_frequency(target_frequency)
    