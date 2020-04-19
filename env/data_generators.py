import numpy as np
from tgym.core import DataGenerator
import random


class DoubleSineSignal(DataGenerator):
    """ Two Sine Superimposed Generator
    """
    @staticmethod
    def _generator(period_1, period_2, amplitude_1, amplitude_2):
        i = 0
        while True:
            i += 1
            noise = random.uniform(0.90, 1.10)
            price = ( amplitude_1*np.sin(2 * i * np.pi / period_1) + \
                amplitude_2 * np.sin(2 * i * np.pi / period_2)) * noise
            yield price 

class TripleSineSignal(DataGenerator):
    """Three Sine Superimposed Generator
    """
    @staticmethod
    def _generator(period_1, period_2, period_3, amplitude_1, amplitude_2, amplitude_3):
        i = 0
        while True:
            i += 1
            noise = random.uniform(0.90, 1.10)
            price = ( amplitude_1*np.sin(2 * i * np.pi / period_1) + \
                amplitude_2 * np.sin(2 * i * np.pi / period_2) + amplitude_3 * np.sin(2 * i * np.pi / period_3)) * noise
            yield price 

class SineSignal(DataGenerator):
    """Sine generator
    """
    @staticmethod
    def _generator(period_1, amplitude_1):
        i = 0
        while True:
            i += 1
           #noise = random.uniform(0.90, 1.10)
            price = amplitude_1*np.sin(2 * i * np.pi / period_1) #* noise
            yield price 

