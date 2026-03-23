import numpy as np

def generate_signal():
    t = np.linspace(0, 1, 2000)
    signal = np.sin(2*np.pi*50*t)
    return signal