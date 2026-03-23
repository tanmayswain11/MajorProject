import numpy as np

def normalize_signal(signal):
    return signal / np.max(np.abs(signal))