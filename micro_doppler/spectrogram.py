import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_spectrogram(input_signal, output_path):
    f, t, Sxx = spectrogram(input_signal, fs=2000)

    plt.pcolormesh(t, f, 10*np.log10(Sxx))
    plt.savefig(output_path)
    plt.close()