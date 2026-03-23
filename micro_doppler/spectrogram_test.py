from spectrogram import generate_spectrogram
from radar_signal.fmcw_radar import generate_signal

sig = generate_signal()
generate_spectrogram(sig, "test.png")