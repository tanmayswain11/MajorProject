from radar_signal.fmcw_radar import generate_signal
from signal_processing.filtering import normalize_signal
from micro_doppler.spectrogram import generate_spectrogram
from model.cnn.predict import predict_image

sig = generate_signal()
sig = normalize_signal(sig)

generate_spectrogram(sig, "output.png")

result = predict_image("output.png")

print("Prediction:", result)