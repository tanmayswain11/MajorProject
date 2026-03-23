from filtering import normalize_signal
from radar_signal.fmcw_radar import generate_signal

sig = generate_signal()
sig = normalize_signal(sig)

print("Processed OK")