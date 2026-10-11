# Dipendenze: numpy, soundfile. Uso: cerca_colpi.py file.wav [secondi_finali]
# Elenca i tratti brevi e isolati (raffiche < 60 ms circondate da quasi silenzio) e il profilo degli ultimi secondi:
# serve a localizzare i "colpi" sentiti a orecchio.
import sys

import numpy as np
import soundfile

audio, rate = soundfile.read(sys.argv[1], dtype="float32")
tail_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
window = int(rate * 0.005)
frames = len(audio) // window
peaks = np.abs(audio[:frames * window]).reshape(frames, window).max(axis=1)
speech_level = np.percentile(peaks[peaks > 0], 90)
loud = peaks > speech_level * 0.05
# raffiche isolate: tratti forti brevi con almeno 60 ms di quasi silenzio prima e dopo
runs, start = [], None
for i, flag in enumerate(np.append(loud, False)):
    if flag and start is None:
        start = i
    elif not flag and start is not None:
        runs.append((start, i))
        start = None
gap = int(0.06 / 0.005)
for (a, b) in runs:
    length_ms = (b - a) * 5
    quiet_before = a >= gap and not loud[a - gap:a].any()
    quiet_after = b + gap <= len(loud) and not loud[b:b + gap].any()
    if length_ms < 60 and quiet_before and quiet_after:
        print(f"raffica isolata a {a * 0.005:.2f} s, {length_ms} ms, picco {peaks[a:b].max() / speech_level:.2f} del parlato")
end_frames = int(tail_seconds / 0.005)
profile = " ".join(f"{p / speech_level:.2f}" for p in peaks[-end_frames::4])
print(f"durata {len(audio) / rate:.2f} s; ultimi {tail_seconds} s (picchi ogni 20 ms, 1 = parlato): {profile}")
