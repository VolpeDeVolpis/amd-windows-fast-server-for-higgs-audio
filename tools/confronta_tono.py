# Dipendenze: soundfile, torchaudio (solo funzioni di calcolo). Tono medio (Hz) a finestre, per due audio.
import sys

import soundfile
import torch
import torchaudio


def pitch_track(path, seconds, window):
    data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))[: int(seconds * rate)]
    values = []
    for start in range(0, wav.numel() - int(window * rate) + 1, int(window * rate)):
        piece = wav[start:start + int(window * rate)]
        loud = piece.abs().mean().item()
        pitch = torchaudio.functional.detect_pitch_frequency(piece.view(1, -1), rate, freq_low=70, freq_high=400)
        values.append((start / rate, pitch.median().item(), loud))
    return values


for path in sys.argv[1:]:
    print(path)
    print("  " + "  ".join(f"{t:>4.0f}s:{p:>4.0f}Hz" for t, p, _ in pitch_track(path, 26, 2.0)))
