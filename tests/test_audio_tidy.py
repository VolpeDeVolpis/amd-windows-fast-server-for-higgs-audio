# Dipendenze: audio_tidy.py, numpy. Prove su segnali sintetici (nessun file vero, nessuna GPU).
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audio_tidy import drop_trailing_fragment  # noqa: E402

RATE = 24000


def tone(seconds, amplitude, frequency=180.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def speech():
    return np.concatenate([tone(0.6, 0.8), silence(0.05), tone(0.5, 0.7), tone(0.4, 0.3)])


def test_weak_burst_after_dip_is_removed():
    clean = speech()
    with_burst = np.concatenate([clean, silence(0.06), tone(0.12, 0.2, 400.0)])
    result = drop_trailing_fragment(with_burst, RATE)
    assert len(result) <= len(clean) + int(0.01 * RATE)
    assert len(result) >= len(clean) - int(0.01 * RATE)


def test_real_final_syllable_is_kept():
    audio = np.concatenate([speech(), silence(0.06), tone(0.25, 0.75)])  # sillaba finale forte
    assert len(drop_trailing_fragment(audio, RATE)) == len(audio)


def test_long_weak_tail_is_kept():
    audio = np.concatenate([speech(), silence(0.06), tone(0.6, 0.2)])  # troppo lungo per essere la raffica
    assert len(drop_trailing_fragment(audio, RATE)) == len(audio)


def test_no_dip_nothing_removed_and_idempotent():
    audio = speech()
    once = drop_trailing_fragment(audio, RATE)
    assert len(once) == len(audio)
    burst = np.concatenate([audio, silence(0.06), tone(0.1, 0.2, 400.0)])
    first = drop_trailing_fragment(burst, RATE)
    assert len(drop_trailing_fragment(first, RATE)) == len(first)
