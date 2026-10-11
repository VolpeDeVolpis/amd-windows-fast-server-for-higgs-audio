# Misure che si adattano al file (nessuna soglia fissa): livelli a finestre, soglia voce/fondo (metodo di Otsu,
# idea presa da un programma di pulizia delle voci precedente), filtro passa-alto (idea dello stesso programma).
# Dipendenze: audio.wav_io. Usato da: audio/clean.
from __future__ import annotations

import math
from array import array
from typing import List, Tuple

from .wav_io import Audio, clamp16

WINDOW_S = 0.02  # lente di misura da 20 ms, non una soglia


def window_levels_db(audio: Audio, window_s: float = WINDOW_S) -> List[float]:
    """Livello (dB) di ogni finestra, mescolando i canali. Il pavimento del logaritmo viene dal file stesso."""
    size = max(1, int(audio.rate * window_s))
    mono = audio.samples if audio.channels == 1 else [
        sum(audio.samples[i:i + audio.channels]) / audio.channels
        for i in range(0, len(audio.samples), audio.channels)]
    rms = []
    for start in range(0, len(mono), size):
        block = mono[start:start + size]
        rms.append(math.sqrt(sum(v * v for v in block) / len(block)))
    floor = max(rms, default=0) * 1e-6 + 1e-9
    return [20 * math.log10(v + floor) for v in rms]


def otsu_threshold(values: List[float], bins: int = 128) -> float:
    """Soglia che separa i due gruppi di livelli del file (fondo e voce): massima varianza fra i gruppi."""
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return low - 1.0
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - low) / (high - low) * bins))] += 1
    centers = [low + (i + 0.5) * (high - low) / bins for i in range(bins)]
    total, total_sum = len(values), sum(c * x for c, x in zip(counts, centers))
    weight_low = sum_low = 0.0
    best, best_var = low, -1.0
    for count, center in zip(counts, centers):
        weight_low += count
        sum_low += count * center
        weight_high = total - weight_low
        if weight_low == 0 or weight_high <= 0:
            continue
        diff = sum_low / weight_low - (total_sum - sum_low) / weight_high
        var = weight_low * weight_high * diff * diff
        if var > best_var:
            best, best_var = center, var
    return best


def speech_windows(audio: Audio) -> Tuple[List[bool], float]:
    levels = window_levels_db(audio)
    if not levels:
        return [], 0.0
    if not any(audio.samples):  # silenzio digitale: nessuna voce
        return [False] * len(levels), max(levels)
    threshold = otsu_threshold(levels)
    return [v > threshold for v in levels], threshold


def highpass(audio: Audio, cutoff_hz: float = 70.0) -> Audio:
    """Filtro passa-alto a un polo per canale: toglie rimbombi e correnti continue molto basse."""
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    alpha = rc / (rc + 1.0 / audio.rate)
    out = array("h", audio.samples)
    for channel in range(audio.channels):
        previous_in = previous_out = 0.0
        for index in range(channel, len(out), audio.channels):
            sample = audio.samples[index]
            previous_out = alpha * (previous_out + sample - previous_in)
            previous_in = sample
            out[index] = clamp16(previous_out)
    return Audio(audio.rate, audio.channels, out)
