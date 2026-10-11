# Misure per controllare l'unione dei pezzi: «colpo» in un punto (salto fra campioni vicini) ed energia/frammenti dentro una pausa.
# Dipendenze: audio.wav_io. Usato da: tests, strumenti di confronto prima/dopo.
from __future__ import annotations

import math
from typing import List, Tuple

from .wav_io import Audio


def jump_near(audio: Audio, index: int, span_ms: float = 5.0) -> int:
    """Salto massimo fra due campioni consecutivi (mono/primo canale) attorno a un punto: un «colpo» è un salto grande."""
    span = int(audio.rate * span_ms / 1000)
    data = audio.samples[audio.channels * max(1, index - span): audio.channels * (index + span): audio.channels]
    return max((abs(data[i] - data[i - 1]) for i in range(1, len(data))), default=0)


def pause_stats(audio: Audio, start: int, end: int, window_ms: float = 2.0) -> Tuple[float, int, float]:
    """(livello rms in dBFS, quanti «frammenti» = gruppi isolati di finestre non mute, quota di campioni diversi da zero)."""
    size = max(1, int(audio.rate * window_ms / 1000))
    data = audio.samples[audio.channels * start: audio.channels * end: audio.channels]
    if not len(data):
        return -200.0, 0, 0.0
    rms = math.sqrt(sum(v * v for v in data) / len(data))
    flags = [any(data[i:i + size]) for i in range(0, len(data), size)]
    fragments = sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))
    return (20 * math.log10(rms / 32768) if rms > 0 else -200.0), fragments, sum(1 for v in data if v) / len(data)


def zero_runs(audio: Audio, min_s: float = 0.1) -> List[Tuple[int, int]]:
    """Tratti di silenzio digitale esatto lunghi almeno min_s (i punti dove stanno le pause inserite)."""
    need, runs, start = int(audio.rate * min_s), [], None
    data = audio.samples[::audio.channels]
    for i, v in enumerate(list(data) + [1]):
        if v == 0 and start is None:
            start = i
        elif v != 0 and start is not None:
            if i - start >= need:
                runs.append((start, i))
            start = None
    return runs


def edge_amplitudes(audio: Audio, start: int, end: int, span_ms: float = 0.5) -> Tuple[int, int]:
    """Ampiezza (picco) del suono subito PRIMA e subito DOPO un tratto di silenzio: se è grande, il suono è stato tagliato di netto = «colpo»."""
    span = max(1, int(audio.rate * span_ms / 1000))
    mono = audio.samples[::audio.channels]
    before = mono[max(0, start - span):start]
    after = mono[end:end + span]
    return (max((abs(v) for v in before), default=0), max((abs(v) for v in after), default=0))


# --- Misure INDIPENDENTI dal rilevatore di tidy.py (usano solo zeri esatti e picchi): servono a non autoconfermarsi ---------------

def _peaks(audio: Audio, window_ms: float) -> List[int]:
    size = max(1, int(audio.rate * window_ms / 1000))
    mono = audio.samples[::audio.channels]
    return [max((abs(v) for v in mono[i:i + size]), default=0) for i in range(0, len(mono), size)]


def envelope_peaks(audio: Audio, start_s: float, end_s: float, step_ms: float = 5.0) -> List[int]:
    """Picco di ogni finestra (es. 5 ms) fra due istanti: l'inviluppo come numeri, per guardarlo senza nessun rilevatore."""
    first, last = int(start_s * audio.rate), int(end_s * audio.rate)
    clipped = Audio(audio.rate, audio.channels, audio.samples[first * audio.channels: last * audio.channels])
    return _peaks(clipped, step_ms)


def faint_tail_ms(audio: Audio, silence_start: int, floor_ratio: float = 0.1) -> float:
    """Quanti ms di suono DEBOLISSIMO (picco sotto il 10% del picco tipico del parlato del file, cioè 20 dB sotto) stanno attaccati PRIMA di un silenzio.
    Una coda lunga = rumore a raffiche prima della pausa. Dopo una buona ripulitura è solo la dissolvenza."""
    reference = sorted(_peaks(audio, 2.0))[int(0.95 * (len(audio.samples) // audio.channels // max(1, int(audio.rate * 0.002))))] if audio.samples else 0
    limit, mono, size = floor_ratio * reference, audio.samples[::audio.channels], max(1, int(audio.rate * 0.002))
    pause = int(audio.rate * 0.060)          # zeri per 60 ms di fila = una vera pausa: la coda finisce lì (zeri più brevi sono solo l'intermittenza)
    index, zeros = silence_start, 0
    while index - size >= 0:
        block = mono[index - size:index]
        if not any(block):
            zeros += size
            if zeros >= pause:
                index += zeros - size
                break
        elif max(abs(v) for v in block) <= limit:
            zeros = 0
        else:
            break
        index -= size
    return (silence_start - index) * 1000.0 / audio.rate


def isolated_faint_runs(audio: Audio, min_gap_ms: float = 60.0, max_len_ms: float = 40.0, floor_ratio: float = 0.05) -> int:
    """Quanti tratti brevi (< max_len_ms) e deboli (picco sotto il 5% del parlato) stanno fra due silenzi esatti di almeno min_gap_ms."""
    mono = audio.samples[::audio.channels]
    reference = sorted(_peaks(audio, 2.0))[int(0.95 * len(_peaks(audio, 2.0)))] if mono else 0
    gap, longest = int(audio.rate * min_gap_ms / 1000), int(audio.rate * max_len_ms / 1000)
    count, i, n = 0, 0, len(mono)
    while i < n:
        if mono[i] == 0:
            i += 1
            continue
        j = i
        while j < n and not (mono[j] == 0 and not any(mono[j:j + max(1, audio.rate // 1000)])):   # un buco < 1 ms non separa due tratti
            j += 1
        before_zeros = i >= gap and not any(mono[i - gap:i])
        after_zeros = j + gap <= n and not any(mono[j:j + gap])
        if before_zeros and after_zeros and (j - i) < longest and max(abs(v) for v in mono[i:j]) <= floor_ratio * reference:
            count += 1
        i = j + 1
    return count
