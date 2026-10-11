# Dependencies: audio_lib/tidy.py and wav_io.py (seam cleanup validated by ear). Used by: server.py.
# Ripulisce ogni pezzo generato prima dell'unione: via le code di rumore del codec ("saltelli"), bordi del parlato sfumati
# (niente "colpo" contro la pausa), parlato identico. Riapplicarla sullo stesso audio non fa danni.
from __future__ import annotations

import sys
from array import array
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio_lib.tidy import tidy_speech  # noqa: E402
from audio_lib.wav_io import Audio, AudioError  # noqa: E402


def to_int16(audio: np.ndarray) -> array:
    return array("h", (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype(np.int16).tobytes())


WINDOW_S = 0.005
DIP_LEVEL = 0.06          # "calo": picco sotto il 6% del livello tipico del parlato del pezzo stesso
MIN_DIP_S = 0.02          # calo di almeno 20 ms
FRAGMENT_MAX_S = 0.40     # piu' corto di una sillaba accentata con la sua coda
FRAGMENT_MAX_PEAK = 0.40  # una sillaba vera arriva vicino al livello del parlato; la raffica finale no


def drop_trailing_fragment(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Removes the short weak burst the model often adds after the last word, before it stops ("colpo" at the end).
    Levels come from the piece itself (90th percentile of 5 ms peaks); repeats while a fragment qualifies."""
    window = max(1, int(sample_rate * WINDOW_S))
    while True:
        frames = len(audio) // window
        if frames < 4:
            return audio
        peaks = np.abs(audio[:frames * window]).reshape(frames, window).max(axis=1)
        if not (peaks > 0).any():
            return audio
        speech_level = float(np.percentile(peaks[peaks > 0], 90))
        quiet = peaks < DIP_LEVEL * speech_level
        last_sound = int(np.flatnonzero(~quiet)[-1]) if (~quiet).any() else -1
        if last_sound < 0:
            return audio
        # inizio del frammento finale: dopo l'ultimo calo lungo almeno MIN_DIP_S prima dell'ultimo suono
        need = max(1, int(MIN_DIP_S / WINDOW_S))
        index, run, dip_end = last_sound, 0, None
        while index >= 0:
            run = run + 1 if quiet[index] else 0
            if run >= need:
                dip_end = index + need  # primo riquadro dopo il calo
                break
            index -= 1
        if dip_end is None:
            return audio
        fragment = peaks[dip_end:last_sound + 1]
        duration_s = (last_sound + 1 - dip_end) * WINDOW_S
        if duration_s > FRAGMENT_MAX_S or fragment.max() > FRAGMENT_MAX_PEAK * speech_level:
            return audio
        dip_start = dip_end - need
        while dip_start > 0 and quiet[dip_start - 1]:
            dip_start -= 1
        audio = audio[:dip_start * window]  # taglia all'inizio del calo


def tidy_chunk(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, str | None]:
    """Float mono audio -> (cleaned float audio, warning). A chunk with no speech at all is returned unchanged
    with a warning, so the caller can report it instead of losing the whole request."""
    # drop_trailing_fragment NON si applica: su parlato vero ha tolto un "ta" finale (dopo la pausa della "t") —
    # tagliare parole e' peggio di un colpo raro. Resta disponibile solo per prove.
    try:
        cleaned, _report = tidy_speech(Audio(sample_rate, 1, to_int16(audio)))
    except AudioError as error:
        return audio, str(error)
    return np.frombuffer(cleaned.samples.tobytes(), dtype=np.int16).astype(np.float32) / 32767.0, None
