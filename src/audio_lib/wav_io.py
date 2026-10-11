# Lettura/scrittura di WAV (PCM 16/24/32 e float, vedi wav_format.py) e piccoli attrezzi comuni (silenzio, unione, guadagno, picco).
# Dipendenze: audio.wav_format. Usato da: audio/*, creation/*, speak/*, engines/fake_engine.
from __future__ import annotations

import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .wav_format import WavFormatError, build_wav, parse_wav


class AudioError(Exception):
    """Problema con un file audio, con messaggio in italiano."""


@dataclass
class Audio:
    rate: int
    channels: int
    samples: array  # 'h' interleaved

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels

    @property
    def seconds(self) -> float:
        return self.frames / self.rate


@dataclass
class StepResult:
    audio: Audio
    message: str


def clamp16(value: float) -> int:
    return -32768 if value < -32768 else 32767 if value > 32767 else int(round(value))


def read_wav(path: Path) -> Audio:
    """Legge PCM 8/16/24/32 bit e virgola mobile 32/64 bit (anche «extensible»): dentro il programma l'audio è a 16 bit."""
    try:
        rate, channels, samples = parse_wav(Path(path).read_bytes())
    except (OSError, WavFormatError, struct.error) as error:
        raise AudioError(f"Non riesco a leggere il file audio ({error}).") from error
    return Audio(rate, channels, samples)


def write_wav(path: Path, audio: Audio, sample_format: str = "pcm16") -> Path:
    """Scrive pcm16 (predefinito, leggibile da tutti), pcm24 o float32."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(build_wav(audio.rate, audio.channels, audio.samples, sample_format))
    except WavFormatError as error:
        raise AudioError(str(error)) from error
    return path


def silence(rate: int, channels: int, seconds: float) -> Audio:
    return Audio(rate, channels, array("h", [0] * (int(round(seconds * rate)) * channels)))


def peak(audio: Audio) -> int:
    return max(max(audio.samples), -min(audio.samples)) if audio.samples else 0


def apply_gain(audio: Audio, factor: float) -> Audio:
    return Audio(audio.rate, audio.channels, array("h", [clamp16(v * factor) for v in audio.samples]))


def join(parts: List[Audio], gap_seconds: float = 0.0) -> Audio:
    if not parts:
        raise AudioError("Niente da unire.")
    first = parts[0]
    result = array("h")
    gap = silence(first.rate, first.channels, gap_seconds).samples
    for index, part in enumerate(parts):
        if part.rate != first.rate or part.channels != first.channels:
            raise AudioError("I pezzi audio hanno formati diversi: non posso unirli.")
        if index:
            result.extend(gap)
        result.extend(part.samples)
    return Audio(first.rate, first.channels, result)
