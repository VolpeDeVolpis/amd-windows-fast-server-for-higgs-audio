# Lettura e scrittura dei file WAV in tutti i formati usati qui: PCM 8/16/24/32 bit e virgola mobile 32/64 bit (il formato 3 di
# Higgs/ComfyUI, che il modulo «wave» di Python non legge: «unknown format: 3»), anche con intestazione «extensible».
# Dentro il programma l'audio è sempre a 16 bit: qui si converte in lettura e si sceglie il formato in scrittura.
# Dipendenze: nessuna (libreria standard). Usato da: audio/wav_io.
from __future__ import annotations

import struct
import sys
from array import array
from typing import Tuple

PCM, FLOAT, EXTENSIBLE = 1, 3, 0xFFFE
WRITE_FORMATS = ("pcm16", "pcm24", "float32")


class WavFormatError(Exception):
    """Il file non è un WAV leggibile (messaggio in italiano)."""


def _int16_from_high_bytes(raw: bytes, width: int, offset_of_high: int) -> array:
    """Tiene i due byte più alti di ogni campione a `width` byte (little-endian): è la conversione a 16 bit."""
    count = len(raw) // width
    out = bytearray(2 * count)
    out[0::2] = raw[offset_of_high - 1: count * width: width]
    out[1::2] = raw[offset_of_high: count * width: width]
    samples = array("h")
    samples.frombytes(bytes(out))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def _from_float(raw: bytes, code: str) -> array:
    values = array(code)
    size = values.itemsize
    values.frombytes(raw[: len(raw) - len(raw) % size])
    if sys.byteorder == "big":
        values.byteswap()
    return array("h", [32767 if v >= 1.0 else -32768 if v <= -1.0 else int(round(v * 32767)) for v in values])


def _decode(tag: int, bits: int, raw: bytes) -> array:
    if tag == PCM:
        if bits == 16:
            samples = array("h")
            samples.frombytes(raw[: len(raw) - len(raw) % 2])
            if sys.byteorder == "big":
                samples.byteswap()
            return samples
        if bits == 8:
            return array("h", [(v - 128) << 8 for v in raw])
        if bits == 24:
            return _int16_from_high_bytes(raw, 3, 2)
        if bits == 32:
            return _int16_from_high_bytes(raw, 4, 3)
    if tag == FLOAT and bits in (32, 64):
        return _from_float(raw, "f" if bits == 32 else "d")
    raise WavFormatError(f"Formato audio non gestito (tipo {tag}, {bits} bit).")


def parse_wav(data: bytes) -> Tuple[int, int, array]:
    """(frequenza, canali, campioni a 16 bit intercalati)."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavFormatError("Non è un file WAV.")
    position, fmt, raw = 12, None, None
    while position + 8 <= len(data):
        name, size = data[position:position + 4], struct.unpack("<I", data[position + 4:position + 8])[0]
        body = position + 8
        if name == b"fmt ":
            tag, channels, rate, _byte_rate, _block, bits = struct.unpack("<HHIIHH", data[body:body + 16])
            if tag == EXTENSIBLE and size >= 40:
                tag = struct.unpack("<H", data[body + 24:body + 26])[0]   # primi due byte del sottoformato
            fmt = (tag, channels, rate, bits)
        elif name == b"data":
            raw = data[body: body + size] if size != 0xFFFFFFFF else data[body:]   # dimensione dichiarata troppo grande: si prende ciò che c'è
            break
        position = body + size + (size & 1)
    if fmt is None or raw is None:
        raise WavFormatError("Al file WAV mancano l'intestazione o i dati.")
    tag, channels, rate, bits = fmt
    if channels < 1 or rate < 1:
        raise WavFormatError("Intestazione WAV non valida.")
    samples = _decode(tag, bits, raw)
    return rate, channels, array("h", samples[: len(samples) - len(samples) % channels])


def build_wav(rate: int, channels: int, samples: array, sample_format: str = "pcm16") -> bytes:
    if sample_format not in WRITE_FORMATS:
        raise WavFormatError(f"Formato di scrittura sconosciuto: {sample_format}")
    data16 = array("h", samples)
    if sys.byteorder == "big":
        data16.byteswap()
    if sample_format == "pcm16":
        tag, bits, payload = PCM, 16, data16.tobytes()
    elif sample_format == "pcm24":
        raw16 = data16.tobytes()
        payload = bytearray(3 * len(data16))
        payload[1::3], payload[2::3] = raw16[0::2], raw16[1::2]
        tag, bits, payload = PCM, 24, bytes(payload)
    else:
        floats = array("f", [v / 32768.0 for v in samples])
        if sys.byteorder == "big":
            floats.byteswap()
        tag, bits, payload = FLOAT, 32, floats.tobytes()
    block = channels * bits // 8
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(payload) + (len(payload) & 1), b"WAVE", b"fmt ", 16,
                         tag, channels, rate, rate * block, block, bits, b"data", len(payload))
    return header + payload + (b"\0" if len(payload) & 1 else b"")


def is_plain_pcm16(data: bytes) -> bool:
    """True se è un WAV PCM a 16 bit semplice (quello che qualunque lettore, anche winsound, suona)."""
    try:
        position = 12
        while position + 8 <= len(data):
            name, size = data[position:position + 4], struct.unpack("<I", data[position + 4:position + 8])[0]
            if name == b"fmt ":
                tag, _channels, _rate, _byte_rate, _block, bits = struct.unpack("<HHIIHH", data[position + 8:position + 24])
                return tag == PCM and bits == 16
            position += 8 + size + (size & 1)
    except struct.error:
        pass
    return False
