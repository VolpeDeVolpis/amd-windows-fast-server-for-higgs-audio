# RIPULISCI IL PARLATO GENERATO: code e teste dei pezzi senza parlato (rumore a bassissimo livello del codec, «saltelli»),
# bordi del parlato con dissolvenza breve (niente «colpo» nella giunzione), pause di VERO silenzio. Nessun valore tarato su un file:
# la soglia nasce dal livello del pezzo stesso (due gruppi di livelli: parlato e fondo; fondo = mediana del gruppo basso).
# Parlato vero = energia CONTINUA: un tratto vale solo se contiene almeno 40 ms sostenuti sopra «fondo + 12 dB» (non bastano picchi
# isolati né una salita lenta di rumore). Si taglia sotto «fondo + 6 dB»; i margini (consonanti deboli, code) si tengono solo attorno a
# parlato vero e sfumano per tutta la loro lunghezza; le pause più corte di 80 ms (occlusive, esitazioni) contano come parlato. Dipendenze: audio.adaptive, audio.wav_io. Usato da:
# speak/say_service, reuse/base_builder. Usato anche da: higgs/audio_tidy.py (server Higgs: NON cambiare firma né comportamento di base senza avvisare).
from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import List, Tuple

from .adaptive import otsu_threshold
from .wav_io import Audio, AudioError


@dataclass(frozen=True)
class TidyParams:
    env_ms: float = 5.0          # finestra con cui si misura il livello
    low_offset_db: float = 6.0   # sopra il fondo di tanti dB c'è parlato
    min_pause_ms: float = 80.0   # pause più corte (occlusive, esitazioni) fanno parte del parlato
    core_offset_db: float = 12.0  # il «cuore» del parlato sta almeno di tanti dB sopra il fondo
    core_ms: float = 40.0        # e deve durare tanto, senza buchi più lunghi di core_gap_ms: altrimenti non è parlato
    core_gap_ms: float = 10.0
    lead_ms: float = 40.0        # margine tenuto PRIMA del parlato (consonanti deboli, attacco)
    tail_ms: float = 30.0        # margine tenuto DOPO il parlato (code, respiri finali); sfuma per intero
    fade_ms: float = 12.0        # dissolvenza sul bordo di ogni tratto tenuto


@dataclass
class TidyReport:
    regions: List[Tuple[int, int]]   # tratti di parlato tenuti (campioni, dopo i margini)
    removed_s: float                 # secondi azzerati o tagliati
    background_db: float
    threshold_db: float


def _envelope_db(audio: Audio, size: int) -> List[float]:
    channels, samples = audio.channels, audio.samples
    levels = []
    step = size * channels
    for start in range(0, len(samples), step):
        block = samples[start:start + step]
        levels.append(math.sqrt(sum(v * v for v in block) / len(block)) if block else 0.0)
    return levels


def _to_db(rms: float) -> float:
    return 20 * math.log10(rms / 32768) if rms > 0 else -200.0


def find_regions(audio: Audio, params: TidyParams = TidyParams()) -> Tuple[List[Tuple[int, int]], float, float]:
    """Tratti di parlato (campioni, margini compresi) + livello del fondo + soglia, tutto ricavato dal pezzo."""
    full, background_db, low = _find_full(audio, params)
    return [(a, b) for a, b, _ca, _cb in full], background_db, low


def _find_full(audio: Audio, params: TidyParams):
    size = max(1, int(audio.rate * params.env_ms / 1000))
    db = [_to_db(v) for v in _envelope_db(audio, size)]
    audible = [v for v in db if v > -199]
    if not audible:
        return [], -200.0, -200.0
    if max(audible) - min(audible) < 3.0:   # livello uniforme (niente fondo da distinguere): è parlato tutto ciò che non è silenzio digitale
        background_db = min(audible) - 1.0
        flags = [v > -199 for v in db]
        return _regions_from(audio, params, flags, flags, size), background_db, background_db
    split = otsu_threshold(audible)
    background = sorted(v for v in audible if v <= split)
    background_db = background[len(background) // 2] if background else min(audible)
    low = min(split, background_db + params.low_offset_db)
    core = max(low, min(split, background_db + params.core_offset_db))
    return _regions_from(audio, params, [v > low for v in db], [v > core for v in db], size), background_db, low


def _runs(flags: List[bool]) -> List[List[int]]:
    runs, start = [], None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append([start, index])
            start = None
    return runs


def _sustained(core_flags: List[bool], start: int, end: int, gap: int, need: int) -> bool:
    """C'è un tratto continuo (buchi fino a `gap` finestre ammessi) di almeno `need` finestre sopra la soglia del cuore?"""
    best, current, hole, seen = 0, 0, 0, False
    for flag in core_flags[start:end]:
        if flag:
            current += hole + 1 if seen else 1
            hole, seen = 0, True
            best = max(best, current)
        elif seen:
            hole += 1
            if hole > gap:
                current, hole, seen = 0, 0, False
    return best >= need


def _regions_from(audio: Audio, params: TidyParams, active: List[bool], core_flags: List[bool], size: int):
    """Da finestre attive/non attive ai tratti di parlato VERO: pause brevi unite, tratti senza energia continua scartati, margini."""
    bridge = max(1, int(params.min_pause_ms / params.env_ms))
    merged: List[List[int]] = []
    for run in _runs(active):
        if merged and run[0] - merged[-1][1] < bridge:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    need, gap = max(1, int(params.core_ms / params.env_ms)), int(params.core_gap_ms / params.env_ms)
    merged = [run for run in merged if _sustained(core_flags, run[0], run[1], gap, need)]
    lead, tail = int(audio.rate * params.lead_ms / 1000), int(audio.rate * params.tail_ms / 1000)
    spans = [(a * size, min(audio.frames, b * size)) for a, b in merged]
    regions = []
    for i, (a, b) in enumerate(spans):
        gap_before = a - (spans[i - 1][1] if i else 0)
        gap_after = (spans[i + 1][0] if i + 1 < len(spans) else audio.frames) - b
        tail_room = min(tail, gap_after * tail // (tail + lead)) if i + 1 < len(spans) else min(tail, gap_after)
        lead_room = min(lead, gap_before * lead // (tail + lead)) if i else min(lead, gap_before)
        regions.append((a - lead_room, b + tail_room, a, b))   # (inizio, fine) con margini + (inizio, fine) del cuore
    return regions


def tidy_speech(audio: Audio, trim_outer: bool = True, params: TidyParams = TidyParams()) -> Tuple[Audio, TidyReport]:
    if not audio.samples:
        raise AudioError("Il pezzo audio è vuoto.")
    regions, background_db, low = _find_full(audio, params)
    if not regions:
        raise AudioError("Nel pezzo non si sente niente: è tutto silenzio.")
    channels = audio.channels
    out = array("h", [0] * len(audio.samples))
    fade = int(audio.rate * params.fade_ms / 1000)
    mono, kept_regions = audio.samples[::channels], []
    for a, b, core_a, core_b in regions:
        a, b = max(0, a), min(audio.frames, b)
        while b > a and mono[b - 1] == 0:   # silenzio digitale già presente: la dissolvenza deve stare sul suono, non sugli zeri
            b -= 1
        while a < b and mono[a] == 0:
            a += 1
        if b <= a:
            continue
        kept_regions.append((a, b))
        out[a * channels:b * channels] = audio.samples[a * channels:b * channels]
        # la dissolvenza copre TUTTO il margine (le raffiche di rumore ai lati del parlato calano fino a zero), almeno fade_ms
        fade_in = min(max(fade, core_a - a), (b - a) // 2)
        fade_out = min(max(fade, b - core_b), (b - a) // 2)
        for k in range(fade_in):
            gain = 0.5 - 0.5 * math.cos(math.pi * k / fade_in)    # 0 -> 1 (coseno rialzato)
            for c in range(channels):
                out[(a + k) * channels + c] = int(out[(a + k) * channels + c] * gain)
        for k in range(fade_out):
            gain = 0.5 - 0.5 * math.cos(math.pi * k / fade_out)
            for c in range(channels):
                out[(b - 1 - k) * channels + c] = int(out[(b - 1 - k) * channels + c] * gain)
    if not kept_regions:
        raise AudioError("Nel pezzo non si sente niente: è tutto silenzio.")
    regions = kept_regions
    first, last = regions[0][0], regions[-1][1]
    if trim_outer:
        out = out[max(0, first) * channels:min(audio.frames, last) * channels]
    kept = sum(min(audio.frames, b) - max(0, a) for a, b in regions)
    removed = (audio.frames - kept) / audio.rate
    return Audio(audio.rate, channels, out), TidyReport(regions, removed, background_db, low)
