# Dipendenze: nessuna. Usato da: server.py.
# Spezza un testo in pezzi da leggere in parallelo: confini di frase, pezzi non oltre max_chars.
from __future__ import annotations

import math
import re

# solo fine frase vera: ";" e ":" lasciano la frase aperta ("il ghiaccio canta: un suono...") e tagliarli spezza l'intonazione
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
CLAUSE_END = re.compile(r"(?<=[,])\s+")


def _split_long(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]
    parts, current = [], ""
    for clause in CLAUSE_END.split(piece):
        candidate = f"{current} {clause}".strip()
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = clause
        else:
            current = candidate
    if current:
        parts.append(current)
    # a clause still too long: cut on spaces
    result = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            result.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            result.append(part)
    return result


PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def split_text(text: str, max_chars: int, min_chars: int) -> list[str]:
    """Paragraphs are never joined; a paragraph is split only when longer than max_chars."""
    chunks = []
    for paragraph in PARAGRAPH_BREAK.split(text.strip()):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            chunks.extend(_split_paragraph(paragraph, max_chars, min_chars))
    return merge_short_neighbours(chunks, max_chars)


def merge_short_neighbours(chunks: list[str], max_chars: int) -> list[str]:
    """Joins adjacent chunks into one generation (smoother audio) while the result stays within the
    longest chunk of the text: the batch already waits for that one, so total time does not grow."""
    if not chunks:
        return chunks
    limit = min(max(len(c) for c in chunks), max_chars)
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        candidate = f"{merged[-1]} {chunk}"
        if len(candidate) <= limit:
            merged[-1] = candidate
        else:
            merged.append(chunk)
    return merged


def _split_paragraph(text: str, max_chars: int, min_chars: int) -> list[str]:
    """A paragraph longer than max_chars becomes the FEWEST chunks of similar length (fewer joins), cut only
    at sentence ends; commas are cut only when a single sentence is longer than max_chars."""
    if len(text) <= max_chars:
        return [text]
    sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    pieces = [p for s in sentences for p in _split_long(s, max_chars)]
    target = len(text) / math.ceil(len(text) / max_chars)  # lunghezza media dei pezzi con il minimo numero di pezzi
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and (len(candidate) > max_chars or (len(current) >= target and len(current) >= min_chars)):
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
