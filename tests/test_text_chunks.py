# Dipendenze: text_chunks.py. Prove sul taglio del testo (nessun file vero).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_chunks import split_text  # noqa: E402


def sentence(words, end="."):
    return " ".join(f"parola{i}" for i in range(words)) + end


def test_colon_and_semicolon_never_cut():
    text = sentence(30, ":") + " " + sentence(30, ";") + " " + sentence(30)
    chunks = split_text(text, 200, 80)
    for chunk in chunks:
        assert not chunk.endswith((":", ";")), chunks


def test_long_paragraph_uses_fewest_balanced_chunks():
    text = " ".join(sentence(12) for _ in range(8))  # ~8 frasi, oltre il doppio del limite
    max_chars = 400
    chunks = split_text(text, max_chars, 80)
    fewest = -(-len(text) // max_chars)
    assert len(chunks) <= fewest + 1
    assert all(len(c) <= max_chars for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_paragraphs_never_joined_across_long_ones_and_short_merged():
    long_paragraph = sentence(40)
    text = f"{long_paragraph}\n\n{sentence(3)}\n\n{sentence(3)}"
    chunks = split_text(text, 500, 80)
    assert chunks[0] == long_paragraph
    assert len(chunks) == 2  # i due brevi uniti fra loro, non oltre il pezzo piu' lungo
