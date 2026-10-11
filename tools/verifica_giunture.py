# Dependencies: audio_lib/measure.py and wav_io.py (misure indipendenti dalla pulizia). Uso: verifica_giunture.py a.wav [b.wav ...]
# Per ogni pausa di vero silenzio: coda debole attaccata prima (ms) e salto fra campioni al bordo (colpo).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audio_lib.measure import faint_tail_ms, jump_near, zero_runs  # noqa: E402
from audio_lib.wav_io import read_wav  # noqa: E402

for path in sys.argv[1:]:
    audio = read_wav(Path(path))
    runs = zero_runs(audio, 0.1)
    tails = [faint_tail_ms(audio, start) for start, _end in runs]
    jumps = [max(jump_near(audio, start), jump_near(audio, end)) for start, end in runs]
    print(f"{Path(path).name}: pause={len(runs)} coda_debole_ms max={max(tails, default=0):.0f} "
          f"media={sum(tails) / max(len(tails), 1):.0f} salto_bordo_max={max(jumps, default=0)}")
