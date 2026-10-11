# Dipendenze: higgs_loader.py, llama_higgs_engine.py, text_chunks.py. Uso: bench_llama_engine.py <testo.txt> <voce> <uscita.wav>
# Genera un testo intero col motore llama.cpp (paragrafi insieme) e misura i secondi: confronto diretto col server torch.
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from higgs_loader import encode_voice, load_bundle  # noqa: E402
from llama_higgs_engine import LlamaHiggsEngine  # noqa: E402
from text_chunks import split_text  # noqa: E402


def main():
    text = Path(sys.argv[1]).read_text(encoding="utf-8-sig")
    voice, out_path = sys.argv[2], Path(sys.argv[3])
    bundle = load_bundle()
    engine = LlamaHiggsEngine(bundle, batch_size=12, max_cache_len=3072)
    ref, ref_text = encode_voice(bundle, voice)
    chunks = split_text(text, 500, 80)
    requests = [(bundle.tokenizer.build_prompt(c, num_ref_tokens=int(ref.shape[0]), reference_text=ref_text or None), ref)
                for c in chunks]
    torch.manual_seed(42)
    start = time.perf_counter()
    delayed = engine.generate(requests, max_new_tokens=1024, temperature=1.0, top_p=0.95, top_k=None)
    gen_s = time.perf_counter() - start
    pieces = []
    with torch.inference_mode():
        for d in delayed:
            pieces.append(bundle.codec.decode(bundle.native.reverse_delay_pattern(d)).numpy())
            pieces.append(np.zeros(int(0.25 * bundle.native.SAMPLE_RATE), dtype=np.float32))
    audio = np.concatenate(pieces)
    soundfile.write(out_path, audio, bundle.native.SAMPLE_RATE)
    total_s = time.perf_counter() - start
    print(f"pezzi={len(chunks)} passi={engine.last_steps} generazione_s={gen_s:.1f} "
          f"ms_a_passo={1000 * gen_s / max(engine.last_steps, 1):.1f} totale_s={total_s:.1f} "
          f"audio_s={len(audio) / bundle.native.SAMPLE_RATE:.1f}", flush=True)


if __name__ == "__main__":
    main()
