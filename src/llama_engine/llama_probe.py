# Dipendenze: llama_bridge.py, numpy (niente torch: processo separato). Uso: llama_probe.py <cartella di make_reference>
# Confronta lo stato finale di llama.cpp con quello torch salvato, poi misura il costo di un passo con 1, 4, 8, 12 sequenze.
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_bridge import LlamaBackbone  # noqa: E402


def cosine(a, b):
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    data = Path(sys.argv[1])
    embeds = np.load(data / "prompt_embeds.npy")
    reference = np.load(data / "reference_hidden.npy")
    length = embeds.shape[0]
    backbone = LlamaBackbone(Path(__file__).resolve().parent / "models" / "higgs_backbone_bf16.gguf")
    for sequences in (1, 4, 8, 12):
        backbone.open(sequences=sequences, tokens_per_sequence=length + 64, max_batch_tokens=max(2048, length))
        for seq in range(sequences):
            last = backbone.prefill(embeds, seq)
        if sequences == 1:
            print(f"coseno ultimo stato torch vs llama.cpp = {cosine(reference[-1], last):.5f}", flush=True)
        step = np.tile(embeds[-1:], (sequences, 1))
        seq_ids = np.arange(sequences, dtype=np.int32)
        backbone.step(step, np.full(sequences, length), seq_ids, np.ones(sequences, dtype=bool))  # riscaldamento
        start = time.perf_counter()
        repeats = 30
        for k in range(1, repeats + 1):
            backbone.step(step, np.full(sequences, length + k), seq_ids, np.ones(sequences, dtype=bool))
        print(f"{sequences} sequenze: {1000 * (time.perf_counter() - start) / repeats:.1f} ms a passo", flush=True)
    backbone.free()


if __name__ == "__main__":
    import threading
    threading.stack_size(64 * 1024 * 1024)  # llama.cpp si aspetta una pila grande come nei suoi programmi
    worker = threading.Thread(target=main)
    worker.start()
    worker.join()
