# Dipendenze: higgs_loader.py, llama_bridge.py. Confronta lo stato finale del "cervello" torch e llama.cpp sullo stesso ingresso,
# e misura quanto costa un passo con 1..N sequenze insieme.
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from higgs_loader import encode_voice, load_bundle  # noqa: E402
from llama_bridge import LlamaBackbone  # noqa: E402


def main():
    import higgs_client
    free_gb = higgs_client.free_video_memory_gb()
    if free_gb is None or free_gb < 14 or higgs_client.comfyui_is_generating():
        sys.exit(f"Scheda video non libera (liberi: {free_gb} GB, ne servono 14): non parto.")
    voice = sys.argv[1]
    text = "Nel cuore delle Alpi, il ghiacciaio custodisce da secoli il silenzio."
    bundle = load_bundle()
    ref, ref_text = encode_voice(bundle, voice)
    ids = bundle.tokenizer.build_prompt(text, num_ref_tokens=int(ref.shape[0]), reference_text=ref_text or None)
    with torch.inference_mode():
        embeds = bundle.model._prompt_embeds(ids, ref, bundle.device)
        reference = bundle.model.backbone.model(inputs_embeds=embeds).last_hidden_state[0, -1].float().cpu().numpy()
    # libera il "cervello" torch prima di caricare quello llama.cpp: mai due copie insieme sulla scheda
    embeds = embeds.cpu()
    bundle.model.backbone.model.layers = None
    bundle = None
    torch.cuda.empty_cache()
    backbone = LlamaBackbone(HERE / "models" / "higgs_backbone_bf16.gguf")
    length = embeds.shape[1]
    backbone.open(sequences=8, tokens_per_sequence=length + 600, max_batch_tokens=max(2048, length))
    flat = embeds[0].float().cpu().numpy()
    want = np.zeros(length, dtype=bool)
    want[-1] = True
    mine = backbone.step(flat, np.arange(length), np.zeros(length), want)[0]
    cosine = float(np.dot(reference, mine) / np.linalg.norm(reference) / np.linalg.norm(mine))
    print(f"coseno torch vs llama.cpp = {cosine:.5f}", flush=True)
    for sequences in (1, 4, 8):
        step_embeds = np.random.randn(sequences, backbone.n_embd).astype(np.float32) * 0.02
        positions = np.full(sequences, length, dtype=np.int32)
        seq_ids = np.arange(sequences, dtype=np.int32)
        for s in range(1, sequences):  # le altre sequenze hanno bisogno di un passato: copio il prefill a mano
            backbone.step(flat, np.arange(length), np.full(length, s), np.zeros(length, dtype=bool))
        start = time.perf_counter()
        for k in range(20):
            backbone.step(step_embeds, positions + k, seq_ids, np.ones(sequences, dtype=bool))
        print(f"{sequences} sequenze: {1000 * (time.perf_counter() - start) / 20:.1f} ms a passo", flush=True)
        backbone.open(sequences=8, tokens_per_sequence=length + 600, max_batch_tokens=max(2048, length))
        backbone.step(flat, np.arange(length), np.zeros(length), want)
    backbone.free()


if __name__ == "__main__":
    main()
