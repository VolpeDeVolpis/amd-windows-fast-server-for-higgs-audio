# Dipendenze: higgs_loader.py (torch, scheda video). Uso: make_reference.py <voce> <cartella uscita>
# Salva un ingresso vero di Higgs (vettori del prompt) e lo stato finale calcolato dal "cervello" originale torch,
# cosi' llama_probe.py puo' confrontarlo in un processo separato senza torch.
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from higgs_loader import encode_voice, load_bundle  # noqa: E402


def main():
    voice, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    text = "Nel cuore delle Alpi, il ghiacciaio custodisce da secoli il silenzio."
    bundle = load_bundle()
    ref, ref_text = encode_voice(bundle, voice)
    ids = bundle.tokenizer.build_prompt(text, num_ref_tokens=int(ref.shape[0]), reference_text=ref_text or None)
    with torch.inference_mode():
        embeds = bundle.model._prompt_embeds(ids, ref, bundle.device)
        hidden = bundle.model.backbone.model(inputs_embeds=embeds).last_hidden_state[0].float().cpu().numpy()
    np.save(out_dir / "prompt_embeds.npy", embeds[0].float().cpu().numpy())
    np.save(out_dir / "reference_hidden.npy", hidden)
    print(f"salvati {embeds.shape[1]} vettori di ingresso e lo stato di riferimento")


if __name__ == "__main__":
    main()
