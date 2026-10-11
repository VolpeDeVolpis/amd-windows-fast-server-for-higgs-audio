# Dipendenze: higgs_loader.py. Codec sulla scheda: con e senza la libreria MIOpen per le convoluzioni.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from higgs_loader import MODEL_DIR, load_native_module

native = load_native_module()
codes = torch.randint(0, 1024, (int(sys.argv[1]) if len(sys.argv) > 1 else 750, 8))
results = {}
for label, use_miopen in (("miopen on", True), ("miopen off", False)):
    torch.backends.cudnn.enabled = use_miopen
    codec = native.HiggsAudioCodec.from_pretrained(MODEL_DIR, device=torch.device("cuda"), dtype=torch.float32)
    with torch.inference_mode():
        codec.decode(codes[:50])
        torch.cuda.synchronize()
        start = time.perf_counter()
        results[label] = codec.decode(codes).double()
        torch.cuda.synchronize()
    print(label, "%.1f s" % (time.perf_counter() - start), flush=True)
    codec = None
    torch.cuda.empty_cache()
difference = (results["miopen on"] - results["miopen off"]).abs().max().item()
print("max_difference", difference)
