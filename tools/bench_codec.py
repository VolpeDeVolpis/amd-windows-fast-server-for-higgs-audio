# Dipendenze: higgs_loader.py. Misura la conversione codici->audio (codec) in varie configurazioni.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from higgs_loader import MODEL_DIR, load_native_module


def timed_decode(codec, codes, label):
    with torch.inference_mode():
        codec.decode(codes[:50])  # warm-up
        if codec.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        audio = codec.decode(codes)
        if codec.device.type == "cuda":
            torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(f"{label}: {seconds:.1f} s for {audio.numel() / 24000:.0f} s audio", flush=True)


def main():
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 750
    native = load_native_module()
    codes = torch.randint(0, 1024, (frames, 8))
    for device_name, dtype in (("cuda", torch.float32), ("cuda", torch.bfloat16), ("cuda", torch.float16), ("cpu", torch.float32)):
        codec = native.HiggsAudioCodec.from_pretrained(MODEL_DIR, device=torch.device(device_name), dtype=dtype)
        timed_decode(codec, codes, f"{device_name} {dtype}")
        del codec
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
