import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from higgs_loader import MODEL_DIR, load_native_module
native = load_native_module()
codes = torch.randint(0, 1024, (500, 8))
out = {}
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    codec = native.HiggsAudioCodec.from_pretrained(MODEL_DIR, device=torch.device("cuda"), dtype=dtype)
    with torch.inference_mode():
        out[dtype] = codec.decode(codes).double()
    del codec
ref = out[torch.float32]
for dtype in (torch.float16, torch.bfloat16):
    noise = ref - out[dtype]
    print(dtype, "SNR_dB=%.1f" % (10 * torch.log10(ref.pow(2).mean() / noise.pow(2).mean())).item())
