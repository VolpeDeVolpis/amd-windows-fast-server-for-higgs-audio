# Dependencies: native.py of the ComfyUI node TTS-Audio-Suite (MIT, NOT included: see THIRD_PARTY.md), torch/transformers.
# Usato da: bench.py, server.py. Carica Higgs v3 senza ComfyUI e prepara la voce di riferimento.
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile
import torch

def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set: see README.md, section Configuration.")
    return Path(value)


# HIGGS_NATIVE_PATH: .../TTS-Audio-Suite/engines/higgs_audio_v3/native.py   (downloaded separately by you)
# HIGGS_MODEL_DIR:   folder with the official Higgs Audio v3 files (config.json, model.safetensors, tokenizer.json...)
# HIGGS_VOICES_DIR:  folder with reference voices (NAME.wav + optional NAME.txt); default ./voices
NATIVE_PATH = _required_path("HIGGS_NATIVE_PATH")
MODEL_DIR = _required_path("HIGGS_MODEL_DIR")
VOICES_DIR = Path(os.environ.get("HIGGS_VOICES_DIR", str(Path(__file__).resolve().parents[1] / "voices")))


def load_native_module():
    spec = importlib.util.spec_from_file_location("higgs_native", NATIVE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["higgs_native"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class HiggsBundle:
    native: object
    model: object
    codec: object
    tokenizer: object
    device: torch.device
    dtype: torch.dtype


def load_bundle(device_name: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                codec_device: str | None = None) -> HiggsBundle:
    """codec_device: where codes become audio ("cpu" = same result as GPU fp32, faster on this PC)."""
    native = load_native_module()
    # MIOpen convolutions on Windows run without workspace and fall back to slow solvers:
    # PyTorch's own GPU convolutions give the same audio ~50x faster (codec only uses convolutions).
    torch.backends.cudnn.enabled = False
    device = torch.device(device_name)
    config = native.read_config(MODEL_DIR)
    model = native.build_native_model(config, dtype, "sdpa")
    native.load_native_weights(model, MODEL_DIR, device, dtype)
    codec = native.HiggsAudioCodec.from_pretrained(MODEL_DIR, device=torch.device(codec_device or device_name),
                                                   dtype=torch.float32)
    tokenizer = native.load_tokenizer(MODEL_DIR)
    return HiggsBundle(native, model, codec, tokenizer, device, dtype)


def read_voice(voice_name: str) -> tuple[torch.Tensor, int, str]:
    data, sample_rate = soundfile.read(str(VOICES_DIR / f"{voice_name}.wav"), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    reference_text = ""
    for candidate in (f"{voice_name}.reference.txt", f"{voice_name}.txt"):
        path = VOICES_DIR / candidate
        if path.is_file():
            reference_text = path.read_text(encoding="utf-8-sig").strip()
            break
    return wav, sample_rate, reference_text


def encode_voice(bundle: HiggsBundle, voice_name: str, max_reference_seconds: float = 100.0):
    """Returns (delayed reference codes, reference text), same steps as the ComfyUI node."""
    native = bundle.native
    wav, sample_rate, reference_text = read_voice(voice_name)
    wav = native.trim_silence_edges(wav, sample_rate, -42.0)
    max_samples = int(max_reference_seconds * sample_rate)
    if wav.numel() > max_samples:
        wav = wav[:max_samples].contiguous()
    raw_codes = bundle.codec.encode_reference(wav, sample_rate)
    return native.apply_delay_pattern(raw_codes), reference_text
