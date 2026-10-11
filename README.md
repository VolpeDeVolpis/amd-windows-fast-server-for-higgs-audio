# Fast Higgs Audio v3 server for AMD GPUs (ROCm on Windows), no ComfyUI

> **Credit.** This project runs **Higgs Audio v3** by **[Boson AI](https://www.boson.ai/higgs-audio)**. It is an unofficial, independent tool, *Derived from Higgs TTS 3, licensed from Boson AI USA, Inc.* It contains **no model weights**. See [Licensing](#licensing-read-this-first) and [NOTICE](NOTICE).
>
> Italiano: [README.it.md](README.it.md)

A small local HTTP server (`POST /tts` -> WAV) that generates speech with Higgs Audio v3 on an **AMD Radeon GPU under Windows with ROCm**, much faster than the same model inside ComfyUI. Everything runs on the GPU, nothing on the CPU.

## Measured results

Same text (74 s of speech), same cloned voice, same card (AMD Radeon RX 7900 XTX, 24 GB), Windows 10, 11 Oct 2026:

| System | Time | Speed-up |
|---|---|---|
| ComfyUI (TTS-Audio-Suite node) | 446 s | 1x |
| This server, torch engine, whole paragraphs (clean GPU) | 129 s | 3.5x |
| **This server, llama.cpp engine (default)** | **17 s** (server start ~32 s, then it stops itself) | **~26x** |

Other measurements (same machine):
- Per generation step: 13 ms with 1 text, 18 ms with 8, 28 ms with 12 texts at once (torch engine: 200-330 ms).
- The "brain" (Qwen3 part) converted to GGUF matches the original model at 99.999 %.
- Audio decoding with MIOpen disabled: 0.2 s instead of 9.6 s for 30 s of speech, output identical (max difference 0.00001).
- Video memory peak 14.7 GB (llama.cpp engine), shared memory +0.2 GB (no spill into system RAM).
- Torch engine total time varies 129-203 s depending on CPU load (it uses one full CPU core).

These numbers come from one machine and one text; yours will differ.

## Why ComfyUI was slow (measured)
1. **A thousand tiny GPU commands per step.** Each generation step sends ~1000 small commands to the GPU and on Windows each has a fixed cost, so the GPU looked idle (0-2 %). Fix: generate **several paragraphs together in the same step**. With 4 texts the step cost grows only 15 %; with 16 the throughput is 7 times higher.
2. **Final conversion to audio.** AMD's MIOpen library on Windows picks extremely slow fallback algorithms. Disabling it (`torch.backends.cudnn.enabled = False`) gives the same audio much faster. *This trick can also be applied to ComfyUI.*
3. Conversion gets much slower on long pieces, so it is done one piece at a time.
4. Moving the Qwen3 "brain" to llama.cpp (one big command per step instead of ~1000) gives the final speed-up on top.

## How it works
- Receives text, keeps paragraphs whole (cuts only at sentence ends above 500 characters), generates up to 12 pieces in one batch, joins them with a 0.25 s pause, cleans the seams (noise tails before pauses, clicks at piece edges).
- The transformer runs in llama.cpp; token sampling and the audio codec stay in torch on the GPU.
- `src/higgs_client.py` can start the server on demand: it first checks free video memory (if it cannot be measured it refuses to start); the server stops itself after 2 idle minutes.
- API: `POST /tts` `{"text","voice","seed"}` -> WAV; `GET /voices`; `GET /health`. Listens on `127.0.0.1:8090` only.

## Requirements
- Windows 10 (tested; Windows 11 not tested), AMD Radeon GPU supported by ROCm for Windows, **>= 18 GB free video memory** recommended (tested on 24 GB; peak ~15 GB). Other cards not tested.
- Python 3.12 with a **ROCm build of PyTorch for Windows** (tested: torch 2.12.0+rocm7.15.0a20260709, torchaudio 2.11.0, transformers 5.13.0) plus `numpy`, `soundfile`, `safetensors`, `tokenizers`, `gguf`.
- The Higgs Audio v3 model files from Boson (download below).
- A copy of `native.py` from the ComfyUI custom node **TTS-Audio-Suite** (MIT), folder `engines/higgs_audio_v3/` (we import it, we do not ship it).
- Fast engine only: the official **llama.cpp HIP (ROCm) Windows release** at commit `e85e15cf6` (the ctypes bridge is valid only for that version) and **LM Studio** installed with its ROCm runtime (supplies `libhipblas.dll`). Without these, use `--engine torch` (3.5x instead of ~26x).

## Installation, step by step
1. Install a ROCm PyTorch for Windows in a Python 3.12 environment, then `pip install numpy soundfile safetensors tokenizers gguf`.
2. **Download the model yourself** from the official page: <https://huggingface.co/bosonai/higgs-audio-v3-tts-4b> (accept Boson's license there, e.g. `huggingface-cli download bosonai/higgs-audio-v3-tts-4b --local-dir <folder>`). Do **not** re-upload it anywhere. Product page: <https://www.boson.ai/higgs-audio>.
3. Get the TTS-Audio-Suite ComfyUI node (clone or download it); you need `engines/higgs_audio_v3/native.py` and the package files next to it.
4. Put your own reference voices in `voices/` (`NAME.wav` + optional `NAME.txt` with its transcript). Only voices you have rights and consent for.
5. Set environment variables (PowerShell or system settings):
   ```powershell
   $env:HIGGS_PYTHON      = "C:\path\to\python.exe"
   $env:HIGGS_MODEL_DIR   = "C:\path\to\higgs-audio-v3-tts-4b"
   $env:HIGGS_NATIVE_PATH = "C:\path\to\TTS-Audio-Suite\engines\higgs_audio_v3\native.py"
   $env:HIGGS_VOICES_DIR  = "C:\path\to\voices"          # optional, default .\voices
   $env:LLAMA_CPP_DIR     = "C:\path\to\llama.cpp-hip"   # fast engine only
   ```
6. Fast engine only: `powershell scripts\make_bridge_dll.ps1` (copies a DLL from *your* LM Studio), then convert the model locally:
   `python src\llama_engine\convert_backbone.py <HIGGS_MODEL_DIR> src\llama_engine\models\higgs_backbone_bf16.gguf`
   (8 GB; keep it local, it is a derivative of Boson's weights).
7. Check video memory is free: `powershell scripts\check_vram.ps1`.
8. Start: `scripts\start_server.bat` (options: `--engine torch` for the slower mode without llama.cpp, `--port`, `--batch`, `--idle-shutdown 0` to never stop).
9. Test: `python tools\send_test.py text.txt YourVoice out.wav`.

## Scripts
`scripts/start_server.bat` start; `scripts/check_vram.ps1` dedicated/shared video memory; `scripts/run_with_turn.ps1` runs one heavy job with lock file, free-memory check and shared-memory watchdog; `scripts/make_bridge_dll.ps1`; `tools/verifica_giunture.py` measures seam artefacts; `tools/bench_*.py`, `profile_step.py`, `snr_codec.py` benchmarks; `tests/` CPU-only unit tests (`python -m pytest tests`).

## Known limits
- Tested on one machine (RX 7900 XTX, Windows). Other AMD cards and drivers unknown.
- The llama.cpp bridge depends on one exact llama.cpp commit and on a DLL taken from LM Studio; updating llama.cpp requires updating the struct definitions (the server stops with a clear message).
- The server uses one full CPU core; heavy CPU load slows it.
- Paragraphs above 500 characters are cut at sentence ends (Higgs stops by itself beyond that); cutting at commas caused audible pitch jumps and was dropped.
- HIP graphs were slower on Windows+AMD and once froze a process; fp16/bf16 codec breaks or alters the sound; SGLang/vLLM are Linux/NVIDIA only.
- Some code comments and messages are still in Italian.

## Licensing (read this first)
- **Higgs Audio v3 is not open source.** It is under the *Boson Higgs TTS 3 Research and Non-Commercial License*: free for research and personal/non-commercial use; **free for digital creators to publish, even monetized, content made with it, provided they credit Boson** (Section II-A); everything else (hosting it as an API/service, embedding it in a product for third parties, redistributing it or derivatives, resale) needs a commercial license from Boson (<https://boson.ai>). Also forbidden: voice cloning or impersonation without consent, deception, using outputs to train other generative models. Read the full text on the model page.
- **Attribution for audio/videos you publish** (required): *"This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio"*, in the audio or clearly visible in the description.
- Boson's older `higgs-audio` GitHub repository is Apache-2.0 and is not used here; its README says v3 is a standalone release. The v3 weights use the license above.
- This repository's own code license: see `LICENSE` (MIT, © 2026 Volpe). Third-party credits: [THIRD_PARTY.md](THIRD_PARTY.md). The code license does **not** change the model license.
- Not affiliated with or endorsed by Boson AI, AMD, LM Studio or ComfyUI.

## How this was made

This project was directed and tested by **Volpe** (idea, requirements, listening tests, hardware) and written by **Claude** (Anthropic's AI assistant, via Claude Code): the analysis of why ComfyUI was slow, the server code, the measurements and this documentation. Every change in the history is co-signed by Claude.
