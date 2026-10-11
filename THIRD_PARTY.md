# Third-party code and materials

Nothing below is redistributed in this repository unless stated. "Used" = you install/download it yourself.

| Component | Author / owner | License | How it is used | Blocks publishing? |
|---|---|---|---|---|
| **Higgs Audio v3 TTS weights, tokenizer, codec** (`bosonai/higgs-audio-v3-tts-4b`) | Boson AI USA, Inc. | Boson Higgs TTS 3 Research and Non-Commercial License (source-available, NOT open source; Creator Use Grant for creators) | Downloaded by the user from Hugging Face. NOT included. The GGUF produced by `convert_backbone.py` is a Derivative Work: keep it local, never upload it. | No for the code. Yes for the weights/GGUF: never publish them. |
| **`native.py` of TTS-Audio-Suite** (ComfyUI node; model builder, codec, tokenizer, delay pattern) | Shmuel Ronen and contributors | MIT | Imported at run time from the user's own copy (`HIGGS_NATIVE_PATH`). NOT included. | No |
| `batch_sampler.py` | derived from `sampler_step` logic of `native.py` above (re-implemented to run batched on the GPU) | MIT (the original); our file under this repo's license, keep the MIT notice of the original | Adapted | No (keep the credit) |
| `fast_engine.py` (torch engine) | re-implements the Qwen3 forward pass using the model modules built by `native.py`; Qwen3 architecture by Alibaba Cloud / Qwen team | Apache-2.0 (Qwen3 / Hugging Face Transformers) | Adapted design, no copied files | No |
| **llama.cpp** (`llama.dll`, `ggml*.dll`, `llama-cli.exe`) | Georgi Gerganov and ggml-org contributors | MIT | Official HIP Windows release downloaded by the user (`LLAMA_CPP_DIR`). NOT included. | No |
| `llama_bridge.py` struct definitions (`llama_model_params`, `llama_context_params`, `llama_batch`) | ggml-org | MIT | Re-typed with `ctypes` from `llama.h` of commit `e85e15cf6`; valid only for that version | No (credit kept here) |
| `gguf` Python package (used by `convert_backbone.py`) | ggml-org | MIT | `pip install gguf` | No |
| **LM Studio `libhipblas.dll` / ROCm runtime** | LM Studio / AMD | AMD ROCm components: various (MIT and others); LM Studio terms for its bundle | `scripts/make_bridge_dll.ps1` copies, on the user's PC, a DLL from the user's own LM Studio install. Never committed (`*.dll` is git-ignored). | **Do not redistribute** these DLLs. |
| PyTorch (ROCm build), torchaudio | PyTorch Foundation / Meta | BSD-3-Clause | Dependency | No |
| Hugging Face `transformers`, `tokenizers`, `safetensors` | Hugging Face | Apache-2.0 | Dependency | No |
| `numpy`, `soundfile` | NumPy developers; Bastian Bechtold | BSD-3-Clause | Dependency | No |
| ROCm for Windows PyTorch wheels | AMD | various open licenses | Dependency | No |

## Our own code
Everything else (server, text chunking, seam cleanup `audio_lib/`, memory check, tools) was written for this project and is under the license chosen in `LICENSE` (see `LICENZA_SCELTA.md`).

## What could prevent publication
1. **Boson license (the only real constraint).**
   - Never publish weights, the GGUF, the `.npy` references or any file derived from the weights.
   - Section IV(a): a repository that uses the Higgs Materials must carry the Boson attribution (file `NOTICE` and the README: done) and a copy of the license reference.
   - Section IV(a)(iv): software "derived from" Higgs must say "Derived from Higgs TTS 3, licensed from Boson AI USA, Inc." (README: done) and must NOT use "Higgs TTS 3" or a confusingly similar name as the leading element of its name. The folder name `amd-windows-fast-server-for-higgs-audio` starts with "higgs": there is a risk it is read as official or confusingly similar. Safer: name the repo something like `amd-windows-fast-server-for-higgs-audio`.
   - Section II-A(c): hosting the model as a service/API, embedding it in a product for third parties, or redistributing it needs a commercial license from Boson. This repo only helps individuals run it locally.
2. `native.py` and llama.cpp are MIT: no obstacle. Check the exact license file of the TTS-Audio-Suite copy you use before releasing.
3. Voices: never publish cloned voices or reference audio of real people (Boson license IV(b)(ii) requires explicit consent).
