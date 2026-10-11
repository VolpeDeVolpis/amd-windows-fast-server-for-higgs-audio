# Dipendenze: llama.dll of the official llama.cpp HIP release (folder LLAMA_CPP_DIR, commit e85e15cf6, strutture copiate da
# llama.h of that commit); the ROCm libraries it lacks come from the LM Studio "vendor" folder, with hipblas.dll in bridge_dll
# (copy of libhipblas.dll under the expected name). LM Studio's own engine (a fork, non-public API) crashes at the end of model loading.
# numpy. Usato da: llama_probe.py, llama_higgs_engine.py.
# Ponte minimo verso llama.cpp: carica il "cervello" GGUF e fa un passo con ingressi a vettori (embedding) per piu'
# sequenze insieme, restituendo lo stato finale (dopo la norma) di ognuna.
from __future__ import annotations

import ctypes as C
import os
import subprocess
from pathlib import Path

import numpy as np

LMSTUDIO_BACKENDS = Path(os.environ.get("USERPROFILE", "")) / ".lmstudio" / "extensions" / "backends"
LLAMA_DIR = Path(os.environ.get("LLAMA_CPP_DIR", ""))  # folder of the official llama.cpp HIP (ROCm) Windows release
BRIDGE_DLL_DIR = Path(__file__).resolve().parent / "bridge_dll"
EXPECTED_COMMIT = "e85e15cf6"  # le strutture qui sotto valgono SOLO per questa versione di llama.cpp


def _find_vendor() -> Path:
    vendors = sorted(LMSTUDIO_BACKENDS.glob("vendor/win-llama-rocm-vendor-*/bin"))
    if not vendors:
        raise RuntimeError("Non trovo le librerie ROCm di LM Studio (servono a llama.cpp per usare la scheda AMD).")
    return vendors[-1]


VENDOR_DIR = _find_vendor()


class ModelParams(C.Structure):
    _fields_ = [("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p), ("n_gpu_layers", C.c_int32),
                ("split_mode", C.c_int32), ("load_mode", C.c_int32), ("lazy_mode", C.c_int32), ("main_gpu", C.c_int32),
                ("tensor_split", C.c_void_p), ("progress_callback", C.c_void_p),
                ("progress_callback_user_data", C.c_void_p), ("kv_overrides", C.c_void_p),
                ("vocab_only", C.c_bool), ("check_tensors", C.c_bool), ("use_extra_bufts", C.c_bool),
                ("no_host", C.c_bool), ("no_alloc", C.c_bool), ("load_mtp", C.c_bool)]


class ContextParams(C.Structure):
    _fields_ = [("n_ctx", C.c_uint32), ("n_batch", C.c_uint32), ("n_ubatch", C.c_uint32), ("n_seq_max", C.c_uint32),
                ("n_rs_seq", C.c_uint32), ("n_outputs_max", C.c_uint32), ("n_outputs_max_per_seq", C.c_uint32),
                ("n_threads", C.c_int32), ("n_threads_batch", C.c_int32),
                ("ctx_type", C.c_int32), ("rope_scaling_type", C.c_int32), ("pooling_type", C.c_int32),
                ("attention_type", C.c_int32), ("flash_attn_type", C.c_int32),
                ("rope_freq_base", C.c_float), ("rope_freq_scale", C.c_float), ("yarn_ext_factor", C.c_float),
                ("yarn_attn_factor", C.c_float), ("yarn_beta_fast", C.c_float), ("yarn_beta_slow", C.c_float),
                ("yarn_orig_ctx", C.c_uint32), ("defrag_thold", C.c_float),
                ("cb_eval", C.c_void_p), ("cb_eval_user_data", C.c_void_p),
                ("type_k", C.c_int32), ("type_v", C.c_int32),
                ("abort_callback", C.c_void_p), ("abort_callback_data", C.c_void_p),
                ("embeddings", C.c_bool), ("offload_kqv", C.c_bool), ("no_perf", C.c_bool), ("op_offload", C.c_bool),
                ("swa_full", C.c_bool), ("kv_unified", C.c_bool),
                ("samplers", C.c_void_p), ("n_samplers", C.c_size_t), ("ctx_other", C.c_void_p)]


class Batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", C.c_void_p), ("embd", C.POINTER(C.c_float)),
                ("pos", C.POINTER(C.c_int32)), ("n_seq_id", C.POINTER(C.c_int32)),
                ("seq_id", C.POINTER(C.POINTER(C.c_int32))), ("logits", C.POINTER(C.c_int8))]


def _check_version():
    output = subprocess.run([str(LLAMA_DIR / "llama-cli.exe"), "--version"], capture_output=True, text=True,
                            timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = output.stdout + output.stderr
    if EXPECTED_COMMIT not in text:
        raise RuntimeError(f"llama.cpp in {LLAMA_DIR} non e' la versione attesa ({text.strip()[:80]}): il ponte va "
                           f"aggiornato con il llama.h di quella versione.")


def _load_library():
    _check_version()
    if not (BRIDGE_DLL_DIR / "hipblas.dll").is_file():
        raise RuntimeError(f"Missing {BRIDGE_DLL_DIR}\\hipblas.dll (copy of LM Studio's libhipblas.dll: run scripts/make_bridge_dll.ps1).")
    for folder in (BRIDGE_DLL_DIR, VENDOR_DIR, LLAMA_DIR):
        os.add_dll_directory(str(folder))
    os.environ["PATH"] = f"{BRIDGE_DLL_DIR};{VENDOR_DIR};{LLAMA_DIR};{os.environ.get('PATH', '')}"
    ggml = C.CDLL(str(LLAMA_DIR / "ggml.dll"))
    ggml.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path(str(LLAMA_DIR).encode("utf-8"))  # motori di calcolo (HIP) della release
    global _GGML
    _GGML = ggml
    lib = C.CDLL(str(LLAMA_DIR / "llama.dll"))
    lib.llama_backend_init.restype = None
    lib.llama_model_default_params.restype = ModelParams
    lib.llama_context_default_params.restype = ContextParams
    lib.llama_model_load_from_file.argtypes = [C.c_char_p, ModelParams]
    lib.llama_model_load_from_file.restype = C.c_void_p
    lib.llama_init_from_model.argtypes = [C.c_void_p, ContextParams]
    lib.llama_init_from_model.restype = C.c_void_p
    lib.llama_decode.argtypes = [C.c_void_p, Batch]
    lib.llama_decode.restype = C.c_int32
    lib.llama_get_embeddings_ith.argtypes = [C.c_void_p, C.c_int32]
    lib.llama_get_embeddings_ith.restype = C.POINTER(C.c_float)
    lib.llama_set_embeddings.argtypes = [C.c_void_p, C.c_bool]
    lib.llama_set_embeddings.restype = None
    lib.llama_free.argtypes = [C.c_void_p]
    lib.llama_model_free.argtypes = [C.c_void_p]
    lib.llama_model_n_embd.argtypes = [C.c_void_p]
    lib.llama_model_n_embd.restype = C.c_int32
    lib.llama_backend_init()
    return lib


class LlamaBackbone:
    """The Qwen3 part of Higgs inside llama.cpp, on the GPU. One context per request batch (memory follows the request)."""

    def __init__(self, gguf_path: Path):
        self.lib = _load_library()
        params = self.lib.llama_model_default_params()
        params.n_gpu_layers = -1  # tutto sulla scheda video
        params.load_mode = 0      # lettura diretta, niente mappatura del file in memoria del PC
        params.no_host = True     # niente memoria del PC "agganciata" alla scheda (conta come memoria condivisa)
        params.use_extra_bufts = False  # niente zone "riorganizzate" del processore (fanno cadere il caricamento da Python)
        self.model = self.lib.llama_model_load_from_file(str(gguf_path).encode("utf-8"), params)
        if not self.model:
            raise RuntimeError(f"llama.cpp non riesce a caricare {gguf_path}")
        self.n_embd = self.lib.llama_model_n_embd(self.model)
        self.ctx = None

    def open(self, sequences: int, tokens_per_sequence: int, max_batch_tokens: int):
        self.close()
        params = self.lib.llama_context_default_params()
        params.n_ctx = sequences * tokens_per_sequence
        params.n_batch = params.n_ubatch = max_batch_tokens
        params.n_seq_max = sequences
        # uscite massime per passo = una per sequenza: la riserva in RAM (150 mila parole x uscite) resta piccola
        params.n_outputs_max = sequences
        params.embeddings = True
        params.pooling_type = 0  # NONE: uno stato per ogni posizione richiesta
        params.flash_attn_type = 1
        params.kv_unified = True
        self.ctx = self.lib.llama_init_from_model(self.model, params)
        if not self.ctx:
            raise RuntimeError("llama.cpp non riesce a creare il contesto (memoria video?)")

    def close(self):
        if self.ctx:
            self.lib.llama_free(self.ctx)
            self.ctx = None

    def prefill(self, embeds: np.ndarray, seq_id: int) -> np.ndarray:
        """Whole prompt of one sequence; returns the final state of its last position [n_embd].
        In embedding mode llama.cpp outputs every position, so the prompt goes in without outputs first
        (embeddings off), then the last position alone with its output."""
        count = embeds.shape[0]
        if count > 1:
            self.lib.llama_set_embeddings(self.ctx, False)
            self.step(embeds[:-1], np.arange(count - 1), np.full(count - 1, seq_id), np.zeros(count - 1, dtype=bool))
            self.lib.llama_set_embeddings(self.ctx, True)
        return self.step(embeds[-1:], np.array([count - 1]), np.array([seq_id]), np.ones(1, dtype=bool))[0]

    def step(self, embeds: np.ndarray, positions: np.ndarray, seq_ids: np.ndarray, want_output: np.ndarray) -> np.ndarray:
        """embeds [T, n_embd] float32; positions/seq_ids [T] int32; want_output [T] bool.
        Returns the final hidden states [n_outputs, n_embd] for the rows with want_output, in order."""
        count = embeds.shape[0]
        embeds = np.ascontiguousarray(embeds, dtype=np.float32)
        positions = np.ascontiguousarray(positions, dtype=np.int32)
        seq_values = np.ascontiguousarray(seq_ids, dtype=np.int32)
        n_seq = np.ones(count, dtype=np.int32)
        seq_pointers = (C.POINTER(C.c_int32) * count)(
            *[C.cast(seq_values.ctypes.data + 4 * i, C.POINTER(C.c_int32)) for i in range(count)])
        logits = np.ascontiguousarray(want_output, dtype=np.int8)
        batch = Batch(count, None, embeds.ctypes.data_as(C.POINTER(C.c_float)),
                      positions.ctypes.data_as(C.POINTER(C.c_int32)), n_seq.ctypes.data_as(C.POINTER(C.c_int32)),
                      C.cast(seq_pointers, C.POINTER(C.POINTER(C.c_int32))), logits.ctypes.data_as(C.POINTER(C.c_int8)))
        result = self.lib.llama_decode(self.ctx, batch)
        if result != 0:
            raise RuntimeError(f"llama_decode ha restituito {result}")
        rows = np.flatnonzero(logits)
        out = np.empty((len(rows), self.n_embd), dtype=np.float32)
        for k, row in enumerate(rows):
            pointer = self.lib.llama_get_embeddings_ith(self.ctx, int(row))
            out[k] = np.ctypeslib.as_array(pointer, shape=(self.n_embd,))
        return out

    def free(self):
        self.close()
        if self.model:
            self.lib.llama_model_free(self.model)
            self.model = None
