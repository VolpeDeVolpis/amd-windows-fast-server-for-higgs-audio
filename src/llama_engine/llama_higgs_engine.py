# Dipendenze: llama_bridge.py (cervello Qwen3 in llama.cpp), batch_sampler.py, HiggsBundle di higgs_loader.py (parti audio
# in torch sulla scheda). Usato da: server.py (--engine llama), bench_llama_engine.py.
# Stessa interfaccia di FastHiggsEngine.generate: piu' testi insieme, ogni passo = un solo comando a llama.cpp.
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from batch_sampler import BatchSamplerState
from llama_bridge import LlamaBackbone

DEFAULT_GGUF = Path(__file__).resolve().parent / "models" / "higgs_backbone_bf16.gguf"


class LlamaHiggsEngine:
    def __init__(self, bundle, batch_size: int, max_cache_len: int, gguf_path: Path = DEFAULT_GGUF):
        self.bundle = bundle
        self.model = bundle.model
        self.device = bundle.device
        self.dtype = bundle.dtype
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        # il cervello torch non serve piu': si liberano i suoi strati (8 GB) prima di caricare quello llama.cpp
        self.model.backbone.model.layers = torch.nn.ModuleList()
        torch.cuda.empty_cache()
        self.backbone = LlamaBackbone(gguf_path)
        self.last_steps = 0

    def _hidden_to_torch(self, hidden: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(hidden).to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def generate(self, requests, *, max_new_tokens: int, temperature: float, top_p, top_k, check_every: int = 16):
        """requests: list of (prompt_ids, reference_codes_delayed). Returns list of delayed codes [L, N] (cpu)."""
        if len(requests) > self.batch_size:
            raise ValueError("more requests than batch slots")
        prompts = [self.model._prompt_embeds(ids, ref, self.device)[0].float().cpu().numpy() for ids, ref in requests]
        lengths = [p.shape[0] for p in prompts]
        if max(lengths) + max_new_tokens > self.max_cache_len:
            raise ValueError(f"cache too short: need {max(lengths) + max_new_tokens}, have {self.max_cache_len}")
        active = len(requests)
        self.backbone.open(sequences=active, tokens_per_sequence=max(lengths) + max_new_tokens + 1,
                           max_batch_tokens=max(lengths))
        try:
            return self._run(prompts, lengths, active, max_new_tokens, temperature, top_p, top_k, check_every)
        finally:
            self.backbone.close()  # memoria del passato restituita alla scheda dopo ogni lotto

    def _run(self, prompts, lengths, active, max_new_tokens, temperature, top_p, top_k, check_every):
        hidden = np.stack([self.backbone.prefill(prompt, seq) for seq, prompt in enumerate(prompts)])
        n = self.model.num_codebooks
        state = BatchSamplerState(active, n, self.device)
        codes_out = torch.zeros((max_new_tokens, active, n), dtype=torch.long, device=self.device)
        valid_out = torch.zeros((max_new_tokens, active), dtype=torch.bool, device=self.device)
        positions = np.array(lengths, dtype=np.int32)
        seq_ids = np.arange(active, dtype=np.int32)
        outputs = np.ones(active, dtype=bool)
        steps_done = 0
        for step in range(max_new_tokens):
            logits = self.model.modality_head.generate(self._hidden_to_torch(hidden)).to(torch.float32)
            codes, valid = state.step(step, logits, temperature, top_p, top_k)
            codes_out[step] = codes
            valid_out[step] = valid
            steps_done = step + 1
            if (step + 1) % check_every == 0 and bool(state.done.all()):
                break
            embeds = self.model.modality_embedding(codes).float().cpu().numpy()
            hidden = self.backbone.step(embeds, positions, seq_ids, outputs)
            positions = positions + 1
        self.last_steps = steps_done
        return [codes_out[:steps_done, s][valid_out[:steps_done, s]].cpu() for s in range(active)]
