# Dipendenze: higgs_loader.py (HiggsBundle), batch_sampler.py. Usato da: bench_fast.py, server.py.
# Motore veloce Higgs v3: memoria dei passi precedenti a dimensione fissa, passo di generazione "registrato"
# una volta sola sulla scheda (HIP/CUDA graph) e ripetuto, piu' testi generati insieme nello stesso passo.
from __future__ import annotations

import torch
import torch.nn.functional as F

from batch_sampler import BatchSamplerState


class FastHiggsEngine:
    def __init__(self, bundle, batch_size: int, max_cache_len: int, use_graph: bool = True):
        self.bundle = bundle
        self.model = bundle.model
        self.backbone = bundle.model.backbone.model
        cfg = self.backbone.config
        self.num_layers = cfg.num_hidden_layers
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.hidden_size = cfg.hidden_size
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = bundle.device
        self.dtype = bundle.dtype
        self.k_cache, self.v_cache = [], []  # allocated per request batch, sized to what it needs (allocate_cache)
        self.cache_positions = torch.arange(max_cache_len, device=self.device)
        self.static_embeds = torch.zeros((batch_size, 1, self.hidden_size), device=self.device, dtype=self.dtype)
        self.static_pos = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        self.static_logits = None
        self.all_slots = torch.arange(batch_size, device=self.device)
        rotary = self.backbone.rotary_emb
        self.inv_freq = rotary.inv_freq.to(device=self.device, dtype=torch.float32)
        self.rope_scaling = float(getattr(rotary, "attention_scaling", 1.0))
        self.graph = None
        self.use_graph = use_graph

    def allocate_cache(self, slots: int, length: int) -> None:
        """Memory of past steps only for this batch (slots x length), so video memory follows the real request."""
        shape = (slots, self.num_kv_heads, length, self.head_dim)
        self.k_cache = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.v_cache = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]

    def release_cache(self) -> None:
        """Gives the batch memory back to the card, so other programs can use it between requests."""
        self.k_cache, self.v_cache = [], []
        torch.cuda.empty_cache()

    # ---- forward of the Qwen3 backbone on the fixed cache ----
    def _forward(self, embeds: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
                 key_limit: int | None = None) -> torch.Tensor:
        """embeds [b, T, H]; positions [b, T]; slots [b] = cache rows; key_limit = cache entries to attend.
        Returns last hidden [b, H]."""
        b, t, _ = embeds.shape
        limit = self.max_cache_len if key_limit is None else key_limit
        freqs = positions.unsqueeze(-1).float() * self.inv_freq  # [b, T, head_dim/2]
        angles = torch.cat((freqs, freqs), dim=-1)
        cos = (angles.cos() * self.rope_scaling).to(self.dtype).unsqueeze(1)
        sin = (angles.sin() * self.rope_scaling).to(self.dtype).unsqueeze(1)
        # mask [b, 1, T, L]: a token sees cache entries up to its own position
        mask = self.cache_positions[:limit].view(1, 1, 1, -1) <= positions.view(b, 1, t, 1)
        hidden = embeds
        row_index = slots.view(b, 1).expand(b, t)
        for layer_index, layer in enumerate(self.backbone.layers):
            attn = layer.self_attn
            residual = hidden
            x = layer.input_layernorm(hidden)
            q = attn.q_norm(attn.q_proj(x).view(b, t, self.num_heads, self.head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            v = attn.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            k_cache, v_cache = self.k_cache[layer_index], self.v_cache[layer_index]
            # write new keys/values at [slot, :, position]
            k_cache[row_index, :, positions] = k.transpose(1, 2)
            v_cache[row_index, :, positions] = v.transpose(1, 2)
            keys = k_cache[:, :, :limit].index_select(0, slots)
            values = v_cache[:, :, :limit].index_select(0, slots)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True)
            out = attn.o_proj(out.transpose(1, 2).reshape(b, t, -1))
            hidden = residual + out
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.backbone.norm(hidden[:, -1, :])

    def _decode_step(self) -> torch.Tensor:
        hidden = self._forward(self.static_embeds, self.static_pos.view(-1, 1), self.all_slots)
        return self.model.modality_head.generate(hidden).to(torch.float32)

    def _capture(self):
        self.static_logits = self._decode_step()  # warm-up outside capture
        if not self.use_graph:
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.static_logits = self._decode_step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self._decode_step()

    def _run_step(self) -> torch.Tensor:
        if self.graph is not None:
            self.graph.replay()
            return self.static_logits
        return self._decode_step()

    @torch.inference_mode()
    def generate(self, requests, *, max_new_tokens: int, temperature: float, top_p, top_k, check_every: int = 16):
        """requests: list of (prompt_ids, reference_codes_delayed). Returns list of delayed codes [L, N]."""
        if len(requests) > self.batch_size:
            raise ValueError("more requests than batch slots")
        prompt_embeds = [self.model._prompt_embeds(ids, ref, self.device).to(self.dtype) for ids, ref in requests]
        lengths = [e.shape[1] for e in prompt_embeds]
        if max(lengths) + max_new_tokens > self.max_cache_len:
            raise ValueError(f"cache too short: need {max(lengths) + max_new_tokens}, have {self.max_cache_len}")
        active = len(requests)  # only occupied slots take part in each step
        self.allocate_cache(active, max(lengths) + max_new_tokens + 1)
        try:
            return self._generate_allocated(prompt_embeds, lengths, active, max_new_tokens, temperature, top_p, top_k,
                                            check_every)
        finally:
            self.release_cache()

    def _generate_allocated(self, prompt_embeds, lengths, active, max_new_tokens, temperature, top_p, top_k, check_every):
        slots = self.all_slots[:active]
        last_hidden = torch.zeros((active, self.hidden_size), device=self.device, dtype=self.dtype)
        for slot, embeds in enumerate(prompt_embeds):
            positions = torch.arange(embeds.shape[1], device=self.device).view(1, -1)
            last_hidden[slot] = self._forward(embeds, positions, slots[slot:slot + 1], embeds.shape[1])[0]
        n = self.model.num_codebooks
        state = BatchSamplerState(active, n, self.device)
        codes_out = torch.zeros((max_new_tokens, active, n), dtype=torch.long, device=self.device)
        valid_out = torch.zeros((max_new_tokens, active), dtype=torch.bool, device=self.device)
        logits = self.model.modality_head.generate(last_hidden).to(torch.float32)
        positions = torch.tensor(lengths, device=self.device)
        key_limit = max(lengths)  # cache entries written so far (upper bound over the batch)
        steps_done = 0
        for step in range(max_new_tokens):
            codes, valid = state.step(step, logits, temperature, top_p, top_k)
            codes_out[step] = codes
            valid_out[step] = valid
            steps_done = step + 1
            if (step + 1) % check_every == 0 and bool(state.done.all()):
                break
            embeds = self.model.modality_embedding(codes).view(active, 1, -1)
            key_limit += 1
            hidden = self._forward(embeds, positions.view(-1, 1), slots, key_limit)
            logits = self.model.modality_head.generate(hidden).to(torch.float32)
            positions = positions + 1
        self.last_steps = steps_done
        results = []
        for slot in range(active):
            rows = codes_out[:steps_done, slot][valid_out[:steps_done, slot]]
            results.append(rows.cpu())
        return results


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
