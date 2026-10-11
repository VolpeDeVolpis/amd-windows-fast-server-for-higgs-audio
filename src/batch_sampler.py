# Dipendenze: solo torch. Usato da: fast_engine.py.
# Campionamento Higgs v3 per piu' sequenze insieme, tutto sulla scheda (nessuna attesa del processore a ogni passo).
# Riproduce la logica di sampler_step del nodo: ritardo iniziale per codebook, fine dopo EOC + (n-2) passi.
from __future__ import annotations

import torch

BOC_ID = 1024
EOC_ID = 1025


def sample_tokens(logits: torch.Tensor, temperature: float, top_p: float | None, top_k: int | None) -> torch.Tensor:
    """logits [R, V] float32 -> sampled ids [R]."""
    if temperature <= 1e-5:
        return logits.argmax(dim=-1)
    logits = logits / float(temperature)
    if top_k is not None and top_k > 0:
        kth = logits.topk(min(int(top_k), logits.size(-1)), dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove).scatter_(-1, sorted_indices, remove)
        logits = torch.where(scatter, float("-inf"), logits)
    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1)


class BatchSamplerState:
    def __init__(self, batch: int, num_codebooks: int, device: torch.device):
        self.num_codebooks = num_codebooks
        self.countdown = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)

    def step(self, step_index: int, logits_b_n_v: torch.Tensor, temperature, top_p, top_k):
        """Returns (codes [B, N], valid [B]); valid = row belongs to the output of that sequence."""
        batch, n, vocab = logits_b_n_v.shape
        valid = ~self.done
        codes = sample_tokens(logits_b_n_v.reshape(batch * n, vocab), temperature, top_p, top_k).view(batch, n)
        if step_index < n:
            if step_index + 1 < n:
                codes[:, step_index + 1:] = BOC_ID
            return codes, valid
        counting = self.countdown >= 0
        self.countdown = torch.where(counting, self.countdown - 1, self.countdown)
        finished_now = counting & (self.countdown <= 0)
        starts = (~counting) & (codes[:, 0] == EOC_ID)
        start_value = max(n - 2, 0)
        self.countdown = torch.where(starts, torch.full_like(self.countdown, start_value), self.countdown)
        if start_value == 0:
            finished_now = finished_now | starts
        self.done = self.done | (finished_now & valid)
        return codes, valid
