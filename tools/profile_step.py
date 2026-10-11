# Dipendenze: higgs_loader.py, fast_engine.py, batch_sampler.py. Misura il costo di ogni pezzo di un passo.
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from batch_sampler import sample_tokens
from fast_engine import FastHiggsEngine
from higgs_loader import load_bundle


def timed(label, fn, repeats):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    print(f"{label}: {1000 * (time.perf_counter() - start) / repeats:.2f} ms")


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    use_graph = "--graph" in sys.argv
    bundle = load_bundle()
    engine = FastHiggsEngine(bundle, batch, 3000, use_graph=use_graph)
    engine.static_pos.fill_(1000)
    with torch.inference_mode():
        engine._capture()
        timed(f"step batch={batch} graph={use_graph}", engine._run_step, 20)
        x = torch.randn(batch * 8, 1026, device=bundle.device)
        timed("sampler top_p", lambda: sample_tokens(x, 1.0, 0.95, None), 50)
        hidden = torch.randn(batch, engine.hidden_size, device=bundle.device, dtype=bundle.dtype)
        layer = engine.backbone.layers[0]
        h3 = hidden.view(batch, 1, -1)
        timed("one mlp", lambda: layer.mlp(h3), 50)
        timed("one q_proj", lambda: layer.self_attn.q_proj(h3), 200)
        w = layer.mlp.gate_proj.weight
        timed("matmul 9728x2560 bf16", lambda: torch.nn.functional.linear(h3, w), 200)
        wf = w.float()
        timed("matmul 9728x2560 fp32", lambda: torch.nn.functional.linear(h3.float(), wf), 200)
        wh = w.half()
        timed("matmul 9728x2560 fp16", lambda: torch.nn.functional.linear(h3.half(), wh), 200)
        cache_k = engine.k_cache[0]
        q = torch.randn(batch, engine.num_heads, 1, engine.head_dim, device=bundle.device, dtype=bundle.dtype)
        mask = engine.cache_positions.view(1, 1, 1, -1) <= engine.static_pos.view(batch, 1, 1, 1)
        timed("sdpa bool mask gqa", lambda: torch.nn.functional.scaled_dot_product_attention(
            q, cache_k, engine.v_cache[0], attn_mask=mask, enable_gqa=True), 50)
        rep = cache_k.repeat_interleave(engine.num_heads // engine.num_kv_heads, dim=1)
        timed("sdpa bool mask repeated", lambda: torch.nn.functional.scaled_dot_product_attention(
            q, rep, rep, attn_mask=mask), 50)
        float_mask = torch.zeros(mask.shape, device=bundle.device, dtype=bundle.dtype).masked_fill(~mask, float("-inf"))
        timed("sdpa float mask repeated", lambda: torch.nn.functional.scaled_dot_product_attention(
            q, rep, rep, attn_mask=float_mask), 50)

        def manual_attention():
            qg = q.view(batch, engine.num_kv_heads, -1, engine.head_dim)
            scores = torch.matmul(qg, cache_k.transpose(-1, -2)) * engine.head_dim ** -0.5
            scores = scores + float_mask
            return torch.matmul(scores.softmax(dim=-1), engine.v_cache[0])
        timed("manual attention", manual_attention, 50)
        kv_new = torch.randn(batch, 1, engine.num_kv_heads, engine.head_dim, device=bundle.device, dtype=bundle.dtype)
        rows = engine.all_slots.view(batch, 1)
        pos = engine.static_pos.view(batch, 1)

        def write_kv():
            cache_k[rows, :, pos] = kv_new
        timed("kv write index_put", write_kv, 50)
        timed("index_select cache", lambda: cache_k.index_select(0, engine.all_slots), 50)
        a = torch.randn(1024, device=bundle.device)
        timed("tiny elementwise", lambda: a.mul(2), 500)
    print(f"vram_alloc_gb={torch.cuda.memory_allocated() / 1e9:.1f}")


if __name__ == "__main__":
    main()
