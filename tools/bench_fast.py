# Dipendenze: higgs_loader.py, fast_engine.py. Misura il motore veloce con 1..N testi insieme.
import argparse
import os
import sys
import time

import soundfile
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fast_engine import FastHiggsEngine
from higgs_loader import encode_voice, load_bundle


def check_against_original(bundle, engine, prompt_ids, ref_delayed):
    """Prefill with the original backbone and with the fast engine: hidden states must match."""
    embeds = bundle.model._prompt_embeds(prompt_ids, ref_delayed, bundle.device)
    reference = bundle.model.backbone.model(inputs_embeds=embeds).last_hidden_state[0, -1].float()
    positions = torch.arange(embeds.shape[1], device=bundle.device).view(1, -1)
    mine = engine._forward(embeds, positions, torch.tensor([0], device=bundle.device))[0].float()
    cosine = torch.nn.functional.cosine_similarity(reference, mine, dim=0).item()
    print(f"check_cosine={cosine:.5f} max_abs_diff={(reference - mine).abs().max().item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    text = open(args.text_file, encoding="utf-8-sig").read().strip()

    bundle = load_bundle()
    ref_delayed, ref_text = encode_voice(bundle, args.voice)
    prompt_ids = bundle.tokenizer.build_prompt(text, num_ref_tokens=int(ref_delayed.shape[0]), reference_text=ref_text or None)
    engine = FastHiggsEngine(bundle, args.batch, len(prompt_ids) + args.max_new_tokens + 8, use_graph=not args.no_graph)
    with torch.inference_mode():
        check_against_original(bundle, engine, prompt_ids, ref_delayed)
    requests = [(prompt_ids, ref_delayed)] * args.batch
    for round_index in range(args.rounds):
        torch.manual_seed(args.seed + round_index)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        delayed_list = engine.generate(requests, max_new_tokens=args.max_new_tokens, temperature=1.0, top_p=0.95, top_k=None)
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        audio_seconds = 0.0
        for slot, delayed in enumerate(delayed_list):
            with torch.inference_mode():
                audio = bundle.codec.decode(bundle.native.reverse_delay_pattern(delayed))
            audio_seconds += audio.numel() / bundle.native.SAMPLE_RATE
            soundfile.write(f"{args.out_prefix}_b{args.batch}_r{round_index}_{slot}.wav", audio.numpy(), bundle.native.SAMPLE_RATE)
        decode_s = time.perf_counter() - t1
        steps = max(d.shape[0] for d in delayed_list)
        print(f"round={round_index} batch={args.batch} graph={not args.no_graph} gen_s={gen_s:.1f} "
              f"ms_per_step={1000 * gen_s / steps:.1f} decode_s={decode_s:.1f} audio_s_total={audio_seconds:.1f} "
              f"audio_s_per_wall_s={audio_seconds / (gen_s + decode_s):.2f} "
              f"lengths={[d.shape[0] for d in delayed_list]} vram_peak_gb={torch.cuda.max_memory_allocated() / 1e9:.1f}")


if __name__ == "__main__":
    main()
