# Dipendenze: higgs_loader.py. Misura il tempo del ciclo originale del nodo (senza ComfyUI), suddiviso per fase.
import argparse
import sys
import time

import torch
import soundfile

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
from higgs_loader import encode_voice, load_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    text = open(args.text_file, encoding="utf-8-sig").read().strip()

    t0 = time.perf_counter()
    bundle = load_bundle()
    torch.cuda.synchronize()
    print(f"load_s={time.perf_counter() - t0:.1f} vram_gb={torch.cuda.memory_allocated() / 1e9:.2f}")

    t0 = time.perf_counter()
    ref_delayed, ref_text = encode_voice(bundle, args.voice)
    print(f"voice_encode_s={time.perf_counter() - t0:.2f} ref_frames={ref_delayed.shape[0]}")

    native = bundle.native
    backbone = bundle.model.backbone.model
    step_times = []
    original_forward = backbone.forward

    def timed_forward(*a, **k):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_forward(*a, **k)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start)
        return result

    backbone.forward = timed_forward
    torch.manual_seed(args.seed)
    prompt_ids = bundle.tokenizer.build_prompt(text, num_ref_tokens=int(ref_delayed.shape[0]), reference_text=ref_text or None)
    t0 = time.perf_counter()
    with torch.inference_mode():
        delayed = bundle.model.generate_codes(prompt_ids, ref_delayed, max_new_tokens=args.max_new_tokens,
                                              temperature=1.0, top_p=0.95, top_k=None)
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        audio = bundle.codec.decode(native.reverse_delay_pattern(delayed))
        decode_s = time.perf_counter() - t1
    seconds_audio = audio.numel() / native.SAMPLE_RATE
    forward_total = sum(step_times[1:])
    steps = len(step_times) - 1
    print(f"prefill_s={step_times[0]:.2f} steps={steps} gen_s={gen_s:.1f} forward_s={forward_total:.1f} "
          f"ms_per_step_forward={1000 * forward_total / max(steps, 1):.1f} "
          f"other_s={gen_s - forward_total - step_times[0]:.1f} decode_s={decode_s:.1f} "
          f"audio_s={seconds_audio:.1f} rtf={(gen_s + decode_s) / seconds_audio:.2f}")
    soundfile.write(args.out, audio.numpy(), native.SAMPLE_RATE)


if __name__ == "__main__":
    main()
