# Dipendenze: pip package gguf (official llama.cpp writer), numpy. Uso: python convert_backbone.py <cartella modello Higgs> <uscita.gguf>
# Copia la parte "cervello" (Qwen3) di Higgs v3 in un file GGUF per llama.cpp, pesi bf16 copiati byte per byte (nessuna perdita).
# Le parti audio (codec, ingressi/uscite audio) restano nel modello originale e girano in torch sulla scheda.
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

import gguf  # noqa: E402

LAYER_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
TEXT_EMBEDDING = "tied.embedding.text_embedding.weight"


def read_safetensors_header(path: Path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def gguf_name(source: str) -> str | None:
    if source == TEXT_EMBEDDING:
        return "token_embd.weight"
    if source == "body.norm.weight":
        return "output_norm.weight"
    if source.startswith("body.layers."):
        index, rest = source[len("body.layers."):].split(".", 1)
        return f"blk.{index}.{LAYER_MAP[rest]}" if rest in LAYER_MAP else None
    return None


def add_vocab(writer, model_dir: Path, vocab_size: int):
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokens = ["" for _ in range(vocab_size)]
    types = [gguf.TokenType.UNUSED] * vocab_size
    for text, token_id in tokenizer["model"]["vocab"].items():
        if token_id < vocab_size:
            tokens[token_id], types[token_id] = text, gguf.TokenType.NORMAL
    for added in tokenizer.get("added_tokens", []):
        if added["id"] < vocab_size:
            tokens[added["id"]] = added["content"]
            types[added["id"]] = gguf.TokenType.CONTROL if added.get("special") else gguf.TokenType.USER_DEFINED
    for index, text in enumerate(tokens):
        if not text:
            tokens[index] = f"[PAD{index}]"
    merges = [m if isinstance(m, str) else " ".join(m) for m in tokenizer["model"].get("merges", [])]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)


def main():
    model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["text_config"]
    weights = model_dir / "model.safetensors"
    header, data_start = read_safetensors_header(weights)
    writer = gguf.GGUFWriter(str(out_path), "qwen3")
    writer.add_name("higgs-tts-3 backbone")
    writer.add_context_length(int(config.get("max_position_embeddings", 8192)))
    writer.add_embedding_length(int(config["hidden_size"]))
    writer.add_block_count(int(config["num_hidden_layers"]))
    writer.add_feed_forward_length(int(config["intermediate_size"]))
    writer.add_head_count(int(config["num_attention_heads"]))
    writer.add_head_count_kv(int(config["num_key_value_heads"]))
    head_dim = int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]))
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_freq_base(float(config.get("rope_theta", 1_000_000.0)))
    writer.add_layer_norm_rms_eps(float(config.get("rms_norm_eps", 1e-6)))
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    add_vocab(writer, model_dir, int(config["vocab_size"]))
    count = 0
    with open(weights, "rb") as handle:
        for source, info in header.items():
            target = gguf_name(source)
            if target is None:
                continue
            if info["dtype"] != "BF16":
                raise ValueError(f"{source}: expected BF16, got {info['dtype']}")
            begin, end = info["data_offsets"]
            handle.seek(data_start + begin)
            raw = np.frombuffer(handle.read(end - begin), dtype=np.uint16).reshape(info["shape"])
            if raw.ndim == 1:  # norme: llama.cpp le vuole in float32 (bf16 -> f32 e' esatto)
                writer.add_tensor(target, (raw.astype(np.uint32) << 16).view(np.float32))
            else:
                writer.add_tensor(target, raw, raw_shape=raw.shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
            count += 1
    expected = 2 + int(config["num_hidden_layers"]) * len(LAYER_MAP)
    if count != expected:
        raise ValueError(f"copied {count} tensors, expected {expected}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"scritto {out_path} con {count} tensori")


if __name__ == "__main__":
    main()
