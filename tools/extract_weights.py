"""
Extract model weights in both FP32 and INT8 layouts for the pure-C inference
binary (c/vm_detect).

Pulls the PyTorch checkpoint from the Hugging Face Hub on first run; override
with `MODEL_ID=<huggingface/repo>` or by passing a local directory as argv[1].

Outputs:
  weights-fp32/    ~1.26 GB, every tensor as float32
  weights-int8/    ~355 MB, large MatMul weights as int8 + per-row float32 scale
                   (conv, LayerNorm, biases, pos_conv remain FP32)

Layout:
  <dir>/feature_extractor/conv{0..6}.weight.bin, conv{i}.bias.bin,
        conv{i}.norm_weight.bin, conv{i}.norm_bias.bin
  <dir>/feature_projection/norm_{weight,bias}.bin, proj.{weight,bias}.bin
  <dir>/pos_conv/weight.bin, bias.bin                 (weight-norm collapsed)
  <dir>/encoder_norm/weight.bin, bias.bin             (post-encoder LN)
  <dir>/encoder/layer_{0..23}/
          ln1.{weight,bias}.bin, ln2.{weight,bias}.bin
          attn_{q,k,v,out}.{weight,bias}.bin
          ffn_{in,out}.{weight,bias}.bin
  <dir>/classifier/projector.{weight,bias}.bin, out.{weight,bias}.bin
  <dir>/manifest.json

For INT8 outputs, each quantized 2D weight is saved as two files:
  <stem>.q8.bin     int8, shape [M, K] row-major
  <stem>.scale.bin  float32, shape [M]   (one scale per output row)
The C loader auto-detects these and dequantizes to FP32 at model load time.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioClassification

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "MODEL_ID", "jakeBland/wav2vec-vm-finetune"
)
FP32_DIR = Path("weights-fp32")
INT8_DIR = Path("weights-int8")


# ---------------------------------------------------------------------------
# FP32 extraction: walk the PyTorch state dict, write each tensor as float32 bin
# ---------------------------------------------------------------------------
def extract_fp32():
    if (FP32_DIR / "manifest.json").exists():
        print(f"[fp32] {FP32_DIR}/ already populated, skipping")
        return
    FP32_DIR.mkdir(exist_ok=True)
    print(f"[fp32] loading {MODEL_ID} ...")
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.eval()
    sd = model.state_dict()
    manifest: dict[str, dict] = {}

    def save(rel: str, tensor: torch.Tensor) -> None:
        path = FP32_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.ascontiguousarray(tensor.detach().cpu().float().numpy())
        arr.tofile(path)
        manifest[rel] = {"shape": list(arr.shape), "dtype": "float32", "numel": int(arr.size)}

    for i in range(7):
        p = f"wav2vec2.feature_extractor.conv_layers.{i}"
        save(f"feature_extractor/conv{i}.weight.bin",      sd[f"{p}.conv.weight"])
        save(f"feature_extractor/conv{i}.bias.bin",        sd[f"{p}.conv.bias"])
        save(f"feature_extractor/conv{i}.norm_weight.bin", sd[f"{p}.layer_norm.weight"])
        save(f"feature_extractor/conv{i}.norm_bias.bin",   sd[f"{p}.layer_norm.bias"])

    save("feature_projection/norm_weight.bin", sd["wav2vec2.feature_projection.layer_norm.weight"])
    save("feature_projection/norm_bias.bin",   sd["wav2vec2.feature_projection.layer_norm.bias"])
    save("feature_projection/proj.weight.bin", sd["wav2vec2.feature_projection.projection.weight"])
    save("feature_projection/proj.bias.bin",   sd["wav2vec2.feature_projection.projection.bias"])

    # Positional conv: collapse weight-norm into a single materialized weight
    def collapse_wn(prefix: str) -> torch.Tensor:
        pairs = [("weight_g", "weight_v"),
                 ("parametrizations.weight.original0", "parametrizations.weight.original1")]
        for gk, vk in pairs:
            if f"{prefix}.{gk}" in sd and f"{prefix}.{vk}" in sd:
                g, v = sd[f"{prefix}.{gk}"], sd[f"{prefix}.{vk}"]
                norm_dims = [d for d in range(v.dim()) if g.shape[d] == 1]
                v_norm = v.pow(2).sum(dim=norm_dims, keepdim=True).sqrt().clamp(min=1e-12)
                return v * (g / v_norm)
        return sd[f"{prefix}.weight"]

    save("pos_conv/weight.bin", collapse_wn("wav2vec2.encoder.pos_conv_embed.conv"))
    save("pos_conv/bias.bin",   sd["wav2vec2.encoder.pos_conv_embed.conv.bias"])

    save("encoder_norm/weight.bin", sd["wav2vec2.encoder.layer_norm.weight"])
    save("encoder_norm/bias.bin",   sd["wav2vec2.encoder.layer_norm.bias"])

    for i in range(24):
        p = f"wav2vec2.encoder.layers.{i}"
        save(f"encoder/layer_{i}/ln1.weight.bin", sd[f"{p}.layer_norm.weight"])
        save(f"encoder/layer_{i}/ln1.bias.bin",   sd[f"{p}.layer_norm.bias"])
        for q in ("q", "k", "v"):
            save(f"encoder/layer_{i}/attn_{q}.weight.bin", sd[f"{p}.attention.{q}_proj.weight"])
            save(f"encoder/layer_{i}/attn_{q}.bias.bin",   sd[f"{p}.attention.{q}_proj.bias"])
        save(f"encoder/layer_{i}/attn_out.weight.bin", sd[f"{p}.attention.out_proj.weight"])
        save(f"encoder/layer_{i}/attn_out.bias.bin",   sd[f"{p}.attention.out_proj.bias"])
        save(f"encoder/layer_{i}/ln2.weight.bin", sd[f"{p}.final_layer_norm.weight"])
        save(f"encoder/layer_{i}/ln2.bias.bin",   sd[f"{p}.final_layer_norm.bias"])
        save(f"encoder/layer_{i}/ffn_in.weight.bin",  sd[f"{p}.feed_forward.intermediate_dense.weight"])
        save(f"encoder/layer_{i}/ffn_in.bias.bin",    sd[f"{p}.feed_forward.intermediate_dense.bias"])
        save(f"encoder/layer_{i}/ffn_out.weight.bin", sd[f"{p}.feed_forward.output_dense.weight"])
        save(f"encoder/layer_{i}/ffn_out.bias.bin",   sd[f"{p}.feed_forward.output_dense.bias"])

    save("classifier/projector.weight.bin", sd["projector.weight"])
    save("classifier/projector.bias.bin",   sd["projector.bias"])
    save("classifier/out.weight.bin",       sd["classifier.weight"])
    save("classifier/out.bias.bin",         sd["classifier.bias"])

    hparams = {
        "sample_rate": 16000, "input_samples": 32000, "conv_channels": 512,
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2], "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "hidden_size": 1024, "ffn_size": 4096, "num_heads": 16, "head_dim": 64,
        "num_layers": 24, "classifier_proj_size": 256, "num_labels": 2,
        "id2label": {"0": "human", "1": "voicemail"},
        "pos_conv_kernel": 128, "pos_conv_groups": 16,
        "layer_norm_eps": 1e-5, "feat_extract_norm": "layer", "do_stable_layer_norm": True,
    }
    total = sum(v["numel"] for v in manifest.values())
    with open(FP32_DIR / "manifest.json", "w") as f:
        json.dump({"hparams": hparams, "tensors": manifest}, f, indent=2)
    print(f"[fp32] wrote {len(manifest)} tensors, {total:,} params ({total * 4 / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# INT8: copy fp32 layout verbatim, but replace the large MatMul weights with
# per-row symmetric int8 + float32 scale pair.
# ---------------------------------------------------------------------------
def _quantize_targets() -> set[str]:
    targets = {
        "feature_projection/proj.weight.bin",
        "classifier/projector.weight.bin",
        "classifier/out.weight.bin",
    }
    for i in range(24):
        for w in ("attn_q", "attn_k", "attn_v", "attn_out", "ffn_in", "ffn_out"):
            targets.add(f"encoder/layer_{i}/{w}.weight.bin")
    # Feature extractor convs 1..6. conv0 is skipped because its flattened K
    # (in_c * kernel = 1 * 10 = 10) isn't a multiple of the int8 kernel's
    # k-chunk (4 for SDOT, 8 for SMMLA). It stays fp32 — the layer is tiny
    # anyway (in_c=1 → the whole conv is ~1% of FE compute).
    for i in range(1, 7):
        targets.add(f"feature_extractor/conv{i}.weight.bin")
    return targets


def extract_int8():
    if (INT8_DIR / "manifest.json").exists():
        print(f"[int8] {INT8_DIR}/ already populated, skipping")
        return
    if not (FP32_DIR / "manifest.json").exists():
        raise SystemExit(f"{FP32_DIR}/ missing — run extract_fp32 first")

    INT8_DIR.mkdir(exist_ok=True)
    manifest = json.load(open(FP32_DIR / "manifest.json"))
    tensors = manifest["tensors"]
    quantize = _quantize_targets()

    saved_fp32_bytes = 0
    saved_int8_bytes = 0
    nq = 0

    for rel, meta in tensors.items():
        src = FP32_DIR / rel
        dst_dir = INT8_DIR / Path(rel).parent
        dst_dir.mkdir(parents=True, exist_ok=True)

        if rel in quantize:
            shape = meta["shape"]
            # Linear weights are 2D (out, in); conv weights are 3D
            # (out_c, in_c, kernel). In both cases per-output-row quantization
            # flattens the trailing dims to match how sgemm treats the weight.
            M = shape[0]
            K = int(np.prod(shape[1:]))
            W = np.fromfile(src, dtype=np.float32).reshape(M, K)
            scale = np.abs(W).max(axis=1) / 127.0
            scale[scale == 0] = 1.0
            Q = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
            stem = Path(rel).stem
            Q.tofile(dst_dir / f"{stem}.q8.bin")
            scale.astype(np.float32).tofile(dst_dir / f"{stem}.scale.bin")
            saved_fp32_bytes += W.nbytes
            saved_int8_bytes += Q.nbytes + scale.astype(np.float32).nbytes
            nq += 1
        else:
            shutil.copy(src, INT8_DIR / rel)

    shutil.copy(FP32_DIR / "manifest.json", INT8_DIR / "manifest.json")
    ratio = saved_fp32_bytes / saved_int8_bytes if saved_int8_bytes else 0
    print(f"[int8] quantized {nq} tensors "
          f"({saved_fp32_bytes / 1e6:.1f} MB -> {saved_int8_bytes / 1e6:.1f} MB, {ratio:.2f}x)")


def report():
    def dir_size(p):
        return sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())

    if FP32_DIR.exists():
        print(f"  {FP32_DIR}/: {dir_size(FP32_DIR) / 1e6:7.1f} MB")
    if INT8_DIR.exists():
        print(f"  {INT8_DIR}/: {dir_size(INT8_DIR) / 1e6:7.1f} MB")


if __name__ == "__main__":
    extract_fp32()
    extract_int8()
    print("\nsizes:")
    report()
