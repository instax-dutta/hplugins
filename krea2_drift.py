# SPDX-License-Identifier: AGPL-3.0-or-later
"""Krea 2 / Krea 2 Turbo structural-conditioning scorer for Heretic 2.1.0.

  ./krea2_drift.py:Krea2Drift

Estimates how badly an ablation would move the picture, without generating one.
The quantity is the fused text tensor the DiT actually consumes:

    Qwen3-VL hidden_states[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    -> drop the 34-token system prefix (user tokens + 5-token suffix remain)
    -> txtfusion (2 layerwise blocks, signed projector, 2 refiner blocks)
    -> txtmlp
    -> (token, 6144) conditioning

Drift is the mean token cosine, averaged across prompts, in percent:

    fused = 100 * (1 - cos)

That number is the score, exactly as Heretic receives and minimizes it: no
deadzone, no threshold, no square, no scale. Optuna's Pareto selection only
sees the order of trials, so any monotone transform of fused would pick the
same trials; the raw value is the simplest and the easiest to read.

The trial line is

    <fused> (max: <max> #<index>, min: <min> #<index>)

all in percent, four decimals. `max` / `min` are the drift of the most and the
least moved prompt; `#index` is its 1-based position in the loaded prompt list
(the line number in the prompt file when every line is one prompt and nothing is
skipped or sampled). Fused close to max means every prompt moved about equally
(a diffuse edit); fused far below max means a few prompts moved much more than
the rest. Display only, the score is `fused`.

Constants. Checked against the official krea-ai/krea-2 encoder.py and the
diffusers Krea 2 docs: taps (2, 5, ..., 35), 12 tapped layers, 34-token prefix,
5-token suffix, 512 text tokens (prefix + text <= 541, then the suffix),
RMSNorm eps 1e-5, text width 2560, 20 heads, 2 layerwise + 2 refiner blocks,
SwiGLU width 6912 (derived from the checkpoint, matches the docs); the prefix
and suffix strings are identical to the official ones. The RMSNorm "scale + 1",
tanh GELU, sigmoid gate, SwiGLU width rule (int(2*d/3)*4 rounded up to 128 =
6912), no RoPE in the fusion blocks and the txtmlp layout (RMSNorm, Linear,
GELU tanh, Linear) were checked line by line against the official mmdit.py and
the diffusers transformer_krea2.py. Measured on the loaded model at every start: the noise
floor that `noise_limit` must stay above.

The text path is transcribed from krea-ai/krea-2 mmdit.py and was checked
numerically against the official TextFusionTransformer + txtmlp on identical
weights and inputs (cos >= 0.99996):
  RMSNorm   : F.rms_norm(x.float(), eps=1e-5, weight=scale + 1.0) -> x.dtype
  QKNorm    : qnorm(q), knorm(k), v
  Attention : wo(sdpa(q,k,v,mask) * sigmoid(gate(x)))   -- no RoPE in fusion
  SwiGLU    : down(silu(gate(x)) * up(x))
  Block     : x + attn(prenorm(x), mask); x + mlp(postnorm(x))
  Fusion    : (b*l, n, d) -> 2 layerwise (mask=None) -> projector over n
              -> (b, l, d) -> 2 refiner (mask) -> txtmlp
  mask      : (B, L) -> (B, 1, L, L), True = attend

Deviations, metric-neutral because baseline and trial share them:
  * default SDPA backend instead of forced CUDNN_ATTENTION;
  * eager RMSNorm instead of @torch.compile;
  * fully-masked padding rows get self-attention instead of NaN. Real queries
    never attend to them, and those rows are dropped before the cosine.

Only the fused baseline is kept (bf16, CPU). The 12-layer tap tensors are
discarded as soon as txtfusion has run. The path itself is one bf16 copy,
parked on CPU between trials so a 16 GB card can hold Qwen3-VL-4B.

Memory lifetime. The plugin never keeps a reference to the Heretic model (or its
layers) between calls: it looks the model up in the context at the start of
init()/get_score() and drops every reference in a `finally`, together with the
forward hooks and the fusion path on the GPU. Heretic frees a model by
`self.model = None` (reload after "Save the model", switching trial in the final
menu); a scorer that still pointed at the old model would keep its ~8 GB alive
and the reload would not fit into a 16 GB card.
"""


from __future__ import annotations

import contextlib
import gc
import glob
import math
import os
import random
import statistics
import time
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import format_dataset_specification, print

# ------------------------------------------------------------------ constants

KREA_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
KREA_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
# krea-ai/krea-2 encoder.py: prompt_template_encode_start_idx = 34 (positions
# dropped from hidden states and mask) and a 5-token suffix that is kept.
PREFIX_TOKENS = 34
SUFFIX_TOKENS = 5

# HF output_hidden_states index: [0] is embeddings, [k] is output of layer k-1.
# krea-ai/krea-2 encoder.py: select_layers = (2, 5, 8, ..., 35), stacked per token.
TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
N_TAPS = len(TAPS)

PROJECTOR_SUFFIX = "txtfusion.projector.weight"
PRENORM_SUFFIX = "txtfusion.layerwise_blocks.0.prenorm.scale"

# Only these dtypes may feed the text path. Anything else (int8, fp8, ...) is a
# quantised tensor that load_state_dict would silently cast into garbage.
_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)

# One fused text path per checkpoint. Instances share it.
_PATHS: dict[tuple, "_TextPath"] = {}


def _checkpoint_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise RuntimeError(f"No .safetensors shards under {path}")
        return files
    if os.path.isfile(path):
        return [path]
    raise RuntimeError(f"checkpoint_path does not exist: {path}")


def _collect(path: str, want: Callable[[str], bool]) -> dict[str, torch.Tensor]:
    """Read only the tensors we need; never materialise the whole checkpoint."""
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for file in _checkpoint_files(path):
        with safe_open(file, framework="pt") as handle:
            for key in handle.keys():
                if key in out or not want(key):
                    continue
                tensor = handle.get_tensor(key)
                if tensor.dtype not in _FLOAT_DTYPES:
                    raise RuntimeError(
                        f"{key}: dtype {tensor.dtype} in {file}. The Krea text path "
                        "(txtfusion/txtmlp) must be stored as bf16/fp16/fp32; this "
                        "looks quantised. Point checkpoint_path at a checkpoint "
                        "whose text path is unquantised (e.g. the original BF16)."
                    )
                if not bool(torch.isfinite(tensor).all()):
                    raise RuntimeError(f"{key}: contains NaN/Inf in {file}")
                out[key] = tensor
    return out


def _resolve_encoder(model: nn.Module):
    """Unwrap PEFT; return (bare Qwen3VLModel, decoder layers, rope owner)."""
    base = model
    get_base = getattr(base, "get_base_model", None)
    if callable(get_base):
        unwrapped = get_base()
        if isinstance(unwrapped, nn.Module):
            base = unwrapped

    inner = getattr(base, "model", None)
    encoder = inner if (inner is not None and hasattr(base, "lm_head")) else base

    lm = getattr(encoder, "language_model", None)
    if lm is not None and hasattr(lm, "layers"):
        layers = lm.layers
    elif hasattr(encoder, "layers"):
        layers = encoder.layers
    else:
        raise TypeError(
            f"No Qwen3-VL decoder layers on {type(model).__name__}; "
            "expected Qwen3VLForConditionalGeneration."
        )
    rope_owner = encoder if hasattr(encoder, "rope_deltas") else None
    return encoder, layers, rope_owner


# ------------------------------------------------- Krea text path (mmdit.py)


class _RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        t = F.rms_norm(
            x.float(), (self.features,), eps=self.eps, weight=self.scale.float() + 1.0
        )
        return t.to(dtype)


class _QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = _RMSNorm(dim)
        self.knorm = _RMSNorm(dim)

    def forward(self, q, k, v):
        return self.qnorm(q), self.knorm(k), v


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int | None = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = _QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v, g = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = q.view(b, l, self.heads, self.headdim).transpose(1, 2)
        k = k.view(b, l, self.kvheads, self.headdim).transpose(1, 2)
        v = v.view(b, l, self.kvheads, self.headdim).transpose(1, 2)
        q, k, v = self.qknorm(q, k, v)
        # No RoPE: TextFusionBlock never passes freqs.
        if self.gqa:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(b, l, self.heads * self.headdim)
        return self.wo(o * torch.sigmoid(g))


class _SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _FusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.prenorm = _RMSNorm(features)
        self.postnorm = _RMSNorm(features)
        self.attn = _Attention(features, heads, kvheads, bias)
        self.mlp = _SwiGLU(features, multiplier, bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class _TextFusion(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [_FusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [_FusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )


class _TxtMLP(nn.Module):
    """mmdit.py: Sequential(RMSNorm, Linear, GELU(tanh), Linear)."""

    def __init__(self, txt_dim: int, features: int):
        super().__init__()
        self.norm = _RMSNorm(txt_dim)
        self.fc1 = nn.Linear(txt_dim, features)
        self.fc2 = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)


def _attn_mask(pad: torch.Tensor) -> torch.Tensor:
    """(B,L) key-padding -> (B,1,L,L) bool, True = attend.

    Fully-masked query rows would yield NaN in SDPA. They are padding and are
    discarded downstream, so give them self-attention; real queries never
    attend to them (their key column is False).
    """
    m = pad[:, None, None, :] & pad[:, None, :, None]
    dead = ~m.any(dim=-1, keepdim=True)
    if bool(dead.any()):
        eye = torch.eye(m.shape[-1], dtype=torch.bool, device=m.device)[None, None]
        m = m | (dead & eye)
    return m


def _derive_multiplier(features: int, observed: int) -> int:
    for mult in range(1, 33):
        d = int(2 * features / 3) * mult
        d = 128 * ((d + 127) // 128)
        if d == observed:
            return mult
    raise RuntimeError(
        f"Cannot derive SwiGLU multiplier: features={features}, mlpdim={observed}"
    )


class _TextPath:
    """Krea's frozen text path: full txtfusion -> txtmlp. Always bf16."""

    def __init__(self, settings: "Settings"):
        if not settings.checkpoint_path:
            raise RuntimeError(
                "Krea2Drift needs checkpoint_path (a Krea 2 safetensors whose "
                "txtfusion/txtmlp weights are bf16, fp16 or fp32)."
            )
        self.chunk_tokens = settings.fusion_chunk_tokens
        self.chunk_prompts = settings.fusion_chunk_prompts
        self.resident = torch.device("cpu")

        def want(key: str) -> bool:
            return ".txtmlp." in key or ".txtfusion." in key

        raw = _collect(settings.checkpoint_path, want)
        if not raw:
            raise RuntimeError(
                f"No txtfusion/txtmlp tensors found in {settings.checkpoint_path}"
            )

        fusion_sd: dict[str, torch.Tensor] = {}
        txt_sd: dict[str, torch.Tensor] = {}
        remap = {"0.": "norm.", "1.": "fc1.", "3.": "fc2."}
        for key, tensor in raw.items():
            if ".txtfusion." in key:
                fusion_sd[key.split(".txtfusion.", 1)[1]] = tensor
            elif ".txtmlp." in key:
                tail = key.split(".txtmlp.", 1)[1]
                head, rest = tail.split(".", 1)
                if head + "." not in remap:
                    raise RuntimeError(f"Unexpected txtmlp key: {key}")
                txt_sd[remap[head + "."] + rest] = tensor
        del raw

        if "projector.weight" not in fusion_sd:
            raise RuntimeError(f"{PROJECTOR_SUFFIX} not found in the checkpoint")
        pw = fusion_sd["projector.weight"].flatten().double()
        if pw.numel() != N_TAPS:
            raise RuntimeError(f"projector: expected {N_TAPS} weights, got {pw.numel()}")
        print(
            f"* Krea projector (1, {N_TAPS}): "
            + ", ".join(f"{v:+.4f}" for v in pw.tolist())
        )

        if "layerwise_blocks.0.prenorm.scale" not in fusion_sd:
            raise RuntimeError(f"{PRENORM_SUFFIX} not found in the checkpoint")
        txt_dim = fusion_sd["layerwise_blocks.0.prenorm.scale"].numel()
        self.txt_dim = txt_dim

        for name in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "norm.scale"):
            if name not in txt_sd:
                raise RuntimeError(f"txtmlp.{name} not found in the checkpoint")
        features = txt_sd["fc1.weight"].shape[0]

        # Parameters are born fp32. Cast BEFORE load so the long-lived copy is
        # bf16 on CPU. Loading straight onto CUDA while an older copy was still
        # referenced is what reserved ~18 GB on a 16 GB card.
        dtype = torch.bfloat16
        qk = fusion_sd["layerwise_blocks.0.attn.qknorm.qnorm.scale"].numel()
        if txt_dim % qk:
            raise RuntimeError(f"txt_dim={txt_dim} not divisible by headdim={qk}")
        heads = txt_dim // qk
        kvheads = fusion_sd["layerwise_blocks.0.attn.wk.weight"].shape[0] // qk
        mult = _derive_multiplier(
            txt_dim, fusion_sd["layerwise_blocks.0.mlp.gate.weight"].shape[0]
        )
        bias = any(k.endswith(".bias") for k in fusion_sd)
        self.fusion = _TextFusion(N_TAPS, txt_dim, heads, mult, bias, kvheads)
        self.fusion.to(dtype=dtype)
        missing, unexpected = self.fusion.load_state_dict(fusion_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "txtfusion state_dict mismatch.\n"
                f"  missing: {sorted(missing)}\n  unexpected: {sorted(unexpected)}"
            )
        self.fusion.eval().requires_grad_(False)
        print(
            f"* Krea txtfusion: dim={txt_dim} heads={heads} kvheads={kvheads} "
            f"multiplier={mult} bias={bias}"
        )

        self.txtmlp = _TxtMLP(txt_dim, features)
        self.txtmlp.to(dtype=dtype)
        self.txtmlp.load_state_dict(txt_sd, strict=True)
        self.txtmlp.eval().requires_grad_(False)
        del fusion_sd, txt_sd
        print(f"* Krea txtmlp: {txt_dim} -> {features}")

    def place(self, device: torch.device) -> None:
        device = torch.device(device)
        if self.resident == device:
            return
        self.fusion.to(device=device)
        self.txtmlp.to(device=device)
        self.resident = device

    @torch.inference_mode()
    def transform(self, taps: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """(12, B, T, D) -> (B, T, features): the conditioning the DiT consumes."""
        x = taps.permute(1, 2, 0, 3).contiguous()  # (B, T, N, D)
        b, l, n, d = x.shape
        if d != self.txt_dim:
            raise RuntimeError(
                f"Qwen hidden size {d} != txtfusion dim {self.txt_dim}. "
                "Heretic must be ablating the Qwen3-VL-4B this Krea checkpoint consumes."
            )

        flat = x.reshape(b * l, n, d)
        del x
        pieces = []
        for chunk in flat.split(self.chunk_tokens):
            c = chunk.contiguous()
            for block in self.fusion.layerwise_blocks:
                c = block(c, mask=None)
            pieces.append(c)
        x = torch.cat(pieces, dim=0).reshape(b, l, n, d)
        del flat, pieces

        x = x.permute(0, 1, 3, 2)  # (B, T, D, N)
        x = self.fusion.projector(x).squeeze(-1)  # (B, T, D)

        pieces = []
        for start in range(0, b, self.chunk_prompts):
            seg = x[start : start + self.chunk_prompts]
            mask = _attn_mask(token_mask[start : start + self.chunk_prompts])
            for block in self.fusion.refiner_blocks:
                seg = block(seg, mask=mask)
            pieces.append(self.txtmlp(seg))
        return torch.cat(pieces, dim=0)


def _shared_path(settings: "Settings") -> _TextPath:
    if not settings.checkpoint_path:
        raise RuntimeError(
            "Krea2Drift needs checkpoint_path (a Krea 2 safetensors whose "
            "txtfusion/txtmlp weights are bf16, fp16 or fp32)."
        )
    key = (
        os.path.abspath(settings.checkpoint_path),
        int(settings.fusion_chunk_tokens),
        int(settings.fusion_chunk_prompts),
    )
    path = _PATHS.get(key)
    if path is not None:
        print("* Krea text path reused (one bf16 copy, parked on CPU)")
        return path
    path = _TextPath(settings)
    _PATHS[key] = path
    return path


# ------------------------------------------------------------------- batching


class _Batch:
    __slots__ = ("input_ids", "attention_mask", "token_mask", "counts", "indices")

    def __init__(self, input_ids, attention_mask, token_mask, counts, indices):
        self.input_ids = input_ids            # (B, W) long, CPU
        self.attention_mask = attention_mask  # (B, W) long, CPU
        self.token_mask = token_mask          # (B, W - prefix) bool, CPU
        self.counts = counts                  # (B,) float32, CPU
        self.indices = indices                # list[int]: original prompt ids


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(dataset="GOOD_prompts.txt", split="[256:]")
    )
    max_length: PositiveInt = 512
    batch_size: NonNegativeInt = Field(
        default=0, description="Cap on prompts per forward. 0 = token budget only."
    )
    max_batch_tokens: PositiveInt = 4096
    max_prompts: PositiveInt = 1024
    sample_seed: int = 42
    baseline_device: Literal["auto", "cuda", "cpu"] = "cpu"
    gpu_headroom_gib: float = Field(default=6.0, ge=0.0)
    measure_noise_floor: bool = True

    checkpoint_path: str | None = Field(
        default=None,
        description="Krea 2 safetensors file or shard directory. txtfusion/txtmlp must be float.",
    )
    # Kept so an existing config loads. Only these values are accepted.
    objective: Literal["fused"] = "fused"
    fusion_mode: Literal["full"] = "full"
    tail_fraction: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Ignored. Accepted so older configs still load.",
    )
    fusion_chunk_tokens: PositiveInt = Field(
        default=8192,
        description="Layerwise chunk, in tokens (each token carries its 12 tap vectors).",
    )
    fusion_chunk_prompts: PositiveInt = Field(
        default=8, description="Refiner + txtmlp chunk, in prompts."
    )
    noise_limit: float = Field(
        default=0.01,
        gt=0.0,
        le=100.0,
        description=(
            "Drift in percent treated as numerical noise. Never enters the score. "
            "At start the plugin measures the real noise (unmodified model, "
            "different batching, all prompts) and refuses to run if the noise "
            "exceeds this value."
        ),
    )


class Krea2Drift(Scorer):
    """Drift of the conditioning Krea 2 Turbo feeds the DiT, in percent (minimize)."""

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Krea 2 drift"

    # ---------------------------------------------------------------- binding

    # The model objects are bound for the duration of one init()/get_score() call
    # only (see _bind/_unbind). Between calls these are None, so the plugin never
    # keeps a model alive that Heretic has dropped (reload after save, trial switch).
    _outer = None
    _encoder = None
    _layers = None
    _rope_owner = None

    def _bind(self, ctx: Context) -> None:
        model = ctx.get_model().model
        if model is None:
            raise RuntimeError("Heretic model is not loaded")
        # Heretic swaps the model object (PeftModel after apply_lora, a fresh one
        # after a reload), so it is resolved on every call. The Krea path does not
        # depend on Qwen weights and is built once.
        if not hasattr(model, "get_input_embeddings"):
            raise TypeError("Model does not expose get_input_embeddings()")
        encoder, layers, rope_owner = _resolve_encoder(model)
        if len(layers) <= max(TAPS) - 1:
            raise RuntimeError(
                f"{len(layers)} decoder layers; tap hidden_states[{max(TAPS)}] "
                f"needs layer {max(TAPS) - 1}."
            )
        encoder.eval()
        self._encoder, self._layers, self._rope_owner = encoder, layers, rope_owner
        self._outer = model
        if not hasattr(self, "_path"):
            self._path = _shared_path(self.settings)

    def _unbind(self) -> None:
        """Drop every reference to the Heretic model."""
        self._outer = self._encoder = self._layers = self._rope_owner = None

    def _device(self) -> torch.device:
        return self._outer.get_input_embeddings().weight.device

    def _use_path(self) -> None:
        # ARA's LBFGS closure keeps fp32 copies of edited weights until a
        # collection. Free those before parking ~0.8 GB of fusion on the GPU.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._path.place(self._device())

    def _release_path(self) -> None:
        path = getattr(self, "_path", None)
        if path is None:
            return
        path.place(torch.device("cpu"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run(self, input_ids, attention_mask, **kwargs):
        owner = self._rope_owner
        saved = getattr(owner, "rope_deltas", None) if owner is not None else None
        try:
            # No position_ids: Qwen3VLModel.get_rope_index is what Krea calls.
            return self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )
        finally:
            if owner is not None:
                owner.rope_deltas = saved

    @contextlib.contextmanager
    def _taps(self, consumer: Callable[[int, torch.Tensor], None]):
        handles = []
        for slot, hs in enumerate(TAPS):
            def hook(_m, _a, output, slot=slot):
                tensor = output[0] if isinstance(output, tuple) else output
                consumer(slot, tensor[:, PREFIX_TOKENS:, :])
            handles.append(self._layers[hs - 1].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _collect_taps(self, batch: _Batch) -> torch.Tensor:
        """One forward; return (12, B, T, D) gathered on the compute device."""
        device = self._device()
        ids = batch.input_ids.to(device, non_blocking=True)
        mask = batch.attention_mask.to(device, non_blocking=True)
        store: dict[int, torch.Tensor] = {}

        def gather(slot: int, states: torch.Tensor) -> None:
            # The slice is a view onto the full layer output. Copy it, or all
            # twelve activations stay alive until the forward returns.
            copied = states.detach().contiguous()
            if copied.device != device:
                copied = copied.to(device, non_blocking=True)
            store[slot] = copied

        if self._use_hooks:
            with self._taps(gather):
                self._run(ids, mask)
        else:
            outputs = self._run(ids, mask, output_hidden_states=True)
            hs = outputs.hidden_states
            if hs is None:
                raise RuntimeError("Qwen3-VL returned no hidden_states")
            for slot, idx in enumerate(TAPS):
                gather(slot, hs[idx][:, PREFIX_TOKENS:, :])
            del outputs, hs
        if len(store) != N_TAPS:
            raise RuntimeError(f"Captured {len(store)} of {N_TAPS} taps")
        return torch.stack([store[s] for s in range(N_TAPS)])

    def _fuse(self, batch: _Batch, taps: torch.Tensor) -> torch.Tensor:
        """Taps -> DiT conditioning, then drop the taps."""
        device = taps.device
        token_mask = batch.token_mask.to(device, non_blocking=True)
        cond = self._path.transform(taps, token_mask)
        del taps, token_mask
        return cond

    # -------------------------------------------------------------- tokenize

    def _tokenize(self, tokenizer) -> list[list[int]]:
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            prefix_ids = tokenizer([KREA_PREFIX], return_tensors="pt")["input_ids"][0].tolist()
            if len(prefix_ids) != PREFIX_TOKENS:
                raise RuntimeError(
                    f"Krea prefix -> {len(prefix_ids)} tokens, expected {PREFIX_TOKENS}. "
                    "Wrong tokenizer, or the template string was edited."
                )
            suffix_ids = tokenizer([KREA_SUFFIX], return_tensors="pt")["input_ids"][0].tolist()
            if len(suffix_ids) != SUFFIX_TOKENS:
                raise RuntimeError(
                    f"Krea suffix -> {len(suffix_ids)} tokens, expected {SUFFIX_TOKENS}."
                )
            block = self.settings.max_length + PREFIX_TOKENS - SUFFIX_TOKENS
            encoded = tokenizer(
                [KREA_PREFIX + t for t in self._texts],
                truncation=True,
                max_length=block,
                padding=False,
                add_special_tokens=True,
                return_attention_mask=False,
            )
            seqs, crossed, truncated = [], 0, 0
            for raw in encoded["input_ids"]:
                ids = raw.tolist() if hasattr(raw, "tolist") else [int(x) for x in raw]
                if ids[:PREFIX_TOKENS] != prefix_ids:
                    crossed += 1  # Krea drops by position too; keep the prompt.
                if len(ids) >= block:
                    truncated += 1
                seqs.append(ids + suffix_ids)
            if crossed:
                print(
                    f"[yellow]* {crossed} caption(s) merge across the template "
                    "boundary; kept (Krea drops by position too).[/]"
                )
            if truncated:
                print(
                    f"[yellow]* {truncated} caption(s) reach max_length="
                    f"{self.settings.max_length} and were truncated.[/]"
                )
            self._pad_id = int(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            return seqs
        finally:
            tokenizer.padding_side = old_side

    def _make_batch(self, indices: list[int]) -> _Batch:
        rows = [self._seqs[i] for i in indices]
        width, n = max(len(s) for s in rows), len(rows)
        input_ids = torch.full((n, width), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, width), dtype=torch.long)
        counts = torch.empty(n, dtype=torch.float32)
        for r, seq in enumerate(rows):
            length = len(seq)
            input_ids[r, :length] = torch.tensor(seq, dtype=torch.long)
            attention_mask[r, :length] = 1
            counts[r] = length - PREFIX_TOKENS
        positions = torch.arange(width - PREFIX_TOKENS).unsqueeze(0)
        token_mask = positions < counts.long().unsqueeze(1)
        return _Batch(input_ids, attention_mask, token_mask, counts, list(indices))

    def _build_batches(
        self, budget: int | None = None, descending: bool = False
    ) -> list[_Batch]:
        budget = budget or self.settings.max_batch_tokens
        cap = self.settings.batch_size
        order = sorted(
            range(len(self._seqs)),
            key=lambda i: len(self._seqs[i]),
            reverse=descending,
        )
        groups, current, width = [], [], 0
        for i in order:
            length = len(self._seqs[i])
            new_width = max(width, length)
            too_big = new_width * (len(current) + 1) > budget
            too_many = cap > 0 and len(current) >= cap
            if current and (too_big or too_many):
                groups.append(current)
                current, new_width = [], length
            current.append(i)
            width = new_width
        if current:
            groups.append(current)
        return [self._make_batch(g) for g in groups]

    # ------------------------------------------------------------------ init

    @torch.inference_mode()
    def init(self, ctx: Context) -> None:
        try:
            self._setup(ctx)
        finally:
            # Also on an error: nothing may keep the model alive after init().
            self._release_path()
            self._unbind()

    def _setup(self, ctx: Context) -> None:
        print()
        print(
            "Loading Krea conditioning prompts from "
            f"[bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        prompts = ctx.load_prompts(self.settings.prompts)
        if not prompts:
            raise ValueError("Krea scorer received an empty prompt dataset")
        if len(prompts) > self.settings.max_prompts:
            rng = random.Random(self.settings.sample_seed)
            idx = sorted(rng.sample(range(len(prompts)), self.settings.max_prompts))
            prompts = [prompts[i] for i in idx]
        self._texts = [p.user for p in prompts]
        print(f"* [bold]{len(self._texts)}[/] captions")

        self._bind(ctx)
        self._seqs = self._tokenize(ctx.get_model().tokenizer)

        lengths = [len(s) for s in self._seqs]
        print(
            "* Token lengths (prefix+user+suffix): "
            f"min={min(lengths)} median={sorted(lengths)[len(lengths) // 2]} "
            f"max={max(lengths)}"
        )
        self._batches = self._build_batches()
        padded = sum(b.input_ids.numel() for b in self._batches)
        print(
            f"* {len(self._batches)} batches, {padded:,} padded tokens "
            f"({100.0 * (1.0 - sum(lengths) / padded):.1f}% padding)"
        )

        self._use_hooks = True
        self._verify_taps()
        print("* Capturing reference conditioning from the base model...")
        self._use_path()
        try:
            self._verify_padding()
            started = time.perf_counter()
            self._baseline = self._capture_baseline()
            elapsed = time.perf_counter() - started
            fus_mib = sum(t.numel() * t.element_size() for t in self._baseline) / (1024**2)
            where = sorted({str(t.device) for t in self._baseline})
            print(
                f"* Reference fused {fus_mib:.0f} MiB on {'/'.join(where)}, "
                f"hooks={'on' if self._use_hooks else 'off'}; captured in "
                f"{elapsed:.1f} s (this is about the cost of one score)"
            )
            limit = float(self.settings.noise_limit)
            print(
                "* Score: mean fused drift in percent, minimized as is "
                "(no deadzone, no scaling)"
            )

            if self.settings.measure_noise_floor:
                try:
                    noise, regrouped = self._batch_noise()
                except Exception as error:  # a broken probe must not kill the study
                    noise = None
                    print(f"[yellow]* Noise floor not measured: {error!r}[/]")
                if noise is not None:
                    if not all(math.isfinite(v) for v in noise):
                        raise RuntimeError("Non-finite conditioning on the unmodified model")
                    mean_noise = max(0.0, statistics.fmean(noise))
                    p99 = max(0.0, sorted(noise)[round(0.99 * (len(noise) - 1))])
                    top = max(0.0, max(noise))
                    margin = (
                        f"{limit / top:.1f}x the max"
                        if top > 0
                        else "above the max (no measurable noise)"
                    )
                    print(
                        f"* Noise floor (unmodified model, regrouped batches for "
                        f"{100.0 * regrouped:.0f}% of prompts): mean "
                        f"[bold]{mean_noise:.4f}%[/], p99 {p99:.4f}%, max {top:.4f}%; "
                        f"noise_limit {limit:.4f}% is {margin}"
                    )
                    if regrouped == 0.0:
                        print(
                            "[yellow]* Every prompt kept its batch-mates, so this "
                            "noise figure is not informative.[/]"
                        )
                    if top > limit:
                        raise RuntimeError(
                            f"Re-batching the unmodified model moves the conditioning "
                            f"by up to {top:.4f}% per prompt, above noise_limit "
                            f"{limit:.4f}%. Scores would carry that noise. Fix the "
                            "model dtype/backend, or raise noise_limit above the "
                            "measured maximum."
                        )
        finally:
            self._release_path()

    def _verify_taps(self) -> None:
        longest = max(range(len(self._seqs)), key=lambda i: len(self._seqs[i]))
        probe = self._make_batch([longest])
        device = self._device()
        bucket: dict[int, torch.Tensor] = {}

        def consume(slot: int, states: torch.Tensor) -> None:
            bucket[slot] = states.detach().contiguous()
            if bucket[slot].device != device:
                bucket[slot] = bucket[slot].to(device)

        with self._taps(consume):
            outputs = self._run(
                probe.input_ids.to(device),
                probe.attention_mask.to(device),
                output_hidden_states=True,
            )
        hs = outputs.hidden_states
        if hs is None:
            raise RuntimeError("Verification forward returned no hidden_states")
        count = len(hs)

        mismatches: list[str] = []
        for slot, idx in enumerate(TAPS):
            if idx >= count:
                mismatches.append(f"hidden_states[{idx}] missing (len={count})")
                continue
            expected = hs[idx][:, PREFIX_TOKENS:, :].to(device)
            got = bucket.get(slot)
            if got is None or got.shape != expected.shape:
                mismatches.append(f"hidden_states[{idx}] shape mismatch")
            elif not torch.equal(got, expected):
                delta = (got.float() - expected.float()).abs().max().item()
                mismatches.append(f"hidden_states[{idx}] max abs diff {delta:.3e}")
        del outputs, hs, bucket

        print(
            f"* hidden_states length = {count} "
            f"({'index 0 is embeddings' if count == len(self._layers) + 1 else 'UNEXPECTED'}); "
            f"taps {list(TAPS)}"
        )
        if count != len(self._layers) + 1:
            print(
                "[yellow]Warning: unexpected hidden_states length; the taps may not "
                "be the ones the DiT saw in training.[/]"
            )
        if mismatches:
            self._use_hooks = False
            print(
                "[yellow]Warning: hooks do not match hidden_states "
                f"({'; '.join(mismatches)}). Falling back to output_hidden_states.[/]"
            )
        else:
            print("* Tap hooks match hidden_states exactly; lm_head is skipped")

    def _verify_padding(self) -> None:
        """Fused conditioning of a short prompt must not depend on its padding."""
        order = sorted(range(len(self._seqs)), key=lambda i: len(self._seqs[i]))
        short, long_ = order[0], order[-1]
        if len(self._seqs[short]) == len(self._seqs[long_]):
            print("* Padding probe skipped: all prompts have the same length")
            return
        solo_batch = self._make_batch([short])
        solo = self._fuse(solo_batch, self._collect_taps(solo_batch))
        pair_batch = self._make_batch([short, long_])
        pair = self._fuse(pair_batch, self._collect_taps(pair_batch))
        n = int(solo_batch.counts[0].item())
        cos = F.cosine_similarity(
            solo[0, :n].float(), pair[0, :n].float(), dim=-1, eps=1e-8
        )
        worst, mean = float(cos.min()), float(cos.mean())
        del solo, pair, cos
        drift = 100.0 * (1.0 - mean)
        limit = float(self.settings.noise_limit)
        print(
            f"* Padding probe: fused cos mean={mean:.6f} min={worst:.6f} "
            f"(prompt drift {drift:.4f}%, noise_limit {limit:.4f}%)"
        )
        # The score is a per-prompt mean, so the probe is judged on the mean too.
        if drift > limit:
            raise RuntimeError(
                f"Padding changes fused conditioning by {drift:.4f}% "
                f"(> noise_limit {limit:.4f}%). Refusing to score: batch "
                "composition would leak into the score."
            )

    def _batch_noise(self) -> tuple[list[float], float]:
        """Noise floor of the score, per prompt, in percent.

        Drift of the UNMODIFIED model when the same prompts are batched
        differently (half the token budget, opposite length order): other
        neighbours, other padded width, other GEMM shapes, other rounding.
        Also returns the share of prompts whose batch composition changed
        (if that is 0 the measurement says nothing).
        """
        where: dict[int, tuple[int, int]] = {}
        for b, batch in enumerate(self._batches):
            for row, prompt in enumerate(batch.indices):
                where[prompt] = (b, row)
        base_sets = [set(batch.indices) for batch in self._batches]
        device = self._device()
        noise = [0.0] * len(self._texts)
        regrouped = 0
        alt = self._build_batches(
            budget=max(1, self.settings.max_batch_tokens // 2), descending=True
        )
        for batch in alt:
            members = set(batch.indices)
            cur = self._fuse(batch, self._collect_taps(batch))
            for row, prompt in enumerate(batch.indices):
                b, base_row = where[prompt]
                regrouped += members != base_sets[b]
                n = int(batch.counts[row].item())
                ref = self._baseline[b][base_row, :n].to(device, non_blocking=True)
                cos = F.cosine_similarity(
                    cur[row, :n].float(), ref.float(), dim=-1, eps=1e-8
                )
                noise[prompt] = 100.0 * (1.0 - float(cos.double().mean()))
            del cur
        return noise, regrouped / max(1, len(self._texts))

    # -------------------------------------------------------------- baseline

    def _to_cpu(self, t: torch.Tensor) -> torch.Tensor:
        mode = self.settings.baseline_device
        if t.device.type != "cuda":
            return t.contiguous()
        keep = mode == "cuda"
        if mode == "auto":
            free, _ = torch.cuda.mem_get_info(t.device)
            keep = free >= self.settings.gpu_headroom_gib * (1024**3)
        if keep:
            return t.contiguous()
        out = t.to("cpu", torch.bfloat16)
        try:
            out = out.pin_memory()
        except Exception:
            pass
        return out.contiguous()

    def _capture_baseline(self) -> list[torch.Tensor]:
        fused_list: list[torch.Tensor] = []
        for batch in self._batches:
            cond = self._fuse(batch, self._collect_taps(batch))
            fused_list.append(self._to_cpu(cond))
            del cond
        return fused_list

    # --------------------------------------------------------------- scoring

    def _prompt_mean(self, cos: torch.Tensor, batch: _Batch) -> torch.Tensor:
        device = cos.device
        mask = batch.token_mask.to(device, non_blocking=True)
        cos = torch.where(mask, cos, torch.zeros_like(cos))
        counts = batch.counts.to(device, non_blocking=True).clamp_min(1.0)
        return cos.sum(dim=-1) / counts

    def _measure(self) -> torch.Tensor | None:
        """Per-prompt cosine of the fused conditioning against the baseline. CPU float64."""
        device = self._device()
        fused = torch.empty(len(self._texts), dtype=torch.float64)
        for i, batch in enumerate(self._batches):
            cur = self._fuse(batch, self._collect_taps(batch))
            base_f = self._baseline[i].to(device, non_blocking=True)
            if cur.shape != base_f.shape:
                raise RuntimeError(
                    f"Fused shape {tuple(cur.shape)} vs baseline {tuple(base_f.shape)}"
                )
            fcos = F.cosine_similarity(cur.float(), base_f.float(), dim=-1, eps=1e-8)
            del cur, base_f
            fused[torch.as_tensor(batch.indices)] = self._prompt_mean(fcos, batch).cpu().double()
            del fcos
        if not torch.isfinite(fused).all():
            return None
        return fused

    @staticmethod
    def _detail(
        max_pct: float, max_index: int, min_pct: float, min_index: int
    ) -> str:
        return (
            f"max: {max_pct:.4f} #{max_index + 1}, "
            f"min: {min_pct:.4f} #{min_index + 1}"
        )

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{value:.4f}"

    def _score(self, value: float, detail: str) -> Score:
        text = self._format_value(value)
        return Score(
            value=float(value),
            rich_display=f"[bold]{text}[/] [italic]({detail})[/]",
            md_display=f"{text} *({detail})*",
        )

    @torch.inference_mode()
    def get_score(self, ctx: Context) -> Score:
        try:
            self._bind(ctx)
            self._use_path()
            cos = self._measure()
        finally:
            self._release_path()
            self._unbind()
        if cos is None:
            return self._catastrophe("non-finite conditioning")
        pct = [100.0 * (1.0 - c) for c in cos.tolist()]
        if not pct or not all(math.isfinite(p) for p in pct):
            return self._catastrophe("non-finite metric")
        # cos can exceed 1 by rounding; do not let a "-0.0000" reach Heretic.
        fused_pct = max(0.0, statistics.fmean(pct))
        max_index = max(range(len(pct)), key=pct.__getitem__)
        min_index = min(range(len(pct)), key=pct.__getitem__)
        return self._score(
            fused_pct,
            self._detail(
                max(0.0, pct[max_index]), max_index, max(0.0, pct[min_index]), min_index
            ),
        )

    def _catastrophe(self, reason: str) -> Score:
        # Above the largest possible drift (every token at cosine -1 = 200%).
        return self._score(201.0, reason)

    def get_baseline_score(self, ctx: Context) -> Score:
        del ctx
        return self._score(0.0, "baseline")
