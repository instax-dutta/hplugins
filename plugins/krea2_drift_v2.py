# SPDX-License-Identifier: AGPL-3.0-or-later
"""Krea 2 / Krea 2 Turbo structural-conditioning scorers for Heretic 2.1.0.

  ./krea2_drift_v2.py:Krea2Good    optimization = "minimize"
  ./krea2_drift_v2.py:Krea2Bad     optimization = "maximize"

Both estimate how far an ablation moves the conditioning, without generating an
image. The quantity is the fused text tensor the DiT actually consumes:

    Qwen3-VL hidden_states[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    -> drop the 34-token system prefix (user tokens + 5-token suffix remain)
    -> txtfusion (2 layerwise blocks, signed projector, 2 refiner blocks)
    -> txtmlp
    -> (token, 6144) conditioning

The Qwen forward stops right after the last tapped layer: the layers behind it
and the final norm do not feed the conditioning, so they are never computed.

The drift of one prompt is 100 * (1 - mean token cosine against the unmodified
model), in percent, clamped at 0 (identical to the original).

  Krea2Good  p90max: the largest drift left after discarding the 10% of prompts
             that moved MOST. Minimized: keep the good prompts unchanged,
             without letting a few strong outliers decide the score.
  Krea2Bad   p90min: the smallest drift left after discarding the 10% of prompts
             that moved LEAST. Maximized: make the bad prompts move, without
             letting a few unmoved ones decide the score.

With n prompts exactly floor(10 * n / 100) are discarded (nearest rank, no
interpolation), so the score is always the drift of a real prompt. Below 10
prompts nothing is discarded and the score is the plain max / min. The value goes
to Heretic as is: no deadzone, no scaling.

The trial line is

    Krea2Good: <p90max> (max: <max> #<index1> #<index2> #<index3>)
    Krea2Bad:  <p90min> (min: <min> #<index1> #<index2> #<index3>)

in percent, four decimals. `max` / `min` is the most extreme overall (before
trimming). The three `#index` values are the 1-based prompt numbers of the three
most extreme drifts on the discarded side, ordered by drift (descending for max,
ascending for min). Display only.

Prompts and noise. The configured prompts (after `split`) are the active set and
their number is the target. With a local text file and a `split`, the lines that
follow the selected slice form a reserve; prefix / suffix apply to them too. At
start every active prompt is measured on the unmodified model with different
batching. Prompts above `noise_limit` are dropped, replaced from the reserve in
file order, and the check repeats on the new batches until all active prompts
pass. After NOISE_TOPUP_PASSES passes they are only dropped, not replaced. With
no reserve left the scorer continues with fewer prompts.

Constants. Checked against the official krea-ai/krea-2 encoder.py and the
diffusers Krea 2 docs: taps (2, 5, ..., 35), 12 tapped layers, 34-token prefix,
5-token suffix, 512 text tokens (prefix + text <= 541, then the suffix),
RMSNorm eps 1e-5, text width 2560, 20 heads, 2 layerwise + 2 refiner blocks,
SwiGLU width 6912 (derived from the checkpoint, matches the docs); the prefix
and suffix strings are identical to the official ones. The RMSNorm "scale + 1",
tanh GELU, sigmoid gate, SwiGLU width rule (int(2*d/3)*4 rounded up to 128 =
6912), no RoPE in the fusion blocks and the txtmlp layout (RMSNorm, Linear,
GELU tanh, Linear) were checked line by line against the official mmdit.py and
the diffusers transformer_krea2.py.

The text path is transcribed from krea-ai/krea-2 mmdit.py and was checked
numerically against the official TextFusionTransformer + txtmlp on identical
weights and inputs (implementation-validation cosine >= 0.99996; this is not the
runtime scorer metric):
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
    never attend to them, and those rows are excluded from the distance metric.

Each scorer keeps only its own fused baseline (bf16, CPU; pinned when the
compute device is CUDA); the 12-layer tap tensors are not kept, and the prompt
texts, token lists and reserve are released at the end of init(). Both scorers
share one bf16 copy of the text path. It stays in RAM for the whole run; a
second copy is uploaded to the GPU for the duration of each call and dropped
afterwards, so a 16 GB card can hold Qwen3-VL-4B.

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
import glob
import math
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, PositiveInt
from rich.markup import escape

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import format_dataset_specification, get_split_slice, print

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

# Share of prompts discarded from the extreme end before taking max / min.
TRIM_PERCENT = 10

# Noise filter passes that may refill from the reserve. Later passes only drop,
# so the filter always ends.
NOISE_TOPUP_PASSES = 7

# VRAM chunking of the text path.
# Layerwise blocks: each token carries 12 tap vectors and costs about 1 MB of
# peak activations (SwiGLU 6912 wide, fp32 RMSNorm copies), measured at the real
# dimensions. 512 tokens ~ 0.5 GB; 8192 would not chunk a 4096-token batch at all
# (~4 GB). The result does not depend on the chunk size, only the peak does.
FUSION_CHUNK_TOKENS = 512
FUSION_CHUNK_PROMPTS = 8  # refiner blocks + txtmlp

# Rows (tokens) per step of the drift computation. Its FP32 temporaries stay at
# about 12 MB each whatever max_batch_tokens is.
DRIFT_CHUNK_ROWS = 512

# Only these dtypes may feed the text path. Anything else (int8, fp8, ...) is a
# quantised tensor that load_state_dict would silently cast into garbage.
_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)

# One fused text path per checkpoint. All scorers share it.
_PATHS: dict[str, "_TextPath"] = {}


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
    """Read only the tensors we need, as private bf16 copies in RAM.

    Each tensor is converted as soon as it is read, so a wider source tensor is
    released at once, and nothing returned stays memory-mapped to the file.
    """
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
                out[key] = tensor.to(torch.bfloat16, copy=True)
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

    language_model = getattr(encoder, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        layers = language_model.layers
    elif hasattr(encoder, "layers"):
        # Keep compatibility with wrappers / Qwen variants that expose the
        # decoder layers directly on the unwrapped encoder.
        layers = encoder.layers
    else:
        raise TypeError(
            f"No Qwen3-VL decoder layers on {type(model).__name__}; "
            "expected Qwen3VLForConditionalGeneration or an encoder exposing .layers."
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
    # Applied unconditionally: asking "is any row dead?" would stall the GPU.
    eye = torch.eye(m.shape[-1], dtype=torch.bool, device=m.device)[None, None]
    return m | (dead & eye)


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

    def __init__(self, checkpoint_path: str):
        self.resident = torch.device("cpu")

        def want(key: str) -> bool:
            return ".txtmlp." in key or ".txtfusion." in key

        raw = _collect(checkpoint_path, want)
        if not raw:
            raise RuntimeError(f"No txtfusion/txtmlp tensors found in {checkpoint_path}")

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

        if "layerwise_blocks.0.prenorm.scale" not in fusion_sd:
            raise RuntimeError(
                f"txtfusion.layerwise_blocks.0.prenorm.scale not found in {checkpoint_path}"
            )
        txt_dim = fusion_sd["layerwise_blocks.0.prenorm.scale"].numel()
        self.txt_dim = txt_dim

        for name in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "norm.scale"):
            if name not in txt_sd:
                raise RuntimeError(f"txtmlp.{name} not found in the checkpoint")
        features = txt_sd["fc1.weight"].shape[0]

        qk = fusion_sd["layerwise_blocks.0.attn.qknorm.qnorm.scale"].numel()
        if txt_dim % qk:
            raise RuntimeError(f"txt_dim={txt_dim} not divisible by headdim={qk}")
        heads = txt_dim // qk
        kvheads = fusion_sd["layerwise_blocks.0.attn.wk.weight"].shape[0] // qk
        mult = _derive_multiplier(
            txt_dim, fusion_sd["layerwise_blocks.0.mlp.gate.weight"].shape[0]
        )
        bias = any(k.endswith(".bias") for k in fusion_sd)
        # Built on the meta device and filled with assign=True: the loaded bf16
        # tensors become the parameters. No randomly initialised fp32 copy of the
        # text path is ever allocated. The path is born on the CPU.
        with torch.device("meta"):
            self.fusion = _TextFusion(N_TAPS, txt_dim, heads, mult, bias, kvheads)
            self.txtmlp = _TxtMLP(txt_dim, features)
        missing, unexpected = self.fusion.load_state_dict(
            fusion_sd, strict=False, assign=True
        )
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

        self.txtmlp.load_state_dict(txt_sd, strict=True, assign=True)
        self.txtmlp.eval().requires_grad_(False)
        print(f"* Krea txtmlp: {txt_dim} -> {features}")

        # The weights are frozen, so the CPU copy stays valid for good: place()
        # uploads a second copy and release() just points back at the first one.
        # Nothing is ever downloaded from the GPU or reallocated in RAM.
        self._parameters = [*self.fusion.parameters(), *self.txtmlp.parameters()]
        self._cpu_data = [parameter.data for parameter in self._parameters]

    def place(self, device: torch.device) -> None:
        device = torch.device(device)
        if self.resident == device:
            return
        for parameter, master in zip(self._parameters, self._cpu_data):
            parameter.data = master.to(device)
        self.resident = device

    def release(self) -> None:
        """Drop the compute-device copy. Also undoes a place() that failed midway."""
        for parameter, master in zip(self._parameters, self._cpu_data):
            parameter.data = master
        self.resident = torch.device("cpu")

    def transform(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """(B, T, 12, D) -> (B, T, features): the conditioning the DiT consumes.

        `x` is used as scratch space: the layerwise blocks write into it.
        """
        b, l, n, d = x.shape
        if d != self.txt_dim:
            raise RuntimeError(
                f"Qwen hidden size {d} != txtfusion dim {self.txt_dim}. "
                "Heretic must be ablating the Qwen3-VL-4B this Krea checkpoint consumes."
            )

        flat = x.reshape(b * l, n, d)  # a view of x
        for start in range(0, b * l, FUSION_CHUNK_TOKENS):
            c = flat[start : start + FUSION_CHUNK_TOKENS]
            for block in self.fusion.layerwise_blocks:
                c = block(c, mask=None)
            flat[start : start + FUSION_CHUNK_TOKENS] = c
        del flat, c

        x = x.permute(0, 1, 3, 2)  # (B, T, D, N)
        x = self.fusion.projector(x).squeeze(-1)  # (B, T, D); frees the 4D buffer

        pieces = []
        for start in range(0, b, FUSION_CHUNK_PROMPTS):
            seg = x[start : start + FUSION_CHUNK_PROMPTS]
            mask = _attn_mask(token_mask[start : start + FUSION_CHUNK_PROMPTS])
            for block in self.fusion.refiner_blocks:
                seg = block(seg, mask=mask)
            pieces.append(self.txtmlp(seg))
        return torch.cat(pieces, dim=0)


def _shared_path(checkpoint_path: str) -> _TextPath:
    key = os.path.abspath(checkpoint_path)
    path = _PATHS.get(key)
    if path is not None:
        print("* Krea text path reused (one bf16 copy, parked on CPU)")
        return path
    path = _TextPath(checkpoint_path)
    _PATHS[key] = path
    return path


# ------------------------------------------------------------------- batching


class _TapsDone(Exception):
    """Ends the Qwen forward: nothing after the last tap feeds the score."""


class _Taps:
    """The tapped states of one forward, in one (B, T, 12, D) tensor."""

    def __init__(self):
        self.states: torch.Tensor | None = None
        self.n_filled = 0

    def put(self, slot: int, states: torch.Tensor, device: torch.device) -> None:
        if self.states is None:
            b, l, d = states.shape
            self.states = states.new_empty((b, l, N_TAPS, d), device=device)
        self.states[:, :, slot, :].copy_(states)
        self.n_filled += 1

    def take(self) -> torch.Tensor:
        """Hand the buffer over; this object keeps no reference to it."""
        if self.n_filled != N_TAPS:
            raise RuntimeError(f"Captured {self.n_filled} of {N_TAPS} taps")
        states, self.states = self.states, None
        return states


class _Batch:
    __slots__ = ("input_ids", "attention_mask", "token_mask", "indices")

    def __init__(self, input_ids, attention_mask, token_mask, indices):
        self.input_ids = input_ids            # (B, W) long, CPU
        self.attention_mask = attention_mask  # (B, W) long, CPU
        self.token_mask = token_mask          # (B, W - prefix) bool, CPU
        self.indices = indices                # list[int]: original prompt ids


class Settings(BaseModel):
    checkpoint_path: str = Field(
        description="Krea 2 safetensors file or shard directory. txtfusion/txtmlp must be float.",
    )
    max_length: PositiveInt = 512
    max_batch_tokens: PositiveInt = 4096
    measure_noise_floor: bool = Field(
        default=True,
        description="Measure per-prompt noise at start and filter by noise_limit.",
    )
    noise_limit: float = Field(
        default=0.01,
        gt=0.0,
        le=100.0,
        description=(
            "Largest per-prompt drift, in percent, accepted as numerical noise. "
            "Never enters the score. Prompts whose noise at start (unmodified "
            "model, different batching) exceeds it are dropped and replaced from "
            "the reserve. Also the limit of the padding probe, which aborts."
        ),
    )


class GoodSettings(Settings):
    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(dataset="GOOD_prompts.txt"),
        description="Prompts whose conditioning must stay unchanged.",
    )


class BadSettings(Settings):
    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(dataset="BAD_prompts.txt"),
        description="Prompts whose conditioning should move.",
    )


class _Krea2Scorer(Scorer):
    """Per-prompt drift of the conditioning Krea 2 Turbo feeds the DiT."""

    # Set by the subclasses.
    _largest: bool  # which end is trimmed: the largest drifts or the smallest
    _failure_value: float

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
            self._path = _shared_path(self.settings.checkpoint_path)

    def _unbind(self) -> None:
        """Drop every reference to the Heretic model."""
        self._outer = self._encoder = self._layers = self._rope_owner = None

    def _device(self) -> torch.device:
        return self._outer.get_input_embeddings().weight.device

    # No gc.collect() around these two: neither the early-stopped forward nor
    # Heretic's ARA step leaves reference cycles behind (tensors are freed by
    # reference counting), and a full collection costs time on every call.

    def _use_path(self) -> None:
        # Blocks cached by the modifier or other scorers go back to the driver
        # before ~0.8 GB of fusion weights are parked on the GPU.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._path.place(self._device())

    def _release_path(self) -> None:
        path = getattr(self, "_path", None)
        if path is None:
            return
        path.release()
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
    def _tap_hooks(self, taps: _Taps, device: torch.device, stop: bool):
        """Fill `taps` with the tapped states; `stop` ends the forward after the last."""

        def keep(slot: int):
            def hook(_module, _args, output):
                states = output[0] if isinstance(output, tuple) else output
                # Copied into the buffer at once: holding the slice itself (a
                # view) would keep all twelve full activations alive.
                taps.put(slot, states[:, PREFIX_TOKENS:, :], device)

            return hook

        def end(_module, _args, _output):
            raise _TapsDone

        handles = []
        try:
            for slot, idx in enumerate(TAPS):
                handles.append(self._layers[idx - 1].register_forward_hook(keep(slot)))
            if stop:
                # Registered after the taps, so it runs after the last one.
                handles.append(self._layers[TAPS[-1] - 1].register_forward_hook(end))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _condition(self, batch: _Batch) -> torch.Tensor:
        """(B, T, features): the conditioning of one batch, on the compute device."""
        device = self._device()
        taps = _Taps()
        with self._tap_hooks(taps, device, stop=True):
            with contextlib.suppress(_TapsDone):
                self._run(
                    batch.input_ids.to(device, non_blocking=True),
                    batch.attention_mask.to(device, non_blocking=True),
                )
        return self._path.transform(
            taps.take(), batch.token_mask.to(device, non_blocking=True)
        )

    # -------------------------------------------------------------- tokenize

    def _tokenize(self, tokenizer, texts: list[str]) -> list[list[int]]:
        # Prefix / suffix / pad_id are fixed for a tokenizer; cache after first call
        # (init top-ups would otherwise re-encode the same template strings).
        if not hasattr(self, "_prefix_ids"):
            prefix_ids = tokenizer([KREA_PREFIX], return_tensors="pt")["input_ids"][
                0
            ].tolist()
            if len(prefix_ids) != PREFIX_TOKENS:
                raise RuntimeError(
                    f"Krea prefix -> {len(prefix_ids)} tokens, expected {PREFIX_TOKENS}. "
                    "Wrong tokenizer, or the template string was edited."
                )
            suffix_ids = tokenizer([KREA_SUFFIX], return_tensors="pt")["input_ids"][
                0
            ].tolist()
            if len(suffix_ids) != SUFFIX_TOKENS:
                raise RuntimeError(
                    f"Krea suffix -> {len(suffix_ids)} tokens, expected {SUFFIX_TOKENS}."
                )
            self._prefix_ids = prefix_ids
            self._suffix_ids = suffix_ids
            self._pad_id = int(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
        prefix_ids = self._prefix_ids
        suffix_ids = self._suffix_ids
        block = self.settings.max_length + PREFIX_TOKENS - SUFFIX_TOKENS
        encoded = tokenizer(
            [KREA_PREFIX + t for t in texts],
            truncation=True,
            max_length=block,
            padding=False,
            return_attention_mask=False,
        )
        seqs, crossed, truncated = [], 0, 0
        for ids in encoded["input_ids"]:
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
        return seqs

    def _make_batch(self, indices: list[int]) -> _Batch:
        rows = [self._seqs[i] for i in indices]
        width, n = max(len(s) for s in rows), len(rows)
        input_ids = torch.full((n, width), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, width), dtype=torch.long)
        lengths: list[int] = []
        for r, seq in enumerate(rows):
            length = len(seq)
            input_ids[r, :length] = torch.as_tensor(seq, dtype=torch.long)
            attention_mask[r, :length] = 1
            lengths.append(length - PREFIX_TOKENS)
        if any(c < 1 for c in lengths):
            raise ValueError("every prompt must contain at least one valid token")
        positions = torch.arange(width - PREFIX_TOKENS).unsqueeze(0)
        token_mask = positions < torch.as_tensor(lengths).unsqueeze(1)
        return _Batch(input_ids, attention_mask, token_mask, list(indices))

    def _build_batches(
        self, budget: int | None = None, descending: bool = False
    ) -> list[_Batch]:
        budget = budget or self.settings.max_batch_tokens
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
            if current and too_big:
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

    def _load_pool(
        self, ctx: Context
    ) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
        """Return (active, reserve) as lists of (number, text).

        Both go through Heretic's loader, so prefix / suffix are applied alike.
        """
        spec = self.settings.prompts
        active, reserve = [], []
        used: set[tuple[str, int]] = set()
        spare: list[tuple[str, int, str]] = []
        for single in spec if isinstance(spec, list) else [spec]:
            selected = [p.user for p in ctx.load_prompts(single)]
            path, start = single.dataset, 0
            local_file = os.path.isfile(path)
            if local_file:
                # realpath also makes aliases / relative references to the same
                # local file share one source identity for overlap detection.
                path = os.path.realpath(path)
                if single.split is not None:
                    whole = single.model_copy(update={"split": None})
                    lines = [p.user for p in ctx.load_prompts(whole)]
                    start, end = get_split_slice(f"_{single.split}", len(lines))
                    spare += [(path, n, t) for n, t in enumerate(lines[end:], end + 1)]
            for number, text in enumerate(selected, start + 1):
                source_key = (path, number)
                # A list of specifications may contain overlapping slices of
                # one text file. Keep each original file line only once.
                if local_file and source_key in used:
                    continue
                active.append((number, text))
                if local_file:
                    used.add(source_key)
        # Overlapping specs must not score a line twice.
        for path, number, text in spare:
            if (path, number) not in used:
                used.add((path, number))
                reserve.append((number, text))
        return active, reserve

    def _top_up(self) -> int:
        """Refill the active set from the reserve. Returns the number added."""
        count = self._target - len(self._texts)
        added, self._reserve = self._reserve[:count], self._reserve[count:]
        self._labels += [number for number, _ in added]
        self._texts += [text for _, text in added]
        return len(added)

    def _setup(self, ctx: Context) -> None:
        print()
        print(
            f"Loading {self.score_name} prompts from "
            f"[bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        active, self._reserve = self._load_pool(ctx)
        if not active:
            raise ValueError(f"{self.score_name}: empty prompt dataset")
        self._labels = [number for number, _ in active]
        self._texts = [text for _, text in active]
        self._target = len(active)
        print(
            f"* Target [bold]{self._target}[/] captions, "
            f"reserve {len(self._reserve)}"
        )

        self._bind(ctx)
        tokenizer = ctx.get_model().tokenizer
        self._seqs = self._tokenize(tokenizer, self._texts)
        self._batches = self._build_batches()
        self._verify_taps()
        self._use_path()
        self._verify_padding()
        print("* Capturing reference conditioning from the base model...")
        self._baseline = self._capture_baseline()
        if self.settings.measure_noise_floor:
            self._filter_noisy_prompts(tokenizer)
        discarded = len(self._texts) * TRIM_PERCENT // 100
        print(
            f"* Using [bold]{len(self._texts)}[/] prompts (target {self._target}); "
            f"the score discards {discarded}"
        )
        # Only needed to select prompts and build batches. Scoring uses the
        # batches, the baseline and the prompt numbers.
        del self._texts, self._seqs, self._reserve, self._target

    def _filter_noisy_prompts(self, tokenizer) -> None:
        """Drop prompts above noise_limit, refill, rebuild, measure again."""
        limit = float(self.settings.noise_limit)
        number = 0
        while True:
            number += 1
            noise, regrouped = self._batch_noise()
            if not all(math.isfinite(v) for v in noise):
                raise RuntimeError("Non-finite conditioning on the unmodified model")
            print(
                f"* Noise floor (unmodified model, regrouped batches): max "
                f"[bold]{max(noise):.4f}%[/], noise_limit {limit:.4f}%"
            )
            if not regrouped:
                print(
                    "[yellow]* Every prompt kept its batch-mates, so this noise "
                    "figure is not informative.[/]"
                )

            drop = [i for i, v in enumerate(noise) if v > limit]
            if not drop:
                return
            print(
                f"[yellow]* Skipping {len(drop)} prompt(s) above noise_limit "
                f"{limit:.4f}% (pass {number}):[/]"
            )
            for i in sorted(drop, key=noise.__getitem__, reverse=True):
                text = " ".join(self._texts[i].split())
                if len(text) > 80:
                    text = text[:77] + "..."
                print(
                    f"  [yellow]#{self._labels[i]}  {noise[i]:.4f}%[/]  {escape(text)}"
                )
            dropped = set(drop)
            keep = [i for i in range(len(self._texts)) if i not in dropped]
            self._texts = [self._texts[i] for i in keep]
            self._seqs = [self._seqs[i] for i in keep]
            self._labels = [self._labels[i] for i in keep]

            added = self._top_up() if number <= NOISE_TOPUP_PASSES else 0
            if added:
                print(
                    f"* Topped up +{added} from the reserve "
                    f"(now {len(self._texts)}, target {self._target}, "
                    f"reserve left {len(self._reserve)})"
                )
            if not self._texts:
                raise RuntimeError(
                    f"{self.score_name}: no prompts left under noise_limit "
                    f"{limit:.4f}%. Fix the model dtype/backend or raise noise_limit."
                )
            if added:
                self._seqs += self._tokenize(tokenizer, self._texts[-added:])
            self._batches = self._build_batches()
            self._baseline = []  # free the old one before building the new one
            self._baseline = self._capture_baseline()

    def _verify_taps(self) -> None:
        """The hooks must see exactly what output_hidden_states returns."""
        longest = max(range(len(self._seqs)), key=lambda i: len(self._seqs[i]))
        probe = self._make_batch([longest])
        device = self._device()
        taps = _Taps()
        with self._tap_hooks(taps, device, stop=False):
            hidden = self._run(
                probe.input_ids.to(device),
                probe.attention_mask.to(device),
                output_hidden_states=True,
            ).hidden_states
        if hidden is None or len(hidden) != len(self._layers) + 1:
            raise RuntimeError(
                f"Expected {len(self._layers) + 1} hidden_states (embeddings + layers), "
                f"got {None if hidden is None else len(hidden)}"
            )
        states = taps.take()
        for slot, idx in enumerate(TAPS):
            expected = hidden[idx][:, PREFIX_TOKENS:, :].to(device)
            if not torch.equal(states[:, :, slot, :], expected):
                raise RuntimeError(f"Tap hook does not match hidden_states[{idx}]")
        print(f"* Tap hooks match hidden_states {list(TAPS)} exactly")

    def _verify_padding(self) -> None:
        """Fused conditioning of a short prompt must not depend on its padding."""
        order = sorted(range(len(self._seqs)), key=lambda i: len(self._seqs[i]))
        short, long_ = order[0], order[-1]
        if len(self._seqs[short]) == len(self._seqs[long_]):
            print("* Padding probe skipped: all prompts have the same length")
            return
        solo_batch = self._make_batch([short])
        solo = self._condition(solo_batch)
        pair = self._condition(self._make_batch([short, long_]))
        n = int(solo_batch.token_mask[0].sum().item())
        valid_tokens = torch.ones((1, n), dtype=torch.bool, device=solo.device)
        cos = _mean_token_cosine(
            pair[0:1, :n, :], solo[0:1, :n, :], valid_tokens
        )
        drift = 100.0 * (1.0 - float(cos[0].item()))
        del solo, pair, valid_tokens, cos
        limit = float(self.settings.noise_limit)
        print(f"* Padding probe: drift {drift:.4f}%, noise_limit {limit:.4f}%")
        # A prompt's drift is a mean over its tokens, so the probe is judged alike.
        if drift > limit:
            raise RuntimeError(
                f"Padding changes fused conditioning by {drift:.4f}% "
                f"(> noise_limit {limit:.4f}%). Refusing to score: batch "
                "composition would leak into the score."
            )

    def _batch_noise(self) -> tuple[list[float], bool]:
        """Noise floor of the score, per prompt, in percent.

        Drift of the UNMODIFIED model when the same prompts are batched
        differently (half the token budget, opposite length order): other
        neighbours, other padded width, other GEMM shapes, other rounding.
        Also returns whether any batch composition changed (if none did, the
        measurement says nothing).
        """
        where: dict[int, tuple[int, int]] = {}
        for b, batch in enumerate(self._batches):
            for row, prompt in enumerate(batch.indices):
                where[prompt] = (b, row)
        base_sets = [set(batch.indices) for batch in self._batches]
        device = self._device()
        noise = [0.0] * len(self._texts)
        cos_parts: list[torch.Tensor] = []
        order: list[int] = []
        regrouped = False
        alt = self._build_batches(
            budget=max(1, self.settings.max_batch_tokens // 2), descending=True
        )
        for batch in alt:
            members = set(batch.indices)
            cur = self._condition(batch)
            # Gather baseline rows into one (B, T, F) tensor, then one cosine
            # over the whole alt batch (same metric as before; fewer kernels).
            bsz, width, features = cur.shape
            # Zeros on padding; only valid tokens (token_mask) enter the cosine.
            ref = cur.new_zeros((bsz, width, features))
            for row, prompt in enumerate(batch.indices):
                b, base_row = where[prompt]
                regrouped = regrouped or members != base_sets[b]
                n = int(batch.token_mask[row].sum().item())
                ref[row, :n].copy_(
                    self._baseline[b][base_row, :n].to(device, non_blocking=True)
                )
            cos = _mean_token_cosine(
                cur, ref, batch.token_mask.to(device, non_blocking=True)
            )
            cos_parts.append(cos)
            order += batch.indices
            del cur, ref
        # One transfer for all prompts instead of a sync per prompt.
        cos_values = torch.cat(cos_parts).cpu().tolist()
        for prompt, cos in zip(order, cos_values):
            noise[prompt] = max(0.0, 100.0 * (1.0 - cos))
        return noise, regrouped

    # -------------------------------------------------------------- baseline

    @staticmethod
    def _to_cpu(t: torch.Tensor) -> torch.Tensor:
        if t.device.type != "cuda":
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
            fused_list.append(self._to_cpu(self._condition(batch)))
        return fused_list

    # --------------------------------------------------------------- scoring

    def _measure(self) -> torch.Tensor | None:
        """Per-prompt mean token cosine against the baseline. CPU float32."""
        device = self._device()
        cos_parts: list[torch.Tensor] = []
        order: list[int] = []
        for i, batch in enumerate(self._batches):
            cur = self._condition(batch)
            base_f = self._baseline[i].to(device, non_blocking=True)
            if cur.shape != base_f.shape:
                raise RuntimeError(
                    f"Fused shape {tuple(cur.shape)} vs baseline {tuple(base_f.shape)}"
                )
            cos = _mean_token_cosine(
                cur, base_f, batch.token_mask.to(device, non_blocking=True)
            )
            del cur, base_f
            cos_parts.append(cos)
            order += batch.indices
        # One transfer for all batches: reading each result back separately
        # would make the CPU wait for the GPU after every batch.
        fused = torch.empty(len(self._labels), dtype=torch.float32)
        fused[torch.as_tensor(order)] = torch.cat(cos_parts).cpu()
        if not torch.isfinite(fused).all():
            return None
        return fused

    def _score(self, value: float, detail: str) -> Score:
        text = f"{value:.4f}"
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
            return self._score(self._failure_value, "non-finite conditioning")
        # cos can exceed 1 by rounding; drift is never negative.
        drift = [max(0.0, 100.0 * (1.0 - c)) for c in cos.tolist()]
        value, extreme, top3 = _trimmed_extreme(drift, self._largest)
        name = "max" if self._largest else "min"
        tags = " ".join(f"#{self._labels[i]}" for i in top3)
        return self._score(
            value, f"{name}: {drift[extreme]:.4f} {tags}"
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return self._score(0.0, "baseline")


def _mean_token_cosine(
    current: torch.Tensor,
    reference: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean token cosine per prompt, from one pass over the two tensors.

    Inputs have shape (batch, tokens, features), contiguous (reshape must be a
    view, or it would copy). Returns float32 tensor of shape (batch,): mean over
    valid tokens of F.cosine_similarity(eps=1e-8). The caller turns it into
    100 * (1 - cos).

    Only tokens selected by token_mask contribute. Tokens are processed in
    chunks of DRIFT_CHUNK_ROWS rows, in FP32, so the temporaries do not grow with
    the batch. No host/device synchronisation happens here; prompts without a
    valid token are rejected on the CPU in _make_batch.
    """
    if current.ndim != 3 or reference.ndim != 3:
        raise ValueError("conditioning tensors must have shape (batch, tokens, features)")
    if current.shape != reference.shape:
        raise ValueError(
            f"conditioning shape mismatch: {tuple(current.shape)} vs {tuple(reference.shape)}"
        )
    if token_mask.shape != current.shape[:2]:
        raise ValueError(
            f"token mask shape {tuple(token_mask.shape)} does not match "
            f"conditioning {tuple(current.shape[:2])}"
        )

    batch_size, token_count, features = current.shape
    rows = batch_size * token_count
    cur2d = current.reshape(rows, features)
    ref2d = reference.reshape(rows, features)

    cosine = torch.empty(rows, device=current.device, dtype=torch.float32)
    for start in range(0, rows, DRIFT_CHUNK_ROWS):
        end = min(start + DRIFT_CHUNK_ROWS, rows)
        cur = cur2d[start:end].float()
        ref = ref2d[start:end].float()
        cosine[start:end] = F.cosine_similarity(cur, ref, dim=-1, eps=1e-8)
        del cur, ref

    valid = token_mask.to(device=current.device, dtype=torch.bool).reshape(rows)
    zeros = torch.zeros_like(cosine)
    cosine_sum = torch.where(valid, cosine, zeros).view(batch_size, -1).sum(1)
    valid_count = valid.view(batch_size, -1).sum(1).float()
    return cosine_sum / valid_count


def _trimmed_extreme(
    values: list[float], largest: bool
) -> tuple[float, int, list[int]]:
    """Discard TRIM_PERCENT of the values from one end.

    Returns:
      - the most extreme value that is left (after discarding),
      - the index of the most extreme value overall (on the same end),
      - up to three indices of the most extreme values overall, ordered by
        drift (descending when largest=True, ascending when largest=False).
    """
    order = sorted(range(len(values)), key=values.__getitem__, reverse=largest)
    discarded = len(values) * TRIM_PERCENT // 100
    top3 = order[: min(3, len(order))]
    return values[order[discarded]], order[0], top3


class Krea2Good(_Krea2Scorer):
    """p90max drift of the good prompts, in percent (minimize)."""

    settings: GoodSettings

    _largest = True
    # Worst value when minimizing: above the largest possible drift (200%).
    _failure_value = 201.0

    @property
    def score_name(self) -> str:
        return "Krea2 good p90max"


class Krea2Bad(_Krea2Scorer):
    """p90min drift of the bad prompts, in percent (maximize)."""

    settings: BadSettings

    _largest = False
    # Worst value when maximizing. A broken model must never win this objective.
    _failure_value = 0.0

    @property
    def score_name(self) -> str:
        return "Krea2 bad p90min"
