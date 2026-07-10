# RD-JEPA: Recurrent Deliberation JEPA

*A latent-space physics world model that buys prediction quality with test-time compute — via mutating convolution kernels that literally evolve per sample.*

**RD-JEPA** is a latent-space world model that utilizes **test-time compute**
to perform iterative physical refinement. Instead of a single feed-forward
pass, it runs an internal "thinking loop" — a **Kernel Lens** of N=4
depthwise conv kernels that **mutate per-sample** during $K$ deliberation
steps, carving physical impossibilities away until the latent is sharp.
Trained as a Joint-Embedding Predictive Architecture (JEPA), it never decodes
pixels during the thinking loop; an asynchronous probing decoder provides an
on-demand "viewport" for visualization.

> **Target.** Consumer GPU (NVIDIA RTX 3070, 8GB VRAM). Dataset: Kubric
> MOVi-A (pre-rendered rigid-body collisions of CLEVER-style shapes),
> passive video — no action modality.

<p align="center">
  <img src="docs/architecture.png" alt="RD-JEPA v3 architecture" width="100%"/>
</p>
<p align="sub"><b>Figure 1.</b> The RD-JEPA v3 pipeline. A spatial latent
<b>[B, 64, 4, 4]</b> (flat 1024 externally) is refined <i>K</i> times by a
<b>Kernel Lens</b> — a bank of N=4 depthwise conv kernels that mutate
per-sample via a mutator network (test-time compute has real meaning: different
inputs → different kernel trajectories). An attention gate selects which
kernels to activate at each step. Energy, contrastive, divergence, and kernel-
diversity regularizers prevent mode collapse during BPTT. A separate
VizDecoder probes the frozen <i>h<sub>K</sub></i> for visualization without
entangling gradients with the thinking loop.
<a href="docs/architecture.svg">SVG</a> · <a href="docs/architecture.drawio">source</a></p>

---

## 1. Introduction

Standard feed-forward world predictors throw away the whole camera at each
step and rebuild the world from scratch. RD-JEPA keeps one camera and
**twists a bank of N conv kernels** $K$ times: each tiny twist carves a
physical impossibility away until the latent image is sharp and viable. The
kernels are not static — they **mutate per-sample** based on the latent state,
so test-time compute is genuine computation, not just a repeated fixed
function. The kernel bank is one set of weights reused at every step, so
VRAM is **constant** in $K$ — whether the model thinks for 3 steps or 50, the
memory footprint does not move. This is what makes the loop tractable on 8 GB.

The loop is not guessing the future; it is performing an optimization
*inside* latent space, tweaking the representation until the "energy" of
physical violations approaches zero. RD-JEPA is an iterative **constraint
solver**, not a brute-force sequence generator. Because each step is a
monotonic residual refinement, a violation head $V_\psi$ can terminate the
loop as soon as the state is physically sound: trivial problems stop at
$k\!\approx\!2$, hard ones run the full $K$. **Dynamic thinking** emerges for
free from this residual setup.

### Contributions (v3)

The v3 architecture replaces the v2 MoE lens bank (N parallel MLPs + a
degenerate soft-routing router) with a **mutating kernel lens**:

1. **Kernels as lenses on the latent space** — each lens is a depthwise conv
   kernel `[C, 3, 3]` applied to the spatial latent `[B, C, 4, 4]` via
   `unfold` + `einsum` (per-sample, per-channel). This gives genuine spatial
   inductive bias (detecting/producing local field patterns) rather than
   unconstrained MLPs on a flat vector.
2. **Mutating kernels (test-time compute)** — the kernel state
   `[B, N, C, 3, 3]` is initialized from learned base kernels (seeded with
   Sobel-x, Sobel-y, Laplacian, identity priors) and **evolves per-sample**
   during the $K$ steps. A mutator network reads the pooled latent and
   produces per-sample kernel deltas each step — the kernels literally
   transform based on what the latent looks like. Different inputs → different
   kernel trajectories. The `step_scale` and `mutation_scale` are learnable
   and `tanh`-bounded.
3. **Attention gate, not MoE router** — a lightweight LayerNorm + MLP →
   softmax gate selects which kernels to activate at each step. No
   load-balance loss, no router-entropy loss. A `kernel_diversity_loss`
   (pairwise cosine similarity between base kernels) prevents all kernels
   from collapsing to identical filters.
4. **Asynchronous probing decoder** — a lightweight RGB decoder trained in
   its own optimizer + backward pass on a *detached* $h_K$, every 4 JEPA
   steps. Zero gradient entanglement with the thinking loop; the JEPA acts
   as the physics engine and the decoder acts as the GPU renderer.
5. **Anti-collapse training** — Latent Energy Conservation
   ($|\,\|h_K\|-\|h_0\|\,|^2$), a Contrastive Dynamics margin loss gated
   by the grounded collision signal, a per-step Divergence Regularizer, and
   a curriculum-K schedule $K_{\min}\!\to\!K_{\max}$ that prevents the lens
   from collapsing to a static universe under BPTT.

## 2. Method

### 2.1 Vision backbone

A lightweight 4-layer strided CNN (Conv→GroupNorm→GELU, stride 2) ending in
a ConvNeXt-flavored depthwise block and a 1×1 conv head maps the stacked
context frames $(s_{t-1}, s_t)$ to a **spatial latent**
$h_0 \in \mathbb{R}^{C\times4\times4}$ ($C{=}64$, flat $d{=}1024$). Spatial
structure is mandatory: the kernel lens operates on the 2D field directly.

An EMA copy $E_{\bar\theta}$ (decay $0.996$) of the encoder produces the
stop-gradient JEPA target from $(s_t, s_{t+1})$. There is **no action
modality** — MOVi is passive video, so the kernel lens refines a purely
visual latent.

### 2.2 The Kernel Lens: mutating depthwise conv kernels

At each deliberation step $k \in \{1,\dots,K\}$ the kernel lens applies N
depthwise conv kernels to the spatial latent, gates them by attention, and
mutates the kernels for the next step. The full step is:

**Step 1 — Per-sample depthwise convolution.** The flat latent is reshaped
to spatial $h_{\mathrm{sp}} \in \mathbb{R}^{B\times C\times4\times4}$ and
unfolding extracts sliding-window patches
$P \in \mathbb{R}^{B\times C\times k^2\times L}$. Each per-sample kernel
$K_n \in \mathbb{R}^{C\times3\times3}$ is applied via einsum:

$$\delta_n = \mathrm{einsum}\big(\text{`bnck,bckl\to bncl`},\; K_n,\; P\big) \quad\to\; [B, C, 16]$$

This is a per-sample, per-channel depthwise convolution — each sample has its
own evolved kernels, not shared ones.

**Step 2 — Attention gate.** The pooled latent $\bar{h} = \mathrm{mean}(h_{\mathrm{sp}}) \in \mathbb{R}^{B\times C}$
is passed through a LayerNorm + MLP → softmax to produce attention weights:

$$g = \mathrm{softmax}\!\Big(\frac{\mathrm{MLP}(\mathrm{LN}(\bar{h}))}{T}\Big) \in \mathbb{R}^N, \quad [B, N]$$

where $T$ is a learnable temperature (clamped $\geq 0.5$).

**Step 3 — Gated sum + residual update.** The per-kernel spatial deltas are
combined by the gate, bounded by `tanh`, and added residually:

$$h_k = h_{k-1} + \tanh(s_{\mathrm{step}}) \cdot \tanh\!\Big(\sum_{n=1}^{N} g_n \, \delta_n\Big)$$

where $s_{\mathrm{step}}$ is a learnable scale bounded by `tanh`.

**Step 4 — Kernel mutation (test-time compute).** A mutator network reads the
*updated* pooled latent and produces per-sample kernel deltas:

$$\Delta K = \tanh\big(s_{\mathrm{mut}}\big) \cdot \tanh\!\big(\mathrm{MLP}_{\mathrm{mut}}(\mathrm{mean}(h_k))\big)$$
$$K_k = K_{k-1} + \Delta K \quad\to\; [B, N, C, 3, 3]$$

The kernels **literally evolve** — this is the test-time compute: different
inputs produce different kernel trajectories. $s_{\mathrm{mut}}$ is a
learnable scale bounded by `tanh`. The mutated kernels feed into the next
step's convolution.

**Initialization.** The base kernels are seeded with physics-inspired spatial
operators — Sobel-x, Sobel-y, Laplacian, identity — so the bank starts with
meaningful edge/blob detectors rather than random noise, then adapts via
training + per-sample mutation.

### 2.3 Dynamic depth & early exit

A lightweight scalar head $V_\psi$ predicts the "physical error" of $h_k$.
If $V_\psi(h_k) < \tau$ (default $0.1$), the loop terminates early — the lens
is in focus — saving compute. Complex scenes run the full $K$; quiet ones
stop at $k\!\approx\!2$. $V_\psi$ is trained both self-supervised (to predict
the residual latent error to the target) and grounded against MOVi's per-frame
collision-force magnitude (a genuine physics quantity, not a binary flag).

### 2.4 Loss functions

The JEPA core loss is **final-only** — MSE between $h_K$ and the
stop-gradient EMA target. Energy/divergence regularizers supervise the whole
trajectory instead.

| Loss | Form | Weight | Purpose |
|---|---|---|---|
| **JEPA** | $\|h_K - \mathrm{sg}(\text{target})\|_2^2$ | 1.0 | latent prediction |
| **Energy conservation** | $\big|\,\|h_K\|_2 - \|h_0\|_2\,\big|^2$ | 0.1 | forbid zeroing the state |
| **Contrastive dynamics** | $\mathrm{ReLU}(m - \|h_K - h_0\|_2)\cdot\mathbf{1}[v_{gt}\!>\!0]$ | 0.05 | penalize stasis when a push existed |
| **Divergence regularization** | $\overline{\big|\,\|h_k\| - \|h_{k-1}\|\,\big|}$ | 0.05 | per-step constant density |
| **Violation aux** | $\mathrm{ReLU}(v_k - v_{k-1}).\mathrm{mean}()$ | 0.01 | monotonic focusing |
| **Violation self-sup** | $\mathrm{MSE}(v_k,\,\|h_k\!-\!\text{target}\|^2)$ | 0.1 | teach $V_\psi$ its own error |
| **Violation grounded** | $\mathrm{Smooth}\ell_1(v_K, v_{gt})$ | 0.1 | collision-force regression |
| **VICReg var + cov** | hinge std + off-diag covariance | 1.0 / 1.0 | collapse safety net |
| **Kernel diversity** | $\overline{\big|\cos(K_i^{\mathrm{base}}, K_j^{\mathrm{base}})\big|}$ | 0.01 | prevent filter collapse |

### 2.5 Asynchronous probing decoder

The JEPA loss never decodes pixels. A separate `VizDecoder` (4 conv-transpose
blocks, ~200K params) is trained in its **own optimizer + backward pass** on
a *detached* $h_K$, every `decoder_interval=4` JEPA steps — zero gradient
entanglement with the thinking loop. The decoder learns to reconstruct $s_t$
from the frozen latent, providing an on-demand "viewport" to see what the
model is imagining without forcing the JEPA to predict pixels during
deliberation.

### 2.6 Curriculum K

To prevent the gradients from vanishing into a collapsed state before the
lens knows how to focus, $K$ is not fixed at training start. A per-epoch
linear schedule ramps $K_{\min}\!=\!1 \to K_{\max}\!=\!15$ over
`curriculum_warmup_epochs=5` (`cfg.resolve_K(epoch)`). Both `train_step`
and `eval_step` use the epoch's $K_{\mathrm{epoch}}$.

## 3. Training & Optimization

### 3.1 VRAM survival

Training a recurrent loop on 8 GB requires aggressive memory management:

- **Gradient checkpointing** — `torch.utils.checkpoint` inside the $K$-loop
  (enabled by default in v3); intermediate activations are discarded and
  recomputed during backward.
- **Automatic Mixed Precision** — bf16 autocast on Ampere.
- **Weight sharing** — the kernel lens is one set of weights reused $K$ times,
  so activation memory is the only $K$-dependent cost (and checkpointing
  caps it).
- **Memory-fraction guard** — `set_per_process_memory_fraction(0.95)`.

### 3.2 Model size

| Component | Params |
|---|---|
| Context encoder $E_\theta$ | 0.49 M |
| EMA target encoder $E_{\bar\theta}$ | 0.49 M (frozen) |
| Kernel Lens (base kernels + gate + mutator) | ~0.33 M |
| Violation head $V_\psi$ | ~0.16 M |
| **JEPA total** | **~1.5 M** |
| Asynchronous probing decoder | 0.21 M |

The v3 kernel lens (~0.33M) is dramatically smaller than the v2 MoE lens
bank (~4.4M of N parallel MLPs + router), making it far more tractable on a
laptop GPU while gaining test-time compute via kernel mutation.

## 4. Quick Start

### 4.1 Environment

A single `uv`-managed Python 3.11 env (PyTorch + `tfrecord` + Pillow).
**No TensorFlow anywhere.**

```bash
uv sync
```

### 4.2 Data

MOVi-A tfds shards are parsed with the pure-Python `tfrecord` package and
emitted as v3 `.npz` caches (RGB frames + `violation_gt`).

```bash
# Easy way (recommended): builds train + dev with recommended defaults
# (50 train shards, force-scale 1.0, 256x256 frames) for an 8 GB laptop.
uv run python scripts/build_data.py
uv run python scripts/build_data.py --max-shards 10   # quick test
uv run python scripts/build_data.py --dev-only         # just the dev split
uv run python scripts/build_data.py --scan-scale        # tune force-scale

# Full control (single split, custom params): see scripts/build_data.py --help
uv run python scripts/build_data.py --tfds-split train --out-split train --force-scale 1.0
uv run python scripts/build_data.py --tfds-split validation --out-split dev
```

### 4.3 Training

```bash
# Easy way (recommended): every Config field is a CLI override (kebab-case).
# The full config is printed before training starts. Run with --help to see
# all options.
uv run python scripts/train.py
uv run python scripts/train.py --exp-name big --epochs 40 --batch-size 256
uv run python scripts/train.py --fast    # 500-sample smoke test

# Metrics + gif dashboard
aim up
```

### 4.4 Development

```bash
uv run ruff check . --fix   # lint + auto-fix
uv run pytest               # tests (22 tests, CPU, synthetic shards)
uv run mypy rd_jepa/        # type check
```

## 5. Repository Layout

```
rd_jepa/
├── config.py              single Config dataclass + resolve_K curriculum
├── losses.py              JEPA + energy + contrastive + divergence + VICReg + kernel diversity
├── train.py               train_step / train_decoder_step / eval_step / train
├── data/loader.py         MoviTransitionDataset (v3 .npz cache)
├── models/
│   ├── rd_jepa.py         RDJEPA: encode -> K-loop kernel lens -> early exit
│   ├── deliberation.py    KernelLens (mutating conv kernels) + ViolationHead
│   ├── encoder.py         spatial CNN encoder
│   └── ema.py             EMA target encoder (BYOL-style stop-grad)
├── eval/
│   ├── probe.py           linear probe train/eval on violation target
│   └── probe_module.py    ViolationProbe (MSE / R²)
└── viz/
    ├── decoder.py         VizDecoder + make_decoder_optimizer (async probing)
    ├── gif_writer.py      deliberation + rollout gif rendering
    └── aim_logger.py      Aim scalar/image logging
scripts/
├── build_data.py          MOVi tfds -> .npz cache (v3 format) + easy CLI
└── train.py               easy training entry point (all Config fields as CLI flags)
tests/test_movi_pipeline.py  22 tests: data + model + kernel lens + losses + decoder contract
docs/
├── architecture.drawio    Figure 1 source (draw.io)
├── architecture.png       Figure 1 (raster, rendered @ 3x scale)
└── architecture.svg       Figure 1 (vector, web-friendly)
data/cache/                MOVi transition cache (v3: RGB + violation_gt)
runs/ + .aim/              experiment outputs
```

## 6. Implementation Notes

- **One Python version** — `uv` manages only 3.11 (training AND data
  generation); the previous dual-Python PhyRE setup has been removed.
- **No TensorFlow** — MOVi tfds shards parsed with `tfrecord` (pure Python).
- **Cache format v3** — RGB frames `[H,W,3]` uint8 + `violation_gt` float
  (collision-force regression target), replacing PhyRE v2's scene-id maps +
  `solved` bool.
- **VICReg** — variance/covariance regularization prevents representation
  collapse (a safety net beyond the energy/contrastive terms).
- **Action modality removed** — MOVi is passive video; the kernel lens
  refines a purely visual latent.
- **Mutating kernel lens** — N=4 depthwise conv kernels that operate on the
  spatial latent and evolve per-sample during the K steps. A mutator network
  reads the pooled latent and produces per-sample kernel deltas each step;
  the kernels literally transform based on the input. An attention gate
  selects which kernels to activate; a `kernel_diversity_loss` prevents
  filter collapse. Seeded with Sobel-x, Sobel-y, Laplacian, identity priors.

## 7. v3 Breaking Changes

- `Config(n_lenses=4, ...)` no longer works — use `n_kernels=4`. The removed
  v2 fields (`n_lenses`, `load_balance_weight`, `router_entropy_weight`) are
  **rejected**. New fields: `n_kernels`, `kernel_size`,
  `kernel_diversity_weight`.
- Old `runs/*/ckpt.pt` files (MoE `LensBank` with `lens.lenses.{i}.*` +
  `lens.router.*` keys) **will not load** — `RDJEPA.lens` is now a
  `KernelLens` whose state-dict keys are `lens.base_kernels`, `lens.gate.*`,
  `lens.mutator.*`. Start fresh from a new run.
- `batch_size` reduced from 256 to 128; `grad_checkpoint` now defaults to
  `True` (laptop-friendly). `hidden_dim` reduced from 256 to 128.
- `latent_dim` fixed to 1024 (was incorrectly 512 in v2 despite the comment).
- `VizDecoder.decoder_loss` no longer detaches internally — the caller
  (`train_decoder_step`) detaches `h_K` explicitly. Direct callers of
  `decoder_loss` must pass `h.detach()`.

## License

See [LICENSE](LICENSE).
