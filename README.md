# RD-JEPA: Recurrent Deliberation JEPA

*A latent-space physics world model that buys prediction quality with test-time compute.*

**RD-JEPA** is a latent-space world model that utilizes **test-time compute**
to perform iterative physical refinement. Instead of a single feed-forward
pass, it runs an internal "thinking loop" — the **Lens** — that applies an
additive-then-subtractive refinement $K$ times until the latent is
physically sharp. Trained as a Joint-Embedding Predictive Architecture
(JEPA), it never decodes pixels during the thinking loop; an asynchronous
probing decoder provides an on-demand "viewport" for visualization.

> **Target.** Consumer GPU (NVIDIA RTX 3070, 8GB VRAM). Dataset: Kubric
> MOVi-A (pre-rendered rigid-body collisions of CLEVR-style shapes),
> passive video — no action modality.

<p align="center">
  <img src="docs/architecture.png" alt="RD-JEPA v2 architecture" width="100%"/>
</p>
<p align="sub"><b>Figure 1.</b> The RD-JEPA v2 pipeline. A spatial latent
<b>[B, 64, 4, 4]</b> is refined <i>K</i> times by a weight-shared Lens whose
subtractive phase is a CFD incompressibility projection (Navier–Stokes
masking). Energy, contrastive, and divergence regularizers prevent mode
collapse during BPTT. A separate VizDecoder probes the frozen <i>h<sub>K</sub></i>
for visualization without entangling gradients with the thinking loop.
[PDF](docs/architecture.pdf) · [SVG](docs/architecture.svg) · [source](docs/architecture.tex)</p>

---

## 1. Introduction

Standard feed-forward world predictors throw away the whole camera at each
step and rebuild the world from scratch. RD-JEPA keeps one camera and
**twists the same lens** $K$ times: each tiny twist carves a physical
impossibility away until the latent image is sharp and viable. The lens is a
single shared refinement function $F_\theta$ reused at every step, so VRAM
is **constant** in $K$ — whether the model thinks for 3 steps or 50, the
memory footprint does not move. This is what makes the loop tractable on
8 GB.

The loop is not guessing the future; it is performing an optimization
*inside* latent space, tweaking the representation until the "energy" of
physical violations approaches zero. RD-JEPA is an iterative **constraint
solver**, not a brute-force sequence generator. Because each step is a
monotonic residual refinement, a violation head $V_\psi$ can terminate the
loop as soon as the state is physically sound: trivial problems stop at
$k\!\approx\!2$, hard ones run the full $K$. **Dynamic thinking** emerges for
free from this residual setup.

### Contributions (v2)

The v2 architecture bakes three core fixes into a single unified model —
no ablation knobs — addressing the three weaknesses of the original design:

1. **Asynchronous probing decoder** — a lightweight RGB decoder trained in
   its own optimizer + backward pass on a *detached* $h_K$, every 4 JEPA
   steps. Zero gradient entanglement with the thinking loop; the JEPA acts
   as the physics engine and the decoder acts as the GPU renderer.
2. **Navier–Stokes masking** — the subtractive phase becomes a
   *Latent Divergence Projection*: fixed Sobel divergence + a learned
   per-sample projection scalar + L2 mass renormalization — the CFD
   incompressibility projection. Density is *redistributed*, never zeroed,
   so the lens handles fluids/deformation instead of just rigid bodies.
3. **Anti-collapse training** — Latent Energy Conservation
   ($|\,\|h_K\|-\|h_0\|\,|^2$), a Contrastive Dynamics margin loss gated
   by the grounded collision signal, a per-step Divergence Regularizer, and
   a curriculum-K schedule $K_{\min}\!\to\!K_{\max}$ that prevents the lens
   from collapsing to a static universe under BPTT.

## 2. Method

### 2.1 Vision backbone

A lightweight 4-layer strided CNN (Conv→GroupNorm→GELU, stride 2) ending in
a ConvNeXt-flavored depthwise block and a 1×1 conv head maps the stacked
context frames $(s_{t-1}, s_t)$ to a **spatial latent**
$h_0 \in \mathbb{R}^{C\times4\times4}$ ($C{=}64$, flat $d{=}1024$ for the
deliberation MLPs). Spatial structure is mandatory: the divergence
projection in the lens needs spatial axes to operate on.

An EMA copy $E_{\bar\theta}$ (decay $0.996$) of the encoder produces the
stop-gradient JEPA target from $(s_t, s_{t+1})$. There is **no action
modality** — MOVi is passive video, so the Lens refines a purely visual
latent.

### 2.2 The Lens: a recurrent refinement function $F_\theta$

At each deliberation step $k \in \{1,\dots,K\}$ the lens produces a residual
delta composed of two fused phases, framed as a fluid simulator:

1. **Additive phase (advection)** — the lens gathers momentum.
$$h^{\mathrm{add}}_k = \mathrm{MLP}_{\mathrm{add}}(h_{k-1})$$

2. **Subtractive phase (projection)** — the CFD incompressibility step. The
   additive update is reshaped to spatial $[\,C,4,4\,]$; a fixed Sobel
   kernel computes the discrete divergence $\nabla\!\cdot\, h^{\mathrm{add}}$;
   a per-channel density $\rho$ is read off and mapped through a small MLP
   to a learned per-sample projection scalar $\alpha\!\in\!(0,1)$; the
   divergence field is subtracted and the result is L2-renormalized so
   $\|h^{\mathrm{proj}}\|_2 \approx \|h^{\mathrm{add}}\|_2$:
$$h^{\mathrm{proj}}_k = \mathrm{Proj}_{\nabla\!\cdot}\!\big(h^{\mathrm{add}}_k\big)$$

3. **Residual update**:
$$h_k = h_{k-1} + \tanh\!\big(h^{\mathrm{proj}}_k\big)$$

The same weights are reused at every $k$ — **constant VRAM** regardless of
$K$ (Section 3.1). The lens cannot "cheat" by zeroing the latent to avoid
the physical-violation loss: mass conservation is built into the projection.

### 2.3 Dynamic depth & early exit

A lightweight scalar head $V_\psi$ predicts the "physical error" of $h_k$.
If $V_\psi(h_k) < \tau$, the loop terminates early — the lens is in focus —
saving compute. Complex scenes run the full $K$; quiet ones stop at
$k\!\approx\!2$. $V_\psi$ is trained both self-supervised (to predict the
residual latent error to the target) and grounded against MOVi's per-frame
collision-force magnitude (a genuine physics quantity, not a binary flag).

### 2.4 Loss functions

The JEPA core loss is **final-only** — MSE between $h_K$ and the
stop-gradient EMA target. (The discounted trajectory loss was removed in v2;
energy/divergence regularizers now supervise the whole trajectory instead.)

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

- **Gradient checkpointing** — `torch.utils.checkpoint` inside the $K$-loop;
  intermediate activations are discarded and recomputed during backward.
- **Automatic Mixed Precision** — bf16 autocast on Ampere.
- **Weight sharing** — the lens is one set of weights reused $K$ times, so
  activation memory is the only $K$-dependent cost (and checkpointing
  caps it).
- **Memory-fraction guard** — `set_per_process_memory_fraction(0.7)`.

### 3.2 Model size

| Component | Params |
|---|---|
| Context encoder $E_\theta$ | 0.49 M |
| EMA target encoder $E_{\bar\theta}$ | 0.49 M (frozen) |
| Lens $F_\theta$ (additive MLP + divergence projection) | 1.08 M |
| Violation head $V_\psi$ | 1.05 M |
| **JEPA total** | **3.11 M** |
| Asynchronous probing decoder | 0.21 M |

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
uv run aim up
```

### 4.4 Development

```bash
uv run ruff check . --fix   # lint + auto-fix
uv run pytest               # tests (14 tests, CPU, synthetic shards)
uv run mypy rd_jepa/        # type check
```

## 5. Repository Layout

```
rd_jepa/
├── config.py              single Config dataclass + resolve_K curriculum
├── losses.py              JEPA + energy + contrastive + divergence + VICReg
├── train.py               train_step / train_decoder_step / eval_step / train
├── data/loader.py         MoviTransitionDataset (v3 .npz cache)
├── models/
│   ├── rd_jepa.py         RDJEPA: encode -> K-loop -> early exit
│   ├── deliberation.py    DeliberationStep + DivergenceProjection + ViolationHead
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
tests/test_movi_pipeline.py  14 tests: data + model + losses + decoder contract
docs/
├── architecture.tex       Figure 1 source (TikZ)
├── architecture.pdf       Figure 1 (vector)
├── architecture.png      Figure 1 (raster, rendered @ 150 dpi)
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
- **Action modality removed** — MOVi is passive video; the Lens refines a
  purely visual latent.

## 7. v2 Breaking Changes

- `Config(K=15, ...)` no longer works — use `K_max=15`. The removed ablation
  fields (`gate`, `latent_shape`, `loss_trajectory`, `gamma`, `tbptt_n`,
  `K`) are **rejected**; the single unified architecture has no toggleable
  paths.
- Old `runs/*/ckpt.pt` files (FLAT latent + sigmoid gate) **will not load**
  into the v2 model — `DeliberationStep` now contains `DivergenceProjection`
  params (Sobel buffers, `mlp_alpha`) and the encoder is spatial-only.
  Start fresh from a new run.
- `VizDecoder.decoder_loss` no longer detaches internally — the caller
  (`train_decoder_step`) detaches `h_K` explicitly. Direct callers of
  `decoder_loss` must pass `h.detach()`.

## License

See [LICENSE](LICENSE).
