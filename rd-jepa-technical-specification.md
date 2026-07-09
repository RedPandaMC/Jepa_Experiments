# Technical Specification: Recurrent Deliberation JEPA (RD-JEPA)

## 1. Project Overview & Vision
**RD-JEPA** is a latent-space world model that utilizes **test-time compute** to perform iterative physical refinement. Instead of a single forward pass, it runs an internal "thinking loop" that applies additive momentum and subtractive pruning to solve physical constraints. By treating future state prediction as a dynamic optimization problem, it dynamically adjusts its deliberation depth to ensure stable, realistic predictions.

**Target Hardware:** Consumer GPU (NVIDIA RTX 3070, 8GB VRAM).
**Target Environment:** 2D Physical Reasoning Environments (PhyRE).

---

## 2. Core Architecture

The model is built on the Joint-Embedding Predictive Architecture (JEPA) paradigm but replaces the standard feed-forward predictor with a **Recurrent Deliberation Loop**.

### 2.1. Vision Backbone (The Encoders)
Given the strict 8GB VRAM constraint, we avoid heavy Vision Transformers (ViTs).
* **Context Encoder ($E_	heta$):** A lightweight convolutional network (e.g., ResNet-18 or a customized 4-layer CNN) that maps the current world state $s_t$ into a compact latent vector $h_0 \in \mathbb{R}^d$ (where $d = 256$ or $512$).
* **Action Encoder ($A_\phi$):** Maps the proposed action $a_t$ (e.g., dropping a ball at coordinates x, y) into the same latent space.

### 2.2. The Deliberation Loop (The Lens)

RD-JEPA does not predict $s_{t+1}$ directly. Instead it iteratively **focuses** a latent state until it is physically sharp, using a single shared refinement function applied $K$ times. We call this the **Lens Paradigm**.

**The camera-lens analogy.** A standard $K$-step predictor throws away the whole camera at each step and rebuilds the world from scratch. RD-JEPA keeps one camera and *twists the same lens* $K$ times: each tiny twist carves a physical impossibility away until the latent image is sharp and viable.

**The lens is a Shared Refinement Function $F_\theta$.** The same small set of weights is reused at every step $k$ — there is no per-step parameter growth. This is the design choice that keeps VRAM **constant** regardless of $K$: whether the model thinks for 3 steps or 50, the memory footprint does not move, which is what makes the loop tractable on 8GB (see §3.1).

**The lens outputs a residual delta, not a new state.** At each deliberation step $k \in \{1, 2, ..., K\}$ the refinement function produces one tiny correction composed of two fused phases, and the new latent is a residual update of the previous one:

`h_k = h_{k-1} + tanh(M_k \odot h_add_k)`

where $F_\theta(h_{k-1}, a_t)$ computes:

1.  **Additive Phase (Extrapolation) — the lens gathers momentum.**
    Predicts the raw momentum-based future from the current blurry state.
    `h_add_k = MLP_add(h_{k-1}, a_t)`
2.  **Subtractive Phase (Pruning/Masking) — the lens filters hallucinations.**
    Projects a mask that identifies and suppresses physical violations (e.g., overlapping solids). The mask is the final glass of the lens.
    `M_k = Gate(MLP_mask(h_add_k))`

**Why this changes everything.**
* *Constant VRAM* — shared weights mean the 8GB budget holds for any $K$.
* *Latent gradient descent* — the loop is not guessing the future; it is performing an optimization *inside* latent space, tweaking the representation until the "energy" of physical violations (measured by $V_\psi$, §2.3) approaches zero. RD-JEPA becomes an iterative **constraint solver**, not a brute-force sequence generator.
* *Focusing implies early exit* — a lens stops twisting once the image is in focus. Because each step is a monotonic residual refinement, the violation head can terminate the loop as soon as the state is physically sound: trivial problems stop at $k\!\approx\!2$, hard ones run the full $K$. Dynamic thinking (§2.3) emerges for free from this residual setup.

### 2.3. Dynamic Depth & Early Exit
To mimic System 2 thinking, the loop evaluates its own physical stability.
* **Energy/Violation Head ($V_\psi$):** A lightweight scalar head predicting the "physical error" of $h_k$.
* If $V_\psi(h_k) < 	au$ (where $	au$ is a hyperparameter threshold), the loop terminates early — **the lens is in focus** — saving compute. Complex puzzles run for the full $K$ steps.

---

## 3. Training & Optimization Strategy

Training a recurrent loop on an 8GB GPU requires aggressive memory management.

### 3.1. VRAM Survival Implementation
* **Gradient Checkpointing:** Use `torch.utils.checkpoint` inside the $K$-loop. Intermediate activations of the $K$ steps are discarded and recomputed during the backward pass.
* **Automatic Mixed Precision (AMP):** All forward/backward passes must be executed in `torch.float16` or `torch.bfloat16` using `torch.cuda.amp.autocast`.
* **Truncated Backpropagation Through Time (TBPTT):** For high $K$ values (e.g., $K > 15$), gradients are detached every $n$ steps to prevent Out-Of-Memory (OOM) errors.

### 3.2. Loss Functions
As a JEPA, the model is trained *without* decoding back to pixels.
1.  **Latent Reconstruction Loss:** $L_{sim} = || h_{final} - 	ext{sg}(E_{ar{	heta}}(s_{t+1})) ||_2^2$ 
    *(where $	ext{sg}$ is stop-gradient and $E_{ar{	heta}}$ is an Exponential Moving Average (EMA) of the encoder).*
2.  **Contrastive Refinement Loss (Optional/Research):** Penalize the model if the intermediate states $h_k$ do not show monotonically decreasing physical violation scores.

---

## 4. Execution Plan (Phased Delivery)

* **Phase 1: Environment & Baseline**
    * Set up the PhyRE dataset.
    * Downsample inputs to $64 	imes 64$ for rapid batching.
    * Train a baseline 1-step MLP predictor to establish a performance floor.
* **Phase 2: Core Network Build**
    * Implement ResNet-18 context encoder and EMA target encoder.
    * Implement the Additive and Subtractive MLPs.
* **Phase 3: The Refinement Loop**
    * Wrap the MLPs in the checkpointed $K$-step loop.
    * Implement the BPTT mechanics and verify VRAM usage on the RTX 3070.
* **Phase 4: Early Exit & Dynamics**
    * Train the Violation Head.
    * Evaluate dynamically varying $K$ across easy vs. hard PhyRE tasks.

---

## 5. Open Decisions for Research & Implementation

Before writing the final training loop, the following architectural decisions need to be tested and locked in:

### Decision 1: The Subtractive Masking Mechanism
* **Option A:** Standard `Sigmoid` gating (values between 0 and 1). Soft, easily differentiable, but might not aggressively zero-out impossible physics.
* **Option B:** `Sparsemax` or `Gumbel-Softmax` for hard, discrete pruning of latent dimensions representing physical space. 
* *Research Task:* Train both on a 500-sample PhyRE subset and compare the gradient flow and validation loss.

### Decision 2: Injecting the Action ($a_t$)
* **Option A:** Concatenate $a_t$ only at the beginning ($h_0$).
* **Option B:** Inject $a_t$ continuously at every step $k$ in the Additive Phase. Continuous injection might keep the model grounded, but could over-parameterize the loop.
* *Research Task:* A/B test condition injection methods.

### Decision 3: Loss Trajectory over $K$ steps
* **Option A:** Apply the JEPA loss *only* to the final output $h_K$.
* **Option B:** Apply a discounted loss to *all* intermediate steps $h_k$ to encourage the model to reach the correct state as fast as possible (e.g., $L_{total} = \sum_{k=1}^K \gamma^{K-k} L_{sim}(h_k)$).
* *Research Task:* Monitor if Option B causes mode collapse or improves gradient stability through the deep loop.

### Decision 4: Spatial Inductive Biases
* Since PhyRE relies heavily on 2D coordinates, should the latent vector $h$ be a flat 1D vector (e.g., `shape=[B, 256]`), or should it retain a spatial grid structure (e.g., `shape=[B, 64, 8, 8]`) during the thinking loop? Grid structures preserve spatial locality but drastically increase computation in the Subtractive Phase.
* *Research Task:* Evaluate memory constraints of a spatial-latent vs. flat-latent loop.
