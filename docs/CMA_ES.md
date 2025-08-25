# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

<div align="center">

<img src="../examples/gifs/cmaes_kbf.gif" width="200" alt="CMA-ES Example">

<p><b>Figure:</b> CMA-ES is a stochastic, derivative-free algorithm for difficult non-linear non-convex problems in continuous domains. It's good for ill-conditioned and non-separable problems.</p>

</div>


## Flow

It maintains and adapts a covariance matrix in eigendecomposition form, $C = B \times D^2 \times B^T$, capturing the shape of the search distribution and enabling sampling by, $x = \text{mean} + \sigma \times B \times D \times z$, (where $z$ is Gaussian noise). It tracks evolution paths (pc for covariance, ps for step size) and uses this to adapt C using rank-1 (pc·pcᵀ) and rank-μ (weighted differences) updates.

### Evolution Paths

| Path | Tracks | Update | Notes |
|------|---------|----------------|----------------|
| **pc** (covariance) | Tracks direction of successful steps | $\mathbf{p}_c = (1-c_c) \cdot \mathbf{p}_c +$ hsig $\cdot \sqrt{c_c(2-c_c)\mu_{eff}} \cdot \mathbf{y}$ | • $c_c$: learning rate<br>• $\mu_{eff}$: effective selection mass<br>• hsig: Heaviside function |
| **ps** (step size) | Tracks magnitude of successful steps | $\mathbf{p}_s = (1-c_s) \cdot \mathbf{p}_s + \sqrt{c_s(2-c_s)\mu_{eff}} \cdot B \cdot D^{-1} \cdot B^T \cdot \mathbf{y}$ | • $c_s$: learning rate<br>• $B \cdot D^{-1} \cdot B^T$: transformed step<br>• $\mathbf{y}$: normalized step vector |

**Note**: 
```math
y = \frac{m_{new} - m_{old}}{\sigma}, \text{hsig} = \begin{cases} 1 & \text{if } \frac{\|\mathbf{p}_s\|}{\chi} > 1.4 \\ 0 & \text{otherwise} \end{cases}
```

### Covariance Matrix Update

**Formula**: 
```math
C = (1-c_1-c_\mu) \cdot C + c_1 \cdot \mathbf{p}_c \mathbf{p}_c^T + c_\mu \cdot \sum_{i=1}^{\mu} w_i \mathbf{y}_i \mathbf{y}_i^T
```

| Component | Purpose | Mathematical Form |
|-----------|---------|-------------------|
| **Decay** | Maintains memory while allowing adaptation | $(1-c_1-c_\mu) \cdot C$ |
| **Rank-1** | Directional adaptation using evolution path | $c_1 \cdot \mathbf{p}_c \mathbf{p}_c^T$ |
| **Rank-μ** | Population-based adaptation using weighted differences | $c_\mu \cdot \sum_{i=1}^{\mu} w_i \mathbf{y}_i \mathbf{y}_i^T$ |
| **Factorization** | Efficient sampling and updates | $C = B \cdot D^2 \cdot B^T$ |

**Where**: B = eigenvectors, D = $\sqrt{\text{eigenvalues}}$, $w_i$ = recombination weights 


```
                          CMA-ES Algorithm Flow

    ┌─────────────────────────────────────────────────────────────┐
    │                      INIT                                   │
    │  • Population size: λ offspring, μ parents                  │
    │  • Mean vector: m = initial guess                           │
    │  • Covariance matrix: C = Identity                          │
    │  • Step size: σ = initial_sigma                             │
    │  • Evolution paths: pc = 0, ps = 0                          │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    SAMPLE GENERATION                        │
    │                                                             │
    │   z₁, z₂, ..., zλ ~ N(0,I)   ◄─── Random vectors            │
    │             │                                               │
    │             ▼                                               │
    │   x₁ = m + σ·B·D·z₁         ◄─── Transform to problem       │
    │   x₂ = m + σ·B·D·z₂              space using current        │
    │   ...                            covariance structure       │
    │   xλ = m + σ·B·D·zλ                                         │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 EVALUATION & SELECTION                      │
    │                                                             │
    │   f₁ = f(x₁), f₂ = f(x₂), ..., fλ = f(xλ)                   │
    │             │                                               │
    │             ▼                                               │
    │   Sort by fitness: x₁ ≥ x₂ ≥ ... ≥ xλ                       │
    │             │                                               │
    │             ▼                                               │
    │   Select μ best: x₁, x₂, ..., xμ                            │
    │   Assign weights: w₁ > w₂ > ... > wμ                        │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    RECOMBINATION                            │
    │                                                             │
    │   m_new = Σᵢ₌₁μ wᵢ · xᵢ    ◄─── Weighted average of         │
    │                                  best solutions             │
    │   y = (m_new - m_old) / σ   ◄─── Normalized step            │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   ADAPTATION                                │
    │                                                             │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
    │  │ Evolution Paths │  │ Covariance      │  │ Step Size   │  │
    │  │                 │  │ Matrix Update   │  │ Control     │  │
    │  │ ps = (1-cs)·ps  │  │                 │  │             │  │
    │  │    + √(...)·y   │  │ C += c₁·pc·pcᵀ  │  │ σ *= exp(   │  │
    │  │                 │  │                 │  │   (‖ps‖/χ   │  │
    │  │ pc = (1-cc)·pc  │  │ C += cμ·Σwᵢ·    │  │    - 1)·... │  │
    │  │    + hsig·√(...)│  │      yᵢ·yᵢᵀ     │  │         )   │  │
    │  │       ·y        │  │                 │  │             │  │
    │  └─────────────────┘  └─────────────────┘  └─────────────┘  │
    │           │                     │                   │       │
    │           └─────────────────────┼───────────────────┘       │
    │                                 ▼                           │
    │                    ┌─────────────────────┐                  │
    │                    │ Eigendecomposition  │                  │
    │                    │   C = B·D²·Bᵀ       │                  │
    │                    │ B: eigenvectors     │                  │
    │                    │ D: √(eigenvalues)   │                  │
    │                    └─────────────────────┘                  │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Converged?    │
                    │ • Max iterations│
                    │ • Tolerance met │
                    │ • σ too small   │
                    └─────┬─────┬─────┘
                          │     │
                     No   │     │ Yes
                          │     ▼
                          │  ┌─────┐
                          │  │DONE │
                          │  └─────┘
                          │
                          └─────────┐
                                    │
     ┌──────────────────────────────┘
     │
     └─► Back to SAMPLE GENERATION
```

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "CMAES": {
            "num_parents": 50,
            "initial_sigma": 1.5,
            "use_active_cma": true,
            "active_cma_ratio": 0.25
        }
    }
}
```

Default values:

```json
{
    "alg_conf": {
        "CMAES": {}
    }
}
```

## Sources and more information

- [CMA-ES + great bibliography](https://cma-es.github.io/)
- [Power iteration eigen decomposition, (not used, but potential FPGA avenue)](https://en.wikipedia.org/wiki/Power_iteration)
- [Active cov adaption, (negative weights)](https://ieeexplore.ieee.org/document/1688662)
