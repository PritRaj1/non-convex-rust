# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

<div align="center">

<img src="../../../examples/gifs/cmaes_kbf.gif" width="200" alt="CMA-ES Example">

<p><b>Figure:</b> CMA-ES is a stochastic, derivative-free algorithm for difficult non-linear non-convex problems in continuous domains. It is particularly well suited for ill-conditioned and non-separable problems.</p>

</div>

## Flowchart

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
            "initial_sigma": 1.5
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
- [Power iteration eigen decomposition](https://en.wikipedia.org/wiki/Power_iteration)
