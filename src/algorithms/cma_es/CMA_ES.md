# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

<div align="center">

<img src="../../../examples/gifs/cmaes_kbf.gif" width="200" alt="CMA-ES Example">

<p><b>Figure:</b> CMA-ES is a stochastic, derivative-free algorithm for difficult non-linear non-convex problems in continuous domains. It is particularly well suited for ill-conditioned and non-separable problems.</p>

</div>

The eigen decomposition of the covariance matrix is computed using power iteration with Rayleigh quotient to improve convergence speed. Please note: this may not be the best method, (e.g. covariance matrix is symmetric so perhaps Lanczos iteration is better). LAPACK could also be used, but I wanted to learn about power iteration.

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
