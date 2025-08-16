# Nelder-Mead

<div align="center">

<img src="../examples/gifs/nm_kbf.gif" width="200" alt="NM Example">

<p><b>Figure:</b> Nelder-Mead (NM) is a derivative-free optimizer that uses a Simplex (geometric figure with N+1 vertices in N-dimensional space) to crawl around the space. The most important part of setting up this method is the choice of the initial Simplex imo.</p>

|    Simplex operations     | What                                 | Why                        |
|--------------|---------------------------------------------|--------------------------------|
| Reflection   | Reflect worst point over centroid          | Explore new region             |
| Expansion    | Extend further past reflection             | Seek larger improvement        |
| Contraction  | Move worst point towards centroid          | Try smaller, safer step        |
| Shrink       | Pull all points towards best point         | Reset simplex, escape stagnation|


</div>

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "NM": {
            "common": {
                "alpha": 1.0,
                "gamma": 2.0,
                "rho": 0.5,
                "sigma": 0.5
            },
            "advanced": {
                "adaptive_parameters": true,
                "restart_strategy": {
                    "Stagnation": {
                        "max_iterations": 30,
                        "threshold": 1e-6
                    }
                },
                "stagnation_detection": {
                    "stagnation_window": 20,
                    "improvement_threshold": 1e-6,
                    "diversity_threshold": 1e-3
                },
                "coefficient_bounds": {
                    "alpha_bounds": [0.1, 3.0],
                    "gamma_bounds": [1.0, 5.0],
                    "rho_bounds": [0.1, 1.0],
                    "sigma_bounds": [0.1, 1.0]
                },
                "adaptation_rate": 0.1,
                "success_history_size": 20,
                "improvement_history_size": 30
            }
        }
    }
}
```

Default values, (nothing needs to be specified):

```json
{
    "alg_conf": {
        "NM": {}
    }
}
```

## Sources and more information

- [Simplex](https://doi:10.1093/comjnl/7.4.308)
- [Downhill Simplex visuals (for convex opt)](https://www.brnt.eu/phd/node10.html#SECTION00622200000000000000)
- [Nelder-Mead may converge to non-stationary points](https://doi:10.1137/S1052623496303482)
- [Adaptive Nelder-Mead in Matlab](https://www.researchgate.net/publication/225691623_Implementing_the_Nelder-Mead_simplex_algorithm_with_adaptive_parameters)