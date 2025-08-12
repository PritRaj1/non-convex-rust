# Parallel Tempering

<div align="center">

<img src="../../../examples/gifs/pt_kbf.gif" width="1000" alt="PT Example">

<p><b>Figure:</b> Parallel Tempering (Replica Exchange Monte Carlo) runs multiple copies ("replicas") of the system at different temperatures in parallel. Replicas periodically attempt to swap states, allowing high-temperature replicas to explore broadly and low-temperature replicas to refine solutions. Each replica can use different updates depending on the problem/conf setup.</p>

</div>

## Updates

| Type         | What                                                                 | When                                                                                 |
|---------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Metropolis-Hastings | Standard random-walk.                                              | Use when gradients are unavailable or the objective is non-differentiable.                  |
| MALA                | Metropolis-Adjusted Langevin Algorithm; uses gradient information.          | Use when gradients are available and you want faster mixing and more efficient exploration. |
| pCN                 | Preconditioned Crank-Nicolson uses a mix of preconditioned noise and old state.   | Use for high-dimensional problems, (see this [notebook](https://github.com/PritRaj1/hilbert-mcmc/blob/main/metropolis_hastings.ipynb) for an explanation). |

## Tempering

Temperatures are scheduled with a dynamic power law relationship:

```math
t_k = \left( \frac{k}{N_t} \right)^p, \quad k = 0, 1, \ldots, N_t, \quad t_k \in [0, 1], \quad p > 0
```

where $N_t$ is the number of temperatures and $p$ is a parameter that controls the rate of temperature increase. 

In early iterations, $p$ is set large (>1) to cluster more temperatures towards smoother replicas. As iterations progress, $p$ is decreased to <1 to shift more temperatures towards detailed replicas.

This is a personal trick I always use. It makes early iterations more explorative and later iterations more exploitative.

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "PT": {
            "common": {
                "num_replicas": 10,
                "power_law_init": 2.0,
                "power_law_final": 0.5,
                "power_law_cycles": 1,
                "alpha": 0.1,
                "omega": 2.1,
                "mala_step_size": 0.1
            },
            "swap_conf": {
                "Always": {}
                // or
                "Periodic": {
                    "swap_frequency": 1.0
                }
                // or
                "Stochastic": {
                    "swap_probability": 0.1
                }
            },
            "update_conf": {
                "MetropolisHastings": {
                    "random_walk_step_size": 0.1
                }
                // or
                "MALA": {
                    "step_size": 0.1,
                    "use_preconditioning": false
                }
                // or
                "PCN": {
                    "step_size": 0.1,
                    "preconditioner": 1.0
                }
                // or
                "Auto": {}
            }
        }
    }
}
```

Default values, (only choices of swap check must be specified): 

```json
{
    "alg_conf": {
        "PT": {
            "common": {},
            "swap_conf": {
                "Always": {}
            },
            "update_conf": {
                "Auto": {}
            }
        }
    }
}
```

## Sources and more information

- [Parallel Tempering](https://arxiv.org/abs/physics/0508111)
- [Chains might be initialized between 0 and 1 similar to simulated annealing](https://doi.org/10.13182/NT90-A34350)
- [Metropolis-Hastings](http://www.jstor.org/stable/2280232)
- [MALA](https://doi.org/10.1063/1.436415)
- [Metropolis-Hastings step size adaptation](https://doi.org/10.1007/BF00143556)
- [Power-law scheduling and minimizing KL divergences between temperature distributions](https://doi.org/10.1016/j.csda.2009.07.025)
- [Cyclic annealing-ish](https://arxiv.org/abs/1903.10145)
- [pCN and other tricks](https://arxiv.org/abs/1202.0709)