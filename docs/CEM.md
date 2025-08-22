# Cross Entropy Method (CEM)

<div align="center">

<img src="../examples/gifs/cem_kbf.gif" width="200" alt="CEM Example">

<p><b>Figure:</b>The Cross Entropy Method (CEM) is a population-based, derivative-free algorithm that holds a distribution over samples and minimises the Kullback-Leibler divergence between the current sampling distribution and the optimal importance sampling distribution, (conditioned on the objective function).</p>

</div>

The Kullback-Leibler divergence between the current sampling distribution and the optimal importance sampling distribution is:

```math
KL(p_\theta \,\|\, p_{\theta^*}) = \int p_\theta(x) \log\left(\frac{p_\theta(x)}{p_{\theta^*}(x)}\right) dx
```

where:
- $p_\theta$ is the current sampling distribution
- $p_{\theta^*}$ is the optimal importance sampling distribution
- $\theta$ represents the distribution parameters (mean, covariance)

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "CEM": {
            "common": {
                "population_size": 100,
                "elite_size": 20,
                "initial_std": 2.0,
                "min_std": 1e-6,
                "max_std": 5.0
            },
            "sampling": {
                "use_antithetic": true,
                "antithetic_ratio": 0.3
            },
            "adaptation": {
                "smoothing_factor": 0.7
            },
            "advanced": {
                "use_restart_strategy": true,
                "restart_frequency": 50,
                "use_covariance_adaptation": true,
                "covariance_regularization": 1e-6
            }
        }
    }
}
```

Default values:

```json
{
    "alg_conf": {
        "CEM": {}
    }
}
```

