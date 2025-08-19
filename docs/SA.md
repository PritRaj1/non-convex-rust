# Simulated Annealing (SA)

<div align="center">

<img src="../examples/gifs/sa_kbf.gif" width="200" alt="SA Example">

<p><b>Figure:</b> The Simulated Annealing (SA) algorithm explores the search space  by accepting both improving and deteriorating moves with a probability that decreases over time, controlled by a temperature parameter. (Example above converged in 81 iters).</p>

</div>

## Generation Mechanisms

| Mechanism | Description | When Used | Advantages |
|-----------|-------------|-----------|------------|
| **Random Drift** | Gaussian perturbation around current position with adaptive step size | When gradients are unavailable | Simple, robust, decent for exploration |
| **MALA (Metropolis-Adjusted Langevin)** | Gradient-guided moves with noise correction | When gradients are available | More efficient, follows objective landscape |
| **Multi-Neighbor** | Generates multiple candidates per iteration | Always (configurable count) | Better exploration, higher acceptance probability |

## Acceptance Criteria

| Criterion | Formula | Description |
|-----------|---------|-------------|
| **Metropolis** | `P(accept) = min(1, exp(Δf / (T * k)))` | Standard SA acceptance for downhill moves |
| **MALA** | `P(accept) = min(1, exp(Δf / (T * k) + Langevin_correction))` | Gradient-aware acceptance with detailed balance |

## Cooling Schedules

| Schedule | Formula | Characteristics | Best For |
|----------|---------|----------------|----------|
| **Exponential** | `T(i) = T₀ * αⁱ` | Fast initial cooling, then gradual | Most problems, balanced exploration/exploitation |
| **Logarithmic** | `T(i) = T₀ / (1 + ln(i))` | Slower cooling, maintains exploration longer | Highly multimodal landscapes |
| **Cauchy** | `T(i) = T₀ / (1 + i)` | Very slow cooling, maximum exploration | Rugged landscapes, many local optima |
| **Adaptive** | Dynamic based on success rate | Self-adjusting temperature | Unknown landscapes, automatic tuning |

## Restart Strategies

| Strategy | Trigger Condition | Purpose |
|----------|-------------------|---------|
| **None** | Never | Continuous optimization without interruption |
| **Periodic** | Every N iterations | Escape local optima, maintain diversity |
| **Stagnation** | No improvement for N iterations | Restart when stuck in local optimum |
| **Diversity** | Population diversity below threshold | Restart when search becomes too focused |


## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "SA": {
            "initial_temp": 100.0,
            "cooling_rate": 0.95,
            "step_size": 0.1,
            "num_neighbors": 10,
            "x_min": -10.0,
            "x_max": 10.0,
            "min_step_size_factor": 0.1,
            "step_size_decay_power": 0.5,
            "min_temp_factor": 0.1,
            "use_adaptive_cooling": true,
            "advanced": {
                "restart_strategy": "None",
                "stagnation_detection": {
                    "stagnation_window": 50,
                    "improvement_threshold": 1e-8
                },
                "adaptive_parameters": true,
                "adaptation_rate": 0.15,
                "improvement_history_size": 50,
                "success_history_size": 50,
                "cooling_schedule": "Exponential"
            }
        }
    }
}
```

Default values:

```json
{
    "alg_conf": {
        "SA": {}
    }
}
```

## Sources and more information

- [Simulated Annealing](https://doi.org/10.1126/science.220.4598.671)
- [Metropolis-Hastings](http://www.jstor.org/stable/2280232)
- [MALA](https://doi.org/10.1063/1.436415)
- [Metropolis-Hastings step size adaptation](https://doi.org/10.1007/BF00143556)
- [Multi-neighbor generation](https://doi.org/10.1016/j.eswa.2024.124484)