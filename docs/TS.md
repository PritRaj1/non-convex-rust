# Tabu Search

<div align="center">

<img src="../examples/gifs/tabu_kbf.gif" width="200" alt="Tabu Example">

<p><b>Figure:</b> The Tabu Search algorithm is a local search method that maintains a tabu list to avoid revisiting recently explored solutions, helping the search escape local optima. The choice and management of the tabu list are central to the method's effectiveness.</p>

</div>

Multiple variants of tabu search are implemented:

- **Standard tabu search**: Uses a fixed size tabu list.
- **Reactive tabu search**: Uses a dynamic tabu list size.
- **Frequency-based tabu search**: Tracks solution visit frequency to prevent over-exploration.
- **Quality-based tabu search**: Maintains quality memory for better solution evaluation.


## Guidance

| Feature | Standard TS | Reactive TS | Frequency TS | Quality TS |
|---------|-------------|-------------|--------------|------------|
| Tabu List Size | Fixed | Dynamic | Frequency-based | Quality-based |
| Memory Usage | Constant | Variable | Frequency tracking | Quality memory |
| Parameter Tuning | Manual | Semi-automatic | Adaptive | Adaptive |
| Adaptation | None | Tabu size only | Full adaptation | Full adaptation |
| Escape Mechanism | Basic | Enhanced | Frequency control | Quality guidance |
| Best For | Well-understood problems | Variable landscapes | High-dimensional | Quality-focused |

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "TS": {
            "common": {
                "num_neighbors": 50,
                "step_size": 0.1,
                "perturbation_prob": 0.3,
                "tabu_list_size": 20,
                "tabu_threshold": 1e-6
            },
            "list_type": {
                "FrequencyBased": {
                    "frequency_threshold": 3,
                    "max_frequency": 10
                }
            },
            "advanced": {
                "adaptive_parameters": true,
                "aspiration_criteria": true,
                "neighborhood_strategy": {
                    "Mixed": {
                        "strategies": [
                            {"Uniform": {"step_size": 0.1, "prob": 0.3}},
                            {"Gaussian": {"sigma": 0.05, "prob": 0.2}},
                            {"Cauchy": {"scale": 0.03, "prob": 0.1}}
                        ]
                    }
                },
                "restart_strategy": {
                    "Stagnation": {
                        "max_iterations": 100,
                        "threshold": 1e-6
                    }
                },
                "intensification_cycles": 5,
                "diversification_threshold": 0.1,
                "success_history_size": 20,
                "adaptation_rate": 0.1
            }
        }
    }
}
```

Default values, (only choices of tabu list must be specified):

```json
{
    "alg_conf": {
        "TS": {
            "list_type": {
                "Standard": {}
            }
        }
    }
}
```

## Sources and more information

- [Tabu Search](https://ieeexplore.ieee.org/document/9091743)
- [Reactive Tabu Search](https://doi.org/10.1287/ijoc.6.2.126)
- [Adaptive Memory Programming, (frequency and quality tracking)](https://www.sciencedirect.com/science/article/abs/pii/S037722170000268X)