# Continous Genetic Algorithm

<div align="center">

<img src="../examples/gifs/cga_kbf.gif" width="200" alt="CGA Example">

<p><b>Figure:</b> The Continous Genetic Algorithm (CGA) adapts the classic genetic algorithm for optimization in continuous (floating-point) search spaces. Increase mutation or crossover to increase exploration. Decrease for exploitation. Use adaptive to handle this auto-magically.</p>

</div>

## Selection

| Selection Methods         | Notes                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| Roulette Wheel | - Maintains diversity well                                                                    |
|                          | - Can be biased and not my go-to for constrained problems                                     |
| Tournament     | - More reflective of biological selection and works well for constrained problems             |
|                          | - Weaker individuals can be considered and tournament size must be tuned to maintain diversity|
| Residual (SRS)       | - My preferred method in statistical sampling, has good theoretical properties and maintains diversity |
|                          | - Somewhat deterministic and fast                                                                      |

## Crossover

| Crossover Methods         | Notes                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| Random | - Standard and boring |
| Heuristic     | - More exploitative and faster to converge (with good initial population) |
|      | - More suitable to continous problems (by intuition). Uses a blend of parent characteristics. |

## Mutation
| Mutation Methods         | Notes                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| Gaussian | - Standard mutation operator using normal distribution with mean 0 and configurable sigma                |
|          | - Good for local search and fine-tuning solutions by adding normally distributed noise                  |
| Uniform  | - Randomly replaces values with new ones uniformly sampled within bounds                               |
|          | - More disruptive than Gaussian, helps maintain diversity through complete value replacement           |
| Non-Uniform | - Decreasing mutation strength over generations based on generation number and shape parameter b     |
|             | - Allows broad exploration early and fine-tuning later through decreasing step sizes                |
| Polynomial | - Popular in multi-objective optimization, uses polynomial probability distribution                   |
|            | - Parameter eta_m controls distribution shape - larger values give smaller perturbations near parent  |

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "CGA": {
            "common": {
                "num_parents": 2,
                "adaptive_parameters": true, // adjusts mutation and crossover to balance exploration vs exploitation
                "success_history_size": 50,
                "adaptation_rate": 0.1
            },
            "selection": {
                "Tournament": { 
                    "tournament_size": 2
                }
                // or
                "Residual": {}
                // or
                "RouletteWheel": {}
            },
            "crossover": {
                "Heuristic": {
                    "crossover_prob": 0.8
                }
                // or
                "Random": {
                    "crossover_prob": 0.8
                }
                // or
                "SimulatedBinary": {
                    "crossover_prob": 0.8,
                    "eta_c": 15.0
                }
            },
            "mutation": {
                "NonUniform": {
                    "mutation_rate": 0.05,
                    "b": 5.0
                }
                // or
                "Gaussian": {
                    "mutation_rate": 0.05,
                    "sigma": 0.1
                }
                // or
                "Uniform": {
                    "mutation_rate": 0.05
                }
                // or
                "Polynomial": {
                    "mutation_rate": 0.05,
                    "eta_m": 20.0
                }
            }
        }
    }
}
```

Default values, (only choices of selection and crossover must be specified):

```json
{
    "alg_conf": {
        "CGA": {
            "common": {},
            "selection": {
                "Residual": {}
            },
            "crossover": {
                "Random": {}
            }
        }
    }
}
```

## Sources and more information

- [Continuous Genetic Algorithm](https://doi.org/10.1002/0471671746.ch3)
- [CGA is more akin to Evolutionary Strategies](https://arxiv.org/abs/1703.03864)
- [However, it is still a GA](https://doi.org/10.1007/BFb0029787)
- [Heuristic or blend crossover](https://doi.org/10.1007/978-3-662-03315-9)
- [SBX crossover](https://content.wolfram.com/sites/13/2018/02/09-2-2.pdf)