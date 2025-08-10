# Tabu Search

<div align="center">

<img src="../../../examples/gifs/tabu_kbf.gif" width="200" alt="Tabu Example">

<p><b>Figure:</b> The Tabu Search algorithm is a local search method that maintains a tabu list to avoid revisiting recently explored solutions, helping the search escape local optima. The choice and management of the tabu list are central to the method's effectiveness.</p>

</div>

Two variants of tabu search are implemented:

- Standard tabu search
    - Uses a fixed size tabu list.
- Reactive tabu search
    - Uses a dynamic tabu list size.

## Guidance

|  | Standard Tabu Search | Reactive Tabu Search |
|---------|---------------------|---------------------|
| Tabu List Size | Fixed size specified by parameter | Dynamically adjusts between min and max size |
| Memory Usage | Constant memory usage | Variable memory usage based on search progress |
| Parameter Tuning | Requires careful tuning of tabu list size | More robust to initial parameter settings |
| Adaptation | No adaptation during search | Adapts tabu list size based on search effectiveness |
| Escape Mechanism | Basic tabu restrictions | Enhanced escape from local optima through size adjustments |
| When to Use | Well-understood problem spaces | Problems with varying landscape complexity |

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "TS": {
            "common": {
                "num_neighbors": 10,
                "step_size": 0.1,
                "perturbation_prob": 0.1
            },
            "tabu_list": {
                "Standard": {
                    "size": 10
                }
                // or
                "Reactive": {
                    "min_size": 10,
                    "max_size": 100
                }
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
            "tabu_list": {
                "Standard": {}
            }
        }
    }
}
```
## Sources and more information

- [Tabu Search](https://ieeexplore.ieee.org/document/9091743)
- [Reactive Tabu Search](https://doi.org/10.1287/ijoc.6.2.126)