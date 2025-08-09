# Simulated Annealing (SA)

<div align="center">

<figure>
  <img src="../../../examples/gifs/sa_kbf.gif" width="200" alt="SA Example">
  <figcaption><b>Figure:</b> Simulated Annealing with either a Metropolis-Hastings update mechanism or Metropolis-Adjusted Langevin Algorithm (MALA) when gradients are available. </figcaption>
</figure>

</div>

Neighbors are generated with a multi-neighbor generation mechanism.

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
            "reheat_after": 20,
            "x_min": -10.0,
            "x_max": 10.0
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