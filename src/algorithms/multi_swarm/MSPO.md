# MSPO

<div align="center">

<figure>
  <img src="../../../examples/gifs/mspo_kbf.gif" width="200" alt="MSPO Example">
  <figcaption><b>Figure:</b> Multi-Swarm Particle Optimization (MSPO) is a population-based optimization algorithm that uses multiple swarms to explore the search space. </figcaption>
</figure>

</div>

Each swarm maintains its own population of particles that move through the search space according to:

- Inertia (w) - Tendency to continue current trajectory
- Cognitive component (c1) - Attraction to particle's personal best position
- Social component (c2) - Attraction to swarm's global best position

## Config example

Fully-defined:

```json
{   
    "alg_conf": {
        "MSPO": {
            "num_swarms": 10,
            "swarm_size": 10,
            "w": 0.729,
            "c1": 1.5,
            "c2": 1.5,
            "x_min": 0.0,
            "x_max": 10.0,
            "exchange_interval": 20,
            "exchange_ratio": 0.05
        }
    }
}
```

Default values:

```json
{
    "alg_conf": {
        "MSPO": {}
    }
}
```

## Sources and more information

- [Multi-Swarm Particle Optimization](https://doi.org/10.1109/ACCESS.2022.3220239)
- [Dynamic and adaptive tactics to improve convergence](https://doi.org/10.1016/j.engappai.2023.106215)