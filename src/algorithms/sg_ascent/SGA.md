# Stochastic Gradient Ascent

<div align="center">

<figure>
  <img src="../../../examples/gifs/sga_kbf.gif" width="200" alt="SGA Example">
  <figcaption><b>Figure:</b> Stochastic Gradient Ascent (SGA) is a gradient-based optimization algorithm. It is a simple and efficient optimization algorithm that is often used in machine learning and deep learning applications. </figcaption>
</figure>

</div>

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "SGA": {
            "learning_rate": 0.1,
            "momentum": 0.9
        }
    }
}

Default values, (nothing needs to be specified):

```json
{
    "alg_conf": {
        "SGA": {}
    }
}
```

## Sources and more information

- [Stochastic Gradient Descent](https://doi.org/10.1037%2Fh0042519)