# Stochastic Gradient Ascent

<div align="center">

<img src="../../../examples/gifs/sga_kbf.gif" width="200" alt="SGA Example">

<p><b>Figure:</b> The Stochastic Gradient Ascent (SGA) algorithm is a simple and efficient gradient-based optimization method. It is widely used in machine learning and deep learning for maximizing objective functions, updating parameters incrementally using noisy gradient estimates.</p>

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