# Adaptive Moment Estimation

<div align="center">

<img src="../../../examples/gifs/adam_kbf.gif" width="200" alt="Adam Example">

<p><b>Figure:</b> The Adam (Adaptive Moment Estimation) algorithm computes adaptive learning rates for each parameter, combining the advantages of <b>AdaGrad</b> and <b>RMSProp</b>.</p>

</div>

At each time step $ t $, the following updates are performed:

1. Gradient of fitness function
```math
g_t = \nabla_{x} f(x_t)
```

2. Update biased first moment estimate (mean)
```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

3. Update biased second raw moment estimate (uncentered variance) 
```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

4. Compute bias-corrected first and second moments
```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

5. Update parameters
```math
x_{t+1} = x_t + \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

Parameters
- $ \alpha $: Learning rate
- $ (\beta_1, \beta_2) \in [0,1]^2 $: Decay rates (typically $0.9$ and $0.999$ respectively)
- $ \epsilon $: Small constant to prevent division by zero (typically $10^{-8}$)

## Config example

Fully-defined:

```json
{
    "alg_conf": {
        "Adam": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        }
    }
}
```

Default values, (nothing needs to be specified):

```json
{
    "alg_conf": {
        "Adam": {}
    }
}
```

## Sources and further information

- [Adam](https://arxiv.org/abs/1412.6980)