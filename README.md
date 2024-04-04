# PINN Blogpost

In this blog post we attempt to reproduce the paper *Solving real-world optimization tasks using physics-informed neural computing*[^X]. ]
The original paper's code is [available on github](https://github.com/jaem-seo/pinn-optimization/tree/b65a4982283d46be4c817d8e3157ca68c39ed88c) and uses the [DeepXDE](https://github.com/lululxvi/deepxde) library to implement the PINNs.
We reproduce three of the examples produced in the paper by writing the network in pure PyTorch, and include a [new example](#spaceship-landing) in order to further test the capabilities if the proposed architecture.


## Pendulum

### Existing code

We found that the code used by the developer is flakier than the paper might suggest. In `data/pendulum` the code sets a random seed as such:

```python
# Set random seed
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
dde.backend.tf.random.set_random_seed(seed)
```

and indeed the code runs well when run like this. However, removing the set seed does not always give good results. Here we present a random sampling of X runs with the above lines ommitted:



### Own implementation

- Point resampling is important
- learning rates and coefficients are important, and depend on the implementation. For example, learning rates had to be lowered when point resampling wasn't implemented 
- about 1 in X runs actually converges to something nice
- average time for a run
  - (did not have time to optimize this)
- 
![run1](pendulum/runs/run_20240404114459.png)

## Spacecraft Swingby

## Shortest Path

## Spaceship Landing


[^X]: Seo, J. Solving real-world optimization tasks using physics-informed neural computing. Sci Rep 14, 202 (2024). https://doi.org/10.1038/s41598-023-49977-3