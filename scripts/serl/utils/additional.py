import jax.numpy as jnp
import numpy as np


def to_python_type(x):
    if isinstance(x, (jnp.ndarray, )):
        return float(x)
    elif hasattr(x, "item"):  # JAX scalar
        return x.item()
    elif isinstance(x, dict):
        return {k: to_python_type(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_python_type(v) for v in x]
    return x
