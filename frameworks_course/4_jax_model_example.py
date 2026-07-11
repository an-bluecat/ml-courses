from pathlib import Path
import pickle

import jax
import jax.numpy as jnp


MODEL_PATH = Path(__file__).with_name("model.pkl")


def predict(params, x):
    """The whole model: y = w*x + b."""
    return params["w"] * x + params["b"]


def loss_fn(params, x, t):
    return jnp.mean((predict(params, x) - t) ** 2)


@jax.jit
def train_step(params, x, t):
    grads = jax.grad(loss_fn)(params, x, t)
    return jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, grads)


def rounded_params(params):
    return jax.tree_util.tree_map(lambda p: round(float(p), 3), params)


def train_and_save():
    xs = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    ts = jnp.array([1.0, 3.0, 5.0, 7.0], dtype=jnp.float32)

    params = {
        "w": jnp.float32(0.0),
        "b": jnp.float32(0.0),
    }

    for step in range(101):
        if step in (0, 1, 5, 20, 100):
            loss = float(loss_fn(params, xs, ts))
            w = float(params["w"])
            b = float(params["b"])
            print(f"step {step:3d}   loss {loss:8.5f}   w {w:6.3f}   b {b:6.3f}")
        params = train_step(params, xs, ts)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(params, f)

    print("saved model.pkl - the ENTIRE model:", rounded_params(params))
    return params


def load_and_serve():
    with MODEL_PATH.open("rb") as f:
        params = pickle.load(f)

    print("loaded:", rounded_params(params))

    for x in [4.0, 10.0, -2.0]:
        prediction = float(predict(params, jnp.float32(x)))
        print(f"request x = {x:5.1f}   ->   prediction {prediction:.3f}")


if __name__ == "__main__":
    train_and_save()
    print()
    load_and_serve()
