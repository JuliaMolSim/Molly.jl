#!/usr/bin/env python
"""Time allegro-jax (https://github.com/mariogeiger/allegro-jax) energy + forces across system
sizes, as a second independent Allegro implementation for the head-to-head in
benchmark/allegro_benchmarks.md.

The model is configured to be COMPARABLE in size to Molly's native reference and the
nequip-allegro run in allegro_torch_bench.py (l_max=2, 2 layers, 4 tensor channels,
16 hidden scalars, 8 bessels, r_c=4.0) — not identical ops, so read it as implementation
throughput at a similar model scale.

Systems: the same jittered cubic lattice (2.5 A spacing, seed 1), H/C species, all-pairs
edges within r_max. Energies/forces in float64 to match the other columns. Writes
results/allegro_jax_<device>.json keyed like the torch bench.

  ~/allegrojax/bin/python allegro_jax_bench.py           # CUDA if visible, else CPU
  JAX_PLATFORMS=cpu ~/allegrojax/bin/python allegro_jax_bench.py
"""
import json
import os
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import e3nn_jax as e3nn  # noqa: E402
import flax.linen as nn  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from allegro_jax import Allegro  # noqa: E402

RC = 4.0
SIZES = [int(x) for x in
         os.environ.get("ALLEGRO_SIZES", "64,128,256,512,1024,2048,4096").split(",")]
DEVICE = jax.devices()[0].platform  # "gpu" or "cpu"
KEY = "cuda" if DEVICE == "gpu" else "cpu"


class Model(nn.Module):
    @nn.compact
    def __call__(self, positions, species, senders, receivers):
        node_attrs = jax.nn.one_hot(species, 2)
        vectors = e3nn.IrrepsArray("1o", positions[receivers] - positions[senders])
        out = Allegro(
            avg_num_neighbors=10.0,
            max_ell=2,
            irreps=4 * e3nn.Irreps("0e + 1o + 2e"),
            mlp_n_hidden=16,
            mlp_n_layers=2,
            n_radial_basis=8,
            radial_cutoff=RC,
            output_irreps=e3nn.Irreps("0e"),
            num_layers=2,
        )(node_attrs, vectors, senders, receivers)
        return jnp.sum(out.array)


def make_system(n, a=2.5, jitter=0.2, seed=1):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.cbrt(n)))
    pts = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if len(pts) == n:
                    break
                pts.append([a * x, a * y, a * z])
    pts = np.array(pts[:n]) + jitter * (2 * rng.random((n, 3)) - 1)
    types = rng.integers(0, 2, n)
    return pts, types


def edge_index(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    i, j = np.where((d < RC) & (d > 1e-8))
    return i, j


def timeit(f, reps=5, samples=8):
    f().block_until_ready()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(samples):
            out = f()
        out.block_until_ready()
        best = min(best, (time.perf_counter() - t0) / samples)
    return best * 1e3  # ms


def main():
    model = Model()
    res = {KEY: {}}
    printed_params = False
    for n in SIZES:
        pos_np, types_np = make_system(n)
        i, j = edge_index(pos_np)
        pos = jnp.asarray(pos_np)
        species = jnp.asarray(types_np)
        senders = jnp.asarray(i)
        receivers = jnp.asarray(j)

        w = model.init(jax.random.PRNGKey(0), pos, species, senders, receivers)
        if not printed_params:
            nparams = sum(x.size for x in jax.tree_util.tree_leaves(w))
            print(f"allegro-jax bench | device={KEY} | params={nparams}")
            printed_params = True

        energy = jax.jit(lambda p: model.apply(w, p, species, senders, receivers))
        forces = jax.jit(jax.grad(lambda p: model.apply(w, p, species, senders, receivers)))

        te = timeit(lambda: energy(pos))
        tf = timeit(lambda: forces(pos))
        res[KEY][str(n)] = {"energy_ms": te, "forces_ms": tf, "edges": int(len(i))}
        print(f"  n={n:5d} edges={len(i):6d} energy={te:8.3f} ms  forces={tf:8.3f} ms")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"allegro_jax_{KEY}.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(res)
    json.dump(prev, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
