#!/usr/bin/env python
"""Time allegro-jax (https://github.com/mariogeiger/allegro-jax) energy + forces across system sizes,
as a second independent Allegro implementation for the head-to-head in benchmark/allegro_benchmarks.md
(Molly native vs nequip-allegro vs allegro-jax).

The model is configured to the SAME scale as Molly's native model and the nequip-allegro run
(l_max=2, 2 layers, 8 tensor channels, 32 hidden scalars, 8 bessels, r_c=4.0) — not identical ops, so
read it as implementation throughput at a matching model scale. Systems are random H/C/N/O at density
0.09 atoms/Å³ (same as the other benches), float64; edges via a KD-tree so sizes reach ~16k atoms.
Writes results/allegro_jax_<device>.json keyed cpu_t<N> / cuda with {energy_ms, forces_ms, edges}.

  ~/allegrojax/bin/python allegro_jax_bench.py                                              # CUDA if visible
  JAX_PLATFORMS=cpu ALLEGRO_JAX_THREADS=8 taskset -c 0-7 ~/allegrojax/bin/python allegro_jax_bench.py   # CPU t8
"""
import json, os, time
import numpy as np
from scipy.spatial import cKDTree

_THREADS = int(os.environ.get("ALLEGRO_JAX_THREADS", "0"))   # 0 = default (all cores); CPU-only knob
if _THREADS == 1:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()

import jax  # noqa: E402

_PLAT = jax.devices()[0].platform.lower()   # "gpu" (cuda), "cpu", or "metal"
_X64 = os.environ.get("ALLEGRO_JAX_X64", "0" if "metal" in _PLAT else "1") == "1"
jax.config.update("jax_enable_x64", _X64)

import e3nn_jax as e3nn  # noqa: E402
import flax.linen as nn  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from allegro_jax import Allegro  # noqa: E402

RC = 4.0
DENSITY = 0.09
NTYPES = 4
SIZES = [int(x) for x in os.environ.get("ALLEGRO_SIZES", "500,1000,2000,4000,8000,15954").split(",")]
if "metal" in _PLAT:
    KEY = "metal"
elif _PLAT == "gpu":
    KEY = "cuda"
else:
    KEY = f"cpu_t{_THREADS}" if _THREADS > 0 else "cpu"


class Model(nn.Module):
    @nn.compact
    def __call__(self, positions, species, senders, receivers):
        node_attrs = jax.nn.one_hot(species, NTYPES)
        vectors = e3nn.IrrepsArray("1o", positions[receivers] - positions[senders])
        out = Allegro(
            avg_num_neighbors=10.0,
            max_ell=2,
            irreps=8 * e3nn.Irreps("0e + 1o + 2e"),   # 8 tensor channels (num_tensor_features)
            mlp_n_hidden=32,                           # num_scalar_features
            mlp_n_layers=2,
            n_radial_basis=8,
            radial_cutoff=RC,
            output_irreps=e3nn.Irreps("0e"),
            num_layers=2,
        )(node_attrs, vectors, senders, receivers)
        return jnp.sum(out.array)


def make_system(n, rng):
    L = (n / DENSITY) ** (1.0 / 3.0)
    pos = rng.uniform(0.0, L, size=(n, 3))
    types = rng.integers(0, NTYPES, n)
    return pos, types, L


def edge_index(pos):
    tree = cKDTree(pos)
    pairs = tree.query_pairs(RC, output_type="ndarray")      # i<j
    ij = np.concatenate([pairs, pairs[:, ::-1]], axis=0)      # both directions
    return ij[:, 0], ij[:, 1]


def timeit(f, max_reps=40, min_reps=3, budget=4.0):
    """Best single-call time in ms. Adaptive: cheap sizes get many samples; the dense large-N cases,
    where one call dominates, stop after min_reps once the budget is exceeded. Always the minimum."""
    f().block_until_ready()
    best = np.inf
    t_start = time.perf_counter()
    for r in range(max_reps):
        t0 = time.perf_counter()
        out = f()
        out.block_until_ready()
        best = min(best, time.perf_counter() - t0)
        if r + 1 >= min_reps and time.perf_counter() - t_start > budget:
            break
    return best * 1e3  # ms


def main():
    model = Model()
    rng = np.random.default_rng(1)
    res = {KEY: {}}
    printed = False
    for n in SIZES:
        pos_np, types_np, L = make_system(n, rng)
        i, j = edge_index(pos_np)
        pos = jnp.asarray(pos_np); species = jnp.asarray(types_np)
        senders = jnp.asarray(i); receivers = jnp.asarray(j)
        w = model.init(jax.random.PRNGKey(0), pos, species, senders, receivers)
        if not printed:
            nparams = sum(x.size for x in jax.tree_util.tree_leaves(w))
            print(f"allegro-jax bench | device={KEY} | params={nparams} | sizes={SIZES}")
            printed = True
        energy = jax.jit(lambda p: model.apply(w, p, species, senders, receivers))
        forces = jax.jit(jax.grad(lambda p: model.apply(w, p, species, senders, receivers)))
        te = timeit(lambda: energy(pos))
        tf = timeit(lambda: forces(pos))
        res[KEY][str(n)] = {"energy_ms": te, "forces_ms": tf, "edges": int(len(i))}
        print(f"  n={n:5d} edges={len(i):7d} energy={te:9.3f} ms  forces={tf:9.3f} ms")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    fkey = "cpu" if KEY.startswith("cpu") else KEY
    path = os.path.join(outdir, f"allegro_jax_{fkey}.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(res)
    json.dump(prev, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
