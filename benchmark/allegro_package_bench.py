#!/usr/bin/env python
"""Time the real `nequip-allegro` package (allegro 0.8.3) energy and forces across system sizes, for
the head-to-head with Molly's native `AllegroPackageModel` (benchmark/allegro.jl) and allegro-jax
(benchmark/allegro_jax_bench.py) in benchmark/allegro_benchmarks.md.

The model is the SAME architecture Molly loads (l_max=2, 2 layers, 32 scalar / 8 tensor features,
TwoBodyBesselScalarEmbed p=6, 8 bessels, avg_nn=10, r_c=4.0, H/C/N/O), so this is an
implementation-vs-implementation throughput comparison at identical model scale (float64).

Energy and forces are timed separately, as in the ANI head-to-head: energy is the inner
SequentialGraphNetwork (`model.model.func`) under no_grad (no force backward); forces is the full
model (autograd gradient of the energy). Systems are random H/C/N/O at density 0.09 atoms/Å³ (same as
the Molly bench); edges are built with a KD-tree so sizes reach ~16k atoms. Writes
results/allegro_torch_<device>.json keyed by cpu_t<N> / cuda with {energy_ms, forces_ms, edges}.

  ALLEGRO_TORCH_DEVICE=cpu  ALLEGRO_TORCH_THREADS=8  <python-with-nequip-allegro> benchmark/allegro_package_bench.py
  ALLEGRO_TORCH_DEVICE=cuda                          <python-with-nequip-allegro> benchmark/allegro_package_bench.py
"""
import os, json, time
import numpy as np
import torch
from scipy.spatial import cKDTree
from nequip.utils.global_state import set_global_state
from nequip.data import AtomicDataDict
from allegro.model import AllegroModel

RC = 4.0
DENSITY = 0.09
TYPE_NAMES = ["H", "C", "N", "O"]
DEVICE = os.environ.get("ALLEGRO_TORCH_DEVICE", "cpu")
SIZES = [int(x) for x in os.environ.get("ALLEGRO_SIZES", "500,1000,2000,4000,8000,15954").split(",")]
if DEVICE == "cpu":
    torch.set_num_threads(int(os.environ.get("ALLEGRO_TORCH_THREADS", "8")))
KEY = "cuda" if DEVICE == "cuda" else f"cpu_t{torch.get_num_threads()}"

CFG = dict(seed=0, model_dtype="float64", l_max=2, r_max=RC, type_names=TYPE_NAMES,
           radial_chemical_embed={"_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                                  "num_bessels": 8, "bessel_trainable": False, "polynomial_cutoff_p": 6},
           num_layers=2, num_scalar_features=32, num_tensor_features=8, avg_num_neighbors=10.0)


def build_model():
    set_global_state(); torch.manual_seed(0)
    return AllegroModel(**CFG).double().to(DEVICE).eval()


def make_system(n, rng):
    L = (n / DENSITY) ** (1.0 / 3.0)
    pos = rng.uniform(0.0, L, size=(n, 3))
    types = rng.integers(0, 4, n)
    return pos, types, L


def edge_index(pos):
    # KD-tree neighbour pairs within r_c (both directions), so this scales to ~16k atoms.
    tree = cKDTree(pos)
    pairs = tree.query_pairs(RC, output_type="ndarray")      # i<j
    ij = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T    # both directions
    return torch.tensor(ij, dtype=torch.long, device=DEVICE), ij.shape[1]


def make_inputs(pos, types):
    ei, ne = edge_index(pos)
    t = torch.tensor(types, dtype=torch.long, device=DEVICE).reshape(-1, 1)
    def build(grad):
        p = torch.tensor(pos, dtype=torch.float64, device=DEVICE, requires_grad=grad)
        return {AtomicDataDict.POSITIONS_KEY: p,
                AtomicDataDict.ATOM_TYPE_KEY: t,
                AtomicDataDict.EDGE_INDEX_KEY: ei}
    return build, ne


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timeit(f, reps=4, samples=3):
    f(); sync()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(samples):
            f()
        sync()
        best = min(best, (time.perf_counter() - t0) / samples)
    return best * 1e3   # ms


def main():
    model = build_model()
    energy_net = model.model.func    # inner SequentialGraphNetwork: total energy, no force backward
    print(f"nequip-allegro bench | device={DEVICE} key={KEY} | "
          f"params={sum(p.numel() for p in model.parameters())} | sizes={SIZES}")
    rng = np.random.default_rng(1)
    res = {KEY: {}}
    for n in SIZES:
        pos, types, L = make_system(n, rng)
        build, ne = make_inputs(pos, types)
        def energy():
            with torch.no_grad():
                return energy_net(build(False))[AtomicDataDict.TOTAL_ENERGY_KEY].sum()
        def forces():
            return model(build(True))[AtomicDataDict.FORCE_KEY]
        te = timeit(energy)
        tf = timeit(forces)
        res[KEY][str(n)] = {"energy_ms": te, "forces_ms": tf, "edges": int(ne)}
        print(f"  n={n:5d} edges={ne:7d} energy={te:9.3f} ms  forces={tf:9.3f} ms")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"allegro_torch_{DEVICE}.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(res)
    json.dump(prev, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
