#!/usr/bin/env python
"""Time the real `nequip-allegro` package (allegro 0.8.3) on one energy + forces evaluation vs
system size, to put the native Molly `AllegroPackageModel` timings (benchmark/allegro.jl) in context
on the SAME hardware. Same architecture as the reference generator (test/allegro_package_reference.py):
l_max=2, 2 layers, 32 scalar / 8 tensor features, TwoBodyBesselScalarEmbed p=6, avg_nn=10, float64.

One `model(data)` call returns total energy AND forces (the model's autograd GradientOutput), matching
the native `allegro_package_energy_and_forces`. The edge list is rebuilt inside the timed region (torch
cdist, cutoff r_max) so both sides include neighbour construction. Random H/C/N/O systems at the same
density (0.09 atoms/Å³) and sizes as benchmark/allegro.jl.

  PKG_DEV=cpu  ALLEGRO_SIZES=100,250,500,1000  <python-with-nequip-allegro> benchmark/allegro_package_bench.py
  PKG_DEV=cuda ALLEGRO_SIZES=100,250,500,1000,2000 <python-with-nequip-allegro> benchmark/allegro_package_bench.py

Writes benchmark/results/allegro_bench.json under key  package_<dev>  (merged with the native keys).
"""
import os, sys, json, time
import numpy as np
import torch
from nequip.utils.global_state import set_global_state
from nequip.data import AtomicDataDict
from allegro.model import AllegroModel

TYPE_NAMES = ["H", "C", "N", "O"]
RC = 4.0
DENSITY = 0.09
CFG = dict(seed=0, model_dtype="float64", l_max=2, r_max=RC, type_names=TYPE_NAMES,
           radial_chemical_embed={"_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                                  "num_bessels": 8, "bessel_trainable": False,
                                  "polynomial_cutoff_p": 6},
           num_layers=2, num_scalar_features=32, num_tensor_features=8, avg_num_neighbors=10.0)

DEV = os.environ.get("PKG_DEV", "cpu")
SIZES = [int(s) for s in os.environ.get("ALLEGRO_SIZES", "100,250,500,1000,2000").split(",")]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "benchmark", "results")


def random_system(n, rng):
    L = (n / DENSITY) ** (1.0 / 3.0)
    pos = rng.uniform(0.0, L, size=(n, 3))
    types = rng.integers(0, 4, size=n)
    return pos, types, L


def eval_once(model, pos, types, device):
    # Rebuild the edge list (cutoff r_max) inside the timed region, then energy + forces in one call.
    p = torch.tensor(pos, dtype=torch.float64, device=device, requires_grad=True)
    with torch.no_grad():
        d = torch.cdist(p, p)
        mask = (d < RC) & (d > 0)
        ei = mask.nonzero(as_tuple=False).t().contiguous()
    data = {AtomicDataDict.POSITIONS_KEY: p,
            AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(types, dtype=torch.long,
                                                       device=device).reshape(-1, 1),
            AtomicDataDict.EDGE_INDEX_KEY: ei}
    out = model(data)
    E = out[AtomicDataDict.TOTAL_ENERGY_KEY].sum()
    F = out[AtomicDataDict.FORCE_KEY]
    # touch the results so lazy work can't be skipped
    return float(E.detach().cpu()), float(F.detach().abs().sum().cpu())


def timeit(fn, reps=4):
    fn()
    best = float("inf")
    for _ in range(reps):
        if DEV == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        best = min(best, (time.time() - t) * 1e3)
    return best


def main():
    set_global_state()
    torch.manual_seed(0)
    device = torch.device(DEV)
    model = AllegroModel(**CFG).double().eval().to(device)
    print(f"package Allegro benchmark | dev={DEV} | sizes={SIZES} | "
          f"{sum(p.numel() for p in model.parameters())} params")
    rng = np.random.default_rng(1)
    rows = {}
    for n in SIZES:
        pos, types, L = random_system(n, rng)
        ms = timeit(lambda: eval_once(model, pos, types, device))
        rows[f"n{n}"] = dict(atoms=n, box_A=L, ms_energy_forces=ms)
        print(f"  N={n:5d}  box={L:.1f} A   {ms:8.2f} ms")

    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, "allegro_bench.json")
    prev = {}
    if os.path.isfile(path):
        with open(path) as f:
            prev = json.load(f)
    key = f"package_{DEV}" + (f"_t{torch.get_num_threads()}" if DEV == "cpu" else "")
    prev[key] = rows
    with open(path, "w") as f:
        json.dump(prev, f, indent=4)
    print("wrote", path, "key", key)


if __name__ == "__main__":
    main()
