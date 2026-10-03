#!/usr/bin/env python
"""Time the REAL nequip-allegro (PyTorch) energy + forces across system sizes, for a head-to-head
with Molly's native Allegro (cf. the ANI-vs-TorchANI comparison in JuliaMolSim/Molly.jl#260).

Model is configured to be COMPARABLE in size to Molly's native reference (l_max=2, 2 layers,
num_tensor_features=4, num_scalar_features=16, num_bessels=8) — not identical ops (the native
bit-match is a follow-up), so this measures implementation throughput at a similar model scale.

Systems: jittered cubic lattice (2.5 A spacing), H/C, all-pairs edges within r_max — the same shape
as benchmark/allegro.jl. Writes results/allegro_torch_<device>.json keyed by cpu_t<N> / cuda with
{energy_ms, forces_ms, edges} per atom count.

  ALLEGRO_TORCH_DEVICE=cpu  ALLEGRO_TORCH_THREADS=8  python allegro_torch_bench.py
  ALLEGRO_TORCH_DEVICE=cuda                          python allegro_torch_bench.py
"""
import os, json, time, itertools
import numpy as np
import torch
from nequip.utils.global_state import set_global_state
from nequip.data import AtomicDataDict
from allegro.model import AllegroModel

RC = 4.0
DEVICE = os.environ.get("ALLEGRO_TORCH_DEVICE", "cpu")
SIZES = [int(x) for x in os.environ.get("ALLEGRO_SIZES", "64,128,256,512,1024,2048,4096").split(",")]
if DEVICE == "cpu":
    torch.set_num_threads(int(os.environ.get("ALLEGRO_TORCH_THREADS", "1")))
COMPILE = os.environ.get("ALLEGRO_TORCH_COMPILE", "0") == "1"   # torch.compile (inductor) — the fast path
KEY = ("cuda" if DEVICE == "cuda" else f"cpu_t{torch.get_num_threads()}") + ("_c" if COMPILE else "")

def build_model():
    set_global_state(); torch.manual_seed(0)
    return AllegroModel(seed=0, model_dtype="float64", l_max=2, r_max=RC, type_names=["H", "C"],
        radial_chemical_embed={"_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                               "num_bessels": 8, "bessel_trainable": False, "polynomial_cutoff_p": 6},
        num_layers=2, num_scalar_features=16, num_tensor_features=4,
        avg_num_neighbors=10.0).double().to(DEVICE).eval()

def make_system(n, a=2.5, jitter=0.2, seed=1):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.cbrt(n)))
    pts = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if len(pts) == n: break
                pts.append([a*x, a*y, a*z])
    pts = np.array(pts[:n]) + jitter * (2*rng.random((n, 3)) - 1)
    types = rng.integers(0, 2, n)
    return pts, types

def edge_index(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    i, j = np.where((d < RC) & (d > 1e-8))
    return torch.tensor(np.stack([i, j]), dtype=torch.long, device=DEVICE), len(i)

def make_inputs(pos, types):
    """Cache the edge index + type tensor; positions are built fresh each call (fresh autograd leaf)."""
    ei, ne = edge_index(pos)
    t = torch.tensor(types, dtype=torch.long, device=DEVICE).reshape(-1, 1)
    def build(grad):
        p = torch.tensor(pos, dtype=torch.float64, device=DEVICE, requires_grad=grad)
        return {AtomicDataDict.POSITIONS_KEY: p,
                AtomicDataDict.ATOM_TYPE_KEY: t,
                AtomicDataDict.EDGE_INDEX_KEY: ei}
    return build, ne

def sync():
    if DEVICE == "cuda": torch.cuda.synchronize()

def timeit(f, reps=5, samples=8):
    f(); sync()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(samples): f()
        sync()
        best = min(best, (time.perf_counter() - t0) / samples)
    return best * 1e3   # ms

def main():
    model = build_model()
    energy_net = model.model.func   # inner SequentialGraphNetwork: total_energy without the force backward
    if COMPILE:
        energy_net = torch.compile(energy_net, dynamic=True)
        model = torch.compile(model, dynamic=True)
    print(f"nequip-allegro bench | device={DEVICE} key={KEY} | compile={COMPILE} | params={sum(p.numel() for p in build_model().parameters())}")
    res = {KEY: {}}
    for n in SIZES:
        pos, types = make_system(n)
        build, ne = make_inputs(pos, types)
        def energy():
            with torch.no_grad():
                return energy_net(build(False))[AtomicDataDict.TOTAL_ENERGY_KEY].sum()
        def forces():
            return model(build(True))[AtomicDataDict.FORCE_KEY]
        te = timeit(energy)
        tf = timeit(forces)
        res[KEY][str(n)] = {"energy_ms": te, "forces_ms": tf, "edges": ne}
        print(f"  n={n:5d} edges={ne:6d} energy={te:8.3f} ms  forces={tf:8.3f} ms")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"allegro_torch_{DEVICE}.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(res)
    json.dump(prev, open(path, "w"), indent=1)
    print("wrote", path)

if __name__ == "__main__":
    main()
