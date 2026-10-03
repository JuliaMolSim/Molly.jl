#!/usr/bin/env python
"""6mrr trajectory-throughput head-to-head, reference side: time one Allegro energy+forces evaluation
on the full 6mrr system (15,954 atoms) for nequip-allegro or allegro-jax on a given device, with a
comparable-size model (l_max=2, 2 layers). Pairs with benchmark/allegro_6mrr_throughput.jl (Molly);
together they build results/allegro_traj_throughput.json for the trajectory figure. OOM / impractical
cases are recorded as such — that is the finding at biomolecular scale.

  IMPL=nequip DEVICE=cpu  THREADS=8 python benchmark/allegro_6mrr_throughput.py     # torch env
  IMPL=nequip DEVICE=cuda           python benchmark/allegro_6mrr_throughput.py
  IMPL=jax    DEVICE=cpu  THREADS=8 JAX_PLATFORMS=cpu taskset -c 0-7 python ...      # jax env
  IMPL=jax    DEVICE=cuda           python benchmark/allegro_6mrr_throughput.py
"""
import os, json, time, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDB  = os.environ.get("ALLEGRO_PDB", os.path.join(ROOT, "data", "6mrr_equil.pdb"))
RES  = os.path.join(ROOT, "benchmark", "results")
IMPL = os.environ.get("IMPL", "nequip")
DEV  = os.environ.get("DEVICE", "cpu")
THREADS = int(os.environ.get("THREADS", "1"))
RC = 4.0


def read_6mrr():
    sym2t = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 3}
    pos, types = [], []
    for l in open(PDB):
        if l.startswith(("ATOM", "HETATM")):
            pos.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            types.append(sym2t[l[76:78].strip()])
    return np.array(pos), np.array(types), np.array([1, 6, 7, 8])


def edge_index(pos):
    n = len(pos); i_list, j_list = [], []
    for a in range(0, n, 2000):                          # chunked to avoid an N^2 dense matrix
        b = min(a + 2000, n)
        d = np.linalg.norm(pos[a:b, None, :] - pos[None, :, :], axis=-1)
        ii, jj = np.where((d < RC) & (d > 1e-8))
        i_list.append(ii + a); j_list.append(jj)
    return np.concatenate(i_list), np.concatenate(j_list)


def timeit(f, reps=3):
    f(); best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best * 1e3


def record(key, ms, status):
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, "allegro_traj_throughput.json")
    prev = json.load(open(p)) if os.path.exists(p) else {}
    prev[key] = {"atoms": 15954, "ms_step": (None if ms is None else float(ms)),
                 "ns_day": (None if ms is None else 8.64 / ms), "status": status}
    json.dump(prev, open(p, "w"), indent=1)
    print("wrote", key, status, ms)


def run_nequip():
    import torch
    from nequip.utils.global_state import set_global_state
    from nequip.data import AtomicDataDict
    from allegro.model import AllegroModel
    set_global_state(); torch.manual_seed(0)
    if DEV == "cpu":
        torch.set_num_threads(THREADS)
    key = f"nequip_{'cuda' if DEV == 'cuda' else f'cpu_t{THREADS}'}"
    pos, types, _ = read_6mrr()
    ei_np = np.stack(edge_index(pos)); ne = ei_np.shape[1]
    print(f"nequip {DEV} | 6mrr {len(pos)} atoms, {ne} edges")
    try:
        model = AllegroModel(seed=0, model_dtype="float64", l_max=2, r_max=RC,
            type_names=["H", "C", "N", "O"],
            radial_chemical_embed={"_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                                   "num_bessels": 8, "bessel_trainable": False, "polynomial_cutoff_p": 6},
            num_layers=2, num_scalar_features=16, num_tensor_features=4,
            avg_num_neighbors=10.0).double().to(DEV).eval()
        ei = torch.tensor(ei_np, dtype=torch.long, device=DEV)
        t = torch.tensor(types, dtype=torch.long, device=DEV).reshape(-1, 1)

        def forces():
            p = torch.tensor(pos, dtype=torch.float64, device=DEV, requires_grad=True)
            f = model({AtomicDataDict.POSITIONS_KEY: p, AtomicDataDict.ATOM_TYPE_KEY: t,
                       AtomicDataDict.EDGE_INDEX_KEY: ei})[AtomicDataDict.FORCE_KEY]
            (torch.cuda.synchronize() if DEV == "cuda" else None)
            return f
        record(key, timeit(forces, reps=2 if DEV == "cpu" else 3), "ok")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        record(key, None, "OOM" if "out of memory" in str(e).lower() else f"error:{str(e)[:80]}")


def run_jax():
    import jax, jax.numpy as jnp, e3nn_jax as e3nn, flax.linen as nn
    from allegro_jax import Allegro
    plat = jax.devices()[0].platform
    key = "jax_cuda" if plat == "gpu" else f"jax_cpu_t{THREADS}"
    pos, types, _ = read_6mrr()
    i, j = edge_index(pos); ne = len(i)
    print(f"jax {key} | 6mrr {len(pos)} atoms, {ne} edges")

    class Model(nn.Module):
        @nn.compact
        def __call__(self, p, s, se, re):
            na = jax.nn.one_hot(s, 4)
            v = e3nn.IrrepsArray("1o", p[re] - p[se])
            out = Allegro(avg_num_neighbors=10.0, max_ell=2, irreps=4 * e3nn.Irreps("0e + 1o + 2e"),
                          mlp_n_hidden=16, mlp_n_layers=2, n_radial_basis=8, radial_cutoff=RC,
                          output_irreps=e3nn.Irreps("0e"), num_layers=2)(na, v, se, re)
            return jnp.sum(out.array)
    try:
        model = Model()
        p = jnp.asarray(pos); s = jnp.asarray(types); se = jnp.asarray(i); re = jnp.asarray(j)
        w = model.init(jax.random.PRNGKey(0), p, s, se, re)
        forces = jax.jit(jax.grad(lambda pp: model.apply(w, pp, s, se, re)))
        forces(p).block_until_ready()
        best = np.inf
        for _ in range(3 if plat == "gpu" else 2):
            t0 = time.perf_counter(); forces(p).block_until_ready(); best = min(best, time.perf_counter() - t0)
        record(key, best * 1e3, "ok")
    except Exception as e:
        record(key, None, "OOM" if "memory" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e) else f"error:{str(e)[:80]}")


if __name__ == "__main__":
    (run_jax if IMPL == "jax" else run_nequip)()
