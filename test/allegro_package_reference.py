#!/usr/bin/env python
"""External Allegro reference by CALLING the real `nequip-allegro` package (not a re-implementation).

Per PR #283 review (Joe Greener): the reference should come from the existing package, so that a
re-implementation's correctness is checked against the actual code rather than against another
re-implementation. This script:

  1. builds a small real `allegro.model.AllegroModel` (deterministic, seed 0, float64),
  2. runs its forward on a few small H/C/N/O frames and records total/atomic energy + forces,
  3. verifies nequip's ASE calculator (`NequIPCalculator`) reproduces that forward exactly — this is
     the path Molly's `ASECalculator` drives via PythonCall,
  4. exports the reference values + the full model weights (state_dict, with the deterministic
     per-linear `alpha` baked in) so the native Julia port can be validated against the package.

Requires the `nequip-allegro` package (allegro 0.8.3 / nequip 0.19, torch, e3nn, ase). It is an
offline generator like `test/allegro_reference.py`; the emitted files are committed under
`data/allegro_reference/` and consumed by the native-model bit-match (follow-up).

Usage:  <python-with-nequip-allegro> test/allegro_package_reference.py [outdir]
"""
import sys, os, json, itertools
import numpy as np
import torch
from nequip.utils.global_state import set_global_state
from nequip.data import AtomicDataDict
from nequip.data.transforms import NeighborListTransform, ChemicalSpeciesToAtomTypeMapper
from nequip.integrations.ase import NequIPCalculator
from allegro.model import AllegroModel
from ase import Atoms

TYPE_NAMES = ["H", "C", "N", "O"]
RC = 4.0
CFG = dict(seed=0, model_dtype="float64", l_max=2, r_max=RC, type_names=TYPE_NAMES,
           radial_chemical_embed={"_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                                  "num_bessels": 8, "bessel_trainable": False,
                                  "polynomial_cutoff_p": 6},
           num_layers=2, num_scalar_features=32, num_tensor_features=8, avg_num_neighbors=10.0)

# small frames: (name, positions Å, atom-type indices into TYPE_NAMES)
FRAMES = [
    ("ch4",  np.array([[0.,0.,0.],[.63,.63,.63],[-.63,-.63,.63],[-.63,.63,-.63],[.63,-.63,-.63]]), [1,0,0,0,0]),
    ("h2o",  np.array([[0.,0.,0.],[.96,0.,0.],[-.24,.93,0.]]), [3,0,0]),
    ("hcn",  np.array([[0.,0.,0.],[1.07,0.,0.],[2.22,0.,0.]]), [0,1,2]),
    ("nh3o", np.array([[0.,0.,0.],[.94,.3,0.],[-.3,.94,.1],[.2,-.2,.95],[1.8,1.2,.4]]), [2,0,0,0,3]),
]

def build():
    return AllegroModel(**CFG).double().eval()

def edge_index(pos, rc):
    ei = [[i, j] for i, j in itertools.permutations(range(len(pos)), 2)
          if np.linalg.norm(pos[j] - pos[i]) < rc]
    return torch.tensor(ei, dtype=torch.long).t().contiguous()

def direct_forward(model, pos, types):
    p = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    data = {AtomicDataDict.POSITIONS_KEY: p,
            AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(types, dtype=torch.long).reshape(-1, 1),
            AtomicDataDict.EDGE_INDEX_KEY: edge_index(pos, RC)}
    out = model(data)
    E = float(out[AtomicDataDict.TOTAL_ENERGY_KEY].detach().sum())
    Ei = out[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().cpu().numpy().reshape(-1).tolist()
    F = out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy().tolist()
    return E, Ei, F

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "..", "data", "allegro_reference")
    os.makedirs(outdir, exist_ok=True)
    set_global_state()
    torch.manual_seed(0)
    model = build()
    print(f"model: {sum(p.numel() for p in model.parameters())} params, types {TYPE_NAMES}")

    # ASE calculator over the SAME in-memory model (species mapper + neighbour list) — the path
    # Molly's ASECalculator uses.
    mapper = ChemicalSpeciesToAtomTypeMapper(
        model_type_names=TYPE_NAMES, chemical_species_to_atom_type_map={t: t for t in TYPE_NAMES})
    calc = NequIPCalculator(model, device="cpu", transforms=[mapper, NeighborListTransform(r_max=RC)])

    systems = []
    ase_ok = True
    for name, pos, types in FRAMES:
        E, Ei, F = direct_forward(model, pos, types)
        atoms = Atoms(symbols=[TYPE_NAMES[t] for t in types], positions=pos, cell=[50]*3, pbc=False)
        atoms.calc = calc
        dE = abs(atoms.get_potential_energy() - E)
        dF = float(np.max(np.abs(atoms.get_forces() - np.array(F))))
        ase_ok &= (dE < 1e-8 and dF < 1e-8)
        systems.append(dict(name=name, coords_A=pos.tolist(), types=types,
                            energy=E, atomic_energy=Ei, forces=F))
        print(f"  {name:5s} E={E:.8f}  ASE |dE|={dE:.1e} max|dF|={dF:.1e}")
    print("ASE calculator == direct package forward:", "PASS" if ase_ok else "FAIL")
    assert ase_ok, "ASE calculator disagrees with the direct forward"

    with open(os.path.join(outdir, "allegro_package_ref.json"), "w") as f:
        json.dump(dict(config={k: v for k, v in CFG.items() if k != "radial_chemical_embed"},
                       type_names=TYPE_NAMES, systems=systems), f, indent=1)
    # weights: state_dict with the non-persistent per-linear alpha baked into each weight.
    buffers = dict(model.named_buffers())
    export = {}
    for k, v in model.state_dict().items():
        arr = v.detach().cpu().numpy()
        if k.endswith(".weight"):
            akey = k[:-len(".weight")] + ".alpha"
            if akey in buffers:
                arr = arr * float(buffers[akey].detach().cpu().numpy())
        export[k.replace(".", "__")] = arr
    np.savez(os.path.join(outdir, "allegro_package_weights.npz"), **export)
    print("wrote", os.path.join(outdir, "allegro_package_ref.json"), "and _weights.npz")

if __name__ == "__main__":
    main()
