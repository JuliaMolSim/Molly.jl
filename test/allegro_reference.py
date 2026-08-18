#!/usr/bin/env python3
"""Offline reference generator for the native Allegro potential in Molly.jl.

This is developer tooling, not run in CI (like test/torchani_reference.py). It exports fixed-input
values from e3nn so the Julia equivariant primitives in src/equivariant/ can be pinned to e3nn's
exact conventions (axis order, "component" normalization, real Wigner-3j signs) — the prerequisite
for loading trained Allegro/NequIP weights. A later revision will also walk a trained Allegro model
and export its weights + per-system reference energies/forces to HDF5/JSON.

Usage:
    pip install "e3nn>=0.5" torch numpy
    python test/allegro_reference.py            # writes data/allegro_reference/e3nn_reference.json

The Julia side (test/equivariant.jl) can then load this file, when present, and assert its real
spherical harmonics and tensor product match e3nn under the recorded convention flags.
"""

import json
import os

import numpy as np
import torch
from e3nn import o3


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "allegro_reference")

# Fixed sample directions (unit vectors), deterministic — no RNG.
SAMPLE_DIRS = [
    [0.3, -0.5, 0.8],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-0.4, 0.7, -0.6],
    [0.577350269, 0.577350269, 0.577350269],
]

LMAX = 2
SH_NORMALIZATION = "component"   # e3nn flag; Σ_m Y_lm^2 = 2l+1
SH_NORMALIZE = True              # normalise the input vector before evaluating


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def export_spherical_harmonics():
    """e3nn real spherical harmonics for l=0..LMAX at each sample direction."""
    irreps = o3.Irreps.spherical_harmonics(LMAX)  # e.g. 1x0e+1x1o+1x2e, e3nn (y,z,x) axis order
    out = []
    for d in SAMPLE_DIRS:
        x = torch.tensor(_unit(d), dtype=torch.float64)
        y = o3.spherical_harmonics(irreps, x, normalize=SH_NORMALIZE, normalization=SH_NORMALIZATION)
        out.append(y.tolist())
    return {"irreps": str(irreps), "dirs": [list(_unit(d)) for d in SAMPLE_DIRS], "values": out}


def export_wigner_3j():
    """Real Wigner-3j (Clebsch-Gordan) tensors for the small couplings used by l<=2 models."""
    couplings = {}
    for l1 in range(LMAX + 1):
        for l2 in range(LMAX + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2, LMAX) + 1):
                w = o3.wigner_3j(l1, l2, l3).to(torch.float64)  # (2l1+1, 2l2+1, 2l3+1)
                couplings[f"{l1},{l2},{l3}"] = w.tolist()
    return couplings


def export_tensor_product():
    """A concrete FullyConnectedTensorProduct output on fixed inputs and weights, for a full
    end-to-end convention check (input irreps, weights and output all recorded)."""
    irreps_in1 = o3.Irreps("2x1o")
    irreps_in2 = o3.Irreps("1x1o")
    irreps_out = o3.Irreps("2x0e+2x1e+2x2e")
    tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out, shared_weights=True)
    torch.manual_seed(0)
    x = torch.arange(1, irreps_in1.dim + 1, dtype=torch.float64) * 0.1
    y = torch.arange(1, irreps_in2.dim + 1, dtype=torch.float64) * 0.1
    with torch.no_grad():
        w = torch.arange(1, tp.weight_numel + 1, dtype=torch.float64) * 0.01
        z = tp(x.unsqueeze(0).double(), y.unsqueeze(0).double(), w.double()).squeeze(0)
    return {
        "irreps_in1": str(irreps_in1), "irreps_in2": str(irreps_in2), "irreps_out": str(irreps_out),
        "x": x.tolist(), "y": y.tolist(), "weights": w.tolist(), "z": z.tolist(),
        "instructions": [list(i) for i in tp.instructions],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = {
        "convention": {
            "lmax": LMAX,
            "sh_normalization": SH_NORMALIZATION,
            "sh_normalize": SH_NORMALIZE,
            "note": "e3nn l=1 axis order is (y, z, x); Molly's internal primitives currently use "
                    "(x, y, z). Pin Molly to this reference (permutation + any sign/scale) before "
                    "loading trained weights.",
        },
        "spherical_harmonics": export_spherical_harmonics(),
        "wigner_3j": export_wigner_3j(),
        "tensor_product": export_tensor_product(),
    }
    path = os.path.join(OUT_DIR, "e3nn_reference.json")
    with open(path, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
