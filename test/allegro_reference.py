#!/usr/bin/env python3
"""Offline reference generator for the native Allegro potential in Molly.jl.

Developer tooling, not run in CI (like test/torchani_reference.py). Requires e3nn + torch. It
writes, into data/allegro_reference/:

  * e3nn_reference.json  — e3nn spherical harmonics / wigner_3j / a tensor product, used to pin
    the Julia equivariant primitives to e3nn's exact conventions.
  * allegro_model.h5     — weights of a small Allegro-style model (random, seeded), in the flat
    layout the Julia loader expects.
  * allegro_model.json   — config, species symbols, and per-system reference energies + forces
    (from finite differences) for a couple of tiny molecules.

The Julia side (test/allegro_potentials.jl) loads these, when present, and checks that the native
implementation reproduces the reference energy (and, once implemented, forces).

Usage:
    pip install "e3nn>=0.5" torch numpy h5py
    python test/allegro_reference.py
"""

import json
import os

import numpy as np
import torch
import h5py
from e3nn import o3

torch.set_default_dtype(torch.float64)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "allegro_reference")
SPECIES = ["H", "C"]  # S = 2 species; indices 0,1

# ----------------------------------------------------------------------------------------------
# e3nn primitive reference (convention pin)
# ----------------------------------------------------------------------------------------------
SAMPLE_DIRS = [[0.3, -0.5, 0.8], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
               [-0.4, 0.7, -0.6], [0.577350269, 0.577350269, 0.577350269]]
LMAX = 2


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def export_primitives():
    irreps = o3.Irreps.spherical_harmonics(LMAX)
    sh = []
    for d in SAMPLE_DIRS:
        x = torch.tensor(_unit(d), dtype=torch.float64)
        sh.append(o3.spherical_harmonics(irreps, x, normalize=True, normalization="component").tolist())
    w3j = {}
    for l1 in range(LMAX + 1):
        for l2 in range(LMAX + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2, LMAX) + 1):
                w3j[f"{l1},{l2},{l3}"] = o3.wigner_3j(l1, l2, l3).tolist()
    return {
        "convention": {"lmax": LMAX, "sh_normalization": "component", "sh_normalize": True,
                       "note": "Molly's src/equivariant primitives are pinned to these values."},
        "spherical_harmonics": {"irreps": str(irreps),
                                "dirs": [list(_unit(d)) for d in SAMPLE_DIRS], "values": sh},
        "wigner_3j": w3j,
    }


# ----------------------------------------------------------------------------------------------
# Small Allegro-style model (numpy + e3nn oracle). See src/equivariant/allegro_model.jl for the
# exactly-matching Julia forward.
# ----------------------------------------------------------------------------------------------
LS = [0, 1, 2]
DIMS = [2 * l + 1 for l in LS]
SHDIM = sum(DIMS)
PAR = {0: 1, 1: -1, 2: 1}


def silu(x):
    return x / (1.0 + np.exp(-x))


def sh(rhat):
    x = torch.tensor(rhat, dtype=torch.float64)
    return o3.spherical_harmonics([0, 1, 2], x, normalize=True, normalization="component").numpy()


def bessel(d, rc, nb):
    n = np.arange(1, nb + 1)
    return np.sqrt(2.0 / rc) * np.sin(n * np.pi * d / rc) / d


def envelope(d, rc, p=6):
    if d >= rc:
        return 0.0
    x = d / rc
    a = (p + 1) * (p + 2) / 2; b = p * (p + 2); c = p * (p + 1) / 2
    return 1 - a * x ** p + b * x ** (p + 1) - c * x ** (p + 2)


def make_paths():
    return [(k1, k2, k3, l1, l2, l3)
            for k1, l1 in enumerate(LS) for k2, l2 in enumerate(LS) for k3, l3 in enumerate(LS)
            if abs(l1 - l2) <= l3 <= l1 + l2 and PAR[l1] * PAR[l2] == PAR[l3]]


def block_slice(k):
    off = sum(DIMS[:k]); return slice(off, off + DIMS[k])


def feat_index(C, k, c, m):
    return sum(DIMS[:k]) * C + c * DIMS[k] + m


def tp_uvu(V, Y, w, C, paths, cgs, woff):
    out = np.zeros(SHDIM * C)
    for pi, (k1, k2, k3, l1, l2, l3) in enumerate(paths):
        cg = cgs[pi]; d1, d2, d3 = cg.shape
        for c in range(C):
            wc = w[woff[pi] + c]
            for i1 in range(d1):
                for i2 in range(d2):
                    for i3 in range(d3):
                        v = cg[i1, i2, i3]
                        if v == 0:
                            continue
                        out[feat_index(C, k3, c, i3)] += wc * v * V[feat_index(C, k1, c, i1)] * Y[block_slice(k2)][i2]
    return out


def build(C=4, H=16, nb=8, rc=4.0, L=2, S=2, seed=0):
    rng = np.random.default_rng(seed)
    paths = make_paths()
    cgs = [o3.wigner_3j(l1, l2, l3).numpy() for (_, _, _, l1, l2, l3) in paths]
    woff = []; off = 0
    for _ in paths:
        woff.append(off); off += C
    n_weights = off
    cfg = dict(C=C, H=H, nb=nb, rc=rc, L=L, S=S, lmax=2, env_p=6, n_weights=n_weights, n_paths=len(paths))
    din = nb + 2 * S
    W = {}
    W['emb_W1'] = rng.standard_normal((H, din)) * 0.3; W['emb_b1'] = np.zeros(H)
    W['emb_W2'] = rng.standard_normal((H, H)) * 0.3; W['emb_b2'] = np.zeros(H)
    W['init_w'] = [rng.standard_normal((C, 1)) * 0.5 for _ in LS]
    W['init_b0'] = rng.standard_normal(C) * 0.1
    W['layers'] = []
    for _ in range(L):
        lw = {}
        lw['tp_W'] = rng.standard_normal((n_weights, H)) * 0.2; lw['tp_b'] = rng.standard_normal(n_weights) * 0.1
        lw['x_W'] = rng.standard_normal((H, H + C)) * 0.2; lw['x_b'] = np.zeros(H)
        lw['lin_w'] = [rng.standard_normal((C, C)) * 0.3 for _ in LS]; lw['lin_b0'] = rng.standard_normal(C) * 0.1
        W['layers'].append(lw)
    W['out_W'] = rng.standard_normal((1, H)) * 0.3; W['out_b'] = rng.standard_normal(1) * 0.1
    return cfg, W, paths, cgs, woff


def edge_energy(cfg, W, paths, cgs, woff, d, rhat, Zi, Zj):
    C, H, nb, rc, S = cfg['C'], cfg['H'], cfg['nb'], cfg['rc'], cfg['S']
    Y = sh(rhat); u = envelope(d, rc); R = bessel(d, rc, nb) * u
    oi = np.zeros(S); oi[Zi] = 1; oj = np.zeros(S); oj[Zj] = 1
    x = silu(W['emb_W1'] @ np.concatenate([R, oi, oj]) + W['emb_b1']); x = W['emb_W2'] @ x + W['emb_b2']
    V = np.zeros(SHDIM * C)
    for k, l in enumerate(LS):
        for c in range(C):
            wl = W['init_w'][k][c, 0]
            for m in range(DIMS[k]):
                V[feat_index(C, k, c, m)] = wl * Y[block_slice(k)][m] * u
        if l == 0:
            for c in range(C):
                V[feat_index(C, 0, c, 0)] += W['init_b0'][c] * u
    for lw in W['layers']:
        w = lw['tp_W'] @ x + lw['tp_b']
        P = tp_uvu(V, Y, w, C, paths, cgs, woff)
        scal = np.array([P[feat_index(C, 0, c, 0)] for c in range(C)])
        x = x + silu(lw['x_W'] @ np.concatenate([x, scal]) + lw['x_b'])
        Vn = np.zeros(SHDIM * C)
        for k, l in enumerate(LS):
            Wl = lw['lin_w'][k]
            for m in range(DIMS[k]):
                for co in range(C):
                    acc = sum(Wl[co, ci] * P[feat_index(C, k, ci, m)] for ci in range(C))
                    if l == 0:
                        acc += lw['lin_b0'][co]
                    Vn[feat_index(C, k, co, m)] = acc
        V = Vn
    return float((W['out_W'] @ x + W['out_b'])[0])


def total_energy(cfg, W, paths, cgs, woff, coords, species, rc):
    n = len(coords); E = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = coords[j] - coords[i]; d = np.linalg.norm(r)
            if d >= rc or d < 1e-8:
                continue
            E += edge_energy(cfg, W, paths, cgs, woff, d, r / d, species[i], species[j])
    return E


def num_forces(cfg, W, paths, cgs, woff, coords, species, rc, h=1e-5):
    n = len(coords); F = np.zeros((n, 3))
    for i in range(n):
        for b in range(3):
            cp = coords.copy(); cp[i, b] += h; cm = coords.copy(); cm[i, b] -= h
            F[i, b] = -(total_energy(cfg, W, paths, cgs, woff, cp, species, rc) -
                        total_energy(cfg, W, paths, cgs, woff, cm, species, rc)) / (2 * h)
    return F


def export_model():
    cfg, W, paths, cgs, woff = build()
    with h5py.File(os.path.join(OUT_DIR, "allegro_model.h5"), "w") as f:
        g = f.create_group("config")
        for k, v in cfg.items():
            g.attrs[k] = v
        f["species"] = [s.encode() for s in SPECIES]
        f["emb_W1"] = W['emb_W1']; f["emb_b1"] = W['emb_b1']
        f["emb_W2"] = W['emb_W2']; f["emb_b2"] = W['emb_b2']
        f["init_w"] = np.stack([W['init_w'][k][:, 0] for k in range(3)], axis=1)
        f["init_b0"] = W['init_b0']; f["out_W"] = W['out_W']; f["out_b"] = W['out_b']
        for li, lw in enumerate(W['layers']):
            lg = f.create_group(f"layer{li}")
            lg["tp_W"] = lw['tp_W']; lg["tp_b"] = lw['tp_b']
            lg["x_W"] = lw['x_W']; lg["x_b"] = lw['x_b']
            lg["lin_w"] = np.stack(lw['lin_w'], axis=2)
            lg["lin_b0"] = lw['lin_b0']

    def mk(coords, species, name):
        coords = np.array(coords, dtype=np.float64)
        E = total_energy(cfg, W, paths, cgs, woff, coords, species, cfg['rc'])
        F = num_forces(cfg, W, paths, cgs, woff, coords, species, cfg['rc'])
        return dict(name=name, coords_A=coords.tolist(), species=list(species),
                    energy=float(E), forces=F.tolist())
    systems = [
        mk([[0.0, 0, 0], [1.1, 0, 0], [0.2, 1.0, 0.3], [-0.5, 0.4, 1.2]], [0, 1, 0, 1], "tetra"),
        mk([[0.0, 0, 0], [1.0, 0.1, -0.2], [-0.3, 0.9, 0.4]], [1, 0, 1], "tri"),
    ]
    with open(os.path.join(OUT_DIR, "allegro_model.json"), "w") as fj:
        json.dump(dict(config=cfg, species=SPECIES, systems=systems), fj, indent=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "e3nn_reference.json"), "w") as f:
        json.dump(export_primitives(), f, indent=2)
    export_model()
    print("wrote e3nn_reference.json, allegro_model.h5, allegro_model.json to", OUT_DIR)


if __name__ == "__main__":
    main()
