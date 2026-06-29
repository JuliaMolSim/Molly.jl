#!/usr/bin/env python3
"""
Generate TorchANI ANI-2x reference data for Molly.jl ML potential validation.
Tested with TorchANI 2.2.4 + PyTorch 2.8.0 + Python 3.9.

Outputs written to data/ani_reference/:
  n2_dimer_ani2x.json   energy, forces, coordinates for N₂ dimer at 1.1 Å
  n2_aevs.json          raw AEV vectors (2 × 1008) for both N atoms
  water_ani2x.json      energy + forces for H₂O at TIP3P geometry (multi-element)
  water_aevs.json       raw AEV vectors (3 × 1008) for O, H, H
  6mrr_ani2x.json       energy + per-atom forces for 6mrr_equil.pdb (slow)
  ani2x.h5              ANI-2x weights + AEV params in HDF5 for the Julia loader

Usage:
  pip install torchani==2.2.4 ase h5py
  python scripts/torchani_reference.py
  python scripts/torchani_reference.py --skip-protein    # skip slow protein run
  python scripts/torchani_reference.py --skip-weights    # skip HDF5 export
"""

import os, json, argparse, warnings
import numpy as np
import torch
import torchani

warnings.filterwarnings("ignore")   # suppress cuaev / urllib warnings
os.makedirs("data/ani_reference", exist_ok=True)

HARTREE_TO_EV = 27.211396132  # CODATA 2018

# ANI-2x element order (must match species_converter output)
ELEMENTS = ["H", "C", "N", "O", "S", "F", "Cl"]
ATOMIC_NUMS = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17}


def get_model():
    return torchani.models.ANI2x(periodic_table_index=True)


def compute_aev_vectors(model, atomic_nums_list, positions_list):
    """
    Compute ANI-2x AEV vectors for a list of atoms.
    Returns numpy array of shape (n_atoms, 1008).

    TorchANI 2.2.4 requires calling species_converter BEFORE aev_computer.
    """
    species_raw = torch.tensor([atomic_nums_list], dtype=torch.long)
    coords      = torch.tensor([positions_list],   dtype=torch.float32)

    # Step 1: convert atomic numbers → model indices
    converted = model.species_converter((species_raw, coords))

    # Step 2: compute AEVs with model-indexed species
    _, aevs = model.aev_computer(converted)   # shape (1, n_atoms, 1008)
    return aevs[0].detach().numpy()           # (n_atoms, 1008)


# -------------------------------------------------------------------------
# N₂ dimer reference
# -------------------------------------------------------------------------

def n2_dimer_reference(model):
    print("=== N₂ dimer (1.1 Å bond, no PBC) ===")

    atomic_nums = [7, 7]
    positions   = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]

    species = torch.tensor([atomic_nums], dtype=torch.long)
    coords  = torch.tensor([positions],   dtype=torch.float32, requires_grad=True)

    result = model((species, coords))
    energy_ha = result.energies.item()
    energy_ev = energy_ha * HARTREE_TO_EV

    grads = torch.autograd.grad(result.energies.sum(), coords)[0]
    forces_ha = (-grads[0]).detach().numpy()
    forces_ev = forces_ha * HARTREE_TO_EV

    # AEVs
    aev_np = compute_aev_vectors(model, atomic_nums, positions)

    data = {
        "description":     "ANI-2x reference: N₂ dimer at 1.1 Å, no PBC",
        "species":         ["N", "N"],
        "atomic_numbers":  atomic_nums,
        "coordinates_A":   positions,
        "energy_hartree":  float(energy_ha),
        "energy_eV":       float(energy_ev),
        "forces_hartree_per_A": forces_ha.tolist(),
        "forces_eV_per_A":      forces_ev.tolist(),
        "torchani_version": torchani.__version__,
    }
    aev_data = {
        "description": "ANI-2x AEV vectors for N₂ dimer; shape (2, 1008)",
        "n_atoms":     2,
        "aev_length":  int(aev_np.shape[1]),
        "species":     ["N", "N"],
        "aevs":        aev_np.tolist(),
    }

    with open("data/ani_reference/n2_dimer_ani2x.json", "w") as f:
        json.dump(data, f, indent=2)
    with open("data/ani_reference/n2_aevs.json", "w") as f:
        json.dump(aev_data, f, indent=2)

    print(f"  E = {energy_ha:.10f} Ha = {energy_ev:.10f} eV")
    print(f"  F[0] = {forces_ev[0].tolist()} eV/Å")
    print(f"  AEV shape: {aev_np.shape}  (saved to n2_aevs.json)")
    return data


# -------------------------------------------------------------------------
# 6mrr protein reference
# -------------------------------------------------------------------------

def protein_reference(model, pdb_path="data/6mrr_equil.pdb"):
    print(f"\n=== 6mrr protein ({pdb_path}) ===")
    try:
        import ase.io
    except ImportError:
        print("  ase not installed — skipping (pip install ase)")
        return None

    try:
        atoms_ase = ase.io.read(pdb_path)
    except Exception as e:
        print(f"  Could not read PDB: {e}")
        return None

    n_atoms  = len(atoms_ase)
    elements = atoms_ase.get_chemical_symbols()
    print(f"  {n_atoms} atoms, elements: {set(elements)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}  (this may take a while on CPU)")
    model   = model.to(device)

    atomic_nums = torch.tensor([atoms_ase.numbers], dtype=torch.long, device=device)
    coords      = torch.tensor([atoms_ase.get_positions()], dtype=torch.float32,
                               device=device, requires_grad=True)

    result   = model((atomic_nums, coords))
    energy_ha = result.energies.item()
    energy_ev = energy_ha * HARTREE_TO_EV

    grads     = torch.autograd.grad(result.energies.sum(), coords)[0]
    forces_ha = (-grads[0]).detach().cpu().numpy()
    forces_ev = forces_ha * HARTREE_TO_EV

    data = {
        "description": f"ANI-2x reference: {pdb_path}, no PBC, device={device}",
        "n_atoms":     n_atoms,
        "elements":    elements,
        "energy_hartree": float(energy_ha),
        "energy_eV":      float(energy_ev),
        "forces_eV_per_A": forces_ev.tolist(),
        "torchani_version": torchani.__version__,
    }
    with open("data/ani_reference/6mrr_ani2x.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"  E = {energy_ha:.6f} Ha = {energy_ev:.6f} eV")
    print(f"  F[0] = {forces_ev[0].tolist()} eV/Å  (saved {n_atoms} atoms)")
    return data


# -------------------------------------------------------------------------
# Export weights + AEV params to HDF5 for the Julia ANIPotential loader
# -------------------------------------------------------------------------

def export_weights_h5(model, output="data/ani_reference/ani2x.h5"):
    print(f"\n=== Exporting ANI-2x weights → {output} ===")
    try:
        import h5py
    except ImportError:
        print("  h5py not installed — skipping (pip install h5py)")
        return

    aev = model.aev_computer

    with h5py.File(output, "w") as h5:
        # ---- AEV hyperparameters ----
        ag = h5.create_group("aev_params")
        ag.create_dataset("Rcr",     data=float(aev.Rcr))
        ag.create_dataset("Rca",     data=float(aev.Rca))
        ag.create_dataset("EtaR",    data=aev.EtaR.squeeze().numpy())
        ag.create_dataset("ShfR",    data=aev.ShfR.squeeze().numpy())
        ag.create_dataset("EtaA",    data=aev.EtaA.squeeze().numpy())
        ag.create_dataset("Zeta",    data=float(aev.Zeta.item()))
        ag.create_dataset("ShfA",    data=aev.ShfA.squeeze().numpy())   # angular shift (θ_s)
        ag.create_dataset("ShfZ",    data=aev.ShfZ.squeeze().numpy())   # radial shift for angular (r_s_A)
        ag.create_dataset("species", data=[e.encode() for e in ELEMENTS])
        # Atomic self-energies (Hartree) — same order as ELEMENTS
        ag.create_dataset("self_energies",
                          data=model.energy_shifter.self_energies.numpy())

        # ---- Neural network weights per ensemble member ----
        ensemble = model.neural_networks   # torchani.nn.Ensemble, len=8
        n_members = len(ensemble)
        print(f"  Ensemble size: {n_members}")

        for ens_idx in range(n_members):
            ens_grp = h5.create_group(str(ens_idx))
            member  = ensemble[ens_idx]   # torchani.nn.ANIModel

            for elem in ELEMENTS:
                elem_seq  = getattr(member, elem)   # torch.nn.Sequential
                elem_grp  = ens_grp.create_group(elem)
                file_idx  = 0   # HDF5 key: 0, 2, 4, 6 (even = Dense layer)

                for module in elem_seq:
                    classname = type(module).__name__
                    if classname == "Linear":
                        layer_grp = elem_grp.create_group(str(file_idx))
                        layer_grp.create_dataset(
                            "weight", data=module.weight.detach().numpy())
                        layer_grp.create_dataset(
                            "bias",   data=module.bias.detach().numpy())
                        file_idx += 2   # skip the CELU activation index

    # Quick verification
    with h5py.File(output, "r") as h5:
        w = h5["0/H/0/weight"]
        print(f"  H layer-0 weight shape: {w.shape}  (expected (256, 1008))")
        rcr = float(h5["aev_params/Rcr"][()])
        rca = float(h5["aev_params/Rca"][()])
        print(f"  Rcr={rcr} Å, Rca={rca} Å")

    print(f"  Saved to {output}")


# -------------------------------------------------------------------------
# H₂O water molecule reference (multi-element: O + 2H)
# -------------------------------------------------------------------------

def water_reference(model):
    """Generate ANI-2x reference for H₂O at TIP3P geometry (no PBC)."""
    print("\n=== H₂O water molecule (TIP3P geometry, no PBC) ===")

    r_oh   = 0.9572          # O-H bond length (Å)
    theta  = 104.52 * np.pi / 180.0   # H-O-H angle (rad)

    # O at origin, H1 along +x, H2 in xy plane
    atomic_nums = [8, 1, 1]
    positions   = [
        [0.0,                  0.0, 0.0],
        [r_oh,                 0.0, 0.0],
        [r_oh * np.cos(theta), r_oh * np.sin(theta), 0.0],
    ]

    species = torch.tensor([atomic_nums], dtype=torch.long)
    coords  = torch.tensor([positions],   dtype=torch.float32, requires_grad=True)

    result    = model((species, coords))
    energy_ha = result.energies.item()
    energy_ev = energy_ha * HARTREE_TO_EV

    grads      = torch.autograd.grad(result.energies.sum(), coords)[0]
    forces_ha  = (-grads[0]).detach().numpy()
    forces_ev  = forces_ha * HARTREE_TO_EV

    aev_np = compute_aev_vectors(model, atomic_nums, positions)

    data = {
        "description":          "ANI-2x reference: H₂O at TIP3P geometry, no PBC",
        "species":              ["O", "H", "H"],
        "atomic_numbers":       atomic_nums,
        "coordinates_A":        positions,
        "energy_hartree":       float(energy_ha),
        "energy_eV":            float(energy_ev),
        "forces_hartree_per_A": forces_ha.tolist(),
        "forces_eV_per_A":      forces_ev.tolist(),
        "torchani_version":     torchani.__version__,
    }
    aev_data = {
        "description": "ANI-2x AEV vectors for H₂O; shape (3, 1008)",
        "n_atoms":     3,
        "aev_length":  int(aev_np.shape[1]),
        "species":     ["O", "H", "H"],
        "aevs":        aev_np.tolist(),
    }

    with open("data/ani_reference/water_ani2x.json", "w") as f:
        json.dump(data, f, indent=2)
    with open("data/ani_reference/water_aevs.json", "w") as f:
        json.dump(aev_data, f, indent=2)

    print(f"  E = {energy_ha:.10f} Ha = {energy_ev:.10f} eV")
    print(f"  F[O]  = {forces_ev[0].tolist()} eV/Å")
    print(f"  F[H1] = {forces_ev[1].tolist()} eV/Å")
    print(f"  F[H2] = {forces_ev[2].tolist()} eV/Å")
    print(f"  AEV shape: {aev_np.shape}  (saved to water_aevs.json)")
    return data


# -------------------------------------------------------------------------
# Print AEV parameter summary
# -------------------------------------------------------------------------

def print_aev_summary(model):
    aev = model.aev_computer
    print("\n=== AEV parameter summary ===")
    print(f"  Rcr (radial cutoff):   {float(aev.Rcr):.4f} Å")
    print(f"  Rca (angular cutoff):  {float(aev.Rca):.4f} Å")
    print(f"  EtaR: {aev.EtaR.squeeze().tolist()}")
    print(f"  ShfR: {aev.ShfR.squeeze().tolist()}")
    print(f"  EtaA: {aev.EtaA.squeeze().tolist()}")
    print(f"  Zeta: {float(aev.Zeta.item())}")
    # TorchANI naming: ShfA = radial shifts for angular Gaussian; ShfZ = angular shifts.
    print(f"  ShfA (radial shifts r_s_A, Å): {aev.ShfA.squeeze().tolist()}")
    print(f"  ShfZ (angular shifts θ_s, rad): {aev.ShfZ.squeeze().tolist()}")

    n_sp   = len(ELEMENTS)
    n_etaR = aev.EtaR.numel()
    n_shfA = aev.ShfA.numel()
    n_etaA = aev.EtaA.numel()
    n_shfZ = aev.ShfZ.numel()
    n_pairs = n_sp * (n_sp + 1) // 2
    n_rad   = n_sp * n_etaR
    n_ang   = n_pairs * n_shfA * n_etaA * n_shfZ
    print(f"\n  Radial AEV:  {n_sp} species × {n_etaR} η_R = {n_rad}")
    print(f"  Angular AEV: {n_pairs} pairs × {n_shfA} ShfA × {n_etaA} EtaA × {n_shfZ} ShfZ = {n_ang}")
    print(f"  Total:       {n_rad + n_ang}")


# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TorchANI reference data")
    parser.add_argument("--skip-protein", action="store_true")
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()

    model = get_model()

    n2_dimer_reference(model)
    water_reference(model)
    print_aev_summary(model)

    if not args.skip_protein:
        protein_reference(model)
    else:
        print("\nSkipping protein (--skip-protein)")

    if not args.skip_weights:
        export_weights_h5(model)
    else:
        print("\nSkipping weight export (--skip-weights)")

    print("\nDone. Files in data/ani_reference/")
