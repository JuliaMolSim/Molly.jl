"""Build the real nequip-allegro model + its ASE calculator (the same small HCNO model as
test/allegro_package_reference.py). Used from Julia via PythonCall so Molly's ASECalculator can
drive the actual package — the external-validation path for Joe's review (#283), and the template
for running an MD trajectory with a trained model.

Point PythonCall at a Python that has nequip-allegro, e.g. set before `using PythonCall`:
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
    ENV["JULIA_PYTHONCALL_EXE"]   = "<env>/bin/python"
"""
import torch
from nequip.utils.global_state import set_global_state
from nequip.data.transforms import NeighborListTransform, ChemicalSpeciesToAtomTypeMapper
from nequip.integrations.ase import NequIPCalculator
from allegro.model import AllegroModel

TYPE_NAMES = ["H", "C", "N", "O"]
RC = 4.0

def make_calc(device="cpu"):
    """The seed-0 random model matching data/allegro_reference/allegro_package_*."""
    set_global_state()
    torch.manual_seed(0)
    m = AllegroModel(seed=0, model_dtype="float64", l_max=2, r_max=RC, type_names=TYPE_NAMES,
        radial_chemical_embed={"_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                               "num_bessels": 8, "bessel_trainable": False, "polynomial_cutoff_p": 6},
        num_layers=2, num_scalar_features=32, num_tensor_features=8,
        avg_num_neighbors=10.0).double().eval()
    return _calc_from_model(m, TYPE_NAMES, device)

def calc_from_compiled(path, type_names, device="cpu"):
    """A trained model compiled with `nequip-compile --target ase` (for the 6mrr trajectory)."""
    return NequIPCalculator.from_compiled_model(path, device=device)

def _calc_from_model(model, type_names, device):
    mapper = ChemicalSpeciesToAtomTypeMapper(
        model_type_names=type_names, chemical_species_to_atom_type_map={t: t for t in type_names})
    return NequIPCalculator(model, device=device, transforms=[mapper, NeighborListTransform(r_max=RC)])
