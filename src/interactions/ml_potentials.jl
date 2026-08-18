# Machine-learning interatomic potentials (core definitions).
#
# The ANIPotential struct, the scalar AEV helpers (cosine_cutoff, celu01) and the public
# function stubs live here in core Molly. The implementations that need Lux/HDF5/
# KernelAbstractions are in ext/MollyLuxExt.jl (loaded when Lux and HDF5 are available).

export
    ANIPotential,
    ani2x_data_dir,
    compute_aevs,
    AllegroPotential,
    allegro_data_dir

# Base type for ML interatomic potentials, a shared supertype for current and future ones.
abstract type AbstractMLPotential end

# ANI energies are produced in Hartree; MD in Molly is typically in eV.
const HARTREE_TO_EV = ustrip(u"eV", 1u"Eh_au")

# Smooth cutoff f_C ([ANI-1] Eq. 2), shared by the CPU and GPU AEV paths.
@inline function cosine_cutoff(r::T, r_c::T) where T
    r < r_c ? T(0.5) * (one(T) + cos(T(π) * r / r_c)) : zero(T)
end

# CELU activation with α=0.1, the nonlinearity between each element network's Dense layers.
# Defined in core Molly so AD backends can register rules without depending on MollyLuxExt.
celu01(x::T) where T = x >= zero(T) ? x : T(0.1) * (exp(x / T(0.1)) - one(T))

"""
    ANIPotential(path; T=Float32, ensemble_idx=nothing)

Load an ANI-2x neural network potential from an HDF5 file exported by
`test/torchani_reference.py`. Requires `Lux` and `HDF5` to be loaded.

The system's `atoms_data` is required, since the element of each atom is read from
`atoms_data[i].element`. The supported elements are H, C, N, O, S, F and Cl.

By default all ensemble members are loaded and their energies averaged. Pass
`ensemble_idx` (one-indexed, `1:8` for ANI-2x) to load only a single member.

Coordinates without units are treated as nm following the Molly convention, and
converted internally to the Å the ANI parameters use. Periodic systems must use a
neighbour finder; the neighbour-list path applies the minimum-image convention.

Note: the ANI-2x weights are `Float32`, so the energy/force paths run in `Float32`
internally regardless of the system's coordinate type.
"""
struct ANIPotential{M, PV, SV, SP, P, SE, D} <: AbstractMLPotential
    model::M          # NamedTuple of per-element Lux.Chain sub-networks (shared architecture)
    ps_vec::PV        # Vector of per-element parameter NamedTuples, one per ensemble member
    st_vec::SV        # Vector of per-element state NamedTuples, one per ensemble member
    species_map::SP   # Dict{String,Int}: element → 1-based index
    aev_params::P     # NamedTuple: η_R, r_s_R, r_c_R, η_A, r_s_A (ShfA, Å), θ_s (ShfZ, rad), ζ, r_c_A
    self_energies::SE # Vector: atomic self-energy per species (Hartree)
    cutoff::D         # max(r_c_R, r_c_A), plain Float (Å)
    buffers::Ref{Any} # lazily-initialized AEVBuffers for zero-allocation AEV computation
end

# Fallback constructor. The real `AbstractString` method is in ext/MollyLuxExt.jl (needs Lux +
# HDF5); `path` is left untyped here so that method is strictly more specific and does not
# overwrite this one (method overwriting is an error during extension precompilation).
function ANIPotential(path; kwargs...)
    error("ANIPotential requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end

"""
    ani2x_data_dir()

Path to the ANI-2x data directory: a lazily-downloaded artifact holding `ani2x.h5` (the model
weights) and `6mrr_ani2x.json` (TorchANI reference energies). Requires `Lux` and `HDF5`. Load
the potential with `ANIPotential(joinpath(ani2x_data_dir(), "ani2x.h5"))`.
"""
function ani2x_data_dir end

"""
    compute_aevs(coords, species_indices, neighbors, boundary, aev_params, n_species)

Compute the Atomic Environment Vectors (AEVs) for all atoms, returning an
`(n_atoms, aev_length)` matrix. `neighbors` is a `NeighborList` (or `nothing` for an all-pairs
build). This is the reference AEV path; the GPU-portable kernel version is `compute_aevs_ka`.
Requires `Lux` and `HDF5`. Implementation is in ext/MollyLuxExt.jl.
"""
function compute_aevs end

# GPU-portable AEV computation (implementation in ext/MollyLuxExt.jl). KernelAbstractions is a
# strong Molly dependency, so only Lux and HDF5 gate the extension.
function compute_aevs_ka(args...; kwargs...)
    error("compute_aevs_ka requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end

# End-to-end on-device ANI energy (implementation in ext/MollyLuxExt.jl).
function compute_ani_energy_ka(args...; kwargs...)
    error("compute_ani_energy_ka requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end

# On-device analytic ANI forces (implementation in ext/MollyLuxExt.jl).
function compute_ani_forces_ka(args...; kwargs...)
    error("compute_ani_forces_ka requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end

# ---- Allegro (equivariant GNN) potential -------------------------------------------------------
#
# Allegro (Musaelian et al. 2023) is a strictly-local O(3)-equivariant potential: its energy is a
# sum over directed edges within a cutoff. The equivariant primitives it is built from (irreps,
# real spherical harmonics, Clebsch-Gordan tensor products, equivariant linear layers) live in
# core Molly under src/equivariant/ and are exercised directly by test/equivariant.jl. The trained
# model container, HDF5 weight loading and AtomsCalculators wiring are provided by an extension
# (loaded with `using Lux, HDF5`); the function stubs below error until then.

# SiLU / swish activation, the scalar nonlinearity in Allegro's latent MLPs. Defined in core so
# AD backends and the extension share one definition.
silu(x::T) where T = x / (one(T) + exp(-x))

"""
    AllegroPotential(path; T=Float32)

Load a native Allegro equivariant neural-network potential from an HDF5 file exported by
`test/allegro_reference.py`. Requires `Lux` and `HDF5` to be loaded.

The element of each atom is read from `atoms_data[i].element` and mapped through the model's
species list. Coordinates without units are treated as nm (Molly convention) and converted to the
Å the model uses; periodic systems must use a neighbour finder (minimum-image convention).

!!! note
    This is under active development. The equivariant primitives (spherical harmonics,
    Clebsch-Gordan tensor products, equivariant linear layers) and their analytic gradients are
    implemented and validated in core Molly; the energy/forces model assembly and weight loading
    land in a follow-up. `path` is left untyped here so the extension's `::AbstractString`
    method is strictly more specific (avoids a precompile-time method overwrite).
"""
struct AllegroPotential{M, SP, D} <: AbstractMLPotential
    model::M           # AllegroModel (config + precomputed tensor-product paths/CG + weights)
    species_map::SP    # Dict{String,Int}: element → 1-based index
    cutoff::D          # r_cutoff, plain Float (Å)
    buffers::Ref{Any}  # lazily-initialized per-edge scratch buffers
end

# Fallback constructor. The real `AbstractString` method is in the extension (needs Lux + HDF5);
# `path` is left untyped here so that method is strictly more specific and does not overwrite this
# one (method overwriting is an error during extension precompilation).
function AllegroPotential(path; kwargs...)
    error("AllegroPotential requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end

"""
    allegro_data_dir()

Path to the Allegro data directory (a lazily-downloaded artifact holding the exported model
weights and reference data). Requires `Lux` and `HDF5`. Implementation is in the extension.
"""
function allegro_data_dir end

# End-to-end on-device Allegro energy (implementation in the extension).
function compute_allegro_energy_ka(args...; kwargs...)
    error("compute_allegro_energy_ka requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end

# On-device analytic Allegro forces (implementation in the extension).
function compute_allegro_forces_ka(args...; kwargs...)
    error("compute_allegro_forces_ka requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end
