# Machine-learning interatomic potentials (core definitions).
#
# The ANIPotential struct, the scalar AEV helpers (cosine_cutoff, celu01) and the public
# function stubs live here in core Molly. The implementations that need Lux/HDF5/
# KernelAbstractions are in ext/MollyLuxExt.jl (loaded when Lux and HDF5 are available).

export
    ANIPotential,
    ani2x_data_dir,
    compute_aevs

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

# Compute atomic environment vectors (AEVs) for all atoms, returning an
# (n_atoms, aev_length) matrix. Implementation is in ext/MollyLuxExt.jl.
function compute_aevs end

# GPU-portable AEV computation with KernelAbstractions (implementation in ext/MollyLuxExt.jl).
function compute_aevs_ka(args...; kwargs...)
    error("compute_aevs_ka requires KernelAbstractions, Lux, and HDF5 to be loaded: " *
          "`using KernelAbstractions, Lux, HDF5`")
end

# End-to-end on-device ANI energy (implementation in ext/MollyLuxExt.jl).
function compute_ani_energy_ka(args...; kwargs...)
    error("compute_ani_energy_ka requires KernelAbstractions, Lux, and HDF5 to be loaded: " *
          "`using KernelAbstractions, Lux, HDF5`")
end

# On-device analytic ANI forces (implementation in ext/MollyLuxExt.jl).
function compute_ani_forces_ka(args...; kwargs...)
    error("compute_ani_forces_ka requires KernelAbstractions, Lux, and HDF5 to be loaded: " *
          "`using KernelAbstractions, Lux, HDF5`")
end
