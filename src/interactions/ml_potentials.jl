# Machine-learning interatomic potentials (core definitions).
#
# The ANIPotential struct, the scalar AEV helpers (cosine_cutoff, celu01) and the public
# function stubs live here in core Molly. The implementations that need Lux/HDF5/
# KernelAbstractions are in ext/MollyLuxExt.jl (loaded when Lux and HDF5 are available).

export
    ANIPotential,
    ani2x_data_dir,
    compute_aevs,
    AllegroPotential

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

# ---- Shared ML-potential unit / coordinate conversion -------------------------------------------
# Molly stores unitless coordinates in nm; the ML potentials work internally in Å. These helpers
# are shared by ANIPotential (ext/MollyLuxExt.jl) and AllegroPotential.

const NM_TO_ANGSTROM = 10

# In-place conversion of coordinates to unitless Å in a pre-allocated buffer (zero allocations).
# The unit check is on the element type (no indexing), so it also works for device arrays.
function coords_to_angstrom_into!(out::AbstractVector{SVector{D,TF}},
                                  coords::AbstractVector{SVector{D,T}}) where {D, TF, T}
    if T <: Real   # unitless Molly coords are nm
        @inbounds for i in eachindex(coords)
            out[i] = SVector{D,TF}(coords[i]) * TF(NM_TO_ANGSTROM)
        end
    else
        @inbounds for i in eachindex(coords)
            out[i] = SVector{D,TF}(ustrip.(u"Å", coords[i]))
        end
    end
    return out
end

# Non-mutating conversion of coordinates to unitless Å, staying on the coords' device. Unitless
# Molly coords are treated as nm. The unit check is on the element type so this works on GPU arrays.
function coords_to_angstrom(coords)
    eltype(eltype(coords)) <: Real ? coords .* NM_TO_ANGSTROM : ustrip_vec.(u"Å", coords)
end

# Convert a boundary to unitless Å (unitless side lengths are treated as nm).
strip_boundary(b::CubicBoundary) =
    unit(b.side_lengths[1]) == NoUnits ? CubicBoundary(b.side_lengths .* NM_TO_ANGSTROM) :
                                         CubicBoundary(ustrip.(u"Å", b.side_lengths))

function strip_boundary(b::TriclinicBoundary{D, T, C, A}) where {D, T, C, A}
    if unit(b.basis_vectors[1][1]) == NoUnits
        bv = SVector(ntuple(i -> b.basis_vectors[i] .* NM_TO_ANGSTROM, 3))
    else
        bv = SVector(ntuple(i -> ustrip.(u"Å", b.basis_vectors[i]), 3))
    end
    return TriclinicBoundary(bv; approx_images=A)
end

# Convert an ML-potential energy (eV, unitless) to the system's energy units. In the no-units case
# the Molly convention is kJ/mol.
function ml_energy_to_units(E_eV, energy_units)
    if energy_units == NoUnits
        return ustrip(u"kJ * mol^-1", E_eV * Unitful.Na * u"eV")
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return uconvert(energy_units, E_eV * Unitful.Na * u"eV")
    else
        return uconvert(energy_units, E_eV * u"eV")
    end
end

# Convert an ML-potential force SVector (eV/Å, unitless) to the system's force units. In the
# no-units case the Molly convention is kJ/mol/nm (the eV/Å → kJ/mol/nm conversion includes the
# Å → nm factor, so no extra scaling is needed).
function ml_force_to_units(fi::SVector{D,T}, force_units) where {D, T}
    if force_units == NoUnits
        return ustrip.(u"kJ * mol^-1 * nm^-1", fi .* (Unitful.Na * u"eV/Å"))
    elseif dimension(force_units) == u"𝐋 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return uconvert.(force_units, fi .* (Unitful.Na * u"eV/Å"))
    else
        return uconvert.(force_units, fi .* u"eV/Å")
    end
end

# ---- Allegro (equivariant GNN) potential -------------------------------------------------------
#
# Allegro (Musaelian et al. 2023) is a strictly-local O(3)-equivariant potential: its energy is a
# sum over directed edges within a cutoff. The equivariant primitives (irreps, real spherical
# harmonics, Clebsch-Gordan tensor products, equivariant linear layers), the model forward and
# analytic forces all live in core Molly (src/equivariant/); only loading weights from an HDF5 file
# needs the extension (loaded with `using HDF5`).

# SiLU / swish activation, the scalar nonlinearity in Allegro's latent MLPs.
silu(x::T) where T = x / (one(T) + exp(-x))

"""
    AllegroPotential(path; T=Float32)

Load a native Allegro equivariant neural-network potential from an HDF5 file exported by
`test/allegro_reference.py`. Requires `HDF5` to be loaded.

The element of each atom is read from `atoms_data[i].element` and mapped through the model's
species list. Coordinates without units are treated as nm (Molly convention) and converted to the
Å the model uses. Energy and analytic forces are computed on the CPU; a GPU-backed system is
supported via a host round-trip.
"""
struct AllegroPotential{M, SP, D} <: AbstractMLPotential
    model::M           # AllegroModel (config + precomputed tensor-product paths/CG + weights)
    species_map::SP    # Dict{String,Int}: element → 1-based index
    cutoff::D          # r_cutoff, plain Float (Å)
    buffers::Ref{Any}  # lazily-initialized per-edge scratch buffers
end

# Fallback constructor. The real `AbstractString` method is in ext/MollyHDF5Ext.jl (needs HDF5);
# `path` is left untyped here so that method is strictly more specific and does not overwrite this
# one (method overwriting is an error during extension precompilation).
function AllegroPotential(path; kwargs...)
    error("AllegroPotential requires HDF5 to be loaded: `using HDF5`")
end

# Energy: sum over directed edges within the cutoff, in the system's energy units.
function AtomsCalculators.potential_energy(sys::System, inter::AllegroPotential; kwargs...)
    m = inter.model
    coords_A = [SVector{3,Float64}(c) for c in from_device(coords_to_angstrom(sys.coords))]
    species = [inter.species_map[sys.atoms_data[i].element] for i in eachindex(sys.coords)]
    E = allegro_total_energy(m, coords_A, species, strip_boundary(sys.boundary), m.r_c)
    return ml_energy_to_units(E, sys.energy_units)
end

# Analytic forces F = -∂E/∂r, accumulated into `fs` in the system's force units.
function AtomsCalculators.forces!(fs, sys::System, inter::AllegroPotential; kwargs...)
    m = inter.model
    coords_A = [SVector{3,Float64}(c) for c in from_device(coords_to_angstrom(sys.coords))]
    species = [inter.species_map[sys.atoms_data[i].element] for i in eachindex(sys.coords)]
    F = allegro_forces(m, coords_A, species, strip_boundary(sys.boundary), m.r_c)  # eV/Å
    inc = [ml_force_to_units(SVector{3,Float64}(F[i]), sys.force_units) for i in eachindex(F)]
    if fs isa Array
        fs .+= inc
    else
        # GPU-backed force buffer: upload the host increment (matching fs's element type) and add
        # on-device to avoid scalar indexing into the GPU array.
        fs .+= to_device(convert.(eltype(fs), inc), array_type(sys))
    end
    return fs
end
