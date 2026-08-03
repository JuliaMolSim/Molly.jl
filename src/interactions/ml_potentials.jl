# Machine-learning interatomic potentials (core definitions).
#
# The abstract type, the ANIPotential struct, the scalar AEV helpers (cosine_cutoff,
# celu01) and the public function stubs live here in core Molly (no Lux/HDF5/KA
# dependency). The implementations that need those packages are in the extensions:
#   ext/MollyLuxExt.jl        — AEV + energy + HDF5 loader, plus the GPU-portable AEV kernels,
#                               on-device energy, and the single analytic forces path / forces!.
#                               Triggered by Lux + HDF5; KernelAbstractions is a strong Molly
#                               dependency so the GPU code needs no separate extension.
#
# See ext/MollyLuxExt.jl for the ANI method overview and per-equation references
# (ANI-1: Smith et al., Chem. Sci. 2017; ANI-2x: Devereux et al., JCTC 2020).

# Base type for ML interatomic potentials. Kept as a shared supertype for current and
# future ML potentials (ANIPotential, ...); useful for dispatch/`isa` checks.
abstract type AbstractMLPotential end

# ANI energies are produced in Hartree; MD in Molly is typically in eV.
const HARTREE_TO_EV = 27.211396132

"""
    cosine_cutoff(r, r_c)

Smooth cutoff function: `0.5*(1+cos(π*r/r_c))` for `r < r_c`, else `0`.
Used by the ANI AEV (radial/angular symmetry functions, [ANI-1] Eq. 2). A plain scalar
function with no Lux/HDF5 dependency, so the CPU and GPU (KA) AEV paths share one
definition.
"""
@inline function cosine_cutoff(r::T, r_c::T) where T
    r < r_c ? T(0.5) * (one(T) + cos(T(π) * r / r_c)) : zero(T)
end

"""
    celu01(x)

CELU activation with α=0.1 — the nonlinearity between Dense layers of each ANI element
network (matches the TorchANI ANI-2x architecture). `celu01(x) = x` for `x ≥ 0`,
`0.1*(exp(x/0.1) - 1)` otherwise. Exported so AD backends can register rules without
depending on MollyLuxExt.
"""
@noinline celu01(x::T) where T = x >= zero(T) ? x : T(0.1) * (exp(x / T(0.1)) - one(T))

# Convert an ANI energy (eV, unitless) to the system's energy units.
function ani_energy_to_units(E_eV, energy_units)
    if energy_units == NoUnits
        return E_eV
    elseif dimension(energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return uconvert(energy_units, E_eV * Unitful.Na * u"eV")
    else
        return uconvert(energy_units, E_eV * u"eV")
    end
end

# Convert an ANI force SVector (eV/Å, unitless) to the system's force units and device.
function ani_force_to_units(fi::SVector{D,T}, force_units, ::Type{AT}) where {D, T, AT}
    f_unit = if force_units == NoUnits
        fi
    elseif dimension(force_units) == u"𝐋 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        uconvert.(force_units, fi .* (Unitful.Na * u"eV/Å"))
    else
        uconvert.(force_units, fi .* u"eV/Å")
    end
    return to_device(f_unit, AT)
end

# Strip length units from a boundary so it is consistent with unit-stripped coords (Å).
# The AEV paths (CPU and GPU kernels) pass the stripped boundary to `vector(...)` for the
# minimum-image convention; keeping a shared definition means Cubic and Triclinic boundaries
# are handled identically on both paths.
strip_boundary(b::CubicBoundary) =
    unit(b.side_lengths[1]) == NoUnits ? b : CubicBoundary(ustrip.(u"Å", b.side_lengths))

function strip_boundary(b::TriclinicBoundary{D, T, C, A}) where {D, T, C, A}
    unit(b.basis_vectors[1][1]) == NoUnits && return b
    bv = SVector(ntuple(i -> ustrip.(u"Å", b.basis_vectors[i]), 3))
    return TriclinicBoundary(bv; approx_images=A)
end

"""
    ANIPotential(path; force_units, energy_units, T, ensemble_idx)

Load an ANI neural network potential from an HDF5 file exported by
`test/torchani_reference.py`. Requires `Lux` and `HDF5` to be loaded.

By default all ensemble members are loaded and energies are averaged.
Pass `ensemble_idx` to load only a specific member (wrapped in a length-1 vector).

Note: the ANI-2x weights are `Float32`, so the energy/force paths run in `Float32`
internally regardless of the system's coordinate type. Periodic systems must use a
neighbour finder (the neighbour-list path applies the minimum-image convention);
boundaries are handled via `vector(...)`.
"""
struct ANIPotential{M, PV, SV, SP, P, SE, D, F, E} <: AbstractMLPotential
    model::M          # NamedTuple of per-element Lux.Chain sub-networks (shared architecture)
    ps_vec::PV        # Vector of per-element parameter NamedTuples, one per ensemble member
    st_vec::SV        # Vector of per-element state NamedTuples, one per ensemble member
    species_map::SP   # Dict{String,Int}: element → 1-based index
    aev_params::P     # NamedTuple: η_R, r_s_R, r_c_R, η_A, r_s_A (ShfA, Å), θ_s (ShfZ, rad), ζ, r_c_A
    self_energies::SE # Vector: atomic self-energy per species (Hartree)
    cutoff::D         # max(r_c_R, r_c_A), plain Float (Å)
    force_units::F
    energy_units::E
    buffers::Ref{Any} # lazily-initialized AEVBuffers for zero-allocation AEV computation
end

# Constructor and implementation are in ext/MollyLuxExt.jl (needs Lux + HDF5).
function ANIPotential(path::AbstractString; kwargs...)
    error("ANIPotential requires Lux and HDF5 to be loaded: `using Lux, HDF5`")
end

"""
    compute_aevs(coords, species_indices, neighbors, boundary, aev_params, n_species)

Compute atomic environment vectors (AEVs) for all atoms.
Returns a `(n_atoms, aev_length)` matrix. Implementation is in ext/MollyLuxExt.jl.
"""
function compute_aevs end

"""
    compute_aevs_ka(coords, species_indices, aev_params, n_species; backend=nothing, neighbors=nothing)

GPU-portable AEV computation using KernelAbstractions — one thread per atom (or one
workgroup per atom with `write_reduce=true`). `coords`/`species` must already live on the
target device; `backend` defaults to `KernelAbstractions.get_backend(coords)`. Requires
`KernelAbstractions`, `Lux`, and `HDF5` to be loaded.
"""
function compute_aevs_ka(args...; kwargs...)
    error("compute_aevs_ka requires KernelAbstractions, Lux, and HDF5 to be loaded: " *
          "`using KernelAbstractions, Lux, HDF5`")
end

"""
    compute_ani_energy_ka(coords, species_indices, pot, n_species; backend=nothing, neighbors=nothing)

End-to-end ANI energy (eV) computed on-device: GPU AEV (`compute_aevs_ka`) followed by
the per-element neural networks run on the same backend, summed over atoms and averaged
over ensemble members. Requires `KernelAbstractions`, `Lux`, and `HDF5`.
"""
function compute_ani_energy_ka(args...; kwargs...)
    error("compute_ani_energy_ka requires KernelAbstractions, Lux, and HDF5 to be loaded: " *
          "`using KernelAbstractions, Lux, HDF5`")
end

"""
    compute_ani_forces_ka(coords, species_indices, pot, n_species; backend=nothing, neighbors=nothing, boundary=nothing)

On-device ANI forces (eV/Å), the analytic counterpart of [`compute_ani_energy_ka`]. Runs the
GPU AEV forward, a manual VJP through the per-element neural networks for `∂E/∂G`, then the
backward radial/angular AEV kernels for `∂E/∂r`, giving `F = -∂E/∂r` averaged over the
ensemble. Returns a `Vector{SVector{3}}`. Obeys the minimum-image convention via `boundary`.
Requires `KernelAbstractions`, `Lux`, and `HDF5`.
"""
function compute_ani_forces_ka(args...; kwargs...)
    error("compute_ani_forces_ka requires KernelAbstractions, Lux, and HDF5 to be loaded: " *
          "`using KernelAbstractions, Lux, HDF5`")
end
