# ANI forces via single-pass Enzyme reverse-mode AD.
# Loaded when Enzyme, Lux, and HDF5 are all in the environment.
# Overrides the finite-difference forces! from MollyLuxExt.
#
# __precompile__(false): method overwriting between extension modules is not
# allowed during precompilation; loading at runtime avoids this restriction.
__precompile__(false)

module MollyLuxEnzymeExt

using Molly
using Molly: from_device, to_device
import AtomsCalculators
using Enzyme, Lux, HDF5
using ChainRulesCore
using StaticArrays, Unitful, LinearAlgebra

# ============================================================================
# AD rules for Molly.celu01
# ============================================================================

# ChainRule (for any ChainRules-compatible AD system):
# d/dx celu(x; α=0.1) = 1 if x ≥ 0, else exp(x/0.1)
function ChainRulesCore.rrule(::typeof(Molly.celu01), x::T) where T
    y = Molly.celu01(x)
    function celu01_pullback(ȳ)
        dx = x >= zero(x) ? one(x) : exp(x / T(0.1))
        return NoTangent(), ȳ * dx
    end
    return y, celu01_pullback
end

# Enzyme augmented_primal: run forward pass and store input as tape for reverse.
# Conditional primal return avoids AugmentedRuleReturnError when primal not requested.
function EnzymeRules.augmented_primal(config,
                                       ::EnzymeRules.Const{typeof(Molly.celu01)},
                                       ::Type{<:EnzymeRules.Active},
                                       x::EnzymeRules.Active)
    val = Molly.celu01(x.val)
    primal = EnzymeRules.needs_primal(config) ? val : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, x.val)  # tape = input x
end

# Enzyme reverse: analytical derivative of celu, bypasses exp() symbol duplication.
function EnzymeRules.reverse(config,
                               ::EnzymeRules.Const{typeof(Molly.celu01)},
                               dret, tape, x::EnzymeRules.Active)
    xval = tape
    d = xval >= zero(xval) ? one(xval) : exp(xval / typeof(xval)(0.1f0))
    return (nothing, dret.val * d)
end

# ============================================================================
# Shared utilities
# ============================================================================

function _lue_strip_boundary(b::CubicBoundary)
    unit(b.side_lengths[1]) == NoUnits ? b :
        CubicBoundary(ustrip.(u"Å", b.side_lengths))
end

function _lue_f32coords(coords::AbstractVector{SVector{D,T}}) where {D,T}
    if unit(first(coords)[1]) == NoUnits
        [SVector{D, Float32}(sv) for sv in coords]
    else
        [SVector{D, Float32}(ustrip_vec(u"Å", sv)) for sv in from_device(coords)]
    end
end

# ============================================================================
# Single-pass Enzyme energy kernel
# ============================================================================

# Full energy (Hartree) for one ensemble member — AEVs + Lux network.
# Differentiable replica of the energy pipeline (forces are F = −∇E):
#   AEVs via compute_aevs — radial [ANI-1] Eq. 3, angular [ANI-1] Eq. 4, cutoff Eq. 2;
#   energy E = Σ_i E_i ([ANI-1] Eq. 1) from the per-element networks + self-energies.
# coords_mat: (3, N) Float32 — the only Active argument.
# All other args are Const so Enzyme doesn't need to differentiate through them.
#
# Batched by species: gather each element's AEV rows into one (aev_len, n_s) matrix
# and call Lux.apply once per element rather than once per atom. This cuts the number
# of Lux.apply calls Enzyme has to trace from n_atoms to n_species, which both speeds
# up the reverse pass and reduces allocation. `group_atoms[s]` lists the atom indices
# of species s; the gather/permutedims are differentiable so forces stay exact.
function _ani_energy_for_ad(
    coords_mat        :: AbstractMatrix{Float32},
    species_idx,
    boundary_strip,
    aev_params,
    n_species         :: Int,
    neighbors,
    group_atoms,
    models_by_species,
    ps_by_species,
    st_by_species,
    self_energies,
)
    n_atoms = size(coords_mat, 2)
    coords  = [SVector{3, Float32}(coords_mat[:, i]) for i in 1:n_atoms]
    aevs    = Molly.compute_aevs(coords, species_idx, neighbors, boundary_strip,
                                 aev_params, n_species)
    E = 0f0
    for s in 1:n_species
        atoms_s = group_atoms[s]
        ns = length(atoms_s)
        ns == 0 && continue
        batch  = permutedims(aevs[atoms_s, :])   # (aev_len, ns)
        out, _ = Lux.apply(models_by_species[s], batch, ps_by_species[s], st_by_species[s])
        E += sum(@view out[1, :]) + self_energies[s] * ns
    end
    return E
end

# ============================================================================
# Enzyme-based forces! (replaces finite-diff version from MollyLuxExt)
# ============================================================================

function AtomsCalculators.forces!(fs,
                                   sys::System{D, AT, T},
                                   inter::ANIPotential;
                                   kwargs...) where {D, AT, T}
    coords_f32  = _lue_f32coords(sys.coords)
    species_idx = [inter.species_map[ad.element] for ad in sys.atoms_data]
    nbrs        = get(kwargs, :neighbors, nothing)
    bdy         = _lue_strip_boundary(sys.boundary)
    bdy_f32     = CubicBoundary(Float32.(bdy.side_lengths))
    n_species   = length(inter.species_map)
    n_atoms     = length(coords_f32)

    idx_to_elem = Dict(v => k for (k, v) in inter.species_map)
    syms        = [Symbol(idx_to_elem[s]) for s in 1:n_species]
    # Atom indices grouped by species (Const for Enzyme — pure index data).
    group_atoms       = [findall(==(s), species_idx) for s in 1:n_species]
    models_by_species = [getfield(inter.model, syms[s]) for s in 1:n_species]

    coords_mat = reduce(hcat, coords_f32)
    n_ens = length(inter.ps_vec)

    # Pre-gather per-member (ps, st); independent per ensemble member.
    ps_all = [[getfield(inter.ps_vec[e], syms[s]) for s in 1:n_species] for e in 1:n_ens]
    st_all = [[getfield(inter.st_vec[e], syms[s]) for s in 1:n_species] for e in 1:n_ens]
    dcoords_all = [zeros(Float32, 3, n_atoms) for _ in 1:n_ens]

    # One reverse-mode pass per ensemble member. Each writes its own gradient buffer and
    # only reads the shared primal coords_mat + Const args, so the passes are independent.
    # `_ani_energy_for_ad` is compiled once (same types) — warm member 1 serially to avoid
    # a compilation race, then run the rest across threads.
    run_member!(e) = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        _ani_energy_for_ad,
        Enzyme.Active,
        Enzyme.Duplicated(coords_mat, dcoords_all[e]),
        Enzyme.Const(species_idx),
        Enzyme.Const(bdy_f32),
        Enzyme.Const(inter.aev_params),
        Enzyme.Const(n_species),
        Enzyme.Const(nbrs),
        Enzyme.Const(group_atoms),
        Enzyme.Const(models_by_species),
        Enzyme.Const(ps_all[e]),
        Enzyme.Const(st_all[e]),
        Enzyme.Const(inter.self_energies),
    )

    run_member!(1)
    if n_ens > 1
        Threads.@threads for e in 2:n_ens
            run_member!(e)
        end
    end

    dcoords_sum = dcoords_all[1]
    for e in 2:n_ens
        dcoords_sum .+= dcoords_all[e]
    end
    dcoords = dcoords_sum ./ n_ens

    for atom_i in 1:n_atoms
        fi = SVector{D, T}(-dcoords[:, atom_i]) * T(Molly.HARTREE_TO_EV)
        fs[atom_i] += Molly.ani_force_to_units(fi, sys.force_units, AT)
    end
    return fs
end

end # module MollyLuxEnzymeExt
