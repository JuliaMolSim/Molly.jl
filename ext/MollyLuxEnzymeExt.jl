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
# coords_mat: (3, N) Float32 — the only Active argument.
# All other args are Const so Enzyme doesn't need to differentiate through them.
function _ani_energy_for_ad(
    coords_mat    :: AbstractMatrix{Float32},
    species_idx,
    boundary_strip,
    aev_params,
    n_species     :: Int,
    neighbors,
    per_atom_model,
    per_atom_ps,
    per_atom_st,
    self_energies,
)
    n_atoms = size(coords_mat, 2)
    coords  = [SVector{3, Float32}(coords_mat[:, i]) for i in 1:n_atoms]
    aevs    = Molly.compute_aevs(coords, species_idx, neighbors, boundary_strip,
                                 aev_params, n_species)
    E = 0f0
    for atom_i in 1:n_atoms
        aev_i = @view aevs[atom_i, :]
        out, _ = Lux.apply(per_atom_model[atom_i],
                           Float32.(reshape(aev_i, :, 1)),
                           per_atom_ps[atom_i],
                           per_atom_st[atom_i])
        E += Float32(out[1]) + self_energies[species_idx[atom_i]]
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
    Ha_to_eV = T(27.211396132)

    coords_f32  = _lue_f32coords(sys.coords)
    species_idx = [inter.species_map[ad.element] for ad in sys.atoms_data]
    nbrs        = get(kwargs, :neighbors, nothing)
    bdy         = _lue_strip_boundary(sys.boundary)
    bdy_f32     = CubicBoundary(Float32.(bdy.side_lengths))
    n_species   = length(inter.species_map)
    n_atoms     = length(coords_f32)

    idx_to_elem    = Dict(v => k for (k, v) in inter.species_map)
    per_atom_model = [getfield(inter.model, Symbol(idx_to_elem[s])) for s in species_idx]

    coords_mat  = reduce(hcat, coords_f32)
    dcoords_sum = zeros(Float32, 3, n_atoms)
    n_ens = length(inter.ps_vec)

    for ens_i in 1:n_ens
        per_atom_ps = [getfield(inter.ps_vec[ens_i], Symbol(idx_to_elem[s])) for s in species_idx]
        per_atom_st = [getfield(inter.st_vec[ens_i], Symbol(idx_to_elem[s])) for s in species_idx]

        dcoords_i = zeros(Float32, 3, n_atoms)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            _ani_energy_for_ad,
            Enzyme.Active,
            Enzyme.Duplicated(coords_mat, dcoords_i),
            Enzyme.Const(species_idx),
            Enzyme.Const(bdy_f32),
            Enzyme.Const(inter.aev_params),
            Enzyme.Const(n_species),
            Enzyme.Const(nbrs),
            Enzyme.Const(per_atom_model),
            Enzyme.Const(per_atom_ps),
            Enzyme.Const(per_atom_st),
            Enzyme.Const(inter.self_energies),
        )

        dcoords_sum .+= dcoords_i
    end
    dcoords = dcoords_sum ./ n_ens

    for atom_i in 1:n_atoms
        fi = SVector{D, T}(-dcoords[:, atom_i]) * Ha_to_eV
        f_unit = if sys.force_units == NoUnits
            fi
        elseif dimension(sys.force_units) == u"𝐋 * 𝐌 * 𝐍^-1 * 𝐓^-2"
            uconvert.(sys.force_units, fi .* (Unitful.Na * u"eV/Å"))
        else
            uconvert.(sys.force_units, fi .* u"eV/Å")
        end
        fs[atom_i] += to_device(f_unit, AT)
    end
    return fs
end

end # module MollyLuxEnzymeExt
