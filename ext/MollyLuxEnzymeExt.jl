# ANI forces via reverse-mode AD (Enzyme + Zygote).
# Loaded when Enzyme, Lux, and HDF5 are all in the environment.
# Overrides the finite-difference forces! from MollyLuxExt.
#
# Strategy:
#   EnzymeRules for Molly.celu01 bypass the LLVM symbol-collision bug in Enzyme ≤ 0.13
#   that fired when exp() from the activation appeared multiple times in one trace.
#   With the rule in place, Enzyme can differentiate the full AEV + Lux pipeline in
#   a single backward pass via _ani_energy_for_ad.
#   Fallback: if single-pass Enzyme still fails for any system, the two-stage path
#   (Zygote for Lux, Enzyme for AEV VJP) remains available as _aev_vjp_kernel.
#
# __precompile__(false): method overwriting between extension modules is not
# allowed during precompilation; loading at runtime avoids this restriction.
__precompile__(false)

module MollyLuxEnzymeExt

using Molly
using Molly: from_device, to_device
import AtomsCalculators
using Enzyme, Lux, HDF5, Zygote
using ChainRulesCore
using StaticArrays, Unitful, LinearAlgebra

# ============================================================================
# AD rules for Molly.celu01
# ============================================================================

# ChainRule (Zygote path): d/dx celu(x; α=0.1) = 1 if x ≥ 0, else exp(x/0.1)
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

# Enzyme reverse: return analytical gradient, bypassing exp() symbol duplication.
function EnzymeRules.reverse(config,
                               ::EnzymeRules.Const{typeof(Molly.celu01)},
                               dret, tape, x::EnzymeRules.Active)
    xval = tape
    d = xval >= zero(xval) ? one(xval) : exp(xval / typeof(xval)(0.1f0))
    return (nothing, dret.val * d)  # (nothing for Const func; gradient for Active x)
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
# Single-pass Enzyme energy kernel (used when celu01 EnzymeRule is in effect)
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
# Two-stage AEV VJP kernel (fallback when single-pass Enzyme is unavailable)
# ============================================================================

# VJP kernel for Enzyme: computes sum(aevs(coords) .* cotangent).
# All non-differentiable data is passed as explicit arguments (no closures over
# mutable state) so Enzyme can prove read-only and avoid EnzymeMutabilityException.
function _aev_vjp_kernel(coords_mat, cotangent, species_idx, neighbors, boundary,
                          aev_params, n_species, n_atoms)
    cs       = [SVector{3, Float32}(coords_mat[:, i]) for i in 1:n_atoms]
    aevs_tmp = Molly.compute_aevs(cs, species_idx, neighbors, boundary, aev_params, n_species)
    return sum(aevs_tmp .* cotangent)
end

# ============================================================================
# Reverse-mode AD forces! (replaces finite-diff version from MollyLuxExt)
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

    # Pre-compute per-atom model/ps/st outside the AD scopes.
    idx_to_elem    = Dict(v => k for (k, v) in inter.species_map)
    per_atom_model = [getfield(inter.model, Symbol(idx_to_elem[s])) for s in species_idx]

    coords_mat  = reduce(hcat, coords_f32)   # (3, n_atoms) Float32
    dcoords_sum = zeros(Float32, 3, n_atoms)
    n_ens = length(inter.ps_vec)

    for ens_i in 1:n_ens
        per_atom_ps = [getfield(inter.ps_vec[ens_i], Symbol(idx_to_elem[s])) for s in species_idx]
        per_atom_st = [getfield(inter.st_vec[ens_i], Symbol(idx_to_elem[s])) for s in species_idx]

        dcoords_i = zeros(Float32, 3, n_atoms)

        # ── Single-pass Enzyme backward through AEVs + Lux ──────────────────
        # Works when the EnzymeRules for Molly.celu01 are in effect (this file).
        # Falls back to two-stage if Enzyme errors at runtime.
        try
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
        catch
            # ── Fallback: two-stage (Zygote for Lux, Enzyme VJP for AEVs) ───
            fill!(dcoords_i, 0f0)
            aevs = Molly.compute_aevs(coords_f32, species_idx, nbrs, bdy_f32,
                                      inter.aev_params, n_species)
            dE_d_aevs = Zygote.gradient(
                aevs_mat -> begin
                    E = 0f0
                    for atom_i in 1:n_atoms
                        aev_i = @view aevs_mat[atom_i, :]
                        out, _ = Lux.apply(per_atom_model[atom_i],
                                           Float32.(reshape(aev_i, :, 1)),
                                           per_atom_ps[atom_i],
                                           per_atom_st[atom_i])
                        E += Float32(out[1]) + inter.self_energies[species_idx[atom_i]]
                    end
                    E
                end,
                aevs,
            )[1]
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                _aev_vjp_kernel,
                Enzyme.Active,
                Enzyme.Duplicated(coords_mat, dcoords_i),
                Enzyme.Const(dE_d_aevs),
                Enzyme.Const(species_idx),
                Enzyme.Const(nbrs),
                Enzyme.Const(bdy_f32),
                Enzyme.Const(inter.aev_params),
                Enzyme.Const(n_species),
                Enzyme.Const(n_atoms),
            )
        end
        dcoords_sum .+= dcoords_i
    end
    dcoords = dcoords_sum ./ n_ens

    # F_i = -∂E/∂r_i (in Hartree/Å) converted to system force_units.
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
