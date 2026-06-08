# ANI forces via two-stage reverse-mode AD.
# Loaded when Enzyme, Lux, and HDF5 are all in the environment.
# Overrides the finite-difference forces! from MollyLuxExt.
#
# Strategy (avoids LLVM symbol-collision bug in Enzyme ≤ 0.13):
#   Stage 1 — Zygote differentiates E wrt AEVs (Lux network only; no exp duplication).
#   Stage 2 — Enzyme differentiates AEVs wrt coords (pure math; no custom activations).
#   Chain rule combines the two: dE/d_coords = J_AEV^T · (dE/d_AEVs).
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
# ChainRule for Molly.celu01 — enables Zygote to differentiate through Lux layers
# ============================================================================

# d/dx celu(x; α=0.1) = 1 if x ≥ 0, else exp(x/0.1)
function ChainRulesCore.rrule(::typeof(Molly.celu01), x::T) where T
    y = Molly.celu01(x)
    function celu01_pullback(ȳ)
        dx = x >= zero(x) ? one(x) : exp(x / T(0.1))
        return NoTangent(), ȳ * dx
    end
    return y, celu01_pullback
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
# Enzyme-differentiable AEV kernel (top-level, no captured mutable state)
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
# Two-stage reverse-mode AD forces! (replaces finite-diff version from MollyLuxExt)
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

        # ── Stage 1: compute AEVs forward (plain Julia, no AD) ──────────────
        aevs = Molly.compute_aevs(coords_f32, species_idx, nbrs, bdy_f32,
                                  inter.aev_params, n_species)

        # ── Stage 2: Zygote — dE/d_AEVs (only Lux network, no exp collision) ─
        dE_d_aevs = Zygote.gradient(
            aevs_mat -> begin
                E = 0f0
                for atom_i in 1:n_atoms
                    aev_i = @view aevs_mat[atom_i, :]
                    out, _ = Lux.apply(per_atom_model[atom_i],
                                       reshape(aev_i, :, 1),
                                       per_atom_ps[atom_i],
                                       per_atom_st[atom_i])
                    E += Float32(out[1]) + inter.self_energies[species_idx[atom_i]]
                end
                E
            end,
            aevs,
        )[1]

        # ── Stage 3: Enzyme VJP — d_AEVs/d_coords via chain rule ───────────
        # Computes d(sum(aevs .* cotangent))/d_coords_mat = J_AEV^T · cotangent.
        # All captured data passed as explicit Const args so Enzyme can prove read-only.
        dcoords_i = zeros(Float32, 3, n_atoms)
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
