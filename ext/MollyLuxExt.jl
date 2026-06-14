# ANI (and future ML potential) support via Lux.jl + HDF5.jl
# Loaded when both Lux and HDF5 are in the user environment.

module MollyLuxExt

using Molly
using Molly: from_device, to_device, vector
import AtomsCalculators
using Lux, HDF5
using StaticArrays, Unitful, Random, LinearAlgebra

# ============================================================================
# Shared infrastructure
# ============================================================================

# Strip units from a coordinate SVector array and return plain Float SVectors.
function strip_coords(coords::AbstractVector{SVector{D,T}}) where {D,T}
    unit(first(coords)[1]) == NoUnits ? collect(coords) :
        ustrip_vec.(u"Å", from_device(coords))   # returns SVector{D, Float64/32}
end

# Strip length units from a CubicBoundary so it is consistent with unit-stripped coords (Å).
function strip_boundary(b::CubicBoundary)
    unit(b.side_lengths[1]) == NoUnits ? b :
        CubicBoundary(ustrip.(u"Å", b.side_lengths))
end

# ============================================================================
# AEV computation  (pure Julia, Enzyme-differentiable in a later PR)
# ============================================================================

# Smooth cutoff: 0.5*(1+cos(π*r/r_c)) for r<r_c, else 0.
Molly.cosine_cutoff(r::T, r_c::T) where T =
    r < r_c ? T(0.5) * (one(T) + cos(T(π) * r / r_c)) : zero(T)

# Local alias for use within this module.
@inline cosine_cutoff(r, r_c) = Molly.cosine_cutoff(r, r_c)

# Replace the d-th component of SVector sv with value x.
@inline function _setcomp(sv::SVector{D,T}, d::Int, x::T) where {D,T}
    SVector{D,T}(ntuple(k -> k == d ? x : sv[k], Val(D)))
end

# Radial AEV sub-vector for atom i.
# Returns Vector of length n_species * n_η_R * n_ShfR.
# EtaR × ShfR are used as an outer product (all combinations), matching TorchANI.
function _radial_aev(coord_i::SVector{D,T},
                     nbr_coords, nbr_species,
                     η_R, r_s_R, r_c_R::T, n_species::Int) where {D,T}
    n_eta = length(η_R)
    n_shf = length(r_s_R)
    G = zeros(T, n_species * n_eta * n_shf)
    for (cj, sj) in zip(nbr_coords, nbr_species)
        dr = cj - coord_i   # displacement from i to j (already absolute for no-PBC path)
        r  = T(norm(dr))
        r >= r_c_R && continue
        fc   = cosine_cutoff(r, r_c_R)
        base = (sj - 1) * n_eta * n_shf
        for ki in 1:n_eta
            for kj in 1:n_shf
                @inbounds G[base + (ki-1)*n_shf + kj] +=
                    T(0.25) * exp(-η_R[ki] * (r - r_s_R[kj])^2) * fc
            end
        end
    end
    return G
end

# Angular AEV sub-vector for atom i.
# Returns Vector of length n_pairs * n_η_A * n_r_s_A * n_θ_s.
# EtaA × r_s_A × θ_s used as outer product, matching TorchANI.
function _angular_aev(coord_i::SVector{D,T},
                      nbr_coords, nbr_species,
                      η_A, r_s_A, θ_s, ζ::T, r_c_A::T,
                      n_species::Int) where {D,T}
    n_pairs = n_species * (n_species + 1) ÷ 2
    n_eta   = length(η_A)
    n_shf_r = length(r_s_A)
    n_th    = length(θ_s)
    G = zeros(T, n_pairs * n_eta * n_shf_r * n_th)
    n_nbr = length(nbr_coords)

    # Cache per-neighbor data within angular cutoff.
    rj   = Vector{T}(undef, n_nbr)
    fcj  = Vector{T}(undef, n_nbr)
    drj  = Vector{SVector{D,T}}(undef, n_nbr)
    ok   = Vector{Bool}(undef, n_nbr)
    for idx in 1:n_nbr
        d  = nbr_coords[idx] - coord_i
        r  = T(norm(d))
        ok[idx]  = r < r_c_A
        rj[idx]  = r
        fcj[idx] = ok[idx] ? cosine_cutoff(r, r_c_A) : zero(T)
        drj[idx] = d
    end

    prefac0 = T(2)^(one(T) - ζ)   # TorchANI angular prefactor: 2^(1-ζ)

    for j in 1:n_nbr
        ok[j] || continue
        sj = nbr_species[j]
        for k in (j+1):n_nbr
            ok[k] || continue
            sk = nbr_species[k]

            # Upper-triangular species pair index (1-based).
            s1, s2 = sj <= sk ? (sj, sk) : (sk, sj)
            pair_idx = (s1 - 1) * n_species - (s1 - 1) * (s1 - 2) ÷ 2 + (s2 - s1 + 1)

            r_avg  = (rj[j] + rj[k]) * T(0.5)
            fc_jk  = fcj[j] * fcj[k]
            cos_th = clamp(dot(drj[j], drj[k]) / (rj[j] * rj[k]), T(-1), T(1))
            theta  = acos(T(0.95) * cos_th)   # 0.95 scaling matches TorchANI to avoid NaN near ±1

            base = (pair_idx - 1) * n_eta * n_shf_r * n_th
            for p in 1:n_eta
                for q in 1:n_shf_r
                    r_factor = prefac0 * fc_jk * exp(-η_A[p] * (r_avg - r_s_A[q])^2)
                    for l in 1:n_th
                        ang = (one(T) + cos(theta - θ_s[l]))^ζ
                        @inbounds G[base + ((p-1)*n_shf_r + (q-1))*n_th + l] += r_factor * ang
                    end
                end
            end
        end
    end
    return G
end

# Compute AEVs for all atoms.
# Returns Matrix{T} of shape (n_atoms, aev_length).
# Pass neighbors=nothing to use an O(N²) all-pairs loop (useful when no NeighborFinder
# is configured, e.g. for small test systems).
function Molly.compute_aevs(coords::AbstractVector{SVector{D,T}},
                      species_indices::AbstractVector{<:Integer},
                      neighbors,
                      boundary,
                      p,          # aev_params NamedTuple
                      n_species::Int) where {D,T}
    n_atoms  = length(coords)
    r_c_R    = T(p.r_c_R)
    r_c_A    = T(p.r_c_A)
    r_c_max  = max(r_c_R, r_c_A)

    # Build per-atom AEV row; use reduce(vcat, ...) so no mutation — required for Zygote.
    rows = map(1:n_atoms) do atom_i
        nbr_coords  = SVector{D,T}[]
        nbr_species = Int[]

        if isnothing(neighbors)
            for j in 1:n_atoms
                j == atom_i && continue
                dr = coords[j] - coords[atom_i]
                norm(dr) < r_c_max || continue
                push!(nbr_coords,  coords[j])
                push!(nbr_species, Int(species_indices[j]))
            end
        else
            for ni in eachindex(neighbors)
                idx_i, idx_j, _ = neighbors[ni]
                nbr_idx = (Int(idx_i) == atom_i) ? Int(idx_j) :
                          (Int(idx_j) == atom_i) ? Int(idx_i) : 0
                nbr_idx == 0 && continue
                dr = vector(coords[atom_i], coords[nbr_idx], boundary)
                push!(nbr_coords,  coords[atom_i] + dr)
                push!(nbr_species, Int(species_indices[nbr_idx]))
            end
        end

        G_rad = _radial_aev(coords[atom_i], nbr_coords, nbr_species,
                            p.η_R, p.r_s_R, r_c_R, n_species)
        G_ang = _angular_aev(coords[atom_i], nbr_coords, nbr_species,
                             p.η_A, p.r_s_A, p.θ_s, T(p.ζ), r_c_A, n_species)
        vcat(G_rad, G_ang)'   # (1, aev_len) row
    end
    return reduce(vcat, rows)  # (n_atoms, aev_len)
end

# ============================================================================
# HDF5 weight loader
# ============================================================================

# Use the public Molly.celu01 so AD backends (Enzyme, Zygote) can register rules.
const _celu01 = Molly.celu01

# Build one element's Lux.Chain from the HDF5 group at /ensemble_idx/element/.
# Dense layer indices 0, 2, 4, ... in the group; activations implicit between them.
function _build_element_model(elem_grp::HDF5.Group)
    layer_ids = sort(parse.(Int, keys(elem_grp)))   # [0, 2, 4, 6]
    n_dense   = length(layer_ids)
    lux_layers = []
    ps_pairs   = Pair{Symbol,NamedTuple}[]

    for (k, li) in enumerate(layer_ids)
        lg  = elem_grp[string(li)]
        # HDF5.jl reads arrays transposed relative to Python (column-major vs row-major).
        # Python stored weight as (out, in); Julia reads it as (in, out) → permute back.
        W   = permutedims(Float32.(read(lg["weight"])))   # (out, in)
        b   = Float32.(read(lg["bias"]))                  # (out,)
        out_sz, in_sz = size(W)
        act = k < n_dense ? _celu01 : identity
        push!(lux_layers, Lux.Dense(in_sz => out_sz, act))
        push!(ps_pairs, Symbol("layer_", k) => (weight = W, bias = b))
    end

    model = Lux.Chain(lux_layers...)
    ps    = NamedTuple(ps_pairs)
    _, st = Lux.setup(Xoshiro(0), model)
    return model, ps, st
end

# Read an HDF5 dataset as a Vector{T}, handling scalar datasets gracefully.
_h5vec(grp, name, T) = T.(vcat(read(grp[name])))

function Molly.ANIPotential(path::String;
                           force_units  = u"eV/Å",
                           energy_units = u"eV",
                           T            = Float32,
                           ensemble_idx = nothing)
    h5open(path, "r") do h5
        # AEV hyperparameters
        ag    = h5["aev_params"]
        r_c_R = T(only(read(ag["Rcr"])))
        r_c_A = T(only(read(ag["Rca"])))
        η_R   = _h5vec(ag, "EtaR", T)
        r_s_R = _h5vec(ag, "ShfR", T)
        η_A   = _h5vec(ag, "EtaA", T)
        # TorchANI naming is counterintuitive: ShfA holds *radial* shifts (Å) and
        # ShfZ holds *angular* shifts (rad) for the angular AEV Gaussian.
        r_s_A = _h5vec(ag, "ShfA", T)   # ShfA = radial shifts for angular Gaussian (Å)
        θ_s   = _h5vec(ag, "ShfZ", T)   # ShfZ = angular shifts θ_s (rad)
        ζ     = T(only(read(ag["Zeta"])))
        species_list = String.(read(ag["species"]))

        aev_params = (η_R=η_R, r_s_R=r_s_R, r_c_R=r_c_R,
                      η_A=η_A, r_s_A=r_s_A, θ_s=θ_s, ζ=ζ, r_c_A=r_c_A)

        species_map   = Dict{String,Int}(s => i for (i, s) in enumerate(species_list))
        self_energies = _h5vec(ag, "self_energies", T)   # (n_species,) Hartree

        # Determine which ensemble members to load.
        # All integer-keyed top-level groups are ensemble members.
        ens_indices = if isnothing(ensemble_idx)
            sort([parse(Int, k) for k in keys(h5) if tryparse(Int, k) !== nothing])
        else
            [ensemble_idx]
        end

        # Build shared model architecture from the first ensemble member.
        syms = Tuple(Symbol.(species_list))
        first_grp = h5[string(first(ens_indices))]
        models_v  = []
        for elem in species_list
            m, _, _ = _build_element_model(first_grp[elem])
            push!(models_v, m)
        end
        model_nt = NamedTuple{syms}(Tuple(models_v))

        # Load parameters and states for each ensemble member.
        ps_list = NamedTuple[]
        st_list = NamedTuple[]
        for idx in ens_indices
            ens_grp = h5[string(idx)]
            ps_v = []; st_v = []
            for elem in species_list
                _, ps, st = _build_element_model(ens_grp[elem])
                push!(ps_v, ps); push!(st_v, st)
            end
            push!(ps_list, NamedTuple{syms}(Tuple(ps_v)))
            push!(st_list, NamedTuple{syms}(Tuple(st_v)))
        end

        cutoff = T(max(r_c_R, r_c_A))
        return ANIPotential(model_nt, ps_list, st_list, species_map,
                            aev_params, self_energies, cutoff, force_units, energy_units)
    end
end

# ============================================================================
# AtomsCalculators interface
# ============================================================================

# Internal: energy (Hartree) for one ensemble member given pre-computed AEVs.
function _ani_energy_single(aevs, species_idx, idx_to_elem, model, self_energies,
                            ps, st, ::Type{T}) where T
    E = zero(T)
    for atom_i in eachindex(species_idx)
        s    = species_idx[atom_i]
        sym  = Symbol(idx_to_elem[s])
        aev_i = @view aevs[atom_i, :]
        out, _ = Lux.apply(getfield(model, sym),
                            Float32.(reshape(aev_i, :, 1)),
                            getfield(ps, sym),
                            getfield(st, sym))
        E += T(out[1]) + self_energies[s]
    end
    return E
end

# Internal: compute raw energy (Hartree) averaged over all ensemble members.
function _ani_raw_energy(coords_strip::AbstractVector{SVector{D,T}},
                         species_idx::AbstractVector{Int},
                         boundary, inter::ANIPotential,
                         neighbors) where {D,T}
    n_species = length(inter.species_map)
    # coords_strip is unit-free (Å); boundary must also be unit-free for vector() to work.
    aevs = Molly.compute_aevs(coords_strip, species_idx, neighbors, strip_boundary(boundary),
                               inter.aev_params, n_species)
    idx_to_elem = Dict(v => k for (k, v) in inter.species_map)
    n_ens = length(inter.ps_vec)
    E = zero(T)
    for i in 1:n_ens
        E += _ani_energy_single(aevs, species_idx, idx_to_elem, inter.model,
                                inter.self_energies, inter.ps_vec[i], inter.st_vec[i], T)
    end
    return E / n_ens
end

function AtomsCalculators.potential_energy(sys::System{D, AT, T},
                                           inter::ANIPotential;
                                           kwargs...) where {D, AT, T}
    coords_strip = strip_coords(sys.coords)
    species_idx  = [inter.species_map[ad.element] for ad in sys.atoms_data]
    nbrs = get(kwargs, :neighbors, nothing)

    E_ha = _ani_raw_energy(coords_strip, species_idx, sys.boundary, inter, nbrs)

    Ha_to_eV = T(27.211396132)
    E_eV = E_ha * Ha_to_eV

    if sys.energy_units == NoUnits
        return E_eV
    elseif dimension(sys.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return uconvert(sys.energy_units, E_eV * Unitful.Na * u"eV")
    else
        return uconvert(sys.energy_units, E_eV * u"eV")
    end
end

# Forces via finite differences (fallback when Enzyme is not loaded).
function AtomsCalculators.forces!(fs,
                                  sys::System{D, AT, T},
                                  inter::ANIPotential;
                                  kwargs...) where {D, AT, T}
    h = T(1e-4)   # Å — must be T (Float64) for numerical precision
    Ha_to_eV = T(27.211396132)

    coords_strip = strip_coords(sys.coords)
    species_idx  = [inter.species_map[ad.element] for ad in sys.atoms_data]
    nbrs = get(kwargs, :neighbors, nothing)

    for atom_i in eachindex(coords_strip)
        fi = zero(SVector{D, T})
        for dim in 1:D
            cp = copy(coords_strip)
            cm = copy(coords_strip)
            cp[atom_i] = _setcomp(cp[atom_i], dim, cp[atom_i][dim] + h)
            cm[atom_i] = _setcomp(cm[atom_i], dim, cm[atom_i][dim] - h)
            Ep = _ani_raw_energy(cp, species_idx, sys.boundary, inter, nbrs)
            Em = _ani_raw_energy(cm, species_idx, sys.boundary, inter, nbrs)
            fi = _setcomp(fi, dim, T(-(Ep - Em) / (2h) * Ha_to_eV))
        end

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

end # module MollyLuxExt
