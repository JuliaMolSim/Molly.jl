export
    LennardJones,
    LJDispersionCorrection,
    LennardJonesSoftCoreBeutler,
    LennardJonesSoftCoreGapsys,
    AshbaughHatch,
    SIMDLennardJones,
    SIMDNeighborFinder,
    SIMDCoulomb,
    PackedFlatSoA,
    SIMDCoulombReactionField,
    SIMDCoulombEwald,
    ClusteredSIMDNeighborFinder,
    ClusteredNeighborList,
    ClusterPairSoA,
    cluster_diagnostics,
    cluster_csr_pair_count,
    check_cluster_csr_consistency

@doc raw"""
    LennardJones(; cutoff, use_neighbors, shortcut, σ_mixing, ϵ_mixing, weight_special)

The Lennard-Jones 6-12 interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]
```
and the force on each atom by
```math
\begin{aligned}
\vec{F}_i &= 24\varepsilon_{ij} \left(2\frac{\sigma_{ij}^{12}}{r_{ij}^{13}} - \frac{\sigma_{ij}^6}{r_{ij}^{7}}\right) \frac{\vec{r}_{ij}}{r_{ij}} \\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \vec{r}_{ij}
\end{aligned}
```

Should be used alongside the [`LJDispersionCorrection`](@ref) general interaction
when the long-range correction to the potential energy is required.
"""
@kwdef struct LennardJones{C, H, S, E, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    weight_special::W = 1
end

use_neighbors(inter::LennardJones) = inter.use_neighbors

function Base.zero(lj::LennardJones{C, H, S, E, W}) where {C, H, S, E, W}
    return LennardJones(
        lj.cutoff,
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        zero(W),
    )
end

function Base.:+(l1::LennardJones, l2::LennardJones)
    return LennardJones(
        l1.cutoff,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.weight_special + l2.weight_special,
    )
end

function inject_interaction(inter::LennardJones, params_dic)
    key_prefix = "inter_LJ_"
    return LennardJones(
        inter.cutoff,
        inter.use_neighbors,
        inter.shortcut,
        inter.σ_mixing,
        inter.ϵ_mixing,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
    )
end

function extract_parameters!(params_dic, inter::LennardJones, ff)
    key_prefix = "inter_LJ_"
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    return params_dic
end

@inline function force(inter::LennardJones,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip.(zero(dr)) * force_units
    end
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::LennardJones, r, (σ2, ϵ))
    six_term = (σ2 / r^2) ^ 3
    return (24ϵ / r) * (2 * six_term ^ 2 - six_term)
end

@inline function potential_energy(inter::LennardJones,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip(zero(dr[1])) * energy_units
    end
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::LennardJones, r, (σ2, ϵ))
    six_term = (σ2 / r^2) ^ 3
    return 4ϵ * (six_term ^ 2 - six_term)
end

@doc raw"""
    LJDispersionCorrection(atoms, dist_cutoff, σ_mixing=LorentzMixing(),
                           ϵ_mixing=GeometricMixing())

The long-range dispersion correction for the [`LennardJones`](@ref) interaction.

Approximately represents contributions from beyond the cutoff distance.
Should be used alongside the [`LennardJones`](@ref) pairwise interaction when the long-range
correction to the potential energy is required.
The potential energy is defined as
```math
E = \frac{8 \pi N^2}{V} \left( \frac{\left< \epsilon_{ij} \sigma_{ij}^{12} \right>}{9 r_c^9} - \frac{\left< \epsilon_{ij} \sigma_{ij}^{6} \right>}{3 r_c^3} \right)
```
The forces are zero.

The number of atoms and atom σ and ϵ values are assumed not to change after setup (the box
volume can change).
Only compatible with 3D systems.
Not compatible with cutoffs other than [`DistanceCutoff`](@ref).
"""
struct LJDispersionCorrection{F}
    factor::F
end

function LJDispersionCorrection(atoms, dist_cutoff, σ_mix=LorentzMixing(),
                                ϵ_mix=GeometricMixing())
    T = typeof(ustrip(dist_cutoff))
    n_atoms = length(atoms)
    atoms_cpu = from_device(atoms)
    at = atoms_cpu[1]
    ϵσ12_sum, ϵσ6_sum = zero(at.ϵ * at.σ^12), zero(at.ϵ * at.σ^6)
    for i in 1:n_atoms
        atom_i = atoms_cpu[i]
        for j in 1:i
            atom_j = atoms_cpu[j]
            σ = σ_mixing(σ_mix, atom_i, atom_j, false)
            ϵ = ϵ_mixing(ϵ_mix, atom_i, atom_j, false)
            ϵσ12_sum += ϵ * σ^12
            ϵσ6_sum  += ϵ * σ^6
        end
    end
    n_pairs = (n_atoms * (n_atoms + 1)) ÷ 2
    ϵσ12_mean = ϵσ12_sum / n_pairs
    ϵσ6_mean  = ϵσ6_sum  / n_pairs
    inner_term = (ϵσ12_mean / (9 * dist_cutoff^9) - ϵσ6_mean / (3 * dist_cutoff^3))
    factor = 8 * T(π) * n_atoms^2 * inner_term
    return LJDispersionCorrection(factor)
end

Base.zero(dc::LJDispersionCorrection) = LJDispersionCorrection(zero(dc.factor))

function Base.:+(dc1::LJDispersionCorrection, dc2::LJDispersionCorrection)
    return LJDispersionCorrection(dc1.factor + dc2.factor)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                        inter::LJDispersionCorrection; kwargs...)
    return inter.factor / volume(sys)
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs, sys,
                                                        inter::LJDispersionCorrection; kwargs...)
    return fs
end

@doc raw"""
    LennardJonesSoftCoreBeutler(; cutoff, α, λ, use_neighbors, shortcut, σ_mixing,
                                ϵ_mixing, weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1).
The potential energy is defined as
```math
V(r_{ij}) = \lambda \left(\frac{C^{(12)}}{r_{LJ}^{12}} - \frac{C^{(6)}}{r_{LJ}^{6}}\right)
```
and the force on each atom by
```math
\vec{F}_i = \lambda \left(\left(\frac{12C^{(12)}}{r_{LJ}^{13}} - \frac{6C^{(6)}}{r_{LJ}^7}\right)\left(\frac{r_{ij}}{r_{LJ}}\right)^5\right) \frac{\vec{r_{ij}}}{r_{ij}}
```
where
```math
r_{LJ} = \left(\frac{\alpha(1-\lambda)C^{(12)}}{C^{(6)}}+r^6\right)^{1/6}
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```

If ``\lambda`` is 1.0, this gives the standard [`LennardJones`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct LennardJonesSoftCoreBeutler{C, A, H, S, E, LM, SCH, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1.0
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreBeutler) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreBeutler{C, A, H, S, E, LM, SCH, W}) where {C, A, H, S, E, LM, SCH, W}
    return LennardJonesSoftCoreBeutler(
        lj.cutoff,
        zero(A),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        lj.scheduler,
        zero(W),
    )
end

function Base.:+(l1::LennardJonesSoftCoreBeutler, l2::LennardJonesSoftCoreBeutler)
    return LennardJonesSoftCoreBeutler(
        l1.cutoff,
        l1.α + l2.α,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.λ_mixing,
        l1.scheduler,
        l1.weight_special + l2.weight_special
    )
end

@inline function force(inter::LennardJonesSoftCoreBeutler,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    # Mix Lambda
    T = typeof(ustrip(atom_i.σ))
    λ_glob = T(λ_mixing(inter.λ_mixing, atom_i, atom_j))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, role_i, role_j)

    # 2. Dispatch to the scheduler for the effective sterics lambda
    λ = T(scale_sterics(inter.scheduler, λ_glob, pair_role))

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return zero_pairwise_force(dr, force_units)
    end

    r = norm(dr)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    # If lambda is 1, the soft core formula reduces to standard LJ
    # We explicity branch to save compute.
    if λ >= 1

        σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
        ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
        σ2 = σ^2
        params = (σ2, ϵ, nothing, nothing)
        
        # Call standard LJ cutoff logic.
        f = force_cutoff(inter.cutoff, inter, r, params) 
        fdr = radial_force_vector(f, r, dr, force_units)
        
        return special ? fdr * inter.weight_special : fdr
    end

    
    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j)^6
    ϵ  = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)

    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    σ6_fac = inter.α * (1 - λ)
    params = (C12, C6, λ, σ6_fac)

    f = force_cutoff(inter.cutoff, inter, r, params)
    fdr = radial_force_vector(f, r, dr, force_units)
    
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard LJ Logic
@inline function pairwise_force(::LennardJonesSoftCoreBeutler, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    six_term = (σ2 / r^2)^3
    return (24 * ϵ / r) * (2 * six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic
function pairwise_force(::LennardJonesSoftCoreBeutler, r, (C12, C6, λ, σ6_fac)::Tuple{Any, Any, Any, Any})
    R = sqrt(cbrt((σ6_fac*(C12/C6))+r^6))
    R6 = R^6
    return λ*(((12*C12)/(R6*R6*R)) - ((6*C6)/(R6*R)))*((r/R)^5)
end

@inline function potential_energy(inter::LennardJonesSoftCoreBeutler,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    # Mix Lambda
    T = typeof(ustrip(atom_i.σ))
    λ_glob = T(λ_mixing(inter.λ_mixing, atom_i, atom_j))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, role_i, role_j)

    # 2. Dispatch to the scheduler for the effective sterics lambda
    λ = T(scale_sterics(inter.scheduler, λ_glob, pair_role))

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end


    # If lambda is 1, the soft core formula reduces to standard LJ
    # We explicity branch to save compute.
    if λ >= 1

        σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
        ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)

        r = norm(dr)
        σ2 = σ^2
        params = (σ2, ϵ, nothing, nothing)

        pe = pe_cutoff(inter.cutoff, inter, r, params)
        
        if special
            return pe * inter.weight_special
        else
            return pe
        end
    end

    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j)^6
    ϵ  = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)

    cutoff = inter.cutoff
    r = norm(dr)
    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    σ6_fac = inter.α * (1 - λ)
    params = (C12, C6, σ6_fac, λ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

# Dispatch 1: Standard LJ Logic
@inline function pairwise_pe(::LennardJonesSoftCoreBeutler, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    inv_r2 = inv(r^2)
    six_term = (σ2 * inv_r2)^3
    return 4 * ϵ * (six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
function pairwise_pe(::LennardJonesSoftCoreBeutler, r, (C12, C6, σ6_fac, λ)::Tuple{Any, Any, Any, Any})
    R6 = (σ6_fac * (C12 / C6)) + r^6
    return λ * ((C12 / (R6 * R6)) - (C6 / R6))
end

@doc raw"""
    LennardJonesSoftCoreGapsys(; cutoff, α, λ, use_neighbors, shortcut, σ_mixing,
                               ϵ_mixing, weight_special)

The Lennard-Jones 6-12 interaction between two atoms with a soft core potential, used for
the appearing and disappearing of atoms.

See [Gapsys et al. 2012](https://doi.org/10.1021/ct300220p).
The potential energy is defined as
```math
V(r_{ij}) = \left\{ \begin{array}{cl}
\lambda \left( \frac{C^{(12)}}{r_{ij}^{12}} - \frac{C^{(6)}}{r_{ij}^{6}} \right), & \text{if} & r \ge r_{LJ} \\
\lambda \left( (\frac{78C^{(12)}}{r_{LJ}^{14}}-\frac{21C^{(6)}}{r_{LJ}^{8}})r_{ij}^2 - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}})r_{ij} + \frac{91C^{(12)}}{r_{LJ}^{12}}-\frac{28C^{(6)}}{r_{LJ}^{6}} \right), & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
\lambda \left( \frac{12C^{(12)}}{r_{ij}^{13}} - \frac{6C^{(6)}}{r_{ij}^{7}} \right)\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\lambda \left( (\frac{-156C^{(12)}}{r_{LJ}^{14}}+\frac{42C^{(6)}}{r_{LJ}^{8}})r_{ij} - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}}) \right)\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
where
```math
r_{LJ} = \alpha \left( \frac{26C^{(12)}(1-\lambda)}{7C^{(6)}} \right)^{\frac{1}{6}}
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```

If ``\lambda`` is 1.0, this gives the standard [`LennardJones`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct LennardJonesSoftCoreGapsys{C, A, H, S, E, LM, SCH, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 0.85
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreGapsys) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreGapsys{C, A, H, S, E, LM, SCH, W}) where {C, A, H, S, E, LM, SCH, W}
    return LennardJonesSoftCoreGapsys(
        lj.cutoff,
        zero(A),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        lj.scheduler,
        zero(W),
    )
end

function Base.:+(l1::LennardJonesSoftCoreGapsys, l2::LennardJonesSoftCoreGapsys)
    return LennardJonesSoftCoreGapsys(
        l1.cutoff,
        l1.α + l2.α,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.λ_mixing,
        l1.scheduler,
        l1.weight_special + l2.weight_special,
    )
end

@inline function force(inter::LennardJonesSoftCoreGapsys,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)

    T = typeof(ustrip(atom_i.σ))
    λ_glob = T(λ_mixing(inter.λ_mixing, atom_i, atom_j))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, role_i, role_j)

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ = T(scale_sterics(inter.scheduler, λ_glob, pair_role))

    if λ <= 0
        return zero_pairwise_force(dr, force_units)
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return zero_pairwise_force(dr, force_units)
    end

    cutoff = inter.cutoff
    r = norm(dr)
    if iszero_value(r)
        return zero_pairwise_force(dr, force_units)
    end

    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    # 3. Fast Path: Standard Lennard Jones
    if λ >= 1
        # Pass standard LJ params tuple (Length 2)
        params = (σ^2, ϵ, nothing, nothing)
        f = force_cutoff(cutoff, inter, r, params)
        fdr = radial_force_vector(f, r, dr, force_units)
        return special ? fdr * inter.weight_special : fdr
    end

    # 4. Alchemical Path: Soft Core Gapsys
    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    val = (26 * σ6 * (1 - λ)) / 7
    R = inter.α * sqrt(cbrt(val))

    # Pass SoftCore params tuple (Length 4)
    params = (C12, C6, λ, R)
    f = force_cutoff(cutoff, inter, r, params)
    fdr = radial_force_vector(f, r, dr, force_units)
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard LJ Logic (Matches Tuple length 2)
@inline function pairwise_force(::LennardJonesSoftCoreGapsys, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    six_term = (σ2 / r^2)^3
    return (24 * ϵ / r) * (2 * six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
@inline function pairwise_force(::LennardJonesSoftCoreGapsys, r, (C12, C6, λ, R)::Tuple{Any, Any, Any, Any})
    r6 = r^6
    if r >= R
        return λ * (((12*C12)/(r6*r6*r)) - ((6*C6)/(r6*r)))
    else
        invR = inv(R)
        invR2 = invR^2
        invR6 = invR^6
        return λ * (((-156*C12*(invR6*invR6*invR2)) + (42*C6*(invR2*invR6)))*r +
                    (168*C12*(invR6*invR6*invR)) - (48*C6*(invR6*invR)))
    end
end

@inline function potential_energy(inter::LennardJonesSoftCoreGapsys,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    T = typeof(ustrip(atom_i.σ))
    λ_glob = T(λ_mixing(inter.λ_mixing, atom_i, atom_j))

    # 1. Fetch alchemical roles from the contiguous array
    role_i = atom_i.alch_role
    role_j = atom_j.alch_role
    pair_role = mix_roles(inter.scheduler, role_i, role_j)

    # 2. Dispatch to the scheduler for the effective sterics lambda
    # Changed scale_elec to scale_sterics
    λ = T(scale_sterics(inter.scheduler, λ_glob, pair_role))

    if λ <= 0
        return ustrip(zero(dr[1])) * energy_units
    end

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end

    cutoff = inter.cutoff
    r = norm(dr)
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    # 3. Fast Path: Standard Lennard Jones
    if λ >= 1
        # Pass standard LJ params tuple (Length 2)
        params = (σ^2, ϵ, nothing, nothing)
        pe = pe_cutoff(cutoff, inter, r, params)
        return special ? pe * inter.weight_special : pe
    end

    # 4. Alchemical Path: Soft Core Gapsys
    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    val = (26 * σ6 * (1 - λ)) / 7
    R = inter.α * sqrt(cbrt(val))

    # Pass SoftCore params tuple (Length 4)
    params = (C12, C6, λ, R)
    pe = pe_cutoff(cutoff, inter, r, params)
    return special ? pe * inter.weight_special : pe
end

# Dispatch 1: Standard LJ Logic (Matches Tuple length 2)
@inline function pairwise_pe(::LennardJonesSoftCoreGapsys, r, (σ2, ϵ, _, _)::Tuple{Any, Any, Nothing, Nothing})
    inv_r2 = inv(r^2)
    six_term = (σ2 * inv_r2)^3
    return 4 * ϵ * (six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
@inline function pairwise_pe(::LennardJonesSoftCoreGapsys, r, (C12, C6, λ, R)::Tuple{Any, Any, Any, Any})
    r6 = r^6
    if r >= R
        return λ * ((C12/(r6*r6)) - (C6/(r6)))
    else
        invR = inv(R)
        invR2 = invR^2
        invR6 = invR^6
        return λ * ((78*C12*(invR6*invR6*invR2)) - (21*C6*(invR2*invR6)))*(r^2) -
                   ((168*C12*(invR6*invR6*invR)) - (48*C6*(invR6*invR)))*r +
                   (91*C12*(invR6*invR6)) - (28*C6*(invR6))
    end
end

@doc raw"""
    AshbaughHatch(; cutoff, use_neighbors, shortcut, ϵ_mixing, σ_mixing,
                  λ_mixing, weight_special)

The Ashbaugh-Hatch potential ($V_{\text{AH}}$) is a modified Lennard-Jones ($V_{\text{LJ}}$)
6-12 interaction between two atoms.

The potential energy is defined as
```math
V_{\text{LJ}}(r_{ij}) = 4\varepsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \\
```
```math
V_{\text{AH}}(r_{ij}) =
    \begin{cases}
      V_{\text{LJ}}(r_{ij}) +\varepsilon_{ij}(1-λ_{ij}) &,  r_{ij}\leq  2^{1/6}σ  \\
       λ_{ij}V_{\text{LJ}}(r_{ij})  &,  2^{1/6}σ \leq r_{ij}
    \end{cases}
```
and the force on each atom by
```math
\vec{F}_{\text{AH}} =
    \begin{cases}
      F_{\text{LJ}}(r_{ij})  &,  r_{ij} \leq  2^{1/6}σ  \\
       λ_{ij}F_{\text{LJ}}(r_{ij})  &,  2^{1/6}σ \leq r_{ij}
    \end{cases}
```
where
```math
\begin{aligned}
\vec{F}_{\text{LJ}}\
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]  \vec{r_{ij}}
\end{aligned}
```

If ``\lambda`` is one this gives the standard [`LennardJones`](@ref) potential.
"""
@kwdef struct AshbaughHatch{C, H, S, E, L, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = LorentzMixing()
    λ_mixing::L = LorentzMixing()
    weight_special::W = 1
end

use_neighbors(inter::AshbaughHatch) = inter.use_neighbors

function Base.zero(lj::AshbaughHatch{C, H, S, E, L, W}) where {C, H, S, E, L, W}
    return AshbaughHatch(
        lj.cutoff,
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        zero(W),
    )
end

function Base.:+(l1::AshbaughHatch, l2::AshbaughHatch)
    return AshbaughHatch(
        l1.cutoff,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
        l1.λ_mixing,
        l1.weight_special + l2.weight_special,
    )
end

@kwdef struct AshbaughHatchAtom{T, M, C, S, E, L}
    index::Int = 1
    atom_type::T = 1
    mass::M = 1.0u"g/mol"
    charge::C = 0.0
    σ::S = 0.0u"nm"
    ϵ::E = 0.0u"kJ * mol^-1"
    λ::L = 1.0
end

@inline function force(inter::AshbaughHatch,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special::Bool=false,
                       args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip.(zero(dr)) * force_units
    end

    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    λ = λ_mixing(inter.λ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ, λ)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

@inline function pairwise_force(::AshbaughHatch, r, (σ2, ϵ, λ))
    r2 = r^2
    six_term = (σ2 / r2) ^ 3
    lj_term = (24ϵ / r) * (2 * six_term ^ 2 - six_term)
    if r2 < (2^(1/3) * σ2)
        return lj_term
    else
        return λ * lj_term
    end
end

@inline function potential_energy(inter::AshbaughHatch,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special::Bool=false,
                                  args...)
    if shortcut_pair(inter.shortcut, atom_i, atom_j, special)
        return ustrip(zero(dr[1])) * energy_units
    end
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j, special)
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j, special)
    λ = λ_mixing(inter.λ_mixing, atom_i, atom_j, special)

    cutoff = inter.cutoff
    r = norm(dr)
    σ2 = σ^2
    params = (σ2, ϵ, λ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

@inline function pairwise_pe(::AshbaughHatch, r, (σ2, ϵ, λ))
    r2 = r^2
    six_term = (σ2 / r2) ^ 3
    lj_term = 4ϵ * (six_term ^ 2 - six_term)
    if r2 < (2^(1/3) * σ2)
        return lj_term + ϵ * (1 - λ)
    else
        return λ * lj_term
    end
end

# Specific interaction used to allow different σ/ϵ for 1-4 interactions
# Assumes no 1-4 Lennard-Jones interaction via the pairwise interactions (weight_special = 0)
struct LennardJones14{S, E, W}
    σ14_mixed::S
    ϵ14_mixed::E
    weight_14::W
end

function Base.zero(::LennardJones14{S, E, W}) where {S, E, W}
    return LennardJones14(zero(S), zero(E), zero(W))
end

function Base.:+(l1::LennardJones14, l2::LennardJones14)
    return LennardJones14(
        l1.σ14_mixed + l2.σ14_mixed,
        l1.ϵ14_mixed + l2.ϵ14_mixed,
        l1.weight_14 + l2.weight_14,
    )
end

@inline function force(inter::LennardJones14, coords_i, coords_l, boundary, args...)
    σ2 = inter.σ14_mixed ^ 2
    dr = vector(coords_i, coords_l, boundary)
    r2 = sum(abs2, dr)
    six_term = (σ2 / r2) ^ 3
    fl = inter.weight_14 * (24 * inter.ϵ14_mixed / r2) * (2 * six_term ^ 2 - six_term) * dr
    fi = -fl
    return SpecificForce2Atoms(fi, fl)
end

@inline function potential_energy(inter::LennardJones14, coords_i, coords_l, boundary, args...)
    σ2 = inter.σ14_mixed ^ 2
    r2 = sum(abs2, vector(coords_i, coords_l, boundary))
    six_term = (σ2 / r2) ^ 3
    return inter.weight_14 * 4 * inter.ϵ14_mixed * (six_term ^ 2 - six_term)
end

########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################

# const NATIVE_SIMD_WIDTH = if HostCPUFeatures.has_avx512f()
#     8  # Enterprise Servers (AVX-512)
# elseif HostCPUFeatures.has_avx2() || HostCPUFeatures.has_avx()
#     4  # Standard Intel/AMD (AVX/AVX2)
# else
#     2  # Apple Silicon (NEON) or legacy chips
# end

# Structs 

const SIMD_WIDTH = 8

abstract type PackedLayout end
struct LJOnlyLayout <: PackedLayout end
struct CoulOnlyLayout <: PackedLayout end
struct LJCoulSplitLayout <: PackedLayout end

struct PackedFlatSoA{T, L<:PackedLayout}
    offsets::Vector{Int}
    split_idxs::Vector{Int}
    adj_list::Vector{Int}
    sigmas::Vector{T}
    eps::Vector{T}
    charges::Vector{T}
    lj_weights::Vector{T}
    coul_weights::Vector{T} # positive = normal pair, negative = special pair, abs(weight) = actual scaling 

    _both_counts::Vector{Int}
    _coul_counts::Vector{Int}
    _both_part_counts::Vector{Int}
    _coul_part_counts::Vector{Int}
    _both_part_pos::Vector{Int}
    _coul_part_pos::Vector{Int}
    _is_both::Vector{Bool}
end

# Holds both the standard Molly list and your SIMD-friendly packed arrays
struct PackedNeighborList{L, P ,S}
    standard_list::L
    packed_data::P
    soa_params::S
end

struct SIMDNeighborFinder{N, I}
    base_finder::N
    inter::I
end

@kwdef struct SIMDLennardJones{C, SC, S, E, W} <: PairwiseInteraction
    cutoff::C = DistanceCutoff(1.0u"nm")
    shortcut::SC = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    weight_special::W = 1
end

abstract type AbstractSIMDCoulomb <: PairwiseInteraction end

@kwdef struct SIMDCoulomb{C, W, T} <: AbstractSIMDCoulomb
    cutoff::C = NoCutoff()
    weight_special::W = 1 
    coulomb_const:: T = 138.93545764
end

# REACTION FIELD INTERACTION 
struct SIMDCoulombReactionField{D, S, W, T, K} <: AbstractSIMDCoulomb
    dist_cutoff::D
    solvent_dielectric::S
    weight_special::W
    coulomb_const::T
    two_krf::K
end

function SIMDCoulombReactionField(;
    dist_cutoff,
    solvent_dielectric = 78.3,
    weight_special = 1,
    coulomb_const = 138.93545764,
)
    F = typeof(ustrip(dist_cutoff))

    sd = F(solvent_dielectric)
    ws = F(weight_special)
    cc = F(ustrip(coulomb_const))

    ws < zero(F) &&
        throw(ArgumentError("signed Coulomb weights require nonnegative weight_special"))

    rc = F(ustrip(dist_cutoff))
    krf = inv(rc^3) * ((sd - one(F)) / (F(2) * sd + one(F)))
    two_krf = F(2) * krf

    return SIMDCoulombReactionField(dist_cutoff, sd, ws, cc, two_krf)
end

struct SIMDCoulombEwald{D, E, W, C, A, B} <: AbstractSIMDCoulomb
    dist_cutoff::D
    error_tol::E
    weight_special::W
    coulomb_const::C
    α::A
    two_α_inv_sqrtπ::B
end

function SIMDCoulombEwald(;
    dist_cutoff,
    error_tol = 0.0005,
    weight_special = 1,
    coulomb_const = 138.93545764,
    α = nothing,
)
    F = typeof(ustrip(dist_cutoff))
    rc = F(ustrip(dist_cutoff))
    tol = F(error_tol)
    ws = F(weight_special)
    cc = F(ustrip(coulomb_const))

    ws < zero(F) && 
        throw(ArgumentError("signed Coulomb weights require nonnegative weight_special"))
    
    alpha_val = isnothing(α) ? sqrt(-log(F(2) * tol)) / rc : F(ustrip(α))

    # 2 / sqrt(pi)
    two_inv_sqrtπ = F(1.1283791670955126)
    two_α_inv_sqrtπ = two_inv_sqrtπ * alpha_val

    return SIMDCoulombEwald(dist_cutoff, tol, ws, cc, alpha_val, two_α_inv_sqrtπ)
end




use_neighbors(::SIMDLennardJones) = true  
use_neighbors(::AbstractSIMDCoulomb) = true

# Duck typing
# If Molly's loggers ask for `.n` or `.list`, route it to the standard_list
function Base.getproperty(nl::PackedNeighborList, sym::Symbol)
    if sym === :standard_list
        return getfield(nl, :standard_list)
    elseif sym === :packed_data
        return getfield(nl, :packed_data)
    elseif sym === :soa_params
        return getfield(nl, :soa_params)
    else
        return getproperty(getfield(nl, :standard_list), sym)
    end
end

canonical_inters(inters::Tuple{<:SIMDLennardJones}) = inters
canonical_inters(inters::Tuple{<:AbstractSIMDCoulomb}) = inters

canonical_inters(inters::Tuple{<:SIMDLennardJones, <:AbstractSIMDCoulomb}) = inters
canonical_inters(inters::Tuple{<:AbstractSIMDCoulomb, <:SIMDLennardJones}) = (inters[2], inters[1])

layout_type(::Tuple{<:SIMDLennardJones}) = LJOnlyLayout
layout_type(::Tuple{<:AbstractSIMDCoulomb}) = CoulOnlyLayout
layout_type(::Tuple{<:SIMDLennardJones, <:AbstractSIMDCoulomb}) = LJCoulSplitLayout



Base.iterate(nl::PackedNeighborList, args...) = iterate(nl.standard_list, args...)
Base.length(nl::PackedNeighborList) = length(nl.standard_list)

@inline Base.:*(c::SIMD.Vec, v::SVector{3}) = SVector(c * v[1], c * v[2], c * v[3])
@inline Base.:*(v::SVector{3}, c::SIMD.Vec) = SVector(c * v[1], c * v[2], c * v[3])

# LJ weights remain ordinary positive scale factors.
@inline function _lj_weight(inter::SIMDLennardJones, is_special, ::Type{T}) where {T}
    return is_special ? T(inter.weight_special) : one(T)
end

# Coulomb-like weights encode specialness in the sign.
@inline function _coul_signed_weight(inter::AbstractSIMDCoulomb, is_special, ::Type{T}) where {T}
    w = is_special ? T(inter.weight_special) : one(T)
    w < zero(T) &&
        throw(ArgumentError("signed Coulomb weights require nonnegative weight_special"))
    return is_special ? -w : w
end

@inline function simd_erfc_from_exp(x::V, exp_mx2::V) where {V <: SIMD.Vec}
    F = eltype(V)

    p  = F(0.3275911)
    a1 = F(0.254829592)
    a2 = F(-0.284496736)
    a3 = F(1.421413741)
    a4 = F(-1.453152027)
    a5 = F(1.061405429)

    t = one(V) / (one(V) + p * x)

    # Equivalent to:
    # (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
    # but with fewer temporaries.
    poly = muladd(a5, t, a4)
    poly = muladd(poly, t, a3)
    poly = muladd(poly, t, a2)
    poly = muladd(poly, t, a1)

    return (poly * t) * exp_mx2
end

# LJ uses normal weights 
@inline function custom_force(inter::SIMDLennardJones, dr, safe_dist2, atom_i,
    atom_j, neigh_weights, cutoff_2)
    f_div_r = force_apply_cutoff(inter.cutoff, inter, safe_dist2, atom_i, atom_j, cutoff_2)
    f_div_r_weighted = f_div_r * neigh_weights
    return f_div_r_weighted * dr
end

@inline function custom_force(inter::SIMDCoulomb, dr, safe_dist2, atom_i, atom_j, signed_weights, cutoff_2)
    f_div_r = force_apply_cutoff(inter.cutoff, inter, safe_dist2, atom_i, atom_j, cutoff_2)
    f_div_r_weighted = f_div_r * abs(signed_weights)
    return f_div_r_weighted * dr
end


# RF has dist_cutoff not AbstractCutOff so uses own formula 
@inline function custom_force(inter::SIMDCoulombReactionField, dr, safe_dist2, atom_i, atom_j, signed_weights, cutoff_2)
    f_div_r = core_force_div_r(inter, safe_dist2, atom_i, atom_j, signed_weights) 
    return f_div_r * dr 
end

@inline function custom_force(
    inter::SIMDCoulombEwald,
    dr,
    safe_dist2,
    atom_i,
    atom_j,
    signed_weights,
    cutoff_2,
)
    f_div_r = core_force_div_r(inter, safe_dist2, atom_i, atom_j, signed_weights)
    return f_div_r * dr
end


@inline function force_apply_cutoff(cutoff::DistanceCutoff, inter, dist_2, atom_i, atom_j, cutoff_2)
    return core_force_div_r(inter, dist_2, atom_i, atom_j)
end

@inline function force_apply_cutoff(cutoff::NoCutoff, inter, dist_2, atom_i, atom_j, cutoff_2)
    return core_force_div_r(inter, dist_2, atom_i, atom_j)
end

@inline function core_force_div_r(inter::SIMDLennardJones, dist_2, atom_i, atom_j)
    sigma_ij = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    eps_ij = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)

    V = typeof(dist_2)
    ElT = eltype(V)

    inv_dist_2 = one(V) / dist_2
    base_ratio_2 = (sigma_ij * sigma_ij) * inv_dist_2
    br2_sq = base_ratio_2 * base_ratio_2
    base_ratio_6 = br2_sq * base_ratio_2

    term = base_ratio_6 * muladd(ElT(48.0), base_ratio_6, ElT(-24.0))
    return eps_ij * inv_dist_2 * term
end

# Old plain coulomb force for old cutoff path 
@inline function core_force_div_r(inter::SIMDCoulomb, dist_2, atom_i, atom_j)
    V = typeof(dist_2)
    inv_dist_2 = one(V) / dist_2
    inv_dist = SIMD.sqrt(inv_dist_2)
    inv_dist_3 = inv_dist_2 * inv_dist # 1.0 / r^3
    
    return inter.coulomb_const * atom_i.q * atom_j.q * inv_dist_3
end

@inline function core_force_div_r(inter::SIMDCoulombReactionField, dist_2, atom_i, atom_j, signed_weights)
    V = typeof(dist_2)

    inv_dist_2 = one(V) / dist_2
    inv_dist = SIMD.sqrt(inv_dist_2)
    inv_dist_3 = inv_dist_2 * inv_dist

    weight = abs(signed_weights)
    rf_scale = vifelse(signed_weights < zero(V), zero(V), one(V))

    keqq = inter.coulomb_const * atom_i.q * atom_j.q
    rf_term = muladd(-rf_scale, inter.two_krf, inv_dist_3)
    return weight * keqq * rf_term
end

@inline function core_force_div_r(
    inter::SIMDCoulombEwald,
    dist_2,
    atom_i,
    atom_j,
    signed_weights,
)
    V = typeof(dist_2)
    F = eltype(V)

    alpha = F(inter.α)
    two_α_inv_sqrtπ = F(inter.two_α_inv_sqrtπ)
    coulomb_const = F(inter.coulomb_const)

    inv_dist_2 = one(V) / dist_2
    inv_dist = SIMD.sqrt(inv_dist_2)
    inv_dist_3 = inv_dist_2 * inv_dist

    # r = sqrt(dist_2), but this reuses inv_dist.
    r = dist_2 * inv_dist

    αr = alpha * r
    αr2 = αr * αr
    exp_mαr2 = SIMD.exp(-αr2)

    erfc_αr = simd_erfc_from_exp(αr, exp_mαr2)

    normal_term = muladd(two_α_inv_sqrtπ * r, exp_mαr2, erfc_αr)

    # Negative signed weight means special electrostatics:
    # special Ewald real-space force becomes ordinary Coulomb.
    special_mask = signed_weights < zero(V)
    ewald_factor = vifelse(special_mask, one(V), normal_term)

    weight = abs(signed_weights)
    keqq = coulomb_const * atom_i.q * atom_j.q

    return weight * keqq * inv_dist_3 * ewald_factor
end

# @inline function core_force_div_r(
#     inter::SIMDCoulombEwald,
#     dist_2,
#     atom_i,
#     atom_j,
#     signed_weights,
# )
#     V = typeof(dist_2)
#     F = eltype(V)

#     alpha = F(inter.α)
#     two_α_inv_sqrtπ = F(inter.two_α_inv_sqrtπ)
#     coulomb_const = F(inter.coulomb_const)

#     inv_dist_2 = one(V) / dist_2
#     inv_dist = SIMD.sqrt(inv_dist_2)
#     inv_dist_3 = inv_dist_2 * inv_dist

#     r = dist_2 * inv_dist

#     αr = alpha * r
#     αr2 = αr * αr
#     exp_mαr2 = SIMD.exp(-αr2)

#     erfc_αr = simd_erfc_from_exp(αr, exp_mαr2)

#     # Standard Ewald Real-Space factor
#     normal_term = muladd(two_α_inv_sqrtπ * r, exp_mαr2, erfc_αr)

#     # === FIXED: The Artifact Subtraction ===
#     weight = abs(signed_weights)
    
#     # Ewald real space must subtract the k-space artifact for special pairs.
#     # For normal pairs (weight=1), this neatly resolves to normal_term - 0.
#     ewald_factor = normal_term - (one(V) - weight)

#     keqq = coulomb_const * atom_i.q * atom_j.q

#     # Notice we removed `weight * keqq` here, because the weight is 
#     # mathematically baked into the ewald_factor now!
#     return keqq * inv_dist_3 * ewald_factor
# end

@inline function simd_geometry(x_i, y_i, z_i, neigh_x::V, neigh_y::V, neigh_z::V, sim_params) where {V <: SIMD.Vec}
    dx = x_i - neigh_x
    dy = y_i - neigh_y
    dz = z_i - neigh_z

    # Periodic boundary wrapping
    dx = muladd(-sim_params.box_x, round(dx * sim_params.inv_box_x), dx)
    dy = muladd(-sim_params.box_y, round(dy * sim_params.inv_box_y), dy)
    dz = muladd(-sim_params.box_z, round(dz * sim_params.inv_box_z), dz)

    # Squared distance
    dist_2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
    
    return SVector(dx, dy, dz), dist_2
end

@inline function simd_force_eval(dr, dist_2, atom_i, atom_j, inter, neigh_weights, cutoff_2)
    # Extract the vector type dynamically so we can make zeros and ones
    V = typeof(dist_2) 

    # Create the interaction-specific mask
    mask = (dist_2 <= cutoff_2) & (dist_2 > zero(V))

    # Prevent divide-by-zero for padded dummies
    safe_dist_2 = vifelse(mask, dist_2, one(V)) 

    # Calculate raw force
    fdr_raw = custom_force(inter, dr, safe_dist_2, atom_i, atom_j, neigh_weights, cutoff_2)
   
    # Apply mask
    f_x = vifelse(mask, fdr_raw[1], zero(V))
    f_y = vifelse(mask, fdr_raw[2], zero(V))
    f_z = vifelse(mask, fdr_raw[3], zero(V))

    return f_x, f_y, f_z
end


@inline extract_cutoff_sq(cutoff::DistanceCutoff) = ustrip(cutoff.dist_cutoff)^2
@inline extract_cutoff_sq(cutoff::NoCutoff) = Inf

# Interaction level cutoff extraction 
# To combat RF and Ewald having direct distance cutoff 
@inline extract_cutoff_sq(inter::SIMDLennardJones) = extract_cutoff_sq(inter.cutoff)
@inline extract_cutoff_sq(inter::SIMDCoulomb) = extract_cutoff_sq(inter.cutoff)
@inline extract_cutoff_sq(inter::SIMDCoulombReactionField) = ustrip(inter.dist_cutoff)^2
@inline extract_cutoff_sq(inter::SIMDCoulombEwald) = ustrip(inter.dist_cutoff)^2

# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING


@inline function gather_xyz(flat_coords, neigh_idxs)
    idx_x = neigh_idxs * 3 - 2
    idx_y = neigh_idxs * 3 - 1
    idx_z = neigh_idxs * 3
    
    x = vgather(flat_coords, idx_x)
    y = vgather(flat_coords, idx_y)
    z = vgather(flat_coords, idx_z)
    
    return x, y, z
end


# Returns either both or coulomb only depending on whether lj_shortcut is true
@inline function pair_bucket(lj_inter, atom_i, atom_j, is_special)
    if shortcut_pair(lj_inter.shortcut, atom_i, atom_j, is_special)
        return :coul_only
    else
        return :both
    end
end
    

@inline _part_index(i, part, n_atoms) = i + (part - 1) * n_atoms # index for atom i in the global 1D array

# Get the first and last index of each 
@inline function _part_bounds(n_items, part, n_parts) # n_items = total neighbour pairs, parts = thread chunk, n_parts = total thread chunks
    first_idx = ((part - 1) * n_items) ÷ n_parts + 1
    last_idx = (part * n_items) ÷ n_parts 
    return first_idx, last_idx # return the bounds as a tuple 
end

@inline lj_applies(lj_inter, atom_i, atom_j, is_special) =
    !Molly.shortcut_pair(lj_inter.shortcut, atom_i, atom_j, is_special)

function empty_packed_data(::Type{L}, ::Type{T}=Float64) where {L<:PackedLayout, T}
    return PackedFlatSoA{T, L}(
        Int[], Int[], Int[],
        T[], T[], T[], T[], T[],
        Int[], Int[], Int[], Int[], Int[], Int[], Bool[],
    )
end

function find_neighbors(sys::System, nf::SIMDNeighborFinder, old_neighbors, step_n::Integer, force_tracking::Bool=false; kwargs...)
    old_standard = isnothing(old_neighbors) ? nothing : old_neighbors.standard_list
    new_standard = Molly.find_neighbors(sys, nf.base_finder, old_standard, step_n, force_tracking; kwargs...)
    needs_rebuild = isnothing(old_neighbors) || force_tracking || iszero(step_n % nf.base_finder.n_steps)

    if !needs_rebuild
        return old_neighbors
    end

    canonical = canonical_inters(nf.inter)

    if isnothing(old_neighbors)
        T_float = typeof(ustrip(sys.atoms[1].σ))
        soa_params = (σ = T_float[ustrip(a.σ) for a in sys.atoms], ϵ = T_float[ustrip(a.ϵ) for a in sys.atoms], q = T_float[ustrip(a.charge) for a in sys.atoms])
        
        # Initialize empty arrays; the builder handles all resizing
        packed_data = empty_packed_data(layout_type(canonical), T_float)
    else
        soa_params = old_neighbors.soa_params
        packed_data = old_neighbors.packed_data
    end
    
    build_packed_adj_list!(packed_data, sys.atoms, new_standard, SIMD_WIDTH, soa_params, canonical...,)

    return PackedNeighborList(new_standard, packed_data, soa_params)
end

function build_packed_adj_list!(packed_data::PackedFlatSoA{T, LJCoulSplitLayout}, atoms, molly_neighbours, N_SIMD, soa_params,
    lj_inter::SIMDLennardJones, coul_inter::AbstractSIMDCoulomb,) where {T}

    n_atoms = length(atoms)
    n_pairs = molly_neighbours.n # total number of neighbour list pairs
    #n_parts = Threads.nthreads() # number of threads - the neighbour list gets split into this many parts 
    n_parts = min(Threads.nthreads(), 8)
    chunk_size = 2 * N_SIMD # how large a chunk is since its unrolled 2x 
    part_len = n_atoms * n_parts # each thread gets n_atom slots so total length 

    resize!(packed_data._both_counts, n_atoms) # vector of ints of size n_atoms - stores no. both counts for each atom
    resize!(packed_data._coul_counts, n_atoms) # vector of ints of size n_atoms - stores no. coul counts for each atom
    resize!(packed_data._both_part_counts, part_len) # vector of ints of size n_atoms * n_threads - each thread writes to its own n_atoms section?
    resize!(packed_data._coul_part_counts, part_len) # same as above except coul count rather than both count 
    resize!(packed_data._both_part_pos, part_len) # ? 
    resize!(packed_data._coul_part_pos, part_len) # ? 
    resize!(packed_data._is_both, n_pairs) # ? 

    # initialise everythin with 0's 
    fill!(packed_data._both_counts, 0)
    fill!(packed_data._coul_counts, 0)
    fill!(packed_data._both_part_counts, 0)
    fill!(packed_data._coul_part_counts, 0)

    # Pass 1: Classify and count, thread-locally by atom.
    Threads.@threads for part in 1:n_parts
        first_idx, last_idx = _part_bounds(n_pairs, part, n_parts) # get the first and last index of the thread 

        @inbounds for idx in first_idx:last_idx
            i, j, is_special = molly_neighbours.list[idx]
            is_both = lj_applies(lj_inter, atoms[i], atoms[j], is_special)
            packed_data._is_both[idx] = is_both

            if is_both
                packed_data._both_part_counts[_part_index(i, part, n_atoms)] += 1
                packed_data._both_part_counts[_part_index(j, part, n_atoms)] += 1
            else
                packed_data._coul_part_counts[_part_index(i, part, n_atoms)] += 1
                packed_data._coul_part_counts[_part_index(j, part, n_atoms)] += 1
            end
        end
    end

    # This is the standard offsets and split idxs that we will use in the end  
    resize!(packed_data.offsets, n_atoms + 1)
    resize!(packed_data.split_idxs, n_atoms)

    # Pass 2 and pass 3: Reduce counts, build padded offsets, and convert per-thread counts into per-thread write cursors.
    current_offset = 1
    @inbounds for i in 1:n_atoms
        # Shared (LJ+Coulomb) Block 
        packed_data.offsets[i] = current_offset # initially, offset i is the first slot of atom i's both block
        both_pos = current_offset # the both_pos will increase 
        both_total = 0

        for part in 1:n_parts # loop over threads, so for atom i we want to assign where thread 1/2/3 starts writing
            k = _part_index(i, part, n_atoms) # index for atom i in a global array 
            # k is the storage location for thread 'parts' data for atom i 
            # both_part_pos is a global 1d array of n_atoms * n_threads
            packed_data._both_part_pos[k] = both_pos     # for atom i in the global array, thread x needs to start writing at both_pos 
            both_count = packed_data._both_part_counts[k]
            both_pos += both_count # now advance the counter by how many neighbours that thread has for that atom
            both_total += both_count
        end

        packed_data._both_counts[i] = both_total
        both_pad = both_total % chunk_size == 0 ? 0 : chunk_size - both_total % chunk_size
        current_offset += both_total + both_pad

        # --- Coulomb-Only Block ---
        packed_data.split_idxs[i] = current_offset
        coul_pos = current_offset # split idx i is the first slot of atom i's coulomb block 
        coul_total = 0

        for part in 1:n_parts
            k = _part_index(i, part, n_atoms)
            packed_data._coul_part_pos[k] = coul_pos
            coul_count = packed_data._coul_part_counts[k]
            coul_pos += coul_count
            coul_total += coul_count
        end

        packed_data._coul_counts[i] = coul_total
        coul_pad = coul_total % chunk_size == 0 ? 0 : chunk_size - coul_total % chunk_size
        current_offset += coul_total + coul_pad
    end

    # By the end of part 2 we have the packed_data.offsets and packed_data.split_idxs sorted and know the total length
    # After pass 3 we end with a n_atoms * n_threads array that for each atom within each thread has the starting position

    # Finalize sizes based on the math above
    packed_data.offsets[n_atoms + 1] = current_offset
    total_len = current_offset - 1 # what the total length of the final padded list will be 

    # # Resize the arrays so that they are the length of the total, padded, adjacent list 
    resize!(packed_data.adj_list, total_len)
    resize!(packed_data.sigmas, total_len)
    resize!(packed_data.eps, total_len)
    resize!(packed_data.charges, total_len)
    resize!(packed_data.lj_weights, total_len)
    resize!(packed_data.coul_weights, total_len)

    # Pass 4: Fill arrays in parallel --> this is like pass 1 but we actually fill final array 
    Threads.@threads for part in 1:n_parts
        first_idx, last_idx = _part_bounds(n_pairs, part, n_parts) # split the pairs by parts and bound each thread

        @inbounds for idx in first_idx:last_idx # same as pass one 
            i, j, is_special = molly_neighbours.list[idx]
            coul_w = _coul_signed_weight(coul_inter, is_special, T)
            lj_w = _lj_weight(lj_inter, is_special, T)

            if packed_data._is_both[idx]
                # remember part index ensures you get the write global index 

                pos_i = packed_data._both_part_pos[_part_index(i, part, n_atoms)] # retrieve the index of atom i to write to 
                packed_data._both_part_pos[_part_index(i, part, n_atoms)] = pos_i + 1 # add one for atom i since we are writing to the original index

                # Write the atom data of the neighbour j to atom i in the packed list 
                packed_data.adj_list[pos_i] = j
                packed_data.sigmas[pos_i] = soa_params.σ[j]
                packed_data.eps[pos_i] = soa_params.ϵ[j]
                packed_data.charges[pos_i] = soa_params.q[j]
                packed_data.lj_weights[pos_i] = lj_w
                packed_data.coul_weights[pos_i] = coul_w

                # Do the same thing for atom j, the other half of the neighbour pair 
                pos_j = packed_data._both_part_pos[_part_index(j, part, n_atoms)]
                packed_data._both_part_pos[_part_index(j, part, n_atoms)] = pos_j + 1 # remember to add 1 to the position in the global 1D array 

                # Fill the details of the atom i at the position j 
                packed_data.adj_list[pos_j] = i
                packed_data.sigmas[pos_j] = soa_params.σ[i]
                packed_data.eps[pos_j] = soa_params.ϵ[i]
                packed_data.charges[pos_j] = soa_params.q[i]
                packed_data.lj_weights[pos_j] = lj_w
                packed_data.coul_weights[pos_j] = coul_w
            else
                # if its not both then write to coulomb 
                # _is_both did one check at the start and was the length of the neighbour pairs
                pos_i = packed_data._coul_part_pos[_part_index(i, part, n_atoms)]
                packed_data._coul_part_pos[_part_index(i, part, n_atoms)] = pos_i + 1

                packed_data.adj_list[pos_i] = j
                packed_data.charges[pos_i] = soa_params.q[j]
                packed_data.coul_weights[pos_i] = coul_w

                pos_j = packed_data._coul_part_pos[_part_index(j, part, n_atoms)]
                packed_data._coul_part_pos[_part_index(j, part, n_atoms)] = pos_j + 1

                packed_data.adj_list[pos_j] = i
                packed_data.charges[pos_j] = soa_params.q[i]
                packed_data.coul_weights[pos_j] = coul_w
            end
        end
    end

    # Pass 5: Pad rows in parallel, per atom 
    Threads.@threads for i in 1:n_atoms
        @inbounds begin
            for pos in (packed_data.offsets[i] + packed_data._both_counts[i]):(packed_data.split_idxs[i] - 1)
                # look at where the offset was + the total counts was until where the split index begins 
                # pad with empty atoms
                packed_data.adj_list[pos] = i
                packed_data.sigmas[pos] = one(T)
                packed_data.eps[pos] = one(T)
                packed_data.charges[pos] = zero(T)
                packed_data.lj_weights[pos] = zero(T)
                packed_data.coul_weights[pos] = zero(T)
            end

            # look at where the split index + coulomb counts ends until where the next index begins and pad 
            for pos in (packed_data.split_idxs[i] + packed_data._coul_counts[i]):(packed_data.offsets[i + 1] - 1)
                packed_data.adj_list[pos] = i
                packed_data.charges[pos] = zero(T)
                packed_data.coul_weights[pos] = zero(T)
            end
        end
    end

    return packed_data
end

@inline _keep_pair(::AbstractSIMDCoulomb, atoms, i, j, is_special) = true

@inline _keep_pair(lj_inter::SIMDLennardJones, atoms, i, j, is_special) =
    !Molly.shortcut_pair(lj_inter.shortcut, atoms[i], atoms[j], is_special)

@inline _simd_pad(count::Int, chunk_size::Int) =
    count % chunk_size == 0 ? 0 : chunk_size - count % chunk_size

@inline function _fill_single_entry!(
    packed_data::PackedFlatSoA{T, LJOnlyLayout},
    pos,
    neigh,
    soa_params,
    inter::SIMDLennardJones,
    weight,
) where {T}
    packed_data.adj_list[pos] = neigh
    packed_data.sigmas[pos] = soa_params.σ[neigh]
    packed_data.eps[pos] = soa_params.ϵ[neigh]
    packed_data.lj_weights[pos] = weight
end

@inline function _fill_single_entry!(
    packed_data::PackedFlatSoA{T, CoulOnlyLayout},
    pos,
    neigh,
    soa_params,
    inter::AbstractSIMDCoulomb,
    weight,
) where {T}
    packed_data.adj_list[pos] = neigh
    packed_data.charges[pos] = soa_params.q[neigh]
    packed_data.coul_weights[pos] = weight
end

@inline function _pad_single_entry!(
    packed_data::PackedFlatSoA{T, LJOnlyLayout},
    pos,
    i,
) where {T}
    packed_data.adj_list[pos] = i
    packed_data.sigmas[pos] = one(T)
    packed_data.eps[pos] = one(T)
    packed_data.lj_weights[pos] = zero(T)
end

@inline function _pad_single_entry!(
    packed_data::PackedFlatSoA{T, CoulOnlyLayout},
    pos,
    i,
) where {T}
    packed_data.adj_list[pos] = i
    packed_data.charges[pos] = zero(T)
    packed_data.coul_weights[pos] = zero(T)
end

function build_packed_adj_list!(
    packed_data::PackedFlatSoA{T, L},
    atoms,
    molly_neighbours,
    N_SIMD,
    soa_params,
    inter,
) where {T, L<:Union{LJOnlyLayout, CoulOnlyLayout}}

    n_atoms = length(atoms)
    n_pairs = molly_neighbours.n
    #n_parts = Threads.nthreads()
    n_parts = min(Threads.nthreads(), 8)
    chunk_size = 2 * N_SIMD
    part_len = n_atoms * n_parts

    resize!(packed_data._both_part_counts, part_len)
    resize!(packed_data._both_part_pos, part_len)
    resize!(packed_data._is_both, n_pairs)

    counts = packed_data._both_part_counts
    pos = packed_data._both_part_pos
    keep = packed_data._is_both
    list = molly_neighbours.list

    fill!(counts, 0)

    #Threads.@threads :static for part in 1:n_parts
    Threads.@threads for part in 1:n_parts
        first_idx, last_idx = _part_bounds(n_pairs, part, n_parts)
        base = (part - 1) * n_atoms

        @inbounds for idx in first_idx:last_idx
            i, j, is_special = list[idx]
            keep_pair = _keep_pair(inter, atoms, i, j, is_special)
            keep[idx] = keep_pair

            if keep_pair
                counts[base + i] += 1
                counts[base + j] += 1
            end
        end
    end

    resize!(packed_data.offsets, n_atoms + 1)
    resize!(packed_data.split_idxs, n_atoms)

    current_offset = 1

    @inbounds for i in 1:n_atoms
        packed_data.offsets[i] = current_offset

        write_pos = current_offset
        total = 0

        k = i
        for _ in 1:n_parts
            c = counts[k]
            pos[k] = write_pos
            write_pos += c
            total += c
            k += n_atoms
        end

        current_offset += total + _simd_pad(total, chunk_size)
        packed_data.split_idxs[i] = current_offset
    end

    packed_data.offsets[n_atoms + 1] = current_offset
    total_len = current_offset - 1

    resize!(packed_data.adj_list, total_len)

    if L === LJOnlyLayout
        resize!(packed_data.sigmas, total_len)
        resize!(packed_data.eps, total_len)
        resize!(packed_data.lj_weights, total_len)
    else
        resize!(packed_data.charges, total_len)
        resize!(packed_data.coul_weights, total_len)
    end

    #Threads.@threads :static for part in 1:n_parts
    Threads.@threads for part in 1:n_parts
        first_idx, last_idx = _part_bounds(n_pairs, part, n_parts)
        base = (part - 1) * n_atoms

        @inbounds for idx in first_idx:last_idx
            keep[idx] || continue

            i, j, is_special = list[idx]
            weight = if L === LJOnlyLayout
                _lj_weight(inter, is_special, T)
            else
                _coul_signed_weight(inter, is_special, T)
            end

            ki = base + i
            pos_i = pos[ki]
            pos[ki] = pos_i + 1
            _fill_single_entry!(packed_data, pos_i, j, soa_params, inter, weight)

            kj = base + j
            pos_j = pos[kj]
            pos[kj] = pos_j + 1
            _fill_single_entry!(packed_data, pos_j, i, soa_params, inter, weight)
        end
    end

    last_part_base = (n_parts - 1) * n_atoms

    #Threads.@threads :static for i in 1:n_atoms
    Threads.@threads for i in 1:n_atoms
        @inbounds for p in pos[last_part_base + i]:(packed_data.offsets[i + 1] - 1)
            _pad_single_entry!(packed_data, p, i)
        end
    end

    return packed_data
end

function pairwise_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords,
    velocities, boundary, neighbors::PackedNeighborList, force_units, n_atoms,
    pairwise_inters_nonl, 
    pairwise_inters_nl::Tuple, 
    ::Val{n_threads}, ::Val{needs_vir}, step_n=0) where {n_threads, needs_vir}

    fill!(fs_nounits, zero(eltype(fs_nounits)))
    #inter = pairwise_inters_nl[1]
    
    # Unpack data 
    packed_data = neighbors.packed_data
    soa_params = neighbors.soa_params

    # Setup box and types 
    FT = eltype(eltype(fs_nounits))
    flat_coords = reinterpret(FT, coords) # zero allocation flattening of the coords as flat float64 array for vgather

    # Calculate box metrics for distances
    box_x = FT(ustrip(boundary.side_lengths[1]))
    box_y = FT(ustrip(boundary.side_lengths[2]))
    box_z = FT(ustrip(boundary.side_lengths[3]))
    inv_box_x = one(FT) / box_x
    inv_box_y = one(FT) / box_y
    inv_box_z = one(FT) / box_z

    # Multithreading part 
    n_t = n_threads
    num_chunks = 4 * n_t #T8 = 8xn_t, #T16 = 4xn_t, #T32 = 4xn_t, #T64 = 2/1xn_t
    n_atoms = length(coords)

    # Allocate chunks
    chunk_size = cld(n_atoms, num_chunks)
    counter = Threads.Atomic{Int}(1) # hardware counter
    
    sim_params = (box_x = box_x, box_y = box_y, box_z = box_z, 
        inv_box_x = inv_box_x, inv_box_y = inv_box_y, inv_box_z = inv_box_z)
                    
    canonical = canonical_inters(pairwise_inters_nl)
    cutoffs = ntuple(k -> extract_cutoff_sq(canonical[k]), length(canonical))

    n_atoms_total = length(coords)
    # Static scheduling bypasses atomic locks entirely.
    # The CPU distributes n_atoms across the threads once, instantly.
    #Threads.@threads :static for i in 1:n_atoms_total
    Threads.@threads :static for i in 1:n_atoms_total
        simd_chunk_forces!(fs_nounits, i, packed_data, soa_params, sim_params, coords, flat_coords, canonical, cutoffs, Val(SIMD_WIDTH))
    end

    # # Spawn the right number of tasks per cores
    # @sync for _ in 1:n_t
    #     Threads.@spawn begin
            
    #         while true
    #             # Get chunk then add 1 
    #             chunk_id = Threads.atomic_add!(counter, 1)
                
    #             # Break if it exceeds
    #             if chunk_id > num_chunks
    #                 break
    #             end

    #             start_idx = (chunk_id - 1) * chunk_size + 1
    #             end_idx   = min(chunk_id * chunk_size, n_atoms)
                
    #             for i in start_idx:end_idx
    #                 simd_chunk_forces!(fs_nounits, i, packed_data, soa_params, sim_params, coords, flat_coords, canonical, cutoffs, Val(SIMD_WIDTH),)
    #             end
    #         end
    #     end
    # end

    return fs_nounits
end

@inline function simd_chunk_forces!(fs_nounits, i, packed_data::PackedFlatSoA{T, LJCoulSplitLayout}, soa_params, sim_params, coords, flat_coords, inters::Tuple{<:SIMDLennardJones, <:AbstractSIMDCoulomb}, cutoffs,
    ::Val{N_SIMD}) where {T, N_SIMD}

    lj_inter = inters[1]
    coul_inter = inters[2]
    lj_cutoff_2 = cutoffs[1]
    coul_cutoff_2 = cutoffs[2]

    VFloat = Vec{N_SIMD, T}
    VInt = Vec{N_SIMD, Int}

    xi = T(ustrip(coords[i][1]))
    yi = T(ustrip(coords[i][2]))
    zi = T(ustrip(coords[i][3]))

    atom_i_proxy = (σ = soa_params.σ[i], ϵ = soa_params.ϵ[i], q = soa_params.q[i])

    f_ix_vec_1 = zero(VFloat); f_iy_vec_1 = zero(VFloat); f_iz_vec_1 = zero(VFloat)
    f_ix_vec_2 = zero(VFloat); f_iy_vec_2 = zero(VFloat); f_iz_vec_2 = zero(VFloat)

    both_start = packed_data.offsets[i]
    both_end   = packed_data.split_idxs[i] - 1

    @inbounds for j in both_start:(2 * N_SIMD):both_end
        neigh_idxs_1 = vload(VInt, packed_data.adj_list, j)
        neigh_idxs_2 = vload(VInt, packed_data.adj_list, j + N_SIMD)

        # Full Memory Load
        atom_j_proxy_1 = (
            σ = vload(VFloat, packed_data.sigmas, j),
            ϵ = vload(VFloat, packed_data.eps, j),
            q = vload(VFloat, packed_data.charges, j)
        )
        atom_j_proxy_2 = (
            σ = vload(VFloat, packed_data.sigmas, j + N_SIMD),
            ϵ = vload(VFloat, packed_data.eps, j + N_SIMD),
            q = vload(VFloat, packed_data.charges, j + N_SIMD)
        )

        lj_weights_1 = vload(VFloat, packed_data.lj_weights, j)
        lj_weights_2 = vload(VFloat, packed_data.lj_weights, j + N_SIMD)
        coul_signed_weights_1 = vload(VFloat, packed_data.coul_weights, j)
        coul_signed_weights_2 = vload(VFloat, packed_data.coul_weights, j + N_SIMD)

        neigh_x_1, neigh_y_1, neigh_z_1 = gather_xyz(flat_coords, neigh_idxs_1)
        neigh_x_2, neigh_y_2, neigh_z_2 = gather_xyz(flat_coords, neigh_idxs_2)

        dr_1, dist_2_1 = simd_geometry(xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, sim_params)
        dr_2, dist_2_2 = simd_geometry(xi, yi, zi, neigh_x_2, neigh_y_2, neigh_z_2, sim_params)

        # Apply LJ
        lj_x_1, lj_y_1, lj_z_1 = simd_force_eval(dr_1, dist_2_1, atom_i_proxy, atom_j_proxy_1, lj_inter, lj_weights_1, lj_cutoff_2)
        lj_x_2, lj_y_2, lj_z_2 = simd_force_eval(dr_2, dist_2_2, atom_i_proxy, atom_j_proxy_2, lj_inter, lj_weights_2, lj_cutoff_2)

        # Apply Coulomb (REUSING THE SAME GEOMETRY!)
        coul_x_1, coul_y_1, coul_z_1 = simd_force_eval(dr_1, dist_2_1, atom_i_proxy, atom_j_proxy_1, coul_inter, coul_signed_weights_1, coul_cutoff_2)
        coul_x_2, coul_y_2, coul_z_2 = simd_force_eval(dr_2, dist_2_2, atom_i_proxy, atom_j_proxy_2, coul_inter, coul_signed_weights_2, coul_cutoff_2)

        f_ix_vec_1 += lj_x_1 + coul_x_1; f_iy_vec_1 += lj_y_1 + coul_y_1; f_iz_vec_1 += lj_z_1 + coul_z_1
        f_ix_vec_2 += lj_x_2 + coul_x_2; f_iy_vec_2 += lj_y_2 + coul_y_2; f_iz_vec_2 += lj_z_2 + coul_z_2
    end

    coul_start = packed_data.split_idxs[i]
    coul_end   = packed_data.offsets[i + 1] - 1

    @inbounds for j in coul_start:(2 * N_SIMD):coul_end
        neigh_idxs_1 = vload(VInt, packed_data.adj_list, j)
        neigh_idxs_2 = vload(VInt, packed_data.adj_list, j + N_SIMD)

        # ONLY Load Charges
        atom_j_proxy_1 = (σ = zero(VFloat), ϵ = zero(VFloat), q = vload(VFloat, packed_data.charges, j))
        atom_j_proxy_2 = (σ = zero(VFloat), ϵ = zero(VFloat), q = vload(VFloat, packed_data.charges, j + N_SIMD))

        coul_signed_weights_1 = vload(VFloat, packed_data.coul_weights, j)
        coul_signed_weights_2 = vload(VFloat, packed_data.coul_weights, j + N_SIMD)

        neigh_x_1, neigh_y_1, neigh_z_1 = gather_xyz(flat_coords, neigh_idxs_1)
        neigh_x_2, neigh_y_2, neigh_z_2 = gather_xyz(flat_coords, neigh_idxs_2)

        # Hoisted Geometry
        dr_1, dist_2_1 = simd_geometry(xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, sim_params)
        dr_2, dist_2_2 = simd_geometry(xi, yi, zi, neigh_x_2, neigh_y_2, neigh_z_2, sim_params)

        # Apply Coulomb ONLY
        coul_x_1, coul_y_1, coul_z_1 = simd_force_eval(dr_1, dist_2_1, atom_i_proxy, atom_j_proxy_1, coul_inter, coul_signed_weights_1, coul_cutoff_2)
        coul_x_2, coul_y_2, coul_z_2 = simd_force_eval(dr_2, dist_2_2, atom_i_proxy, atom_j_proxy_2, coul_inter, coul_signed_weights_2, coul_cutoff_2)

        f_ix_vec_1 += coul_x_1; f_iy_vec_1 += coul_y_1; f_iz_vec_1 += coul_z_1
        f_ix_vec_2 += coul_x_2; f_iy_vec_2 += coul_y_2; f_iz_vec_2 += coul_z_2
    end

    f_ix = sum(f_ix_vec_1 + f_ix_vec_2)
    f_iy = sum(f_iy_vec_1 + f_iy_vec_2)
    f_iz = sum(f_iz_vec_1 + f_iz_vec_2)

    @inbounds fs_nounits[i] += SVector(f_ix, f_iy, f_iz)
end

@inline function _load_atom_j_proxy(
    packed_data::PackedFlatSoA{T, LJOnlyLayout},
    j,
    ::Type{V},
) where {T, V}
    return (
        σ = vload(V, packed_data.sigmas, j),
        ϵ = vload(V, packed_data.eps, j),
        q = zero(V),
    )
end

@inline function _load_atom_j_proxy(
    packed_data::PackedFlatSoA{T, CoulOnlyLayout},
    j,
    ::Type{V},
) where {T, V}
    return (
        σ = zero(V),
        ϵ = zero(V),
        q = vload(V, packed_data.charges, j),
    )
end

@inline _load_weights(packed_data::PackedFlatSoA{T, LJOnlyLayout}, j, ::Type{V}) where {T, V} =
    vload(V, packed_data.lj_weights, j)

@inline _load_weights(packed_data::PackedFlatSoA{T, CoulOnlyLayout}, j, ::Type{V}) where {T, V} =
    vload(V, packed_data.coul_weights, j)


@inline function simd_chunk_forces!(
    fs_nounits,
    i,
    packed_data::PackedFlatSoA{T, L},
    soa_params,
    sim_params,
    coords,
    flat_coords,
    inters::Tuple{I},
    cutoffs,
    ::Val{N_SIMD},
) where {T, L<:Union{LJOnlyLayout, CoulOnlyLayout}, I, N_SIMD}

    VFloat = Vec{N_SIMD, T}
    VInt = Vec{N_SIMD, Int}

    inter = inters[1]
    cutoff_2 = cutoffs[1]

    xi = T(ustrip(coords[i][1]))
    yi = T(ustrip(coords[i][2]))
    zi = T(ustrip(coords[i][3]))

    atom_i_proxy = (
        σ = soa_params.σ[i],
        ϵ = soa_params.ϵ[i],
        q = soa_params.q[i],
    )

    f_ix_vec_1 = zero(VFloat); f_iy_vec_1 = zero(VFloat); f_iz_vec_1 = zero(VFloat)
    f_ix_vec_2 = zero(VFloat); f_iy_vec_2 = zero(VFloat); f_iz_vec_2 = zero(VFloat)

    start_idx = packed_data.offsets[i]
    end_idx = packed_data.offsets[i + 1] - 1

    @inbounds for j in start_idx:(2 * N_SIMD):end_idx
        neigh_idxs_1 = vload(VInt, packed_data.adj_list, j)
        neigh_idxs_2 = vload(VInt, packed_data.adj_list, j + N_SIMD)

        atom_j_proxy_1 = _load_atom_j_proxy(packed_data, j, VFloat)
        atom_j_proxy_2 = _load_atom_j_proxy(packed_data, j + N_SIMD, VFloat)

        weights_1 = _load_weights(packed_data, j, VFloat)
        weights_2 = _load_weights(packed_data, j + N_SIMD, VFloat)

        neigh_x_1, neigh_y_1, neigh_z_1 = gather_xyz(flat_coords, neigh_idxs_1)
        neigh_x_2, neigh_y_2, neigh_z_2 = gather_xyz(flat_coords, neigh_idxs_2)

        dr_1, dist_2_1 = simd_geometry(xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, sim_params)
        dr_2, dist_2_2 = simd_geometry(xi, yi, zi, neigh_x_2, neigh_y_2, neigh_z_2, sim_params)

        f_x_1, f_y_1, f_z_1 = simd_force_eval(dr_1, dist_2_1, atom_i_proxy, atom_j_proxy_1, inter, weights_1, cutoff_2)
        f_x_2, f_y_2, f_z_2 = simd_force_eval(dr_2, dist_2_2, atom_i_proxy, atom_j_proxy_2, inter, weights_2, cutoff_2)

        f_ix_vec_1 += f_x_1; f_iy_vec_1 += f_y_1; f_iz_vec_1 += f_z_1
        f_ix_vec_2 += f_x_2; f_iy_vec_2 += f_y_2; f_iz_vec_2 += f_z_2
    end

    f_ix = sum(f_ix_vec_1 + f_ix_vec_2)
    f_iy = sum(f_iy_vec_1 + f_iy_vec_2)
    f_iz = sum(f_iz_vec_1 + f_iz_vec_2)

    @inbounds fs_nounits[i] += SVector(f_ix, f_iy, f_iz)

    return nothing
end


########################################################################
# Cluster-based SIMD nonbonded path
########################################################################

# Spatial sorter on step 0 and on rebuilds

# 8x8 SIMD kernel - to replace simd_chunk_forces! Loop over cluster list, loading two vectors and using shufflevector()

########################################################################
# Clustered SIMD prototype: GROMACS/Pall-Hess style neighbor rebuild
#
# Prototype scope:
#   - CubicBoundary only
#   - SIMDLennardJones only
#   - scalar force loop first, no SIMD yet
#   - exclusions/special pairs are read from the base neighbor finder masks
#
# Revisit:
#   - SIMD inner loop over j lanes
#   - Coulomb / reaction-field / PME direct-space path
#   - dynamic pruning between full neighbor rebuilds
#   - cluster-cell pair search instead of O(n_clusters^2) AABB scan
#   - compact storage of image shifts and masks
########################################################################

const CLUSTER_WIDTH = SIMD_WIDTH

struct ClusteredSIMDNeighborFinder{N, I, CW, R}
    base_finder::N
    inter::I
    prune_inner_fraction::R
    build_standard_list::Bool
end

function ClusteredSIMDNeighborFinder(base_finder, inter; cluster_width::Integer=CLUSTER_WIDTH, prune_inner_fraction=0.0, build_standard_list::Bool=true,)
    
    1 <= cluster_width <= 8 ||
        throw(ArgumentError("cluster_width must be in 1:8 because UInt64 stores CW x CW masks"))
    
    0 <= prune_inner_fraction <= 1 ||
        throw(ArgumentError("prune_inner_fraction must be between 0 and 1"))
    
        return ClusteredSIMDNeighborFinder{typeof(base_finder), typeof(inter), cluster_width, typeof(prune_inner_fraction),}(
        base_finder,
        inter,
        prune_inner_fraction,
        build_standard_list,
    )
end

cluster_width(::ClusteredSIMDNeighborFinder{N, I, CW, R}) where {N, I, CW, R} = CW

mutable struct ClusterPairSoA{T}
    n_atoms::Int # num atoms in system
    n_slots::Int # total packed storage slots = n_clusters * cluster_size (inc. padded dummy slots)
    n_clusters::Int # so we can loop over n_clusters and to index bounding boxes / masks / pair_list entries
    nx::Int # num spatial bins in x direction during cluster construction
    ny::Int # num spatial bins in y direction
    cluster_col::Vector{Int32}
    col_first_cluster::Vector{Int32}
    col_last_cluster::Vector{Int32}

    # Atom and slot mapping - a slot is an individual entry within a block of slots (a cluster)
    atom_to_slot::Vector{Int32} # atom 17 --> slot 21
    slot_to_atom::Vector{Int32} # writing forces back from clustered storage into original Molly order

    # Packed particle data - for contiguous loads - avoids vgathers
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    sigma::Vector{T}
    epsilon::Vector{T}

    # Store force accumulated per packed slot - same order as x/y/Z
    # Would be mapped back to original atom ordering using slot_to_atom
    fx::Vector{T}
    fy::Vector{T}
    fz::Vector{T}

    # Thread local force buffers to avoid writes and enable 3rd law
    fx_chunks::Vector{Vector{T}}
    fy_chunks::Vector{Vector{T}}
    fz_chunks::Vector{Vector{T}}

    cluster_active_masks::Vector{UInt64} # bit mask saying which lanes correspond to real atoms
    # Axis-aligned bounding box of each cluster
    xmin::Vector{T}
    xmax::Vector{T}
    ymin::Vector{T}
    ymax::Vector{T}
    zmin::Vector{T}
    zmax::Vector{T}

    # Actual cluster-pair list
    pair_i::Vector{Int32}
    pair_j::Vector{Int32}
    shift_x::Vector{T}
    shift_y::Vector{T}
    shift_z::Vector{T}
    bbox_dist2::Vector{T}

    pair_masks::Vector{UInt64}
    special_masks::Vector{UInt64}

    pair_i_chunks::Vector{Vector{Int32}}
    pair_j_chunks::Vector{Vector{Int32}}
    shift_x_chunks::Vector{Vector{T}}
    shift_y_chunks::Vector{Vector{T}}
    shift_z_chunks::Vector{Vector{T}}
    bbox_dist2_chunks::Vector{Vector{T}}
    pair_masks_chunks::Vector{Vector{UInt64}}
    special_masks_chunks::Vector{Vector{UInt64}}

    ci_offsets::Vector{Int32}
    cj_list::Vector{Int32}
    csr_shift_x::Vector{T}
    csr_shift_y::Vector{T}
    csr_shift_z::Vector{T}
    csr_bbox_dist2::Vector{T}
    csr_pair_masks::Vector{UInt64}
    csr_special_masks::Vector{UInt64}

end

mutable struct ClusterOnlyStandardList
    n::Int
    list::Vector{Tuple{Int32, Int32, Bool}}
end

ClusterOnlyStandardList() = ClusterOnlyStandardList(0, Tuple{Int32, Int32, Bool}[])

Base.length(nl::ClusterOnlyStandardList) = nl.n
Base.iterate(nl::ClusterOnlyStandardList, state=1) =
    state > nl.n ? nothing : (nl.list[state], state + 1)

struct ClusteredNeighborList{L, P}
    standard_list::L
    cluster_data::P
end

function Base.getproperty(nl::ClusteredNeighborList, sym::Symbol)
    if sym === :standard_list
        return getfield(nl, :standard_list)
    elseif sym === :cluster_data
        return getfield(nl, :cluster_data)
    else
        return getproperty(getfield(nl, :standard_list), sym)
    end
end

Base.iterate(nl::ClusteredNeighborList, args...) = iterate(nl.standard_list, args...)
Base.length(nl::ClusteredNeighborList) = length(nl.standard_list)
Base.getindex(nl::ClusteredNeighborList, i::Integer) = nl.standard_list[i]
Base.firstindex(::ClusteredNeighborList) = 1
Base.lastindex(nl::ClusteredNeighborList) = length(nl)
Base.eachindex(nl::ClusteredNeighborList) = eachindex(nl.standard_list)

function empty_cluster_data(::Type{T}=Float64) where {T}
    return ClusterPairSoA{T}(
        0, 0, 0, 0, 0, Int32[], Int32[], Int32[],
        Int32[], Int32[],
        T[], T[], T[],
        T[], T[],
        T[], T[], T[],
        Vector{T}[], Vector{T}[], Vector{T}[],
        UInt64[],
        T[], T[], T[], T[], T[], T[],
        Int32[], Int32[],
        T[], T[], T[], T[],
        UInt64[], UInt64[],
        Vector{Int32}[], Vector{Int32}[],
        Vector{T}[], Vector{T}[], Vector{T}[], Vector{T}[],
        Vector{UInt64}[], Vector{UInt64}[],
        Int32[], Int32[],
        T[], T[], T[], T[],
        UInt64[], UInt64[],

    )
end

@inline _cluster_index(slot::Integer, ::Val{CW}) where {CW} = (slot - 1) ÷ CW + 1
@inline _lane_bit(lane::Integer) = UInt64(1) << (lane - 1)

@inline function _lane_pair_bit(lane_i::Integer, lane_j::Integer, ::Val{CW}) where {CW}
    return UInt64(1) << ((lane_i - 1) * CW + (lane_j - 1))
end

@inline function _has_lane_pair(mask::UInt64, lane_i::Integer, lane_j::Integer, ::Val{CW}) where {CW}
    return (mask & _lane_pair_bit(lane_i, lane_j, Val(CW))) != 0
end

@generated function _cluster_rotate_register(v::Vec{N, T}, ::Val{Shift}) where {N, T, Shift}
    indices = Tuple((i - 1 + Shift) % N for i in 1:N)
    return :(shufflevector(v, Val($indices)))
end

@generated function _cluster_pairmask_vec(mask::UInt64, ::Val{Shift}, ::Val{CW}) where {Shift, CW}
    lanes = map(1:CW) do lane_i
        lane_j = ((lane_i - 1 + Shift) % CW) + 1
        bit = UInt64(1) << ((lane_i - 1) * CW + (lane_j - 1))
        :((mask & $(bit)) != 0)
    end

    return :(Vec{$CW, Bool}(($(lanes...),)))
end

@inline function _cluster_wrap_coord(x::T, box_length::T) where {T}
    xw = x - box_length * floor(x / box_length)
    return xw >= box_length ? zero(T) : xw
end

@inline function _cluster_min_image_delta(a::T, b::T, box_length::T) where {T}
    d = b - a
    return d - box_length * round(d / box_length)
end

@inline function _cluster_column_index(x, y, box_x, box_y, nx::Integer, ny::Integer)
    ix = clamp(floor(Int, x * nx / box_x) + 1, 1, nx)
    iy = clamp(floor(Int, y * ny / box_y) + 1, 1, ny)
    return ix + (iy - 1) * nx
end

function _cluster_box_lengths(boundary::CubicBoundary, dist_unit, ::Type{T}) where {T}
    return (
        T(ustrip(dist_unit, boundary.side_lengths[1])),
        T(ustrip(dist_unit, boundary.side_lengths[2])),
        T(ustrip(dist_unit, boundary.side_lengths[3])),
    )
end

_cluster_box_lengths(boundary, dist_unit, ::Type{T}) where {T} =
    throw(ArgumentError("ClusteredSIMDNeighborFinder prototype currently supports CubicBoundary only"))

function _resize_cluster_storage!(data::ClusterPairSoA{T}, n_atoms, n_slots) where {T}
    resize!(data.atom_to_slot, n_atoms)
    resize!(data.slot_to_atom, n_slots)
    resize!(data.x, n_slots)
    resize!(data.y, n_slots)
    resize!(data.z, n_slots)
    resize!(data.sigma, n_slots)
    resize!(data.epsilon, n_slots)
    resize!(data.fx, n_slots)
    resize!(data.fy, n_slots)
    resize!(data.fz, n_slots)
    return data
end

function _build_cluster_order!(
    data::ClusterPairSoA{T},
    atoms,
    coords,
    boundary,
    ::Val{CW},
) where {T, CW}
    n_atoms = length(coords)
    dist_unit = unit(first(first(coords)))
    box_x, box_y, box_z = _cluster_box_lengths(boundary, dist_unit, T)

    density = T(n_atoms) / (box_x * box_y * box_z)
    grid_spacing = cbrt(T(CW) / density)
    nx = max(1, floor(Int, box_x / grid_spacing))
    ny = max(1, floor(Int, box_y / grid_spacing))

    x_by_atom = Vector{T}(undef, n_atoms)
    y_by_atom = Vector{T}(undef, n_atoms)
    z_by_atom = Vector{T}(undef, n_atoms)
    columns = [Int32[] for _ in 1:(nx * ny)]

    @inbounds for atom_i in 1:n_atoms
        x = _cluster_wrap_coord(T(ustrip(dist_unit, coords[atom_i][1])), box_x)
        y = _cluster_wrap_coord(T(ustrip(dist_unit, coords[atom_i][2])), box_y)
        z = _cluster_wrap_coord(T(ustrip(dist_unit, coords[atom_i][3])), box_z)

        x_by_atom[atom_i] = x
        y_by_atom[atom_i] = y
        z_by_atom[atom_i] = z

        col = _cluster_column_index(x, y, box_x, box_y, nx, ny)
        push!(columns[col], Int32(atom_i))
    end

    n_slots = 0
    @inbounds for col in columns
        isempty(col) && continue
        n_slots += cld(length(col), CW) * CW
    end
    n_clusters = n_slots ÷ CW

    data.n_atoms = n_atoms
    data.n_slots = n_slots
    data.n_clusters = n_clusters
    data.nx = nx
    data.ny = ny

    resize!(data.cluster_col, n_clusters)
    resize!(data.col_first_cluster, nx * ny)
    resize!(data.col_last_cluster, nx * ny)
    fill!(data.col_first_cluster, Int32(0))
    fill!(data.col_last_cluster, Int32(0))

    _resize_cluster_storage!(data, n_atoms, n_slots)
    fill!(data.slot_to_atom, Int32(0))

    slot = 1
    @inbounds for col_idx in eachindex(columns)
        col = columns[col_idx]
        isempty(col) && continue

        sort!(col; by = atom_i -> z_by_atom[Int(atom_i)])

        col_slots = cld(length(col), CW) * CW
        first_cluster = _cluster_index(slot, Val(CW))
        last_slot = slot + col_slots - 1
        last_cluster = _cluster_index(last_slot, Val(CW))

        data.col_first_cluster[col_idx] = Int32(first_cluster)
        data.col_last_cluster[col_idx] = Int32(last_cluster)

        for cluster_i in first_cluster:last_cluster
            data.cluster_col[cluster_i] = Int32(col_idx)
        end

        for atom_i32 in col
            atom_i = Int(atom_i32)
            data.atom_to_slot[atom_i] = Int32(slot)
            data.slot_to_atom[slot] = atom_i32
            data.x[slot] = x_by_atom[atom_i]
            data.y[slot] = y_by_atom[atom_i]
            data.z[slot] = z_by_atom[atom_i]
            data.sigma[slot] = T(ustrip(atoms[atom_i].σ))
            data.epsilon[slot] = T(ustrip(atoms[atom_i].ϵ))
            slot += 1
        end

        while slot <= last_slot
            data.slot_to_atom[slot] = Int32(0)
            data.x[slot] = zero(T)
            data.y[slot] = zero(T)
            data.z[slot] = zero(T)
            data.sigma[slot] = one(T)
            data.epsilon[slot] = zero(T)
            slot += 1
        end
    end

    return data
end

function _build_cluster_bounds!(data::ClusterPairSoA{T}, ::Val{CW}) where {T, CW}
    resize!(data.cluster_active_masks, data.n_clusters)
    resize!(data.xmin, data.n_clusters); resize!(data.xmax, data.n_clusters)
    resize!(data.ymin, data.n_clusters); resize!(data.ymax, data.n_clusters)
    resize!(data.zmin, data.n_clusters); resize!(data.zmax, data.n_clusters)

    @inbounds for ci in 1:data.n_clusters
        xmin = T(Inf); xmax = T(-Inf)
        ymin = T(Inf); ymax = T(-Inf)
        zmin = T(Inf); zmax = T(-Inf)
        active_mask = UInt64(0)
        first_slot = (ci - 1) * CW + 1

        for lane in 1:CW
            slot = first_slot + lane - 1
            if data.slot_to_atom[slot] != 0
                active_mask |= _lane_bit(lane)
                x = data.x[slot]; y = data.y[slot]; z = data.z[slot]
                xmin = min(xmin, x); xmax = max(xmax, x)
                ymin = min(ymin, y); ymax = max(ymax, y)
                zmin = min(zmin, z); zmax = max(zmax, z)
            end
        end

        data.cluster_active_masks[ci] = active_mask
        data.xmin[ci] = xmin; data.xmax[ci] = xmax
        data.ymin[ci] = ymin; data.ymax[ci] = ymax
        data.zmin[ci] = zmin; data.zmax[ci] = zmax
    end

    return data
end

@inline function _interval_sep(amin::T, amax::T, bmin::T, bmax::T) where {T}
    if amax < bmin
        return bmin - amax
    elseif bmax < amin
        return amin - bmax
    else
        return zero(T)
    end
end

function _bbox_distance2_shift(data::ClusterPairSoA{T}, ci, cj, box_x, box_y, box_z) where {T}
    best_dist2 = T(Inf)
    best_sx = zero(T); best_sy = zero(T); best_sz = zero(T)

    for ix in -1:1, iy in -1:1, iz in -1:1
        sx = T(ix) * box_x
        sy = T(iy) * box_y
        sz = T(iz) * box_z

        dx = _interval_sep(data.xmin[ci], data.xmax[ci], data.xmin[cj] + sx, data.xmax[cj] + sx)
        dy = _interval_sep(data.ymin[ci], data.ymax[ci], data.ymin[cj] + sy, data.ymax[cj] + sy)
        dz = _interval_sep(data.zmin[ci], data.zmax[ci], data.zmin[cj] + sz, data.zmax[cj] + sz)

        dist2 = dx * dx + dy * dy + dz * dz
        if dist2 < best_dist2
            best_dist2 = dist2
            best_sx = sx; best_sy = sy; best_sz = sz
        end
    end

    return best_dist2, best_sx, best_sy, best_sz
end

function _bbox_distance2_known_xy_shift(
    data::ClusterPairSoA{T},
    ci,
    cj,
    sx::T,
    sy::T,
    box_z::T,
) where {T}
    dx = _interval_sep(
        data.xmin[ci],
        data.xmax[ci],
        data.xmin[cj] + sx,
        data.xmax[cj] + sx,
    )

    dy = _interval_sep(
        data.ymin[ci],
        data.ymax[ci],
        data.ymin[cj] + sy,
        data.ymax[cj] + sy,
    )

    best_dist2 = T(Inf)
    best_sz = zero(T)

    for iz in -1:1
        sz = T(iz) * box_z

        dz = _interval_sep(
            data.zmin[ci],
            data.zmax[ci],
            data.zmin[cj] + sz,
            data.zmax[cj] + sz,
        )

        dist2 = dx * dx + dy * dy + dz * dz

        if dist2 < best_dist2
            best_dist2 = dist2
            best_sz = sz
        end
    end

    return best_dist2, sx, sy, best_sz
end


@inline function _cluster_eligible(eligible, i::Integer, j::Integer)
    eligible === nothing && return true
    return eligible[i, j] || eligible[j, i]
end

@inline function _cluster_special(special, i::Integer, j::Integer)
    special === nothing && return false
    return special[i, j] || special[j, i]
end

@inline function _cluster_lj_pair_ok(lj::SIMDLennardJones, atoms, eligible, i, j, is_special)
    _cluster_eligible(eligible, i, j) || return false
    return !shortcut_pair(lj.shortcut, atoms[i], atoms[j], is_special)
end

function _cluster_full_pair_masks(
    data::ClusterPairSoA,
    atoms,
    lj::SIMDLennardJones,
    eligible,
    special,
    ci::Integer,
    cj::Integer,
    ::Val{CW},
) where {CW}
    pair_mask = UInt64(0)
    special_mask = UInt64(0)

    first_i = (ci - 1) * CW + 1
    first_j = (cj - 1) * CW + 1

    @inbounds for lane_i in 1:CW
        slot_i = first_i + lane_i - 1
        atom_i = Int(data.slot_to_atom[slot_i])
        atom_i == 0 && continue

        for lane_j in 1:CW
            ci == cj && lane_j <= lane_i && continue

            slot_j = first_j + lane_j - 1
            atom_j = Int(data.slot_to_atom[slot_j])
            atom_j == 0 && continue

            is_special = _cluster_special(special, atom_i, atom_j)
            _cluster_lj_pair_ok(lj, atoms, eligible, atom_i, atom_j, is_special) || continue

            bit = _lane_pair_bit(lane_i, lane_j, Val(CW))
            pair_mask |= bit
            is_special && (special_mask |= bit)
        end
    end

    return pair_mask, special_mask
end

function _cluster_pruned_pair_masks(
    data::ClusterPairSoA{T},
    atoms,
    lj::SIMDLennardJones,
    eligible,
    special,
    ci::Integer,
    cj::Integer,
    cutoff2::T,
    sx::T,
    sy::T,
    sz::T,
    ::Val{CW},
) where {T, CW}
    pair_mask = UInt64(0)
    special_mask = UInt64(0)

    first_i = (ci - 1) * CW + 1
    first_j = (cj - 1) * CW + 1

    @inbounds for lane_i in 1:CW
        slot_i = first_i + lane_i - 1
        atom_i = Int(data.slot_to_atom[slot_i])
        atom_i == 0 && continue

        xi = data.x[slot_i]
        yi = data.y[slot_i]
        zi = data.z[slot_i]

        for lane_j in 1:CW
            ci == cj && lane_j <= lane_i && continue

            slot_j = first_j + lane_j - 1
            atom_j = Int(data.slot_to_atom[slot_j])
            atom_j == 0 && continue
            
            dx = xi - (data.x[slot_j] + sx)
            dy = yi - (data.y[slot_j] + sy)
            dz = zi - (data.z[slot_j] + sz)
            r2 = dx * dx + dy * dy + dz * dz
            
            r2 <= cutoff2 || continue
            
            is_special = _cluster_special(special, atom_i, atom_j)
            _cluster_lj_pair_ok(lj, atoms, eligible, atom_i, atom_j, is_special) || continue
            
            bit = _lane_pair_bit(lane_i, lane_j, Val(CW))
            pair_mask |= bit
            is_special && (special_mask |= bit)
        end
    end

    return pair_mask, special_mask
end

function _ensure_cluster_pair_chunks!(data::ClusterPairSoA{T}, n_parts::Integer) where {T}
    resize!(data.pair_i_chunks, n_parts)
    resize!(data.pair_j_chunks, n_parts)
    resize!(data.shift_x_chunks, n_parts)
    resize!(data.shift_y_chunks, n_parts)
    resize!(data.shift_z_chunks, n_parts)
    resize!(data.bbox_dist2_chunks, n_parts)
    resize!(data.pair_masks_chunks, n_parts)
    resize!(data.special_masks_chunks, n_parts)

    @inbounds for part in 1:n_parts
        if !isassigned(data.pair_i_chunks, part)
            data.pair_i_chunks[part] = Int32[]
            data.pair_j_chunks[part] = Int32[]
            data.shift_x_chunks[part] = T[]
            data.shift_y_chunks[part] = T[]
            data.shift_z_chunks[part] = T[]
            data.bbox_dist2_chunks[part] = T[]
            data.pair_masks_chunks[part] = UInt64[]
            data.special_masks_chunks[part] = UInt64[]
        end
    end

    return data
end


function _build_cluster_pairs!(
    data::ClusterPairSoA{T},
    atoms,
    boundary,
    pairlist_cutoff,
    lj::SIMDLennardJones,
    eligible,
    special,
    prune_inner_fraction,
    n_threads::Integer,
    ::Val{CW},
) where {T, CW}
    dist_unit = unit(pairlist_cutoff)
    box_x, box_y, box_z = _cluster_box_lengths(boundary, dist_unit, T)
    cutoff = T(ustrip(dist_unit, pairlist_cutoff))
    cutoff2 = cutoff * cutoff

    rB = T(prune_inner_fraction) * cutoff
    rB2 = rB * rB

    empty!(data.pair_i)
    empty!(data.pair_j)
    empty!(data.shift_x)
    empty!(data.shift_y)
    empty!(data.shift_z)
    empty!(data.bbox_dist2)
    empty!(data.pair_masks)
    empty!(data.special_masks)

    grid_dx = box_x / T(data.nx)
    grid_dy = box_y / T(data.ny)

    n_parts = max(1, min(Int(n_threads), max(data.n_clusters, 1)))
    _ensure_cluster_pair_chunks!(data, n_parts)

    @inbounds for part in 1:n_parts
        empty!(data.pair_i_chunks[part])
        empty!(data.pair_j_chunks[part])
        empty!(data.shift_x_chunks[part])
        empty!(data.shift_y_chunks[part])
        empty!(data.shift_z_chunks[part])
        empty!(data.bbox_dist2_chunks[part])
        empty!(data.pair_masks_chunks[part])
        empty!(data.special_masks_chunks[part])
    end

    Threads.@threads for part in 1:n_parts
        pair_i_local = data.pair_i_chunks[part]
        pair_j_local = data.pair_j_chunks[part]
        shift_x_local = data.shift_x_chunks[part]
        shift_y_local = data.shift_y_chunks[part]
        shift_z_local = data.shift_z_chunks[part]
        bbox_dist2_local = data.bbox_dist2_chunks[part]
        pair_masks_local = data.pair_masks_chunks[part]
        special_masks_local = data.special_masks_chunks[part]

        @inbounds for ci in part:n_parts:data.n_clusters
            mask_i = data.cluster_active_masks[ci]
            mask_i == 0 && continue

            ix_min = floor(Int, (data.xmin[ci] - cutoff) / grid_dx)
            ix_max = floor(Int, (data.xmax[ci] + cutoff) / grid_dx)
            iy_min = floor(Int, (data.ymin[ci] - cutoff) / grid_dy)
            iy_max = floor(Int, (data.ymax[ci] + cutoff) / grid_dy)

            zlo = data.zmin[ci] - cutoff
            zhi = data.zmax[ci] + cutoff
            use_direct_z_window = zlo >= zero(T) && zhi <= box_z

            for raw_iy in iy_min:iy_max
                jy = mod(raw_iy, data.ny) + 1
                sy_xy = T(fld(raw_iy, data.ny)) * box_y

                for raw_ix in ix_min:ix_max
                    jx = mod(raw_ix, data.nx) + 1
                    sx_xy = T(fld(raw_ix, data.nx)) * box_x

                    col_j = jx + (jy - 1) * data.nx

                    first_cj = Int(data.col_first_cluster[col_j])
                    last_cj = Int(data.col_last_cluster[col_j])
                    first_cj == 0 && continue

                    for cj in first_cj:last_cj
                        cj < ci && continue

                        if use_direct_z_window
                            data.zmin[cj] > zhi && break
                            data.zmax[cj] < zlo && continue
                        end

                        mask_j = data.cluster_active_masks[cj]
                        mask_j == 0 && continue

                        bbox_dist2, sx, sy, sz = _bbox_distance2_known_xy_shift(
                            data,
                            ci,
                            cj,
                            sx_xy,
                            sy_xy,
                            box_z,
                        )

                        bbox_dist2 <= cutoff2 || continue

                        pair_mask, special_mask = if bbox_dist2 <= rB2
                            _cluster_full_pair_masks(
                                data,
                                atoms,
                                lj,
                                eligible,
                                special,
                                ci,
                                cj,
                                Val(CW),
                            )
                        else
                            _cluster_pruned_pair_masks(
                                data,
                                atoms,
                                lj,
                                eligible,
                                special,
                                ci,
                                cj,
                                cutoff2,
                                sx,
                                sy,
                                sz,
                                Val(CW),
                            )
                        end

                        pair_mask == 0 && continue

                        push!(pair_i_local, Int32(ci))
                        push!(pair_j_local, Int32(cj))
                        push!(shift_x_local, sx)
                        push!(shift_y_local, sy)
                        push!(shift_z_local, sz)
                        push!(bbox_dist2_local, bbox_dist2)
                        push!(pair_masks_local, pair_mask)
                        push!(special_masks_local, special_mask)
                    end
                end
            end
        end
    end

    @inbounds for part in 1:n_parts
        append!(data.pair_i, data.pair_i_chunks[part])
        append!(data.pair_j, data.pair_j_chunks[part])
        append!(data.shift_x, data.shift_x_chunks[part])
        append!(data.shift_y, data.shift_y_chunks[part])
        append!(data.shift_z, data.shift_z_chunks[part])
        append!(data.bbox_dist2, data.bbox_dist2_chunks[part])
        append!(data.pair_masks, data.pair_masks_chunks[part])
        append!(data.special_masks, data.special_masks_chunks[part])
    end

    return data
end

function _build_cluster_csr_from_flat!(data::ClusterPairSoA{T}) where {T}
    n_clusters = data.n_clusters
    n_pairs = length(data.pair_i)

    resize!(data.ci_offsets, n_clusters + 1)
    fill!(data.ci_offsets, Int32(0))

    # First store counts in ci_offsets[ci + 1].
    @inbounds for p in 1:n_pairs
        ci = Int(data.pair_i[p])
        data.ci_offsets[ci + 1] += Int32(1)
    end

    # Convert counts to 1-based CSR offsets.
    data.ci_offsets[1] = Int32(1)

    @inbounds for ci in 1:n_clusters
        data.ci_offsets[ci + 1] += data.ci_offsets[ci]
    end

    resize!(data.cj_list, n_pairs)
    resize!(data.csr_shift_x, n_pairs)
    resize!(data.csr_shift_y, n_pairs)
    resize!(data.csr_shift_z, n_pairs)
    resize!(data.csr_bbox_dist2, n_pairs)
    resize!(data.csr_pair_masks, n_pairs)
    resize!(data.csr_special_masks, n_pairs)

    write_pos = Vector{Int32}(undef, n_clusters)

    @inbounds for ci in 1:n_clusters
        write_pos[ci] = data.ci_offsets[ci]
    end

    @inbounds for p in 1:n_pairs
        ci = Int(data.pair_i[p])
        dst = Int(write_pos[ci])
        write_pos[ci] += Int32(1)

        data.cj_list[dst] = data.pair_j[p]
        data.csr_shift_x[dst] = data.shift_x[p]
        data.csr_shift_y[dst] = data.shift_y[p]
        data.csr_shift_z[dst] = data.shift_z[p]
        data.csr_bbox_dist2[dst] = data.bbox_dist2[p]
        data.csr_pair_masks[dst] = data.pair_masks[p]
        data.csr_special_masks[dst] = data.special_masks[p]
    end

    return data
end




function _refresh_cluster_coordinates!(
    data::ClusterPairSoA{T},
    coords,
    boundary,
) where {T}
    dist_unit = unit(first(first(coords)))
    box_x, box_y, box_z = _cluster_box_lengths(boundary, dist_unit, T)

    @inbounds for slot in 1:data.n_slots
        atom_i = Int(data.slot_to_atom[slot])

        if atom_i == 0
            data.x[slot] = zero(T)
            data.y[slot] = zero(T)
            data.z[slot] = zero(T)
        else
            data.x[slot] = _cluster_wrap_coord(T(ustrip(dist_unit, coords[atom_i][1])), box_x)
            data.y[slot] = _cluster_wrap_coord(T(ustrip(dist_unit, coords[atom_i][2])), box_y)
            data.z[slot] = _cluster_wrap_coord(T(ustrip(dist_unit, coords[atom_i][3])), box_z)
        end
    end

    return data
end


function build_cluster_pair_list!(
    data::ClusterPairSoA{T},
    atoms,
    coords,
    boundary,
    pairlist_cutoff,
    lj::SIMDLennardJones,
    eligible,
    special,
    prune_inner_fraction,
    n_threads::Integer,
    ::Val{CW},
) where {T, CW}
    #t0 = time_ns()
    _build_cluster_order!(data, atoms, coords, boundary, Val(CW))
    #t1 = time_ns()
    _build_cluster_bounds!(data, Val(CW))
    #t2 = time_ns()
    _build_cluster_pairs!(data, atoms, boundary, pairlist_cutoff, lj, eligible, special, prune_inner_fraction, n_threads, Val(CW))
    #t3 = time_ns()
    _build_cluster_csr_from_flat!(data)


    # println(
    # "cluster rebuild timings: ",
    # "order_ms=", (t1 - t0) / 1e6,
    # " bounds_ms=", (t2 - t1) / 1e6,
    # " pairs_ms=", (t3 - t2) / 1e6,
    # )
    return data
end

_cluster_base_n_steps(nf) = hasproperty(nf, :n_steps) ? nf.n_steps : 1
_cluster_base_eligible(nf) = hasproperty(nf, :eligible) ? nf.eligible : nothing
_cluster_base_special(nf) = hasproperty(nf, :special) ? nf.special : nothing

function find_neighbors(
    sys::System,
    nf::ClusteredSIMDNeighborFinder,
    old_neighbors=nothing,
    step_n::Integer=0,
    force_recompute::Bool=false;
    kwargs...,
)
    lj = only(canonical_inters(nf.inter))

    old_standard = if isnothing(old_neighbors)
        nothing
    else
        old_neighbors.standard_list
    end

    new_standard = if nf.build_standard_list
        Molly.find_neighbors(
            sys,
            nf.base_finder,
            old_standard,
            step_n,
            force_recompute;
            kwargs...,
        )
    else
        isnothing(old_standard) ? ClusterOnlyStandardList() : old_standard
    end

    needs_rebuild = isnothing(old_neighbors) ||
                force_recompute ||
                iszero(step_n % _cluster_base_n_steps(nf.base_finder))

    if !needs_rebuild
        _refresh_cluster_coordinates!(
            old_neighbors.cluster_data,
            sys.coords,
            sys.boundary,
        )
        return old_neighbors
    end

    dist_unit = unit(first(first(sys.coords)))
    T_float = typeof(ustrip(dist_unit, sys.coords[1][1]))
    cluster_data = isnothing(old_neighbors) ? empty_cluster_data(T_float) : old_neighbors.cluster_data

    build_cluster_pair_list!(
        cluster_data,
        sys.atoms,
        sys.coords,
        sys.boundary,
        nf.base_finder.dist_cutoff,
        lj,
        _cluster_base_eligible(nf.base_finder),
        _cluster_base_special(nf.base_finder),
        nf.prune_inner_fraction,
        Int(get(kwargs, :n_threads, Threads.nthreads())),
        Val(cluster_width(nf)),
    )

    return ClusteredNeighborList(new_standard, cluster_data)
end

@inline function _cluster_lj_force_dr(
    lj::SIMDLennardJones,
    atoms,
    atom_i::Integer,
    atom_j::Integer,
    special::Bool,
    dx::T,
    dy::T,
    dz::T,
) where {T}
    atom_i_data = atoms[atom_i]
    atom_j_data = atoms[atom_j]

    σ = T(ustrip(σ_mixing(lj.σ_mixing, atom_i_data, atom_j_data, special)))
    ϵ = T(ustrip(ϵ_mixing(lj.ϵ_mixing, atom_i_data, atom_j_data, special)))
    weight = special ? T(lj.weight_special) : one(T)

    r2 = dx * dx + dy * dy + dz * dz
    inv_r2 = inv(r2)
    sr2 = (σ * σ) * inv_r2
    sr6 = sr2 * sr2 * sr2
    f_div_r = weight * ϵ * inv_r2 * sr6 * (T(48) * sr6 - T(24))

    return SVector(f_div_r * dx, f_div_r * dy, f_div_r * dz)
end

@inline _cluster_pair_sigma(
    ::LorentzMixing,
    atoms,
    atom_i,
    atom_j,
    special,
    sigma_i::T,
    sigma_j::T,
) where {T} = (sigma_i + sigma_j) / T(2)

@inline function _cluster_pair_sigma(
    mixing,
    atoms,
    atom_i,
    atom_j,
    special,
    sigma_i::T,
    sigma_j::T,
) where {T}
    return T(ustrip(σ_mixing(mixing, atoms[atom_i], atoms[atom_j], special)))
end

@inline _cluster_pair_epsilon(
    ::GeometricMixing,
    atoms,
    atom_i,
    atom_j,
    special,
    epsilon_i::T,
    epsilon_j::T,
) where {T} = sqrt(epsilon_i * epsilon_j)

@inline function _cluster_pair_epsilon(
    mixing,
    atoms,
    atom_i,
    atom_j,
    special,
    epsilon_i::T,
    epsilon_j::T,
) where {T}
    return T(ustrip(ϵ_mixing(mixing, atoms[atom_i], atoms[atom_j], special)))
end

@inline function _cluster_lj_force_components_packed(
    lj::SIMDLennardJones,
    atoms,
    atom_i::Integer,
    atom_j::Integer,
    special::Bool,
    sigma_i::T,
    epsilon_i::T,
    sigma_j::T,
    epsilon_j::T,
    dx::T,
    dy::T,
    dz::T,
) where {T}
    sigma = _cluster_pair_sigma(
        lj.σ_mixing,
        atoms,
        atom_i,
        atom_j,
        special,
        sigma_i,
        sigma_j,
    )

    epsilon = _cluster_pair_epsilon(
        lj.ϵ_mixing,
        atoms,
        atom_i,
        atom_j,
        special,
        epsilon_i,
        epsilon_j,
    )

    weight = special ? T(lj.weight_special) : one(T)

    r2 = dx * dx + dy * dy + dz * dz
    inv_r2 = inv(r2)
    sr2 = (sigma * sigma) * inv_r2
    sr6 = sr2 * sr2 * sr2
    f_div_r = weight * epsilon * inv_r2 * sr6 * (T(48) * sr6 - T(24))

    return f_div_r * dx, f_div_r * dy, f_div_r * dz
end

function _clustered_lj_forces_simd8_chunk!(
    fx,
    fy,
    fz,
    data::ClusterPairSoA{T},
    boundary,
    lj::SIMDLennardJones,
    first_pair::Integer,
    step_pair::Integer,
) where {T}
    VFloat = Vec{SIMD_WIDTH, T}

    box_x = T(ustrip(boundary.side_lengths[1]))
    box_y = T(ustrip(boundary.side_lengths[2]))
    box_z = T(ustrip(boundary.side_lengths[3]))
    lj_cutoff2 = T(extract_cutoff_sq(lj))
    special_weight = T(lj.weight_special)

    @inbounds for pair_idx in first_pair:step_pair:length(data.pair_i)
        ci = Int(data.pair_i[pair_idx])
        cj = Int(data.pair_j[pair_idx])
        pair_mask = data.pair_masks[pair_idx]
        special_mask = data.special_masks[pair_idx]

        first_i = (ci - 1) * SIMD_WIDTH + 1
        first_j = (cj - 1) * SIMD_WIDTH + 1

        xi = vload(VFloat, data.x, first_i)
        yi = vload(VFloat, data.y, first_i)
        zi = vload(VFloat, data.z, first_i)
        sigma_i = vload(VFloat, data.sigma, first_i)
        epsilon_i = vload(VFloat, data.epsilon, first_i)

        xj = vload(VFloat, data.x, first_j)
        yj = vload(VFloat, data.y, first_j)
        zj = vload(VFloat, data.z, first_j)
        sigma_j = vload(VFloat, data.sigma, first_j)
        epsilon_j = vload(VFloat, data.epsilon, first_j)

        fix = zero(VFloat); fiy = zero(VFloat); fiz = zero(VFloat)
        fjx = zero(VFloat); fjy = zero(VFloat); fjz = zero(VFloat)

        img_sx = data.shift_x[pair_idx]
        img_sy = data.shift_y[pair_idx]
        img_sz = data.shift_z[pair_idx]

        Base.Cartesian.@nexprs 8 shift -> begin
            s = shift - 1
            inv_s = (SIMD_WIDTH - s) % SIMD_WIDTH

            active = _cluster_pairmask_vec(pair_mask, Val(s), Val(SIMD_WIDTH))
            special = _cluster_pairmask_vec(special_mask, Val(s), Val(SIMD_WIDTH))

            xj_s = _cluster_rotate_register(xj, Val(s))
            yj_s = _cluster_rotate_register(yj, Val(s))
            zj_s = _cluster_rotate_register(zj, Val(s))
            sigma_j_s = _cluster_rotate_register(sigma_j, Val(s))
            epsilon_j_s = _cluster_rotate_register(epsilon_j, Val(s))

            # dx = xi - xj_s
            # dy = yi - yj_s
            # dz = zi - zj_s

            # dx = muladd(-box_x, round(dx / box_x), dx)
            # dy = muladd(-box_y, round(dy / box_y), dy)
            # dz = muladd(-box_z, round(dz / box_z), dz)

            dx = xi - (xj_s + img_sx)
            dy = yi - (yj_s + img_sy)
            dz = zi - (zj_s + img_sz)

            r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
            valid = active & (r2 <= lj_cutoff2)
            safe_r2 = vifelse(valid, r2, one(VFloat))

            inv_r2 = one(VFloat) / safe_r2
            sigma = (sigma_i + sigma_j_s) / T(2)
            epsilon = SIMD.sqrt(epsilon_i * epsilon_j_s)
            weight = vifelse(special, special_weight, one(VFloat))

            sr2 = (sigma * sigma) * inv_r2
            sr6 = sr2 * sr2 * sr2
            f_div_r = weight * epsilon * inv_r2 * sr6 * (T(48) * sr6 - T(24))
            f_div_r = vifelse(valid, f_div_r, zero(VFloat))

            fx_ij = f_div_r * dx
            fy_ij = f_div_r * dy
            fz_ij = f_div_r * dz

            fix += fx_ij
            fiy += fy_ij
            fiz += fz_ij

            fjx += _cluster_rotate_register(-fx_ij, Val(inv_s))
            fjy += _cluster_rotate_register(-fy_ij, Val(inv_s))
            fjz += _cluster_rotate_register(-fz_ij, Val(inv_s))
        end

        if ci == cj
            vstore(vload(VFloat, fx, first_i) + fix + fjx, fx, first_i)
            vstore(vload(VFloat, fy, first_i) + fiy + fjy, fy, first_i)
            vstore(vload(VFloat, fz, first_i) + fiz + fjz, fz, first_i)
        else
            vstore(vload(VFloat, fx, first_i) + fix, fx, first_i)
            vstore(vload(VFloat, fy, first_i) + fiy, fy, first_i)
            vstore(vload(VFloat, fz, first_i) + fiz, fz, first_i)

            vstore(vload(VFloat, fx, first_j) + fjx, fx, first_j)
            vstore(vload(VFloat, fy, first_j) + fjy, fy, first_j)
            vstore(vload(VFloat, fz, first_j) + fjz, fz, first_j)
        end
    end

    return nothing
end

function _clustered_lj_forces_packed_chunk!(
    fx,
    fy,
    fz,
    atoms,
    data::ClusterPairSoA{T},
    boundary,
    lj::SIMDLennardJones,
    first_pair::Integer,
    step_pair::Integer,
) where {T}
    CW = data.n_clusters == 0 ? 0 : data.n_slots ÷ data.n_clusters

    if CW == SIMD_WIDTH && lj.σ_mixing isa LorentzMixing && lj.ϵ_mixing isa GeometricMixing
        _clustered_lj_forces_simd8_chunk!(
            fx,
            fy,
            fz,
            data,
            boundary,
            lj,
            first_pair,
            step_pair,
        )
        return nothing
    end

    box_x = T(ustrip(boundary.side_lengths[1]))
    box_y = T(ustrip(boundary.side_lengths[2]))
    box_z = T(ustrip(boundary.side_lengths[3]))
    lj_cutoff2 = T(extract_cutoff_sq(lj))

    @inbounds for pair_idx in first_pair:step_pair:length(data.pair_i)
        ci = Int(data.pair_i[pair_idx])
        cj = Int(data.pair_j[pair_idx])
        mask = data.pair_masks[pair_idx]
        special_mask = data.special_masks[pair_idx]

        first_i = (ci - 1) * CW + 1
        first_j = (cj - 1) * CW + 1

        for lane_i in 1:CW
            slot_i = first_i + lane_i - 1
            atom_i = Int(data.slot_to_atom[slot_i])
            atom_i == 0 && continue

            xi = data.x[slot_i]; yi = data.y[slot_i]; zi = data.z[slot_i]
            sigma_i = data.sigma[slot_i]
            epsilon_i = data.epsilon[slot_i]

            for lane_j in 1:CW
                _has_lane_pair(mask, lane_i, lane_j, Val(CW)) || continue

                slot_j = first_j + lane_j - 1
                atom_j = Int(data.slot_to_atom[slot_j])
                atom_j == 0 && continue

                # dx = xi - data.x[slot_j]
                # dy = yi - data.y[slot_j]
                # dz = zi - data.z[slot_j]

                # dx = muladd(-box_x, round(dx / box_x), dx)
                # dy = muladd(-box_y, round(dy / box_y), dy)
                # dz = muladd(-box_z, round(dz / box_z), dz)

                sx = data.shift_x[pair_idx]
                sy = data.shift_y[pair_idx]
                sz = data.shift_z[pair_idx]

                dx = xi - (data.x[slot_j] + sx)
                dy = yi - (data.y[slot_j] + sy)
                dz = zi - (data.z[slot_j] + sz)

                r2 = dx * dx + dy * dy + dz * dz
                r2 <= lj_cutoff2 || continue

                special = _has_lane_pair(special_mask, lane_i, lane_j, Val(CW))
                fij_x, fij_y, fij_z = _cluster_lj_force_components_packed(
                    lj,
                    atoms,
                    atom_i,
                    atom_j,
                    special,
                    sigma_i,
                    epsilon_i,
                    data.sigma[slot_j],
                    data.epsilon[slot_j],
                    dx,
                    dy,
                    dz,
                )

                fx[slot_i] += fij_x
                fy[slot_i] += fij_y
                fz[slot_i] += fij_z

                fx[slot_j] -= fij_x
                fy[slot_j] -= fij_y
                fz[slot_j] -= fij_z
            end
        end
    end

    return nothing
end

function _clustered_lj_forces_simd8_csr_chunk!(
    fx,
    fy,
    fz,
    data::ClusterPairSoA{T},
    boundary,
    lj::SIMDLennardJones,
    first_ci::Integer,
    step_ci::Integer,
) where {T}
    VFloat = Vec{SIMD_WIDTH, T}

    lj_cutoff2 = T(extract_cutoff_sq(lj))
    special_weight = T(lj.weight_special)

    @inbounds for ci in first_ci:step_ci:data.n_clusters
        first = Int(data.ci_offsets[ci])
        last = Int(data.ci_offsets[ci + 1]) - 1
        first > last && continue

        first_i = (ci - 1) * SIMD_WIDTH + 1

        xi = vload(VFloat, data.x, first_i)
        yi = vload(VFloat, data.y, first_i)
        zi = vload(VFloat, data.z, first_i)
        sigma_i = vload(VFloat, data.sigma, first_i)
        epsilon_i = vload(VFloat, data.epsilon, first_i)

        for pair_idx in first:last
            cj = Int(data.cj_list[pair_idx])
            pair_mask = data.csr_pair_masks[pair_idx]
            special_mask = data.csr_special_masks[pair_idx]

            first_j = (cj - 1) * SIMD_WIDTH + 1

            xj = vload(VFloat, data.x, first_j)
            yj = vload(VFloat, data.y, first_j)
            zj = vload(VFloat, data.z, first_j)
            sigma_j = vload(VFloat, data.sigma, first_j)
            epsilon_j = vload(VFloat, data.epsilon, first_j)

            fix = zero(VFloat); fiy = zero(VFloat); fiz = zero(VFloat)
            fjx = zero(VFloat); fjy = zero(VFloat); fjz = zero(VFloat)

            img_sx = data.csr_shift_x[pair_idx]
            img_sy = data.csr_shift_y[pair_idx]
            img_sz = data.csr_shift_z[pair_idx]

            Base.Cartesian.@nexprs 8 shift -> begin
                s = shift - 1
                inv_s = (SIMD_WIDTH - s) % SIMD_WIDTH

                active = _cluster_pairmask_vec(pair_mask, Val(s), Val(SIMD_WIDTH))
                special = _cluster_pairmask_vec(special_mask, Val(s), Val(SIMD_WIDTH))

                xj_s = _cluster_rotate_register(xj, Val(s))
                yj_s = _cluster_rotate_register(yj, Val(s))
                zj_s = _cluster_rotate_register(zj, Val(s))
                sigma_j_s = _cluster_rotate_register(sigma_j, Val(s))
                epsilon_j_s = _cluster_rotate_register(epsilon_j, Val(s))

                dx = xi - (xj_s + img_sx)
                dy = yi - (yj_s + img_sy)
                dz = zi - (zj_s + img_sz)

                r2 = muladd(dx, dx, muladd(dy, dy, dz * dz))
                valid = active & (r2 <= lj_cutoff2)
                safe_r2 = vifelse(valid, r2, one(VFloat))

                inv_r2 = one(VFloat) / safe_r2
                sigma = (sigma_i + sigma_j_s) / T(2)
                epsilon = SIMD.sqrt(epsilon_i * epsilon_j_s)
                weight = vifelse(special, special_weight, one(VFloat))

                sr2 = (sigma * sigma) * inv_r2
                sr6 = sr2 * sr2 * sr2
                f_div_r = weight * epsilon * inv_r2 * sr6 * (T(48) * sr6 - T(24))
                f_div_r = vifelse(valid, f_div_r, zero(VFloat))

                fx_ij = f_div_r * dx
                fy_ij = f_div_r * dy
                fz_ij = f_div_r * dz

                fix += fx_ij
                fiy += fy_ij
                fiz += fz_ij

                fjx += _cluster_rotate_register(-fx_ij, Val(inv_s))
                fjy += _cluster_rotate_register(-fy_ij, Val(inv_s))
                fjz += _cluster_rotate_register(-fz_ij, Val(inv_s))
            end

            if ci == cj
                vstore(vload(VFloat, fx, first_i) + fix + fjx, fx, first_i)
                vstore(vload(VFloat, fy, first_i) + fiy + fjy, fy, first_i)
                vstore(vload(VFloat, fz, first_i) + fiz + fjz, fz, first_i)
            else
                vstore(vload(VFloat, fx, first_i) + fix, fx, first_i)
                vstore(vload(VFloat, fy, first_i) + fiy, fy, first_i)
                vstore(vload(VFloat, fz, first_i) + fiz, fz, first_i)

                vstore(vload(VFloat, fx, first_j) + fjx, fx, first_j)
                vstore(vload(VFloat, fy, first_j) + fjy, fy, first_j)
                vstore(vload(VFloat, fz, first_j) + fjz, fz, first_j)
            end
        end
    end

    return nothing
end

function _clustered_lj_forces_packed_csr_chunk!(
    fx,
    fy,
    fz,
    atoms,
    data::ClusterPairSoA{T},
    boundary,
    lj::SIMDLennardJones,
    first_ci::Integer,
    step_ci::Integer,
) where {T}
    CW = data.n_clusters == 0 ? 0 : data.n_slots ÷ data.n_clusters

    if CW == SIMD_WIDTH && lj.σ_mixing isa LorentzMixing && lj.ϵ_mixing isa GeometricMixing
        _clustered_lj_forces_simd8_csr_chunk!(
            fx,
            fy,
            fz,
            data,
            boundary,
            lj,
            first_ci,
            step_ci,
        )
        return nothing
    end

    lj_cutoff2 = T(extract_cutoff_sq(lj))

    @inbounds for ci in first_ci:step_ci:data.n_clusters
        first = Int(data.ci_offsets[ci])
        last = Int(data.ci_offsets[ci + 1]) - 1
        first > last && continue

        first_i = (ci - 1) * CW + 1

        for pair_idx in first:last
            cj = Int(data.cj_list[pair_idx])
            mask = data.csr_pair_masks[pair_idx]
            special_mask = data.csr_special_masks[pair_idx]

            sx = data.csr_shift_x[pair_idx]
            sy = data.csr_shift_y[pair_idx]
            sz = data.csr_shift_z[pair_idx]

            first_j = (cj - 1) * CW + 1

            for lane_i in 1:CW
                slot_i = first_i + lane_i - 1
                atom_i = Int(data.slot_to_atom[slot_i])
                atom_i == 0 && continue

                xi = data.x[slot_i]
                yi = data.y[slot_i]
                zi = data.z[slot_i]
                sigma_i = data.sigma[slot_i]
                epsilon_i = data.epsilon[slot_i]

                for lane_j in 1:CW
                    _has_lane_pair(mask, lane_i, lane_j, Val(CW)) || continue

                    slot_j = first_j + lane_j - 1
                    atom_j = Int(data.slot_to_atom[slot_j])
                    atom_j == 0 && continue

                    dx = xi - (data.x[slot_j] + sx)
                    dy = yi - (data.y[slot_j] + sy)
                    dz = zi - (data.z[slot_j] + sz)

                    r2 = dx * dx + dy * dy + dz * dz
                    r2 <= lj_cutoff2 || continue

                    special = _has_lane_pair(special_mask, lane_i, lane_j, Val(CW))

                    fij_x, fij_y, fij_z = _cluster_lj_force_components_packed(
                        lj,
                        atoms,
                        atom_i,
                        atom_j,
                        special,
                        sigma_i,
                        epsilon_i,
                        data.sigma[slot_j],
                        data.epsilon[slot_j],
                        dx,
                        dy,
                        dz,
                    )

                    fx[slot_i] += fij_x
                    fy[slot_i] += fij_y
                    fz[slot_i] += fij_z

                    fx[slot_j] -= fij_x
                    fy[slot_j] -= fij_y
                    fz[slot_j] -= fij_z
                end
            end
        end
    end

    return nothing
end


function _ensure_cluster_force_chunks!(data::ClusterPairSoA{T}, n_threads::Integer) where {T}
    resize!(data.fx_chunks, n_threads)
    resize!(data.fy_chunks, n_threads)
    resize!(data.fz_chunks, n_threads)

    @inbounds for thread_i in 1:n_threads
        if !isassigned(data.fx_chunks, thread_i)
            data.fx_chunks[thread_i] = Vector{T}(undef, data.n_slots)
            data.fy_chunks[thread_i] = Vector{T}(undef, data.n_slots)
            data.fz_chunks[thread_i] = Vector{T}(undef, data.n_slots)
        else
            resize!(data.fx_chunks[thread_i], data.n_slots)
            resize!(data.fy_chunks[thread_i], data.n_slots)
            resize!(data.fz_chunks[thread_i], data.n_slots)
        end
    end

    return data
end

function _scatter_cluster_forces!(fs, data::ClusterPairSoA{T}) where {T}
    @inbounds for slot in 1:data.n_slots
        atom_i = Int(data.slot_to_atom[slot])
        atom_i == 0 && continue
        fs[atom_i] += SVector(data.fx[slot], data.fy[slot], data.fz[slot])
    end

    return fs
end

function _reduce_scatter_cluster_forces!(fs, data::ClusterPairSoA{T}, n_threads::Integer) where {T}
    @inbounds for slot in 1:data.n_slots
        atom_i = Int(data.slot_to_atom[slot])
        atom_i == 0 && continue

        fx = zero(T)
        fy = zero(T)
        fz = zero(T)

        for thread_i in 1:n_threads
            fx += data.fx_chunks[thread_i][slot]
            fy += data.fy_chunks[thread_i][slot]
            fz += data.fz_chunks[thread_i][slot]
        end

        fs[atom_i] += SVector(fx, fy, fz)
    end

    return fs
end

function pairwise_forces_loop!(
    fs_nounits,
    fs_chunks,
    vir_nounits,
    vir_chunks,
    atoms,
    coords,
    velocities,
    boundary,
    neighbors::ClusteredNeighborList,
    force_units,
    n_atoms,
    pairwise_inters_nonl,
    pairwise_inters_nl::Tuple{<:SIMDLennardJones},
    ::Val{n_threads},
    ::Val{needs_vir},
    step_n=0,
) where {n_threads, needs_vir}
    isempty(pairwise_inters_nonl) ||
        throw(ArgumentError("ClusteredSIMDNeighborFinder prototype only supports neighbor-list LJ interactions"))
    needs_vir &&
        throw(ArgumentError("ClusteredSIMDNeighborFinder prototype does not implement virial yet"))

    lj = pairwise_inters_nl[1]
    data = neighbors.cluster_data

    if n_threads == 1
        fill!(fs_nounits, zero(eltype(fs_nounits)))
        fill!(data.fx, zero(eltype(data.fx)))
        fill!(data.fy, zero(eltype(data.fy)))
        fill!(data.fz, zero(eltype(data.fz)))

        # _clustered_lj_forces_packed_chunk!(
        #     data.fx,
        #     data.fy,
        #     data.fz,
        #     atoms,
        #     data,
        #     boundary,
        #     lj,
        #     1,
        #     1,
        # )

        _clustered_lj_forces_packed_csr_chunk!(
            data.fx,
            data.fy,
            data.fz,
            atoms,
            data,
            boundary,
            lj,
            1,
            1,
        )

        _scatter_cluster_forces!(fs_nounits, data)
    else
        fill!(fs_nounits, zero(eltype(fs_nounits)))
        _ensure_cluster_force_chunks!(data, n_threads)

        @inbounds for thread_i in 1:n_threads
            fill!(data.fx_chunks[thread_i], zero(eltype(data.fx)))
            fill!(data.fy_chunks[thread_i], zero(eltype(data.fy)))
            fill!(data.fz_chunks[thread_i], zero(eltype(data.fz)))
        end

        Threads.@threads for thread_i in 1:n_threads
            # _clustered_lj_forces_packed_chunk!(
            #     data.fx_chunks[thread_i],
            #     data.fy_chunks[thread_i],
            #     data.fz_chunks[thread_i],
            #     atoms,
            #     data,
            #     boundary,
            #     lj,
            #     thread_i,
            #     n_threads,
            # )

            _clustered_lj_forces_packed_csr_chunk!(
                data.fx_chunks[thread_i],
                data.fy_chunks[thread_i],
                data.fz_chunks[thread_i],
                atoms,
                data,
                boundary,
                lj,
                thread_i,
                n_threads,
            )
        end

        _reduce_scatter_cluster_forces!(fs_nounits, data, n_threads)
    end

    return fs_nounits
end

cluster_pair_count(data::ClusterPairSoA) = length(data.pair_i)
cluster_candidate_pair_count(data::ClusterPairSoA) = sum(count_ones, data.pair_masks; init=0)
cluster_dummy_fraction(data::ClusterPairSoA) =
    data.n_slots == 0 ? 0.0 : (data.n_slots - data.n_atoms) / data.n_slots

cluster_csr_pair_count(data::ClusterPairSoA) =
    isempty(data.ci_offsets) ? 0 : Int(data.ci_offsets[end] - 1)


function cluster_diagnostics(data::ClusterPairSoA)
    return (
        n_atoms = data.n_atoms,
        n_clusters = data.n_clusters,
        cluster_width = data.n_clusters == 0 ? 0 : data.n_slots ÷ data.n_clusters,
        nx = data.nx,
        ny = data.ny,
        dummy_fraction = cluster_dummy_fraction(data),
        cluster_pairs = cluster_pair_count(data),
        candidate_particle_pairs = cluster_candidate_pair_count(data),
        csr_cluster_pairs = cluster_csr_pair_count(data),

    )
end

function check_cluster_csr_consistency(data::ClusterPairSoA)
    flat_n = length(data.pair_i)
    csr_n = isempty(data.ci_offsets) ? 0 : Int(data.ci_offsets[end] - 1)

    flat_n == csr_n || return false
    length(data.cj_list) == flat_n || return false
    length(data.csr_shift_x) == flat_n || return false
    length(data.csr_shift_y) == flat_n || return false
    length(data.csr_shift_z) == flat_n || return false
    length(data.csr_bbox_dist2) == flat_n || return false
    length(data.csr_pair_masks) == flat_n || return false
    length(data.csr_special_masks) == flat_n || return false
    length(data.ci_offsets) == data.n_clusters + 1 || return false

    cursor = copy(data.ci_offsets[1:end - 1])

    @inbounds for p in 1:flat_n
        ci = Int(data.pair_i[p])
        1 <= ci <= data.n_clusters || return false

        dst = Int(cursor[ci])
        dst < Int(data.ci_offsets[ci + 1]) || return false
        cursor[ci] += Int32(1)

        data.cj_list[dst] == data.pair_j[p] || return false
        data.csr_shift_x[dst] == data.shift_x[p] || return false
        data.csr_shift_y[dst] == data.shift_y[p] || return false
        data.csr_shift_z[dst] == data.shift_z[p] || return false
        data.csr_bbox_dist2[dst] == data.bbox_dist2[p] || return false
        data.csr_pair_masks[dst] == data.pair_masks[p] || return false
        data.csr_special_masks[dst] == data.special_masks[p] || return false
    end

    @inbounds for ci in 1:data.n_clusters
        cursor[ci] == data.ci_offsets[ci + 1] || return false
    end

    return true
end



function pairwise_forces_loop!(
    fs_nounits,
    fs_chunks,
    vir_nounits,
    vir_chunks,
    atoms,
    coords,
    velocities,
    boundary,
    neighbors::ClusteredNeighborList,
    force_units,
    n_atoms,
    pairwise_inters_nonl,
    pairwise_inters_nl::Tuple{<:SIMDLennardJones},
    ::Val{1},
    ::Val{needs_vir},
    step_n=0,
) where {needs_vir}
    isempty(pairwise_inters_nonl) ||
        throw(ArgumentError("ClusteredSIMDNeighborFinder prototype only supports neighbor-list LJ interactions"))
    needs_vir &&
        throw(ArgumentError("ClusteredSIMDNeighborFinder prototype does not implement virial yet"))

    lj = pairwise_inters_nl[1]
    data = neighbors.cluster_data

    fill!(fs_nounits, zero(eltype(fs_nounits)))
    fill!(data.fx, zero(eltype(data.fx)))
    fill!(data.fy, zero(eltype(data.fy)))
    fill!(data.fz, zero(eltype(data.fz)))

    # _clustered_lj_forces_packed_chunk!(
    #     data.fx,
    #     data.fy,
    #     data.fz,
    #     atoms,
    #     data,
    #     boundary,
    #     lj,
    #     1,
    #     1,
    # )

    _clustered_lj_forces_packed_csr_chunk!(
        data.fx,
        data.fy,
        data.fz,
        atoms,
        data,
        boundary,
        lj,
        1,
        1,
    )

    _scatter_cluster_forces!(fs_nounits, data)

    return fs_nounits
end
