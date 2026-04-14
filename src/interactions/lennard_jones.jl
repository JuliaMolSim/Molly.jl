export
    LennardJones,
    LJDispersionCorrection,
    LennardJonesSoftCoreBeutler,
    LennardJonesSoftCoreGapsys,
    AshbaughHatch,
    SIMDLennardJones,
    SIMDNeighborFinder,
    SIMDCoulomb,
    PackedFlatSoA

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

##################
##################
##################

# const NATIVE_SIMD_WIDTH = if HostCPUFeatures.has_avx512f()
#     8  # Enterprise Servers (AVX-512)
# elseif HostCPUFeatures.has_avx2() || HostCPUFeatures.has_avx()
#     4  # Standard Intel/AMD (AVX/AVX2)
# else
#     2  # Apple Silicon (NEON) or legacy chips
# end

# Structs 

const SIMD_WIDTH = 8
struct PackedFlatSoA{T}
    offsets::Vector{Int}
    adj_list::Vector{Int}
    sigmas::Vector{T}
    eps::Vector{T}
    charges::Vector{T}
    weights::Vector{T}
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

use_neighbors(::SIMDLennardJones) = true  


@kwdef struct SIMDCoulomb{C, W, T} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    weight_special::W = 1 
    coulomb_const:: T = 138.93545764
end

use_neighbors(::SIMDCoulomb) = true


# --- DUCK TYPING MAGIC ---
# If Molly's loggers ask for `.n` or `.list`, secretly route it to the standard_list
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

Base.iterate(nl::PackedNeighborList, args...) = iterate(nl.standard_list, args...)
Base.length(nl::PackedNeighborList) = length(nl.standard_list)
# -------------------------


@inline Base.:*(c::SIMD.Vec, v::SVector{3}) = SVector(c * v[1], c * v[2], c * v[3])
@inline Base.:*(v::SVector{3}, c::SIMD.Vec) = SVector(c * v[1], c * v[2], c * v[3])


@inline function custom_force(inter, dr, safe_dist2, atom_i, atom_j, neigh_weights, cutoff_2)
    f_div_r = force_apply_cutoff(inter.cutoff, inter, safe_dist2, atom_i, atom_j, cutoff_2)
    f_div_r_weighted = f_div_r * neigh_weights
    return f_div_r_weighted * dr
end

@inline function force_apply_cutoff(cutoff::DistanceCutoff, inter, dist_2, atom_i, atom_j, cutoff_2)
    return core_force_div_r(inter, dist_2, atom_i, atom_j)
end

@inline function force_apply_cutoff(cutoff::NoCutoff, inter, dist_2, atom_i, atom_j, cutoff_2)
    return core_force_div_r(inter, dist_2, atom_i, atom_j)
end

@inline function core_force_div_r(inter::SIMDLennardJones, dist_2, atom_i, atom_j)
    # LJ specific mixing
    sigma_ij = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    eps_ij = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)

    inv_dist_2 = 1.0 / dist_2
    base_ratio_2 = (sigma_ij * sigma_ij) * inv_dist_2
    br2_sq = base_ratio_2 * base_ratio_2
    base_ratio_6 = br2_sq * base_ratio_2

    term = base_ratio_6 * muladd(48.0, base_ratio_6, -24.0)
    return eps_ij * inv_dist_2 * term
end

@inline function core_force_div_r(inter::SIMDCoulomb, dist_2, atom_i, atom_j)
    inv_dist_2 = 1.0 / dist_2
    inv_dist = SIMD.sqrt(inv_dist_2)
    inv_dist_3 = inv_dist_2 * inv_dist # 1.0 / r^3
    
    return inter.coulomb_const * atom_i.q * atom_j.q * inv_dist_3
    #return inter.coulomb_const * atom_i.q * atom_j.q * inv_dist_2

end

# SIMD is agnostic to the specific interaction because interaction gets passed down 
@inline function simd_force(x_i, y_i, z_i, neigh_x::V, neigh_y::V, neigh_z::V, 
    sim_params, atom_i, atom_j, inter, neigh_weights, cutoff_2) where {V <: Vec}


    #cutoff_2 = hasfield(typeof(inter.cutoff), :dist_cutoff) ? ustrip(inter.cutoff.dist_cutoff)^2 : Inf

    # NOTE: every variable in this function except the central atom i and the box dimensions, is a vector of 8 numbers

    # Boundary math - neigh_x is a vector of 8 numbers 
    dx = x_i - neigh_x
    dy = y_i - neigh_y
    dz = z_i - neigh_z

    # Calculate how many box lengths apart and wrap the neighbour to the closest image 
    dx = muladd(-sim_params.box_x, round(dx * sim_params.inv_box_x), dx)
    dy = muladd(-sim_params.box_y, round(dy * sim_params.inv_box_y), dy)
    dz = muladd(-sim_params.box_z, round(dz * sim_params.inv_box_z), dz)

    # Calculate the square of the distance 
    dist_2 = muladd(dx, dx, muladd(dy, dy, dz * dz))

    # Create 8 true or false values 
    mask = (dist_2 < cutoff_2) & (dist_2 > 0.0)
    #mask = dist_2 > 0.0

   # Swap any NaN distances to be one(V) so there is no divide-by-zero 
   # After the function, use the mask values to check if the force should be zero
   safe_dist_2 = vifelse(mask, dist_2, one(V)) 

   # Package as an svector to allow dr * force syntax 
   dr = SVector(dx, dy, dz)

   # Pass to the user defined force function which will calculate garbage forces for any dummy atoms 
   fdr_raw = custom_force(inter, dr, safe_dist_2, atom_i, atom_j, neigh_weights, cutoff_2)
   
   # Zero out forces for atoms outside the cutoff (or padded dummy atoms)
   f_x = vifelse(mask, fdr_raw[1], zero(V))
   f_y = vifelse(mask, fdr_raw[2], zero(V))
   f_z = vifelse(mask, fdr_raw[3], zero(V))

    return f_x, f_y, f_z
end




@inline function simd_chunk_forces!(fs_nounits, i, packed_data, soa_params, sim_params, coords, flat_coords, pairwise_inters_nl, ::Val{N_SIMD}) where {N_SIMD}

    VFloat = Vec{N_SIMD, Float64}
    VInt   = Vec{N_SIMD, Int}
    
    # Use the offsets to find where Atom i's data lives in the flat 1d arrays
    start_idx = packed_data.offsets[i]
    end_idx   = packed_data.offsets[i+1] - 1

    # For some reason quicker than indexing from flat_coords
    xi = ustrip(coords[i][1])
    yi = ustrip(coords[i][2])
    zi = ustrip(coords[i][3])

    # Central atom proxy
    atom_i_proxy = (σ = soa_params.σ[i], ϵ = soa_params.ϵ[i], q = soa_params.q[i])

    # Two sets of accumulators for 2x ILP
    f_ix_vec_1 = zero(VFloat); f_iy_vec_1 = zero(VFloat); f_iz_vec_1 = zero(VFloat)
    f_ix_vec_2 = zero(VFloat); f_iy_vec_2 = zero(VFloat); f_iz_vec_2 = zero(VFloat)
    
    # Loop through from 1 until length of offsets which is length of atoms minus the 1 difference added at the start - no pointer chasing
    @inbounds for j in start_idx:(2 * N_SIMD):end_idx
        
        # Load the 8 atom IDs of the different neighbors for both chunks 
        neigh_idxs_1 = vload(VInt, packed_data.adj_list, j)
        neigh_idxs_2 = vload(VInt, packed_data.adj_list, j + N_SIMD)

        # Load the sigmas and epsilons of the 8 neighbours into the vector registers for both chunks 
        neigh_sigmas_1 = vload(VFloat, packed_data.sigmas, j)
        neigh_eps_1    = vload(VFloat, packed_data.eps, j)
        neigh_charges_1 = vload(VFloat, packed_data.charges, j)
        neigh_sigmas_2 = vload(VFloat, packed_data.sigmas, j + N_SIMD)
        neigh_eps_2    = vload(VFloat, packed_data.eps, j + N_SIMD)
        neigh_charges_2 = vload(VFloat, packed_data.charges, j + N_SIMD)

        neigh_weights_1 = vload(VFloat, packed_data.weights, j)
        neigh_weights_2 = vload(VFloat, packed_data.weights, j + N_SIMD)

        #neigh_tupl = ntuple(i -> vload(...), 5)

        # Build Proxies
        atom_j_proxy_1 = (σ = neigh_sigmas_1, ϵ = neigh_eps_1, q = neigh_charges_1)
        atom_j_proxy_2 = (σ = neigh_sigmas_2, ϵ = neigh_eps_2, q = neigh_charges_2)

        # Calculate the x y and z indices using each of the atoms indexes (neigh_idx) from the SIMD chunk of the list
        idx_x_1 = neigh_idxs_1 * 3 - 2
        idx_y_1 = neigh_idxs_1 * 3 - 1
        idx_z_1 = neigh_idxs_1 * 3
        idx_x_2 = neigh_idxs_2 * 3 - 2
        idx_y_2 = neigh_idxs_2 * 3 - 1
        idx_z_2 = neigh_idxs_2 * 3
        
        # Gather the x y and z coordinates of the 8 neighboours using the generated x/y/z idxs - this gathers 8 of each coordinate
        neigh_x_1 = vgather(flat_coords, idx_x_1)
        neigh_y_1 = vgather(flat_coords, idx_y_1)
        neigh_z_1 = vgather(flat_coords, idx_z_1)
        neigh_x_2 = vgather(flat_coords, idx_x_2)
        neigh_y_2 = vgather(flat_coords, idx_y_2)
        neigh_z_2 = vgather(flat_coords, idx_z_2)
        
        # REPLACE THE FOR LOOP WITH THIS:
        f_x_1, f_y_1, f_z_1, f_x_2, f_y_2, f_z_2 = compute_all_forces(
            pairwise_inters_nl, xi, yi, zi, 
            neigh_x_1, neigh_y_1, neigh_z_1, 
            neigh_x_2, neigh_y_2, neigh_z_2, 
            sim_params, atom_i_proxy, 
            atom_j_proxy_1, atom_j_proxy_2, 
            neigh_weights_1, neigh_weights_2
        )
        
        # Add to accumulators
        f_ix_vec_1 += f_x_1; f_iy_vec_1 += f_y_1; f_iz_vec_1 += f_z_1
        f_ix_vec_2 += f_x_2; f_iy_vec_2 += f_y_2; f_iz_vec_2 += f_z_2

        # for inter in pairwise_inters_nl
        #     # Pass the scalar values for atom i and then the vectors of 8 values for the neighbours, alongside the box params

        #     cutoff_2 = ustrip(inter.cutoff.dist_cutoff)^2

        #     f_x_chunk_1, f_y_chunk_1, f_z_chunk_1 = simd_force(xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, 
        #         sim_params, atom_i_proxy, atom_j_proxy_1, inter, neigh_weights_1, cutoff_2)
            
        #     # Do the same for the second chunk
        #     f_x_chunk_2, f_y_chunk_2, f_z_chunk_2 = simd_force(xi, yi, zi, neigh_x_2, neigh_y_2, neigh_z_2, 
        #         sim_params, atom_i_proxy, atom_j_proxy_2, inter, neigh_weights_2, cutoff_2)
            
        #     # Adds the 8 force values into running totals 
        #     f_ix_vec_1 += f_x_chunk_1; f_iy_vec_1 += f_y_chunk_1; f_iz_vec_1 += f_z_chunk_1
        #     f_ix_vec_2 += f_x_chunk_2; f_iy_vec_2 += f_y_chunk_2; f_iz_vec_2 += f_z_chunk_2
        # end

        # # USE THE GENERATED UNROLLER:
        # f_x_1, f_y_1, f_z_1, f_x_2, f_y_2, f_z_2 = apply_all_interactions(
        #     pairwise_inters_nl, xi, yi, zi, 
        #     neigh_x_1, neigh_y_1, neigh_z_1, 
        #     neigh_x_2, neigh_y_2, neigh_z_2, 
        #     sim_params, atom_i_proxy, 
        #     atom_j_proxy_1, atom_j_proxy_2, 
        #     neigh_weights_1, neigh_weights_2
        # )
        
        # # Add to accumulators
        # f_ix_vec_1 += f_x_1; f_iy_vec_1 += f_y_1; f_iz_vec_1 += f_z_1
        # f_ix_vec_2 += f_x_2; f_iy_vec_2 += f_y_2; f_iz_vec_2 += f_z_2
    end
    
    # Horizontal sum across both chunks to get a single scalar force for the central atom
    f_ix = sum(f_ix_vec_1 + f_ix_vec_2)
    f_iy = sum(f_iy_vec_1 + f_iy_vec_2)
    f_iz = sum(f_iz_vec_1 + f_iz_vec_2)

    @inbounds fs_nounits[i] += SVector(f_ix, f_iy, f_iz)
end

# Recursive Step: Calculate one interaction, add it to the rest
@inline function compute_all_forces(inters::Tuple, xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, neigh_x_2, neigh_y_2, neigh_z_2, sim_params, atom_i_proxy, atom_j_proxy_1, atom_j_proxy_2, neigh_weights_1, neigh_weights_2)
    
    # Grab the exact, strictly typed interaction
    inter = first(inters)
    cutoff_2 = extract_cutoff_sq(inter.cutoff)

    # Calculate both chunks for THIS interaction
    f1x, f1y, f1z = simd_force(xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, sim_params, atom_i_proxy, atom_j_proxy_1, inter, neigh_weights_1, cutoff_2)
    f2x, f2y, f2z = simd_force(xi, yi, zi, neigh_x_2, neigh_y_2, neigh_z_2, sim_params, atom_i_proxy, atom_j_proxy_2, inter, neigh_weights_2, cutoff_2)

    # Recurse to get the forces for the remaining interactions in the tuple
    rest_f1x, rest_f1y, rest_f1z, rest_f2x, rest_f2y, rest_f2z = compute_all_forces(Base.tail(inters), xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, neigh_x_2, neigh_y_2, neigh_z_2, sim_params, atom_i_proxy, atom_j_proxy_1, atom_j_proxy_2, neigh_weights_1, neigh_weights_2)

    # Add them all together at compile time
    return f1x + rest_f1x, f1y + rest_f1y, f1z + rest_f1z, f2x + rest_f2x, f2y + rest_f2y, f2z + rest_f2z
end

# Base Case: When the tuple is empty, return zero vectors to close out the recursion!
@inline function compute_all_forces(::Tuple{}, xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, neigh_x_2, neigh_y_2, neigh_z_2, sim_params, atom_i_proxy, atom_j_proxy_1, atom_j_proxy_2, neigh_weights_1, neigh_weights_2)
    V = typeof(neigh_x_1)
    return zero(V), zero(V), zero(V), zero(V), zero(V), zero(V)
end

@inline extract_cutoff_sq(cutoff::DistanceCutoff) = ustrip(cutoff.dist_cutoff)^2
@inline extract_cutoff_sq(cutoff::NoCutoff) = Inf

# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING


# Notice the '!' - this mutates the existing packed_data
function build_packed_adj_list!(packed_data::PackedFlatSoA, atoms, molly_neighbors, N_SIMD, soa_params, my_inter)
    n_atoms = length(atoms)
    
    # Pass 1: Count neighbors (1 single flat allocation, virtually instant)
    counts = zeros(Int, n_atoms)
    for idx in 1:molly_neighbors.n
        (i, j, is_special) = molly_neighbors.list[idx]
        if !shortcut_pair(my_inter.shortcut, atoms[i], atoms[j], is_special)
            counts[i] += 1
            counts[j] += 1
        end
    end
    
    # Pass 2: Calculate padded offsets
    resize!(packed_data.offsets, n_atoms + 1)
    current_offset = 1
    chunk_size = 2 * N_SIMD
    
    @inbounds for i in 1:n_atoms
        packed_data.offsets[i] = current_offset
        c = counts[i]
        rem = c % chunk_size
        pad = rem != 0 ? chunk_size - rem : 0
        current_offset += c + pad
    end
    packed_data.offsets[n_atoms + 1] = current_offset
    total_len = current_offset - 1
    
    # Pass 3: Resize flat arrays to exactly the needed length
    resize!(packed_data.adj_list, total_len)
    resize!(packed_data.sigmas, total_len)
    resize!(packed_data.eps, total_len)
    resize!(packed_data.charges, total_len)
    resize!(packed_data.weights, total_len)
    
    insert_idx = copy(packed_data.offsets)
    
    # Pass 4: Fill the arrays directly
    for idx in 1:molly_neighbors.n
        (i, j, is_special) = molly_neighbors.list[idx]
        if shortcut_pair(my_inter.shortcut, atoms[i], atoms[j], is_special)
           continue
        end
        w = is_special ? my_inter.weight_special : 1.0
        
        pos_i = insert_idx[i]
        packed_data.adj_list[pos_i] = j
        packed_data.sigmas[pos_i] = soa_params.σ[j]
        packed_data.eps[pos_i] = soa_params.ϵ[j]
        packed_data.charges[pos_i] = soa_params.q[j]
        packed_data.weights[pos_i] = w
        insert_idx[i] += 1
        
        pos_j = insert_idx[j]
        packed_data.adj_list[pos_j] = i
        packed_data.sigmas[pos_j] = soa_params.σ[i]
        packed_data.eps[pos_j] = soa_params.ϵ[i]
        packed_data.charges[pos_j] = soa_params.q[i]
        packed_data.weights[pos_j] = w
        insert_idx[j] += 1
    end
    
    # Pass 5: Fill padding with dummy atoms (dist = 0 masks them out)
    @inbounds for i in 1:n_atoms
        start_pad = insert_idx[i]
        end_pad = packed_data.offsets[i+1] - 1
        for pos in start_pad:end_pad
            packed_data.adj_list[pos] = i 
            packed_data.sigmas[pos] = 1.0
            packed_data.eps[pos] = 1.0
            packed_data.charges[pos] = 0.0
            packed_data.weights[pos] = 1.0
        end
    end
    
    return packed_data
end

function find_neighbors(sys::System, nf::SIMDNeighborFinder, old_neighbors, step_n::Integer, force_tracking::Bool=false; kwargs...)
    
    old_standard = isnothing(old_neighbors) ? nothing : old_neighbors.standard_list
    new_standard = Molly.find_neighbors(sys, nf.base_finder, old_standard, step_n, force_tracking; kwargs...)
    
    # The correct clock check!
    needs_rebuild = isnothing(old_neighbors) || force_tracking || iszero(step_n % nf.base_finder.n_steps)
    #needs_rebuild = isnothing(old_neighbors) || (new_standard != old_standard)
    #needs_rebuild = isnothing(old_neighbors) || iszero(step_n % nf.base_finder.n_steps)


    if !needs_rebuild
        return old_neighbors
    end

    # --- THE ZERO-ALLOCATION CACHE LOGIC ---
    if isnothing(old_neighbors)
        # STEP 0: Allocate everything ONCE
        soa_params = (σ = [ustrip(a.σ) for a in sys.atoms], ϵ = [ustrip(a.ϵ) for a in sys.atoms], q = [ustrip(a.charge) for a in sys.atoms])
        packed_data = PackedFlatSoA{Float64}(Int[], Int[], Float64[], Float64[], Float64[], Float64[])
    else
        # STEP 10, 20, 30: Reuse the params and the memory from the previous step!
        soa_params = old_neighbors.soa_params
        packed_data = old_neighbors.packed_data
    end

    # Mutate the arrays in-place!
    build_packed_adj_list!(packed_data, sys.atoms, new_standard, 8, soa_params, nf.inter)

    # Wrap it back up and send it to the next step
    return PackedNeighborList(new_standard, packed_data, soa_params)
end

# Fallback for when forces() is called manually outside of the simulate loop!
function find_neighbors(sys::System, nf::SIMDNeighborFinder; kwargs...)
    # Route it to your main function, passing the cached neighbors and step 0
    return find_neighbors(sys, nf, nothing, 0, false; kwargs...)
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
    box_x = ustrip(boundary.side_lengths[1])
    box_y = ustrip(boundary.side_lengths[2])
    box_z = ustrip(boundary.side_lengths[3])
    inv_box_x = 1.0 / box_x
    inv_box_y = 1.0 / box_y
    inv_box_z = 1.0 / box_z



    # Multithreading part 
    n_t = Threads.nthreads()
    num_chunks = 8 * n_t
    n_atoms = length(coords)

    # Allocate chunks
    chunk_size = cld(n_atoms, num_chunks)
    counter = Threads.Atomic{Int}(1) # hardware counter

    # Spawn the right number of tasks per cores
    @sync for _ in 1:n_t
        Threads.@spawn begin
            
            while true
                # Get chunk then add 1 
                chunk_id = Threads.atomic_add!(counter, 1)
                
                # Break if it exceeds
                if chunk_id > num_chunks
                    break
                end
                
                start_idx = (chunk_id - 1) * chunk_size + 1
                end_idx   = min(chunk_id * chunk_size, n_atoms)
                for i in start_idx:end_idx

                    #cutoff_2 = ustrip(inter.cutoff.dist_cutoff)^2
                    #cutoff_2 = hasfield(typeof(inter.cutoff), :dist_cutoff) ? ustrip(inter.cutoff.dist_cutoff)^2 : Inf

                    # Package box metrics 
                    sim_params = (
                        box_x = box_x, box_y = box_y, box_z = box_z,
                        inv_box_x = inv_box_x, inv_box_y = inv_box_y, inv_box_z = inv_box_z
                    )

                    simd_chunk_forces!(fs_nounits, i, packed_data, soa_params, sim_params, coords, flat_coords, pairwise_inters_nl, Val(SIMD_WIDTH))
                end
            end
        end
    end

    return fs_nounits
end


#####

# @inline extract_cutoff_sq(cutoff::DistanceCutoff) = ustrip(cutoff.dist_cutoff)^2
# @inline extract_cutoff_sq(cutoff::NoCutoff) = Inf

# # Notice the @generated macro!
# @generated function apply_all_interactions(inters::Tuple, xi, yi, zi, nx1, ny1, nz1, nx2, ny2, nz2, sim_params, ai, aj1, aj2, w1, w2)

#     # fieldcount looks at the Tuple TYPE at compile time (e.g., 2 for LJ and Coulomb)
#     N = fieldcount(inters) 
    
#     # 1. Initialize the accumulators inside a 'quote' block (an Abstract Syntax Tree)
#     ex = quote
#         V = typeof(nx1)
#         f1x = zero(V); f1y = zero(V); f1z = zero(V)
#         f2x = zero(V); f2y = zero(V); f2z = zero(V)
#     end
    
#     # 2. Iterate through the interactions and write hardcoded math for each one
#     for i in 1:N
#         push!(ex.args, quote
#             # NO LOCAL VARIABLES FOR INTER! Pass it directly to guarantee strict types.
            
#             # Chunk 1 Math
#             fx1, fy1, fz1 = simd_force(xi, yi, zi, nx1, ny1, nz1, sim_params, ai, aj1, inters[$i], w1, extract_cutoff_sq(inters[$i].cutoff))
#             f1x += fx1; f1y += fy1; f1z += fz1
            
#             # Chunk 2 Math
#             fx2, fy2, fz2 = simd_force(xi, yi, zi, nx2, ny2, nz2, sim_params, ai, aj2, inters[$i], w2, extract_cutoff_sq(inters[$i].cutoff))
#             f2x += fx2; f2y += fy2; f2z += fz2
#         end)
#     end
    
#     # 3. Return the final vectors
#     push!(ex.args, :(return f1x, f1y, f1z, f2x, f2y, f2z))
    
#     return ex
# end

