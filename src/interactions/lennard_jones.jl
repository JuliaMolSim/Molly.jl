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

# struct PackedFlatSoA{T}
#     offsets::Vector{Int}
#     split_idxs::Vector{Int} 
#     adj_list::Vector{Int}
#     sigmas::Vector{T}
#     eps::Vector{T}
#     charges::Vector{T}
#     lj_weights::Vector{T}
#     coul_weights::Vector{T}
    
#     # NEW: Zero-allocation scratchpads for the builder!
#     _both_counts::Vector{Int}
#     _coul_counts::Vector{Int}
#     _both_insert::Vector{Int}
#     _coul_insert::Vector{Int}
# end

struct PackedFlatSoA{T}
    offsets::Vector{Int}
    split_idxs::Vector{Int}
    adj_list::Vector{Int}
    sigmas::Vector{T}
    eps::Vector{T}
    charges::Vector{T}
    lj_weights::Vector{T}
    coul_weights::Vector{T}

    # Thread-local scratchpads (Flattened 1D arrays to prevent false sharing)
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
    mask = (dist_2 < cutoff_2) & (dist_2 > 0.0)

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

# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING
# VERSION USING OVERLOADED NEIGHBOURS WORKING



# function find_neighbors(sys::System, nf::SIMDNeighborFinder, old_neighbors, step_n::Integer, force_tracking::Bool=false; kwargs...)
    
#     old_standard = isnothing(old_neighbors) ? nothing : old_neighbors.standard_list
#     new_standard = Molly.find_neighbors(sys, nf.base_finder, old_standard, step_n, force_tracking; kwargs...)
    
#     needs_rebuild = isnothing(old_neighbors) || force_tracking || iszero(step_n % nf.base_finder.n_steps)

#     if !needs_rebuild
#         return old_neighbors
#     end

#     if isnothing(old_neighbors)
#         soa_params = (σ = [ustrip(a.σ) for a in sys.atoms], ϵ = [ustrip(a.ϵ) for a in sys.atoms], q = [ustrip(a.charge) for a in sys.atoms])
        
#         # Allocate the 8 arrays required by the new split list struct
#         # packed_data = PackedFlatSoA{Float64}(
#         #     Int[], Int[], Int[], Float64[], Float64[], Float64[], Float64[], Float64[]
#         # )
#         packed_data = PackedFlatSoA{Float64}(
#             Int[], Int[], Int[], Float64[], Float64[], Float64[], Float64[], Float64[],
#             zeros(Int, length(sys.atoms)), zeros(Int, length(sys.atoms)), # Counts
#             Int[], Int[]                                                  # Inserts
#         )
#     else
#         soa_params = old_neighbors.soa_params
#         packed_data = old_neighbors.packed_data
#     end

#     # Assuming nf.inter is a tuple like (SIMDCoulomb(), SIMDLennardJones())
#     # We pass the specific interactions down to the builder:
#     coul_inter = nf.inter[1]
#     lj_inter = nf.inter[2] 
    
#     # Uncommented and passed explicit interactions!
#     build_packed_adj_list!(packed_data, sys.atoms, new_standard, 8, soa_params, lj_inter, coul_inter)

#     return PackedNeighborList(new_standard, packed_data, soa_params)
# end



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
    
    sim_params = (
                        box_x = box_x, box_y = box_y, box_z = box_z,
                        inv_box_x = inv_box_x, inv_box_y = inv_box_y, inv_box_z = inv_box_z
                    )

 


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
                
                   # Unpack the tuple here so simd_chunk_forces! gets strict types
                coul_inter = pairwise_inters_nl[1]
                lj_inter = pairwise_inters_nl[2]

                lj_cutoff_2 = extract_cutoff_sq(lj_inter.cutoff)
                coul_cutoff_2 = extract_cutoff_sq(coul_inter.cutoff)

                for i in start_idx:end_idx
                    

                    # Pass lj_inter and coul_inter explicitly!
                    simd_chunk_forces!(fs_nounits, i, packed_data, soa_params, sim_params, coords, flat_coords, lj_inter, coul_inter, lj_cutoff_2, coul_cutoff_2, Val(SIMD_WIDTH))
                end
            end
        end
    end

    return fs_nounits
end

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

# function build_packed_adj_list!(packed_data::PackedFlatSoA, atoms, molly_neighbours, N_SIMD, soa_params, lj_inter, coul_inter)
#     n_atoms = length(atoms)
#     chunk_size = 2 * N_SIMD

#     # both_counts = zeros(Int, n_atoms) # count neighbours in each bucket
#     # coul_counts = zeros(Int, n_atoms)
#     both_counts = fill!(packed_data._both_counts, 0)
#     coul_counts = fill!(packed_data._coul_counts, 0)

#     # Count how many neighbours each atom has in each bucket 
#     for idx in 1:molly_neighbours.n
#         i, j, is_special = molly_neighbours.list[idx] # get specific neighbour list 

#         bucket = pair_bucket(lj_inter, atoms[i], atoms[j], is_special)

#         if bucket === :both # add to both counts or coul counts
#             both_counts[i] += 1
#             both_counts[j] += 1
#         else
#             coul_counts[i] += 1
#             coul_counts[j] += 1
#         end
#     end

#     # Convert the counts of each bucket into padded offsets for both buckets 
#     resize!(packed_data.offsets, n_atoms + 1)
#     resize!(packed_data.split_idxs, n_atoms)

#     current_offset = 1 # initial offset will be the first atom 
#     @inbounds for i in 1:n_atoms
#         packed_data.offsets[i] = current_offset # set the offset for atom i 

#         # Add the number of both neighbours to the current offset 
#         c_both = both_counts[i]
#         rem_both = c_both % chunk_size
#         pad_both = rem_both == 0 ? 0 : chunk_size - rem_both
#         current_offset += c_both + pad_both 

#         packed_data.split_idxs[i] = current_offset # mark where coulomb only starts 

#         # Add the number of coulomb neighbours to current offset 
#         c_coul = coul_counts[i]
#         rem_coul = c_coul % chunk_size
#         pad_coul = rem_coul == 0 ? 0 : chunk_size - rem_coul 
#         current_offset += c_coul + pad_coul 
#     end

#     packed_data.offsets[n_atoms + 1] = current_offset
#     total_len = current_offset - 1 # for resizing 

#     # Resize arrays 
#     resize!(packed_data.adj_list, total_len)
#     resize!(packed_data.sigmas, total_len)
#     resize!(packed_data.eps, total_len)
#     resize!(packed_data.charges, total_len)
#     resize!(packed_data.lj_weights, total_len)
#     resize!(packed_data.coul_weights, total_len)

#     # Copy because we need to use the idx as a moving pointer as neighbours are added
#     # both_insert_idx = copy(packed_data.offsets)
#     # coul_insert_idx = copy(packed_data.split_idxs)

#     resize!(packed_data._both_insert, n_atoms + 1)
#     both_insert_idx = copyto!(packed_data._both_insert, packed_data.offsets)
    
#     resize!(packed_data._coul_insert, n_atoms)
#     coul_insert_idx = copyto!(packed_data._coul_insert, packed_data.split_idxs)

#     for idx in 1:molly_neighbours.n
#         i, j, is_special = molly_neighbours.list[idx]
#         bucket = pair_bucket(lj_inter, atoms[i], atoms[j], is_special)
        
#         lj_w = is_special ? lj_inter.weight_special : 1.0 # need to check if its special for the weights
#         coul_w = is_special ? coul_inter.weight_special : 1.0

#         if bucket === :both
#             # insert atom i 
#             pos_i = both_insert_idx[i]
#             packed_data.adj_list[pos_i] = j
#             packed_data.sigmas[pos_i] = soa_params.σ[j]
#             packed_data.eps[pos_i] = soa_params.ϵ[j]
#             packed_data.charges[pos_i] = soa_params.q[j]
#             packed_data.lj_weights[pos_i] = lj_w
#             packed_data.coul_weights[pos_i] = coul_w
#             both_insert_idx[i] += 1 # added a new neighbour now so must increment index

#             # insert atom j in pair 
#             pos_j = both_insert_idx[j]
#             packed_data.adj_list[pos_j] = i
#             packed_data.sigmas[pos_j] = soa_params.σ[i]
#             packed_data.eps[pos_j] = soa_params.ϵ[i]
#             packed_data.charges[pos_j] = soa_params.q[i]
#             packed_data.lj_weights[pos_j] = lj_w
#             packed_data.coul_weights[pos_j] = coul_w
#             both_insert_idx[j] += 1
#         else
#             # Insert for i (Coulomb only)
#             pos_i = coul_insert_idx[i]
#             packed_data.adj_list[pos_i] = j
#             packed_data.charges[pos_i] = soa_params.q[j]
#             packed_data.coul_weights[pos_i] = coul_w
#             coul_insert_idx[i] += 1

#             # Insert for j
#             pos_j = coul_insert_idx[j]
#             packed_data.adj_list[pos_j] = i
#             packed_data.charges[pos_j] = soa_params.q[i]
#             packed_data.coul_weights[pos_j] = coul_w
#             coul_insert_idx[j] += 1
#         end
#     end

#     # Pad both blocks with dummy atoms
#     @inbounds for i in 1:n_atoms
#         # Pad Shared Block
#         for pos in both_insert_idx[i]:(packed_data.split_idxs[i] - 1)
#             packed_data.adj_list[pos] = i
#             packed_data.sigmas[pos] = 1.0
#             packed_data.eps[pos] = 1.0
#             packed_data.charges[pos] = 0.0
#             packed_data.lj_weights[pos] = 0.0
#             packed_data.coul_weights[pos] = 0.0
#         end

#         # Pad coulomb-only block
#         for pos in coul_insert_idx[i]:(packed_data.offsets[i+1] - 1)
#             packed_data.adj_list[pos] = i
#             packed_data.charges[pos] = 0.0
#             packed_data.coul_weights[pos] = 0.0
#         end
#     end

# end
    
@inline function simd_chunk_forces!(fs_nounits, i, packed_data, soa_params, sim_params, coords, flat_coords, lj_inter, coul_inter, lj_cutoff_2, coul_cutoff_2,
    ::Val{N_SIMD}) where {N_SIMD}

    VFloat = Vec{N_SIMD, Float64}
    VInt = Vec{N_SIMD, Int}

    xi = ustrip(coords[i][1])
    yi = ustrip(coords[i][2])
    zi = ustrip(coords[i][3])

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
        coul_weights_1 = vload(VFloat, packed_data.coul_weights, j)
        coul_weights_2 = vload(VFloat, packed_data.coul_weights, j + N_SIMD)

        neigh_x_1, neigh_y_1, neigh_z_1 = gather_xyz(flat_coords, neigh_idxs_1)
        neigh_x_2, neigh_y_2, neigh_z_2 = gather_xyz(flat_coords, neigh_idxs_2)

        # 🚨 HOISTED GEOMETRY: Calculated strictly ONCE per pair!
        dr_1, dist_2_1 = simd_geometry(xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, sim_params)
        dr_2, dist_2_2 = simd_geometry(xi, yi, zi, neigh_x_2, neigh_y_2, neigh_z_2, sim_params)

        # Apply LJ
        lj_x_1, lj_y_1, lj_z_1 = simd_force_eval(dr_1, dist_2_1, atom_i_proxy, atom_j_proxy_1, lj_inter, lj_weights_1, lj_cutoff_2)
        lj_x_2, lj_y_2, lj_z_2 = simd_force_eval(dr_2, dist_2_2, atom_i_proxy, atom_j_proxy_2, lj_inter, lj_weights_2, lj_cutoff_2)

        # Apply Coulomb (REUSING THE SAME GEOMETRY!)
        coul_x_1, coul_y_1, coul_z_1 = simd_force_eval(dr_1, dist_2_1, atom_i_proxy, atom_j_proxy_1, coul_inter, coul_weights_1, coul_cutoff_2)
        coul_x_2, coul_y_2, coul_z_2 = simd_force_eval(dr_2, dist_2_2, atom_i_proxy, atom_j_proxy_2, coul_inter, coul_weights_2, coul_cutoff_2)

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

        coul_weights_1 = vload(VFloat, packed_data.coul_weights, j)
        coul_weights_2 = vload(VFloat, packed_data.coul_weights, j + N_SIMD)

        neigh_x_1, neigh_y_1, neigh_z_1 = gather_xyz(flat_coords, neigh_idxs_1)
        neigh_x_2, neigh_y_2, neigh_z_2 = gather_xyz(flat_coords, neigh_idxs_2)

        # Hoisted Geometry
        dr_1, dist_2_1 = simd_geometry(xi, yi, zi, neigh_x_1, neigh_y_1, neigh_z_1, sim_params)
        dr_2, dist_2_2 = simd_geometry(xi, yi, zi, neigh_x_2, neigh_y_2, neigh_z_2, sim_params)

        # Apply Coulomb ONLY
        coul_x_1, coul_y_1, coul_z_1 = simd_force_eval(dr_1, dist_2_1, atom_i_proxy, atom_j_proxy_1, coul_inter, coul_weights_1, coul_cutoff_2)
        coul_x_2, coul_y_2, coul_z_2 = simd_force_eval(dr_2, dist_2_2, atom_i_proxy, atom_j_proxy_2, coul_inter, coul_weights_2, coul_cutoff_2)

        f_ix_vec_1 += coul_x_1; f_iy_vec_1 += coul_y_1; f_iz_vec_1 += coul_z_1
        f_ix_vec_2 += coul_x_2; f_iy_vec_2 += coul_y_2; f_iz_vec_2 += coul_z_2
    end

    f_ix = sum(f_ix_vec_1 + f_ix_vec_2)
    f_iy = sum(f_iy_vec_1 + f_iy_vec_2)
    f_iz = sum(f_iz_vec_1 + f_iz_vec_2)

    @inbounds fs_nounits[i] += SVector(f_ix, f_iy, f_iz)

    
end


@inline _part_index(i, part, n_atoms) = i + (part - 1) * n_atoms

@inline function _part_bounds(n_items, part, n_parts)
    first_idx = ((part - 1) * n_items) ÷ n_parts + 1
    last_idx = (part * n_items) ÷ n_parts
    return first_idx, last_idx
end

@inline lj_applies(lj_inter, atom_i, atom_j, is_special) =
    !Molly.shortcut_pair(lj_inter.shortcut, atom_i, atom_j, is_special)

function build_packed_adj_list!(packed_data::PackedFlatSoA, atoms, molly_neighbours, N_SIMD, soa_params, lj_inter, coul_inter)
    n_atoms = length(atoms)
    n_pairs = molly_neighbours.n
    n_parts = Threads.nthreads()
    chunk_size = 2 * N_SIMD
    part_len = n_atoms * n_parts

    resize!(packed_data._both_counts, n_atoms)
    resize!(packed_data._coul_counts, n_atoms)
    resize!(packed_data._both_part_counts, part_len)
    resize!(packed_data._coul_part_counts, part_len)
    resize!(packed_data._both_part_pos, part_len)
    resize!(packed_data._coul_part_pos, part_len)
    resize!(packed_data._is_both, n_pairs)

    fill!(packed_data._both_counts, 0)
    fill!(packed_data._coul_counts, 0)
    fill!(packed_data._both_part_counts, 0)
    fill!(packed_data._coul_part_counts, 0)

    # Pass 1: Classify and count, thread-locally by atom.
    Threads.@threads for part in 1:n_parts
        first_idx, last_idx = _part_bounds(n_pairs, part, n_parts)

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

    # Pass 2: Reduce counts and build padded offsets.
    resize!(packed_data.offsets, n_atoms + 1)
    resize!(packed_data.split_idxs, n_atoms)

    current_offset = 1
    @inbounds for i in 1:n_atoms
        packed_data.offsets[i] = current_offset

        both_total = 0
        coul_total = 0

        for part in 1:n_parts
            both_total += packed_data._both_part_counts[_part_index(i, part, n_atoms)]
            coul_total += packed_data._coul_part_counts[_part_index(i, part, n_atoms)]
        end

        packed_data._both_counts[i] = both_total
        packed_data._coul_counts[i] = coul_total

        both_pad = both_total % chunk_size == 0 ? 0 : chunk_size - both_total % chunk_size
        current_offset += both_total + both_pad

        packed_data.split_idxs[i] = current_offset

        coul_pad = coul_total % chunk_size == 0 ? 0 : chunk_size - coul_total % chunk_size
        current_offset += coul_total + coul_pad
    end

    packed_data.offsets[n_atoms + 1] = current_offset
    total_len = current_offset - 1

    resize!(packed_data.adj_list, total_len)
    resize!(packed_data.sigmas, total_len)
    resize!(packed_data.eps, total_len)
    resize!(packed_data.charges, total_len)
    resize!(packed_data.lj_weights, total_len)
    resize!(packed_data.coul_weights, total_len)

    # Pass 3: Convert per-thread counts into per-thread write cursors.
    @inbounds for i in 1:n_atoms
        both_pos = packed_data.offsets[i]
        coul_pos = packed_data.split_idxs[i]

        for part in 1:n_parts
            k = _part_index(i, part, n_atoms)

            packed_data._both_part_pos[k] = both_pos
            both_pos += packed_data._both_part_counts[k]

            packed_data._coul_part_pos[k] = coul_pos
            coul_pos += packed_data._coul_part_counts[k]
        end
    end

    # Pass 4: Fill arrays in parallel.
    Threads.@threads for part in 1:n_parts
        first_idx, last_idx = _part_bounds(n_pairs, part, n_parts)

        @inbounds for idx in first_idx:last_idx
            i, j, is_special = molly_neighbours.list[idx]
            coul_w = is_special ? coul_inter.weight_special : 1.0

            if packed_data._is_both[idx]
                lj_w = is_special ? lj_inter.weight_special : 1.0

                pos_i = packed_data._both_part_pos[_part_index(i, part, n_atoms)]
                packed_data._both_part_pos[_part_index(i, part, n_atoms)] = pos_i + 1

                packed_data.adj_list[pos_i] = j
                packed_data.sigmas[pos_i] = soa_params.σ[j]
                packed_data.eps[pos_i] = soa_params.ϵ[j]
                packed_data.charges[pos_i] = soa_params.q[j]
                packed_data.lj_weights[pos_i] = lj_w
                packed_data.coul_weights[pos_i] = coul_w

                pos_j = packed_data._both_part_pos[_part_index(j, part, n_atoms)]
                packed_data._both_part_pos[_part_index(j, part, n_atoms)] = pos_j + 1

                packed_data.adj_list[pos_j] = i
                packed_data.sigmas[pos_j] = soa_params.σ[i]
                packed_data.eps[pos_j] = soa_params.ϵ[i]
                packed_data.charges[pos_j] = soa_params.q[i]
                packed_data.lj_weights[pos_j] = lj_w
                packed_data.coul_weights[pos_j] = coul_w
            else
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

    # Pass 5: Pad rows in parallel.
    Threads.@threads for i in 1:n_atoms
        @inbounds begin
            for pos in (packed_data.offsets[i] + packed_data._both_counts[i]):(packed_data.split_idxs[i] - 1)
                packed_data.adj_list[pos] = i
                packed_data.sigmas[pos] = 1.0
                packed_data.eps[pos] = 1.0
                packed_data.charges[pos] = 0.0
                packed_data.lj_weights[pos] = 0.0
                packed_data.coul_weights[pos] = 0.0
            end

            for pos in (packed_data.split_idxs[i] + packed_data._coul_counts[i]):(packed_data.offsets[i + 1] - 1)
                packed_data.adj_list[pos] = i
                packed_data.charges[pos] = 0.0
                packed_data.coul_weights[pos] = 0.0
            end
        end
    end

    return packed_data
end

function find_neighbors(sys::System, nf::SIMDNeighborFinder, old_neighbors, step_n::Integer, force_tracking::Bool=false; kwargs...)
    old_standard = isnothing(old_neighbors) ? nothing : old_neighbors.standard_list
    new_standard = Molly.find_neighbors(sys, nf.base_finder, old_standard, step_n, force_tracking; kwargs...)
    needs_rebuild = isnothing(old_neighbors) || force_tracking || iszero(step_n % nf.base_finder.n_steps)

    if !needs_rebuild
        return old_neighbors
    end

    if isnothing(old_neighbors)
        soa_params = (σ = [ustrip(a.σ) for a in sys.atoms], ϵ = [ustrip(a.ϵ) for a in sys.atoms], q = [ustrip(a.charge) for a in sys.atoms])
        
        # Initialize empty arrays; the builder handles all resizing
        packed_data = PackedFlatSoA{Float64}(
            Int[], Int[], Int[], Float64[], Float64[], Float64[], Float64[], Float64[],
            Int[], Int[], Int[], Int[], Int[], Int[], Bool[]
        )
    else
        soa_params = old_neighbors.soa_params
        packed_data = old_neighbors.packed_data
    end

    coul_inter = nf.inter[1]
    lj_inter = nf.inter[2] 
    
    build_packed_adj_list!(packed_data, sys.atoms, new_standard, 8, soa_params, lj_inter, coul_inter)

    return PackedNeighborList(new_standard, packed_data, soa_params)
end
