export
    LennardJones,
    LennardJonesSoftCoreBeutler,
    LennardJonesSoftCoreGapsys,
    AshbaughHatch

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

The potential energy does not include the long range dispersion correction present
in some other implementations that approximately represents contributions from
beyond the cutoff distance.
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
@kwdef struct LennardJonesSoftCoreBeutler{C, A, H, S, E, LM, SCH, R, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 0.85
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    roles::R = AlchemicalRole[]
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreBeutler) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreBeutler{C, A, H, S, E, LM, SCH, R, W}) where {C, A, H, S, E, LM, SCH, R, W}
    return LennardJonesSoftCoreBeutler(
        lj.cutoff,
        zero(A),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        lj.scheduler,
        lj.roles,
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
        l1.roles,
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
    role_i = inter.roles[atom_i.index]
    role_j = inter.roles[atom_j.index]
    pair_role = mix_roles(role_i, role_j)

    # 2. Dispatch to the scheduler for the effective sterics lambda
    λ = T(scale_elec(inter.scheduler, λ_glob, pair_role))

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end

    # If lambda is 1, the soft core formula reduces to standard LJ
    # We explicity branch to save compute.
    if λ >= 1.0

        σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
        ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
        r = norm(dr)
        σ2 = σ^2
        params = (σ2, ϵ, nothing, nothing)
        
        # Call standard LJ cutoff logic.
        f = force_cutoff(inter.cutoff, inter, r, params) 
        fdr = (f / r) * dr
        
        return special ? fdr * inter.weight_special : fdr
    end

    
    σ6 = σ_mixing(inter.σ_mixing, atom_i, atom_j)^6
    ϵ  = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)

    r = norm(dr)
    C6 = 4 * ϵ * σ6
    C12 = C6 * σ6
    σ6_fac = inter.α * (1 - λ)
    params = (C12, C6, λ, σ6_fac)

    f = force_cutoff(inter.cutoff, inter, r, params)
    fdr = (f / r) * dr
    
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard LJ Logic
@inline function pairwise_force(::LennardJonesSoftCoreBeutler, r, (σ2, ϵ, _, _)::Tuple{<:Quantity, <:Quantity, Nothing, Nothing})
    six_term = (σ2 / r^2)^3
    return (24 * ϵ / r) * (2 * six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic
function pairwise_force(::LennardJonesSoftCoreBeutler, r, (C12, C6, λ, σ6_fac)::Tuple{<:Quantity, <:Quantity, <:Real, <:Real})
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
    role_i = inter.roles[atom_i.index]
    role_j = inter.roles[atom_j.index]
    pair_role = mix_roles(role_i, role_j)

    # 2. Dispatch to the scheduler for the effective sterics lambda
    λ = T(scale_elec(inter.scheduler, λ_glob, pair_role))

    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end


    # If lambda is 1, the soft core formula reduces to standard LJ
    # We explicity branch to save compute.
    if λ >= 1.0

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
@inline function pairwise_pe(::LennardJonesSoftCoreBeutler, r, (σ2, ϵ, _, _)::Tuple{<:Quantity, <:Quantity, Nothing, Nothing})
    inv_r2 = inv(r^2)
    six_term = (σ2 * inv_r2)^3
    return 4 * ϵ * (six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
function pairwise_pe(::LennardJonesSoftCoreBeutler, r, (C12, C6, σ6_fac, λ))
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
@kwdef struct LennardJonesSoftCoreGapsys{C, A, H, S, E, LM, SCH, R, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 0.85
    use_neighbors::Bool = false
    shortcut::H = LJZeroShortcut()
    σ_mixing::S = LorentzMixing()
    ϵ_mixing::E = GeometricMixing()
    λ_mixing::LM = MinimumMixing()
    scheduler::SCH = DefaultLambdaScheduler()
    roles::R = AlchemicalRole[]
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreGapsys) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreGapsys{C, A, H, S, E, LM, SCH, R, W}) where {C, A, H, S, E, LM, SCH, R, W}
    return LennardJonesSoftCoreGapsys(
        lj.cutoff,
        zero(A),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        lj.λ_mixing,
        lj.scheduler,
        lj.roles,
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
        l1.roles,
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
    role_i = inter.roles[atom_i.index]
    role_j = inter.roles[atom_j.index]
    pair_role = mix_roles(role_i, role_j)

    # 2. Dispatch to the scheduler for the effective sterics lambda
    λ = T(scale_elec(inter.scheduler, λ_glob, pair_role))
    
    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end

    cutoff = inter.cutoff
    r = norm(dr)
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    # 3. Fast Path: Standard Lennard Jones
    if λ >= 1.0
        # Pass standard LJ params tuple (Length 2)
        params = (σ^2, ϵ, nothing, nothing)
        f = force_cutoff(cutoff, inter, r, params)
        fdr = (f / r) * dr
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
    fdr = (f / r) * dr
    return special ? fdr * inter.weight_special : fdr
end

# Dispatch 1: Standard LJ Logic (Matches Tuple length 2)
@inline function pairwise_force(::LennardJonesSoftCoreGapsys, r, (σ2, ϵ, _, _)::Tuple{<:Quantity, <:Quantity, Nothing, Nothing})
    six_term = (σ2 / r^2)^3
    return (24 * ϵ / r) * (2 * six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
@inline function pairwise_force(::LennardJonesSoftCoreGapsys, r, (C12, C6, λ, R)::Tuple{<:Quantity, <:Quantity, <:Real, <:Quantity})
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
    λ = T(λ_mixing(inter.λ_mixing, atom_i, atom_j))
    if shortcut_pair(inter.shortcut, atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end

    cutoff = inter.cutoff
    r = norm(dr)
    σ = σ_mixing(inter.σ_mixing, atom_i, atom_j)
    ϵ = ϵ_mixing(inter.ϵ_mixing, atom_i, atom_j)
    σ6 = σ^6

    # 3. Fast Path: Standard Lennard Jones
    if λ >= 1.0
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
@inline function pairwise_pe(::LennardJonesSoftCoreGapsys, r, (σ2, ϵ, _, _)::Tuple{<:Quantity, <:Quantity, Nothing, Nothing})
    inv_r2 = inv(r^2)
    six_term = (σ2 * inv_r2)^3
    return 4 * ϵ * (six_term^2 - six_term)
end

# Dispatch 2: Soft Core Logic (Matches Tuple length 4)
@inline function pairwise_pe(::LennardJonesSoftCoreGapsys, r, (C12, C6, λ, R)::Tuple{<:Quantity, <:Quantity, <:Real, <:Quantity})
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
# Assumes no 1-4 Lennard-Jones interaction via the pairwise forces (weight_special = 0)
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

@inline function force(inter::LennardJones14, coords_i, coords_j, coords_k,
                       coords_l, boundary, args...)
    σ2 = inter.σ14_mixed ^ 2
    dr = vector(coords_i, coords_l, boundary)
    r2 = sum(abs2, dr)
    six_term = (σ2 / r2) ^ 3
    fl = inter.weight_14 * (24 * inter.ϵ14_mixed / r2) * (2 * six_term ^ 2 - six_term) * dr
    fi = -fl
    fj, fk = zero(fl), zero(fl)
    return SpecificForce4Atoms(fi, fj, fk, fl)
end

@inline function potential_energy(inter::LennardJones14, coords_i, coords_j, coords_k,
                                  coords_l, boundary, args...)
    σ2 = inter.σ14_mixed ^ 2
    r2 = sum(abs2, vector(coords_i, coords_l, boundary))
    six_term = (σ2 / r2) ^ 3
    return inter.weight_14 * 4 * inter.ϵ14_mixed * (six_term ^ 2 - six_term)
end
