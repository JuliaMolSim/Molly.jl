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
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}^{6}}{r_{ij}^{6}}\right)^2 -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right] \vec{r}_{ij}
\end{aligned}
```

The potential energy does not include the long range dispersion correction present
in some other implementations that approximately represents contributions from
beyond the cutoff distance.
"""
@kwdef struct LennardJones{C, H, S, E, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
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
    if inter.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

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
    if inter.shortcut(atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end
    σ = inter.σ_mixing(atom_i, atom_j)
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

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
    LennardJonesSoftCoreBeutler(; λ, α, p, cutoff, use_neighbors, inter_state_a, inter_state_b, shortcut, weight_special)

    The Lennard-Jones soft core interaction as described by
    [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1). This interaction is a linear
    interpolation between two Lennard-Jones interactions A and B controlled by a parameter λ. The
    parameters `inter_state_a` and `inter_state_b` can be a named tuple of `σ_mixing` and `ϵ_mixing`
    for the interaction in the respective state or `nothing` for no interaction in that state.

    The potential energy is defined as
    ````math
    V(r)&=(1-\lambda)V^A(r_A)+λV^B(r_B) \\
    r_A&=\left(\alpha\sigma_A^6\lambda^p+r^6\right)^\frac{1}{6} \\
    r_B&=\left(\alpha\sigma_B^6(1-\lambda)^p+r^6\right)^\frac{1}{6}
    ````
    with $V^A$ and $V^B$ being the regular Lennard-Jones potentials in states $A$ and $B$ respectively.

    The force on each atom is
    ```math
    F(r)=(1-\lambda)F^A(r_A)\left(\frac{r}{r_A}\right)^5 + \lambda F^B(r_B)\left(\frac{r}{r_B}\right)^5
    ```
    with $F^A$ and $F^B$ being the regular Lennard-Jones forces in states $A$ and $B$ respectively.
"""
@kwdef struct LennardJonesSoftCoreBeutler{L, A, P, C, F, W, SA, SB} <: PairwiseInteraction
    λ::L = 1
    α::A = 1
    p::P = 1
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    inter_state_a::SA = nothing
    inter_state_b::SB = (σ_mixing = lorentz_σ_mixing, ϵ_mixing = geometric_ϵ_mixing)
    shortcut::F = lj_zero_shortcut
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreBeutler) = inter.use_neighbors

function Base.zero(inter::LennardJonesSoftCoreBeutler{L, A, P, C, F, W, SA, SB}) where {L, A, P, C, F, W, SA, SB}
    return LennardJonesSoftCoreBeutler{L, A, P, C, F, W, SA, SB}(
        zero(inter.λ),
        zero(inter.α),
        zero(inter.p),
        inter.cutoff,
        inter.use_neighbors,
        inter.inter_state_a,
        inter.inter_state_b,
        inter.shortcut,
        zero(inter.weight_special)
    )
end

function Base.:+(l1::LennardJonesSoftCoreBeutler, l2::LennardJonesSoftCoreBeutler)
    return LennardJonesSoftCoreBeutler(
        l1.λ + l2.λ,
        l1.α + l2.α,
        l1.p + l2.p,
        l1.cutoff,
        l1.use_neighbors,
        l1.inter_state_a,
        l1.inter_state_b,
        l1.shortcut,
        l1.weight_special + l2.weight_special
    )
end

const lennard_jones = LennardJones()

@inline function Molly.force(
        sc::LennardJonesSoftCoreBeutler,
        dr,
        atom_i,
        atom_j,
        force_units = u"kJ * mol^-1 * nm^-1",
        special = false,
        args...)
    
    if sc.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end

    zero_force = ustrip(zero(dr[1])) * force_units

    r = norm(dr)
    params = (sc.λ, sc.α, sc.p, sc.inter_state_a, sc.inter_state_b, atom_i, atom_j, zero_force)

    f = force_cutoff(sc.cutoff, sc, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * sc.weight_special
    else
        return fdr
    end
end

@inline function get_ra(r6, σ2, α, λ, p)
    return cbrt(sqrt(α * σ2^3 * λ^p + r6))
end

@inline function get_rb(r6, σ2, α, λ, p)
    return cbrt(sqrt(α * σ2^3 * (1 - λ)^p + r6))
end

@inline function pairwise_force(::LennardJonesSoftCoreBeutler, r,
        (λ, α, p, inter_state_a, inter_state_b, atom_i, atom_j, zero_force))
    r6 = r^6

    force_term_a, force_term_b = zero_force, zero_force

    # force for system in state A
    if !isnothing(inter_state_a)
        σ_a = inter_state_a.σ_mixing(atom_i, atom_j)
        ϵ_a = inter_state_a.ϵ_mixing(atom_i, atom_j)

        σ_a2 = σ_a^2
        r_a = get_ra(r6, σ_a2, α, λ, p)

        force_term_a = (1 - λ) * pairwise_force(lennard_jones, r_a, (σ_a2, ϵ_a)) * (r / r_a)^5
    end

    # force for system in state B
    if !isnothing(inter_state_b)
        σ_b = inter_state_b.σ_mixing(atom_i, atom_j)
        ϵ_b = inter_state_b.ϵ_mixing(atom_i, atom_j)

        σ_b2 = σ_b^2
        r_b = get_rb(r6, σ_b2, α, λ, p)

        force_term_b = λ * pairwise_force(lennard_jones, r_b, (σ_b2, ϵ_b)) * (r / r_b)^5
    end

    return force_term_a + force_term_b
end

@inline function Molly.potential_energy(
        sc::LennardJonesSoftCoreBeutler,
        dr,
        atom_i,
        atom_j,
        energy_units = u"kJ * mol^-1",
        special = false,
        args...)
    zero_energy = ustrip(zero(dr[1])) * energy_units
    if sc.shortcut(atom_i, atom_j)
        return zero_energy
    end

    r = norm(dr)
    params = (sc.λ, sc.α, sc.p, sc.inter_state_a, sc.inter_state_b, atom_i, atom_j, zero_energy)

    pe = pe_cutoff(sc.cutoff, sc, r, params)
    if special
        return pe * sc.weight_special
    else
        return pe
    end
end

@inline function pairwise_pe(
        ::LennardJonesSoftCoreBeutler, r, (
            λ, α, p, inter_state_a, inter_state_b, atom_i, atom_j, zero_energy))
    r6 = r^6

    energy_term_a, energy_term_b = zero_energy, zero_energy

    # energy for system in state A
    if !isnothing(inter_state_a)
        σ_a = inter_state_a.σ_mixing(atom_i, atom_j)
        ϵ_a = inter_state_a.ϵ_mixing(atom_i, atom_j)

        σ_a2 = σ_a^2
        r_a = get_ra(r6, σ_a2, α, λ, p)

        energy_term_a = (1 - λ) * pairwise_pe(lennard_jones, r_a, (σ_a2, ϵ_a))
    end

    # energy for system in state B
    if !isnothing(inter_state_b)
        σ_b = inter_state_b.σ_mixing(atom_i, atom_j)
        ϵ_b = inter_state_b.ϵ_mixing(atom_i, atom_j)

        σ_b2 = σ_b^2
        r_b = get_rb(r6, σ_b2, α, λ, p)

        energy_term_b = λ * pairwise_pe(lennard_jones, r_b, (σ_b2, ϵ_b))
    end

    return energy_term_a + energy_term_b
end

@inline function ∂H_∂λ(
        sc::LennardJonesSoftCoreBeutler,
        dr,
        atom_i,
        atom_j,
        energy_units = u"kJ * mol^-1",
        special = false)
    zero_energy = ustrip(zero(dr[1])) * energy_units
    if sc.shortcut(atom_i, atom_j)
        return zero_energy
    end

    r = norm(dr)
    r6 = r^6
    pα_6 = sc.p * sc.α / 6

    term_state_a, term_state_b = zero_energy, zero_energy

    # ∂V/∂λ for system in state A
    if !isnothing(sc.inter_state_a)
        σ_a = sc.inter_state_a.σ_mixing(atom_i, atom_j)
        ϵ_a = sc.inter_state_a.ϵ_mixing(atom_i, atom_j)

        σ_a2 = σ_a^2
        r_a = get_ra(r6, σ_a2, α, λ, p)

        term_state_a = pairwise_pe(lennard_jones, r_a, (σ_a2, ϵ_a)) +
                       (1 - sc.λ) * pairwise_force(lennard_jones, r_a, (σ_a2, ϵ_a)) /
                       r_a^5 * σ_a2^3 * sc.λ^(p - 1)
    end

    # ∂V/∂λ for system in state B
    if !isnothing(sc.inter_state_b)
        σ_b = sc.inter_state_b.σ_mixing(atom_i, atom_j)
        ϵ_b = sc.inter_state_b.ϵ_mixing(atom_i, atom_j)

        σ_b2 = σ_b^2
        r_b = get_rb(r6, σ_b2, α, λ, p)

        term_state_b = pairwise_pe(lennard_jones, r_b, (σ_b2, ϵ_b)) +
                       pα_6 * sc.λ * pairwise_force(lennard_jones, r_b, (σ_b2, ϵ_b)) /
                       r_b^5 * σ_b2^3 * (1 - sc.λ)^(sc.p - 1)
    end

    if special
        return (term_state_b - term_state_a) * sc.weight_special
    else
        return term_state_b - term_state_a
    end
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
\frac{C^{(12)}}{r_{ij}^{12}} - \frac{C^{(6)}}{r_{ij}^{6}}, & \text{if} & r \ge r_{LJ} \\
(\frac{78C^{(12)}}{r_{LJ}^{14}}-\frac{21C^{(6)}}{r_{LJ}^{8}})r_{ij}^2 - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}})r_{ij} + \frac{91C^{(12)}}{r_{LJ}^{12}}-\frac{28C^{(6)}}{r_{LJ}^{6}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
(\frac{12C^{(12)}}{r_{ij}^{13}} - \frac{6C^{(6)}}{r_{ij}^{7}})\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
((\frac{-156C^{(12)}}{r_{LJ}^{14}}+\frac{42C^{(6)}}{r_{LJ}^{8}})r_{ij} - (\frac{168C^{(12)}}{r_{LJ}^{13}}-\frac{48C^{(6)}}{r_{LJ}^{7}}))\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
where
```math
r_{LJ} = \alpha(\frac{26C^{(12)}(1-\lambda)}{7C^{(6)}})^{\frac{1}{6}}
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
@kwdef struct LennardJonesSoftCoreGapsys{C, A, L, H, S, E, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
end

use_neighbors(inter::LennardJonesSoftCoreGapsys) = inter.use_neighbors

function Base.zero(lj::LennardJonesSoftCoreGapsys{C, A, L, H, S, E, W}) where {C, A, L, H, S, E, W}
    return LennardJonesSoftCoreGapsys(
        lj.cutoff,
        zero(A),
        zero(L),
        lj.use_neighbors,
        lj.shortcut,
        lj.σ_mixing,
        lj.ϵ_mixing,
        zero(W),
    )
end

function Base.:+(l1::LennardJonesSoftCoreGapsys, l2::LennardJonesSoftCoreGapsys)
    return LennardJonesSoftCoreGapsys(
        l1.cutoff,
        l1.α + l2.α,
        l1.λ + l2.λ,
        l1.use_neighbors,
        l1.shortcut,
        l1.σ_mixing,
        l1.ϵ_mixing,
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
    if inter.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end
    σ6 = inter.σ_mixing(atom_i, atom_j)^6
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r = norm(dr)
    C6 = 4 * ϵ * σ6
    params = (C6 * σ6, C6)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(inter::LennardJonesSoftCoreGapsys, r, (C12, C6))
    R = inter.α*sqrt(cbrt((26*(C12/C6)*(1-inter.λ)/7)))
    r6 = r^6
    invR = inv(R)
    invR2 = invR^2
    invR6 = invR^6
    if r >= R
        return (((12*C12)/(r6*r6*r))-((6*C6)/(r6*r)))
    elseif r < R
        return (((-156*C12*(invR6*invR6*invR2)) + (42*C6*(invR2*invR6)))*r +
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
    if inter.shortcut(atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end
    σ6 = inter.σ_mixing(atom_i, atom_j)^6
    ϵ = inter.ϵ_mixing(atom_i, atom_j)

    cutoff = inter.cutoff
    r = norm(dr)
    C6 = 4 * ϵ * σ6
    params = (C6 * σ6, C6)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(inter::LennardJonesSoftCoreGapsys, r, (C12, C6))
    R = inter.α*sqrt(cbrt((26*(C12/C6)*(1-inter.λ)/7)))
    r6 = r^6
    invR = inv(R)
    invR2 = invR^2
    invR6 = invR^6
    if r >= R
        return (C12/(r6*r6))-(C6/(r6))
    elseif r < R
        return ((78*C12*(invR6*invR6*invR2)) - (21*C6*(invR2*invR6)))*(r^2) -
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
&= \frac{24\varepsilon_{ij}}{r_{ij}^2} \left[2\left(\frac{\sigma_{ij}^{6}}{r_{ij}^{6}}\right)^2 -\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]  \vec{r_{ij}}
\end{aligned}
```

If ``\lambda`` is one this gives the standard [`LennardJones`](@ref) potential.
"""
@kwdef struct AshbaughHatch{C, H, S, E, L, W} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    shortcut::H = lj_zero_shortcut
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = lorentz_ϵ_mixing
    λ_mixing::L = lorentz_λ_mixing
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
    if inter.shortcut(atom_i, atom_j)
        return ustrip.(zero(dr)) * force_units
    end

    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    σ = inter.σ_mixing(atom_i, atom_j)
    λ = inter.λ_mixing(atom_i, atom_j)

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
    if inter.shortcut(atom_i, atom_j)
        return ustrip(zero(dr[1])) * energy_units
    end
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    σ = inter.σ_mixing(atom_i, atom_j)
    λ = inter.λ_mixing(atom_i, atom_j)

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
