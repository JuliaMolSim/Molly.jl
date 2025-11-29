export
    Coulomb,
    CoulombSoftCoreBeutler,
    CoulombSoftCoreGapsys,
    CoulombReactionField,
    CoulombEwald,
    Yukawa

const coulomb_const = 138.93545764u"kJ * mol^-1 * nm" # 1 / 4πϵ0

@doc raw"""
    Coulomb(; cutoff, use_neighbors, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}}
```
"""
@kwdef struct Coulomb{C, W, T} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
end

use_neighbors(inter::Coulomb) = inter.use_neighbors

function Base.zero(coul::Coulomb{C, W, T}) where {C, W, T}
    return Coulomb(coul.cutoff, coul.use_neighbors, zero(W), zero(T))
end

function Base.:+(c1::Coulomb, c2::Coulomb)
    return Coulomb(
        c1.cutoff,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function inject_interaction(inter::Coulomb, params_dic)
    key_prefix = "inter_CO_"
    return Coulomb(
        inter.cutoff,
        inter.use_neighbors,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
    )
end

function extract_parameters!(params_dic, inter::Coulomb, ff)
    key_prefix = "inter_CO_"
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
    return params_dic
end

@inline function force(inter::Coulomb{C},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where C
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::Coulomb, r, (ke, qi, qj))
    return (ke * qi * qj) / r^2
end

@inline function potential_energy(inter::Coulomb{C},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...) where C
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::Coulomb, r, (ke, qi, qj))
    return (ke * qi * qj) * inv(r)
end

@doc raw"""
    CoulombSoftCoreBeutler(; cutoff, α, λ, use_neighbors, σ_mixing, ϵ_mixing,
                           weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1).
The potential energy is defined as
```math
V(r_{ij}) = \lambda\frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_Q^{1/6}}
```
and the force on each atom by
```math
\vec{F}_i = \lambda\frac{1}{4\pi\epsilon_0} \frac{q_iq_jr_{ij}^5}{r_Q^{7/6}}\frac{\vec{r_{ij}}}{r_{ij}}
```
where
```math
r_{Q} = \left(\frac{\alpha(1-\lambda)C^{(12)}}{C^{(6)}}\right)+r_{ij}^6
```
and
```math
C^{(12)} = 4\epsilon\sigma^{12}
C^{(6)} = 4\epsilon\sigma^{6}
```
If ``\lambda`` is 1.0, this gives the standard [`Coulomb`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct CoulombSoftCoreBeutler{C, A, L, S, E, W, T, R} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    use_neighbors::Bool = false
    σ_mixing::S = lorentz_σ_mixing
    ϵ_mixing::E = geometric_ϵ_mixing
    weight_special::W = 1
    coulomb_const::T = coulomb_const
    σ6_fac::R = (α * (1-λ))
end

use_neighbors(inter::CoulombSoftCoreBeutler) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreBeutler{C, A, L, S, E, W, T, R}) where {C, A, L, S, E, W, T, R}
    return CoulombSoftCoreBeutler(
        coul.cutoff,
        zero(A),
        zero(L),
        coul.use_neighbors,
        coul.σ_mixing,
        coul.ϵ_mixing,
        zero(W),
        zero(T),
        zero(R),
    )
end

function Base.:+(c1::CoulombSoftCoreBeutler, c2::CoulombSoftCoreBeutler)
    return CoulombSoftCoreBeutler(
        c1.cutoff,
        c1.α + c2.α,
        c1.λ + c2.λ,
        c1.use_neighbors,
        c1.σ_mixing,
        c1.ϵ_mixing,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.σ6_fac + c2.σ6_fac,
    )
end

@inline function force(inter::CoulombSoftCoreBeutler,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ6 = inter.σ_mixing(atom_i, atom_j)^6
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    C6 = 4 * ϵ * σ6
    params = (ke, qi, qj, C6 * σ6, C6, inter.σ6_fac, inter.λ)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::CoulombSoftCoreBeutler, r, (ke, qi, qj, C12, C6, σ6_fac, λ))
    r3 = r^3
    R = ((σ6_fac*(C12/C6))+(r3*r3))*sqrt(cbrt(((σ6_fac*(C12/C6))+(r3*r3))))
    return λ * ke * ((qi*qj)/R) * (r3*r*r)
end

@inline function potential_energy(inter::CoulombSoftCoreBeutler,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    σ6 = inter.σ_mixing(atom_i, atom_j)^6
    ϵ = inter.ϵ_mixing(atom_i, atom_j)
    C6 = 4 * ϵ * σ6
    params = (ke, qi, qj, C6 *σ6, C6, inter.σ6_fac, inter.λ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::CoulombSoftCoreBeutler, r, (ke, qi, qj, C12, C6, σ6_fac, λ))
    R = sqrt(cbrt((σ6_fac*(C12/C6))+r^6))
    return λ * ke * ((qi * qj)/R)
end

@doc raw"""
    CoulombSoftCoreGapsys(; cutoff, α, λ, σQ, use_neighbors, weight_special, coulomb_const)

The Coulomb electrostatic interaction between two atoms with a soft core, used for
the appearing and disappearing of atoms.

See [Gapsys et al. 2012](https://doi.org/10.1021/ct300220p).
The potential energy is defined as
```math
V(r_{ij}) = \left\{ \begin{array}{cl}
\frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\frac{1}{4\pi\epsilon_0} (\frac{q_iq_j}{r_{Q}^3}r_{ij}^2-\frac{3q_iq_j}{r_{Q}^2}r_{ij}+\frac{3q_iq_j}{r_{Q}}), & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
and the force on each atom by
```math
\vec{F}_i = \left\{ \begin{array}{cl}
\frac{1}{4\pi\epsilon_0} \frac{q_iq_j}{r_{ij}^2}\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \ge r_{LJ} \\
\frac{1}{4\pi\epsilon_0} (\frac{-2q_iq_j}{r_{Q}^3}r_{ij}+\frac{3q_iq_j}{r_{Q}^2})\frac{\vec{r_{ij}}}{r_{ij}}, & \text{if} & r \lt r_{LJ} \\
\end{array} \right.
```
where
```math
r_{Q} = \alpha(1-\lambda)^{1/6}(1+σ_Q|qi*qj|)
```

If ``\lambda`` is 1.0, this gives the standard [`Coulomb`](@ref) potential and means
the atom is fully turned on.
If ``\lambda`` is zero the interaction is turned off.
``\alpha`` determines the strength of softening the function.
"""
@kwdef struct CoulombSoftCoreGapsys{C, A, L, S, W, T, R} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    α::A = 1
    λ::L = 0
    σQ::S = 1.0
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
    σ6_fac::R = (α * sqrt(cbrt(1-λ)))
end

use_neighbors(inter::CoulombSoftCoreGapsys) = inter.use_neighbors

function Base.zero(coul::CoulombSoftCoreGapsys{C, A, L, S, W, T, R}) where {C, A, L, S, W, T, R}
    return CoulombSoftCoreGapsys(
        coul.cutoff,
        zero(A),
        zero(L),
        zero(Q),
        coul.use_neighbors,
        zero(W),
        zero(T),
        zero(R),
    )
end

function Base.:+(c1::CoulombSoftCoreGapsys, c2::CoulombSoftCoreGapsys)
    return CoulombSoftCoreGapsys(
        c1.cutoff,
        c1.α + c2.α,
        c1.λ + c2.λ,
        c1.σQ + c2.σQ,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.σ6_fac + c2.σ6_fac,
    )
end

@inline function force(inter::CoulombSoftCoreGapsys,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj, inter.σQ, inter.σ6_fac, inter.λ)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::CoulombSoftCoreGapsys, r, (ke, qi, qj, σQ, σ6_fac, λ))
    qij = qi * qj
    R = σ6_fac*(oneunit(r)+(σQ*abs(qij)))
    if r >= R
        return λ * ke * (qij/(r^2))
    elseif r < R
        return λ * ke * (-(((2*qij)/(R^3)) * r) + ((3*qij)/(R^2)))
    end
end

@inline function potential_energy(inter::CoulombSoftCoreGapsys,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r = norm(dr)
    cutoff = inter.cutoff
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (ke, qi, qj, inter.σQ, inter.σ6_fac, inter.λ)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::CoulombSoftCoreGapsys, r, (ke, qi, qj, σQ, σ6_fac, λ))
    qij = qi * qj
    R = σ6_fac*(oneunit(r)+(σQ*abs(qij)))
    if r >= R
        return λ * ke * (qij/r)
    elseif r < R
        return λ * ke * (((qij/(R^3))*(r^2))-(((3*qij)/(R^2))*r)+((3*qij)/R))
    end
end

const crf_solvent_dielectric = 78.3

"""
    CoulombReactionField(; dist_cutoff, solvent_dielectric, use_neighbors, weight_special,
                            coulomb_const)

The Coulomb electrostatic interaction modified using the reaction field approximation
between two atoms.
"""
@kwdef struct CoulombReactionField{D, S, W, T} <: PairwiseInteraction
    dist_cutoff::D
    solvent_dielectric::S = crf_solvent_dielectric
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
end

use_neighbors(inter::CoulombReactionField) = inter.use_neighbors

function Base.zero(coul::CoulombReactionField{D, S, W, T}) where {D, S, W, T}
    return CoulombReactionField{D, S, W, T}(
        zero(D),
        zero(S),
        coul.use_neighbors,
        zero(W),
        zero(T),
    )
end

function Base.:+(c1::CoulombReactionField, c2::CoulombReactionField)
    return CoulombReactionField(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.solvent_dielectric + c2.solvent_dielectric,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
    )
end

function inject_interaction(inter::CoulombReactionField, params_dic)
    key_prefix = "inter_CRF_"
    return CoulombReactionField(
        dict_get(params_dic, key_prefix * "dist_cutoff", inter.dist_cutoff),
        dict_get(params_dic, key_prefix * "solvent_dielectric", inter.solvent_dielectric),
        inter.use_neighbors,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
    )
end

function extract_parameters!(params_dic, inter::CoulombReactionField, ff)
    key_prefix = "inter_CRF_"
    params_dic[key_prefix * "dist_cutoff"] = inter.dist_cutoff
    params_dic[key_prefix * "solvent_dielectric"] = inter.solvent_dielectric
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
    return params_dic
end

@inline function force(inter::CoulombReactionField,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r2 = sum(abs2, dr)
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)
    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = (1 / (inter.dist_cutoff ^ 3)) * 0
    else
        # These values could be pre-computed but this way is easier for AD
        krf = (1 / (inter.dist_cutoff ^ 3)) * ((inter.solvent_dielectric - 1) /
              (2 * inter.solvent_dielectric + 1))
    end

    f = (ke * qi * qj) * (inv(r) - 2 * krf * r2) * inv(r2)

    if special
        return f * dr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return f * dr * (r <= inter.dist_cutoff)
    end
end

@inline function potential_energy(inter::CoulombReactionField,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    ke = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)
    if special
        # 1-4 interactions do not use the reaction field approximation
        krf = (1 / (inter.dist_cutoff ^ 3)) * 0
        crf = (1 /  inter.dist_cutoff     ) * 0
    else
        krf = (1 / (inter.dist_cutoff ^ 3)) * ((inter.solvent_dielectric - 1) /
              (2 * inter.solvent_dielectric + 1))
        crf = (1 /  inter.dist_cutoff     ) * ((3 * inter.solvent_dielectric) /
              (2 * inter.solvent_dielectric + 1))
    end

    pe = (ke * qi * qj) * (inv(r) + krf * r2 - crf)

    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * (r <= inter.dist_cutoff)
    end
end

"""
    CoulombEwald(; dist_cutoff, error_tol=0.0005, use_neighbors=false, weight_special=1,
                 coulomb_const=coulomb_const, approximate_erfc=true)

The short range Ewald electrostatic interaction between two atoms.

Should be used alongside the [`Ewald`](@ref) or [`PME`](@ref) general interaction,
which provide the long-range term.
`dist_cutoff` and `error_tol` should match the general interaction.

`dist_cutoff` is the cutoff distance for short range interactions.
`approximate_erfc` determines whether to use a fast approximation to the erfc function.
"""
struct CoulombEwald{T, D, W, C, A} <: PairwiseInteraction
    dist_cutoff::D
    error_tol::T
    use_neighbors::Bool
    weight_special::W
    coulomb_const::C
    α::A
    approximate_erfc::Bool
end

function CoulombEwald(; dist_cutoff, error_tol=0.0005, use_neighbors=false,
                      weight_special=1, coulomb_const=coulomb_const, approximate_erfc=true)
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    return CoulombEwald(dist_cutoff, error_tol, use_neighbors, weight_special, coulomb_const,
                        α, approximate_erfc)
end

use_neighbors(inter::CoulombEwald) = inter.use_neighbors

function Base.zero(coul::CoulombEwald{T, D, W, C, A}) where {T, D, W, C, A}
    return CoulombEwald(
        zero(D),
        zero(T),
        coul.use_neighbors,
        zero(W),
        zero(C),
        zero(A),
        coul.approximate_erfc,
    )
end

function Base.:+(c1::CoulombEwald, c2::CoulombEwald)
    return CoulombEwald(
        c1.dist_cutoff + c2.dist_cutoff,
        c1.error_tol + c2.error_tol,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.α + c2.α,
        c1.approximate_erfc,
    )
end

function inject_interaction(inter::CoulombEwald, params_dic)
    key_prefix = "inter_CE_"
    return CoulombEwald(
        dict_get(params_dic, key_prefix * "dist_cutoff", inter.dist_cutoff),
        inter.error_tol,
        inter.use_neighbors,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_special),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
        inter.α,
        inter.approximate_erfc,
    )
end

function extract_parameters!(params_dic, inter::CoulombEwald, ff)
    key_prefix = "inter_CE_"
    params_dic[key_prefix * "dist_cutoff"] = inter.dist_cutoff
    params_dic[key_prefix * "weight_14"] = inter.weight_special
    params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
    return params_dic
end

function calc_erfc(αr::T, exp_mαr2, approximate_erfc) where T
    if approximate_erfc
        # See the OpenMM source code, Abramowitz and Stegun 1964, and Hastings 1995
        t = inv(one(T) + T(0.3275911) * αr)
        return (T(0.254829592)+(T(-0.284496736)+(T(1.421413741)+
                                    (T(-1.453152027)+T(1.061405429)*t)*t)*t)*t)*t*exp_mαr2
    else
        return erfc(αr)
    end
end

@inline function force(inter::CoulombEwald{T},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...) where T
    r2 = sum(abs2, dr)
    ke, α = inter.coulomb_const, inter.α
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)
    inv_r = inv(r)
    αr = α * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    f = ke * qi * qj * inv_r^3
    if special
        # Special interactions excluded from reciprocal calculation
        # so have a standard interaction
        return f * dr * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return f * dr * (erfc_αr + 2 * αr * exp_mαr2 / sqrt(T(π))) * (r <= inter.dist_cutoff)
    end
end

@inline function potential_energy(inter::CoulombEwald,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special=false,
                                  args...)
    r2 = sum(abs2, dr)
    ke, α = inter.coulomb_const, inter.α
    qi, qj = atom_i.charge, atom_j.charge
    r = sqrt(r2)
    inv_r = inv(r)
    αr = α * r
    exp_mαr2 = exp(-αr^2)
    erfc_αr = calc_erfc(αr, exp_mαr2, inter.approximate_erfc)
    pe = ke * qi * qj * inv_r
    if special
        return pe * inter.weight_special * (r <= inter.dist_cutoff)
    else
        return pe * erfc_αr * (r <= inter.dist_cutoff)
    end
end

@doc raw"""
    Yukawa(; cutoff, use_neighbors, weight_special, coulomb_const, kappa)

The Yukawa electrostatic interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}} \exp(-\kappa r_{ij})
```
and the force on each atom by
```math
F(r_{ij}) = \frac{q_i q_j}{4 \pi \varepsilon_0 r_{ij}^2} \exp(-\kappa r_{ij})\left(\kappa r_{ij} + 1\right) \vec{r}_{ij}
```
"""
@kwdef struct Yukawa{C, W, T, K} <: PairwiseInteraction
    cutoff::C = NoCutoff()
    use_neighbors::Bool = false
    weight_special::W = 1
    coulomb_const::T = coulomb_const
    kappa::K = 1.0*u"nm^-1"
end

use_neighbors(inter::Yukawa) = inter.use_neighbors

function Base.zero(yukawa::Yukawa{C, W, T, K}) where {C, W, T, K}
    return Yukawa(
        yukawa.cutoff,
        yukawa.use_neighbors,
        zero(W),
        zero(T),
        zero(K),
    )
end

function Base.:+(c1::Yukawa, c2::Yukawa)
    return Yukawa(
        c1.cutoff,
        c1.use_neighbors,
        c1.weight_special + c2.weight_special,
        c1.coulomb_const + c2.coulomb_const,
        c1.kappa + c2.kappa,
    )
end

@inline function force(inter::Yukawa,
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special=false,
                       args...)
    r = norm(dr)
    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    kappa = inter.kappa
    params = (coulomb_const, qi, qj, kappa)

    f = force_cutoff(cutoff, inter, r, params)
    fdr = (f / r) * dr
    if special
        return fdr * inter.weight_special
    else
        return fdr
    end
end

function pairwise_force(::Yukawa, r, (coulomb_const, qi, qj, kappa))
    return (coulomb_const * qi * qj) * exp(-kappa * r) * (kappa * r + 1) / r^2
end

@inline function potential_energy(inter::Yukawa,
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special::Bool=false,
                                  args...)
    r = norm(dr)
    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = atom_i.charge, atom_j.charge
    params = (coulomb_const, qi, qj, inter.kappa)

    pe = pe_cutoff(cutoff, inter, r, params)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function pairwise_pe(::Yukawa, r, (coulomb_const, qi, qj, kappa))
    return (coulomb_const * qi * qj) * inv(r) * exp(-kappa * r)
end
