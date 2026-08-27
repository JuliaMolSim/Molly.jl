export FENEBond

@doc raw"""
    FENEBond(; k, r0, σ, ϵ)

A finitely extensible non-linear elastic (FENE) bond between two atoms, see
[Kremer and Grest 1990](https://doi.org/10.1063/1.458541).

The potential energy is defined as
```math
V(r) = -\frac{1}{2} k r^2_0 \ln \left( 1 - \left( \frac{r}{r_0} \right) ^2 \right) + V_{\text{WCA}}(r)
```
where the WCA contribution is given by
```math
V_{\text{WCA}}(r) =
    \begin{cases}
      4\varepsilon \left[ \left( \frac{\sigma}{r} \right) ^{12} - \left( \frac{\sigma}{r} \right) ^6 \right] + \varepsilon & r < 2^{1/6}\sigma\\
      0 & r \geq 2^{1/6}\sigma\\
    \end{cases}
```
"""
struct FENEBond{K, D, E}
    k::K
    r0::D
    σ::D
    ϵ::E
end

FENEBond(; k, r0, σ, ϵ) = FENEBond{typeof(k), typeof(r0), typeof(ϵ)}(k, r0, σ, ϵ)

function Base.zero(::Type{FENEBond{K, D, E}}) where {K, D, E}
    return FENEBond(k=zero(K), r0=zero(D), σ=zero(D), ϵ=zero(E))
end

Base.zero(b::FENEBond) = zero(typeof(b))

function Base.:+(b1::FENEBond, b2::FENEBond)
    return FENEBond(k=(b1.k + b2.k), r0=(b1.r0 + b2.r0), σ=(b1.σ + b2.σ), ϵ=(b1.ϵ + b2.ϵ))
end

function inject_interaction(inter::FENEBond, inter_type, params_dic)
    key_prefix = "inter_FB_$(inter_type)_"
    return FENEBond(
        dict_get(params_dic, key_prefix * "k" , inter.k ),
        dict_get(params_dic, key_prefix * "r0", inter.r0),
        dict_get(params_dic, key_prefix * "σ" , inter.σ ),
        dict_get(params_dic, key_prefix * "ϵ" , inter.ϵ ),
    )
end

function extract_parameters!(params_dic,
                             inter::InteractionList2Atoms{<:Any, <:AbstractVector{<:FENEBond}},
                             ff)
    for (bond_type, bond) in zip(inter.types, from_device(inter.inters))
        key_prefix = "inter_FB_$(bond_type)_"
        if !haskey(params_dic, key_prefix * "k")
            params_dic[key_prefix * "k" ] = bond.k
            params_dic[key_prefix * "r0"] = bond.r0
            params_dic[key_prefix * "σ" ] = bond.σ
            params_dic[key_prefix * "ϵ" ] = bond.ϵ
        end
    end
    return params_dic
end

@inline function force(b::FENEBond, coord_i, coord_j, boundary, args...)
    dr = vector(coord_i, coord_j, boundary)
    r = sqrt(sum(abs2, dr))
    r2 = r^2
    r2inv = inv(r2)
    r6inv = r2inv^3
    σ6 = b.σ^6
    fwca_divr = zero(b.k)
    fmag_divr = zero(fwca_divr)

    if r < (b.σ * 2 ^ (1 / 6))
        fwca_divr = 24 * b.ϵ * r2inv * (2 * (σ6 * r6inv) ^ 2 - σ6 * r6inv)
    end
    fmag_divr = fwca_divr - b.k / (1 - r2 / b.r0^2)

    f = fmag_divr * dr
    return SpecificForce2Atoms(-f, f)
end

@inline function potential_energy(b::FENEBond, coord_i, coord_j, boundary, args...)
    dr = vector(coord_i, coord_j, boundary)
    r = sqrt(sum(abs2, dr))
    r2 = r^2
    r2inv = inv(r2)
    r6inv = r2inv^3
    σ6 = b.σ^6
    r02 = b.r0^2
    uwca = zero(b.ϵ)

    if r < (b.σ * 2 ^ (1 / 6))
        uwca = 4 * b.ϵ * ((σ6 * r6inv) ^ 2 - σ6 * r6inv) + b.ϵ
    end
    return -(b.k / 2) * r02 * log(1 - r2 / r02) + uwca
end
