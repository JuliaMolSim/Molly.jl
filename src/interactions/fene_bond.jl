export FENEBond

@doc raw"""
    FENEBond(; k, r0, ϵ, σ)

A finitely extensible nonlinear elastic (FENE) bond between two atoms ([Kremer, Grest, J Chem Phys, 92, 5057 (1990)](https://doi.org/10.1063/1.458541)). The potential energy is given by

```math
U(r) = -\frac{1}{2} k r^2_0 \ln (1 - (\frac{r}{r_0})^2) + U_{\text{WCA}}(r)
```
where the WCA contribution is given by
```math
U_{\text{WCA}}(r) =
    \begin{cases}
      4\epsilon \big[ (\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^6 \big] + \epsilon & r < 2^{1/6}\sigma\\
      0 & r \geq 2^{1/6}\sigma\\
    \end{cases}       
```
"""
struct FENEBond{D, K, E} <: SpecificInteraction
    k::K
    r0::D
    ϵ::E
    σ::D
end

FENEBond(; k, r0, ϵ, σ) = FENEBond{typeof(r0), typeof(k), typeof(ϵ)}(k, r0, ϵ, σ)

@inline @inbounds function force(b::FENEBond, coord_i, coord_j, boundary)
    ab = vector(coord_i, coord_j, boundary)
    r = norm(ab)
    r2 = r^2
    r2inv = r2^-1#1 / r2
    r6inv = r2inv^3
    σ6 = b.σ^6
    fwca_divr = zero(b.k)
    fmag_divr = zero(fwca_divr)

    if r < (b.σ * 2^(1/6))
        fwca_divr = 24 * b.ϵ * r2inv * ( 2 * (σ6 * r6inv)^2 - σ6 * r6inv)
    end
    fmag_divr = fwca_divr - b.k / (1 - r2 / b.r0^2)
    
    f = fmag_divr * ab
    return SpecificForce2Atoms(-f, +f)
end

@inline @inbounds function potential_energy(b::FENEBond, coord_i, coord_j, boundary)
    dr = vector(coord_i, coord_j, boundary)
    r = norm(dr)
    r2 = r^2
    r6inv = r2inv^3
    r02 = b.r0^2
    

    uwca = zero(b.ϵ)#0.0
    if r < b.σ * 2^(1/6)
        uwca = 4 * b.ϵ * ((σ6 * r6inv)^2 - σ6*r6inv) + b.ϵ
    end
    return (b.k / 2) * r02 * log(1 - r2 / r02) + uwca
end
