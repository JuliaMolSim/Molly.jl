export Buckingham

@doc raw"""
    Buckingham(; cutoff, use_neighbors, weight_special)

The Buckingham interaction between two atoms.

The potential energy is defined as
```math
V(r_{ij}) = A_{ij} \exp(-B_{ij} r_{ij}) - \frac{C_{ij}}{r_{ij}^6}
```
and the force on each atom by
```math
\vec{F}_i = \left( A_{ij} B_{ij} \exp(-B_{ij} r_{ij}) - 6 \frac{C_{ij}}{r_{ij}^7} \right) \frac{\vec{r}_{ij}}{r_{ij}}
```
The parameters are derived from the atom parameters according to
```math
\begin{aligned}
A_{ij} &= (A_{ii} A_{jj})^{1/2} \\
B_{ij} &= \frac{2}{\frac{1}{B_{ii}} + \frac{1}{B_{jj}}} \\
C_{ij} &= (C_{ii} C_{jj})^{1/2}
\end{aligned}
```
so atoms that use this interaction should have fields `A`, `B` and `C` available.
"""
struct Buckingham{C, W} <: PairwiseInteraction
    cutoff::C
    use_neighbors::Bool
    weight_special::W
end

function Buckingham(;
                    cutoff=NoCutoff(),
                    use_neighbors=false,
                    weight_special=1)
    return Buckingham{typeof(cutoff), typeof(weight_special)}(
        cutoff, use_neighbors, weight_special)
end

use_neighbors(inter::Buckingham) = inter.use_neighbors

@inline function force(inter::Buckingham{C},
                       dr,
                       atom_i,
                       atom_j,
                       force_units=u"kJ * mol^-1 * nm^-1",
                       special::Bool=false,
                       args...) where C
    if (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
       (iszero_value(atom_i.C) || iszero_value(atom_j.C))
        return ustrip.(zero(dr)) * force_units
    end

    Aij = sqrt(atom_i.A * atom_j.A)
    Bij = 2 / (inv(atom_i.B) + inv(atom_j.B))
    Cij = sqrt(atom_i.C * atom_j.C)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    params = (Aij, Bij, Cij)

    f = force_divr_with_cutoff(inter, r2, params, cutoff, force_units)
    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr(::Buckingham, r2, invr2, (A, B, C))
    r = sqrt(r2)
    return A * B * exp(-B * r) / r - 6 * C * invr2^4
end

@inline function potential_energy(inter::Buckingham{C},
                                  dr,
                                  atom_i,
                                  atom_j,
                                  energy_units=u"kJ * mol^-1",
                                  special::Bool=false,
                                  args...) where C
    if (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
       (iszero_value(atom_i.C) || iszero_value(atom_j.C))
        return ustrip(zero(dr[1])) * energy_units
    end

    Aij = sqrt(atom_i.A * atom_j.A)
    Bij = 2 / (inv(atom_i.B) + inv(atom_j.B))
    Cij = sqrt(atom_i.C * atom_j.C)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    params = (Aij, Bij, Cij)

    pe = potential_with_cutoff(inter, r2, params, cutoff, energy_units)
    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::Buckingham, r2, invr2, (A, B, C))
    return A * exp(-B * sqrt(r2)) - C * invr2^3
end
