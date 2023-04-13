export Buckingham

@doc raw"""
    Buckingham(; cutoff, nl_only, weight_special, force_units, energy_units)

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
struct Buckingham{C, W, F, E} <: PairwiseInteraction
    cutoff::C
    nl_only::Bool
    weight_special::W
    force_units::F
    energy_units::E
end

function Buckingham(;
                    cutoff=NoCutoff(),
                    nl_only=false,
                    weight_special=1,
                    force_units=u"kJ * mol^-1 * nm^-1",
                    energy_units=u"kJ * mol^-1")
    return Buckingham{typeof(cutoff), typeof(weight_special), typeof(force_units), typeof(energy_units)}(
        cutoff, nl_only, weight_special, force_units, energy_units)
end

@inline @inbounds function force(inter::Buckingham{C},
                                    dr,
                                    coord_i,
                                    coord_j,
                                    atom_i,
                                    atom_j,
                                    boundary,
                                    special::Bool=false) where C
    if (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
       (iszero_value(atom_i.C) || iszero_value(atom_j.C))
        return ustrip.(zero(coord_i)) * inter.force_units
    end

    Aij = sqrt(atom_i.A * atom_j.A)
    Bij = 2 / (inv(atom_i.B) + inv(atom_j.B))
    Cij = sqrt(atom_i.C * atom_j.C)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    params = (Aij, Bij, Cij)

    if cutoff_points(C) == 0
        f = force_divr_nocutoff(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        f = force_divr_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip.(zero(coord_i)) * inter.force_units

        if r2 < cutoff.sqdist_activation
            f = force_divr_nocutoff(inter, r2, inv(r2), params)
        else
            f = force_divr_cutoff(cutoff, r2, inter, params)
        end
    end

    if special
        return f * dr * inter.weight_special
    else
        return f * dr
    end
end

function force_divr_nocutoff(::Buckingham, r2, invr2, (A, B, C))
    r = sqrt(r2)
    return A * B * exp(-B * r) / r - 6 * C * invr2^4
end

@inline @inbounds function potential_energy(inter::Buckingham{C},
                                            dr,
                                            coord_i,
                                            coord_j,
                                            atom_i,
                                            atom_j,
                                            boundary,
                                            special::Bool=false) where C
    if (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
       (iszero_value(atom_i.C) || iszero_value(atom_j.C))
        return ustrip(zero(coord_i[1])) * inter.energy_units
    end

    Aij = sqrt(atom_i.A * atom_j.A)
    Bij = 2 / (inv(atom_i.B) + inv(atom_j.B))
    Cij = sqrt(atom_i.C * atom_j.C)

    cutoff = inter.cutoff
    r2 = sum(abs2, dr)
    params = (Aij, Bij, Cij)

    if cutoff_points(C) == 0
        pe = potential(inter, r2, inv(r2), params)
    elseif cutoff_points(C) == 1
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        pe = potential_cutoff(cutoff, r2, inter, params)
    elseif cutoff_points(C) == 2
        r2 > cutoff.sqdist_cutoff && return ustrip(zero(coord_i[1])) * inter.energy_units

        if r2 < cutoff.sqdist_activation
            pe = potential(inter, r2, inv(r2), params)
        else
            pe = potential_cutoff(cutoff, r2, inter, params)
        end
    end

    if special
        return pe * inter.weight_special
    else
        return pe
    end
end

function potential(::Buckingham, r2, invr2, (A, B, C))
    return A * exp(-B * sqrt(r2)) - C * invr2^3
end
