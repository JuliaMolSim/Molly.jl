# Cutoff strategies for long-range interactions

export
    NoCutoff,
    DistanceCutoff,
    ShiftedPotentialCutoff,
    ShiftedForceCutoff,
    CubicSplineCutoff

abstract type AbstractCutoff{P} end

Base.:+(c1::T, ::T) where {T <: AbstractCutoff} = c1

function pe_cutoff(cutoff::AbstractCutoff{0}, inter, r, params)
    return pairwise_pe(inter, r, params)
end

function pe_cutoff(cutoff::AbstractCutoff{1}, inter, r, params)
    return pe_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff)
end

function pe_cutoff(cutoff::AbstractCutoff{2}, inter, r, params)
    return ifelse(
        r <= cutoff.dist_activation,
        pairwise_pe(inter, r, params),
        pe_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff),
    )
end

function force_cutoff(cutoff::AbstractCutoff{0}, inter, r, params)
    return pairwise_force(inter, r, params)
end

function force_cutoff(cutoff::AbstractCutoff{1}, inter, r, params)
    return force_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff)
end

function force_cutoff(cutoff::AbstractCutoff{2}, inter, r, params)
    return ifelse(
        r <= cutoff.dist_activation,
        pairwise_force(inter, r, params),
        force_apply_cutoff(cutoff, inter, r, params) * (r <= cutoff.dist_cutoff),
    )
end

"""
    NoCutoff()

Placeholder cutoff that does not alter the potential or force.
"""
struct NoCutoff <: AbstractCutoff{0} end

@doc raw"""
    DistanceCutoff(dist_cutoff)

Cutoff that sets the potential and force to be zero past a specified cutoff distance.

```math
\begin{aligned}
V_c(r) &= \begin{cases}
V(r), r \le r_c \\
0, r > r_c
\end{cases} \\
F_c(r) &= \begin{cases}
F(r), r \le r_c \\
0, r > r_c
\end{cases}
\end{aligned}
```
"""
struct DistanceCutoff{P, D} <: AbstractCutoff{1}
    dist_cutoff::D
end

DistanceCutoff(dist_cutoff::D) where D = DistanceCutoff{1, D}(dist_cutoff)

pe_apply_cutoff(::DistanceCutoff, inter, r, params) = pairwise_pe(inter, r, params)
force_apply_cutoff(::DistanceCutoff, inter, r, params) = pairwise_force(inter, r, params)

@doc raw"""
    ShiftedPotentialCutoff(dist_cutoff)

Cutoff that shifts the potential to be continuous at a specified cutoff distance.

```math
\begin{aligned}
V_c(r) &= \begin{cases}
V(r) - V(r_c), r \le r_c \\
0, r > r_c
\end{cases} \\
F_c(r) &= \begin{cases}
F(r), r \le r_c \\
0, r > r_c
\end{cases}
\end{aligned}
```
"""
struct ShiftedPotentialCutoff{P, D} <: AbstractCutoff{1}
    dist_cutoff::D
end

ShiftedPotentialCutoff(dist_cutoff::D) where D = ShiftedPotentialCutoff{1, D}(dist_cutoff)

function pe_apply_cutoff(cutoff::ShiftedPotentialCutoff, inter, r, params)
    pe_r = pairwise_pe(inter, r, params)
    pe_cut = pairwise_pe(inter, cutoff.dist_cutoff, params)
    return pe_r - pe_cut
end

function force_apply_cutoff(::ShiftedPotentialCutoff, inter, r, params)
    return pairwise_force(inter, r, params)
end

@doc raw"""
    ShiftedForceCutoff(dist_cutoff)

Cutoff that shifts the force to be continuous at a specified cutoff distance.

```math
\begin{aligned}
V_c(r) &= \begin{cases}
V(r) - (r-r_c) V'(r_c) - V(r_c), r \le r_c \\
0, r > r_c
\end{cases} \\
F_c(r) &= \begin{cases}
F(r) - F(r_c), r \le r_c \\
0, r > r_c
\end{cases}
\end{aligned}
```
"""
struct ShiftedForceCutoff{P, D} <: AbstractCutoff{1}
    dist_cutoff::D
end

ShiftedForceCutoff(dist_cutoff::D) where D = ShiftedForceCutoff{1, D}(dist_cutoff)

function pe_apply_cutoff(cutoff::ShiftedForceCutoff, inter, r, params)
    pe_r = pairwise_pe(inter, r, params)
    pe_cut = pairwise_pe(inter, cutoff.dist_cutoff, params)
    f_cut = pairwise_force(inter, cutoff.dist_cutoff, params)
    return pe_r + (r - cutoff.dist_cutoff) * f_cut - pe_cut
end

function force_apply_cutoff(cutoff::ShiftedForceCutoff, inter, r, params)
    f_r = pairwise_force(inter, r, params)
    f_cut = pairwise_force(inter, cutoff.dist_cutoff, params)
    return f_r - f_cut
end

@doc raw"""
    CubicSplineCutoff(dist_activation, dist_cutoff)

Cutoff that interpolates between the true potential at an activation distance
and zero at a cutoff distance using a cubic Hermite spline.

```math
\begin{aligned}
V_c(r) &= \begin{cases}
V(r), r \le r_a \\
(2t^3 - 3t^2 + 1) V(r_a) + (t^3 - 2t^2 + t) (r_c - r_a) V'(r_a), r_a < r \le r_c \\
0, r > r_c
\end{cases} \\
F_c(r) &= \begin{cases}
F(r), r \le r_a \\
\frac{-(6t^2 - 6t) V(r_a)}{r_c - r_a} - (3t^2 - 4t + 1) V'(r_a), r_a < r \le r_c \\
0, r > r_c
\end{cases} \\
t &= \frac{r - r_a}{r_c - r_a}
\end{aligned}
```
"""
struct CubicSplineCutoff{P, D} <: AbstractCutoff{2}
    dist_activation::D
    dist_cutoff::D
end

function CubicSplineCutoff(dist_activation::D, dist_cutoff) where D
    if dist_cutoff <= dist_activation
        throw(ArgumentError("the cutoff radius $dist_cutoff must be larger " *
                            "than the activation radius $dist_activation"))
    end
    return CubicSplineCutoff{2, D}(dist_activation, dist_cutoff)
end

function pe_apply_cutoff(cutoff::CubicSplineCutoff, inter, r, params)
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff - cutoff.dist_activation)
    pe_act = pairwise_pe(inter, cutoff.dist_activation, params)
    dpe_dr_act = -pairwise_force(inter, cutoff.dist_activation, params)
    return (2t^3 - 3t^2 + 1) * pe_act + (t^3 - 2t^2 + t) *
           (cutoff.dist_cutoff - cutoff.dist_activation) * dpe_dr_act
end

function force_apply_cutoff(cutoff::CubicSplineCutoff, inter, r, params)
    t = (r - cutoff.dist_activation) / (cutoff.dist_cutoff - cutoff.dist_activation)
    pe_act = pairwise_pe(inter, cutoff.dist_activation, params)
    dpe_dr_act = -pairwise_force(inter, cutoff.dist_activation, params)
    return -(6t^2 - 6t) * pe_act / (cutoff.dist_cutoff - cutoff.dist_activation) -
                    (3t^2 - 4t + 1) * dpe_dr_act
end
