# Bias potentials

export
    LinearBias,
    bias_gradient,
    SquareBias,
    FlatBottomSquareBias,
    PeriodicFlatBottomBias,
    BiasPotential

@doc raw"""
    LinearBias(k, cv_target)

A linear bias on a collective variable (CV) towards a target value.

The potential energy is defined as
```math
V(\boldsymbol{s}) = k |\boldsymbol{s} - \boldsymbol{s}_t|
```
where $s$ and $s_t$ are the system and target CV values respectively.
"""
struct LinearBias{K, C}
    k::K
    cv_target::C
end

function potential_energy(lb::LinearBias, cv_sim; kwargs...)
    return lb.k * abs(cv_sim - lb.cv_target)
end

function bias_gradient(lb::LinearBias, cv_sim)
    return lb.k * (cv_sim - lb.cv_target) / abs(cv_sim - lb.cv_target)
end

@doc raw"""
    SquareBias(k, cv_target)

A square (harmonic) bias on a collective variable (CV) towards a target value.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (\boldsymbol{s} - \boldsymbol{s}_t)^2
```
where $s$ and $s_t$ are the system and target CV values respectively.
"""
struct SquareBias{K, C}
    k::K
    cv_target::C
end

function potential_energy(sb::SquareBias, cv_sim; kwargs...)
    return (sb.k / 2) * (cv_sim - sb.cv_target)^2
end

function bias_gradient(sb::SquareBias, cv_sim)
    return sb.k * (cv_sim - sb.cv_target)
end

@doc raw"""
    FlatBottomSquareBias(k, r_fb, cv_target)

A flat-bottomed square (harmonic) bias on a collective variable (CV) towards a target value.

The bias is zero when the value of the collective variable does not deviate
from `cv_target` by more than `r_fb`, and is square (harmonic) outside this range.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (|\boldsymbol{s} - \boldsymbol{s}_t| - r_{fb})^2 H
```
where $s$ and $s_t$ are the system and target CV values respectively, and
```math
H = \left\{ \begin{array}{cl}
0 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| < r_{fb} \\
1 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_t| \geq r_{fb} \\
\end{array} \right.
```
"""
struct FlatBottomSquareBias{K, R, C}
    k::K
    r_fb::R
    cv_target::C
end

function potential_energy(fb::FlatBottomSquareBias, cv_sim; kwargs...)
    d_abs = abs(cv_sim - fb.cv_target)
    H = (d_abs < fb.r_fb ? 0 : 1)
    return (fb.k / 2) * (d_abs - fb.r_fb)^2 * H
end

function bias_gradient(fb::FlatBottomSquareBias, cv_sim)
    d = cv_sim - fb.cv_target
    d_abs = abs(d)
    H = (d_abs < fb.r_fb ? 0 : 1)
    return H * fb.k * (d_abs - fb.r_fb) * d / d_abs
end

@doc raw"""
    PeriodicFlatBottomBias(k, cv_target, width)

A periodic flat-bottom potential on a collective variable (CV).
Behaves like a harmonic potential outside the `width`, but is zero inside.
Handles -π to π periodicity.

The potential energy is defined as:
V(ξ) = 0                                if |d| <= width
V(ξ) = 0.5 * k * (|d| - width)^2        if |d| >  width
where d is the minimum image distance between ξ and cv_target.
"""
struct PeriodicFlatBottomBias{K, T, W}
    k::K
    cv_target::T
    width::W
end

function potential_energy(pb::PeriodicFlatBottomBias, cv_sim; kwargs...)
    # Calculate signed distance in periodic bounds [-π, π]
    d = cv_sim - pb.cv_target
    d_wrapped = d - 2π * round(d / 2π)
    
    # Check flat-bottom condition
    dist = abs(d_wrapped)
    
    if dist <= pb.width
        # Inside flat region: return zero with correct units/type
        # We calculate a dummy energy to get the zero of the correct type
        return zero(0.5 * pb.k * pb.width^2)
    else
        # Outside: Harmonic penalty
        disp = dist - pb.width
        return 0.5 * pb.k * disp^2
    end
end

function bias_gradient(pb::PeriodicFlatBottomBias, cv_sim)
    # 1. Calculate signed distance in periodic bounds [-π, π]
    d = cv_sim - pb.cv_target
    d_wrapped = d - 2π * round(d / 2π)
    
    dist = abs(d_wrapped)
    
    if dist <= pb.width
        # Inside flat region: zero gradient
        return zero(pb.k * pb.width)
    else
        # Outside: Gradient is k * (|d| - w) * sign(d)
        disp = dist - pb.width
        return pb.k * disp * sign(d_wrapped)
    end
end

"""
    BiasPotential(cv_type, bias_type)

A potential to bias a simulation along a collective variable (CV), implemented
as an AtomsCalculators.jl calculator.

The `cv_type` could for example be [`CalcDist`](@ref) and the `bias_type`
could be [`LinearBias`](@ref).

Forces resulting from the bias potential are evaluated in two steps, specfically by
(1) calculating the gradient of the bias potential with respect to the value of the CV, and
(2) calculating the gradient of the CV with respect to the atomic coordinates.

Gradients can be calculated with either automatic differentiation or explicitly defined
gradient functions.
Enzyme should be imported in the first case.

Not currently compatible with virial calculation.
"""
struct BiasPotential{C, B}
    cv_type::C
    bias_type::B
end

function AtomsCalculators.potential_energy(sys, bias::BiasPotential; kwargs...)
    if bias.cv_type.correction == :pbc
        coords = unwrap_molecules(sys)
    else
        coords = sys.coords
    end

    cv_sim = calculate_cv(
        bias.cv_type,
        from_device(coords),
        from_device(sys.atoms),
        sys.boundary,
        from_device(sys.velocities);
        kwargs...,
    )

    return potential_energy(bias.bias_type, cv_sim; kwargs...)
end

function AtomsCalculators.forces!(
    fs, sys, bias::BiasPotential;
    needs_vir::Bool = false,
    buffers::Union{Nothing, BuffersCPU, BuffersGPU} = nothing, # Dummy to be able to have explicit kwarg. In reality a buffer will always be passed
    kwargs...
)
    if bias.cv_type.correction == :pbc
        coords = unwrap_molecules(sys)
    else
        coords = sys.coords
    end

    # Gradient of CV with respect to coordinates
    d_coords, cv_sim = cv_gradient(
        bias.cv_type,
        from_device(coords),
        from_device(sys.atoms),
        sys.boundary,
        from_device(sys.velocities),
    )

    # Gradient of bias function with respect to CV
    d_bias = bias_gradient(bias.bias_type, cv_sim)

    fs_svec = d_bias .* d_coords
    
    if needs_vir && bias.cv_type.has_virial
        calculate_virial(bias.cv_type, coords, fs_svec, buffers, sys.boundary)
    end

    fs .-= to_device(fs_svec, typeof(fs))
    return fs
end
