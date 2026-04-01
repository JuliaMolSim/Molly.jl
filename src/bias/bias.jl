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

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `cv_target`: The target value of the collective variable.
"""
struct LinearBias{K, C}
    k::K
    cv_target::C
end

Unitful.ustrip(lb::LinearBias) = LinearBias(ustrip(lb.k), ustrip(lb.cv_target))

function potential_energy(lb::LinearBias, cv_sim; kwargs...)
    return lb.k * abs(cv_sim - lb.cv_target)
end

"""
    bias_gradient(bias::BiasType, cv_sim::Real)

Calculate the gradient of a bias potential with respect to the value of a
collective variable.

# Arguments
- `b::BiasType`: A struct that defines the type of bias to be used.
- `cv_sim::Real`: The value of a measured collective variable given
    the coordinates of a simulation.
"""
function bias_gradient(lb::LinearBias, cv_sim)
    return lb.k * (cv_sim - lb.cv_target) / abs(cv_sim - lb.cv_target)
end

@doc raw"""
    SquareBias(k, cv_target)

A harmonic bias on a collective variable (CV) towards a target value.

The potential energy is defined as
```math
V(\boldsymbol{s}) = \frac{1}{2} k (\boldsymbol{s} - \boldsymbol{s}_t)^2
```
where $s$ and $s_t$ are the system and target CV values respectively.

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `cv_target`: The target value of the collective variable.
"""
struct SquareBias{K, C}
    k::K
    cv_target::C
end

Unitful.ustrip(sb::SquareBias) = SquareBias(ustrip(sb.k), ustrip(sb.cv_target))

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

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `r_fb`: Width of flat-bottom potential well. Inside this region the
    bias potential is always 0.
- `cv_target`: The target value of the collective variable.
"""
struct FlatBottomSquareBias{K, R, C}
    k::K
    r_fb::R
    cv_target::C
end

Unitful.ustrip(fb::FlatBottomSquareBias) = FlatBottomSquareBias(
    ustrip(fb.k),
    ustrip(fb.r_fb),
    ustrip(fb.cv_target),
)

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

A flat-bottomed square (harmonic) bias on a collective variable (CV) towards a target value.

The bias is zero when the value of the collective variable does not deviate
from `cv_target` by more than `r_fb`, and is square (harmonic) outside this range.

This variant handles periodicity in the CV wrapping around the (-π, π) range.

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

# Arguments
- `k`: The energy constant for the bias. Must be compliant with the
    [`System`](@ref) energy units.
- `r_fb`: Width of flat-bottom potential well. Inside this region the
    bias potential is always 0.
- `cv_target`: The target value of the collective variable.
"""
struct PeriodicFlatBottomBias{K, R, T}
    k::K
    r_bf::R
    cv_target::T
end

Unitful.ustrip(pb::PeriodicFlatBottomBias) = PeriodicFlatBottomBias(
    ustrip(pb.k),
    ustrip(pb.r_bf),
    ustrip(pb.cv_target),
)

function potential_energy(pb::PeriodicFlatBottomBias, cv_sim; kwargs...)
    # Calculate signed distance in periodic bounds [-π, π] with inferred units
    FT = typeof(ustrip(pb.cv_target))
    d = cv_sim - pb.cv_target
    twopi = 2 * pi * oneunit(d)
    d_wrapped = d - twopi * round(d / twopi)
    
    dist = abs(d_wrapped)
    
    if dist <= pb.r_bf
        return zero(pb.k * pb.r_bf^2)
    else
        disp = dist - pb.r_bf
        return FT(0.5) * pb.k * disp^2
    end
end

function bias_gradient(pb::PeriodicFlatBottomBias, cv_sim)
    d = cv_sim - pb.cv_target
    twopi = 2 * pi * oneunit(d)
    d_wrapped = d - twopi * round(d / twopi)
    
    dist = abs(d_wrapped)
    
    if dist <= pb.r_bf
        return zero(pb.k * pb.r_bf)
    else
        disp = dist - pb.r_bf
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

Virial contributions must be explicitly defined.
"""
struct BiasPotential{C, B}
    cv_type::C
    bias_type::B
end

Unitful.ustrip(bp::BiasPotential) = BiasPotential(_strip_units(bp.cv_type), _strip_units(bp.bias_type))

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
    buffers = nothing, # Dummy to be able to have explicit kwarg. In reality a buffer will always be passed
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
        calculate_virial!(buffers.virial, bias.cv_type, from_device(coords), -fs_svec, from_device(sys.atoms), sys.boundary)
    end

    fs .-= to_device(fs_svec, typeof(fs))
    return fs
end
