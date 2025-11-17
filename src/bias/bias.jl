# Bias potentials 

export
    LinearBias,
    SquareBias,
    FlatBottomBias,
    BiasPotential,
    bias_gradient

@doc raw"""
    LinearBias(k, cv_target) 

Apply a linear bias on a collective variable (CV) towards a target value `cv_target`.

The potential energy is defined as
```math
V_{bias}(\boldsymbol{s}) = k |\boldsymbol{s} - \boldsymbol{s}_{target}|
```
where $s$ and $s_{target}$ are system and target CV values, respectively. 
"""
struct LinearBias{K, CVT}   
    k::K
    cv_target::CVT 
end

function potential_energy(lb::LinearBias, cv_sim; kwargs...)
    pe = lb.k * abs(cv_sim - lb.cv_target)
    return pe
end

function bias_gradient(lb::LinearBias, cv_sim)
    bias_grad = lb.k * (cv_sim - lb.cv_target) / abs(cv_sim - lb.cv_target)  
    return bias_grad
end

@doc raw"""
    SquareBias(k, cv_target) 

Apply a harmonic/squared bias on a collective variable towards a target value `cv_target`.

The potential energy is defined as
```math
V_{bias}(\boldsymbol{s}) = \frac{1}{2} k (\boldsymbol{s} - \boldsymbol{s}_{target})^2
```
where $s$ and $s_{target}$ are system and target CV values, respectively. 
"""
struct SquareBias{K, CVT}
    k::K
    cv_target::CVT             
end

function potential_energy(sb::SquareBias, cv_sim; kwargs...)  
    pe = 0.5 * sb.k * (cv_sim - sb.cv_target)^2
    return pe
end

function bias_gradient(sb::SquareBias, cv_sim)
    bias_grad = sb.k * (cv_sim - sb.cv_target)  
    return bias_grad
end

@doc raw"""
    FlatBottomBias(k, r_fb, cv_target) 

Apply a flat-bottomed bias on a collective variable towards a target value `cv_target`.
The potential equals zero when the value of the collective variable does not deviate from `cv_target` by more than `r_fb`.

The potential energy is defined as
```math
V_{bias}(\boldsymbol{s}) = \frac{1}{2} k (|\boldsymbol{s} - \boldsymbol{s}_{target}| - r_{fb})^2 H 
```
where where $s$ and $s_{target}$ are system and target CV values, respectively, and
```math
H = \left\{ \begin{array}{cl}
0 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_{target}| < r_{fb} \\
1 & \text{if} & |\boldsymbol{s} - \boldsymbol{s}_{target}| \geq r_{fb} \\
\end{array} \right.
```
"""
struct FlatBottomBias{K, R, CVT}
    k::K
    r_fb::R
    cv_target::CVT
end

function potential_energy(fb::FlatBottomBias, cv_sim; kwargs...)  
    d_abs = abs(cv_sim - fb.cv_target)
    if d_abs < fb.r_fb  
        H = 0
    else
        H = 1
    end
    pe = ( 0.5 * fb.k * (d_abs - fb.r_fb)^2 ) * H   
    return pe
end

function bias_gradient(fb::FlatBottomBias, cv_sim)
    d = cv_sim - fb.cv_target
    d_abs = abs(cv_sim - fb.cv_target)   
    if d_abs < fb.r_fb  
        H = 0
    else
        H = 1
    end
    bias_grad = H * fb.k * (d_abs - fb.r_fb) * d / d_abs
    return bias_grad
end

@doc raw"""
    BiasPotential(cv_type, bias_type) 

Bias a simulation along the collective variable CV of type `cv_type` and using a bias potential of type `bias_type`.

The `cv_type` could for example be `CalcDist`, and the `bias_type` could be `SquareBias`.

Forces resulting from the bias potential are evaluated in two steps, specfically by
(1) calculating the derivative of the bias potential with respect to the value of the CV, and 
(2) calculating the gradient of the CV with respect to the atomic coordinates. 

Gradients can be calculated with either automatic differentiation or explicitely defined gradient functions. These approaches can be mixed so that one step uses a pre-defined gradient function and the other step uses AD. 
"""
struct BiasPotential{CV, B} 
    cv_type::CV 
    bias_type::B 
end

function AtomsCalculators.potential_energy(   
        sys, bias::BiasPotential; kwargs...,
    )

    if bias.cv_type.correction == :pbc
        coords = Molly.unwrap_molecules(sys) 
    else
        coords = sys.coords 
    end

    cv_sim = calculate_cv(bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities; kwargs...) 
    energy = potential_energy(bias.bias_type, cv_sim; kwargs...) 
    
    return energy   
end 

function AtomsCalculators.forces!(             
        fs, sys, bias::BiasPotential; kwargs...,
    )

    if bias.cv_type.correction == :pbc
        coords = Molly.unwrap_molecules(sys) 
    else
        coords = sys.coords 
    end

    # gradient of cv with respect to coordinates
    d_coords, cv_sim = Molly.cv_gradient(bias.cv_type, coords, sys.atoms, sys.boundary, sys.velocities) 

    # gradient of bias function with respect to cv
    d_bias = bias_gradient(bias.bias_type, cv_sim) 

    # calc forces 
    fs .-= d_bias .* d_coords 

    return fs
end