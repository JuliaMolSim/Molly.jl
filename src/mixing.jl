# Mixing functions for non-bonded parameters

@public LJZeroShortcut,
    BuckinghamZeroShortcut,
    mixing_atom_fields,
    LorentzMixing,
    σ_mixing,
    ϵ_mixing,
    λ_mixing,
    A_mixing,
    B_mixing,
    C_mixing,
    GeometricMixing,
    WaldmanHaglerMixing,
    FenderHalseyMixing,
    InverseMixing,
    MinimumMixing,
    ProductMixing,
    ExceptionList,
    MixingException

shortcut_pair(::Nothing, args...) = false

"""
    LJZeroShortcut()

Skip a pairwise interaction when the `σ` or `ϵ` of either atom is zero.

Give this when setting up a pairwise interaction such as [`LennardJones`](@ref).
"""
struct LJZeroShortcut end

function shortcut_pair(::LJZeroShortcut, atom_i, atom_j, args...)
    return iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ) 
end

"""
    BuckinghamZeroShortcut()

Skip a pairwise interaction when the `A` or `C` of either atom is zero.

Give this when setting up a pairwise interaction such as [`Buckingham`](@ref).
"""
struct BuckinghamZeroShortcut end

function shortcut_pair(::BuckinghamZeroShortcut, atom_i, atom_j, args...)
    return (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
           (iszero_value(atom_i.C) || iszero_value(atom_j.C))
end

function params_mixing(λ_params, params, args...)
    return params
end

function params_mixing(λ_params, params::Tuple, args...)
    return sum(((1-λ_params),λ_params) .* params)
end

"""
    mixing_atom_fields(mixing)

Extra atom fields that a mixing rule reads.

Custom mixing schemes should implement this function if fields not specified by
[`Molly.required_atom_fields`](@ref) are required by the mixing rule.
"""
mixing_atom_fields(mr) = ()

@doc raw"""
    LorentzMixing()

Mix parameters according to their arithmetic mean.

For example:
```math
\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}
```
Give this when setting up a pairwise interaction such as [`LennardJones`](@ref).
"""
struct LorentzMixing end

function xy_mixing(::LorentzMixing, x, y, args...) 
    return (x + y) / 2
end

function xy_mixing(::LorentzMixing, x::Tuple, y::Tuple, λ_params, alch_role) 
    xA, xB = x
    yA, yB = y
    A = (xA + yA) / 2
    B = (xB + yB) / 2
    A, B = switchAB(alch_role, A, B)
    return (1-λ_params)*A + λ_params*B
end

"""
    σ_mixing(mixing, atom_i, atom_j, args...)

The rule for mixing σ parameters for a given mixing scheme.

Custom mixing schemes can implement this function.
"""
σ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.σ , atom_j.σ, args...)

"""
    ϵ_mixing(mixing, atom_i, atom_j, args...)

The rule for mixing ϵ parameters for a given mixing scheme.

Custom mixing schemes can implement this function.
"""
ϵ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.ϵ , atom_j.ϵ, args...)

"""
    λ_mixing(mixing, atom_i, atom_j, args...)

The rule for mixing λ parameters for a given mixing scheme.

Custom mixing schemes can implement this function.
"""
λ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.λ , atom_j.λ, args...)

"""
    A_mixing(mixing, atom_i, atom_j, args...)

The rule for mixing A ([`Buckingham`](@ref)) parameters for a given mixing scheme.

Custom mixing schemes can implement this function.
"""
A_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.A , atom_j.A, args...)

"""
    B_mixing(mixing, atom_i, atom_j, args...)

The rule for mixing B ([`Buckingham`](@ref)) parameters for a given mixing scheme.

Custom mixing schemes can implement this function.
"""
B_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.B , atom_j.B, args...)

"""
    C_mixing(mixing, atom_i, atom_j, args...)

The rule for mixing C ([`Buckingham`](@ref)) parameters for a given mixing scheme.

Custom mixing schemes can implement this function.
"""
C_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.C , atom_j.C, args...)

@doc raw"""
    GeometricMixing()

Mix parameters according to their geometric mean.

For example:
```math
\varepsilon_{ij} = \sqrt{\varepsilon_i \varepsilon_j}
```
Give this when setting up a pairwise interaction such as [`LennardJones`](@ref).
"""
struct GeometricMixing end

function xy_mixing(::GeometricMixing, x, y, args...) 
    return sqrt(x * y)
end

function xy_mixing(::GeometricMixing, x::Tuple, y::Tuple, λ_params, alch_role, args...) 
    xA,xB = x
    yA,yB = y
    A = sqrt(xA*yA)
    B = sqrt(xB*yB)
    return (1-λ_params)*A + λ_params*B
end

σ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.σ, atom_j.σ, args...)
ϵ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.ϵ, atom_j.ϵ, args...)
λ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.λ, atom_j.λ, args...)
A_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.A, atom_j.A, args...)
B_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.B, atom_j.B, args...)
C_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.C, atom_j.C, args...)

@doc raw"""
    WaldmanHaglerMixing()

Mix `σ` and `ϵ` parameters according to the Waldman-Hagler rules.

See [Waldman and Hagler 1993](https://doi.org/10.1002/jcc.540140909).
```math
\begin{aligned}
\sigma_{ij} &= \left( \frac{\sigma_i^6 + \sigma_j^6}{2} \right)^{\frac{1}{6}} \\
\varepsilon_{ij} &= 2 \sqrt{\varepsilon_i \varepsilon_j} \frac{\sigma_i^3 \sigma_j^3}{\sigma_i^6 + \sigma_j^6}
\end{aligned}
```
Give this when setting up a pairwise interaction such as [`LennardJones`](@ref).
"""
struct WaldmanHaglerMixing end

function σ_mixing(::WaldmanHaglerMixing, atom_i, atom_j, args...)
    T = typeof(ustrip(atom_i.σ))
    return ((atom_i.σ^6 + atom_j.σ^6) / 2) ^ T(1/6)
end

function ϵ_mixing(::WaldmanHaglerMixing, atom_i, atom_j, args...)
    return 2 * sqrt(atom_i.ϵ * atom_j.ϵ) * ((atom_i.σ^3 * atom_j.σ^3) / (atom_i.σ^6 + atom_j.σ^6))
end

@doc raw"""
    FenderHalseyMixing()

Mix `ϵ` parameters according to the Fender-Halsey rule.

See [Fender and Halsey 1962](https://doi.org/10.1063/1.1701284).
```math
\varepsilon_{ij} = \frac{2 \varepsilon_i \varepsilon_j}{\varepsilon_i + \varepsilon_j}
```
Give this when setting up a pairwise interaction such as [`LennardJones`](@ref).
"""
struct FenderHalseyMixing end

function ϵ_mixing(::FenderHalseyMixing, atom_i, atom_j, args...)
    return (2 * atom_i.ϵ * atom_j.ϵ) / (atom_i.ϵ + atom_j.ϵ)
end

@doc raw"""
    InverseMixing()

Mix `B` parameters according to their harmonic mean.

```math
B_{ij} = \frac{2}{\frac{1}{B_i} + \frac{1}{B_j}}
```
Give this when setting up a pairwise interaction such as [`Buckingham`](@ref).
"""
struct InverseMixing end

B_mixing(::InverseMixing, atom_i, atom_j, args...) = 2 / (inv(atom_i.B) + inv(atom_j.B))

# Dict can be used on CPU but doesn't seem faster than ExceptionList for a few exceptions
function get_pair(d::Dict, i, j, default)
    k1 = (i, j)
    k2 = (j, i)
    if haskey(d, k1)
        return d[k1]
    elseif haskey(d, k2)
        return d[k2]
    else
        return default
    end
end

@doc raw"""
    MinimumMixing()

Mix `λ` parameters by taking the minimum over the atoms in an interaction.

For an interaction between atoms ``i``, ``j``, ``k``, ...:
```math
\lambda_{ijk \dots} = \min(\lambda_i, \lambda_j, \lambda_k, \dots)
```
This works for pairwise and specific interactions with any number of atoms.
Give this when setting up an alchemical interaction, where it is the default.
"""
struct MinimumMixing end

@inline λ_mixing(m::MinimumMixing, atoms::Tuple, args...) = min(map(a -> a.λ, atoms)...)

@doc raw"""
    ProductMixing()

Mix `λ` parameters according to their product.

```math
\lambda_{ij} = \lambda_i \lambda_j
```
This only works for interactions between two atoms.
Give this when setting up an alchemical interaction.
"""
struct ProductMixing end

@inline λ_mixing(m::ProductMixing, (a, b), args...) = a.λ * b.λ

"""
    ExceptionList(dict)
    ExceptionList(keys, values)

Define a set of atom pairs that have a specified mixing value.

For example, giving `Dict((1, 2) => 0.3u"nm")` means that pairs of atom type
1 and 2 have a σ value of 0.3 nm.
Provide when setting up a [`Molly.MixingException`](@ref).
"""
struct ExceptionList{N, K, V} # GPU-compatible dictionary-like object for pair lookup
    keys::SVector{N, K}
    values::SVector{N, V}
end

function ExceptionList(d)
    n = length(d)
    ks = SVector{n}(collect(keys(d)))
    vs = SVector{n}(d[k] for k in ks)
    return ExceptionList(ks, vs)
end

# Avoiding branches helps GPU performance
function get_pair(d::ExceptionList{N}, i, j, default) where N
    k1 = (i, j)
    k2 = (j, i)
    val = default
    for ki in 1:N
        if d.keys[ki] == k1 || d.keys[ki] == k2
            val = d.values[ki]
        end
    end
    return val
end

"""
    MixingException(mixing, exceptions)

Provide mixing exceptions (NBFix) for specific pairs of atom types.

`mixing` is a mixing rule such as [`Molly.LorentzMixing`](@ref), used for pairs that are not
exceptions, and `exceptions` is a [`Molly.ExceptionList`](@ref) keyed by atom type.
Give this when setting up a pairwise interaction such as [`LennardJones`](@ref).
"""
struct MixingException{M, E}
    mixing::M
    exceptions::E
end

mixing_atom_fields(::MixingException) = (:atom_type,)

function σ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = σ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end

function ϵ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = ϵ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end

@inline function λ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = λ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end
