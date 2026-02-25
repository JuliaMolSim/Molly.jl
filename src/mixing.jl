# Mixing functions for non-bonded parameters

shortcut_pair(::Nothing, args...) = false

struct LJZeroShortcut end

function shortcut_pair(::LJZeroShortcut, atom_i, atom_j, args...)
    return iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
           iszero_value(atom_i.σ) || iszero_value(atom_j.σ)
end

struct BuckinghamZeroShortcut end

function shortcut_pair(::BuckinghamZeroShortcut, atom_i, atom_j, args...)
    return (iszero_value(atom_i.A) || iszero_value(atom_j.A)) &&
           (iszero_value(atom_i.C) || iszero_value(atom_j.C))
end

struct LorentzMixing end

xy_mixing(::LorentzMixing, x, y, args...) = (x + y) / 2
σ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.σ , atom_j.σ, args...)
ϵ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.ϵ , atom_j.ϵ, args...)
λ_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.λ , atom_j.λ, args...)
A_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.A , atom_j.A, args...)
B_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.B , atom_j.B, args...)
C_mixing(m::LorentzMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.C , atom_j.C, args...)

struct GeometricMixing end

xy_mixing(::GeometricMixing, x, y, args...) = sqrt(x * y)
σ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.σ, atom_j.σ, args...)
ϵ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.ϵ, atom_j.ϵ, args...)
λ_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.λ, atom_j.λ, args...)
A_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.A, atom_j.A, args...)
B_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.B, atom_j.B, args...)
C_mixing(m::GeometricMixing, atom_i, atom_j, args...) = xy_mixing(m, atom_i.C, atom_j.C, args...)

struct WaldmanHaglerMixing end

function σ_mixing(::WaldmanHaglerMixing, atom_i, atom_j, args...)
    T = typeof(ustrip(atom_i.σ))
    return ((atom_i.σ^6 + atom_j.σ^6) / 2) ^ T(1/6)
end

function ϵ_mixing(::WaldmanHaglerMixing, atom_i, atom_j, args...)
    return 2 * sqrt(atom_i.ϵ * atom_j.ϵ) * ((atom_i.σ^3 * atom_j.σ^3) / (atom_i.σ^6 + atom_j.σ^6))
end

struct FenderHalseyMixing end

function ϵ_mixing(::FenderHalseyMixing, atom_i, atom_j, args...)
    return (2 * atom_i.ϵ * atom_j.ϵ) / (atom_i.ϵ + atom_j.ϵ)
end

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

# GPU-compatible dictionary-like object for pair lookup
struct ExceptionList{N, K, V}
    keys::SVector{N, K}
    values::SVector{N, V}
end

function ExceptionList(d::AbstractDict)
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

# Provide exceptions (NBFix) for specific pairs of atom types
struct MixingException{M, E}
    mixing::M
    exceptions::E
end

function σ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = σ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end

function ϵ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = ϵ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end

function λ_mixing(me::MixingException, atom_i, atom_j, args...)
    default = λ_mixing(me.mixing, atom_i, atom_j, args...)
    return get_pair(me.exceptions, atom_i.atom_type, atom_j.atom_type, default)
end
