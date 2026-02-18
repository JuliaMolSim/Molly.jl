# Mixing functions for non-bonded parameters

struct NoShortcut end

shortcut_pair(::NoShortcut, args...) = false

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

σ_mixing(::LorentzMixing, atom_i, atom_j, args...) = (atom_i.σ + atom_j.σ) / 2
ϵ_mixing(::LorentzMixing, atom_i, atom_j, args...) = (atom_i.ϵ + atom_j.ϵ) / 2
λ_mixing(::LorentzMixing, atom_i, atom_j, args...) = (atom_i.λ + atom_j.λ) / 2
A_mixing(::LorentzMixing, atom_i, atom_j, args...) = (atom_i.A + atom_j.A) / 2
B_mixing(::LorentzMixing, atom_i, atom_j, args...) = (atom_i.B + atom_j.B) / 2
C_mixing(::LorentzMixing, atom_i, atom_j, args...) = (atom_i.C + atom_j.C) / 2

struct GeometricMixing end

σ_mixing(::GeometricMixing, atom_i, atom_j, args...) = sqrt(atom_i.σ * atom_j.σ)
ϵ_mixing(::GeometricMixing, atom_i, atom_j, args...) = sqrt(atom_i.ϵ * atom_j.ϵ)
λ_mixing(::GeometricMixing, atom_i, atom_j, args...) = sqrt(atom_i.λ * atom_j.λ)
A_mixing(::GeometricMixing, atom_i, atom_j, args...) = sqrt(atom_i.A * atom_j.A)
B_mixing(::GeometricMixing, atom_i, atom_j, args...) = sqrt(atom_i.B * atom_j.B)
C_mixing(::GeometricMixing, atom_i, atom_j, args...) = sqrt(atom_i.C * atom_j.C)

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
