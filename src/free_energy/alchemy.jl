const AlchemicalRole = UInt8

const CoreRole::AlchemicalRole   = 0x00
const InsertRole::AlchemicalRole = 0x01
const DeleteRole::AlchemicalRole = 0x02

#= 
The logic found in this file is a reinterpretation of how OpenFE deals
with alchemical transformations. The original logic can be found in:

https://github.com/OpenFreeEnergy/openfe/blob/main/src/openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py

=#

# Rule for combining roles during a pairwise interaction.
# Dispatched on the scheduler to allow custom overriding by users.
@inline function mix_roles(::Any, role_i::AlchemicalRole, role_j::AlchemicalRole)
    if role_i == InsertRole || role_j == InsertRole
        return InsertRole
    elseif role_i == DeleteRole || role_j == DeleteRole
        return DeleteRole
    else
        return CoreRole
    end
end

struct DefaultLambdaScheduler end

@inline function scale_sterics(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return λ < T(0.5) ? T(2.0) * λ : T(1.0)
    elseif role == DeleteRole
        return λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
    else
        return λ
    end
end

@inline function scale_elec(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
    elseif role == DeleteRole
        return T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
    else
        return λ
    end
end

struct NAMDLambdaScheduler end

@inline function scale_sterics(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < (T(2.0) / T(3.0)) ? (T(3.0) / T(2.0)) * λ : T(1.0))
    elseif role == DeleteRole
        return T(λ < (T(1.0) / T(3.0)) ? T(0.0) : (λ - (T(1.0) / T(3.0))) * (T(3.0) / T(2.0)))
    else
        return λ
    end
end

@inline function scale_elec(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
    elseif role == DeleteRole
        return T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
    else
        return λ
    end
end

struct QuartersLambdaScheduler end

@inline function scale_sterics(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return λ < T(0.5) ? T(0.0) : (λ > T(0.75) ? T(1.0) : T(4.0) * (λ - T(0.5)))
    elseif role == DeleteRole
        return λ < T(0.25) ? T(0.0) : (λ > T(0.5) ? T(1.0) : T(4.0) * (λ - T(0.25)))
    else
        return λ
    end
end

@inline function scale_elec(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return λ < T(0.75) ? T(0.0) : T(4.0) * (λ - T(0.75))
    elseif role == DeleteRole
        return λ < T(0.25) ? T(4.0) * λ : T(1.0)
    else
        return λ
    end
end

struct EleScaledLambdaScheduler end

@inline function scale_sterics(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return λ < T(0.5) ? T(2.0) * λ : T(1.0)
    elseif role == DeleteRole
        return λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
    else
        return λ
    end
end

@inline function scale_elec(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return λ < T(0.5) ? T(0.0) : sqrt(T(2.0) * (λ - T(0.5)))
    elseif role == DeleteRole
        return λ < T(0.5) ? (T(2.0) * λ)^2 : T(1.0)
    else
        return λ
    end
end