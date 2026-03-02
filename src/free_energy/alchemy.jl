const AlchemicalRole = UInt8

const CoreRole::AlchemicalRole   = 0x00
const InsertRole::AlchemicalRole = 0x01
const DeleteRole::AlchemicalRole = 0x02

abstract type AbstractLambdaScheduler end

#= 
The logic found in this file is a reinterpretation of how OpenFE deals
with alchemical transformations. The original logic can be found in:

https://github.com/OpenFreeEnergy/openfe/blob/main/src/openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py

=#

# Rule for combining roles during a pairwise interaction.
# Dispatched on the scheduler to allow custom overriding by users.
@inline function mix_roles(::AbstractLambdaScheduler, role_i::AlchemicalRole, role_j::AlchemicalRole)
    if role_i == InsertRole || role_j == InsertRole
        return InsertRole
    elseif role_i == DeleteRole || role_j == DeleteRole
        return DeleteRole
    else
        return CoreRole
    end
end

struct DefaultLambdaScheduler <: AbstractLambdaScheduler end

@inline function scale_sterics(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < 0.5 ? 2.0 * λ : 1.0)
    elseif role == DeleteRole
        return T(λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5))
    else
        return λ
    end
end

@inline function scale_elec(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5))
    elseif role == DeleteRole
        return T(λ < 0.5 ? 2.0 * λ : 1.0)
    else
        return λ
    end
end

struct NAMDLambdaScheduler <: AbstractLambdaScheduler end

@inline function scale_sterics(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < (2.0 / 3.0) ? (3.0 / 2.0) * λ : 1.0)
    elseif role == DeleteRole
        return T(λ < (1.0 / 3.0) ? 0.0 : (λ - (1.0 / 3.0)) * (3.0 / 2.0))
    else
        return λ
    end
end

@inline function scale_elec(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5))
    elseif role == DeleteRole
        return T(λ < 0.5 ? 2.0 * λ : 1.0)
    else
        return λ
    end
end

struct QuartersLambdaScheduler <: AbstractLambdaScheduler end

@inline function scale_sterics(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < 0.5 ? 0.0 : (λ > 0.75 ? 1.0 : 4.0 * (λ - 0.5)))
    elseif role == DeleteRole
        return T(λ < 0.25 ? 0.0 : (λ > 0.5 ? 1.0 : 4.0 * (λ - 0.25)))
    else
        return λ
    end
end

@inline function scale_elec(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < 0.75 ? 0.0 : 4.0 * (λ - 0.75))
    elseif role == DeleteRole
        return T(λ < 0.25 ? 4.0 * λ : 1.0)
    else
        return λ
    end
end

struct EleScaledLambdaScheduler <: AbstractLambdaScheduler end

@inline function scale_sterics(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < 0.5 ? 2.0 * λ : 1.0)
    elseif role == DeleteRole
        return T(λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5))
    else
        return λ
    end
end

@inline function scale_elec(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole) where T
    if role == InsertRole
        return T(λ < 0.5 ? 0.0 : sqrt(2.0 * (λ - 0.5)))
    elseif role == DeleteRole
        return T(λ < 0.5 ? (2.0 * λ)^2 : 1.0)
    else
        return λ
    end
end