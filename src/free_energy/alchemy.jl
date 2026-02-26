export 
    AbstractLambdaScheduler,
    DefaultLambdaScheduler,
    NAMDLambdaScheduler,
    QuartersLambdaScheduler,
    EleScaledLambdaScheduler,
    AlchemicalRole,
    CoreRole,
    InsertRole,
    DeleteRole

const AlchemicalRole = UInt8

const CoreRole::AlchemicalRole   = 0x00
const InsertRole::AlchemicalRole = 0x01
const DeleteRole::AlchemicalRole = 0x02

abstract type AbstractLambdaScheduler end

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

@inline function scale_sterics(::DefaultLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < 0.5 ? 2.0 * λ : 1.0
    elseif role == DeleteRole
        return λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5)
    else
        return λ
    end
end

@inline function scale_elec(::DefaultLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5)
    elseif role == DeleteRole
        return λ < 0.5 ? 2.0 * λ : 1.0
    else
        return λ
    end
end

struct NAMDLambdaScheduler <: AbstractLambdaScheduler end

@inline function scale_sterics(::NAMDLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < (2.0 / 3.0) ? (3.0 / 2.0) * λ : 1.0
    elseif role == DeleteRole
        return λ < (1.0 / 3.0) ? 0.0 : (λ - (1.0 / 3.0)) * (3.0 / 2.0)
    else
        return λ
    end
end

@inline function scale_elec(::NAMDLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5)
    elseif role == DeleteRole
        return λ < 0.5 ? 2.0 * λ : 1.0
    else
        return λ
    end
end

struct QuartersLambdaScheduler <: AbstractLambdaScheduler end

@inline function scale_sterics(::QuartersLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < 0.5 ? 0.0 : (λ > 0.75 ? 1.0 : 4.0 * (λ - 0.5))
    elseif role == DeleteRole
        return λ < 0.25 ? 0.0 : (λ > 0.5 ? 1.0 : 4.0 * (λ - 0.25))
    else
        return λ
    end
end

@inline function scale_elec(::QuartersLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < 0.75 ? 0.0 : 4.0 * (λ - 0.75)
    elseif role == DeleteRole
        return λ < 0.25 ? 4.0 * λ : 1.0
    else
        return λ
    end
end

struct EleScaledLambdaScheduler <: AbstractLambdaScheduler end

@inline function scale_sterics(::EleScaledLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < 0.5 ? 2.0 * λ : 1.0
    elseif role == DeleteRole
        return λ < 0.5 ? 0.0 : 2.0 * (λ - 0.5)
    else
        return λ
    end
end

@inline function scale_elec(::EleScaledLambdaScheduler, λ::Real, role::AlchemicalRole)
    if role == InsertRole
        return λ < 0.5 ? 0.0 : sqrt(2.0 * (λ - 0.5))
    elseif role == DeleteRole
        return λ < 0.5 ? (2.0 * λ)^2 : 1.0
    else
        return λ
    end
end