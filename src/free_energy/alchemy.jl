export 
    AbstractLambdaScheduler,
    DefaultLambdaScheduler

@enum AlchemicalRole CoreRole=0 InsertRole=1 DeleteRole=2

# Rule for combining roles during a pairwise interaction. 
# (Assumes inserting and deleting atoms do not directly interact with each other).
@inline function mix_roles(role_i::AlchemicalRole, role_j::AlchemicalRole)
    if role_i == InsertRole || role_j == InsertRole
        return InsertRole
    elseif role_i == DeleteRole || role_j == DeleteRole
        return DeleteRole
    else
        return CoreRole
    end
end

abstract type AbstractLambdaScheduler end

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