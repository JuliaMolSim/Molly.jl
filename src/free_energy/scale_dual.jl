#= 
The logic found in this file is a reinterpretation of how OpenFE deals
with alchemical transformations. The original logic can be found in:

https://github.com/OpenFreeEnergy/openfe/blob/main/src/openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py

Adjusted to support dual topology transformations, which are not supported in OpenFE.

=#

@inline function scale(::Any, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == CoreIRole
        return λ, one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ)
    else
        return one(λ), one(λ)
    end
end

@inline function scale_virial(::Any, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        return λ
    elseif role == DeleteRole
        return (1 - λ)
    else
        return one(λ)
    end
end

@inline function scale_torsion(::Any, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == CoreIRole
        return (λ,λ,λ,λ,λ,λ)
    elseif role == CoreDRole
        return ((1-λ),(1-λ),(1-λ),(1-λ),(1-λ),(1-λ))
    else
        return (one(λ),one(λ),one(λ),one(λ),one(λ),one(λ))
    end
end

################################
### Default Lambda Scheduler ###
################################

@inline function scale_sterics(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

@inline function scale_elec(::DefaultLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

###############################
### Linear Lambda Scheduler ###
###############################

@inline function scale_sterics(::LinearLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole 
        return λ, one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == DeleteRole 
        return (1-λ), one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

@inline function scale_elec(::LinearLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        return λ, one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        return (1-λ), one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

#####################################
### GROMACS RBFE Lambda Scheduler ###
#####################################

@inline scale_sterics(::GROMACSLambdaRBFEScheduler, λ::T, role::AlchemicalRole, dual::Val{true},
                      args...) where T =
    scale_sterics(LinearLambdaScheduler(), λ, role, dual, args...)

@inline scale_elec(::GROMACSLambdaRBFEScheduler, λ::T, role::AlchemicalRole, dual::Val{true},
                   args...) where T =
    scale_elec(LinearLambdaScheduler(), λ, role, dual, args...)

#####################################
### GROMACS ABFE Lambda Scheduler ###
#####################################

@inline function scale_sterics(::GROMACSLambdaABFEScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T    
    if role == InsertRole || role == CoreIRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole || role == CoreDRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

@inline function scale_elec(::GROMACSLambdaABFEScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T 
    if role == InsertRole || role == CoreIRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole || role == CoreDRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

#############################
### OpenMM Test Scheduler ###
#############################

@inline function scale_sterics(::OpenFEScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

@inline function scale_elec(::OpenFEScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

#############################
### NAMD Lambda Scheduler ###
#############################

@inline function scale_sterics(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = T(λ < (T(2.0) / T(3.0)) ? (T(3.0) / T(2.0)) * λ : T(1.0))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = T(λ < (T(1.0) / T(3.0)) ? T(0.0) : (λ - (T(1.0) / T(3.0))) * (T(3.0) / T(2.0)))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), λ
    end
end

@inline function scale_elec(::NAMDLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = T(λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5)))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = T(λ < T(0.5) ? T(2.0) * λ : T(1.0))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

#################################
### Quarters Lambda Scheduler ###
#################################

@inline function scale_sterics(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(0.0) : (λ > T(0.75) ? T(1.0) : T(4.0) * (λ - T(0.5)))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.25) ? T(0.0) : (λ > T(0.5) ? T(1.0) : T(4.0) * (λ - T(0.25)))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

@inline function scale_elec(::QuartersLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = λ < T(0.75) ? T(0.0) : T(4.0) * (λ - T(0.75))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.25) ? T(4.0) * λ : T(1.0)
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

##############################################
### Electrostatics Scaled Lambda Scheduler ###
##############################################

@inline function scale_sterics(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(2.0) * λ : T(1.0)
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.5) ? T(0.0) : T(2.0) * (λ - T(0.5))
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

@inline function scale_elec(::EleScaledLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == InsertRole
        λ = λ < T(0.5) ? T(0.0) : sqrt(T(2.0) * (λ - T(0.5)))
        return λ, one(λ), one(λ)
    elseif role == DeleteRole
        λ = λ < T(0.5) ? (T(2.0) * λ)^2 : T(1.0)
        return (1-λ), one(λ), one(λ)
    elseif role == CoreIRole
        return λ, one(λ), one(λ)
    elseif role == CoreDRole
        return (1-λ), one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end