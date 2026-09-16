# Alchemical Roles
const AlchemicalRole = Int32

const EnvRole::AlchemicalRole    = Int32(0)
const CoreRole::AlchemicalRole   = Int32(1)
const CoreIRole::AlchemicalRole  = Int32(2)
const CoreDRole::AlchemicalRole  = Int32(3)
const InsertRole::AlchemicalRole = Int32(4)
const DeleteRole::AlchemicalRole = Int32(5)

# SoftCore potential options
abstract type SoftCore end

struct DefaultSoftCore <: SoftCore end
struct BeutlerSoftCore <: SoftCore end
struct GapsysSoftCore  <: SoftCore end
struct ScaledSoftCore  <: SoftCore end

# Lambda Schedulers
@kwdef struct DefaultLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct LinearLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct GROMACSLambdaABFEScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct GROMACSLambdaRBFEScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct OpenFEScheduler
    dual::Bool = false
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = true
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct NAMDLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct QuartersLambdaScheduler 
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct EleScaledLambdaScheduler 
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end
@kwdef struct DiffusionLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
    intraC::Bool = true
end

# `Val(scheduler.dual)` builds a type from a runtime field, which the GPU compiler cannot
# lower. Branching on the Bool instead keeps both `Val`s compile-time literals, so these are
# the forms to use anywhere a kernel may reach.
@inline scale_dual(s, λ, role) =
    s.dual ? scale(s, λ, role, Val(true)) : scale(s, λ, role, Val(false))

@inline scale_torsion_dual(s, λ, role) =
    s.dual ? scale_torsion(s, λ, role, Val(true)) : scale_torsion(s, λ, role, Val(false))

@inline scale_bias_dual(s, λ, role) =
    s.dual ? scale_bias(s, λ, role, Val(true)) : scale_bias(s, λ, role, Val(false))

@inline scale_sterics_dual(s, λ, role) =
    s.dual ? scale_sterics(s, λ, role, Val(true)) : scale_sterics(s, λ, role, Val(false))

@inline scale_elec_dual(s, λ, role) =
    s.dual ? scale_elec(s, λ, role, Val(true)) : scale_elec(s, λ, role, Val(false))

@inline scale_virial_dual(s, λ, role) =
    s.dual ? scale_virial(s, λ, role, Val(true)) : scale_virial(s, λ, role, Val(false))

@inline function mix_default(roles::Tuple{Vararg{AlchemicalRole}})
    if any(x->x==InsertRole, roles)
        return InsertRole
    elseif any(x->x==DeleteRole, roles)
        return DeleteRole
    elseif any(x->x==CoreRole, roles)
        return CoreRole
    elseif any(x->x==CoreIRole, roles)
        return CoreIRole
    elseif any(x->x==CoreDRole, roles)
        return CoreDRole
    else
        return EnvRole
    end
end

@inline function mix_special(roles::Tuple{Vararg{AlchemicalRole}})
    if all(x->x==InsertRole, roles)
        return EnvRole
    elseif any(x->x==InsertRole, roles)
        return InsertRole
    elseif all(x->x==DeleteRole, roles)
        return EnvRole
    elseif any(x->x==DeleteRole, roles)
        return DeleteRole
    elseif any(x->x==CoreRole, roles)
        return CoreRole
    elseif any(x->x==CoreIRole, roles)
        return CoreIRole
    elseif any(x->x==CoreDRole, roles)
        return CoreDRole
    else
        return EnvRole
    end
end

@inline function mix_roles(inter::Any, roles::Tuple{Vararg{AlchemicalRole}}; type="")
    if type=="coulomb" && inter.intraC
        return mix_special(roles)
    elseif type=="LJ" && inter.intraLJ
        return mix_special(roles)
    else
        return mix_default(roles)
    end
end

"""
    switchAB(alch_role, A, B)

The pair of end-state parameters an alchemical role interpolates between.

`InsertRole` and `DeleteRole` atoms exist in only one end state, so they hold that state's
parameters at both ends and let the coupling do the work. Only the core roles genuinely morph
A -> B.

Branches on the role *value*. Dispatching on `Val(alch_role)` instead would build a type from a
runtime field, which is not GPU-compilable — see the B20 family in `FREE_ENERGY_TABLES.md`.
"""
@inline function switchAB(alch_role, A, B)
    if alch_role == DeleteRole
        return A, A
    elseif alch_role == InsertRole
        return B, B
    else
        return A, B
    end
end
