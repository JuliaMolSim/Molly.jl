using Molly
using Molly: AlchemicalRole, EnvRole   # roles are not exported by Molly

export
    ProbabilityLambdaScheduler

# `scaled` marks a bonded term that is switched off with its substituent whatever the roles of its
#   atoms. The scheduler is stored in every bonded term, so this is set for each term on its own.
struct ProbabilitySchedule
    scaled::Bool
end
ProbabilitySchedule() = ProbabilitySchedule(false)
const ProbabilityLambdaScheduler = Molly.LambdaScheduler{ProbabilitySchedule}

always_scaled(s::ProbabilityLambdaScheduler) =
    Molly.LambdaScheduler(ProbabilitySchedule(true), s.dual, s.LJindividual, s.LJspecial,
                          s.Cindividual, s.Cspecial, s.intraLJ, s.Tscaled)

# Each alchemical atom carries the λ of its own substituent, the fraction that substituent is on
const CorePRole::AlchemicalRole = Int32(6)   # a copy of a core atom
const ProbRole::AlchemicalRole = Int32(7)    # an atom unique to one substituent
# A unique atom whose bonded terms with core atoms are switched off with its substituent: a soft
#   bond, which opens a ring that is closed through the core
const SoftRole::AlchemicalRole = Int32(8)
# A pooled counter-ion that only changes its charge with λ: the mix_* functions treat it as the environment,
#   so its Lennard-Jones stays on in every state and its pairs get no soft core; only its own charge (λq, per
#   atom in `scale_elec`, which PME and the Ewald pairs take) follows λ. A pure charge controller
const ChargeRole::AlchemicalRole = Int32(9)

@inline is_unique(role) = role == ProbRole || role == SoftRole

@inline function mix_default_prob(roles::Tuple{Vararg{AlchemicalRole}})
    if any(x->x==SoftRole, roles) && any(x->x==CorePRole, roles)
        return CorePRole
    elseif any(is_unique, roles)
        return ProbRole
    elseif any(x->x==CorePRole, roles)
        return CorePRole
    else
        return EnvRole
    end
end

@inline function mix_special_prob(roles::Tuple{Vararg{AlchemicalRole}})
    if all(is_unique, roles)
        return EnvRole
    elseif any(is_unique, roles)
        return ProbRole
    elseif any(x->x==CorePRole, roles)
        return CorePRole
    else
        return EnvRole
    end
end

@inline function mix_torsion_prob(roles::Tuple{Vararg{AlchemicalRole}})
    if any(x->x==CorePRole, roles)
        return CorePRole
    else
        return EnvRole
    end
end

@inline function Molly.mix_roles(inter::ProbabilityLambdaScheduler, roles::Tuple{Vararg{AlchemicalRole}}; lj=false,
                           torsion=false)
    if inter.schedule.scaled
        return any(x->x!=EnvRole, roles) ? CorePRole : EnvRole
    elseif torsion && inter.Tscaled
        return mix_torsion_prob(roles)
    elseif lj && inter.intraLJ
        return mix_special_prob(roles)
    else
        return mix_default_prob(roles)
    end
end

####################################
### Probability Lambda Scheduler ###
####################################

# Bonded terms of unique atoms stay on so that a switched off substituent keeps its geometry,
#   except those of a soft atom with a core atom, which `mix_roles` hands over as `CorePRole`
@inline function Molly.scale(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if role == CorePRole
        return λ, one(λ)
    else
        return one(λ), one(λ)
    end
end

# Molly builds the weight of each term of a `PeriodicTorsionλ{N}` from this one, so no `args...`
#   here: that would be ambiguous with Molly's `Val(N)` method
@inline function Molly.scale_torsion(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}) where T
    if role == CorePRole
        return λ, one(λ)
    else
        return one(λ), one(λ)
    end
end

@inline function Molly.scale_sterics(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if is_unique(role) || role == CorePRole
        return λ, one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

@inline function Molly.scale_elec(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{true}, args...) where T
    if is_unique(role) || role == CorePRole || role == ChargeRole
        return λ, one(λ), one(λ)
    else
        return one(λ), one(λ), one(λ)
    end
end

# The bonded terms of a unique atom are not scaled, so their virial has to carry the λ of the atom.
#   Molly's `virial_lambda_factor` does not know the roles above, and its bonded types only have
#   the scheduler in common as their last type parameter.
const ProbBonded = Union{
    Molly.HarmonicBondλ{<:Any, <:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.HarmonicAngleλ{<:Any, <:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.CosineAngleλ{<:Any, <:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.HarmonicTorsionλ{<:Any, <:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.MorseBondλ{<:Any, <:Any, <:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.UreyBradleyλ{<:Any, <:Any, <:Any, <:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.RBTorsionλ{<:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.CMAPTorsionλ{<:Any, <:Any, <:ProbabilityLambdaScheduler},
    Molly.PeriodicTorsionλ{<:Any, <:Any, <:Any, <:Any, <:ProbabilityLambdaScheduler},
}

@inline function Molly.virial_lambda_factor(inter::ProbBonded, atoms)
    λ = Molly.λ_mixing(inter.λ_mixing, atoms)
    roles = map(a -> a.alch_role, atoms)
    # A torsion that is scaled through its core copies already carries λ in its force
    scaled = Molly.is_torsion(inter) && inter.scheduler.Tscaled && mix_torsion_prob(roles) != EnvRole
    scaled |= inter.scheduler.schedule.scaled
    return (!scaled && mix_default_prob(roles) == ProbRole) ? λ : one(λ)
end

# Single topology cannot hold more than two states, so it is not supported. These two are never
#   reached, but a GPU kernel compiles both branches of `scheduler.dual`
@inline Molly.scale_sterics(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{false}, args...) where T =
    one(λ), one(λ), one(λ)

@inline Molly.scale_elec(::ProbabilityLambdaScheduler, λ::T, role::AlchemicalRole, dual::Val{false}, args...) where T =
    one(λ), one(λ), one(λ)
