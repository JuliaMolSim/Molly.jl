export
    LambdaScheduler,
    DefaultLambdaScheduler,
    LinearLambdaScheduler,
    GROMACSLambdaABFEScheduler,
    GROMACSLambdaRBFEScheduler,
    OpenFEScheduler,
    NAMDLambdaScheduler,
    QuartersLambdaScheduler,
    EleScaledLambdaScheduler

# Alchemical Roles
const AlchemicalRole = Int32

const EnvRole::AlchemicalRole    = Int32(0)
const CoreRole::AlchemicalRole   = Int32(1)
const CoreIRole::AlchemicalRole  = Int32(2)
const CoreDRole::AlchemicalRole  = Int32(3)
const InsertRole::AlchemicalRole = Int32(4)
const DeleteRole::AlchemicalRole = Int32(5)

# Supertype of the λ versions of the bonded interactions, see `virial_lambda_factor`
abstract type AlchemicalBondedInteraction end

# SoftCore potential options
abstract type SoftCore end

struct BeutlerSoftCore <: SoftCore end
struct GapsysSoftCore  <: SoftCore end
struct ScaledSoftCore  <: SoftCore end

# Lambda Schedulers
# The schedule of a `LambdaScheduler` is one of these empty types, which the scaling functions
#   dispatch on through the aliases below
struct DefaultSchedule end
struct LinearSchedule end
struct GROMACSABFESchedule end
struct GROMACSRBFESchedule end
struct OpenFESchedule end
struct NAMDSchedule end
struct QuartersSchedule end
struct EleScaledSchedule end

"""
    LambdaScheduler(schedule=DefaultSchedule(); dual=true, LJindividual=false,
                    LJspecial=false, Cindividual=false, Cspecial=false, intraLJ=false,
                    Tscaled=true)

Turns `global_λ` into the couplings of the steric, electrostatic and bonded interactions of each
alchemical role.

`schedule` selects how the couplings are staged along `global_λ`. Each schedule has an alias
that is normally used instead: [`DefaultLambdaScheduler`](@ref), [`LinearLambdaScheduler`](@ref),
[`GROMACSLambdaABFEScheduler`](@ref), [`GROMACSLambdaRBFEScheduler`](@ref),
[`OpenFEScheduler`](@ref), [`NAMDLambdaScheduler`](@ref), [`QuartersLambdaScheduler`](@ref) and
[`EleScaledLambdaScheduler`](@ref), e.g. `OpenFEScheduler(dual=false)`. Schedulers are passed to
[`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref).

# Arguments
- `dual=true`: whether to use dual topology (energy scaling) or single topology (parameter
    scaling). `false` by default for [`OpenFEScheduler`](@ref).
- `LJindividual=false`: in single topology, whether the Lennard-Jones parameters are scaled for
    each atom before mixing (`true`) or mixed for each state and then interpolated (`false`).
- `LJspecial=false`: the same choice for the Lennard-Jones 1-4 interactions.
- `Cindividual=false`: the same choice for the Coulomb interactions. `true` by default for
    [`OpenFEScheduler`](@ref).
- `Cspecial=false`: the same choice for the Coulomb 1-4 interactions.
- `intraLJ=false`: whether the Lennard-Jones interactions between alchemical atoms are kept on
    (`true`) or scaled with the atoms (`false`).
- `Tscaled=true`: whether a torsion that spans core and unique atoms follows the core atom and is
    scaled (`true`) or follows the unique atom and stays on at every `global_λ` (`false`). `false`
    by default for [`OpenFEScheduler`](@ref), which does not scale these torsions, so that Molly
    reproduces it exactly. Torsions of unique atoms only are never scaled either way.

Coulomb interactions between alchemical atoms are always scaled with the atoms, as the PME mesh
scales the charge of each atom.
"""
struct LambdaScheduler{S}
    schedule::S
    dual::Bool
    LJindividual::Bool
    LJspecial::Bool
    Cindividual::Bool
    Cspecial::Bool
    intraLJ::Bool
    Tscaled::Bool
end

# OpenFE uses single topology with individually scaled Coulomb parameters and does not scale the
#   torsions that span core and unique atoms
function LambdaScheduler(schedule=DefaultSchedule(); dual=!(schedule isa OpenFESchedule),
                         LJindividual=false, LJspecial=false,
                         Cindividual=(schedule isa OpenFESchedule), Cspecial=false,
                         intraLJ=false, Tscaled=!(schedule isa OpenFESchedule))
    return LambdaScheduler(schedule, dual, LJindividual, LJspecial, Cindividual, Cspecial,
                           intraLJ, Tscaled)
end

# Allows the aliases to be called like types, e.g. `DefaultLambdaScheduler(dual=false)`
LambdaScheduler{S}(; kwargs...) where {S} = LambdaScheduler(S(); kwargs...)

"""
    DefaultLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                           Cindividual=false, Cspecial=false, intraLJ=false,
                           Tscaled=true)

Lambda scheduler that turns off the electrostatics of an atom before its sterics.

Atoms that only exist in the first state lose their charges between `global_λ = 0` and `0.5`
and their Lennard-Jones interactions between `0.5` and `1`. Atoms that only exist in the second
state gain their Lennard-Jones interactions between `0` and `0.5` and their charges between `0.5`
and `1`, so an atom is never charged without its Lennard-Jones core. Core atoms, present in both
states, are interpolated linearly.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const DefaultLambdaScheduler = LambdaScheduler{DefaultSchedule}

"""
    LinearLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                          Cindividual=false, Cspecial=false, intraLJ=false,
                          Tscaled=true)

Lambda scheduler that scales the electrostatics and sterics of all alchemical atoms linearly and
at the same time, from `global_λ = 0` to `1`.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const LinearLambdaScheduler = LambdaScheduler{LinearSchedule}

"""
    GROMACSLambdaABFEScheduler(; dual=true, LJindividual=false, LJspecial=false,
                               Cindividual=false, Cspecial=false, intraLJ=false,
                               Tscaled=true)

Lambda scheduler for absolute free energy calculations, following GROMACS.

The charges of the alchemical atoms are scaled between `global_λ = 0` and `0.5` and their
Lennard-Jones interactions between `0.5` and `1`, so decoupled atoms lose their charges first.
With [`AbsoluteFESystem`](@ref), PME is replaced by the two grid scheme of GROMACS, where the
grids of the two end states are mixed with the electrostatic coupling. Inserted atoms follow the
same stages and would gain their charges before their Lennard-Jones interactions, so use
[`DefaultLambdaScheduler`](@ref) for insertions.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const GROMACSLambdaABFEScheduler = LambdaScheduler{GROMACSABFESchedule}

"""
    GROMACSLambdaRBFEScheduler(; dual=true, LJindividual=false, LJspecial=false,
                               Cindividual=false, Cspecial=false, intraLJ=false,
                               Tscaled=true)

Lambda scheduler for relative free energy calculations, following GROMACS.

The electrostatics and sterics are scaled linearly, as in [`LinearLambdaScheduler`](@ref). With
[`RelativeFESystem`](@ref), PME is replaced by the two grid scheme of GROMACS, where the grids of
the two end states are mixed with the electrostatic coupling.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const GROMACSLambdaRBFEScheduler = LambdaScheduler{GROMACSRBFESchedule}

"""
    OpenFEScheduler(; dual=false, LJindividual=false, LJspecial=false,
                    Cindividual=true, Cspecial=false, intraLJ=false,
                    Tscaled=false)

Lambda scheduler that reproduces the relative free energy setup of
[OpenFE](https://github.com/OpenFreeEnergy/openfe).

The electrostatics and sterics follow the same stages as [`DefaultLambdaScheduler`](@ref), but
the defaults use single topology with individual scaling of the Coulomb parameters. As in OpenFE,
the charges of the 1-4 interactions are scaled for each state, and the alchemical atoms do not
contribute to the Lennard-Jones dispersion correction. Use `LJsoftcore=:gapsys` and
`Csoftcore=:scaled` to match OpenFE.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const OpenFEScheduler = LambdaScheduler{OpenFESchedule}

"""
    NAMDLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                        Cindividual=false, Cspecial=false, intraLJ=false,
                        Tscaled=true)

Lambda scheduler with overlapping electrostatic and steric stages, similar to the separate
electrostatic and van der Waals windows in NAMD.

Atoms that only exist in the first state lose their charges between `global_λ = 0` and `0.5` and
their Lennard-Jones interactions between `1/3` and `1`. Atoms that only exist in the second state
gain their Lennard-Jones interactions between `0` and `2/3` and their charges between `0.5` and
`1`. Core atoms are interpolated linearly.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const NAMDLambdaScheduler = LambdaScheduler{NAMDSchedule}

"""
    QuartersLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                            Cindividual=false, Cspecial=false, intraLJ=false,
                            Tscaled=true)

Lambda scheduler that removes the atoms of the first state before adding the atoms of the second
state.

Atoms that only exist in the first state lose their charges between `global_λ = 0` and `0.25` and
their Lennard-Jones interactions between `0.25` and `0.5`. Atoms that only exist in the second
state gain their Lennard-Jones interactions between `0.5` and `0.75` and their charges between
`0.75` and `1`. Core atoms are interpolated linearly over the full range.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const QuartersLambdaScheduler = LambdaScheduler{QuartersSchedule}

"""
    EleScaledLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                             Cindividual=false, Cspecial=false, intraLJ=false,
                             Tscaled=true)

Lambda scheduler with the stages of [`DefaultLambdaScheduler`](@ref) and a non-linear
electrostatic coupling.

The charges of atoms that only exist in the first state are coupled by `1 - (2 global_λ)^2`
between `global_λ = 0` and `0.5`, and those of atoms that only exist in the second state by
`sqrt(2 (global_λ - 0.5))` between `0.5` and `1`. The charges therefore change slowly close to
full coupling and quickly close to decoupling.

A [`LambdaScheduler`](@ref), see there for the keyword arguments.
"""
const EleScaledLambdaScheduler = LambdaScheduler{EleScaledSchedule}

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

# Torsions: a torsion of inserted, deleted or environment atoms only is not scaled, the dummy
# atoms keep their own torsions at both end states. One that also has a core atom follows that
# core atom, so only the torsions of one end state act on the core at a time. Without this the
# insert and delete roles win in `mix_default` and a torsion spanning the core and a dummy atom
# is never scaled, which leaves the torsions of both end states on at every λ.
@inline function mix_torsion(roles::Tuple{Vararg{AlchemicalRole}})
    if any(x->x==CoreRole, roles)
        return CoreRole
    elseif any(x->x==CoreIRole, roles)
        return CoreIRole
    elseif any(x->x==CoreDRole, roles)
        return CoreDRole
    else
        return EnvRole
    end
end

# `lj=true` for Lennard-Jones pairs, where `intraLJ` keeps the pairs within one group on,
# `torsion=true` for the torsions, see `mix_torsion`
@inline function mix_roles(inter::Any, roles::Tuple{Vararg{AlchemicalRole}}; lj=false,
                           torsion=false)
    if torsion && inter.Tscaled
        return mix_torsion(roles)
    elseif lj && inter.intraLJ
        return mix_special(roles)
    else
        return mix_default(roles)
    end
end

# Whether a λ bonded interaction is a torsion, set to `true` by the λ torsion types
@inline is_torsion(inter) = false

# End-state parameters an alchemical role interpolates between; insert and delete atoms keep one
# state's parameters at both ends. Branches on the role value: `Val(alch_role)` is not GPU-compilable.
@inline function switchAB(alch_role, A, B)
    if alch_role == DeleteRole
        return A, A
    elseif alch_role == InsertRole
        return B, B
    else
        return A, B
    end
end

# Factor on a specific interaction's virial contribution. The default of 1 is correct whenever
# the force already carries the alchemical scaling.
@inline virial_lambda_factor(inter, atoms) = 1

# `AbsoluteFESystem` and `RelativeFESystem` convert every specific interaction to its λ
# counterpart, so only the λ types can appear on an alchemical atom and only they need the
# coupling supplied.
@inline function virial_lambda_factor(inter::AlchemicalBondedInteraction, atoms)
    λ_glob = λ_mixing(inter.λ_mixing, atoms)
    roles = map(a -> a.alch_role, atoms)
    # A torsion that follows a core atom is scaled in its force already, so its virial needs no
    #   second factor. The terms that are not scaled, those of the dummy atoms only, get the
    #   weight of the end state they belong to here.
    if is_torsion(inter) && inter.scheduler.Tscaled && mix_torsion(roles) != EnvRole
        return one(λ_glob)
    end
    return scale_virial_dual(inter.scheduler, λ_glob, mix_default(roles))
end

# The energy scaling `λ` and the end state weights `λ_params` of a λ bonded interaction. The
#   physics is that of the plain interaction built by `plain_interaction(inter, λ_params)`.
@inline function bonded_lambda(inter::AlchemicalBondedInteraction, atoms)
    T = typeof(ustrip(first(atoms).λ))
    λ_glob = T(λ_mixing(inter.λ_mixing, atoms))
    role = mix_roles(inter.scheduler, map(a -> a.alch_role, atoms); torsion=is_torsion(inter))
    return scale_dual(inter.scheduler, λ_glob, role)
end
