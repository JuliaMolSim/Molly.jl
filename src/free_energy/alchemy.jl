export
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

# SoftCore potential options
abstract type SoftCore end

struct DefaultSoftCore <: SoftCore end
struct BeutlerSoftCore <: SoftCore end
struct GapsysSoftCore  <: SoftCore end
struct ScaledSoftCore  <: SoftCore end

# Lambda Schedulers
"""
    DefaultLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                           Cindividual=false, Cspecial=false, intraLJ=false)

Lambda scheduler that turns off the electrostatics of an atom before its sterics.

Atoms that only exist in the first state lose their charges between `global_λ = 0` and `0.5`
and their Lennard-Jones interactions between `0.5` and `1`. Atoms that only exist in the second
state gain their Lennard-Jones interactions between `0` and `0.5` and their charges between `0.5`
and `1`, so an atom is never charged without its Lennard-Jones core. Core atoms, present in both
states, are interpolated linearly.

Schedulers are passed to [`AbsoluteFESystem`](@ref) and [`RelativeFESystem`](@ref), and all of
them take the keyword arguments below.

# Arguments
- `dual=true`: whether to use dual topology (energy scaling) or single topology (parameter
    scaling).
- `LJindividual=false`: in single topology, whether the Lennard-Jones parameters are scaled for
    each atom before mixing (`true`) or mixed for each state and then interpolated (`false`).
- `LJspecial=false`: the same choice for the Lennard-Jones 1-4 interactions.
- `Cindividual=false`: the same choice for the Coulomb interactions.
- `Cspecial=false`: the same choice for the Coulomb 1-4 interactions.
- `intraLJ=false`: whether the Lennard-Jones interactions between alchemical atoms are kept on
    (`true`) or scaled with the atoms (`false`).

Coulomb interactions between alchemical atoms are always scaled with the atoms, as the PME mesh
scales the charge of each atom.
"""
@kwdef struct DefaultLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
end
"""
    LinearLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                          Cindividual=false, Cspecial=false, intraLJ=false)

Lambda scheduler that scales the electrostatics and sterics of all alchemical atoms linearly and
at the same time, from `global_λ = 0` to `1`.

See [`DefaultLambdaScheduler`](@ref) for the keyword arguments.
"""
@kwdef struct LinearLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
end
"""
    GROMACSLambdaABFEScheduler(; dual=true, LJindividual=false, LJspecial=false,
                               Cindividual=false, Cspecial=false, intraLJ=false)

Lambda scheduler for absolute free energy calculations, following GROMACS.

The charges of the alchemical atoms are scaled between `global_λ = 0` and `0.5` and their
Lennard-Jones interactions between `0.5` and `1`, so decoupled atoms lose their charges first.
With [`AbsoluteFESystem`](@ref), PME is replaced by the two grid scheme of GROMACS, where the
grids of the two end states are mixed with the electrostatic coupling. Inserted atoms follow the
same stages and would gain their charges before their Lennard-Jones interactions, so use
[`DefaultLambdaScheduler`](@ref) for insertions.

See [`DefaultLambdaScheduler`](@ref) for the keyword arguments.
"""
@kwdef struct GROMACSLambdaABFEScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
end
"""
    GROMACSLambdaRBFEScheduler(; dual=true, LJindividual=false, LJspecial=false,
                               Cindividual=false, Cspecial=false, intraLJ=false)

Lambda scheduler for relative free energy calculations, following GROMACS.

The electrostatics and sterics are scaled linearly, as in [`LinearLambdaScheduler`](@ref). With
[`RelativeFESystem`](@ref), PME is replaced by the two grid scheme of GROMACS, where the grids of
the two end states are mixed with the electrostatic coupling.

See [`DefaultLambdaScheduler`](@ref) for the keyword arguments.
"""
@kwdef struct GROMACSLambdaRBFEScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
end
"""
    OpenFEScheduler(; dual=false, LJindividual=false, LJspecial=false,
                    Cindividual=true, Cspecial=false, intraLJ=false)

Lambda scheduler that reproduces the relative free energy setup of
[OpenFE](https://github.com/OpenFreeEnergy/openfe).

The electrostatics and sterics follow the same stages as [`DefaultLambdaScheduler`](@ref), but
the defaults use single topology with individual scaling of the Coulomb parameters. As in OpenFE,
the charges of the 1-4 interactions are scaled for each state, and the alchemical atoms do not
contribute to the Lennard-Jones dispersion correction. Use `LJsoftcore="gapsys"` and
`Csoftcore="scaled"` to match OpenFE.

See [`DefaultLambdaScheduler`](@ref) for the keyword arguments.
"""
@kwdef struct OpenFEScheduler
    dual::Bool = false
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = true
    Cspecial::Bool = false
    intraLJ::Bool = false
end
"""
    NAMDLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                        Cindividual=false, Cspecial=false, intraLJ=false)

Lambda scheduler with overlapping electrostatic and steric stages, similar to the separate
electrostatic and van der Waals windows in NAMD.

Atoms that only exist in the first state lose their charges between `global_λ = 0` and `0.5` and
their Lennard-Jones interactions between `1/3` and `1`. Atoms that only exist in the second state
gain their Lennard-Jones interactions between `0` and `2/3` and their charges between `0.5` and
`1`. Core atoms are interpolated linearly.

See [`DefaultLambdaScheduler`](@ref) for the keyword arguments.
"""
@kwdef struct NAMDLambdaScheduler
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
end
"""
    QuartersLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                            Cindividual=false, Cspecial=false, intraLJ=false)

Lambda scheduler that removes the atoms of the first state before adding the atoms of the second
state.

Atoms that only exist in the first state lose their charges between `global_λ = 0` and `0.25` and
their Lennard-Jones interactions between `0.25` and `0.5`. Atoms that only exist in the second
state gain their Lennard-Jones interactions between `0.5` and `0.75` and their charges between
`0.75` and `1`. Core atoms are interpolated linearly over the full range.

See [`DefaultLambdaScheduler`](@ref) for the keyword arguments.
"""
@kwdef struct QuartersLambdaScheduler 
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
end
"""
    EleScaledLambdaScheduler(; dual=true, LJindividual=false, LJspecial=false,
                             Cindividual=false, Cspecial=false, intraLJ=false)

Lambda scheduler with the stages of [`DefaultLambdaScheduler`](@ref) and a non-linear
electrostatic coupling.

The charges of atoms that only exist in the first state are coupled by `1 - (2 global_λ)^2`
between `global_λ = 0` and `0.5`, and those of atoms that only exist in the second state by
`sqrt(2 (global_λ - 0.5))` between `0.5` and `1`. The charges therefore change slowly close to
full coupling and quickly close to decoupling.

See [`DefaultLambdaScheduler`](@ref) for the keyword arguments.
"""
@kwdef struct EleScaledLambdaScheduler 
    dual::Bool = true
    LJindividual::Bool = false
    LJspecial::Bool = false
    Cindividual::Bool = false
    Cspecial::Bool = false
    intraLJ::Bool = false
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
    if type=="LJ" && inter.intraLJ
        return mix_special(roles)
    else
        return mix_default(roles)
    end
end

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