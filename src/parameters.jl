# Taking gradients with respect to force field parameters

export
    ParameterPlan,
    inject_gradients,
    extract_parameters

@public parameter_prefix,
    parameter_fields

"""
    parameter_prefix(inter)
    parameter_prefix(inter, inter_type)

The prefix that the parameters of an interaction have in a parameter dictionary, for
example `"inter_HB_CT/CT_"`.

Return `nothing`, the default, for an interaction with no differentiable parameters.
Specific interactions are given the interaction type as a second argument, so their keys
can depend on it. Define this along with [`Molly.parameter_fields`](@ref) to make a custom
interaction work with [`extract_parameters`](@ref) and [`inject_gradients`](@ref).
"""
parameter_prefix(inter) = nothing
parameter_prefix(inter, inter_type) = parameter_prefix(inter)

"""
    parameter_fields(T)

A tuple of `(field, key_suffix)` pairs naming the differentiable parameters of an
interaction type, for example `((:k, "k"), (:r0, "r0"))`.

The key of a parameter is [`Molly.parameter_prefix`](@ref) followed by its suffix. Interactions
whose parameters do not correspond one-to-one with fields, such as
[`PeriodicTorsion`](@ref), instead define `Molly.n_parameters`, `Molly.parameter_keys`,
`Molly.parameter_values` and `Molly.inject_parameters` directly.
"""
parameter_fields(::Type) = ()
parameter_fields(x) = parameter_fields(typeof(x))

@inline n_parameters(::Type{T}) where {T} = length(parameter_fields(T))
@inline n_parameters(x) = n_parameters(typeof(x))
@inline n_parameters_val(::Type{T}) where {T} = Val(n_parameters(T))

# Key suffixes, in the order the values are given in
@inline parameter_keys(::Type{T}) where {T} = map(last, parameter_fields(T))
@inline parameter_keys(x) = parameter_keys(typeof(x))

# Current values of the parameters, in the same order, with any units they are stored with
@inline parameter_values(inter) = map(f -> getfield(inter, first(f)), parameter_fields(inter))

# Convert a parameter value to the type of the field it replaces. On a system with units,
#   a plain number is taken to be in the units of the field, which are re-attached, and a
#   value with units is converted to them. Otherwise this is `convert`, so the unitless and
#   differentiable path is unchanged.
@inline convert_parameter(::Type{T}, x) where {T} = convert(T, x)
@inline convert_parameter(::Type{T}, x::Real) where {T <: Unitful.AbstractQuantity} =
    convert(T, x * unit(T))

# A field that is not a parameter, as it is put into the injected copy of an interaction.
#   `LambdaScheduler` rebuilds itself field by field, since Enzyme cannot deduce the type of a
#   plain copy of its `Bool`s and errors on λ interactions with no other such field.
@inline copied_field(x) = x

# A copy of `inter` with its differentiable parameters replaced by `values`, which are in the
#   order given by `parameter_fields`, either plain numbers in the units of the fields or
#   values with units
@inline inject_parameters(inter, vals...) = inject_parameters(inter, vals)

@generated function inject_parameters(inter::T, vals::Tuple) where {T}
    fields = parameter_fields(T)
    n = length(fields)
    if length(vals.parameters) != n
        return :(throw(ArgumentError("expected $($n) parameter values for $($T)")))
    end
    n == 0 && return :(inter)
    pos = Dict{Symbol, Int}(first(f) => i for (i, f) in enumerate(fields))
    args = Expr[]
    for (i, fname) in enumerate(fieldnames(T))
        if haskey(pos, fname)
            push!(args, :(convert_parameter($(fieldtype(T, i)), vals[$(pos[fname])])))
        else
            push!(args, :(copied_field(getfield(inter, $i))))
        end
    end
    return Expr(:call, T, args...)
end

# Atoms are handled like an interaction with one element per atom, keyed by the atom type
#   except for the charge, which is keyed by the charge type of the atom
@inline atom_parameter_values(at) = (at.mass, at.charge, at.σ, at.ϵ)

function atom_parameter_keys(ad)
    prefix = string("atom_", ad.atom_type, "_")
    charge_type = (isempty(ad.charge_type) ? ad.atom_type : ad.charge_type)
    return (prefix * "mass", string("atom_", charge_type, "_charge"), prefix * "σ", prefix * "ϵ")
end

@inline function inject_atom(at, m, q, s, e)
    return Atom(at.index, at.atom_type, convert_parameter(typeof(at.mass), m),
                convert_parameter(typeof(at.charge), q), convert_parameter(typeof(at.σ), s),
                convert_parameter(typeof(at.ϵ), e), at.λ, at.alch_role)
end

# Atom keys whose atoms do not all have the same value, for example the masses of a type
#   after hydrogen mass repartitioning or the zero masses of the B copies in a dual topology
#   system. A single value would overwrite those values, so these keys are not parameters.
function inconsistent_atom_keys(atoms, atoms_data)
    first_values = Dict{String, Any}()
    inconsistent = Set{String}()
    for (at, ad) in zip(atoms, atoms_data)
        for (key, val) in zip(atom_parameter_keys(ad), atom_parameter_values(at))
            get!(first_values, key, val) == val || push!(inconsistent, key)
        end
    end
    return inconsistent
end

# Single topology systems store the parameters of both end states in each atom
function check_scalar_atoms(atoms)
    if any(at -> any(v -> v isa Tuple, atom_parameter_values(at)), atoms)
        throw(ArgumentError("atoms with parameters for two end states, as in a single " *
                            "topology RelativeFESystem, are not supported; use dual topology"))
    end
end

# Whether an interaction stores the parameters of both end states, as the λ interactions of a
#   single topology system do; such an interaction has no single value per key
single_topology(inter) = false

function check_dual_topology(inters)
    if any(single_topology, inters)
        throw(ArgumentError("interactions with parameters for two end states, as in a single " *
                            "topology RelativeFESystem, are not supported; use dual topology"))
    end
end

# Keys of specific interactions whose interactions do not all have the same value, for example
#   a bond type that the two end states of a dual topology system parametrise differently. A
#   single value would overwrite some of them, so these keys are not parameters.
function inconsistent_inter_keys(specific_inter_lists)
    first_values = Dict{String, Any}()
    inconsistent = Set{String}()
    for il in specific_inter_lists
        for (inter, t) in zip(from_device(il.inters), il.types)
            prefix = parameter_prefix(inter, t)
            isnothing(prefix) && continue
            for (suffix, val) in zip(parameter_keys(inter), parameter_values(inter))
                get!(first_values, prefix * suffix, val) == val || push!(inconsistent, prefix * suffix)
            end
        end
    end
    return inconsistent
end

"""
    ParameterPlan(sys, params_dic)

The mapping from the entries of `params_dic` to the parameters of `sys`, built
once and reused on every call to [`inject_gradients`](@ref).

Atom keys are described in [`extract_parameters`](@ref). It is an error to give an atom key
whose atoms have different values, or an interaction key whose interactions have different
values, which [`extract_parameters`](@ref) leaves out, or to use a single topology system.

Building the plan is where the dictionary keys are formed and looked up. A plan is valid
for any dictionary with the same keys, so it can be reused across a gradient calculation
and across finite differencing, but it must be rebuilt if the set of keys or the system
changes.

`sys` may have units. The dictionary values are then plain numbers in the units that each
parameter is stored with in `sys`, as returned by [`extract_parameters`](@ref).
"""
struct ParameterPlan{V, A, I}
    keys::Vector{String}           # Distinct parameters that this system uses, in slot order
    slots::Vector{Int32}           # Slot of each flat entry, 0 to keep the element's own value
    defaults::Vector{V}            # The element's own value for each flat entry
    ranges::Vector{UnitRange{Int}} # One per (target, parameter), into the flat vector
    target_starts::Vector{Int}     # Target t owns ranges[target_starts[t] + 1 : ...]
    atoms_host::A                  # Host copies of the elements being injected into, so that
    inters_host::I                 #   a GPU run does not copy them back every call
end

Base.length(plan::ParameterPlan) = length(plan.keys)

function Base.show(io::IO, plan::ParameterPlan)
    print(io, "ParameterPlan with ", length(plan.keys), " parameters over ",
          length(plan.slots), " slots")
end

@inline function target_views(flat, plan::ParameterPlan, target::Integer, ::Val{N}) where {N}
    start = @inbounds plan.target_starts[target]
    return ntuple(i -> @inbounds(view(flat, plan.ranges[start + i])), Val(N))
end

@inline function target_scalars(flat, plan::ParameterPlan, target::Integer, ::Val{N}) where {N}
    start = @inbounds plan.target_starts[target]
    return ntuple(i -> @inbounds(flat[first(plan.ranges[start + i])]), Val(N))
end

function plan_value_type(params_dic, sys)
    V = valtype(params_dic)
    isconcretetype(V) && V <: Number && return V
    isempty(params_dic) && return float_type(sys)
    return mapreduce(typeof, promote_type, values(params_dic))
end

# Slot of a key, allocating a new one the first time a key that the dictionary has is seen
function plan_slot!(keys_v, key_index, params_dic, key)
    s = get(key_index, key, Int32(0))
    s != 0 && return s
    haskey(params_dic, key) || return Int32(0)
    push!(keys_v, key)
    s = Int32(length(keys_v))
    key_index[key] = s
    return s
end

# One block of the flat vector: one entry per element for each of `n_params` parameters,
# laid out parameter by parameter so each is a contiguous view
function plan_block!(plan_state, keys_per_element, values_per_element, n_params)
    keys_v, key_index, params_dic, slots, defaults, ranges = plan_state
    for f in 1:n_params
        start = length(slots) + 1
        for i in eachindex(keys_per_element)
            key = keys_per_element[i][f]
            push!(slots, plan_slot!(keys_v, key_index, params_dic, key))
            push!(defaults, ustrip(values_per_element[i][f]))
        end
        push!(ranges, start:length(slots))
    end
    return n_params
end

function ParameterPlan(sys::System, params_dic)
    V = plan_value_type(params_dic, sys)
    keys_v, key_index = String[], Dict{String, Int32}()
    slots, defaults = Int32[], V[]
    ranges, target_starts = UnitRange{Int}[], Int[]
    state = (keys_v, key_index, params_dic, slots, defaults, ranges)

    # Atoms
    push!(target_starts, length(ranges))
    atoms_cpu = from_device(sys.atoms)
    if length(sys.atoms_data) == length(atoms_cpu)
        check_scalar_atoms(atoms_cpu)
        for key in inconsistent_atom_keys(atoms_cpu, sys.atoms_data)
            if haskey(params_dic, key)
                throw(ArgumentError("the atoms with parameter $key have different values, " *
                    "which it would overwrite; remove it from the dictionary or give them " *
                    "different atom or charge types"))
            end
        end
        plan_block!(state, map(atom_parameter_keys, sys.atoms_data),
                    map(atom_parameter_values, atoms_cpu), 4)
    else
        # No atom data, so no atom types to key on
        for _ in 1:4
            push!(ranges, (length(slots) + 1):length(slots))
        end
    end

    # Specific interaction lists
    for key in inconsistent_inter_keys(values(sys.specific_inter_lists))
        if haskey(params_dic, key)
            throw(ArgumentError("the interactions with parameter $key have different values, " *
                "which it would overwrite; remove it from the dictionary or give them " *
                "different types"))
        end
    end
    for il in values(sys.specific_inter_lists)
        push!(target_starts, length(ranges))
        inters_cpu = from_device(il.inters)
        check_dual_topology(inters_cpu)
        suffixes = parameter_keys(eltype(inters_cpu))
        if length(suffixes) == 0 || isnothing(parameter_prefix(first(inters_cpu), ""))
            for _ in suffixes
                push!(ranges, (length(slots) + 1):length(slots))
            end
        else
            keys = [map(suffix -> parameter_prefix(inter, t) * suffix, suffixes)
                    for (inter, t) in zip(inters_cpu, il.types)]
            plan_block!(state, keys, map(parameter_values, inters_cpu), length(suffixes))
        end
    end

    # Pairwise and general interactions, one element each
    for inter in (values(sys.pairwise_inters)..., values(sys.general_inters)...)
        push!(target_starts, length(ranges))
        suffixes = parameter_keys(typeof(inter))
        prefix = parameter_prefix(inter)
        if length(suffixes) == 0 || isnothing(prefix)
            for _ in suffixes
                push!(ranges, (length(slots) + 1):length(slots))
            end
        else
            plan_block!(state, [map(suffix -> prefix * suffix, suffixes)],
                        [parameter_values(inter)], length(suffixes))
        end
    end

    inters_host = map(il -> from_device(il.inters), values(sys.specific_inter_lists))
    return ParameterPlan{V, typeof(atoms_cpu), typeof(inters_host)}(
        keys_v, slots, defaults, ranges, target_starts, atoms_cpu, inters_host)
end

inject_interaction(inter, args...) = inter

# A copy of a general interaction that is consistent with new atom charges, for an
#   interaction that caches values computed from the charges
update_charges(inter, atoms) = inter

# The values of the plan's parameters, in slot order
function plan_values(params_dic, plan::ParameterPlan{V}) where {V}
    ks = plan.keys
    vals = Vector{V}(undef, length(ks))
    @inbounds for i in eachindex(ks)
        vals[i] = dict_get(params_dic, ks[i], zero(V))
    end
    return vals
end

# One value per (element, parameter), taking the element's own value where the dictionary
# has no entry. This is the only place the dictionary is touched, once per distinct key
# rather than once per element.
function parameter_flat(params_dic, plan::ParameterPlan{V}) where {V}
    vals = plan_values(params_dic, plan)
    slots, defaults = plan.slots, plan.defaults
    flat = Vector{V}(undef, length(slots))
    @inbounds for j in eachindex(slots)
        s = slots[j]
        flat[j] = (s > 0 ? vals[s] : defaults[j])
    end
    return flat
end

@inline function inject_atoms(sys, flat, plan, ::Type{AT}) where {AT}
    isempty(plan.ranges[1]) && return sys.atoms
    m, q, s, e = target_views(flat, plan, 1, Val(4))
    return to_device(inject_atom.(plan.atoms_host, m, q, s, e), AT)
end

# Whether the dictionary sets the charge of any atom
@inline injects_charges(plan::ParameterPlan) = !isempty(plan.ranges[2]) && any(!iszero, view(plan.slots, plan.ranges[2]))

@inline function inject_list(il, inters_host, flat, plan, target, ::Type{AT}) where {AT}
    N = n_parameters_val(eltype(inters_host))
    return inject_list_n(il, inters_host, flat, plan, target, N, AT)
end

@inline inject_list_n(il, inters_host, flat, plan, target, ::Val{0}, ::Type{AT}) where {AT} = il

@inline function inject_list_n(il, inters_host, flat, plan, target, ::Val{N},
                               ::Type{AT}) where {N, AT}
    vs = target_views(flat, plan, target, Val(N))
    isempty(first(vs)) && return il
    return replace_inters(il, to_device(inject_parameters.(inters_host, vs...), AT))
end

@inline replace_inters(il::InteractionList1Atoms, inters) =
    InteractionList1Atoms(il.is, inters, il.types, il.data)
@inline replace_inters(il::InteractionList2Atoms, inters) =
    InteractionList2Atoms(il.is, il.js, inters, il.types, il.data)
@inline replace_inters(il::InteractionList3Atoms, inters) =
    InteractionList3Atoms(il.is, il.js, il.ks, inters, il.types, il.data)
@inline replace_inters(il::InteractionList4Atoms, inters) =
    InteractionList4Atoms(il.is, il.js, il.ks, il.ls, inters, il.types, il.data)
@inline replace_inters(il::InteractionList5Atoms, inters) =
    InteractionList5Atoms(il.is, il.js, il.ks, il.ls, il.ms, inters, il.types, il.data)

@inline function inject_inter(inter, flat, plan, target)
    N = n_parameters_val(typeof(inter))
    return inject_inter_n(inter, flat, plan, target, N)
end

@inline inject_inter_n(inter, flat, plan, target, ::Val{0}) = inter

@inline function inject_inter_n(inter, flat, plan, target, ::Val{N}) where {N}
    isempty(plan.ranges[plan.target_starts[target] + 1]) && return inter
    return inject_parameters(inter, target_scalars(flat, plan, target, Val(N)))
end

@inline inject_lists(::Tuple{}, ::Tuple, flat, plan, target, ::Type) = ()
@inline inject_lists(ils::Tuple, hosts::Tuple, flat, plan, target, ::Type{AT}) where {AT} =
    (inject_list(first(ils), first(hosts), flat, plan, target, AT),
     inject_lists(Base.tail(ils), Base.tail(hosts), flat, plan, target + 1, AT)...)

@inline inject_inters(::Tuple{}, flat, plan, target) = ()
@inline inject_inters(inters::Tuple, flat, plan, target) =
    (inject_inter(first(inters), flat, plan, target),
     inject_inters(Base.tail(inters), flat, plan, target + 1)...)

# General interactions may instead define `inject_interaction(inter, params_dic, sys)`, for
# parameters that are not fields of the interaction - implicit solvent reads per-element
# values from the system. `parameter_prefix` returns a literal per type, so this branch is
# resolved at compile time.
@inline inject_generals(::Tuple{}, flat, plan, target, params_dic, sys) = ()

@inline function inject_generals(inters::Tuple, flat, plan, target, params_dic, sys)
    inter = first(inters)
    injected = if isnothing(parameter_prefix(inter))
        inject_interaction(inter, params_dic, sys)
    else
        inject_inter(inter, flat, plan, target)
    end
    return (injected,
            inject_generals(Base.tail(inters), flat, plan, target + 1, params_dic, sys)...)
end

@inline function injected_masses(sys, flat, plan, ::Type{AT}) where {AT}
    isempty(plan.ranges[1]) && return sys.masses, sys.total_mass
    m_host = first(target_views(flat, plan, 1, Val(4)))
    masses_host = convert_parameter.(eltype(sys.masses), m_host)
    return to_device(masses_host, AT), sum(masses_host)
end

# Construct directly to avoid large compile times from validation, which has already been done
@inline function system_with_parameters(sys::System{D, AT, T, TH}, atoms, coords,
                                        velocities, pis, sis, gis, masses,
                                        total_mass) where {D, AT, T, TH}
    return System{D, AT, T, TH, typeof(atoms), typeof(coords), typeof(sys.boundary),
                  typeof(velocities), typeof(sys.atoms_data), typeof(sys.topology),
                  typeof(pis), typeof(sis), typeof(gis), typeof(sys.constraints),
                  typeof(sys.virtual_sites), typeof(sys.virtual_site_flags),
                  typeof(sys.neighbor_finder), typeof(sys.loggers), typeof(sys.force_units),
                  typeof(sys.energy_units), typeof(sys.k), typeof(masses),
                  typeof(total_mass), typeof(sys.data)}(
        atoms, coords, sys.boundary, velocities, sys.atoms_data, sys.topology, pis, sis,
        gis, sys.constraints, sys.virtual_sites, sys.virtual_site_flags, sys.neighbor_finder,
        sys.loggers, sys.df, sys.force_units, sys.energy_units, sys.k, masses, total_mass,
        sys.data, sys.grad_safe, sys.launch_config)
end

"""
    inject_gradients(sys, params_dic)
    inject_gradients(sys, params_dic, plan)
    inject_gradients(sys, params_dic, plan, coords)

A copy of `sys` with the parameters in `params_dic` injected, for use inside a function
being differentiated with respect to `params_dic`.

`plan` is a [`ParameterPlan`](@ref) resolving the dictionary keys to the parameters of the
system. Build it once outside the function being differentiated and pass it in; the
two-argument form builds one on every call, which is convenient but much slower. `coords`
replaces the coordinates of `sys`, so that a gradient can also be taken with respect to
them.

`sys` may have units, in which case the values in `params_dic` are plain numbers in the
units that each parameter is stored with in `sys`, as returned by
[`extract_parameters`](@ref). The unit of each field is re-attached when it is replaced.
Gradients are only supported for systems without units.

When charges are injected, the charge sums that a [`PME`](@ref) with `fixed_charges=true`
caches are recomputed from the new charges.
"""
inject_gradients(sys::System, params_dic) =
    inject_gradients(sys, params_dic, ParameterPlan(sys, params_dic))

inject_gradients(sys::System, params_dic, plan::ParameterPlan) =
    inject_gradients(sys, params_dic, plan, sys.coords)

function inject_gradients(sys::System{<:Any, AT}, params_dic, plan::ParameterPlan,
                          coords) where {AT}
    flat = parameter_flat(params_dic, plan)

    atoms = inject_atoms(sys, flat, plan, AT)
    # Walked recursively rather than with `ntuple`, so that the heterogeneous tuples are
    # unrolled and each element keeps its concrete type
    n_lists = length(sys.specific_inter_lists)
    n_pairwise = length(sys.pairwise_inters)
    sis = inject_lists(sys.specific_inter_lists, plan.inters_host, flat, plan, 2, AT)
    pis = inject_inters(sys.pairwise_inters, flat, plan, 2 + n_lists)
    gis = inject_generals(sys.general_inters, flat, plan, 2 + n_lists + n_pairwise,
                          params_dic, sys)
    if injects_charges(plan)
        gis = map(inter -> update_charges(inter, atoms), gis)
    end

    masses, total_mass = injected_masses(sys, flat, plan, AT)
    velocities = copy(sys.velocities)
    return system_with_parameters(sys, atoms, coords, velocities, pis, sis, gis,
                                  masses, total_mass)
end

# Add every parameter of an interaction to the dictionary, keeping the first value seen for
# a repeated key so that the dictionary holds one entry per interaction type
function extract_block!(params_dic, prefix, keys_t, values_t)
    isnothing(prefix) && return params_dic
    @inbounds for i in eachindex(keys_t)
        key = prefix * keys_t[i]
        haskey(params_dic, key) || (params_dic[key] = ustrip(values_t[i]))
    end
    return params_dic
end

"""
    extract_parameters(sys, ff=nothing; strictness=:warn)

A dictionary of the force field parameters of a system, keyed by strings, for use with
[`inject_gradients`](@ref).

Atoms contribute `atom_<type>_mass`, `atom_<type>_σ` and `atom_<type>_ϵ` for each atom type,
and `atom_<charge type>_charge` for each charge type, the `charge_type` of
[`AtomData`](@ref) (the atom type if empty). A key whose atoms have different values, for
example masses after hydrogen mass repartitioning, is left out, so that injecting the
dictionary leaves those values unchanged. Leaving out a charge, σ or ϵ key is reported
according to `strictness`: `:warn` to emit a warning (the default), `:nowarn` to suppress it
or `:error` to error; leaving out a mass key is not reported. Likewise, an interaction key whose
interactions have different values, for example a bond type that the two end states of a
[`RelativeFESystem`](@ref) parametrise differently, is left out and reported. Single topology
systems, whose atoms and interactions store the parameters of two end states, are not
supported.
An interaction contributes its parameters if it defines [`Molly.parameter_prefix`](@ref)
and [`Molly.parameter_fields`](@ref). The λ interactions of an [`AbsoluteFESystem`](@ref) or
a dual topology [`RelativeFESystem`](@ref) have the keys of the interactions they replace, so
an alchemical system has the keys of the systems it was built from.

The values are plain numbers of the float type of the system. On a system with units they
are in the units that each parameter is stored with in the system, for example kJ mol⁻¹ nm⁻²
for a bond force constant in a system set up with the default units, and the units are
re-attached by [`ParameterPlan`](@ref) and [`inject_gradients`](@ref).
"""
function extract_parameters(sys::System{<:Any, <:Any, T}, ff=nothing;
                            strictness=default_strictness()) where T
    check_strictness(strictness)
    params_dic = Dict{String, T}()

    atoms_cpu = from_device(sys.atoms)
    if length(sys.atoms_data) == length(atoms_cpu)
        check_scalar_atoms(atoms_cpu)
        for (at, ad) in zip(atoms_cpu, sys.atoms_data)
            extract_block!(params_dic, "", atom_parameter_keys(ad), atom_parameter_values(at))
        end
        inconsistent = inconsistent_atom_keys(atoms_cpu, sys.atoms_data)
        for key in inconsistent
            delete!(params_dic, key)
        end
        # Masses that differ within a type are expected, e.g. hydrogen mass repartitioning
        left_out = sort!([k for k in inconsistent if !endswith(k, "_mass")])
        if !isempty(left_out)
            report_issue("the atoms of each of these parameters have different values, so " *
                "they are left out: $(join(left_out, ", ")); give the atoms different atom types " *
                "or charge types to make them parameters", strictness)
        end
    end

    for il in values(sys.specific_inter_lists)
        inters_cpu = from_device(il.inters)
        check_dual_topology(inters_cpu)
        for (inter, t) in zip(inters_cpu, il.types)
            extract_block!(params_dic, parameter_prefix(inter, t), parameter_keys(inter),
                           parameter_values(inter))
        end
    end
    inconsistent_inters = sort!(collect(inconsistent_inter_keys(values(sys.specific_inter_lists))))
    if !isempty(inconsistent_inters)
        foreach(key -> delete!(params_dic, key), inconsistent_inters)
        report_issue("the interactions of each of these parameters have different values, so " *
            "they are left out: $(join(inconsistent_inters, ", ")); give the interactions " *
            "different types to make them parameters", strictness)
    end

    for inter in (values(sys.pairwise_inters)..., values(sys.general_inters)...)
        if isnothing(parameter_prefix(inter))
            extract_parameters!(params_dic, inter, ff)
        else
            extract_block!(params_dic, parameter_prefix(inter), parameter_keys(inter),
                           parameter_values(inter))
        end
    end

    return params_dic
end

# Allow custom function
extract_parameters!(params_dic, inter, ff) = params_dic
