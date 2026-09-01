# Taking gradients with respect to force field parameters

export
    parameter_prefix,
    parameter_fields,
    ParameterPlan,
    extract_parameters,
    inject_gradients

"""
    parameter_prefix(inter)
    parameter_prefix(inter, inter_type)

The prefix that the parameters of an interaction have in a parameter dictionary, for
example `"inter_HB_CT/CT_"`.

Return `nothing`, the default, for an interaction with no differentiable parameters.
Specific interactions are given the interaction type as a second argument, so their keys
can depend on it. Define this along with [`parameter_fields`](@ref) to make a custom
interaction work with [`extract_parameters`](@ref) and `inject_gradients`.
"""
parameter_prefix(inter) = nothing
parameter_prefix(inter, inter_type) = parameter_prefix(inter)

"""
    parameter_fields(T)

A tuple of `(field, key_suffix)` pairs naming the differentiable parameters of an
interaction type, for example `((:k, "k"), (:r0, "r0"))`.

The key of a parameter is [`parameter_prefix`](@ref) followed by its suffix. Interactions
whose parameters do not correspond one-to-one with fields, such as
[`PeriodicTorsion`](@ref), instead define `parameter_keys`, `parameter_values` and
`inject_parameters` directly.
"""
parameter_fields(::Type) = ()
parameter_fields(x) = parameter_fields(typeof(x))

@inline n_parameters(::Type{T}) where {T} = length(parameter_fields(T))
@inline n_parameters(x) = n_parameters(typeof(x))
@inline n_parameters_val(::Type{T}) where {T} = Val(n_parameters(T))

# Key suffixes, in the order the values are given in
@inline parameter_keys(::Type{T}) where {T} = map(last, parameter_fields(T))
@inline parameter_keys(x) = parameter_keys(typeof(x))

# Current values of the parameters, in the same order
@inline parameter_values(inter) = map(f -> getfield(inter, first(f)), parameter_fields(inter))

"""
    inject_parameters(inter, values)

A copy of `inter` with its differentiable parameters replaced by `values`, which are in the
order given by [`parameter_fields`](@ref).
"""
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
            push!(args, :(convert($(fieldtype(T, i)), vals[$(pos[fname])])))
        else
            push!(args, :(getfield(inter, $i)))
        end
    end
    return Expr(:call, T, args...)
end

# Atoms are handled like an interaction with a per-atom-type prefix
const atom_parameter_keys = ("mass", "σ", "ϵ")

@inline atom_parameter_values(at) = (at.mass, at.σ, at.ϵ)

@inline function inject_atom(at, m, s, e)
    return Atom(at.index, at.atom_type, convert(typeof(at.mass), m), at.charge,
                convert(typeof(at.σ), s), convert(typeof(at.ϵ), e), at.λ, at.alch_role)
end

"""
    ParameterPlan(sys, params_dic)

The mapping from the entries of `params_dic` to the parameters of `sys`, built
once and reused on every call to `inject_gradients`.

Building the plan is where the dictionary keys are formed and looked up. A plan is valid
for any dictionary with the same keys, so it can be reused across a gradient calculation
and across finite differencing, but it must be rebuilt if the set of keys or the system
changes.
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

# One block of the flat vector: `n_elements` entries for each of the parameters named by
# `suffixes`, laid out parameter by parameter so each is a contiguous view
function plan_block!(plan_state, prefixes, values_per_element, suffixes)
    keys_v, key_index, params_dic, slots, defaults, ranges = plan_state
    n_params = length(suffixes)
    for f in 1:n_params
        start = length(slots) + 1
        for (i, prefix) in enumerate(prefixes)
            key = prefix * suffixes[f]
            push!(slots, plan_slot!(keys_v, key_index, params_dic, key))
            push!(defaults, values_per_element[i][f])
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
        prefixes = [string("atom_", ad.atom_type, "_") for ad in sys.atoms_data]
        plan_block!(state, prefixes, map(atom_parameter_values, atoms_cpu), atom_parameter_keys)
    else
        # No atom data, so no atom types to key on
        for _ in atom_parameter_keys
            push!(ranges, (length(slots) + 1):length(slots))
        end
    end

    # Specific interaction lists
    for il in values(sys.specific_inter_lists)
        push!(target_starts, length(ranges))
        inters_cpu = from_device(il.inters)
        suffixes = parameter_keys(eltype(inters_cpu))
        if length(suffixes) == 0 || isnothing(parameter_prefix(first(inters_cpu), ""))
            for _ in suffixes
                push!(ranges, (length(slots) + 1):length(slots))
            end
        else
            prefixes = [parameter_prefix(inter, t) for (inter, t) in zip(inters_cpu, il.types)]
            plan_block!(state, prefixes, map(parameter_values, inters_cpu), suffixes)
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
            plan_block!(state, [prefix], [parameter_values(inter)], suffixes)
        end
    end

    inters_host = map(il -> from_device(il.inters), values(sys.specific_inter_lists))
    return ParameterPlan{V, typeof(atoms_cpu), typeof(inters_host)}(
        keys_v, slots, defaults, ranges, target_starts, atoms_cpu, inters_host)
end

inject_interaction(inter, args...) = inter

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
    m, s, e = target_views(flat, plan, 1, Val(3))
    return to_device(inject_atom.(plan.atoms_host, m, s, e), AT)
end

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
    m_host = first(target_views(flat, plan, 1, Val(3)))
    masses_host = convert.(eltype(sys.masses), m_host)
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
        haskey(params_dic, key) || (params_dic[key] = values_t[i])
    end
    return params_dic
end

"""
    extract_parameters(sys, ff=nothing)

A dictionary of the force field parameters of a system, keyed by strings, for use with
`inject_gradients`.

An interaction contributes its parameters if it defines [`parameter_prefix`](@ref) and
[`parameter_fields`](@ref).
"""
function extract_parameters(sys::System{<:Any, <:Any, T}, ff=nothing) where T
    params_dic = Dict{String, T}()

    atoms_cpu = from_device(sys.atoms)
    if length(sys.atoms_data) == length(atoms_cpu)
        for (at, ad) in zip(atoms_cpu, sys.atoms_data)
            extract_block!(params_dic, string("atom_", ad.atom_type, "_"),
                           atom_parameter_keys, atom_parameter_values(at))
        end
    end

    for il in values(sys.specific_inter_lists)
        inters_cpu = from_device(il.inters)
        for (inter, t) in zip(inters_cpu, il.types)
            extract_block!(params_dic, parameter_prefix(inter, t), parameter_keys(inter),
                           parameter_values(inter))
        end
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
