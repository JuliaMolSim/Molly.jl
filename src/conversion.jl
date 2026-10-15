# Conversion of a System to another float type, to unitless values or to another device,
#   done by `System(sys; float_type, units, array_type)`. Methods for unit handling are in
#   units.jl and methods for specific types are in the files defining them.

#=
    SystemConversion

The target of a System conversion, passed to `convert_component`.

The type parameters are the new float type, array type and `float_type_high`.
`strip_units` and `grad_safe` are the settings of the new system, and `same_device` is
whether it is on the same device as the original one.
`atoms`, `coords` and `boundary` are the converted values, available when converting
the parts of the system that depend on them such as the neighbor finder and PME.
=#
struct SystemConversion{T, AT, TH, A, C, B}
    strip_units::Bool
    grad_safe::Bool
    same_device::Bool
    atoms::A # Needed for PME constructors
    coords::C # Needed for CellListMapNeighborFinder
    boundary::B # For PME and CellListMapNeighborFinder
end

function SystemConversion{T, AT, TH}(strip_units, grad_safe, same_device=true, atoms=nothing,
                                     coords=nothing, boundary=nothing) where {T, AT, TH}
    return SystemConversion{T, AT, TH, typeof(atoms), typeof(coords), typeof(boundary)}(
                strip_units, grad_safe, same_device, atoms, coords, boundary)
end

#=
    convert_component(conversion, x)

Convert `x`, part of a System, to the float type, units and device given by a
`SystemConversion`.

This is used by `System(sys; float_type, units, array_type)`.
Floats are converted to the new float type.
Values with units are converted the same way and have their units stripped if required,
in which case stored units become `NoUnits`.
Integers, strings, symbols and functions are returned unchanged.
Tuples, named tuples and static arrays are converted element-wise.
Other arrays are converted element-wise on the CPU and moved to the new device if their
elements are bits types.
Other structs are rebuilt from their converted fields by calling the constructor of the same
name with the fields in order, and an error is thrown if that fails or gives a struct with
different field values.
Custom types for which this does not work, for example because they hold buffers or have
an inner constructor taking different arguments, should define a method of this function.
=#
convert_component(c::SystemConversion, x) = convert_struct(c, x)

convert_component(::SystemConversion{T}, x::AbstractFloat) where {T} = T(x)
convert_component(::SystemConversion{T}, x::Complex{<:AbstractFloat}) where {T} = Complex{T}(x)

const UnconvertedComponent = Union{Number, AbstractString, Symbol, Nothing, Function, Type}

convert_component(::SystemConversion, x::UnconvertedComponent) = x

convert_component(c::SystemConversion, x::Union{Tuple, NamedTuple}) = map(y -> convert_component(c, y), x)
convert_component(c::SystemConversion, x::StaticArray) = map(y -> convert_component(c, y), x)

# Dictionaries are converted value by value and stay on the CPU
function convert_component(c::SystemConversion, x::AbstractDict)
    return Dict(k => convert_component(c, v) for (k, v) in x)
end

function convert_component(c::SystemConversion{<:Any, AT}, x::AbstractArray) where AT
    y = map(el -> convert_component(c, el), from_device(x))
    if AT <: AbstractGPUArray && isbitstype(eltype(y))
        return to_device(y, AT)
    else
        return y
    end
end

function convert_struct(c::SystemConversion, x::X) where X
    n = fieldcount(X)
    if !isstructtype(X) || n == 0
        return x
    end
    fields = ntuple(i -> convert_component(c, getfield(x, i)), Val(n))
    if all(i -> fields[i] === getfield(x, i), 1:n)
        return x
    end
    W = X.name.wrapper
    y = try
        W(fields...)
    catch e
        e isa MethodError || rethrow()
        throw_conversion_error(W)
    end
    # Catch constructors that take different arguments to the fields
    if !(typeof(y).name.wrapper === W && all(i -> isequal(getfield(y, i), fields[i]), 1:n))
        throw_conversion_error(W)
    end
    return y
end

function throw_conversion_error(W)
    throw(ArgumentError("$W can not be converted to another float type, unit setting or " *
                        "device by rebuilding it from its converted fields, a method of " *
                        "Molly.convert_component is required"))
end

# Build a struct like `x` with some fields replaced
function replace_fields(x::X, new_fields::NamedTuple) where X
    return X.name.wrapper((get(new_fields, f, getfield(x, f)) for f in fieldnames(X))...)
end

function convert_component(c::SystemConversion, at::Atom)
    return Atom(at.index, at.atom_type, convert_component(c, at.mass),
                convert_component(c, at.charge), convert_component(c, at.σ),
                convert_component(c, at.ϵ), convert_component(c, at.λ), at.alch_role)
end

#=
    convert_system(sys, args, float_type, float_type_high, array_type, units, data,
                   grad_safe, strictness, launch_config)

Build the system given by the copy constructor of `sys` with the keyword arguments `args`,
converted to a new float type, array type and unit setting. Loggers are dropped, with a
warning, unless they were given as an argument, since their histories and observables are
tied to the original float type, units and device.
=#
function convert_system(sys, args, ::Type{T}, ::Type{TH}, ::Type{AT}, units::Bool, data,
                        grad_safe::Bool, strictness, launch_config) where {T, TH, AT}
    if units && sys.energy_units == NoUnits
        throw(ArgumentError("units can not be added to a System without units"))
    end
    if args.loggers === sys.loggers && length(sys.loggers) > 0
        report_issue("the loggers are dropped when converting a System to another float type, " *
                     "unit setting or device, give new loggers with the loggers keyword " *
                     "argument to keep them", strictness)
        loggers = ()
    else
        loggers = args.loggers
    end
    same_device = (AT === array_type(sys))
    c_atoms = SystemConversion{T, AT, TH}(!units, grad_safe, same_device)
    atoms = convert_component(c_atoms, args.atoms)
    coords = convert_component(c_atoms, args.coords)
    boundary = convert_component(c_atoms, args.boundary)
    c = SystemConversion{T, AT, TH}(!units, grad_safe, same_device, atoms, coords, boundary)
    return System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=convert_component(c, args.velocities),
        atoms_data=args.atoms_data,
        topology=args.topology,
        pairwise_inters=convert_component(c, args.pairwise_inters),
        specific_inter_lists=convert_component(c, args.specific_inter_lists),
        general_inters=convert_component(c, args.general_inters),
        constraints=convert_component(c, args.constraints),
        virtual_sites=convert_component(c, args.virtual_sites),
        neighbor_finder=convert_component(c, args.neighbor_finder),
        loggers=loggers,
        force_units=(units ? args.force_units : NoUnits),
        energy_units=(units ? args.energy_units : NoUnits),
        k=convert_component(c, args.k),
        float_type=T,
        float_type_high=TH,
        data=data,
        grad_safe=grad_safe,
        strictness=strictness,
        launch_config=launch_config,
    )
end
