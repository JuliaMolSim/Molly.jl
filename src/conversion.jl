# Conversion of a System to another float type, to unitless values or to another device,
#   done by `System(sys; float_type, units, array_type)`

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

function convert_component(c::SystemConversion, x::Unitful.Quantity)
    val = convert_component(c, ustrip(x))
    return (c.strip_units ? val : val * unit(x))
end

const UnconvertedComponent = Union{Number, AbstractString, Symbol, Nothing, Function, Type}

convert_component(::SystemConversion, x::UnconvertedComponent) = x

# Units stored in a struct, such as the force and energy units of `MullerBrown`
convert_component(c::SystemConversion, x::Unitful.Units) = (c.strip_units ? NoUnits : x)

convert_component(c::SystemConversion, x::Union{Tuple, NamedTuple}) = map(y -> convert_component(c, y), x)
convert_component(c::SystemConversion, x::StaticArray) = map(y -> convert_component(c, y), x)

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

function convert_component(c::SystemConversion, b::TriclinicBoundary{<:Any, <:Any, <:Any, A}) where A
    return TriclinicBoundary(convert_component(c, b.basis_vectors); approx_images=A)
end

# AngleConstraint stores the third distance rather than the angle, so the angle is found
#   from the converted distances
function convert_component(c::SystemConversion, ac::AngleConstraint)
    d_ij, d_jk, d_ik = convert_component.((c,), (ac.dist_ij, ac.dist_jk, ac.dist_ik))
    angle_ijk = acos(clamp((d_ij^2 + d_jk^2 - d_ik^2) / (2 * d_ij * d_jk), -1, 1))
    return AngleConstraint(ac.i, ac.j, ac.k, angle_ijk, d_ij, d_jk)
end

# Constraints are converted on the CPU, the System constructor moves them to the device
function convert_component(c::SystemConversion, sr::SHAKE_RATTLE)
    clusters = map((sr.clusters12, sr.clusters23, sr.clusters34, sr.angle_clusters)) do cl
        cl_cpu = host_constraint_clusters(cl)
        if length(cl_cpu) == 0
            return cl_cpu
        else
            return StructArray([convert_component(c, cd) for cd in cl_cpu])
        end
    end
    return SHAKE_RATTLE(clusters..., convert_component(c, sr.dist_tolerance),
                        convert_component(c, sr.vel_tolerance), sr.gpu_block_size,
                        sr.max_iters)
end

function convert_component(c::SystemConversion, lincs::LINCS)
    # The distance constraints from the angle constraints are added at the end
    n_angle = (isnothing(lincs.angle_constraints) ? 0 : length(lincs.angle_constraints))
    n_dist = length(lincs.dist_constraints) - 3 * n_angle
    dist_constraints = [convert_component(c, dc) for dc in lincs.dist_constraints[1:n_dist]]
    if n_angle > 0
        angle_constraints = [convert_component(c, ac) for ac in lincs.angle_constraints]
    else
        angle_constraints = nothing
    end
    return LINCS(
        masses=mass.(from_device(c.atoms)),
        dist_tolerance=convert_component(c, lincs.dist_tolerance),
        vel_tolerance=convert_component(c, lincs.vel_tolerance),
        dist_constraints=dist_constraints,
        angle_constraints=angle_constraints,
        n_rec=lincs.lincs_data.n_rec,
        n_iter=lincs.lincs_data.n_iter,
        iter_vel_correction=lincs.iter_vel_correction,
        gpu_block_size=lincs.gpu_block_size,
    )
end

# Neighbor finders are rebuilt from their eligible and special matrices, which are passed on
#   as they are stored, sparse or dense. On the same device the type of neighbor finder is
#   kept if it can be used by the new system. Otherwise the neighbor finder that system setup
#   would choose for the new device, gradient safety and boundary is used, so that moving a
#   system to another device and back gives the original type of neighbor finder.
function convert_component(c::SystemConversion{<:Any, AT},
                           nf::Union{DistanceNeighborFinder, TreeNeighborFinder,
                                     CellListMapNeighborFinder, GPUNeighborFinder,
                                     GPUCellListNeighborFinder}) where AT
    dist_cutoff = convert_component(c, nf.dist_cutoff)
    if isnothing(nf.eligible)
        # The output modes of GPUCellListNeighborFinder that ignore the exceptions store no
        #   matrices, which is the same as every pair being eligible and none special
        matrices = (n_atoms=nf.n_atoms,)
    else
        matrices = (eligible=from_device(nf.eligible), special=from_device(nf.special))
    end
    on_gpu = AT <: AbstractGPUArray
    gpu_nf_allowed = on_gpu && uses_gpu_neighbor_finder(AT) && !c.grad_safe
    if c.same_device
        if nf isa GPUNeighborFinder && gpu_nf_allowed
            return GPUNeighborFinder(; matrices..., dist_cutoff=dist_cutoff,
                                     n_steps=nf.n_steps, array_type=AT)
        elseif nf isa GPUCellListNeighborFinder && on_gpu
            return GPUCellListNeighborFinder(; matrices..., dist_cutoff=dist_cutoff,
                                             n_steps=nf.n_steps, array_type=AT,
                                             max_neighbors=nf.max_neighbors,
                                             output=nf.output, ragged=nf.ragged)
        elseif nf isa TreeNeighborFinder && !on_gpu
            return TreeNeighborFinder(; matrices..., dist_cutoff=dist_cutoff,
                                      n_steps=nf.n_steps)
        elseif nf isa CellListMapNeighborFinder && !on_gpu
            return CellListMapNeighborFinder(; matrices..., dist_cutoff=dist_cutoff,
                                             n_steps=nf.n_steps, x0=c.coords,
                                             boundary=c.boundary)
        elseif nf isa DistanceNeighborFinder
            return DistanceNeighborFinder(; matrices..., dist_cutoff=dist_cutoff,
                                          n_steps=nf.n_steps, array_type=AT)
        end
    end
    # The choice made by system setup
    if gpu_nf_allowed
        return GPUNeighborFinder(; matrices..., dist_cutoff=dist_cutoff, n_steps=nf.n_steps,
                                 array_type=AT)
    elseif on_gpu && gpu_cell_list_suitable(c.boundary, dist_cutoff)
        return GPUCellListNeighborFinder(; matrices..., dist_cutoff=dist_cutoff,
                                         n_steps=nf.n_steps, array_type=AT, ragged=false)
    elseif !on_gpu && !has_infinite_boundary(c.boundary) &&
                minimum(box_sides(c.boundary)) >= 2 * dist_cutoff
        return CellListMapNeighborFinder(; matrices..., dist_cutoff=dist_cutoff,
                                         n_steps=nf.n_steps, x0=c.coords,
                                         boundary=c.boundary)
    end
    return DistanceNeighborFinder(; matrices..., dist_cutoff=dist_cutoff, n_steps=nf.n_steps,
                                  array_type=AT)
end

# The number of threads a PME on CPU was set up for, which sets the size of its buffers
pme_n_threads(pme) = (pme.recip_conv_buffer isa Vector ? length(pme.recip_conv_buffer) :
                      Threads.nthreads())

# PME is rebuilt through its constructor, which sets up the grids, buffers and FFT plans.
#   α is kept from the original, as it is in CoulombEwald and EwaldExclusionData.
function convert_component(c::SystemConversion{<:Any, <:Any, TH}, pme::PME) where TH
    pme_new = PME(
        convert_component(c, pme.dist_cutoff),
        c.atoms,
        c.boundary;
        error_tol=convert_component(c, pme.error_tol),
        order=pme.order,
        ϵr=convert_component(c, pme.ϵr),
        # The charge sums are not stored for gradient safe PME, so fixed charges are not
        #   assumed in that case
        fixed_charges=!isnothing(pme.pc_sum),
        mesh_dims=pme.mesh_dims,
        scheduler=convert_component(c, pme.scheduler),
        float_type_high=TH,
        grad_safe=c.grad_safe,
        n_threads=pme_n_threads(pme),
    )
    return replace_fields(pme_new, (α=convert_component(c, pme.α),))
end

# The λ field holds the mesh weight rather than λ, so it is kept from the original
function convert_component(c::SystemConversion, pme::PME_λ)
    pme_new = PME_λ(
        convert_component(c, pme.dist_cutoff),
        c.atoms,
        c.boundary;
        error_tol=convert_component(c, pme.error_tol),
        order=pme.order,
        ϵr=convert_component(c, pme.ϵr),
        fixed_charges=false,
        mesh_dims=pme.mesh_dims,
        scheduler=convert_component(c, pme.scheduler),
        grad_safe=c.grad_safe,
        n_threads=pme_n_threads(pme),
    )
    return replace_fields(pme_new, (α=convert_component(c, pme.α),
                                    λ=convert_component(c, pme.λ)))
end

# The buffers of implicit solvent depend on the device and the number of threads, so they are
#   set up again
function convert_component(c::SystemConversion{T, AT}, inter::AbstractGBSA) where {T, AT}
    params = NamedTuple(f => convert_component(c, getfield(inter, f))
                        for f in fieldnames(typeof(inter)) if !startswith(String(f), "buffer_"))
    # The number of threads of an implicit solvent interaction on CPU sets its force buffers
    if inter.buffer_force_chunks isa AbstractVector
        n_threads = length(inter.buffer_force_chunks)
    else
        n_threads = Threads.nthreads()
    end
    bufs = gbsa_buffers(AT, T, length(params.offset_radii), params.offset_radii,
                        params.sa_factor, params.dist_cutoff, params.offset,
                        params.factor_solute, n_threads)
    buffers = (buffer_Bs=bufs.Bs, buffer_B_grads=bufs.B_grads,
               buffer_atom_charges=bufs.atom_charges, buffer_born_forces=bufs.born_forces,
               buffer_born_forces_scaled=bufs.born_forces_scaled,
               buffer_born_forces_mod=bufs.born_forces_mod, buffer_Is_nounits=bufs.Is_nounits,
               buffer_pes=bufs.pes, buffer_fs_mat=bufs.fs_mat,
               buffer_force_chunks=bufs.force_chunks)
    return replace_fields(inter, merge(params, buffers))
end

function convert_component(::SystemConversion, ::ANIPotential)
    throw(ArgumentError("ANIPotential can not be converted to another float type, unit " *
                        "setting or device, it should be given to the new system with the " *
                        "general_inters keyword argument"))
end

#=
    convert_system(sys, args, float_type, float_type_high, array_type, units, data,
                   grad_safe, strictness, launch_config)

Build the system given by the copy constructor of `sys` with the keyword arguments `args`,
converted to a new float type, array type and unit setting. Loggers are dropped unless they
were given as an argument, since their histories and observables are tied to the original
float type, units and device.
=#
function convert_system(sys, args, ::Type{T}, ::Type{TH}, ::Type{AT}, units::Bool, data,
                        grad_safe::Bool, strictness, launch_config) where {T, TH, AT}
    if units && sys.energy_units == NoUnits
        throw(ArgumentError("units can not be added to a System without units"))
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
        loggers=(args.loggers === sys.loggers ? () : args.loggers),
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
