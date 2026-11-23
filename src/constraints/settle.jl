export
    SETTLE,
    SetupSETTLE

"""
    SETTLE(; masses, angle_constraints, dist_tolerance=1e-8u"nm",
           vel_tolerance=1e-8u"nm^2 * ps^-1", gpu_block_size=128)

Constrain rigid three-site molecules, such as rigid water, using the SETTLE algorithm.

SETTLE solves the position and velocity constraints of each molecule analytically rather
than iteratively, so the constraints are satisfied to machine precision every step.
See [Miyamoto and Kollman 1992](https://doi.org/10.1002/jcc.540130805).
The position step follows the GROMACS implementation, which avoids computing the
center of mass explicitly to reduce energy drift.

Each [`AngleConstraint`](@ref) is one molecule, with the central atom `j` (oxygen for water)
and two equivalent atoms `i` and `k` (hydrogens for water).
The angle of each constraint must be at most 150°: in flatter triangles a normal time step
can move the atoms out of plane by more than the height of the triangle, which SETTLE can
not solve.
A molecule that is too distorted to be settled is left at its unconstrained positions and
gives a warning, or an error with `strictness=:error`.
Other constraints in the same system should be given to a separate constraint algorithm,
such as [`LINCS`](@ref) or [`SHAKE_RATTLE`](@ref), in the `constraints` tuple of the
[`System`](@ref).
The constraints should be on the CPU, even for a GPU system; the [`System`](@ref)
constructor moves the constraint data to the device of the system.

# Arguments
- `masses`: the masses of the atoms in the system.
- `angle_constraints`: a vector of [`AngleConstraint`](@ref) objects.
- `dist_tolerance=1e-8u"nm"`: the tolerance used by [`check_position_constraints`](@ref),
    should have the same units as the coordinates.
- `vel_tolerance=1e-8u"nm^2 * ps^-1"`: the tolerance used by
    [`check_velocity_constraints`](@ref).
- `gpu_block_size=128`: the number of threads per block to use for GPU calculations.
"""
struct SETTLE{C, D, V, F}
    clusters::C
    dist_tolerance::D
    vel_tolerance::V
    gpu_block_size::Int
    error_flag::F # Set when a molecule is too distorted to be settled
end

# Largest angle in degrees of a triangle given to SETTLE, see the SETTLE docstring
const settle_max_angle = 150

settle_angle(ac::AngleConstraint) = acosd(clamp(
    (ac.dist_ij^2 + ac.dist_jk^2 - ac.dist_ik^2) / (2 * ac.dist_ij * ac.dist_jk), -1, 1))

settle_near_linear(ac::AngleConstraint) = settle_angle(ac) > settle_max_angle

# Whether an angle constraint is a rigid triangle that SETTLE can solve
function settle_compatible(ac::AngleConstraint, masses)
    return isapprox(ac.dist_ij, ac.dist_jk; rtol=1e-6) &&
           isapprox(masses[ac.i], masses[ac.k]; rtol=1e-6) && !settle_near_linear(ac)
end

function SETTLE(; masses,
                angle_constraints,
                dist_tolerance=1e-8u"nm",
                vel_tolerance=1e-8u"nm^2 * ps^-1",
                gpu_block_size::Integer=128)
    ustrip(dist_tolerance) > 0 || throw(ArgumentError("dist_tolerance must be greater than zero"))
    ustrip(vel_tolerance ) > 0 || throw(ArgumentError("vel_tolerance must be greater than zero" ))
    if isnothing(angle_constraints) || length(angle_constraints) == 0
        throw(ArgumentError("angle_constraints must be provided and non-empty for SETTLE"))
    end
    if isa(angle_constraints, AbstractGPUArray) || isa(masses, AbstractGPUArray)
        throw(ArgumentError("constraints and masses should be passed to SETTLE on CPU, data " *
                            "will be moved to GPU later"))
    end

    seen_atoms = Set{Int}()
    for ac in angle_constraints
        for a in (ac.i, ac.j, ac.k)
            if a in seen_atoms
                throw(ArgumentError("atom $a appears in more than one SETTLE constraint"))
            end
            push!(seen_atoms, a)
        end
        if !isapprox(ac.dist_ij, ac.dist_jk; rtol=1e-6)
            throw(ArgumentError("SETTLE requires equal distances to the central atom, found " *
                                "$(ac.dist_ij) and $(ac.dist_jk) for atoms $(ac.i)/$(ac.j)/$(ac.k)"))
        end
        if !isapprox(masses[ac.i], masses[ac.k]; rtol=1e-6)
            throw(ArgumentError("SETTLE requires equal masses for the non-central atoms, found " *
                                "$(masses[ac.i]) and $(masses[ac.k]) for atoms $(ac.i) and $(ac.k)"))
        end
        if settle_near_linear(ac)
            throw(ArgumentError("SETTLE requires angles of at most $(settle_max_angle)°, found " *
                                "$(round(settle_angle(ac); digits=2))° for atoms " *
                                "$(ac.i)/$(ac.j)/$(ac.k)"))
        end
    end

    clusters = StructArray(AngleClusterData(ac) for ac in angle_constraints)
    return SETTLE(clusters, dist_tolerance, vel_tolerance, Int(gpu_block_size), zeros(Int32, 1))
end

function Base.show(io::IO, s::SETTLE)
    print(io, "SETTLE with ", length(s.clusters), " rigid molecules")
end

"""
    SetupSETTLE(; dist_tolerance=1e-6u"nm", vel_tolerance=1e-6u"nm^2 * ps^-1",
                gpu_block_size=128, constraint_algorithm=SetupLINCS())

Set up constraints using the SETTLE algorithm for rigid three-site molecules such as
rigid water.

Passed to the [`System`](@ref) constructor from files, where angle constraints that
[`SETTLE`](@ref) can solve (equal distances to the central atom, equal masses of the
other two atoms and an angle of at most 150°, such as with `rigid_water=true`) are given to
[`SETTLE`](@ref).
The remaining constraints are set up with `constraint_algorithm`, which can be
[`SetupLINCS`](@ref) or [`SetupSHAKE_RATTLE`](@ref).
See [`SETTLE`](@ref) for the other argument descriptions.
"""
struct SetupSETTLE{D, V, C}
    dist_tolerance::D
    vel_tolerance::V
    gpu_block_size::Int
    constraint_algorithm::C
end

function SetupSETTLE(; dist_tolerance=1e-6u"nm",
                     vel_tolerance=1e-6u"nm^2 * ps^-1",
                     gpu_block_size::Integer=128,
                     constraint_algorithm=SetupLINCS())
    ustrip(dist_tolerance) > 0 || throw(ArgumentError("dist_tolerance must be greater than zero"))
    ustrip(vel_tolerance ) > 0 || throw(ArgumentError("vel_tolerance must be greater than zero" ))
    if constraint_algorithm isa SetupSETTLE
        throw(ArgumentError("the constraint_algorithm of SetupSETTLE can not be SetupSETTLE"))
    end
    return SetupSETTLE(dist_tolerance, vel_tolerance, Int(gpu_block_size), constraint_algorithm)
end

function build_constraint_algorithms(T, dist_constraints, angle_constraints, atoms_data,
                                     units, strictness, masses, ca::SetupSETTLE)
    settle_acs = [ac for ac in angle_constraints if settle_compatible(ac, masses)]
    other_acs  = [ac for ac in angle_constraints if !settle_compatible(ac, masses)]
    n_near_linear = count(settle_near_linear, angle_constraints)
    if n_near_linear > 0
        report_issue("$n_near_linear angle constraints have angles above " *
                     "$(settle_max_angle)°, which SETTLE can not solve reliably, so they are " *
                     "given to $(nameof(typeof(ca.constraint_algorithm))); near-linear angle " *
                     "constraints are poorly conditioned and that algorithm may not satisfy " *
                     "them either", strictness)
    end
    algos = ()
    if length(settle_acs) > 0
        settle = SETTLE(
            masses=masses,
            angle_constraints=settle_acs,
            dist_tolerance=convert_setup_quantity(ca.dist_tolerance, units, T),
            vel_tolerance=convert_setup_quantity(ca.vel_tolerance, units, T),
            gpu_block_size=ca.gpu_block_size,
        )
        algos = (settle,)
    end
    if length(dist_constraints) > 0 || length(other_acs) > 0
        other = build_constraint_algorithm(T, dist_constraints, other_acs, atoms_data, units,
                                           strictness, masses, ca.constraint_algorithm)
        algos = (algos..., other)
    end
    return algos
end

cluster_keys(::SETTLE) = (:clusters,)

function constrained_atom_inds(s::SETTLE)
    atom_inds = Int[]
    for cl in from_device(s.clusters)
        push!(atom_inds, cl.k1, cl.k2, cl.k3)
    end
    return unique_ind_list(atom_inds)
end

function constrained_atom_pairs(s::SETTLE)
    atom_pairs = Tuple{Int, Int, eltype(from_device(s.clusters).dist12)}[]
    for cl in from_device(s.clusters)
        push!(atom_pairs, sort_pair(cl.k1, cl.k2, cl.dist12),
                          sort_pair(cl.k1, cl.k3, cl.dist13),
                          sort_pair(cl.k2, cl.k3, cl.dist23))
    end
    return unique_pair_list(atom_pairs)
end

# The constraints as given to the constructor, with the central atom as atom j
function settle_angle_constraints(s::SETTLE)
    return [AngleConstraint(Int(c.k2), Int(c.k1), Int(c.k3),
                            acos(clamp((c.dist12^2 + c.dist13^2 - c.dist23^2) /
                                       (2 * c.dist12 * c.dist13), -1, 1)),
                            c.dist12, c.dist13)
            for c in from_device(s.clusters)]
end

function setup_constraints!(s::SETTLE, neighbor_finder, arr_type)
    if !isnothing(neighbor_finder)
        disable_constrained_interactions!(neighbor_finder, s.clusters)
    end
    return move_constraints_to_device(s, arr_type)
end

function move_constraints_to_device(s::SETTLE, ::Type{AT}) where {AT <: AbstractGPUArray}
    return SETTLE(replace_storage(AT, s.clusters), s.dist_tolerance, s.vel_tolerance,
                  s.gpu_block_size, to_device(zeros(Int32, 1), AT))
end

#=
Position step for one molecule, following settleTemplate in GROMACS (src/gromacs/mdlib/settle.cpp)
which is derived from the original code of Miyamoto and Kollman 1992.
The symbols correspond to appendix A of the paper: the molecule is A (central atom k1) and
B, C (atoms k2 and k3), subscript 0 is the old geometry, 1 the unconstrained new geometry
and 3 the constrained new geometry. The new geometry is expressed relative to the center of
mass without computing the center of mass itself, which reduces energy drift.
=#
@inline function settle_cluster!(cluster_idx, k1s, k2s, k3s, dist12s, dist23s, r_t1, r_t2,
                                 ms, boundary, error_flag)
    k1 = k1s[cluster_idx]
    k2 = k2s[cluster_idx]
    k3 = k3s[cluster_idx]
    dOH = dist12s[cluster_idx]
    dHH = dist23s[cluster_idx]
    FT = typeof(ustrip(dOH))
    almost_zero = FT(1e-12)

    # Canonical triangle, eq. A1: ra is the distance from the center of mass to A,
    #   rb from the center of mass to the B-C line and rc is half the B-C distance
    mA, mB = ms[k1], ms[k2]
    wh = mB / (mA + 2 * mB)
    rc = dHH / 2
    h = sqrt(dOH^2 - rc^2)
    ra = 2 * wh * h
    rb = h - ra

    r_t1_k1 = r_t1[k1]
    b0 = vector(r_t1_k1, r_t1[k2], boundary)
    c0 = vector(r_t1_k1, r_t1[k3], boundary)
    r_t2_k1 = r_t2[k1]
    r_t2_k2 = r_t2[k2]
    r_t2_k3 = r_t2[k3]
    doh2 = vector(r_t2_k1, r_t2_k2, boundary)
    doh3 = vector(r_t2_k1, r_t2_k3, boundary)

    # Unconstrained positions relative to their center of mass
    a1 = -(doh2 + doh3) * wh
    b1 = doh2 + a1
    c1 = doh3 + a1

    # Frame with Z' normal to the old plane and Y'Z' containing A1
    n0 = cross(b0, c0)
    ax = cross(a1, n0)
    ay = cross(n0, ax)
    ex = ax / norm(ax)
    ey = ay / norm(ay)
    ez = n0 / norm(n0)

    b0x, b0y = dot(ex, b0), dot(ey, b0)
    c0x, c0y = dot(ex, c0), dot(ey, c0)
    a1z = dot(ez, a1)
    b1x, b1y, b1z = dot(ex, b1), dot(ey, b1), dot(ez, b1)
    c1x, c1y, c1z = dot(ex, c1), dot(ey, c1), dot(ez, c1)

    # Rotations ψ and ϕ from the Z' coordinates, eqs. A8-A10
    sinphi = a1z / ra
    cosphi2 = 1 - sinphi^2
    # Negated comparisons also catch NaN
    distorted = !(cosphi2 > almost_zero)
    cosphi = sqrt(max(cosphi2, almost_zero))
    sinpsi = (b1z - c1z) / (2 * rc * cosphi)
    cospsi2 = 1 - sinpsi^2
    distorted |= !(cospsi2 > almost_zero)
    cospsi = sqrt(max(cospsi2, almost_zero))

    # Eq. A3
    a2y = ra * cosphi
    b2x = -rc * cospsi
    t1 = -rb * cosphi
    t2 = rc * sinpsi * sinphi
    b2y = t1 - t2
    c2y = t1 + t2

    # Rotation θ, eqs. A14-A17
    α = b2x * (b0x - c0x) + b0y * b2y + c0y * c2y
    β = b2x * (c0y - b0y) + b0x * b2y + c0x * c2y
    γ = b0x * b1y - b1x * b0y + c0x * c1y - c1x * c0y
    α2β2 = α^2 + β^2
    disc = α2β2 - γ^2
    distorted |= !(disc >= zero(disc)) | !(α2β2 > zero(α2β2))
    sinthe = (α * γ - β * sqrt(max(disc, zero(disc)))) / α2β2
    costhe = sqrt(max(1 - sinthe^2, zero(FT)))

    # Constrained positions in the frame, eq. A4
    a3 = ex * (-a2y * sinthe)                + ey * (a2y * costhe)                + ez * a1z
    b3 = ex * (b2x * costhe - b2y * sinthe)  + ey * (b2x * sinthe + b2y * costhe) + ez * b1z
    c3 = ex * (-b2x * costhe - c2y * sinthe) + ey * (-b2x * sinthe + c2y * costhe) + ez * c1z

    # A distorted molecule is left at its unconstrained positions
    if distorted
        error_flag[1] = Int32(1)
    else
        r_t2[k1] = r_t2_k1 + (a3 - a1)
        r_t2[k2] = r_t2_k2 + (b3 - b1)
        r_t2[k3] = r_t2_k3 + (c3 - c1)
    end
    return nothing
end

@kernel inbounds=true function settle_kernel!(@Const(k1s), @Const(k2s), @Const(k3s),
                                              @Const(dist12s), @Const(dist23s),
                                              @Const(r_t1), r_t2, @Const(ms), boundary,
                                              error_flag)
    idx = @index(Global, Linear)
    if idx <= length(k1s)
        settle_cluster!(idx, k1s, k2s, k3s, dist12s, dist23s, r_t1, r_t2, ms, boundary,
                        error_flag)
    end
end

function check_settle_error!(s::SETTLE, strictness)
    if !iszero(only(from_device(s.error_flag)))
        fill!(s.error_flag, zero(Int32))
        report_issue("SETTLE found one or more molecules too distorted to be settled, " *
                     "which usually means the simulation is unstable; these molecules " *
                     "were left at their unconstrained positions, so their constraints " *
                     "are not satisfied", strictness; maxlog=1)
    end
    return nothing
end

function apply_position_constraints!(sys::System{D},
                                     ca::SETTLE,
                                     r_pre_unconstrained_update;
                                     context=nothing,
                                     n_threads::Integer=Threads.nthreads(),
                                     strictness=default_strictness(),
                                     kwargs...) where D
    D == 3 || throw(ArgumentError("SETTLE requires a 3D system, found $D dimensions"))
    context = isnothing(context) ? default_shake_position_constraint_context() : context
    validate_shake_position_virial_context(context)
    coords_before = context.needs_virial ? copyto_constraint_scratch!(context.coords_buffer, sys.coords) : nothing

    n_clusters = length(ca.clusters)
    backend = get_backend(r_pre_unconstrained_update)
    KernelAbstractions.synchronize(backend)
    ms = masses(sys)
    cl = ca.clusters

    if sys.coords isa AbstractGPUArray
        kernel! = settle_kernel!(backend, ca.gpu_block_size)
        kernel!(cl.k1, cl.k2, cl.k3, cl.dist12, cl.dist23, r_pre_unconstrained_update,
                sys.coords, ms, sys.boundary, ca.error_flag; ndrange=n_clusters)
        KernelAbstractions.synchronize(backend)
        if context.needs_virial && !isnothing(coords_before)
            accumulate_shake_cluster23_virial_gpu!(r_pre_unconstrained_update, coords_before,
                            sys.coords, ms, sys.boundary, cl, context, backend, ca.gpu_block_size)
            KernelAbstractions.synchronize(backend)
        end
    else
        apply_clusters_cpu!(settle_cluster!, n_clusters, n_threads, cl.k1, cl.k2, cl.k3,
                            cl.dist12, cl.dist23, r_pre_unconstrained_update, sys.coords, ms,
                            sys.boundary, ca.error_flag)
        if context.needs_virial && !isnothing(coords_before)
            accumulate_shake_cluster_virial!(r_pre_unconstrained_update, coords_before,
                                             sys.coords, ms, sys.boundary, cl, context)
        end
    end
    check_settle_error!(ca, strictness)
    return sys
end

# The velocity step of SETTLE solves the 3x3 linear system for the constraint forces of the
#   triangle exactly (eq. B1 of Miyamoto and Kollman 1992), which is what the RATTLE solve
#   for angle clusters does, so that is used here
function apply_velocity_constraints!(sys::System, ca::SETTLE; context=nothing,
                                     n_threads::Integer=Threads.nthreads(), kwargs...)
    context = isnothing(context) ? default_shake_velocity_constraint_context() : context
    validate_shake_velocity_virial_context(context)
    velocities_before = context.needs_virial ? copyto_constraint_scratch!(context.velocities_buffer, sys.velocities) : nothing

    backend = get_backend(sys.velocities)
    KernelAbstractions.synchronize(backend)
    on_gpu = sys.velocities isa AbstractGPUArray
    ms = masses(sys)
    cl = ca.clusters

    rattle_clusters!(on_gpu, rattle3_angle_kernel!, rattle3_angle_cluster!, length(cl), ca,
                     backend, n_threads, cl.k1, cl.k2, cl.k3, sys.coords, sys.velocities, ms,
                     sys.boundary)
    KernelAbstractions.synchronize(backend)

    if context.needs_virial && !isnothing(velocities_before)
        if on_gpu
            accumulate_shake_cluster23_virial_gpu!(sys.coords, velocities_before,
                            sys.velocities, ms, sys.boundary, cl, context, backend,
                            ca.gpu_block_size)
            KernelAbstractions.synchronize(backend)
        else
            accumulate_shake_cluster_virial!(sys.coords, velocities_before, sys.velocities,
                                             ms, sys.boundary, cl, context)
        end
    end
    return sys
end

function check_position_constraints(sys::System{<:Any, <:Any, FT}, ca::SETTLE) where FT
    err_unit = unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.dist_tolerance)
        throw(ArgumentError("distance tolerance units in SETTLE ($(unit(ca.dist_tolerance))) " *
                            "are inconsistent with system coordinate units ($err_unit)"))
    end
    backend = get_backend(sys.coords)
    max_storage = allocate(backend, FT, length(ca.clusters))
    max_dist_error(backend, 128)(ca.clusters, sys.coords, sys.boundary, max_storage;
                                 ndrange=length(ca.clusters))
    KernelAbstractions.synchronize(backend)
    return maximum(Array(max_storage)) < ustrip(ca.dist_tolerance)
end

function check_velocity_constraints(sys::System{<:Any, <:Any, FT}, ca::SETTLE) where FT
    err_unit = unit(eltype(eltype(sys.velocities))) * unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.vel_tolerance)
        throw(ArgumentError("velocity tolerance units in SETTLE ($(unit(ca.vel_tolerance))) " *
                    "are inconsistent with system velocity and coordinate units ($err_unit)"))
    end
    backend = get_backend(sys.coords)
    max_storage = allocate(backend, FT, length(ca.clusters))
    max_vel_error(backend, 128)(ca.clusters, sys.coords, sys.velocities, sys.boundary,
                                max_storage; ndrange=length(ca.clusters))
    KernelAbstractions.synchronize(backend)
    return maximum(Array(max_storage)) < ustrip(ca.vel_tolerance)
end

# Conversion to another float type, units or device, see `System(sys; ...)`
# Constraints are converted on the CPU, the System constructor moves them to the device
function convert_component(c::SystemConversion, s::SETTLE)
    return SETTLE(
        masses=mass.(from_device(c.atoms)),
        angle_constraints=[convert_component(c, ac) for ac in settle_angle_constraints(s)],
        dist_tolerance=convert_component(c, s.dist_tolerance),
        vel_tolerance=convert_component(c, s.vel_tolerance),
        gpu_block_size=s.gpu_block_size,
    )
end
