export
    SHAKE_RATTLE,
    check_position_constraints,
    check_velocity_constraints,
    setup_constraints!


"""
    SHAKE_RATTLE(n_atoms,
                dist_tolerance,
                vel_tolerance;
                dist_constraints = nothing,
                angle_constraints = nothing,
                gpu_block_size = 64,
                max_iters = 25)

Constrain distances during a simulation using the SHAKE and RATTLE algorithms. 
Either or both of `dist_constraints` and `angle_constraitns` must be passed.

Velocity constraints will be imposed for simulators that integrate velocities such as
[`VelocityVerlet`](@ref).
See [Ryckaert et al. 1977](https://doi.org/10.1016/0021-9991(77)90098-5) for SHAKE,
[Andersen 1983](https://doi.org/10.1016/0021-9991(83)90014-1) for RATTLE and
[Elber et al. 2011](https://doi.org/10.1140%2Fepjst%2Fe2011-01525-9) for a derivation
of the linear system solved to satisfy the RATTLE algorithm.
[Krautler et al. 2000](https://onlinelibrary.wiley.com/doi/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V) for the M-SHAKE algorithm


# Arguments
- `n_atoms`: Total number of atoms in the system.
- `dist_tolerance`: the tolerance used to end the iterative procedure when calculating
    position constraints, should have the same units as the coordinates.
- `vel_tolerance`: the tolerance used to end the iterative procedure when calculating
    velocity constraints, should have the same units as the velocities * the coordinates.
- `dist_constraints`: A vector of [`DistanceConstraint`](@ref) objects that define the
    distance constraints to be applied. If `nothing`, no distance constraints are applied.
- `angle_constraints`: A vector of [`AngleConstraint`](@ref) objects that define the
    angle constraints to be applied. If `nothing`, no angle constraints are applied.
- `gpu_block_size`: The number of threads per block to use for GPU calculations.
- `max_iters`: The maximum number of iterations to perform when doing SHAKE. Defaults to 25.
"""
struct SHAKE_RATTLE{A, B, C, D, E, F, S}
    clusters12::A
    clusters23::B
    clusters34::C
    angle_clusters::D
    dist_tolerance::E
    vel_tolerance::F
    gpu_block_size::Integer
    max_iters::Integer
    stats::S # keeps track of iters, average, variance, and max/min
end

function SHAKE_RATTLE(n_atoms,
                     dist_tolerance,
                     vel_tolerance;
                     dist_constraints = nothing,
                     angle_constraints = nothing,
                     gpu_block_size = 64, max_iters = 25)

    @assert ustrip(dist_tolerance) > 0.0 "dist_tolerance must be greater than zero"
    @assert ustrip(vel_tolerance) > 0.0 "vel_tolerance must be greater than zero"
    @assert !(isnothing(dist_constraints) && isnothing(angle_constraints)) "At least one of dist_constraints or angle_constraints must be provided"
    @assert length(dist_constraints) > 0 || length(angle_constraints) > 0 "At least one of dist_constraints or angle_constraints must be non-empty"

    if !isnothing(dist_constraints) && length(dist_constraints) == 0
        @warn "You passed an empty vector for `dist_constraints`, no distance constraints will be applied."
        dist_constraints = nothing
    end

    if !isnothing(angle_constraints) && length(angle_constraints) == 0
        @warn "You passed an empty vector for `angle_constraints`, no angle constraints will be applied."
        angle_constraints = nothing
    end

    if float_type(dist_tolerance) isa Float32 
        if ustrip(dist_tolerance) <= Float32(1e-6) 
            @warn "Using Float32 with a SHAKE dist_tolerance less than 1e-6. Might have convergence issues."
        end
    end

    if float_type(vel_tolerance) isa Float32 
        if ustrip(vel_tolerance) <= Float32(1e-6) 
            @warn "Using Float32 with a RATTLE vel_tolerance less than 1e-6. Might have convergence issues."
        end
    end

    if isa(dist_constraints, AbstractGPUArray) || isa(angle_constraints, AbstractGPUArray)
        error("Constraints should be passd to SHAKE_RATTLE on CPU. Data will be moved to GPU later.")
    end


    clusters12, clusters23, clusters34, angle_clusters = build_clusters(n_atoms, dist_constraints, angle_constraints)

    A = typeof(clusters12)
    B = typeof(clusters23)
    C = typeof(clusters34)
    D = typeof(angle_clusters)

    stats = Series(Mean(), Variance(), Extrema())

    return SHAKE_RATTLE{A, B, C, D, typeof(dist_tolerance), typeof(vel_tolerance), typeof(stats)}(
        clusters12, clusters23, clusters34, angle_clusters, dist_tolerance, vel_tolerance, gpu_block_size, max_iters, stats)
end

function SHAKE_RATTLE(sr::SHAKE_RATTLE, clusters12, clusters23, clusters34, angle_clusters)
    A = typeof(clusters12)
    B = typeof(clusters23)
    C = typeof(clusters34)
    D = typeof(angle_clusters)

    return SHAKE_RATTLE{A, B, C, D, typeof(sr.dist_tolerance), typeof(sr.vel_tolerance), typeof(sr.stats)}(
        clusters12, clusters23, clusters34, angle_clusters,
        sr.dist_tolerance, sr.vel_tolerance, sr.gpu_block_size, sr.max_iters, sr.stats)
end

cluster_keys(::SHAKE_RATTLE) = [:clusters12, :clusters23, :clusters34, :angle_clusters]
iters_avg(sr::SHAKE_RATTLE) = value(sr.stats[1])
iters_var(sr::SHAKE_RATTLE) = value(sr.stats[2])
iters_min(sr::SHAKE_RATTLE) = value(sr.stats[3].min)
iters_max(sr::SHAKE_RATTLE) = value(sr.stats[3].max)
iters_nmin(sr::SHAKE_RATTLE) = value(sr.stats[3].nmin)
iters_nmax(sr::SHAKE_RATTLE) = value(sr.stats[3].nmax)
  
function setup_constraints!(sr::SHAKE_RATTLE, neighbor_finder, arr_type)

    # Disable Neighbor interactions that are constrained
    if typeof(neighbor_finder) != NoNeighborFinder
        disable_constrained_interactions!(neighbor_finder, sr.clusters12)
        disable_constrained_interactions!(neighbor_finder, sr.clusters23)
        disable_constrained_interactions!(neighbor_finder, sr.clusters34)
        disable_constrained_interactions!(neighbor_finder, sr.angle_clusters)
    end

    # Move to proper backend, if CPU do nothing
    if arr_type <: AbstractGPUArray

        clusters12_gpu = []; clusters23_gpu = []
        clusters34_gpu = []; angle_clusters_gpu = []

        if length(sr.clusters12) > 0
            clusters12_gpu = replace_storage(arr_type, sr.clusters12)
        end
        if length(sr.clusters23) > 0
            clusters23_gpu = replace_storage(arr_type, sr.clusters23)
        end
        if length(sr.clusters34) > 0
            clusters34_gpu = replace_storage(arr_type, sr.clusters34)
        end
        if length(sr.angle_clusters) > 0
            angle_clusters_gpu = replace_storage(arr_type, sr.angle_clusters)
        end

        sr = SHAKE_RATTLE(sr, clusters12_gpu, clusters23_gpu, clusters34_gpu, angle_clusters_gpu)
    end

    # neighboor_finder is also modified
    # but returning only sr makes life easier in types.jl
    return sr

end


"""
    check_position_constraints(sys, constraints)

Checks if the position constraints are satisfied by the current coordinates of `sys`.
"""
function check_position_constraints(sys, ca::SHAKE_RATTLE)
    max_err = typemin(float_type(sys)) * unit(eltype(eltype(sys.coords)))
    for cluster_type in cluster_keys(ca)
        clusters = getproperty(ca, cluster_type)
        for cluster in clusters
            for constraint in cluster.constraints
                dr = vector(sys.coords[constraint.i], sys.coords[constraint.j], sys.boundary)
                err = abs(norm(dr) - constraint.dist)
                if max_err < err
                    max_err = err
                end
            end
        end
    end
    return max_err < ca.dist_tolerance
end

"""
    check_velocity_constraints(sys, constraints)

Checks if the velocity constraints are satisfied by the current velocities of `sys`.
"""
function check_velocity_constraints(sys::System, ca::SHAKE_RATTLE)
    max_err = typemin(float_type(sys)) * unit(eltype(eltype(sys.velocities))) * unit(eltype(eltype(sys.coords)))
    for cluster_type in cluster_keys(ca)
        clusters = getproperty(ca, cluster_type)
        for cluster in clusters
            for constraint in cluster.constraints
                dr = vector(sys.coords[constraint.i], sys.coords[constraint.j], sys.boundary)
                v_diff = sys.velocities[constraint.j] .- sys.velocities[constraint.i]
                err = abs(dot(dr, v_diff))
                if max_err < err
                    max_err = err
                end
            end
        end
    end
    return max_err < ca.vel_tolerance
end
