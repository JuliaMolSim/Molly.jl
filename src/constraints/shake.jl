export
    SHAKE_RATTLE,
    check_position_constraints,
    check_velocity_constraints,
    setup_constraints!

const CC = ConstraintCluster #alias so next line is shorter


"""
    SHAKE_RATTLE(constraints, n_atoms, dist_tolerance, vel_tolerance)

Constrain distances during a simulation using the SHAKE and RATTLE algorithms.

Velocity constraints will be imposed for simulators that integrate velocities such as
[`VelocityVerlet`](@ref).
See [Ryckaert et al. 1977](https://doi.org/10.1016/0021-9991(77)90098-5) for SHAKE,
[Andersen 1983](https://doi.org/10.1016/0021-9991(83)90014-1) for RATTLE and
[Elber et al. 2011](https://doi.org/10.1140%2Fepjst%2Fe2011-01525-9) for a derivation
of the linear system solved to satisfy the RATTLE algorithm.
[Krautler et al. 2000](https://onlinelibrary.wiley.com/doi/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V) for the M-SHAKE algorithm


# Arguments
- `constraints`: A vector of constraints to be imposed on the system.
- `n_atoms`: Total number of atoms in the system.
- `dist_tolerance`: the tolerance used to end the iterative procedure when calculating
    position constraints, should have the same units as the coordinates.
- `vel_tolerance`: the tolerance used to end the iterative procedure when calculating
    velocity constraints, should have the same units as the velocities * the coordinates.
- `gpu_block_size`: The number of threads per block to use for GPU calculations.
- `max_iters`: The maximum number of iterations to perform when doing SHAKE. Defaults to 25.
"""
struct SHAKE_RATTLE{A <: CC, B <: CC, C <: CC, D <: CC, E, F, S}
    clusters12::AbstractVector{A}
    clusters23::AbstractVector{B}
    clusters34::AbstractVector{C}
    angle_clusters::AbstractVector{D}
    dist_tolerance::E
    vel_tolerance::F
    gpu_block_size::Integer
    max_iters::Integer
    stats::S # keeps track of iters, average, variance, and max/min
end

function SHAKE_RATTLE(constraints, n_atoms, dist_tolerance, vel_tolerance;
                         gpu_block_size = 64, max_iters = 25)

    clusters12, clusters23, clusters34, angle_clusters = build_clusters(n_atoms, constraints)

    @assert ustrip(dist_tolerance) > 0.0 "dist_tolerance must be greater than zero"
    @assert ustrip(vel_tolerance) > 0.0 "vel_tolerance must be greater than zero"

    A = eltype(clusters12)
    B = eltype(clusters23)
    C = eltype(clusters34)
    D = eltype(angle_clusters)

    stats = Series(Mean(), Variance(), Extrema())

    return SHAKE_RATTLE{A, B, C, D, typeof(dist_tolerance), typeof(vel_tolerance), typeof(stats)}(
        clusters12, clusters23, clusters34, angle_clusters, dist_tolerance, vel_tolerance, gpu_block_size, max_iters, stats)
end

cluster_keys(::SHAKE_RATTLE) = [:clusters12, :clusters23, :clusters34, :angle_clusters]
iters_avg(sr::SHAKE_RATTLE) = value(sr.stats[1])
iters_var(sr::SHAKE_RATTLE) = value(sr.stats[2])
iters_min(sr::SHAKE_RATTLE) = value(sr.stats[3].min)
iters_max(sr::SHAKE_RATTLE) = value(sr.stats[3].max)
iters_nmin(sr::SHAKE_RATTLE) = value(sr.stats[3].nmin)
iters_nmax(sr::SHAKE_RATTLE) = value(sr.stats[3].nmax)
  
function setup_constraints!(neighbor_finder, sr::SHAKE_RATTLE)

    # Disable Neighbor interactions that are constrained
    if typeof(neighbor_finder) != NoNeighborFinder
        disable_constrained_interactions!(neighbor_finder, sr.clusters12)
        disable_constrained_interactions!(neighbor_finder, sr.clusters23)
        disable_constrained_interactions!(neighbor_finder, sr.clusters34)
        disable_constrained_interactions!(neighbor_finder, sr.angle_clusters)
    end

    return neighbor_finder

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
