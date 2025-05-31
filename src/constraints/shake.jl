export
    SHAKE_RATTLE,
    check_position_constraints,
    check_velocity_constraints

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
"""
struct SHAKE_RATTLE{A, B, C, D, V}
    clusters1::A
    clusters2::B
    clusters3::C
    dist_tolerance::D
    vel_tolerance::V
    coord_ordering::Vector{Int}
    inv_ordering::Vector{Int}
end

function SHAKE_RATTLE(constraints, n_atoms, dist_tolerance, vel_tolerance)

    clusters1, clusters2, clusters3 = build_clusters(n_atoms, constraints)

    # Generate coordinate ordering. This places single constraints first
    # and in the order matching the `clusters1` list (i.e. first two atoms 
    # correspond to the first constraint). Then the 3-constraint and 4-constraint
    # clusters and finally all the other unconstrained atoms. This ordering is 
    # is a one-time cost to improve GPU-performance.

    #* I THINK I NEED TO SORT BY NUMBER OF ATOMS IN 
    #* A CLUSTER AND NOT NUMBER OF CONSTRAINTS IN A CLUSTER
    #* NEED TO PARSE OUT 4-atom 4-cosntraint clusters too

    #* DOUBLE CHECK THAT ORDER IS CONSISTENT WITH IMPLEMENTATION
    #* CENTER IS ASSUMED TO BE FIRST OFTEN.
    coord_ordering = []
    for cluster in clusters1
        push!(coord_ordering, cluster[1].i, cluster[1].j)
    end
    for cluster in clusters2
        atoms_in_cluster = unique([cluster.i; cluster.j])
        push!(coord_ordering, atoms_in_cluster)
    end
    for cluster in clusters3
        atoms_in_cluster = unique([cluster.i; cluster.j])
        push!(coord_ordering, atoms_in_cluster)
    end

    inv_ordering = invperm(coord_ordering) #*TODO MOVE THESE TO BACKEND

    # coord_map = Dict(orig => new for (new, orig) in enumerate(coord_ordering))

    return SHAKE_RATTLE{typeof(clusters), typeof(dist_tolerance), typeof(vel_tolerance)}(
            clusters1, clusters2, clusters3, dist_tolerance, vel_tolerance, coord_ordering, inv_ordering)
end

function kronickerδ(x::T, y::T)
    if x == y
        return one(T)
    else
        return zero(T)
    end
end
  
function A_matrix_helper!(A, clusters, masses)
    M = length(clusters[1])
    for cluster in clusters
        for a in 1:M
            m1 = masses[cluster[a].i]
            m2 = masses[cluster[a].j]
            for b in 1:M 
                δa1b1 = kronickerδ(cluster[a].i, cluster[b].i)
                δa1b2 = kronickerδ(cluster[a].i, cluster[b].j)
                δa2b2 = kronickerδ(cluster[a].j, cluster[b].j)
                δa2b1 = kronickerδ(cluster[a].j, cluster[b].i)
                A[a, b] = ((δa1b1 + δa1b2)/m1) + ((δa2b2 + δa2b1)/m2)
            end
        end
    end
    return A
end

function precompute_A_components(sr::SHAKE_RATTLE, masses)

    # Construct on CPU we can move to GPU after
    Aₘ₂ = KernelAbstractions.zeros(CPU(), T, length(sr.clusters2), 2, 2) 
    Aₘ₃ = KernelAbstractions.zeros(CPU(), T, length(sr.clusters3), 3, 3) 

    # Construct mass component of the A matrix to avoid
    # to avoid if-else inside the GPU kernel.
    # Eqn 15 in J. Comp. Chem., Vol. 22, No. 5, 501–508
    A_matrix_helper!(Aₘ₂, sr.clusters2, masses)
    A_matrix_helper!(Aₘ₃, sr.clusters3, masses)

    return Aₘ₂, Aₘ₃
end

function setup_constraints!(neighbor_finder, sr::SHAKE_RATTLE)

    # Compute and Move A matricies to Backend
    Aₘ₂, Aₘ₃ = precompute_A_components(sr, masses(sys))
    #*TODO MOVE TO BACKEND

    # Disable Neighbor interactions that are constrained
    if typeof(neighbor_finder) != NoNeighborFinder
        disable_constrained_interactions!(neighbor_finder, sr.clusters1)
        disable_constrained_interactions!(neighbor_finder, sr.clusters2)
        disable_constrained_interactions!(neighbor_finder, sr.clusters3)
    end

    return neighbor_finder

end



# function apply_position_constraints!(sys, ca::SHAKE_RATTLE, coord_storage;
#                                      n_threads::Integer=Threads.nthreads())
#     # SHAKE updates
#     converged = false

#     while !converged
#         Threads.@threads for cluster in ca.clusters
#             for constraint in cluster.constraints
#                 k1, k2 = constraint.i, constraint.j

#                 # Vector between the atoms after unconstrained update (s)
#                 s12 = vector(sys.coords[k1], sys.coords[k2], sys.boundary)

#                 # Vector between the atoms before unconstrained update (r)
#                 r12 = vector(coord_storage[k1], coord_storage[k2], sys.boundary)

#                 if abs(norm(s12) - constraint.dist) > ca.dist_tolerance
#                     m1_inv = inv(mass(sys.atoms[k1]))
#                     m2_inv = inv(mass(sys.atoms[k2]))
#                     a = (m1_inv + m2_inv)^2 * sum(abs2, r12)
#                     b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
#                     c = sum(abs2, s12) - (constraint.dist)^2
#                     D = b^2 - 4*a*c

#                     if ustrip(D) < 0.0
#                         @warn "SHAKE determinant negative, setting to 0.0"
#                         D = zero(D)
#                     end

#                     # Quadratic solution for g
#                     α1 = (-b + sqrt(D)) / (2*a)
#                     α2 = (-b - sqrt(D)) / (2*a)

#                     g = abs(α1) <= abs(α2) ? α1 : α2

#                     # Update positions
#                     δri1 = r12 .* (g*m1_inv)
#                     δri2 = r12 .* (-g*m2_inv)

#                     sys.coords[k1] += δri1
#                     sys.coords[k2] += δri2
#                 end
#             end
#         end

#         converged = check_position_constraints(sys, ca)
#     end
#     return sys
# end

# function apply_velocity_constraints!(sys, ca::SHAKE_RATTLE; n_threads::Integer=Threads.nthreads())
#     # RATTLE updates
#     converged = false

#     while !converged
#         Threads.@threads for cluster in ca.clusters
#             for constraint in cluster.constraints
#                 k1, k2 = constraint.i, constraint.j

#                 inv_m1 = inv(mass(sys.atoms[k1]))
#                 inv_m2 = inv(mass(sys.atoms[k2]))

#                 # Vector between the atoms after SHAKE constraint
#                 r_k1k2 = vector(sys.coords[k1], sys.coords[k2], sys.boundary)

#                 # Difference between unconstrainted velocities
#                 v_k1k2 = sys.velocities[k2] .- sys.velocities[k1]

#                 err = abs(dot(r_k1k2, v_k1k2))
#                 if err > ca.vel_tolerance
#                     # Re-arrange constraint equation to solve for Lagrange multiplier
#                     # This has a factor of dt which cancels out in the velocity update
#                     λₖ = -dot(r_k1k2, v_k1k2) / (dot(r_k1k2, r_k1k2) * (inv_m1 + inv_m2))

#                     # Correct velocities
#                     sys.velocities[k1] -= inv_m1 .* λₖ .* r_k1k2
#                     sys.velocities[k2] += inv_m2 .* λₖ .* r_k1k2
#                 end
#             end
#         end

#         converged = check_velocity_constraints(sys, ca)
#     end
#     return sys
# end

"""
    check_position_constraints(sys, constraints)

Checks if the position constraints are satisfied by the current coordinates of `sys`.
"""
function check_position_constraints(sys, ca::SHAKE_RATTLE)
    max_err = typemin(float_type(sys)) * unit(eltype(eltype(sys.coords)))
    for cluster in ca.clusters
        for constraint in cluster.constraints
            dr = vector(sys.coords[constraint.i], sys.coords[constraint.j], sys.boundary)
            err = abs(norm(dr) - constraint.dist)
            if max_err < err
                max_err = err
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
    for cluster in ca.clusters
        for constraint in cluster.constraints
            dr = vector(sys.coords[constraint.i], sys.coords[constraint.j], sys.boundary)
            v_diff = sys.velocities[constraint.j] .- sys.velocities[constraint.i]
            err = abs(dot(dr, v_diff))
            if max_err < err
                max_err = err
            end
        end
    end
    return max_err < ca.vel_tolerance
end



# # Add indices of atoms which do not participate in constraints
# unconstrained_coords = setdiff(1:n_atoms, coord_ordering)
# push!(coord_ordering, unconstrained_coords)

# # Update (i,j) indices of each constraint
# index_map = Dict(orig => new for (new, orig) in enumerate(coord_ordering))

# for cluster_array in (clusters1, clusters2, clusters3)
#     for cluster in cluster_array
#         for constraint in cluster
#             constraint.i = index_map[constraint.i]
#             constraint.j = index_map[constraint.j]
#         end
#     end
# end