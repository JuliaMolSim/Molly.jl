export SHAKE_RATTLE

"""
SHAKE_RATTLE(coords, tolerance, init_posn_tol)

A constraint algorithm to set bonds distances in a simulation. Velocity constraints will be imposed
for simulators that integrate velocities (e.g. [`VelocityVerlet`](@ref)).

See [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/) for a derivation of the linear
system solved to satisfy the RATTLE algorithm.

# Arguments
- constraints::AbstractVector{<:Constraint}: A vector of constraints to be imposed on the system.
- n_atoms::Integer: The number of atoms in the system.
- dist_tolerance: Tolerance used to end iterative procedure when calculating constraints.
    Should have same units as coords.
- vel_tolerance: Tolerance used to end iterative procedure when calculating velocity constraints.
    Should have same units as velocities*coords.
"""
struct SHAKE_RATTLE{D, V} <: ConstraintAlgorithm
    clusters::AbstractVector{<:ConstraintCluster}
    dist_tolerance::D
    vel_tolerance::V
end

function SHAKE_RATTLE(constraints::AbstractVector{<:Constraint}, n_atoms, dist_tolerance, vel_tolerance)

    #If this becomes a memory issue n_atoms could probably be number of atoms constrained
    clusters = build_clusters(n_atoms, constraints)

    return SHAKE_RATTLE{typeof(dist_tolerance), typeof(vel_tolerance)}(clusters, dist_tolerance, vel_tolerance)
end

function position_constraints!(sys::System, constraint_algo::SHAKE_RATTLE;
     n_threads::Integer=Threads.nthreads())

    # Threads.@threads for group_id in 1:n_threads #& can only paralellize over independent clusters
    #     for i in group_id:n_threads:length(sys.constraints)
    #         SHAKE_update!(sys, constraint_algo, sys.constraints[i])
    #     end
    # end

    SHAKE_updates!(sys, constraint_algo)

    return sys
end



function SHAKE_updates!(sys, ca::SHAKE_RATTLE)
    converged = false

    while !converged
        for cluster in ca.clusters #& illegal to parallelize this
            for constraint in cluster.constraints

                k1, k2 = constraint.atom_idxs
        
                # Distance vector after unconstrained update (s)
                s12 = vector(sys.coords[k2], sys.coords[k1], sys.boundary) #& extra allocation

                # Distance vector between the atoms before unconstrained update (r)
                r12 = vector(sys.hidden_storage.ca_coord_storage[k2], sys.hidden_storage.ca_coord_storage[k1], sys.boundary) #& extra allocation

                if abs(norm(s12) - constraint.dist) > ca.dist_tolerance
                    m1_inv = 1/mass(sys.atoms[k1])
                    m2_inv = 1/mass(sys.atoms[k2])
                    a = (m1_inv + m2_inv)^2 * norm(r12)^2 #* can remove sqrt in norm here
                    b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
                    c = norm(s12)^2 - ((constraint.dist)^2) #* can remove sqrt in norm here
                    D = (b^2 - 4*a*c)
                    
                    if ustrip(D) < 0.0
                        @warn "SHAKE determinant negative, setting to 0.0"
                        D = zero(D)
                    end

                    # Quadratic solution for g
                    α1 = (-b + sqrt(D)) / (2*a)
                    α2 = (-b - sqrt(D)) / (2*a)

                    g = abs(α1) <= abs(α2) ? α1 : α2 #* why take smaller one?

                    # Update positions
                    δri1 = r12 .* (-g*m1_inv)
                    δri2 = r12 .* (g*m2_inv)

                    sys.coords[k1] += δri1
                    sys.coords[k2] += δri2
                end
            end
    
        end
        
        max_err = typemin(float_type(sys))*unit(sys.coords[1][1])
        for cluster in ca.clusters
            for constraint in cluster.constraints
                k1, k2 = constraint.atom_idxs
                err = abs(norm(vector(sys.coords[k2], sys.coords[k1], sys.boundary)) - constraint.dist)
                if max_err < err
                    max_err = err
                end
            end
        end

        if max_err < ca.dist_tolerance
            converged = true
        end
    end
end

# function SHAKE_update!(sys, ca::SHAKE_RATTLE, cluster::ConstraintCluster{1})

#     constraint = cluster.constraints[1]

#     # Index of atoms in bond k
#     k1, k2 = constraint.atom_idxs

#     # Distance vector after unconstrained update (s)
#     s12 = vector(sys.coords[k2], sys.coords[k1], sys.boundary) #& extra allocation

#     # Distance vector between the atoms before unconstrained update (r)
#     r12 = vector(ca.coord_storage[k2], ca.coord_storage[k1], sys.boundary) #& extra allocation


#     m1_inv = 1/mass(sys.atoms[k1])
#     m2_inv = 1/mass(sys.atoms[k2])
#     a = (m1_inv + m2_inv)^2 * norm(r12)^2 #* can remove sqrt in norm here
#     b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
#     c = norm(s12)^2 - ((constraint.dist)^2) #* can remove sqrt in norm here
#     D = (b^2 - 4*a*c)
    
#     if ustrip(D) < 0.0
#         @warn "SHAKE determinant negative: $D, setting to 0.0"
#         D = zero(D)
#     end

#     # Quadratic solution for g = 2*λ*dt^2
#     α1 = (-b + sqrt(D)) / (2*a)
#     α2 = (-b - sqrt(D)) / (2*a)

#     g = abs(α1) <= abs(α2) ? α1 : α2 #* why take smaller one?

#     # Update positions
#     δri1 = r12 .* (-g*m1_inv)
#     δri2 = r12 .* (g*m2_inv)

#     sys.coords[k1] += δri1
#     sys.coords[k2] += δri2


#     #Version that modifies accelerations
#     # lambda = g/(2*(dt^2))

#     # accels[k1] += (lambda/m1).*r12
#     # accels[k2] -= (lambda/m2).*r12

# end