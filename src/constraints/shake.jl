export SHAKE_RATTLE,
    check_position_constraints,
    check_velocity_constraints

"""
SHAKE_RATTLE(constraints::AbstractVector{<:Constraint}, n_atoms, dist_tolerance, vel_tolerance)

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
struct SHAKE_RATTLE{CC, D, V}
    clusters::CC
    dist_tolerance::D
    vel_tolerance::V
end

function SHAKE_RATTLE(constraints, n_atoms::Int, dist_tolerance, vel_tolerance)

    #If this becomes a memory issue n_atoms could probably be number of atoms constrained
    clusters = build_clusters(n_atoms, constraints)

    return SHAKE_RATTLE{eltype(clusters), typeof(dist_tolerance), typeof(vel_tolerance)}(clusters, dist_tolerance, vel_tolerance)
end

function position_constraints!(sys::System, constraint_algo::SHAKE_RATTLE, coord_storage;
     n_threads::Integer=Threads.nthreads())

    SHAKE_updates!(sys, constraint_algo, coord_storage)

    return sys
end



function SHAKE_updates!(sys, ca::SHAKE_RATTLE, coord_storage)
    converged = false

    while !converged
        for cluster in ca.clusters #& illegal to parallelize this
            for constraint in cluster.constraints

                k1, k2 = constraint.i, constraint.j
        
                # Distance vector after unconstrained update (s)
                s12 = vector(sys.coords[k2], sys.coords[k1], sys.boundary) #& extra allocation

                # Distance vector between the atoms before unconstrained update (r)
                r12 = vector(coord_storage[k2], coord_storage[k1], sys.boundary) #& extra allocation

                println(ustrip(norm(s12) - constraint.dist))

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

                    g = abs(α1) <= abs(α2) ? α1 : α2

                    # Update positions
                    δri1 = r12 .* (-g*m1_inv)
                    δri2 = r12 .* (g*m2_inv)

                    sys.coords[k1] += δri1
                    sys.coords[k2] += δri2
                end
            end
    
        end
        
        converged = check_position_constraints(sys, ca)
    end
end



function velocity_constraints!(sys::System, constraint_algo::SHAKE_RATTLE;
    n_threads::Integer=Threads.nthreads())

   RATTLE_updates!(sys, constraint_algo)

   return sys
end

function RATTLE_updates!(sys, ca::SHAKE_RATTLE)

   converged = false

   while !converged
       for cluster in ca.clusters #& illegal to parallelize this
           for constraint in cluster.constraints

               # Index of atoms in bond k
               k1, k2 = constraint.i, constraint.j

               # Inverse of masses of atoms in bond k
               inv_m1 = 1/mass(sys.atoms[k1])
               inv_m2 = 1/mass(sys.atoms[k2])

               # Distance vector between the atoms after SHAKE constraint
               r_k1k2 = vector(sys.coords[k2], sys.coords[k1], sys.boundary)

               # Difference between unconstrainted velocities
               v_k1k2 = sys.velocities[k2] .- sys.velocities[k1]

               err = abs(dot(r_k1k2, v_k1k2))
               if err > ca.vel_tolerance
                   # Re-arrange constraint equation to solve for Lagrange multiplier
                   # Technically this has a factor of dt which cancels out in the velocity update
                   λₖ = -dot(r_k1k2,v_k1k2)/(dot(r_k1k2,r_k1k2)*(inv_m1 + inv_m2))

                   # Correct velocities
                   sys.velocities[k1] -= (inv_m1 .* λₖ .* r_k1k2)
                   sys.velocities[k2] += (inv_m2 .* λₖ .* r_k1k2)
               end

           end
       end

       converged = check_velocity_constraints(sys, ca)
   end

end



"""
check_position_constraints(sys::System, ca::SHAKE_RATTLE)

Checks if the position constraints are satisfied by current positions of `sys`.
"""
function check_position_constraints(sys::System, ca::SHAKE_RATTLE)

    max_err = typemin(float_type(sys))*unit(sys.coords[1][1])
    for cluster in ca.clusters
        for constraint in cluster.constraints
            err = abs(norm(vector(sys.coords[constraint.j], sys.coords[constraint.i], sys.boundary)) - constraint.dist)
            if max_err < err
                max_err = err
            end
        end
    end

    return (max_err < ca.dist_tolerance)
end

"""
check_velocity_constraints(sys::System, ca::SHAKE_RATTLE)

Checks if the velocity constraints are satisfied by current velocities of `sys`.
"""
function check_velocity_constraints(sys::System, ca::SHAKE_RATTLE)

    max_err = typemin(float_type(sys))*unit(sys.velocities[1][1])*unit(sys.coords[1][1])
    for cluster in ca.clusters
        for constraint in cluster.constraints
            err = abs(dot(vector(sys.coords[constraint.j], sys.coords[constraint.i], sys.boundary),
                                (sys.velocities[constraint.j] .- sys.velocities[constraint.i])))
            if max_err < err
                max_err = err
            end
        end
    end

    return (max_err < ca.vel_tolerance)
end
