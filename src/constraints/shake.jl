export SHAKE

"""
SHAKE_RATTLE(coords, tolerance, init_posn_tol)

A constraint algorithm to set bonds distances in a simulation. Velocity constraints will be imposed
for simulators that integrate velocities (e.g. [`VelocityVerlet`](@ref)).

See [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/) for a derivation of the linear
system solved to satisfy the RATTLE algorithm.

# Arguments
- coord_storage: An empty array with same type and size as coords in sys, `similar(sys.coords)` is best
- vel_storage: An empty array with the same type and size as velocities in sys, `similar(sys.velocities)` is best.
    This argument can safely be passed as `nothing` if the integrator does not use velocity constraints like [`StormerVerlet`](@ref).
- tolerance: Tolerance used to end iterative procedure when calculating constraints.
    Default is 1e-4.
- init_posn_tol: Tolerance used when checking if system initial positions satisfy position constraints. 
    Default is `nothing.`
"""
struct SHAKE_RATTLE{UC, VC, T, I} <: ConstraintAlgorithm
    coord_storage::UC #Used as storage to avoid re-allocating arrays
    vel_storage::Union{VC, Nothing}
    tolerance::T
    init_posn_tol::Union{I,Nothing}
end

function SHAKE_RATTLE(coord_storage, vel_storage; tolerance=1e-4, init_posn_tol = nothing)
    return SHAKE_RATTLE{typeof(coord_storage), typeof(vel_storage), typeof(tolerance), typeof(init_posn_tol)}(
        coord_storage, vel_storage, tolerance, init_posn_tol)
end

save_positions!(constraint_algo::SHAKE_RATTLE, c) = (constraint_algo.coord_storage .= c)

function apply_position_constraints!(sys::System, constraint_algo::SHAKE_RATTLE, accels, dt;
     n_threads::Integer=Threads.nthreads())

    Threads.@threads for group_id in 1:n_threads #& can only paralellize over independent clusters
        for i in group_id:n_threads:length(sys.constraints)
            SHAKE_update!(sys, constraint_algo, sys.constraints[i])
        end
    end

    return sys
end


function SHAKE_update!(sys, ca::Union{SHAKE,RATTLE}, cluster::ConstraintCluster{1})

    constraint = cluster.constraints[1]

    # Index of atoms in bond k
    k1, k2 = constraint.atom_idxs

    # Distance vector after unconstrained update (s)
    s12 = vector(sys.coords[k2], sys.coords[k1], sys.boundary) #& extra allocation

    # Distance vector between the atoms before unconstrained update (r)
    r12 = vector(ca.coord_storage[k2], ca.coord_storage[k1], sys.boundary) #& extra allocation


    m1_inv = 1/mass(sys.atoms[k1])
    m2_inv = 1/mass(sys.atoms[k2])
    a = (m1_inv + m2_inv)^2 * norm(r12)^2 #* can remove sqrt in norm here
    b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
    c = norm(s12)^2 - ((constraint.dist)^2) #* can remove sqrt in norm here
    D = (b^2 - 4*a*c)
    
    if ustrip(D) < 0.0
        @warn "SHAKE determinant negative: $D, setting to 0.0"
        D = zero(D)
    end

    # Quadratic solution for g = 2*λ*dt^2
    α1 = (-b + sqrt(D)) / (2*a)
    α2 = (-b - sqrt(D)) / (2*a)

    g = abs(α1) <= abs(α2) ? α1 : α2 #* why take smaller one?

    # Update positions
    δri1 = r12 .* (-g*m1_inv)
    δri2 = r12 .* (g*m2_inv)

    sys.coords[k1] += δri1
    sys.coords[k2] += δri2


    #Version that modifies accelerations
    # lambda = g/(2*(dt^2))

    # accels[k1] += (lambda/m1).*r12
    # accels[k2] -= (lambda/m2).*r12

end


# function SHAKE_updates!(sys, ca::SHAKE, clusters::ConstraintCluster{1})
#     converged = false

#     while !converged
#         for cluster in clusters
#             for constraint in cluster.constraints #* CANNOT Parallelize over this

#                 k1, k2 = constraint.atom_idxs
        
#                 # Distance vector after unconstrained update (s)
#                 s12 = vector(sys.coords[k2], sys.coords[k1], sys.boundary) #& extra allocation

#                 # Distance vector between the atoms before unconstrained update (r)
#                 r12 = vector(ca.coord_storage[k2], ca.coord_storage[k1], sys.boundary) #& extra allocation

#                 if abs(norm(s12) - constraint.dist) > ca.tolerance
#                     m1_inv = 1/mass(sys.atoms[k1])
#                     m2_inv = 1/mass(sys.atoms[k2])
#                     a = (m1_inv + m2_inv)^2 * norm(r12)^2 #* can remove sqrt in norm here
#                     b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
#                     c = norm(s12)^2 - ((constraint.dist)^2) #* can remove sqrt in norm here
#                     D = (b^2 - 4*a*c)
                    
#                     if ustrip(D) < 0.0
#                         @warn "SHAKE determinant negative, setting to 0.0"
#                         D = zero(D)
#                     end

#                     # Quadratic solution for g
#                     α1 = (-b + sqrt(D)) / (2*a)
#                     α2 = (-b - sqrt(D)) / (2*a)

#                     g = abs(α1) <= abs(α2) ? α1 : α2 #* why take smaller one?

#                     # Update positions
#                     δri1 = r12 .* (-g*m1_inv)
#                     δri2 = r12 .* (g*m2_inv)

#                     sys.coords[k1] += δri1
#                     sys.coords[k2] += δri2
#                 end
#             end
    
#         end
        
#         #* FIX
#         lengths = map(sys.constraints) do cluster
#             for constraint in cluster.constraints
#                 k1, k2 = constraint.atom_idxs
#             return norm(vector(sys.coords[k2], sys.coords[k1], sys.boundary)) - cluster.constraints[1].dist
#         end

#         if maximum(lengths) < ca.tolerance
#             converged = true
#         end
#     end
# end