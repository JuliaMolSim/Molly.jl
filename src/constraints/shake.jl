export SHAKE

"""
    SHAKE(coords, tolerance)

Constrains a set of bonds to defined distances.
"""
struct SHAKE{UC, T} <: ConstraintAlgorithm
    unupdated_coords::UC #Used as storage to avoid re-allocating arrays
    tolerance::T
end

function SHAKE(unupdated_coords; tolerance=1e-6u"Å")
    return SHAKE{typeof(unupdated_coords), typeof(tolerance)}(
        unupdated_coords, tolerance)
end

save_positions!(constraint_algo::SHAKE, c) = (constraint_algo.unupdated_coords .= c)
save_velocities!(constraint_algo::SHAKE, v) = constraint_algo


function apply_position_constraint!(sys, constraint_algo::SHAKE, 
    constraint_cluster::ConstraintCluster{1}, accels_t, dt)

    SHAKE_update!(sys, constraint_algo, constraint_cluster, accels_t, dt)

end


function apply_velocity_constraint!(sys, constraint_algo::SHAKE, 
    constraint_cluster::ConstraintCluster)

    #TODO: SHould these just be nothing, or do you arbitrarblty zero out the bond velocities???

end


#TODO: I do not think we actually need to iterate here its analytical solution
#TODO: Modify forces instead of positions?
function SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{1}, accels_t, dt)

    constraint = cluster.constraints[1]

    # Index of atoms in bond k
    k1, k2 = constraint.atom_idxs

    converged = false

    while !converged #TODO THIS SHOULDNT BE NECESSARY

    # println("r0 of atom $(k1): $(ca.unupdated_coords[k1])")
    # println("r0 of atom $(k2): $(ca.unupdated_coords[k2])")
    # println("s0 of atom $(k1): $(sys.coords[k1])")
    # println("s0 of atom $(k2): $(sys.coords[k2])")

        # Distance vector between the atoms before unconstrained update
        r01 = vector(ca.unupdated_coords[k2], ca.unupdated_coords[k1], sys.boundary)

        # Distance vector after unconstrained update
        s01 = vector(sys.coords[k2], sys.coords[k1], sys.boundary)
        # println("r0 $(r01)")
        # println("s0 $(s01)")

        if abs(norm(s01) - constraint.dist) > ca.tolerance
            m1 = mass(sys.atoms[k1])
            m2 = mass(sys.atoms[k2])
            a = (1/m1 + 1/m2)^2 * norm(r01)^2
            b = 2 * (1/m1 + 1/m2) * dot(r01, s01)
            c = norm(s01)^2 - ((constraint.dist)^2)
            D = (b^2 - 4*a*c)
            
            if ustrip(D) < 0.0
                @warn "SHAKE determinant negative, setting to 0.0"
                throw(error())
                D = zero(D)
            end

            # Quadratic solution for g (technically 2*g*dt^2 but 2*dt^2 will cancel later)
            α1 = (-b + sqrt(D)) / (2*a)
            α2 = (-b - sqrt(D)) / (2*a)

            g = abs(α1) <= abs(α2) ? α1 : α2

            #Update positions and forces
            sys.coords[k1] += r01 .* (g/m1)
            sys.coords[k2] -= r01 .* (g/m2)

            # accels_t[k1] += ustrip.((g/(2*dt^2)).*r01) * unit(accels_t[k1][1])
            # accels_t[k2] -= ustrip.((g/(2*dt^2)).*r01) * unit(accels_t[k1][1])

            # println(abs(norm(vector(sys.coords[k2], sys.coords[k1], sys.boundary)) - constraint.dist))
        end

        length = abs(norm(vector(sys.coords[k2], sys.coords[k1], sys.boundary)) - constraint.dist)

        if length < ca.tolerance
            converged = true
        end
    end
    return accels_t
end

# TODO: Manually implement matrix inversion
SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{2}) = nothing
SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{3}) = nothing
SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{4}) = nothing

#Implement later, see:
# https://onlinelibrary.wiley.com/doi/abs/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/
#Currently code is setup for independent constraints, but M-SHAKE does not care about that
# SHAKE_update!(sys, cluster::ConstraintClusterP{D}) where {D >= 5} = nothing
