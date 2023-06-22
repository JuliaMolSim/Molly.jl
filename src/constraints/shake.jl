export SHAKE

"""
    SHAKE(unconstrained_coords, tolerance)

Constrains a set of bonds to defined distances.
"""
struct SHAKE{UC, T} <: ConstraintAlgorithm
    unconstrained_coords::UC #Used as storage to avoid re-allocating arrays
    tolerance::T
end

function SHAKE(unconstrained_coords; tolerance=1e-10u"nm")
    return SHAKE{typeof(unconstrained_coords), typeof(tolerance)}(
        unconstrained_coords, tolerance)
end

#Avoid allocations and un-necessary copies
update_unconstrained_positions!(constraint_algo::SHAKE, uc) = (constraint_algo.unconstrained_coords .= uc)
update_unconstrained_velocities!(constraint_algo::SHAKE, uv) = constraint_algo

function apply_position_constraint!(sys, constraint::SHAKE, 
    constraint_cluster::ConstraintCluster)

    SHAKE_algo(sys, constraint_cluster, constraint_cluster.unconstrained_coords)

end


function apply_velocity_constraint!(sys, constraint::SHAKE, 
    constraint_cluster::ConstraintCluster)

    #TODO: SHould these just be nothing, or do you arbitrarblty zero out the bond velocities???

end


#TODO: I do not think we actually need to iterate here its analytical solution
function SHAKE_algo(sys, cluster::ConstraintCluster{1}, unconstrained_coords)

    constraint = cluster.constraints[1]

    # Index of atoms in bond k
    k1, k2 = constraint.atom_idxs

    converged = false

    while !converged #TODO Dont think this is necessary

        # Distance vector between the atoms before unconstrained update
        r01 = vector(unconstrained_coords[k2], unconstrained_coords[k1], sys.boundary)

        # Distance vector after unconstrained update
        s01 = vector(sys.coords[k2], sys.coords[k1], sys.boundary)

        if abs(norm(s01) - constraint.dist) > constraint.tolerance
            m0 = mass(sys.atoms[k1])
            m1 = mass(sys.atoms[k2])
            a = (1/m0 + 1/m1)^2 * norm(r01)^2
            b = 2 * (1/m0 + 1/m1) * dot(r01, s01)
            c = norm(s01)^2 - ((constraint.dist)^2)
            D = (b^2 - 4*a*c)
            
            if ustrip(D) < 0.0
                @warn "SHAKE determinant negative, setting to 0.0"
                D = zero(D)
            end

            # Quadratic solution for g
            α1 = (-b + sqrt(D)) / (2*a)
            α2 = (-b - sqrt(D)) / (2*a)

            g = abs(α1) <= abs(α2) ? α1 : α2

            # g needs to be divided by dt^2???

            # Update positions
            δrk1 = r01 .* ( g/m0)
            δrk2 = r01 .* (-g/m1)

            sys.coords[k1] += δrk1
            sys.coords[k2] += δrk2
        end

    
        length = [abs(norm(vector(sys.coords[k2], sys.coords[k1], sys.boundary)) - constraint.dist)]

        if maximum(length) < constraint.tolerance
            converged = true
        end
    end
end

# TODO
SHAKE_algo(sys, cluster::ConstraintCluster{2}) = nothing
SHAKE_algo(sys, cluster::ConstraintCluster{3}) = nothing
SHAKE_algo(sys, cluster::ConstraintCluster{4}) = nothing

#Implement later, see:
# https://onlinelibrary.wiley.com/doi/abs/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/
#Currently code is setup for independent constraints, but M-SHAKE does not care about that
# SHAKE_algo(sys, cluster::ConstraintClusterP{D}) where {D >= 5} = nothing
