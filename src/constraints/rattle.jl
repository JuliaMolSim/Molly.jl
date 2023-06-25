export RATTLE
"""
    RATTLE(tolerance, coords, velocities)

Constrains a set of bonds to defined distances in a way that the velocities also satisfy the constraints.

See [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/) for a derivation of the linear
system solved to satisfy the RATTLE algorithm.
"""
struct RATTLE{UC,UV,T} <: ConstraintAlgorithm
    unupdated_coords::UC 
    unconstrained_velocities::UV #Used as storage to avoid re-allocating arrays
    tolerance::T
end

function RATTLE(coords, unconstrained_velocities; tolerance=1e-10u"nm")
    return RATTLE{typeof(unupdated_coords), typeof(unconstrained_velocities),typeof(tolerance)}(
            unupdated_coords, unconstrained_velocities, tolerance = tolerance)
end

save_positions!(constraint_algo::RATTLE, c) = (constraint_algo.unupdated_coords .= c)
save_velocities!(constraint_algo::RATTLE, v) = (constraint_algo.unconstrained_velocities .= v)


function apply_position_constraint!(sys, constraint_algo::RATTLE, 
    constraint_cluster::ConstraintCluster)

    SHAKE_algo(sys, constraint_cluster, constraint_algo.unupdated_coords)

end


function apply_velocity_constraint!(sys, constraint_algo::RATTLE, 
    constraint_cluster::ConstraintCluster)

    RATTLE_algo(sys, constraint_cluster, constraint_algo.unconstrained_velocities)

end



# This is just for half step of velocity

"""
RATTLE solution for a single distance constraint between atoms i and j,
where atoms i and j do NOT participate in any other constraints.
"""
function RATTLE_algo(sys, cluster::ConstraintCluster{1}, unconstrainted_velocities)

    constraint = cluster.constraints[1]

    # Index of atoms in bond k
    k1, k2 = constraint.atom_idxs

    # Inverse of masses of atoms in bond k
    inv_m1 = 1/mass(sys.atoms[k1])
    inv_m2 = 1/mass(sys.atoms[k2])

    # Distance vector between the atoms after SHAKE constraint
    r_k1k2 = vector(sys.coords[k2], sys.coords[k1], sys.boundary)

    # Difference between unconstrainted velocities
    v_k1k2 = unconstrainted_velocities[k2] .- unconstrainted_velocities[k1]

    # Re-arrange constraint equation to solve for Lagrange multiplier
    # Technically this has a factor of dt which cancels out in the velocity update
    λₖ = -dot(r_k1k2,v_k1k2)/(dot(r_k1k2,r_k1k2)*(inv_m1 + inv_m2))

    # Correct velocities
    sys.velocities[k1] .-= (inv_m1 .* λₖ .* r_k1k2)
    sys.coords[k2] .+= (inv_m1 .* λₖ .* r_k1k2)

end

# TODO
RATTLE_algo(sys, cluster::ConstraintCluster{2}, unconstrainted_velocities) = nothing
RATTLE_algo(sys, cluster::ConstraintCluster{3}, unconstrainted_velocities) = nothing
RATTLE_algo(sys, cluster::ConstraintCluster{4}, unconstrainted_velocities) = nothing

# Implement later
# RATTLE_algo(sys, cluster::ConstraintCluster{D}) where {D >= 5} = nothing