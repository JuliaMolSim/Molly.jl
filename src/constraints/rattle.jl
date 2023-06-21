export RATTLE
"""
    RATTLE(dists, is, js)

Constrains a set of bonds to defined distances in a way that the velocities also satisfy the constraints.
"""
struct RATTLE{C,E,V} <: VelocityConstraintAlgorithm
    is::C
    js::C
    tolerance::E
    unconstrained_velocity_storage::V
end

function RATTLE(is, js, unconstrained_velocity_storage; tolerance=1e-10)
    @assert length(is) == length(js) "Constraint lengths do not match"
    return RATTLE{typeof(dists), typeof(is), typeof(tolerance), typeof(unconstrained_velocity_storage)}(
        is, js, unconstrained_velocity_storage, tolerance = tolerance)
end


function apply_position_constraints!(sys, constraint::RATTLE, 
    constraint_cluster::SmallConstraintCluster, unconstrained_coords)

end

function apply_position_constraints!(sys, constraint::RATTLE, 
    constraint_cluster::LargeConstraintCluster, unconstrained_coords)

end

function apply_velocity_constraints!(sys, constraint::RATTLE, 
    constraint_cluster::SmallConstraintCluster, unconstrained_velocities)

end

function apply_velocity_constraints!(sys, constraint::RATTLE, 
    constraint_cluster::LargeConstraintCluster, unconstrained_velocities)

end



# This is just for half step of velocity
# Rattle for a single distance constraint between atoms i and j
# Atoms i and j do NOT participate in any other constraints
function RATTLE2(sys, k, constraint::RATTLE)

    # Index of atoms in bond k
    k1 = constraint.is[k]
    k2 = constraint.js[k]

    # Inverse of masses of atoms in bond k
    inv_m1 = 1/mass(sys.atoms[k1])
    inv_m2 = 1/mass(sys.atoms[k2])

    # Distance vector between the atoms after SHAKE constraint
    r_k1k2 = vector(sys.coords[k2], sys.coords[k1], sys.boundary)

    # Difference between unconstrainted velocities
    v_k1k2 = sys.velocities[k2] .- sys.velocities[k1]

    # Re-arrange constraint equation to solve for Lagrange multiplier
    # Technically this has a factor of dt which cancels out in the velocity update
    λₖ = -dot(r_k1k2,v_k1k2)/(dot(r_k1k2,r_k1k2)*(inv_m1 + inv_m2))

    # Correct velocities
    sys.velocities[k1] .-= (inv_m1 .* λₖ .* r_k1k2)
    sys.coords[k2] .+= (inv_m1 .* λₖ .* r_k1k2)

end
