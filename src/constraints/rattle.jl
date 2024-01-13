export RATTLE

"""
RATTLE(coords; tolerance=1e-4, init_posn_tol = nothing)

Constrains a set of bonds to defined distances in a way that the velocities also satisfy the constraints.

See [this paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/) for a derivation of the linear
system solved to satisfy the RATTLE algorithm.

# Arguments
- coords: An empty array with same type and size as coords in sys, `similar(sys.coords)` is best
- tolerance: Tolerance used to end iterative procedure when calculating constraint forces. This
    is not a tolerance on the error in positions or velocities, but a lower `tolerance` should
    result in smaller error. Default is `1e-4`.
- init_posn_tol: Tolerance used when checking if system initial positions satisfy position constraints. 
    Default is `nothing.`
"""
struct RATTLE{CS,T,I} <: PositionAndVelocityConstraintAlgorithm
    coord_storage::CS 
    tolerance::T
    init_posn_tol::Union{I,Nothing}
end

function RATTLE(coords; tolerance=1e-4, init_posn_tol = nothing)
    return RATTLE{typeof(coords), typeof(tolerance), typeof(init_posn_tol)}(
        coords, tolerance, init_posn_tol)
end

save_positions!(constraint_algo::RATTLE, c) = (constraint_algo.coord_storage .= c)


function apply_position_constraint!(sys, constraint_algo::RATTLE, 
    constraint_cluster::ConstraintCluster{1})

    SHAKE_update!(sys, constraint_algo, constraint_cluster)

end

function apply_velocity_constraint!(sys, constraint_algo::RATTLE, 
    constraint_cluster::ConstraintCluster)

    RATTLE_update!(sys, constraint_cluster)

end


"""
RATTLE solution for a single distance constraint between atoms i and j,
where atoms i and j do NOT participate in any other constraints.
"""
function RATTLE_update!(sys, cluster::ConstraintCluster{1})

    constraint = cluster.constraints[1]

    # Index of atoms in bond k
    k1, k2 = constraint.atom_idxs

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
    sys.velocities[k1] -= (inv_m1 .* λₖ .* r_k1k2)
    sys.velocities[k2] += (inv_m2 .* λₖ .* r_k1k2)

end

# TODO
RATTLE_update!(sys, ca::RATTLE, cluster::ConstraintCluster{2}) = nothing
RATTLE_update!(sys, ca::RATTLE, cluster::ConstraintCluster{3}) = nothing
RATTLE_update!(sys, ca::RATTLE, cluster::ConstraintCluster{4}) = nothing

# Implement later
# RATTLE_update!(sys, cluster::ConstraintCluster{D}) where {D >= 5} = nothing