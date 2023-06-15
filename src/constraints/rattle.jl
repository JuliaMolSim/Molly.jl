export RATTLE
"""
    RATTLE(dists, is, js)

Constrains a set of bonds to defined distances in a way that the velocities also satisfy the constraints.
"""
struct RATTLE{D,B,T,C,E} <: VelocityConstraintAlgorithm
    A::D
    b::B
    dt::T
    is::C
    js::C
    tolerance::E
end

function RATTLE(is, js, dt, tolerance=1e-10)
    @assert length(is) == length(js) "Constraint lengths do not match"
    #Allocate storage for linear system
    A = zeros(length(is), lengths(is))
    b = zeros(length(is))
    return RATTLE{typeof(dists), typeof(is), typeof(tolerance)}(A, b, dt, is, js, tolerance = tolerance)
end

# Find isolated 2, 3 & 4 atom clusters
function RATTLE_setup(constraint::RATTLE)

    rattle2_pairs = []
    # rattle3_pairs = []
    # rattle4_pairs = []

    #Exhasutive search through constraints to find conflicts -- will not scale to 3,4 bond clusters
    for r1 in eachindex(constraint.is)
        # Atoms that are part of the bond
        i_r1 = constraint.is[r1]
        j_r1 = constraint.js[r1]

        is_isolated = i_r1 ∉ constraint.is[constraint.is .!= r1] &&
                      i_r1 ∉ constraint.js &&
                      j_r1 ∉ constraint.js[constraint.js .!= r1] &&
                      j_r1 ∉ constraints.is

        if is_isolated
            push!(rattle2_pairs, r1)
        end

    end


end


function apply_constraints!(sys, constraint::RATTLE, 
    constraint_cluster::SmallConstraintCluster, unconstrained_velocities)

end

function apply_constraints!(sys, constraint::RATTLE, 
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