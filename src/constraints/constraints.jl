export
    apply_position_constraints!, apply_velocity_constraints!, IsolatedDistanceConstraint

include("shake.jl")
include("rattle.jl")

# Add new algorithms here later
PositionConstraintAlgorithms = Union{SHAKE}
VelocityConstraintAlgorithms = Union{RATTLE}

abstract type Constraint end

struct DistanceConstraint{D} <: Constraint
    atom_idxs::SVector{Int}
    dist::D
end

# Think this can be implemented as distance constraints
# LAMMPS only allows this with 2 bond constraints and 1 angle constraint
# struct AngleConstraint{A} <: Constraint
#     atom_idxs::SVector{Int}
#     angle::A
# end

abstract type ConstraintCluster end

"""
Cluster of at most 4 bonds between 2,3,4 or 5 atoms around one central atom.
These atoms cannot participate in constraints outside of this cluster.
"""
struct SmallConstraintCluster <: ConstraintCluster
    constraints::Vector{<:Constraint}
end

function SmallConstraintCluster(constraints)
    #1 bond, 2 bonds, 2 bonds 1 angle, 3 bonds, 3 bonds 1 angle, 4 bonds
    @assert length(cosntraints) < 5 "Small constraint can only contain up to 4 constraints"
    return
end 

"""
All other constraints that are not isoalted in the system. These are much
more expensive to solve than the isolated constraints.
"""
struct LargeConstraintCluster <: ConstraintCluster
    constraints::Vector{<:Constraint}
end

#Update dispatch later
function apply_position_constraints!(sys, constraint_algo::PositionConstraintAlgorithms, 
    isolated_constraint_clusters::ConstraintCluster, unconstrained_coords)

    for constraint_cluster in isolated_constraint_clusters
        apply_constraints!(sys, constraint_algo, constraint_cluster, unconstrained_coords)
    end
end

function apply_velocity_constraints!(sys, constraint_algo::VelocityConstraintAlgorithms,
     isolated_constraint_clusters::ConstraintCluster, unconstrained_velocities)

    for constraint_cluster in isolated_constraint_clusters
        apply_constraints!(sys, constraint_algo, constraint_cluster, unconstrained_velocities)
    end
end


