export
    apply_position_constraints!, apply_velocity_constraints!, 
    DistanceConstraint, AngleConstraint, 
    NoSystemConstraints, SystemConstraints

include("shake.jl")
include("rattle.jl")


"""
Placeholder struct for [`System`](@ref) constructor when the system does not require constraints.
If you would like to add constraints to the system they can be passed with the 
[`SystemConstraints`](@ref) struct.
"""
struct NoSystemConstraints end
# Add new algorithms here later
PositionConstraintAlgorithms = Union{NoSystemConstraints, SHAKE}
VelocityConstraintAlgorithms = Union{NoSystemConstraints, RATTLE}


"""
A data container used in the construction of a [`System`](@ref) struct that specifies the
constraint algorithms used during the
"""
struct SystemConstraints
    position_constraint <: PositionConstraintAlgorithms
    velocity_constraint <: VelocityConstraintAlgorithms
end

function SystemConstraints(; position_constraint = nothing, velocity_constraint = nothing)
    return SystemConstraints(; position_constraint = position_constraint, velocity_constraint = velocity_constraint)
end


abstract type Constraint end

"""
Constraint between two atoms that maintains the distance between the two atoms.
# Arguments
- `atom_idxs::SVector{Int}` : The indices of atoms in the system participating in this constraint
- `dist::D` : Euclidean distance between the two atoms.
"""
struct DistanceConstraint{D} <: Constraint
    atom_idxs::SVector{Int}
    dist::D
end

"""
Constraint between three atoms that maintains a common angle (e.g. a water molecule)
# Arguments
- `atom_idxs::SVector{Int}` : The indices of atoms in the system participating in this constraint.
    The first atom in this list is assumed to be the central atom if the `central_atom_idx` keyword is not specified
- `angle::A` : Angle between the atoms.
- `central_atom_idx::Int` : The index if the atom in `atom_idxs` that corresponds to the central atom of the constraint.
"""
struct AngleConstraint{A,CA} <: Constraint
    atom_idxs::SVector{Int}
    angle::A
    central_atom_idx::Int
end

function AngleConstraint(atom_idxs, angle; central_atom = 1)
    return {typeof(angle)}(atom_idxs, angle; central_atom = central_atom)
end


"""
A group of constraints  where all atoms participating in the 
cluster are not in another cluster.
"""
abstract type ConstraintCluster end

"""
Cluster of at most 4 bonds between 2,3,4 or 5 atoms around one central atom.
These atoms CANNOT participate in constraints outside of this cluster.
"""
struct SmallConstraintCluster <: ConstraintCluster
    constraints::Vector{<:Constraint}
end

function SmallConstraintCluster(constraints)
    #1 bond, 2 bonds, 2 bonds 1 angle, 3 bonds, 3 bonds 1 angle, 4 bonds
    @assert length(cosntraints) <= 4 "Small constraint can only contain up to 4 constraints"


    return
end 

"""
All other constraints that in the system. These are much
more expensive to solve than the isolated constraints.
"""
struct LargeConstraintCluster <: ConstraintCluster
    constraints::Vector{<:Constraint}
end

# Parse the constraints passed into the System constructor
# into a more useable format.
function build_clusters(sys)


    return small_clusters, large_clusters
end

function check_clusters(sys; small_clusters = nothing, large_clusters = nothing)

    # Check that constraints are unique -- HOW??
    # Check that angle constraints in small are only with 2 bonds??
    # Check that initial conditions satisfy the constraints

    for cluster in small_clusters

    end

    for cluster in large_clusters

    end
end

# For Verlet/Velocity Verlet position step
function apply_position_constraints!(sys, constraint_algo::PositionConstraintAlgorithms, 
    isolated_constraint_clusters::ConstraintCluster, unconstrained_coords)

    for constraint_cluster in isolated_constraint_clusters
        apply_constraints!(sys, constraint_algo, constraint_cluster, unconstrained_coords)
    end
end

# For Velocity Verlet half step
function apply_velocity_constraints!(sys, constraint_algo::VelocityConstraintAlgorithms,
     isolated_constraint_clusters::ConstraintCluster, unconstrained_velocities)

    for constraint_cluster in isolated_constraint_clusters
        apply_constraints!(sys, constraint_algo, constraint_cluster, unconstrained_velocities)
    end
end

apply_position_constraints!(sys::System, constraint_algo::NoSystemConstraints, args...) = nothing
apply_velocity_constraints!(sys::System, constraint_algo::NoSystemConstraints, args...) = nothing
