export
    apply_position_constraints!, apply_velocity_constraints!, 
    DistanceConstraint, AngleConstraint, 
    NoSystemConstraints

include("shake.jl")
include("rattle.jl")


"""
Placeholder struct for [`System`](@ref) constructor when the system does not require constraints.
An example of a constraint is [`SHAKE`](@ref).
"""
struct NoSystemConstraints end

# Add new algorithms here later
PositionConstraintAlgorithms = Union{SHAKE, RATTLE}
VelocityConstraintAlgorithms = Union{RATTLE}


abstract type Constraint end

"""
Constraint between two atoms that maintains the distance between the two atoms.
# Arguments
- `atom_idxs::SVector{Int}` : The indices of atoms in the system participating in this constraint
- `dist::D` : Euclidean distance between the two atoms.
"""
struct DistanceConstraint{D} <: Constraint
    atom_idxs::SVector{2,Int}
    dist::D
end

# """
# Constraint between three atoms that maintains a common angle (e.g. a water molecule)
# # Arguments
# - `atom_idxs::SVector{Int}` : The indices of atoms in the system participating in this constraint.
#     The first atom in this list is assumed to be the central atom if the `central_atom_idx` keyword is not specified
# - `angle::A` : Angle between the atoms.
# - `central_atom_idx::Int` : The index if the atom in `atom_idxs` that corresponds to the central atom of the constraint.
# """
# struct AngleConstraint{A,D} <: Constraint
#     atom_idxs::SVector{3,Int}
#     angle::A
#     dists::D
#     central_atom_idx::Int
# end

# function AngleConstraint(atom_idxs, angle, dists; central_atom = 1)
#     return {typeof(angle), typeof(dists)}(atom_idxs, angle, dists; central_atom = central_atom)
# end


"""
A group of constraints  where all atoms participating in the 
cluster are not in another cluster.
"""
abstract type ConstraintCluster end

"""
Cluster of at most 4 bonds between 2,3,4 or 5 atoms around one central atom.
These atoms CANNOT participate in constraints outside of this cluster.
Small clusters include: 
    1 bond, 2 bonds, 1 angle, 3 bonds, 1 bond 1 angle, 4 bonds
Note that an angle constraints will be implemented as 3 distance constraints.
"""
struct SmallConstraintCluster <: ConstraintCluster
    constraints::Vector{<:Constraint}
end

function SmallConstraintCluster(constraints)
    @assert length(cosntraints) <= 4 "Small constraint can only contain up to 4 constraints"
    return SmallConstraintCluster(constraints)
end 


"""
All constraints in the system that contain more than 5 atoms. These are much
more expensive to solve than an isolated constraint.
"""
struct LargeConstraintCluster <: ConstraintCluster
    constraints::Vector{<:Constraint}
end


# """
# Loops through all of the constraints in the system and converts the 
# angle constraints into 3 distance constraints.
# """
# function angle_to_dist_constraints!(sys)
#     for (i, cosntraint) in enumerate(sys.constraints)
#         if typeof(constraint) == AngleConstraint
#             d1 = DistanceConstraint()
#             d2 = DistanceConstraint()
#             d3 = DistanceConstraint()
#         end
#     end
# end

function constraint_setup(sys)
    angle_to_dist_constraints!(sys)
    small_clusters, large_clusters = build_clusters(sys)
    check_initial_constraints(sys, small_clusters, large_clusters)
    return small_clusters, large_clusters
end



"""
Parse the constraints into small and large clusters. 
Small clusters can be solved faster whereas large
clusters fall back to slower, generic methods.
"""
function build_clusters(sys)

    # Build directed graph so we don't double count
    # bonds when calling neighbors()
    n_atoms = length(sys.atoms)
    constraint_graph = SimpleDiGraph(n_atoms)

    # Add edges to graph
    for constraint in sys.constraints
        edge_added = add_edge!(constraint_graph, constraint.atom_idxs[1], constraint.atom_idxs[2])
        if !edge_added
            @warn "Duplicated constraint in System. It will be ignored."
        end
    end

    cc = connected_components(constraint_graph)

    small_clusters = Vector{SmallConstraintCluster}(undef,0)
    large_clusters = Vector{LargeConstraintCluster}(undef,0)

    #Loop through connected regions and convert to clusters
    for (cluster_idx, atom_idxs) in enumerate(cc)
        if length(atom_ids) <= 4
            for ai in atom_idxs
                neigh_idxs = neighbors(constraint_graph, ai)
                for neigh_idx in neigh_idxs
                    push!(small_clusters, SmallConstraintCluster(ai, neigh_idx))
                end
            end
        else
            for ai in atom_idxs
                neigh_idxs = neighbors(constraint_graph, ai)
                for neigh_idx in neigh_idxs
                    push!(large_clusters, LargeConstraintCluster(ai, neigh_idx))
                end
            end
        end
    end

    return small_clusters, large_clusters
end

"""
Verifies that the initial conditions of the system satisfy the constraints.
"""
function check_initial_constraints(sys, small_clusters, large_clusters)

    for cluster in small_clusters
        for constraint in cluster.constraints
            r_actual = norm(sys.coords[constraint.atom_idxs[1]] .- sys.coords[constraint.atom_idxs[2]])
            if !isapprox(r_actual, constraint.dist)
                throw(ArgumentError("Constraint between atoms $(constraint.atom_idxs[1]) 
                    and $(constraint.atom_idxs[2]) is not satisfied by the initial conditions."))
            end
        end
    end

    for cluster in large_clusters
        for constraint in cluster.constraints
            r_actual = norm(sys.coords[constraint.atom_idxs[1]] .- sys.coords[constraint.atom_idxs[2]])
            if !isapprox(r_actual, constraint.dist)
                throw(ArgumentError("Constraint between atoms $(constraint.atom_idxs[1]) 
                    and $(constraint.atom_idxs[2]) is not satisfied by the initial conditions."))
            end
        end
    end
end

# For Verlet/Velocity Verlet position step
function apply_position_constraints!(sys, constraint_algo::PositionConstraintAlgorithms, 
    isolated_constraint_clusters::ConstraintCluster, unconstrained_coords)

    if typeof(constraint_algo) == RATTLE
        for constraint_cluster in isolated_constraint_clusters
            #WILL BREAK ATM -- FIX LATER
            apply_constraints!(sys, SHAKE, constraint_cluster, unconstrained_coords)
        end
    else
        for constraint_cluster in isolated_constraint_clusters
            apply_constraints!(sys, constraint_algo, constraint_cluster, unconstrained_coords)
        end
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

