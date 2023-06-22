export
    DistanceConstraint, AngleConstraint, 
    NoSystemConstraints

"""
Supertype for all constraint algorithms.
"""
abstract type ConstraintAlgorithm end


"""
Placeholder struct for [`System`](@ref) constructor when the system does not require constraints.
An example of a constraint is [`SHAKE`](@ref).
"""
struct NoSystemConstraints end


"""
Supertype for all types of constraint.
"""
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
# - `atom_idxs::SVector{2,Int}` : The indices of atoms in the system participating in this constraint.
#     The first atom in this list is assumed to be the central atom if the `central_atom_idx` keyword is not specified
# - `angle::A` : Angle between the atoms.
# - `central_atom_idx::Int` : The index if the atom that corresponds to the central atom of the constraint.
# """
# struct AngleConstraint{A,D} <: Constraint
#     edge_atom_idxs::SVector{2,Int}
#     central_atom_idx::Int
#     angle::A
#     dists::D
# end

# function AngleConstraint(atom_idxs, angle, dists, central_atom)
#     return {typeof(angle), typeof(dists)}(atom_idxs, angle, dists, central_atom)
# end


"""
Atoms in a cluster do not participate in any other constraints outside of that cluster.
"Small" clusters contain at most 4 bonds between 2,3,4 or 5 atoms around one central atom.
Small clusters include: 1 bond, 2 bonds, 1 angle, 3 bonds, 1 bond 1 angle, 4 bonds
Note that an angle constraints will be implemented as 3 distance constraints. These constraints
use special methods that improve computational performance. Any constraint not listed above
will come at a performance penatly.
"""
struct ConstraintCluster{N}
    constraints::SVector{N, <:Constraint}
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

function constraint_setup(coords, constraints)
    n_atoms = length(coords)
    # angle_to_dist_constraints!(sys)
    clusters = build_clusters(n_atoms, constraints)
    check_initial_constraints(coords, clusters)
    return clusters
end



"""
Parse the constraints into small and large clusters. 
Small clusters can be solved faster whereas large
clusters fall back to slower, generic methods.
"""
function build_clusters(n_atoms, constraints)

    constraint_graph = SimpleDiGraph(n_atoms)
    idx_dist_pairs = spzeros(n_atoms, n_atoms) * unit(constraints[1].dist)

    # Store constraints as directed edges, direction is arbitrary but necessary
    for constraint in constraints
        edge_added = add_edge!(constraint_graph, constraint.atom_idxs[1], constraint.atom_idxs[2])
        if edge_added
            idx_dist_pairs[constraint.atom_idxs[1],constraint.atom_idxs[2]] = constraint.dist
            idx_dist_pairs[constraint.atom_idxs[2],constraint.atom_idxs[1]] = constraint.dist
        else
            @warn "Duplicated constraint in System. It will be ignored."
        end
    end

    # Get groups of constraints that are connected to eachother 
    cc = connected_components(constraint_graph)

    # Initialze empty vector of clusters
    clusters = Vector{ConstraintCluster}(undef, 0)

    #Loop through connected regions and convert to clusters
    for (cluster_idx, atom_idxs) in enumerate(cc)
        # Loop atoms in connected region to build cluster
        connected_constraints = Vector{DistanceConstraint}(undef, 0)
        for ai in atom_idxs
            neigh_idxs = neighbors(constraint_graph, ai)
            for neigh_idx in neigh_idxs
                push!(connected_constraints,
                     DistanceConstraint(SVector(ai, neigh_idx), idx_dist_pairs[ai,neigh_idx]))
            end
        end
        connected_constraints = convert(SVector{length(connected_constraints)}, connected_constraints)
        push!(clusters, ConstraintCluster(connected_constraints))
    end

    return clusters
end

"""
Verifies that the initial conditions of the system satisfy the constraints.
"""
function check_initial_constraints(coords, clusters)

    for cluster in clusters
        for constraint in cluster.constraints
            r_actual = norm(coords[constraint.atom_idxs[1]] .- coords[constraint.atom_idxs[2]])
            if !isapprox(r_actual, constraint.dist)
                throw(ArgumentError("Constraint between atoms $(constraint.atom_idxs[1]) 
                    and $(constraint.atom_idxs[2]) is not satisfied by the initial conditions."))
            end
        end
    end

end

# For Verlet/Velocity Verlet position step
"""
Updates the coordinates of a [`System`](@ref) based on the constraints.
"""
function apply_position_constraints!(sys, constraint_algo::ConstraintAlgorithm, 
    constraint_clusters::ConstraintCluster)

    for constraint_cluster in constraint_clusters
        apply_position_constraint!(sys, constraint_algo, constraint_cluster)
    end
end

# For Velocity Verlet half step
"""
Updates the velocities of a [`System`](@ref) based on the constraints.
"""
function apply_velocity_constraints!(sys, constraint_algo::ConstraintAlgorithm,
     constraint_clusters::ConstraintCluster)

    for constraint_cluster in constraint_clusters
        apply_velocity_constraint!(sys, constraint_algo, constraint_cluster)
    end
end

apply_position_constraints!(sys::System, constraint_algo::NoSystemConstraints, args...) = nothing
apply_velocity_constraints!(sys::System, constraint_algo::NoSystemConstraints, args...) = nothing


