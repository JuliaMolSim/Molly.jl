export
    DistanceConstraint,
    check_position_constraints,
    check_velocity_constraints

"""
Supertype for all constraint algorithms.

Should implement the following methods:
position_constraints!(sys::System, constraint_algo::ConstraintAlgorithm, accels, dt; n_threads::Integer=Threads.nthreads())
velocity_constraints!(sys::System, constraint_algo::ConstraintAlgorithm; n_threads::Integer=Threads.nthreads())
"""
abstract type ConstraintAlgorithm end


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
    atom_idxs::SVector{2,<:Integer}
    dist::D
end

# """
# Constraint between three atoms that maintains a common angle (e.g. a water molecule).
# 180° angles are not supported as they cause issues with RATTLE/SHAKE 
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
    constraints::SVector{N,<:Constraint}
    n_unique_atoms::Integer
end

function ConstraintCluster(constraints)

    #Count # of unique atoms in cluster
    atom_ids = []
    for constraint in constraints
        for atom_idx in constraint.atom_idxs
            push!(atom_ids, atom_idx)
        end
    end

    #Not sure this is always true? not true for rings
    # if length(unique(atom_ids)) < length(constraints)
    #     throw(ArgumentError("System is overconstrained. There are more constraints than unique atoms\
    #     in cluster with atoms: $(atom_ids)."))
    # end

    return ConstraintCluster{length(constraints)}(constraints, length(unique(atom_ids)))

end

Base.length(cc::ConstraintCluster) = length(cc.constraints)


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

##### Constraint Setup ######

"""
Disables interactions between atoms in a constraint. This prevents forces
from blowing up as atoms in a bond are typically very close to eachother.
"""
function disable_intra_constraint_interactions!(neighbor_finder,
     constraint_clsuters::AbstractVector{ConstraintCluster})

    # Loop through constraints and modify eligible matrix
    for cluster in constraint_clsuters
        for constraint in cluster.constraints
            neighbor_finder.eligible[constraint.atom_idxs[1], constraint.atom_idxs[2]] = false
            neighbor_finder.eligible[constraint.atom_idxs[2], constraint.atom_idxs[1]] = false
        end
    end

    return neighbor_finder
end


"""
Parse the constraints into clusters. 
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
    cc = connected_components(constraint_graph) #& this can return non-connected things sometimes??
    # Initialze empty vector of clusters
    clusters = Vector{ConstraintCluster}(undef, 0)

    #Loop through connected regions and convert to clusters
    for (cluster_idx, atom_idxs) in enumerate(cc)
        # Loop atoms in connected region to build cluster
        if length(atom_idxs) > 1 #connected_components gives unconnected verticies as well
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
    end

    return clusters
end


##### High Level Constraint Functions ######
function save_ca_positions!(sys::System, c)
    if length(sys.constraint_algorithms) > 0
        sys.hidden_storage.ca_coord_storage .= c
    end
end

function apply_position_constraints!(sys::System; n_threads::Integer=Threads.nthreads())

   for ca in sys.constraint_algorithms
       position_constraints!(sys, ca, n_threads = n_threads)
   end

   return sys

end

function apply_position_constraints!(sys::System, apply_vel_correction::Val{true}, dt;
     n_threads::Integer=Threads.nthreads())

     if length(sys.constraint_algorithms) > 0
        sys.hidden_storage.ca_vel_storage .= -sys.coords ./ dt

        for ca in sys.constraint_algorithms
            position_constraints!(sys, ca, n_threads = n_threads)
        end

        sys.hidden_storage.ca_vel_storage .+= sys.coords ./ dt

        sys.velocities .+= sys.hidden_storage.ca_vel_storage
    end

   return sys
end


function apply_velocity_constraints!(sys::System; n_threads::Integer=Threads.nthreads())
    
    for ca in sys.constraint_algorithms
        velocity_constraints!(sys, ca, n_threads = n_threads)
    end

    return sys
end

# """
# Verifies that the initial conditions of the system satisfy the constraints.
# """
# function check_initial_constraints(coords, clusters, init_tol)

#     if init_tol !== nothing
#         for cluster in clusters
#             for constraint in cluster.constraints
#                 r_actual = norm(coords[constraint.atom_idxs[1]] .- coords[constraint.atom_idxs[2]])
#                 if !isapprox(r_actual, constraint.dist, atol = init_tol)
#                     throw(ArgumentError("Constraint between atoms $(constraint.atom_idxs[1]) 
#                         and $(constraint.atom_idxs[2]) is not satisfied by the initial conditions."))
#                 end
#             end
#         end
#     end

# end

function check_position_constraints(sys::System, ca::ConstraintAlgorithm)

    max_err = typemin(float_type(sys))*unit(sys.coords[1][1])
    for cluster in ca.clusters
        for constraint in cluster.constraints
            k1, k2 = constraint.atom_idxs
            err = abs(norm(vector(sys.coords[k2], sys.coords[k1], sys.boundary)) - constraint.dist)
            if max_err < err
                max_err = err
            end
        end
    end

    return (max_err < ca.dist_tolerance)
end

function check_velocity_constraints(sys::System, ca::ConstraintAlgorithm)

    max_err = typemin(float_type(sys))*unit(sys.velocities[1][1])*unit(sys.coords[1][1])
    for cluster in ca.clusters
        for constraint in cluster.constraints
            k1, k2 = constraint.atom_idxs
            err = abs(dot(vector(sys.coords[k2], sys.coords[k1], sys.boundary), (sys.velocities[k2] .- sys.velocities[k1])))
            if max_err < err
                max_err = err
            end
        end
    end

    return (max_err < ca.vel_tolerance)
end



"""
Re-calculates the # of degrees of freedom in the system due to the constraints.
All constrained molecules with 3 or more atoms are assumed to be non-linear because
180° bond angles are not supported. The table below shows the break down of 
DoF for different types of structures in the system where D is the dimensionality.
When using constraint algorithms the vibrational DoFs are removed from a molecule.

DoF           | Monoatomic | Linear Molecule | Non-Linear Molecule |
Translational |     D      |       D         |        D            |
Rotational    |     0      |     D - 1       |        D            |
Vibrational   |     0      |  D*N - (2D - 1) |    D*N - 2D         |
Total         |     D      |      D*N        |       D*N           |

"""
#& Im not sure this is generic
function n_dof_lost(D::Int, constraint_clusters)

    # Bond constraints remove vibrational DoFs
    vibrational_dof_lost = 0
    #Assumes constraints are a non-linear chain
    for cluster in constraint_clusters
        N = cluster.n_unique_atoms
        # If N > 2 assume non-linear (e.g. breaks for CO2)
        vibrational_dof_lost += ((N == 2) ? D*N - (2*D - 1) : D*(N - 2))
    end

    return vibrational_dof_lost

end

function n_dof(D::Int, N_atoms::Int, boundary)
    return D*N_atoms - (D - (num_infinte_boundary(boundary)))
end
# function n_dof(D::Int, N_atoms::Int, boundary, constraint_clusters)

#     # Momentum only conserved in directions with PBC
#     dof = 3*N_atoms

#     #Assumes constraints are a non-linear chain
#     for cluster in constraint_clusters
#         dof -= length(cluster.constraints)
#     end

#     return dof - (D - (num_infinte_boundary(boundary)))

# end


