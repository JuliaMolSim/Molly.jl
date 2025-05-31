export
    DistanceConstraint,
    disable_constrained_interactions!,
    apply_position_constraints!,
    apply_velocity_constraints!


struct NoConstraints end

"""
    DistanceConstraint(i, j, dist)

Constraint between two atoms that maintains a fixed distance between the two atoms.
"""
struct DistanceConstraint{D}
    i::Int
    j::Int
    dist::D
end


"""
    disable_constrained_interactions!(neighbor_finder, constraint_clusters)

Disables neighbor list interactions between atoms in a constraint.
"""
function disable_constrained_interactions!(neighbor_finder, constraint_clusters)
    for cluster in constraint_clusters
        for constraint in cluster
            neighbor_finder.eligible[constraint.i, constraint.j] = false
            neighbor_finder.eligible[constraint.j, constraint.i] = false
        end
    end
    return neighbor_finder
end

function build_clusters(n_atoms, constraints)
    constraint_graph = SimpleDiGraph(n_atoms)
    idx_dist_pairs = spzeros(n_atoms, n_atoms) * unit(constraints[1].dist)

    # Store constraints as directed edges, direction is arbitrary but necessary
    for constraint in constraints
        edge_added = add_edge!(constraint_graph, constraint.i, constraint.j)
        if edge_added
            idx_dist_pairs[constraint.i, constraint.j] = constraint.dist
            idx_dist_pairs[constraint.j, constraint.i] = constraint.dist
        else
            @warn "Duplicate constraint in the system, it will be ignored"
        end
    end

    # Get groups of constraints that are connected to eachother
    cc = connected_components(constraint_graph)

    # Loop through connected regions and convert to clusters
    clusters1 = []; clusters2 = []; clusters3 = []
    for (_, atom_idxs) in enumerate(cc)
        # Loop over atoms in connected region to build cluster
        if length(atom_idxs) > 1 # connected_components gives unconnected vertices as well
            # connected_constraints = []
            is = []; js = []; dists = []
            for ai in atom_idxs
                neigh_idxs = neighbors(constraint_graph, ai)
                for neigh_idx in neigh_idxs
                    push!(is, ai)
                    push!(js, neigh_idx)
                    push!(dists, idx_dist_pairs[ai, neigh_idx])
                    # constraint = DistanceConstraint(ai, neigh_idx, idx_dist_pairs[ai, neigh_idx])
                    # push!(connected_constraints, constraint)
                end
            end

            cluster = StructArray{DistanceConstraint}((MVector(is...), MVector(js...), MVector(dists...)))
            N = length(clsuter)
            # cluster = SVector(connected_constraints...)
            if N == 1
                push!(clusters1, cluster)
            elseif N == 2
                push!(clusters2, cluster)
            elseif N == 3
                push!(clusters3, cluster)
            else
                @warn "Constraint clusters with more than 3 constraints are unsupported, skipping."
            end
        end
    end

    return [clusters1...], [clusters2...], [clusters3...]
end


"""
    apply_position_constraints!(sys, coord_storage)
    apply_position_constraints!(sys, coord_storage, vel_storage, dt)

Applies the system constraints to the coordinates.

If `vel_storage` and `dt` are provided then velocity corrections are applied as well.
"""
function apply_position_constraints!(sys, coord_storage)
    for ca in sys.constraints
        apply_position_constraints!(sys, ca, coord_storage)
    end
    return sys
end

function apply_position_constraints!(sys, coord_storage, vel_storage, dt)

    if length(sys.constraints) > 0

        vel_storage .= -sys.coords ./ dt

        for ca in sys.constraints
            apply_position_constraints!(sys, ca, coord_storage)
        end

        vel_storage .+= sys.coords ./ dt
        sys.velocities .+= vel_storage
    end

    return sys
end

"""
    apply_velocity_constraints!(sys)

Applies the system constraints to the velocities.
"""
function apply_velocity_constraints!(sys)
    for ca in sys.constraints
        apply_velocity_constraints!(sys, ca)
    end
    return sys
end

#=
    n_dof_lost(D::Integer, constraint_clusters)

Calculate the number of degrees of freedom lost from the system due to the constraints.

All constrained molecules with 3 or more atoms are assumed to be non-linear.
The table below shows the degrees of freedom for different types of structures in the
system where D is the dimensionality.
When using constraint algorithms the vibrational degrees of freedom are removed from a molecule.

| DoF           | Monoatomic | Linear Molecule | Non-Linear Molecule |
| ------------- | ---------- | --------------- | ------------------- |
| Translational |     D      |       D         |        D            |
| Rotational    |     0      |     D - 1       |        D            |
| Vibrational   |     0      |  D*N - (2D - 1) |    D*N - 2D         |
| Total         |     D      |      D*N        |       D*N           |

=#
function n_dof_lost(D::Integer, constraint_clusters)
    # Bond constraints remove vibrational DoFs
    vibrational_dof_lost = 0
    # Assumes constraints are a non-linear chain (e.g., breaks for angle constraints)
    for cluster in constraint_clusters
        N = cluster.n_unique_atoms
        # If N > 2 assume non-linear (e.g. breaks for CO2)
        vibrational_dof_lost += ((N == 2) ? D*N - (2*D - 1) : D*(N - 2))
    end
    return vibrational_dof_lost
end

function n_dof(D::Integer, n_atoms::Integer, boundary)
    return D * n_atoms - (D - n_infinite_dims(boundary))
end
