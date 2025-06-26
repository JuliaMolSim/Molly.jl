export
    DistanceConstraint,
    AngleConstraint,
    disable_constrained_interactions!,
    apply_position_constraints!,
    apply_velocity_constraints!

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
    AngleConstraint(i, j, k, angle_jk, dist_ij, dist_ik)

Constraint between three atoms that maintains a fixed angle, angle_jk, and two bond lengths. 
Atoms j and k must be connected to a shared central atom with fixed bond distances given by
dist_ij and dist_ik. Internally, an `AngleConstraint` is converted to 3 distance constraints. None
of the atoms in this constraint are allowed to be constrained with another atom not
included in this constraint. Angle can be passed as Unitful deg/rad, or as a plain
number in radians. You cannot constrain linear molecules like CO2.

# Arguments
- `i`, `j`, `k`: The indices of the atoms in the constraint.
- `angle_jk`: The angle between atoms j and k in radians if not Unitful.
- `dist_ij`: The distance between atoms i and j.
- `dist_ik`: The distance between atoms i and k.

For example, a water molecule can be defined as:
```julia
ac = AngleConstraint(1, 2, 3, 104.5u"°", 0.9572u"Å", 0.9572u"Å")
ac = AngleConstraint(1, 2, 3, 104.5 * π / 180, 0.9572u"Å", 0.9572u"Å")
```
where atom 1 is oxygen and atoms 2/3 are hydrogen.
"""
struct AngleConstraint{D}
    i::Int
    j::Int
    k::Int
    dist_ij::D
    dist_ik::D
    dist_jk::D
end

function AngleConstraint(i, j, k, angle_jk, dist_ij, dist_ik)

    cosθ = cos(angle_jk)

    if cosθ == -1.0
        error(ArgumentError("Cannot constrain linear molecules."))
    end

    dist_jk = sqrt((dist_ij*dist_ij) + (dist_ik*dist_ik) -
                    (2 * dist_ij * dist_ik * cosθ))

    return AngleConstraint{typeof(dist_ij)}(i, j, k, dist_ij, dist_ik, dist_jk)
end

struct ConstraintCluster{D <: DistanceConstraint}
    constraints::StructArray{D}
    unique_atoms::Vector{Int}
end

n_unique(cc::ConstraintCluster) = length(cc.unique_atoms)

"""
    disable_constrained_interactions!(neighbor_finder, constraint_clusters)

Disables neighbor list interactions between atoms in a constraint.
"""
function disable_constrained_interactions!(
        neighbor_finder,
        constraint_clusters::AbstractVector{ConstraintCluster}
    )

    for cluster in constraint_clusters
        for constraint in cluster.constraints
            neighbor_finder.eligible[constraint.i, constraint.j] = false
            neighbor_finder.eligible[constraint.j, constraint.i] = false
        end
    end
    return neighbor_finder
end

# This finds the unique atom indicies given a the lists (is and js)
# of atoms that participate in a constraint clusters.
# The first atom in the result is the atom at the center of the cluster
function order_atoms(is, js)

    counts = Dict{Int,Int}()
    for atom in is
        counts[atom] = get(counts, atom, 0) + 1
    end

    for atom in js
        counts[atom] = get(counts, atom, 0) + 1
    end

    central_atoms = [atom for (atom, cnt) in counts if cnt > 1]
    unique_atoms = collect(keys(counts))

    if length(central_atoms) == 1
        central = central_atoms[1]
        unique_atoms = [central; filter(x -> x != central, unique_atoms)]
        return unique_atoms
    elseif length(central_atoms) == 0 # Will trigger if just 1 bond in constraint (e.g. C-C)
        return unique_atoms
    else
        @error "Cannot find central atom. You are not permitted to constraint chains of 4 atoms (e.g., C-C-C-C)"
    end
end

# This makes sure the first atom in each distance constraint
# is the central atom of the cluster. It also calls `order_atoms`
# which ensures that the first atom in the cluster is the central atom
function make_cluster(raw_is, raw_js, raw_ds)

    # For bond constraints none of this matters
    if length(raw_is) == 2
        cluster = StructArray{DistanceConstraint}((
                        MVector(raw_is...),
                        MVector(raw_js...),
                        MVector(raw_ds...)
                        ))
        return ConstraintCluster(cluster, unique([raw_is; raw_js]))
    end

    # find unique & central
    unique_idxs = order_atoms(raw_is, raw_js)
    central = unique_idxs[1]
    others  = unique_idxs[2:end]
    M       = length(others)

    # build sorted  DistanceConstraints
    is = fill(central, M) # ALL first atoms are the central atom
    js = others
    dists = Vector{eltype(raw_ds)}(undef, M)

    # Figure out which distance goes with new pairings
    for (a, o) in enumerate(others)
        for (idx, (i0, j0)) in enumerate(zip(raw_is, raw_js))
            if (i0 == central && j0 == o) || (j0 == central && i0 == o)
                dists[a] = raw_ds[idx]
                break
            end
        end
    end

    #* NEEds to be MVector??? dont remember why I did that
    cluster = StructArray{DistanceConstraint}((MVector(is...), MVector(js...), MVector(dists...)))
    return ConstraintCluster(cluster, unique_idxs)
end

function build_clusters(n_atoms, constraints)
    constraint_graph = SimpleDiGraph(n_atoms)
    idx_dist_pairs = spzeros(n_atoms, n_atoms) * unit(constraints[1].dist)

    # Store constraints as directed edges, direction is arbitrary but necessary
    for constraint in constraints
        #TODO - Add support for angle constraints
        if constraint isa AngleConstraint
            error(ArgumentError("Cannot constrain linear molecules."))
        end

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
    
    clusters12 = ConstraintCluster[]; clusters23 = ConstraintCluster[]
    clusters34 = ConstraintCluster[]; clusters_angle = ConstraintCluster[]
    # Loop through connected regions and convert to clusters
    for (_, atom_idxs) in enumerate(cc)
        # Loop over atoms in connected region to build cluster
        if length(atom_idxs) > 1 # connected_components gives unconnected vertices as well
            is = Int[]; js = Int[]; dists = []
            for ai in atom_idxs
                neigh_idxs = neighbors(constraint_graph, ai)
                for neigh_idx in neigh_idxs
                    push!(is, ai)
                    push!(js, neigh_idx)
                    push!(dists, idx_dist_pairs[ai, neigh_idx])
                end
            end

            #* WILL NEED TO UPDATE THIS FOR ANGLE CONSTRAINTS!
            #* ALL ATOMS WILL TRIGGER AS "CENTRAL" ATOMS 
            constraint_cluster = make_cluster(is, js, [dists...])
            N_constraint = length(is)
            N_unique = n_unique(constraint_cluster)

            if N_constraint == 1 && N_unique == 2 # Single bond constraint between two atoms
                push!(clusters12, constraint_cluster)
            elseif N_constraint == 2 && N_unique == 3 # Central atom with 2 bonds constrained
                push!(clusters23, constraint_cluster)
            elseif N_constraint == 3 && N_unique == 4 # Central atom 3 bonds constrained
                push!(clusters34, constraint_cluster)
            elseif N_constraint == 3 && N_unique == 3 # 3 atoms, with 2 bonds + 1 angle constraint
                error("Angle constraints not supported yet")
                push!(clusters_angle, constraint_cluster)
            else
                @error "Constraint clusters with more than 3 constraints or too few unique atoms are not unsupported. Got $(N_constraint) constraints and $(N_unique) unique atoms."
            end
        end
    end
    return clusters12, clusters23, clusters34, clusters_angle
    # return [clusters12...], [clusters23...], [clusters34...], [clusters_angle...]
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
function n_dof_lost(D::Integer, constraint_clusters::AbstractVector{ConstraintCluster})
    # Bond constraints remove vibrational DoFs
    vibrational_dof_lost = 0
    # Assumes constraints are a non-linear chain (e.g., breaks for angle constraints)
    for cluster in constraint_clusters
        N = n_unique(cluster)
        # If N > 2 assume non-linear (e.g. breaks for CO2)
        vibrational_dof_lost += ((N == 2) ? D*N - (2*D - 1) : D*(N - 2))
    end
    return vibrational_dof_lost
end

function n_dof(D::Integer, n_atoms::Integer, boundary)
    return D * n_atoms - (D - n_infinite_dims(boundary))
end


"""
    apply_position_constraints!(sys, coord_storage)
    apply_position_constraints!(sys, coord_storage, vel_storage, dt)

Applies the system constraints to the coordinates.

If `vel_storage` and `dt` are provided then velocity corrections are applied as well.
"""
function apply_position_constraints!(sys, coord_storage; n_threads::Integer=Threads.nthreads())
    for ca in sys.constraints
        apply_position_constraints!(sys, ca, coord_storage; n_threads = n_threads)
    end
    return sys
end

function apply_position_constraints!(sys, coord_storage, vel_storage, dt;
                                        n_threads::Integer=Threads.nthreads())

    if length(sys.constraints) > 0

        vel_storage .= -sys.coords ./ dt

        for ca in sys.constraints
            apply_position_constraints!(sys, ca, coord_storage; n_threads = n_threads)
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
function apply_velocity_constraints!(sys; n_threads::Integer=Threads.nthreads())
    for ca in sys.constraints
        apply_velocity_constraints!(sys, ca)
    end
    return sys
end