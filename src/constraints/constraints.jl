export
    DistanceConstraint,
    AngleConstraint,
    disable_constrained_interactions!,
    apply_position_constraints!,
    apply_velocity_constraints!,
    check_position_constraints,
    check_velocity_constraints


abstract type Constraint{D} end

"""
    DistanceConstraint(i, j, dist)


    DistanceConstraint(i, j, dist)

Constraint between two atoms that maintains a fixed distance between the two atoms.
"""
struct DistanceConstraint{D, I <: Integer} <: Constraint{D}
    i::I
    j::I
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
struct AngleConstraint{D, I <: Integer} <: Constraint{D}
    i::I # Central atom
    j::I
    k::I
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

    return AngleConstraint{typeof(dist_ij), typeof(i)}(i, j, k, dist_ij, dist_ik, dist_jk)
end

to_distance_constraints(ac::AngleConstraint) = (
    DistanceConstraint(ac.i, ac.j, ac.dist_ij), # arm 1
    DistanceConstraint(ac.i, ac.k, ac.dist_ik), # arm 2
    DistanceConstraint(ac.j, ac.k, ac.dist_jk) # angle
) 


function to_cluster_data(ac::AngleConstraint)
    return AngleClusterData(Int32(ac.i), Int32(ac.j), Int32(ac.k), 
                            ac.dist_ij, ac.dist_ik, ac.dist_jk)
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
function make_cluster_data(raw_is, raw_js, raw_ds)

    raw_unique_idxs = unique([raw_is; raw_js])
    N_unique_atoms = length(raw_unique_idxs)
    N_constraints = length(raw_is)

    is_single_bond = (N_constraints == 1)
    is_angle_cluster = (N_constraints == 3 && N_unique_atoms == 3)

    if is_single_bond || is_angle_cluster # no central atom, order agnostic
        return ConstraintKernelData(Int32.(raw_unique_idxs)..., raw_ds...)
    else # order matters, need to place central atom first
        unique_idxs = order_atoms(raw_is, raw_js)
        central = unique_idxs[1]
        others  = unique_idxs[2:end]
        M       = length(others)

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

        return ConstraintKernelData(Int32.(unique_idxs)..., dists...)
    end
end


# Kernel to support when neighbor finder is on GPU
@kernel function disable_pairs!(eligible, idx_i, idx_j)
    tid = @index(Global, Linear)
    if tid <= length(idx_i)
        i = idx_i[tid]
        j = idx_j[tid]
        @inbounds eligible[i, j] = false
        @inbounds eligible[j, i] = false
    end
end


function disable_constrained_interactions!(
        neighbor_finder,
        constraint_clusters::AbstractVector
    )


    atom_interactions = interactions.(constraint_clusters)

    if isa(neighbor_finder.eligible, AbstractGPUArray)

        # Collect pairs first
        i_idx = Int[]
        j_idx = Int[]

        for interaction_list in atom_interactions
            for (i, j, _) in interaction_list
                push!(i_idx, i)
                push!(j_idx, j)
            end
        end

        nf_backend = get_backend(neighbor_finder.eligible)
        is_backend = get_backend(i_idx)

        i_idx = to_backend(i_idx, is_backend, nf_backend)
        j_idx = to_backend(j_idx, is_backend, nf_backend)

        # 1024 is block size
        kernel = disable_pairs!(nf_backend, 1024)
        kernel(neighbor_finder.eligible, i_idx, j_idx, ndrange = length(i_idx))

        return neighbor_finder

    else # KernelAbstractions does not like the BitMatrix used by eligible on CPU

        for interaction_list in atom_interactions
            for (i, j, _) in interaction_list
                neighbor_finder.eligible[i, j] = false
                neighbor_finder.eligible[j, i] = false
            end
        end

        return neighbor_finder
    end

end

# Check for interactions between angle and non-angle clusters,
# builds only non-angle clusters
function build_central_atom_clusters(
        num_atoms::Integer,
        dist_constraints::AbstractVector{<:DistanceConstraint{D}}
    ) where D

    # Store constraints as directed edges, direction is arbitrary but necessary
    constraint_graph = SimpleDiGraph(num_atoms)
    idx_dist_pairs = spzeros(float_type(D), num_atoms, num_atoms) * unit(D)
    
    for constraint in dist_constraints
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
    
    clusters12 = Cluster12Data{D}[]
    clusters23 = Cluster23Data{D}[]
    clusters34 = Cluster34Data{D}[]

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

            cluster_data = make_cluster_data(is, js, [dists...])
            N_constraint = n_constraints(cluster_data)
            N_unique = n_atoms_cluster(cluster_data)

            if N_constraint == 1 && N_unique == 2 # Single bond constraint between two atoms
                push!(clusters12, cluster_data)
            elseif N_constraint == 2 && N_unique == 3 # Central atom with 2 bonds constrained
                push!(clusters23, cluster_data)
            elseif N_constraint == 3 && N_unique == 4 # Central atom 3 bonds constrained
                push!(clusters34, cluster_data)
            elseif N_constraint == 3 && N_unique == 3 # angle constraint
                # Skip angle constraints, we will build them later if needed
                continue
            else
                @error "Constraint clusters with more than 3 constraints or too few unique atoms are not unsupported. Got $(N_constraint) constraints and $(N_unique) unique atoms."
            end
        end
    end

    return StructArray(clusters12), StructArray(clusters23), StructArray(clusters34)
end

function build_clusters(
        n_atoms::Integer,
        dist_constraints::AbstractVector{DistanceConstraint{D}},
        angle_constraints::AbstractVector{AngleConstraint{D}}
    ) where D

    #! avoid mutating input, dont love this...
    dist_constraints = copy(dist_constraints)

    # Convert angle constraints to distance constraints.
    # This is purely to check if they are connected to other
    # clusters in the DAG
    for ac in angle_constraints
        push!(dist_constraints, to_distance_constraints(ac)...)
    end

    # Check for interactions between clusters and build non-angle clusters
    clusters12, clusters23, clusters34 = 
        build_central_atom_clusters(n_atoms, dist_constraints)

    # Now that we know angle_constraints do not interact with
    # any of the distance constraints we can build their clusters
    clusters_angle = ConstraintCluster{3,3}[]
    for ac in angle_constraints
        push!(clusters_angle, to_cluster(ac))
    end

    return clusters12, clusters23, clusters34, [clusters_angle...]
end

function build_clusters(
        n_atoms::Integer,
        dist_constraints::Nothing,
        angle_constraints::AbstractVector{<:AngleConstraint{D}}
    ) where D
    acc = StructArray(to_cluster_data(ac) for ac in angle_constraints)
    #* These are non-concrete...what should I do instead??
    return NoClusterData[], NoClusterData[], NoClusterData[], acc
end 

function build_clusters(
        n_atoms::Integer,
        dist_constraints::AbstractVector{<:DistanceConstraint{D}}, 
        angle_constraints::Nothing
    ) where D

    clusters12, clusters23, clusters34 = 
        build_central_atom_clusters(n_atoms, dist_constraints)

    return clusters12, clusters23, clusters34, NoClusterData[]

end

# Fallback
function build_clusters(
    n_atoms::Integer,
    dist_constraints::Nothing,
    angle_constraints::Nothing
)
    return NoClusterData[], NoClusterData[], NoClusterData[], NoClusterData[]
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
function n_dof_lost(D::Integer, constraint_clusters::AbstractVector)
    # Bond constraints remove vibrational DoFs
    vibrational_dof_lost = 0
    # Assumes constraints are a non-linear chain
    for cluster in constraint_clusters
        N = n_atoms_cluster(cluster)
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


"""
    check_position_constraints(sys, constraints)

Checks if the position constraints are satisfied by the current coordinates of `sys`.
"""
function check_position_constraints(sys, ca)
    max_err = typemin(float_type(sys)) * unit(eltype(eltype(sys.coords)))
    for cluster_type in cluster_keys(ca)
        clusters = getproperty(ca, cluster_type)
        for cluster in clusters
            for (i, j, dist) in interactions(cluster)
                dr = vector(sys.coords[i], sys.coords[j], sys.boundary)
                err = abs(norm(dr) - dist)
                if max_err < err
                    max_err = err
                end
            end
        end
    end
    return max_err < ca.dist_tolerance
end

"""
    check_velocity_constraints(sys, constraints)

Checks if the velocity constraints are satisfied by the current velocities of `sys`.
"""
function check_velocity_constraints(sys::System, ca)
    max_err = typemin(float_type(sys)) * unit(eltype(eltype(sys.velocities))) * unit(eltype(eltype(sys.coords)))
    for cluster_type in cluster_keys(ca)
        clusters = getproperty(ca, cluster_type)
        for cluster in clusters
            for (i, j, _) in interactions(cluster)
                dr = vector(sys.coords[i], sys.coords[j], sys.boundary)
                v_diff = sys.velocities[j] .- sys.velocities[i]
                err = abs(dot(dr, v_diff))
                if max_err < err
                    max_err = err
                end
            end
        end
    end
    return max_err < ca.vel_tolerance
end