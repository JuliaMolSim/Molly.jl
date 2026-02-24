export
    DistanceConstraint,
    AngleConstraint,
    apply_position_constraints!,
    apply_velocity_constraints!,
    check_position_constraints,
    check_velocity_constraints,
    check_constraints

"""
    DistanceConstraint(i, j, dist)

Constraint between two atoms that maintains a fixed distance between the atoms.
"""
struct DistanceConstraint{D, I}
    i::I
    j::I
    dist::D
end

"""
    AngleConstraint(i, j, k, angle_ijk, dist_ij, dist_jk)

Constraint between three atoms that maintains a fixed angle and two bond lengths.

Atoms `i` and `k` should be connected to central atom `j` with fixed bond distances
given by `dist_ij` and `dist_jk`, forming the angle `angle_ijk` in radians.
Internally, an `AngleConstraint` is converted into 3 distance constraints.
None of the atoms in this constraint should be constrained with atoms not part
of this constraint.

For example, a water molecule can be defined as
`AngleConstraint(1, 2, 3, deg2rad(104.5), 0.9572u"Å", 0.9572u"Å")`
where atom 2 is oxygen and atoms 1/3 are hydrogen.

Linear molecules like CO2 can not be constrained.
"""
struct AngleConstraint{D, I}
    i::I
    j::I # Central atom, consistent with HarmonicAngle
    k::I
    dist_ij::D
    dist_jk::D
    dist_ik::D

    function AngleConstraint(i, j, k, angle_ijk, dist_ij, dist_jk)
        cos_θ = cos(angle_ijk)
        if cos_θ == -1
            throw(ArgumentError("disallowed linear angle constraint found between atoms $i/$j/$k"))
        end
        dist_ik = sqrt(dist_ij^2 + dist_jk^2 - 2*dist_ij*dist_jk*cos_θ)
        return new{typeof(dist_ij), typeof(i)}(i, j, k, dist_ij, dist_jk, dist_ik)
    end
end

to_distance_constraints(ac::AngleConstraint) = (
    DistanceConstraint(ac.i, ac.j, ac.dist_ij), # i-j bond
    DistanceConstraint(ac.j, ac.k, ac.dist_jk), # j-k bond
    DistanceConstraint(ac.i, ac.k, ac.dist_ik), # i-j-k angle
)

# This finds the unique atom indices given the lists (is and js)
# of atoms that participate in a constraint clusters.
# The first atom in the result is the atom at the center of the cluster
function order_atoms(is, js)
    counts = Dict{Int, Int}()
    for atom in is
        counts[atom] = get(counts, atom, 0) + 1
    end
    for atom in js
        counts[atom] = get(counts, atom, 0) + 1
    end

    central_atoms = [atom for (atom, cnt) in counts if cnt > 1]
    unique_atoms = collect(keys(counts))

    if length(central_atoms) == 1
        central = only(central_atoms)
        unique_atoms = [central; filter(x -> x != central, unique_atoms)]
        return unique_atoms
    elseif length(central_atoms) == 0 # Will trigger if just 1 bond in constraint (e.g. C-C)
        return unique_atoms
    else
        error("cannot find central atom, constraint chains of 4 atoms (e.g. C-C-C-C) " *
              "are not permitted")
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

    if is_single_bond || is_angle_cluster # No central atom, order agnostic
        return ConstraintKernelData(Int32.(raw_unique_idxs)..., raw_ds...)
    else # Order matters, need to place central atom first
        unique_idxs = order_atoms(raw_is, raw_js)
        central = unique_idxs[1]
        others  = unique_idxs[2:end]
        M = length(others)
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
@kernel function disable_pairs!(eligible, @Const(idx_i), @Const(idx_j))
    tid = @index(Global, Linear)
    if tid <= length(idx_i)
        i = idx_i[tid]
        j = idx_j[tid]
        @inbounds eligible[i, j] = false
        @inbounds eligible[j, i] = false
    end
end

function disable_constrained_interactions!(neighbor_finder, constraint_clusters)
    atom_interactions = cluster_interactions.(constraint_clusters)
    if isa(neighbor_finder.eligible, AbstractGPUArray)
        i_idx, j_idx = Int[], Int[]

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

        kernel = disable_pairs!(nf_backend, 1024)
        kernel(neighbor_finder.eligible, i_idx, j_idx, ndrange=length(i_idx))
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
function build_central_atom_clusters(num_atoms::Integer,
                                     dist_constraints::AbstractVector{<:DistanceConstraint{D}},
                                     strictness) where D
    # Store constraints as directed edges, direction is arbitrary but necessary
    constraint_graph = SimpleDiGraph(num_atoms)
    idx_dist_pairs = spzeros(D, num_atoms, num_atoms)

    for constraint in dist_constraints
        i, j = constraint.i, constraint.j
        edge_added = add_edge!(constraint_graph, i, j)
        if edge_added
            idx_dist_pairs[i, j] = constraint.dist
            idx_dist_pairs[j, i] = constraint.dist
        else
            report_issue(
                "Duplicate constraint in the system between atoms $i and $j, it will be ignored",
                strictness,
            )
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
            is, js = Int[], Int[]
            dists = []
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
                error("constraint clusters with more than 3 constraints or too few unique " *
                      "atoms are not unsupported, found $N_constraint constraints and " *
                      "$N_unique unique atoms")
            end
        end
    end

    return StructArray(clusters12), StructArray(clusters23), StructArray(clusters34)
end

function build_clusters(n_atoms::Integer, dist_constraints, angle_constraints, strictness)
    if isnothing(dist_constraints)
        dist_constraints_cp = []
    else
        dist_constraints_cp = copy(dist_constraints)
    end

    # Convert angle constraints to distance constraints
    # This is purely to check if they are connected to other
    # clusters in the DAG
    if !isnothing(angle_constraints)
        for ac in angle_constraints
            push!(dist_constraints_cp, to_distance_constraints(ac)...)
        end
    end

    # Check for interactions between clusters and build non-angle clusters
    clusters12, clusters23, clusters34 = build_central_atom_clusters(n_atoms,
                                                    [dist_constraints_cp...], strictness)

    # Now that we know angle_constraints do not interact with
    # any of the distance constraints we can build their clusters
    if !isnothing(angle_constraints) && length(angle_constraints) > 0
        clusters_angle = StructArray(AngleClusterData(ac) for ac in angle_constraints)
    else
        clusters_angle = NoClusterData[]
    end
    return clusters12, clusters23, clusters34, clusters_angle
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

Apply the coordinate constraints to the system.

If `vel_storage` and `dt` are provided then velocity constraints are applied as well.
"""
function apply_position_constraints!(sys, coord_storage; n_threads::Integer=Threads.nthreads())
    for ca in sys.constraints
        apply_position_constraints!(sys, ca, coord_storage; n_threads=n_threads)
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

Apply the velocity constraints to the system.
"""
function apply_velocity_constraints!(sys; n_threads::Integer=Threads.nthreads())
    for ca in sys.constraints
        apply_velocity_constraints!(sys, ca)
    end
    return sys
end

@kernel inbounds=true function max_dist_error(@Const(clusters),
                                              @Const(coords),
                                              boundary,
                                              maximums::AbstractVector{T}) where T
    cluster_idx = @index(Global, Linear)
    if cluster_idx <= length(clusters)
        cluster_max = typemin(T)
        for (i, j, dist) in cluster_interactions(clusters[cluster_idx])
            dr = vector(coords[i], coords[j], boundary)
            err = ustrip(abs(norm(dr) - dist))
            cluster_max = max(err, cluster_max)
        end
        maximums[cluster_idx] = cluster_max
    end
end

@kernel inbounds=true function max_vel_error(@Const(clusters),
                                             @Const(coords),
                                             @Const(velocities),
                                             boundary,
                                             maximums::AbstractVector{T}) where T
    cluster_idx = @index(Global, Linear)
    if cluster_idx <= length(clusters)
        cluster_max = typemin(T)
        for (i, j, _) in cluster_interactions(clusters[cluster_idx])
            dr = vector(coords[i], coords[j], boundary)
            v_diff = velocities[j] .- velocities[i]
            err = ustrip(abs(dot(dr, v_diff)))
            cluster_max = max(err, cluster_max)
        end
        maximums[cluster_idx] = cluster_max
    end
end

"""
    check_position_constraints(sys)
    check_position_constraints(sys, constraints)

Check whether the coordinates of a system satisfy the position constraints.
"""
check_position_constraints(sys) = all(ca -> check_position_constraints(sys, ca), sys.constraints)

function check_position_constraints(sys::System{<:Any, <:Any, FT}, ca) where FT
    err_unit = unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.dist_tolerance)
        throw(ArgumentError("distance tolerance units in SHAKE ($(unit(ca.dist_tolerance))) " *
                            "are inconsistent with system coordinate units ($err_unit)"))
    end

    cluster_maxes = FT[]
    backend = get_backend(sys.coords)
    err_kernel = max_dist_error(backend, 128)

    for cluster_type in cluster_keys(ca)
        clusters = getproperty(ca, cluster_type)
        if length(clusters) > 0
            max_storage = allocate(backend, FT, length(clusters))
            err_kernel(clusters, sys.coords, sys.boundary, max_storage; ndrange=length(clusters))
            push!(cluster_maxes, reduce(max, Array(max_storage)))
        end
    end

    KernelAbstractions.synchronize(backend)
    return maximum(cluster_maxes) < ustrip(ca.dist_tolerance)
end

"""
    check_velocity_constraints(sys)
    check_velocity_constraints(sys, constraints)

Check whether the velocities of a system satisfy the velocity constraints.
"""
check_velocity_constraints(sys) = all(ca -> check_velocity_constraints(sys, ca), sys.constraints)

function check_velocity_constraints(sys::System{<:Any, <:Any, FT}, ca) where FT
    err_unit = unit(eltype(eltype(sys.velocities))) * unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.vel_tolerance)
        throw(ArgumentError("velocity tolerance units in RATTLE ($(unit(ca.vel_tolerance))) " *
                    "are inconsistent with system velocity and coordinate units ($err_unit)"))
    end

    cluster_maxes = FT[]
    backend = get_backend(sys.coords)
    err_kernel = max_vel_error(backend, 128)

    for cluster_type in cluster_keys(ca)
        clusters = getproperty(ca, cluster_type)
        if length(clusters) > 0
            max_storage = allocate(backend, FT, length(clusters))
            err_kernel(clusters, sys.coords, sys.velocities, sys.boundary, max_storage;
                       ndrange=length(clusters))
            push!(cluster_maxes, reduce(max, Array(max_storage)))
        end
    end

    KernelAbstractions.synchronize(backend)
    return maximum(cluster_maxes) < ustrip(ca.vel_tolerance)
end

"""
    check_constraints(sys)
    check_constraints(sys, constraints)

Check whether the coordinates and velocities of a system satisfy the
coordinate and velocity constraints.
"""
check_constraints(sys) = all(ca -> check_constraints(sys, ca), sys.constraints)

function check_constraints(sys, ca)
    return check_position_constraints(sys, ca) && check_velocity_constraints(sys, ca)
end

# No-op when backends are same
to_backend(arr, old::T, new::T) where {T <: Backend} = arr

# Allocates and copies when backends are different
function to_backend(arr, old::A, new::B) where {A <: Backend, B <: Backend}
    out = allocate(new, eltype(arr), size(arr))
    copy!(out, arr)
    return out
end

# These types will enable coalesced memory access via StructArray{ConstraintKernelData}
abstract type ConstraintKernelData{D, N, M} end

n_constraints(::ConstraintKernelData{<:Any, N}) where {N} = N
n_atoms_cluster(::ConstraintKernelData{<:Any, <:Any, M}) where {M} = M

struct NoClusterData <: ConstraintKernelData{Nothing, 0, 0} end

# This is effectivelly just a distance constraint
struct Cluster12Data{D} <: ConstraintKernelData{D, 1, 2}
    k1::Int32
    k2::Int32
    dist12::D
end

function ConstraintKernelData(k1::Int32, k2::Int32, dist12::D) where D
    return Cluster12Data{D}(k1, k2, dist12)
end

cluster_interactions(kd::Cluster12Data) = ((kd.k1, kd.k2, kd.dist12),)

struct Cluster23Data{D} <: ConstraintKernelData{D, 2, 3}
    k1::Int32
    k2::Int32
    k3::Int32
    dist12::D
    dist13::D
end

function ConstraintKernelData(k1::Int32, k2::Int32, k3::Int32, dist12::D, dist13::D) where D
    return Cluster23Data{D}(k1, k2, k3, dist12, dist13)
end

cluster_interactions(kd::Cluster23Data) = (
    (kd.k1, kd.k2, kd.dist12),
    (kd.k1, kd.k3, kd.dist13),
)
idx_keys(::Type{<:Cluster23Data}) = (:k1, :k2, :k3)
dist_keys(::Type{<:Cluster23Data}) = (:dist12, :dist13)

struct Cluster34Data{D} <: ConstraintKernelData{D, 3, 4}
    k1::Int32
    k2::Int32
    k3::Int32
    k4::Int32
    dist12::D
    dist13::D
    dist14::D
end

function ConstraintKernelData(k1::Int32, k2::Int32, k3::Int32, k4::Int32, dist12::D,
                              dist13::D, dist14::D) where D
    return Cluster34Data{D}(k1, k2, k3, k4, dist12, dist13, dist14)
end

cluster_interactions(kd::Cluster34Data) = (
    (kd.k1, kd.k2, kd.dist12),
    (kd.k1, kd.k3, kd.dist13),
    (kd.k1, kd.k4, kd.dist14),
)
idx_keys(::Type{<:Cluster34Data}) = (:k1, :k2, :k3, :k4)
dist_keys(::Type{<:Cluster34Data}) = (:dist12, :dist13, :dist14)

struct AngleClusterData{D} <: ConstraintKernelData{D, 3, 3}
    k1::Int32 # Central atom, different to AngleConstraint
    k2::Int32
    k3::Int32
    dist12::D
    dist13::D
    dist23::D
end

function AngleClusterData(ac::AngleConstraint)
    # Note switch of central atom from j (consistent with HarmonicAngle)
    # to i (consistent with other constraint clusters)
    return AngleClusterData(Int32(ac.j), Int32(ac.i), Int32(ac.k),
                            ac.dist_ij, ac.dist_jk, ac.dist_ik)
end

function ConstraintKernelData(k1::Int32, k2::Int32, k3::Int32, dist12::D,
                              dist13::D, dist23::D) where D
    return AngleClusterData{D}(k1, k2, k3, dist12, dist13, dist23)
end

cluster_interactions(kd::AngleClusterData) = (
    (kd.k1, kd.k2, kd.dist12),
    (kd.k1, kd.k3, kd.dist13),
    (kd.k2, kd.k3, kd.dist23),
)
idx_keys(::Type{<:AngleClusterData}) = (:k1, :k2, :k3)
dist_keys(::Type{<:AngleClusterData}) = (:dist12, :dist13, :dist23)
