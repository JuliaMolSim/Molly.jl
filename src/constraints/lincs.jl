export
    LINCS,
    SetupLINCS

# Internal types for LINCS algorithm

struct LincsCouplingMatrix{R, N, C}
    range::R       # length K+1, row pointers (CSR format)
    neighbors::N   # coupled constraint indices
    coef::C        # mass-weighted coupling coefficients
end

struct LincsCouplingDense{CI, CC, NC}
    coupled_indices::CI   # Int[max_coupled, K_padded]
    coupled_coef::CC      # T[max_coupled, K_padded]
    n_coupled::NC         # Int[K_padded]
    max_coupled::Int
end

#=
Maps each atom to the constraints it takes part in, as a CSR where the constraint index
is signed by which end of the constraint the atom is: `+i` when the atom is `atom2[i]`
and `-i` when it is `atom1[i]`.
With this mapping the update is gathered per atom instead of per constraint, which is
conflict free. It is only built on CPU, the GPU path uses the atoms of each workgroup in
`LincsWorkgroupAtoms` instead.
=#
struct LincsAtomScatter{R, C}
    range::R
    constraints::C
end

function build_lincs_atom_scatter(atom1, atom2, n_atoms::Integer)
    K = length(atom1)
    range = zeros(Int32, n_atoms + 1)
    for i in 1:K
        range[atom1[i] + 1] += Int32(1)
        range[atom2[i] + 1] += Int32(1)
    end
    range[1] = Int32(1)
    for a in 1:n_atoms
        range[a + 1] += range[a]
    end
    cursor = copy(range)
    constraints = Vector{Int32}(undef, 2 * K)
    for i in 1:K
        a1, a2 = atom1[i], atom2[i]
        constraints[cursor[a1]] = Int32(-i)
        cursor[a1] += Int32(1)
        constraints[cursor[a2]] = Int32(i)
        cursor[a2] += Int32(1)
    end
    return LincsAtomScatter(range, constraints)
end

#=
The atoms of each workgroup of the GPU kernels that keep the coordinates or velocities in
shared memory, see `lincs_shared_position_kernel!`.
Every constraint that moves an atom is in the same workgroup, so the workgroup can keep the
coordinates or velocities of its atoms in shared memory between the iterations, and gather
the change to each atom from the constraints it is in rather than scattering the changes
with atomic operations into a buffer in device memory.
`local_atoms[s, i]` is the slot in the workgroup of atom `s` of constraint `i`.
`gather[k + (s - 1) * max_deg, i]` lists the constraints that move that atom, if constraint
`i` applies it, by position in the workgroup and signed as in `LincsAtomScatter`, padded
with zero.
`coupled_local` holds the constraints of `coupled_indices` by position in the workgroup,
padded with the position of the constraint itself.
=#
struct LincsWorkgroupAtoms{LA, G, CL}
    local_atoms::LA
    gather::G
    coupled_local::CL
    max_deg::Int
end

function build_lincs_workgroup_atoms(perm, atom1, atom2, apply_mask, coupled_indices,
                                     n_coupled, block_size)
    K_padded = length(perm)
    # Positions in a workgroup fit in 16 bits, which keeps the data each kernel reads small
    block_size <= typemax(Int16) ÷ 2 || error("LINCS gpu_block_size $block_size is too large")
    local_atoms = ones(Int16, 2, K_padded)
    contributions = Dict{Tuple{Int, Int16}, Vector{Int16}}()
    for g in 1:cld(K_padded, block_size)
        slot_of = Dict{Int, Int16}()
        for li in 1:block_size
            i = (g - 1) * block_size + li
            (i > K_padded || perm[i] == 0) && continue
            for (s, a) in ((1, atom1[i]), (2, atom2[i]))
                slot = get!(slot_of, a, Int16(length(slot_of) + 1))
                local_atoms[s, i] = slot
                push!(get!(Vector{Int16}, contributions, (g, slot)),
                      s == 1 ? Int16(-li) : Int16(li))
            end
        end
    end
    max_deg = maximum(length, values(contributions); init=1)
    gather = zeros(Int16, 2 * max_deg, K_padded)
    max_coupled = size(coupled_indices, 1)
    coupled_local = Matrix{Int16}(undef, max_coupled, K_padded)
    for i in 1:K_padded
        g = (i - 1) ÷ block_size + 1
        li = i - (g - 1) * block_size
        for j in 1:max_coupled
            coupled_local[j, i] = (j <= n_coupled[i] ? coupled_indices[j, i] -
                                                       (g - 1) * block_size : li)
        end
        perm[i] == 0 && continue
        for (s, bit) in ((1, 0x01), (2, 0x02))
            iszero(apply_mask[i] & bit) && continue
            for (k, c) in enumerate(contributions[(g, local_atoms[s, i])])
                gather[k + (s - 1) * max_deg, i] = c
            end
        end
    end
    return LincsWorkgroupAtoms(local_atoms, gather, coupled_local, max_deg)
end

struct LincsData{A1, A2, L, IM, SD, CM, AS, AM}
    atom1::A1
    atom2::A2
    lengths::L
    invmass::IM
    sdiag::SD
    coupling::CM
    n_rec::Int
    n_iter::Int
    scatter::AS # `LincsAtomScatter` on CPU, `LincsWorkgroupAtoms` on GPU
    apply_mask::AM # `nothing` on CPU, see `lincs_apply_owned!`
end

struct LincsWorkspace{BV, R, S, TM, BL, FS}
    B::BV
    rhs::R
    sol::S
    tmp::TM
    blcc::BL
    factor_sum::FS
end

"""
    LINCS(; masses, dist_tolerance=1e-8u"nm", vel_tolerance=1e-8u"nm^2 * ps^-1",
          dist_constraints=nothing, angle_constraints=nothing, n_rec=4, n_iter=1,
          iter_vel_correction=false, gpu_block_size=128)

Constrain bond distances during a simulation using the LINCS (LINear Constraint Solver)
algorithm.

LINCS is a non-iterative constraint algorithm that uses matrix expansion to approximate
the inverse of the constraint coupling matrix. It is typically faster than
[`SHAKE_RATTLE`](@ref) for large systems but is approximate for ring topologies.
Either or both of `dist_constraints` and `angle_constraints` must be given.
The masses and constraints should always be on the CPU, even for a GPU system; the
[`System`](@ref) constructor moves the constraint data to the device of the system.
[`SetupLINCS`](@ref) provides LINCS parameters when setting up a system from a file.

Velocity constraints are applied by projecting out the velocity components along the
constraints, see `iter_vel_correction`.
See [Hess et al. 1997](https://doi.org/10.1002/(SICI)1096-987X(199709)18:12<1463::AID-JCC4>3.0.CO;2-H)
for the original LINCS paper.

Not compatible with gradient calculation using Enzyme.

# Arguments
- `masses`: vector of atom masses.
- `dist_tolerance=1e-8u"nm"`: the tolerance for checking position constraints, should
    have the same units as the coordinates.
- `vel_tolerance=1e-8u"nm^2 * ps^-1"`: the tolerance for checking velocity constraints,
    should have the same units as the velocities times the coordinates.
- `dist_constraints`: a vector of [`DistanceConstraint`](@ref) objects.
- `angle_constraints`: a vector of [`AngleConstraint`](@ref) objects. Each angle constraint
    is converted into 3 distance constraints internally. LINCS requires that angle
    constraints are isolated: none of their atoms may participate in distance constraints
    or in other angle constraints.
- `n_rec=4`: order of the matrix expansion for coupling matrix inversion. Higher values
    improve accuracy for coupled constraints at the cost of performance.
- `n_iter=1`: number of outer correction iterations for rotational lengthening. Higher
    values improve accuracy for strongly perturbed bonds.
- `iter_vel_correction=false`: whether to use iterative velocity constraint solving.
    When `false`, velocity correction uses one LINCS projection as in
    GROMACS. When `true`, additional iterative velocity corrections are performed.
- `gpu_block_size=128`: the number of threads per block to use for GPU calculations.
"""
struct LINCS{CL, LD, LW, DC, AC, E, F, DB, CI}
    clusters::CL
    lincs_data::LD
    workspace::LW
    dist_constraints::DC
    angle_constraints::AC
    dist_tolerance::E
    vel_tolerance::F
    iter_vel_correction::Bool
    gpu_block_size::Int
    delta_buf::DB         # 3 × n_atoms matrix for atomic scatter (or nothing on CPU)
    constrained_atoms::CI # sorted unique atom indices in constraints (or nothing on CPU)
end

function validate_angle_constraints(dist_constraints, angle_constraints)
    dist_atoms = Set{Int}()
    for dc in dist_constraints
        push!(dist_atoms, dc.i, dc.j)
    end

    # Index of the angle constraint each atom appears in, so that shared atoms can be
    #   found without comparing every pair of angle constraints
    angle_atoms = Dict{Int, Int}()

    for (idx, ac) in enumerate(angle_constraints)
        ac_atoms = (ac.i, ac.j, ac.k)
        for a in ac_atoms
            if a in dist_atoms
                throw(ArgumentError(
                    "angle constraint $idx (atoms $(ac.i)/$(ac.j)/$(ac.k)) shares " *
                    "atom $a with a distance constraint; LINCS requires angle " *
                    "constraints to be isolated"))
            end
        end

        for a in ac_atoms
            idx2 = get(angle_atoms, a, 0)
            if !iszero(idx2)
                ac2 = angle_constraints[idx2]
                throw(ArgumentError(
                    "angle constraint $idx (atoms $(ac.i)/$(ac.j)/$(ac.k)) shares " *
                    "atom $a with angle constraint $idx2 (atoms $(ac2.i)/$(ac2.j)/$(ac2.k)); " *
                    "LINCS requires angle constraints to be isolated"))
            end
        end
        for a in ac_atoms
            angle_atoms[a] = idx
        end
    end
end

function LINCS(; masses,
               dist_tolerance=1e-8u"nm",
               vel_tolerance=1e-8u"nm^2 * ps^-1",
               dist_constraints=nothing,
               angle_constraints=nothing,
               n_rec::Integer=4,
               n_iter::Integer=1,
               iter_vel_correction::Bool=false,
               gpu_block_size::Integer=128)
    ustrip(dist_tolerance) > 0 || throw(ArgumentError("dist_tolerance must be greater than zero"))
    ustrip(vel_tolerance)  > 0 || throw(ArgumentError("vel_tolerance must be greater than zero" ))
    n_rec  < 0 && throw(ArgumentError("n_rec cannot be negative" ))
    n_iter < 0 && throw(ArgumentError("n_iter cannot be negative"))

    dc_present = !isnothing(dist_constraints) && length(dist_constraints) > 0
    ac_present = !isnothing(angle_constraints) && length(angle_constraints) > 0

    if !dc_present && !ac_present
        throw(ArgumentError("at least one of dist_constraints or angle_constraints must " *
                            "be provided and non-empty for LINCS"))
    end

    all_dist_constraints = dc_present ? collect(dist_constraints) : DistanceConstraint[]

    if ac_present
        validate_angle_constraints(all_dist_constraints, angle_constraints)
        for ac in angle_constraints
            append!(all_dist_constraints, to_distance_constraints(ac))
        end
    end

    if isa(all_dist_constraints, AbstractGPUArray)
        throw(ArgumentError("constraints should be passed to LINCS on CPU"))
    end

    clusters = StructArray([Cluster12Data(Int32(dc.i), Int32(dc.j), dc.dist)
                            for dc in all_dist_constraints])

    lincs_data = build_lincs_data(all_dist_constraints, masses; n_rec=Int(n_rec), n_iter=Int(n_iter))
    workspace = create_lincs_workspace(lincs_data)

    stored_angle_constraints = ac_present ? collect(angle_constraints) : nothing

    return LINCS(clusters, lincs_data, workspace, all_dist_constraints,
                 stored_angle_constraints, dist_tolerance, vel_tolerance,
                 iter_vel_correction, Int(gpu_block_size), nothing, nothing)
end

function Base.show(io::IO, lincs::LINCS)
    n_ac = isnothing(lincs.angle_constraints) ? 0 : length(lincs.angle_constraints)
    n_dc = length(lincs.dist_constraints) - 3 * n_ac # avoid counting angle constraints as distance constraints
    print(io, "LINCS with ", n_dc, " distance and ", n_ac, " angle constraints (n_rec=",
          lincs.lincs_data.n_rec, ", n_iter=", lincs.lincs_data.n_iter,
          ", iter_vel_correction=", lincs.iter_vel_correction, ")")
end

"""
    SetupLINCS(; dist_tolerance=1e-6u"nm", vel_tolerance=1e-6u"nm^2 * ps^-1",
               n_rec=4, n_iter=1, iter_vel_correction::Bool=false,
               gpu_block_size::Integer=128)

Set up constraints using the LINCS (LINear Constraint Solver) algorithm.

Passed to the [`System`](@ref) constructor from files, where it creates a set of
[`LINCS`](@ref) constraints.
See [`LINCS`](@ref) for argument descriptions.
"""
struct SetupLINCS{D, V}
    dist_tolerance::D
    vel_tolerance::V
    n_rec::Int
    n_iter::Int
    iter_vel_correction::Bool
    gpu_block_size::Int
end

function SetupLINCS(; dist_tolerance=1e-6u"nm",
                    vel_tolerance=1e-6u"nm^2 * ps^-1",
                    n_rec::Integer=4,
                    n_iter::Integer=1,
                    iter_vel_correction::Bool=false,
                    gpu_block_size::Integer=128)
    ustrip(dist_tolerance) > 0 || throw(ArgumentError("dist_tolerance must be greater than zero"))
    ustrip(vel_tolerance)  > 0 || throw(ArgumentError("vel_tolerance must be greater than zero" ))
    n_rec  < 0 && throw(ArgumentError("n_rec cannot be negative" ))
    n_iter < 0 && throw(ArgumentError("n_iter cannot be negative"))
    return SetupLINCS(dist_tolerance, vel_tolerance, n_rec, n_iter,
                      iter_vel_correction, gpu_block_size)
end

function build_constraint_algorithm(T, dist_constraints, angle_constraints, atoms_data,
                                    units, strictness, masses, ca::SetupLINCS)
    return LINCS(
        masses=masses,
        dist_tolerance=convert_setup_quantity(ca.dist_tolerance, units, T),
        vel_tolerance=convert_setup_quantity(ca.vel_tolerance, units, T),
        dist_constraints=[dist_constraints...],
        angle_constraints=[angle_constraints...],
        n_rec=ca.n_rec,
        n_iter=ca.n_iter,
        iter_vel_correction=ca.iter_vel_correction,
        gpu_block_size=ca.gpu_block_size,
    )
end

function constrained_atom_inds(lincs::LINCS)
    atom_inds = Int[]
    for dc in lincs.dist_constraints
        push!(atom_inds, dc.i, dc.j)
    end
    return unique_ind_list(atom_inds)
end

function constrained_atom_pairs(lincs::LINCS)
    D = typeof(first(lincs.dist_constraints).dist)
    atom_pairs = Tuple{Int, Int, D}[]
    for dc in lincs.dist_constraints
        push!(atom_pairs, sort_pair(dc.i, dc.j, dc.dist))
    end
    return unique_pair_list(atom_pairs)
end

cluster_keys(::LINCS) = (:clusters,)

# --- Setup functions ---

function build_lincs_coupling_matrix(atom1, atom2, invmass, sdiag, ::Type{T}) where T
    K = length(atom1)

    atom_to_constraints = Dict{Int, Vector{Int}}()
    for (ci, (a1, a2)) in enumerate(zip(atom1, atom2))
        for a in (a1, a2)
            push!(get!(Vector{Int}, atom_to_constraints, a), ci)
        end
    end

    neighbor_lists = [Int[] for _ in 1:K]
    coef_lists = [T[] for _ in 1:K]

    for (ci, (a1_i, a2_i)) in enumerate(zip(atom1, atom2))
        for a in (a1_i, a2_i)
            for cj in atom_to_constraints[a]
                cj == ci && continue

                a1_j = atom1[cj]

                # Sign convention (GROMACS):
                # -1 if both constraints use center in same position (both atom1 or both atom2)
                # +1 if center is atom1 of one and atom2 of the other
                same_side_i = (a == a1_i)
                same_side_j = (a == a1_j)
                sign = (same_side_i == same_side_j) ? T(-1) : T(1)

                coef = sign * invmass[a] * sdiag[ci] * sdiag[cj]
                push!(neighbor_lists[ci], cj)
                push!(coef_lists[ci], coef)
            end
        end
    end

    # Deduplicate: a pair sharing two atoms would appear twice
    for ci in 1:K
        seen = Dict{Int, Int}()
        dedup_neighbors = Int[]
        dedup_coefs = T[]
        for (idx, cj) in enumerate(neighbor_lists[ci])
            if haskey(seen, cj)
                dedup_coefs[seen[cj]] += coef_lists[ci][idx]
            else
                seen[cj] = length(dedup_neighbors) + 1
                push!(dedup_neighbors, cj)
                push!(dedup_coefs, coef_lists[ci][idx])
            end
        end
        neighbor_lists[ci] = dedup_neighbors
        coef_lists[ci] = dedup_coefs
    end

    # Pack into CSR format
    range = Vector{Int}(undef, K + 1)
    range[1] = 1
    for ci in 1:K
        range[ci + 1] = range[ci] + length(neighbor_lists[ci])
    end
    ncc = range[K + 1] - 1
    neighbors = Vector{Int}(undef, max(ncc, 0))
    coef = Vector{T}(undef, max(ncc, 0))
    for ci in 1:K
        idx_start = range[ci]
        for (j, cj) in enumerate(neighbor_lists[ci])
            neighbors[idx_start + j - 1] = cj
            coef[idx_start + j - 1] = coef_lists[ci][j]
        end
    end

    return LincsCouplingMatrix(range, neighbors, coef)
end

function build_lincs_data(dist_constraints::AbstractVector{<:DistanceConstraint},
                          masses::AbstractVector;
                          n_rec::Integer=4, n_iter::Integer=1)
    T = typeof(float(ustrip(masses[1])))
    K = length(dist_constraints)

    atom1 = [dc.i for dc in dist_constraints]
    atom2 = [dc.j for dc in dist_constraints]
    lengths = T[ustrip(dc.dist) for dc in dist_constraints]

    raw_masses = T[ustrip(m) for m in masses]
    invmass = inv.(raw_masses)

    sdiag = Vector{T}(undef, K)
    for i in 1:K
        sdiag[i] = inv(sqrt(invmass[atom1[i]] + invmass[atom2[i]]))
    end

    coupling = build_lincs_coupling_matrix(atom1, atom2, invmass, sdiag, T)
    scatter = build_lincs_atom_scatter(atom1, atom2, length(masses))
    return LincsData(atom1, atom2, lengths, invmass, sdiag, coupling, n_rec, n_iter, scatter,
                     nothing)
end

function create_lincs_workspace(data::LincsData)
    T = eltype(data.lengths)
    K = length(data.atom1)
    ncc = length(data.coupling.neighbors)
    B = Vector{SVector{3, T}}(undef, K)
    rhs = zeros(T, K)
    sol = zeros(T, K)
    tmp = zeros(T, K)
    blcc = zeros(T, ncc)
    factor_sum = zeros(T, K)
    return LincsWorkspace(B, rhs, sol, tmp, blcc, factor_sum)
end

# --- GPU grouping and dense coupling layout ---

function group_constraints_for_gpu(atom1, atom2, block_size)
    K = length(atom1)
    K == 0 && return Int[]

    # Build constraint adjacency via shared atoms
    atom_to_constraints = Dict{Int, Vector{Int}}()
    for (ci, (a1, a2)) in enumerate(zip(atom1, atom2))
        for a in (a1, a2)
            push!(get!(Vector{Int}, atom_to_constraints, a), ci)
        end
    end

    # BFS to find connected components
    visited = falses(K)
    components = Vector{Vector{Int}}()
    for ci in 1:K
        visited[ci] && continue
        comp = Int[]
        queue = [ci]
        visited[ci] = true
        while !isempty(queue)
            c = popfirst!(queue)
            push!(comp, c)
            for a in (atom1[c], atom2[c])
                for cj in get(atom_to_constraints, a, Int[])
                    if !visited[cj]
                        visited[cj] = true
                        push!(queue, cj)
                    end
                end
            end
        end
        push!(components, comp)
    end

    for comp in components
        if length(comp) > block_size
            error(
                "LINCS connected component of $(length(comp)) coupled constraints exceeds " *
                "gpu_block_size=$block_size; increase gpu_block_size in the LINCS constructor " *
                "to at least $(length(comp)), or use CPU constraints for this system",
            )
        end
    end

    sort!(components, by=length, rev=true)

    # Pack components into blocks, padding with 0 (dummy)
    perm = Int[]
    pos_in_block = 0
    for comp in components
        nc = length(comp)
        if nc > block_size - pos_in_block && pos_in_block > 0
            append!(perm, zeros(Int, block_size - pos_in_block))
            pos_in_block = 0
        end
        append!(perm, comp)
        pos_in_block = (pos_in_block + nc) % block_size
    end
    if pos_in_block > 0
        append!(perm, zeros(Int, block_size - pos_in_block))
    end

    return perm
end

function build_dense_coupling(perm, coupling_csr::LincsCouplingMatrix, ::Type{T}) where T
    K_padded = length(perm)

    # Build inverse permutation (original constraint index → new position)
    inv_perm = Dict{Int, Int}()
    for (new_i, old_c) in enumerate(perm)
        if old_c != 0
            inv_perm[old_c] = new_i
        end
    end

    max_coupled = 0
    for old_c in perm
        old_c == 0 && continue
        nc = coupling_csr.range[old_c + 1] - coupling_csr.range[old_c]
        max_coupled = max(max_coupled, nc)
    end
    max_coupled = max(max_coupled, 1)

    # Build dense arrays (column-major: [max_coupled, K_padded])
    coupled_indices = ones(Int, max_coupled, K_padded)
    coupled_coef_arr = zeros(T, max_coupled, K_padded)
    n_coupled_arr = zeros(Int, K_padded)

    for (new_i, old_c) in enumerate(perm)
        old_c == 0 && continue
        start = coupling_csr.range[old_c]
        stop = coupling_csr.range[old_c + 1] - 1
        nc = stop - start + 1
        n_coupled_arr[new_i] = nc
        for (j, n) in enumerate(start:stop)
            coupled_indices[j, new_i] = inv_perm[coupling_csr.neighbors[n]]
            coupled_coef_arr[j, new_i] = coupling_csr.coef[n]
        end
    end

    return LincsCouplingDense(coupled_indices, coupled_coef_arr, n_coupled_arr, max_coupled)
end

# --- Core algorithm ---

@inline function lincs_bond_vector(coords, a1, a2, boundary)
    return ustrip.(vector(coords[a2], coords[a1], boundary))
end

# --- GPU kernels ---
const default_lincs_block_size = 128
# The most coupled constraints for which `lincs_fused_position_correction_kernel!` is used
const max_coupled_registers = 8

# These fused kernels rely on @synchronize (workgroup barrier) between matrix expansion
# iterations. Correctness requires that all coupled constraints within a connected
# component reside in the same workgroup. This is ensured by group_constraints_for_gpu,
# which packs connected components into blocks, and by passing gpu_block_size as the
# workgroup size to the KernelAbstractions kernel constructor.
# It also means that every constraint that moves an atom is in the same workgroup, so
# once the workgroup has added its changes to `delta_buf` it can apply them to its atoms
# itself rather than a separate kernel doing so, see `lincs_apply_owned!`.

#=
Add the changes accumulated in `delta_buf` to the coordinates or velocities `x` of the
atoms that constraint `i` applies, and zero them, after a workgroup barrier.
`apply_mask[i]` has bit 1 set if the constraint applies its first atom and bit 2 if it
applies its second, with each constrained atom applied by one constraint. Called by every
thread of the workgroup. The changes are read with an atomic swap since they were added
with atomics.
=#
@inline function lincs_apply_atom!(x, delta_buf, a, unit_scale)
    T = eltype(delta_buf)
    d1 = Atomix.@atomicswap delta_buf[1, a] = zero(T)
    d2 = Atomix.@atomicswap delta_buf[2, a] = zero(T)
    d3 = Atomix.@atomicswap delta_buf[3, a] = zero(T)
    x[a] += SVector(d1, d2, d3) .* unit_scale
    return x
end

@inline function lincs_apply_owned!(x, delta_buf, a1, a2, mask, unit_scale)
    if !iszero(mask & 0x01)
        lincs_apply_atom!(x, delta_buf, a1, unit_scale)
    end
    if !iszero(mask & 0x02)
        lincs_apply_atom!(x, delta_buf, a2, unit_scale)
    end
    return x
end

@kernel inbounds=true function lincs_fused_position_kernel!(
        delta_buf,
        B, rhs, sol, tmp,
        coords, @Const(old_coords),
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, n_rec, boundary, factor_sum, needs_virial, @Const(apply_mask),
        unit_scale)
    i = @index(Global, Linear)
    @uniform T = eltype(lengths)

    a1, a2 = atom1[i], atom2[i]

    diff_old = lincs_bond_vector(old_coords, a1, a2, boundary)
    inv_len = inv(sqrt(dot(diff_old, diff_old)))
    B_i = diff_old * inv_len
    B[i] = B_i

    diff_new = lincs_bond_vector(coords, a1, a2, boundary)
    proj = dot(B_i, diff_new)
    val = sdiag[i] * (proj - lengths[i])
    rhs[i] = val
    sol[i] = val

    if n_rec > 0
        @synchronize
    end

    nc = n_coupled_arr[i]
    for rec in 1:n_rec
        mvb = zero(T)
        for j in 1:max_coupled
            if j <= nc
                cj = coupled_indices[j, i]
                blcc_val = coupled_coef[j, i] * dot(B_i, B[cj])
                src_val = isodd(rec) ? rhs[cj] : tmp[cj]
                mvb += blcc_val * src_val
            end
        end
        if isodd(rec)
            tmp[i] = mvb
        else
            rhs[i] = mvb
        end
        sol[i] += mvb
        if rec < n_rec
            @synchronize
        end
    end

    factor = sdiag[i] * sol[i]
    if needs_virial
        factor_sum[i] += factor
    end
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end

    @synchronize
    lincs_apply_owned!(coords, delta_buf, a1, a2, apply_mask[i], unit_scale)
end

#=
The position solve followed by the `n_iter` correction iterations for rotational
lengthening in one kernel, doing what `lincs_fused_position_kernel!` and then
`lincs_fused_correction_kernel!` do. As every constraint that moves an atom is in the
workgroup, the coordinates the corrections start from are those the workgroup has applied,
which a barrier makes visible, so the corrections do not need a launch each. The
coefficients of the coupled constraints only depend on the directions from the old
coordinates, so they are found once, with `max_coupled` given as a type parameter so that
they are kept in registers, as is the solution of each constraint. The operations are done
in the same order as in the two kernels, so the results are the same.
=#
# The indices and coefficients of the constraints coupled to constraint `i`, padded with `i`
#   and zero to `max_coupled`
@inline function lincs_coupled_coefs(coupled_indices, coupled_coef, B, B_i, i, nc,
                                     ::Val{max_coupled}) where max_coupled
    coupled = ntuple(j -> @inbounds(j <= nc ? coupled_indices[j, i] : i), Val(max_coupled))
    blcc = ntuple(Val(max_coupled)) do j
        @inline
        @inbounds(j <= nc ? coupled_coef[j, i] * dot(B_i, B[coupled[j]]) :
                            zero(eltype(coupled_coef)))
    end
    return coupled, blcc
end

# The sum over the coupled constraints of the coefficient times `src` of the constraint, in
#   order, unrolled so that the tuples stay in registers
@inline function lincs_coupled_sum(blcc::Tuple, coupled::Tuple, src, nc)
    T = eltype(src)
    terms = ntuple(Val(length(blcc))) do j
        @inline
        @inbounds(j <= nc ? blcc[j] * src[coupled[j]] : zero(T))
    end
    return foldl(+, terms; init=zero(T))
end

@kernel inbounds=true function lincs_fused_position_correction_kernel!(
        delta_buf,
        B, rhs, tmp,
        coords, @Const(old_coords),
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        ::Val{max_coupled}, n_rec, n_iter, boundary, factor_sum, needs_virial,
        @Const(apply_mask), unit_scale) where max_coupled
    i = @index(Global, Linear)
    @uniform T = eltype(lengths)

    a1, a2 = atom1[i], atom2[i]
    length_i, sdiag_i = lengths[i], sdiag[i]
    invmass_1, invmass_2 = invmass[a1], invmass[a2]
    mask_i = apply_mask[i]

    diff_old = lincs_bond_vector(old_coords, a1, a2, boundary)
    inv_len = inv(sqrt(dot(diff_old, diff_old)))
    B_i = diff_old * inv_len
    B[i] = B_i

    diff_new = lincs_bond_vector(coords, a1, a2, boundary)
    proj = dot(B_i, diff_new)
    val = sdiag_i * (proj - length_i)

    @synchronize

    nc = n_coupled_arr[i]
    coupled, blcc = lincs_coupled_coefs(coupled_indices, coupled_coef, B, B_i, i, nc,
                                        Val(max_coupled))

    for iter in 0:n_iter
        if iter > 0
            # The coordinates applied by the workgroup
            @synchronize
            diff = lincs_bond_vector(coords, a1, a2, boundary)
            dlen2 = 2 * length_i^2 - dot(diff, diff)
            p = sqrt(max(dlen2, zero(T)))
            val = sdiag_i * (length_i - p)
        end
        rhs[i] = val
        sol_i = val

        if n_rec > 0
            @synchronize
        end
        for rec in 1:n_rec
            mvb = lincs_coupled_sum(blcc, coupled, (isodd(rec) ? rhs : tmp), nc)
            if isodd(rec)
                tmp[i] = mvb
            else
                rhs[i] = mvb
            end
            sol_i += mvb
            if rec < n_rec
                @synchronize
            end
        end

        factor = sdiag_i * sol_i
        if needs_virial
            factor_sum[i] += factor
        end
        delta = B_i * factor
        for dim in 1:3
            d = delta[dim]
            Atomix.@atomic delta_buf[dim, a1] -= invmass_1 * d
            Atomix.@atomic delta_buf[dim, a2] += invmass_2 * d
        end

        @synchronize
        lincs_apply_owned!(coords, delta_buf, a1, a2, mask_i, unit_scale)
    end
end

#=
LINCS on the GPU with the data of each workgroup in shared memory: the directions,
right-hand sides and solutions of its constraints, and the coordinates or velocities of its
atoms, see `LincsWorkgroupAtoms`. Each atom is loaded by the constraint that applies it,
which adds the changes from the constraints the atom is in, in a fixed order, and writes the
atom back at the end. This does the operations of `lincs_fused_position_correction_kernel!`
and `lincs_fused_velocity_kernel!` without the atomic operations in device memory and with
one read and write of each atom, apart from the order in which the changes to an atom are
added. The directions are written to `B` when the virial is needed.
=#
const max_deg_registers = 8

@inline lincs_shared_vec(arr, k) = @inbounds SVector(arr[1, k], arr[2, k], arr[3, k])

@inline function lincs_shared_store!(arr, k, v)
    @inbounds arr[1, k] = v[1]
    @inbounds arr[2, k] = v[2]
    @inbounds arr[3, k] = v[3]
    return arr
end

# The coefficients of the constraints coupled to the one at position `li` of the
#   workgroup, from the directions in shared memory, padded with zero to `max_coupled`
@inline function lincs_shared_coupled_coefs(coupled_local, coupled_coef, B_sh, B_i, i, nc,
                                            ::Val{max_coupled}) where max_coupled
    coupled = ntuple(j -> @inbounds(coupled_local[j, i]), Val(max_coupled))
    # Without inlining the closure is called with its captures on the stack, which every
    #   thread then reads and writes in device memory
    blcc = ntuple(Val(max_coupled)) do j
        @inline
        @inbounds(j <= nc ? coupled_coef[j, i] * dot(B_i, lincs_shared_vec(B_sh, coupled[j])) :
                            zero(eltype(coupled_coef)))
    end
    return coupled, blcc
end

# The change to an atom from the constraints listed for it in `gather`, which are signed by
#   which end of the constraint the atom is, each change found as in the scatter kernels
@inline function lincs_gather_delta(gather, i, offset, ::Val{max_deg}, B_sh, f_sh,
                                    invmass_a) where max_deg
    T = eltype(f_sh)
    d1, d2, d3 = zero(T), zero(T), zero(T)
    @inbounds for k in 1:max_deg
        c = gather[k + offset, i]
        if c != 0
            lc = abs(c)
            factor = f_sh[lc]
            if c < 0
                d1 -= invmass_a * (B_sh[1, lc] * factor)
                d2 -= invmass_a * (B_sh[2, lc] * factor)
                d3 -= invmass_a * (B_sh[3, lc] * factor)
            else
                d1 += invmass_a * (B_sh[1, lc] * factor)
                d2 += invmass_a * (B_sh[2, lc] * factor)
                d3 += invmass_a * (B_sh[3, lc] * factor)
            end
        end
    end
    return SVector(d1, d2, d3)
end

# The matrix expansion for the right-hand side `val` of the constraint at position `li`,
#   which every thread of the workgroup calls, returning the solution
# Only for the GPU kernels, as the CPU backend of KernelAbstractions only allows
#   `@synchronize` in the body of a kernel
@inline function lincs_shared_solve(rhs_sh, tmp_sh, li, val, blcc, coupled, nc, n_rec)
    @inbounds rhs_sh[li] = val
    sol_i = val
    if n_rec > 0
        @synchronize()
    end
    for rec in 1:n_rec
        mvb = lincs_coupled_sum(blcc, coupled, (isodd(rec) ? rhs_sh : tmp_sh), nc)
        if isodd(rec)
            @inbounds tmp_sh[li] = mvb
        else
            @inbounds rhs_sh[li] = mvb
        end
        sol_i += mvb
        if rec < n_rec
            @synchronize()
        end
    end
    return sol_i
end

@kernel inbounds=true function lincs_shared_position_kernel!(
        coords, @Const(old_coords),
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(invmass), @Const(sdiag),
        @Const(coupled_local), @Const(coupled_coef), @Const(n_coupled_arr),
        @Const(local_atoms), @Const(gather), B,
        ::Val{max_coupled}, ::Val{max_deg}, ::Val{block_size}, n_rec, n_iter, boundary,
        factor_sum, needs_virial, @Const(apply_mask),
        unit_scale) where {max_coupled, max_deg, block_size}
    i = @index(Global, Linear)
    li = @index(Local, Linear)
    @uniform T = eltype(lengths)
    @uniform C = eltype(coords)
    B_sh = @localmem T (3, block_size)
    rhs_sh = @localmem T (block_size,)
    tmp_sh = @localmem T (block_size,)
    f_sh = @localmem T (block_size,)
    x_sh = @localmem C (2 * block_size,)

    a1, a2 = atom1[i], atom2[i]
    la1, la2 = local_atoms[1, i], local_atoms[2, i]
    length_i, sdiag_i = lengths[i], sdiag[i]
    invmass_1, invmass_2 = invmass[a1], invmass[a2]
    mask_i = apply_mask[i]

    x1, x2 = coords[a1], coords[a2]
    if !iszero(mask_i & 0x01)
        x_sh[la1] = x1
    end
    if !iszero(mask_i & 0x02)
        x_sh[la2] = x2
    end

    diff_old = lincs_bond_vector(old_coords, a1, a2, boundary)
    inv_len = inv(sqrt(dot(diff_old, diff_old)))
    B_i = diff_old * inv_len
    if needs_virial
        B[i] = B_i
    end
    lincs_shared_store!(B_sh, li, B_i)

    diff_new = ustrip.(vector(x2, x1, boundary))
    proj = dot(B_i, diff_new)
    val = sdiag_i * (proj - length_i)

    @synchronize

    nc = n_coupled_arr[i]
    coupled, blcc = lincs_shared_coupled_coefs(coupled_local, coupled_coef, B_sh, B_i, i,
                                               nc, Val(max_coupled))

    for iter in 0:n_iter
        if iter > 0
            # The coordinates applied by the workgroup
            @synchronize
            diff = ustrip.(vector(x_sh[la2], x_sh[la1], boundary))
            dlen2 = 2 * length_i^2 - dot(diff, diff)
            p = sqrt(max(dlen2, zero(T)))
            val = sdiag_i * (length_i - p)
        end
        sol_i = lincs_shared_solve(rhs_sh, tmp_sh, li, val, blcc, coupled, nc, n_rec)

        factor = sdiag_i * sol_i
        if needs_virial
            factor_sum[i] += factor
        end
        f_sh[li] = factor

        @synchronize
        if !iszero(mask_i & 0x01)
            x_sh[la1] += lincs_gather_delta(gather, i, 0, Val(max_deg), B_sh, f_sh,
                                            invmass_1) .* unit_scale
        end
        if !iszero(mask_i & 0x02)
            x_sh[la2] += lincs_gather_delta(gather, i, max_deg, Val(max_deg), B_sh, f_sh,
                                            invmass_2) .* unit_scale
        end
    end

    if !iszero(mask_i & 0x01)
        coords[a1] = x_sh[la1]
    end
    if !iszero(mask_i & 0x02)
        coords[a2] = x_sh[la2]
    end
end

# The velocity solve, followed by `n_iter` correction iterations as
#   `lincs_fused_velocity_correction_kernel!` does
@kernel inbounds=true function lincs_shared_velocity_kernel!(
        @Const(coords), velocities,
        @Const(atom1), @Const(atom2), @Const(invmass), @Const(sdiag),
        @Const(coupled_local), @Const(coupled_coef), @Const(n_coupled_arr),
        @Const(local_atoms), @Const(gather), B,
        ::Val{max_coupled}, ::Val{max_deg}, ::Val{block_size}, n_rec, n_iter, boundary,
        factor_sum, needs_virial, @Const(apply_mask),
        unit_scale) where {max_coupled, max_deg, block_size}
    i = @index(Global, Linear)
    li = @index(Local, Linear)
    @uniform T = eltype(sdiag)
    @uniform V = eltype(velocities)
    B_sh = @localmem T (3, block_size)
    rhs_sh = @localmem T (block_size,)
    tmp_sh = @localmem T (block_size,)
    f_sh = @localmem T (block_size,)
    v_sh = @localmem V (2 * block_size,)

    a1, a2 = atom1[i], atom2[i]
    la1, la2 = local_atoms[1, i], local_atoms[2, i]
    sdiag_i = sdiag[i]
    invmass_1, invmass_2 = invmass[a1], invmass[a2]
    mask_i = apply_mask[i]

    v1, v2 = velocities[a1], velocities[a2]
    if !iszero(mask_i & 0x01)
        v_sh[la1] = v1
    end
    if !iszero(mask_i & 0x02)
        v_sh[la2] = v2
    end

    diff = lincs_bond_vector(coords, a1, a2, boundary)
    inv_len = inv(sqrt(dot(diff, diff)))
    B_i = diff * inv_len
    if needs_virial
        B[i] = B_i
    end
    lincs_shared_store!(B_sh, li, B_i)

    dv = ustrip.(v2 - v1)
    val = -sdiag_i * dot(B_i, dv)

    @synchronize

    nc = n_coupled_arr[i]
    coupled, blcc = lincs_shared_coupled_coefs(coupled_local, coupled_coef, B_sh, B_i, i,
                                               nc, Val(max_coupled))

    for iter in 0:n_iter
        if iter > 0
            # The velocities applied by the workgroup
            @synchronize
            dv = ustrip.(v_sh[la2] - v_sh[la1])
            val = -sdiag_i * dot(B_i, dv)
        end
        sol_i = lincs_shared_solve(rhs_sh, tmp_sh, li, val, blcc, coupled, nc, n_rec)

        factor = sdiag_i * sol_i
        if needs_virial
            factor_sum[i] += factor
        end
        f_sh[li] = factor

        @synchronize
        if !iszero(mask_i & 0x01)
            v_sh[la1] += lincs_gather_delta(gather, i, 0, Val(max_deg), B_sh, f_sh,
                                            invmass_1) .* unit_scale
        end
        if !iszero(mask_i & 0x02)
            v_sh[la2] += lincs_gather_delta(gather, i, max_deg, Val(max_deg), B_sh, f_sh,
                                            invmass_2) .* unit_scale
        end
    end

    if !iszero(mask_i & 0x01)
        velocities[a1] = v_sh[la1]
    end
    if !iszero(mask_i & 0x02)
        velocities[a2] = v_sh[la2]
    end
end

# Whether the shared memory kernels can be used for the constraints
function use_lincs_shared_kernels(data)
    return data.scatter isa LincsWorkgroupAtoms &&
           data.coupling.max_coupled <= max_coupled_registers &&
           data.scatter.max_deg <= max_deg_registers
end

@kernel inbounds=true function lincs_fused_correction_kernel!(
        delta_buf,
        @Const(B), rhs, sol, tmp,
        coords,
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, n_rec, boundary, factor_sum, needs_virial, @Const(apply_mask),
        unit_scale)
    i = @index(Global, Linear)
    @uniform T = eltype(lengths)

    a1, a2 = atom1[i], atom2[i]

    # Correction RHS (rotational lengthening) + recompute B from current coords
    diff = lincs_bond_vector(coords, a1, a2, boundary)
    dlen2 = 2 * lengths[i]^2 - dot(diff, diff)
    p = sqrt(max(dlen2, zero(T)))
    val = sdiag[i] * (lengths[i] - p)
    rhs[i] = val
    sol[i] = val

    B_i = B[i]

    if n_rec > 0
        @synchronize
    end

    nc = n_coupled_arr[i]
    for rec in 1:n_rec
        mvb = zero(T)
        for j in 1:max_coupled
            if j <= nc
                cj = coupled_indices[j, i]
                blcc_val = coupled_coef[j, i] * dot(B_i, B[cj])
                src_val = isodd(rec) ? rhs[cj] : tmp[cj]
                mvb += blcc_val * src_val
            end
        end
        if isodd(rec)
            tmp[i] = mvb
        else
            rhs[i] = mvb
        end
        sol[i] += mvb
        if rec < n_rec
            @synchronize
        end
    end

    factor = sdiag[i] * sol[i]
    if needs_virial
        factor_sum[i] += factor
    end
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end

    @synchronize
    lincs_apply_owned!(coords, delta_buf, a1, a2, apply_mask[i], unit_scale)
end

@kernel inbounds=true function lincs_fused_velocity_kernel!(
        delta_buf,
        B, rhs, sol, tmp,
        @Const(coords), velocities,
        @Const(atom1), @Const(atom2), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, n_rec, boundary, factor_sum, needs_virial, @Const(apply_mask),
        unit_scale)
    i = @index(Global, Linear)
    @uniform T = eltype(sdiag)

    a1, a2 = atom1[i], atom2[i]

    diff = lincs_bond_vector(coords, a1, a2, boundary)
    inv_len = inv(sqrt(dot(diff, diff)))
    B_i = diff * inv_len
    B[i] = B_i
    dv = ustrip.(velocities[a2] - velocities[a1])
    val = -sdiag[i] * dot(B_i, dv)
    rhs[i] = val
    sol[i] = val

    if n_rec > 0
        @synchronize
    end

    nc = n_coupled_arr[i]
    for rec in 1:n_rec
        mvb = zero(T)
        for j in 1:max_coupled
            if j <= nc
                cj = coupled_indices[j, i]
                blcc_val = coupled_coef[j, i] * dot(B_i, B[cj])
                src_val = isodd(rec) ? rhs[cj] : tmp[cj]
                mvb += blcc_val * src_val
            end
        end
        if isodd(rec)
            tmp[i] = mvb
        else
            rhs[i] = mvb
        end
        sol[i] += mvb
        if rec < n_rec
            @synchronize
        end
    end

    factor = sdiag[i] * sol[i]
    if needs_virial
        factor_sum[i] += factor
    end
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end

    @synchronize
    lincs_apply_owned!(velocities, delta_buf, a1, a2, apply_mask[i], unit_scale)
end

@kernel inbounds=true function lincs_fused_velocity_correction_kernel!(
        delta_buf,
        @Const(B), rhs, sol, tmp,
        velocities,
        @Const(atom1), @Const(atom2), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, n_rec, factor_sum, needs_virial, @Const(apply_mask), unit_scale)
    i = @index(Global, Linear)
    @uniform T = eltype(sdiag)

    a1, a2 = atom1[i], atom2[i]

    B_i = B[i]
    dv = ustrip.(velocities[a2] - velocities[a1])
    val = -sdiag[i] * dot(B_i, dv)
    rhs[i] = val
    sol[i] = val

    if n_rec > 0
        @synchronize
    end

    nc = n_coupled_arr[i]
    for rec in 1:n_rec
        mvb = zero(T)
        for j in 1:max_coupled
            if j <= nc
                cj = coupled_indices[j, i]
                blcc_val = coupled_coef[j, i] * dot(B_i, B[cj])
                src_val = isodd(rec) ? rhs[cj] : tmp[cj]
                mvb += blcc_val * src_val
            end
        end
        if isodd(rec)
            tmp[i] = mvb
        else
            rhs[i] = mvb
        end
        sol[i] += mvb
        if rec < n_rec
            @synchronize
        end
    end

    factor = sdiag[i] * sol[i]
    if needs_virial
        factor_sum[i] += factor
    end
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end

    @synchronize
    lincs_apply_owned!(velocities, delta_buf, a1, a2, apply_mask[i], unit_scale)
end

@kernel inbounds=true function lincs_accumulate_virial_kernel!(
        constraint_virial_nounits,
        @Const(B),
        @Const(lengths),
        @Const(sdiag),
        @Const(factor_sum),
        @Const(atom1),
        @Const(atom2),
        @Const(atoms),
        virial_scale)
    i = @index(Global, Linear)
    if !iszero(sdiag[i]) && !iszero(lengths[i])
        a1 = atom1[i]
        a2 = atom2[i]
        λ = constraint_virial_lambda(atoms, a1, a2)
        B_i = B[i]
        coeff = λ * virial_scale * (-lengths[i] * factor_sum[i])
        slot = virial_slot(i, size(constraint_virial_nounits, 3))

        for alpha in 1:3
            for beta in 1:3
                Atomix.@atomic constraint_virial_nounits[alpha, beta, slot] +=
                    coeff * B_i[alpha] * B_i[beta]
            end
        end
    end
end

# --- CPU solve path ---

# One iteration of the LINCS matrix expansion, reading `src` and writing `dst` and `sol`.
# Every constraint writes only its own entry, so the iteration is split over chunks with a
# barrier between iterations.
function lincs_expand!(K::Integer, n_chunks::Integer, crange, neighbors, blcc, src, dst,
                       sol, ::Type{T}) where T
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        @inbounds for i in constraint_chunk_range(K, chunk_i, n_chunks)
            mvb = zero(T)
            for n in crange[i]:(crange[i + 1] - 1)
                mvb += blcc[n] * src[neighbors[n]]
            end
            dst[i] = mvb
            sol[i] += mvb
        end
    end
    return nothing
end

# Apply the corrections from the solve to the atoms, gathering the constraints that each
# atom takes part in so that the loop can be split over chunks, see `LincsAtomScatter`
function lincs_gather!(coords, n_chunks::Integer, scatter::LincsAtomScatter, sdiag, sol, B,
                       invmass, unit_scale)
    atom_range, atom_constraints = scatter.range, scatter.constraints
    n_atoms = length(atom_range) - 1
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        @inbounds for a in constraint_chunk_range(n_atoms, chunk_i, n_chunks)
            n_start, n_stop = atom_range[a], atom_range[a + 1] - Int32(1)
            n_start > n_stop && continue
            delta = zero(eltype(B))
            for n in n_start:n_stop
                signed_i = atom_constraints[n]
                i = abs(signed_i)
                contribution = B[i] * (sdiag[i] * sol[i])
                # `atom1` is moved against the bond vector and `atom2` along it
                delta = (signed_i > 0 ? (delta + contribution) : (delta - contribution))
            end
            coords[a] += (invmass[a] * delta) .* unit_scale
        end
    end
    return nothing
end

# The virial needs the per-constraint correction factors, which the gather above does not
# accumulate since it runs over atoms
function accumulate_lincs_factors!(factor_sum, K::Integer, sdiag, sol)
    @inbounds for i in 1:K
        factor_sum[i] += sdiag[i] * sol[i]
    end
    return nothing
end

function lincs_solve!(coords, data::LincsData, ws::LincsWorkspace, unit_scale)
    return lincs_solve!(coords, data, ws, unit_scale, nothing, 1)
end

function lincs_solve!(coords, data::LincsData, ws::LincsWorkspace, unit_scale,
                      factor_sum, n_threads::Integer=Threads.nthreads())
    T = eltype(data.lengths)
    coupling = data.coupling
    K = length(data.atom1)
    n_chunks = n_constraint_chunks(K, n_threads)

    # Matrix expansion: n_rec iterations
    # Ping-pong: each iteration reads what the last one wrote. ws.rhs/ws.tmp still point to
    #   the original arrays, and callers reassign ws.rhs before reuse
    src, dst = ws.rhs, ws.tmp
    for rec in 1:data.n_rec
        lincs_expand!(K, n_chunks, coupling.range, coupling.neighbors, ws.blcc, src, dst,
                      ws.sol, T)
        src, dst = dst, src
    end

    if !isnothing(factor_sum)
        accumulate_lincs_factors!(factor_sum, K, data.sdiag, ws.sol)
    end
    lincs_gather!(coords, n_chunks, data.scatter, data.sdiag, ws.sol, ws.B, data.invmass,
                  unit_scale)
    return nothing
end

# Unit bond vectors from the coordinates before the unconstrained update, and the initial
# right hand side from the coordinates after it
function lincs_position_rhs!(K::Integer, n_chunks::Integer, atom1, atom2, old_coords,
                             coords, boundary, B, rhs, sdiag, lengths)
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        @inbounds for i in constraint_chunk_range(K, chunk_i, n_chunks)
            a1, a2 = atom1[i], atom2[i]
            diff_old = lincs_bond_vector(old_coords, a1, a2, boundary)
            B_i = diff_old * inv(sqrt(dot(diff_old, diff_old)))
            B[i] = B_i
            diff_new = lincs_bond_vector(coords, a1, a2, boundary)
            rhs[i] = sdiag[i] * (dot(B_i, diff_new) - lengths[i])
        end
    end
    return nothing
end

# Runtime coupling coefficients: blcc = coef * dot(B[i], B[neighbor]).
# Each constraint writes its own slice of the coupling CSR
function lincs_coupling_coefs!(K::Integer, n_chunks::Integer, B, crange, neighbors, coef,
                               blcc)
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        @inbounds for i in constraint_chunk_range(K, chunk_i, n_chunks)
            B_i = B[i]
            for n in crange[i]:(crange[i + 1] - 1)
                blcc[n] = coef[n] * dot(B_i, B[neighbors[n]])
            end
        end
    end
    return nothing
end

# Right hand side for the correction iterations that undo rotational lengthening
function lincs_correction_rhs!(K::Integer, n_chunks::Integer, atom1, atom2, coords,
                               boundary, lengths, sdiag, rhs, ::Type{T}) where T
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        @inbounds for i in constraint_chunk_range(K, chunk_i, n_chunks)
            a1, a2 = atom1[i], atom2[i]
            diff = lincs_bond_vector(coords, a1, a2, boundary)
            dlen2 = 2 * lengths[i]^2 - dot(diff, diff)
            if dlen2 < zero(T)
                warn_lincs_stretched(a1, a2)
            end
            p = sqrt(max(dlen2, zero(T)))
            rhs[i] = sdiag[i] * (lengths[i] - p)
        end
    end
    return nothing
end

@noinline function warn_lincs_stretched(a1, a2)
    @warn "LINCS correction: bond $(a1)-$(a2) stretched beyond sqrt(2) * target " *
          "length, constraint may be unreliable" maxlog=1
    return nothing
end

function lincs_apply!(coords, old_coords, data::LincsData, ws::LincsWorkspace,
                      boundary, context, n_threads::Integer=Threads.nthreads())
    T = eltype(data.lengths)
    coupling = data.coupling
    K = length(data.atom1)
    n_chunks = n_constraint_chunks(K, n_threads)
    unit_scale = oneunit(eltype(eltype(coords)))
    factor_sum = context.needs_virial ? ws.factor_sum : nothing
    if !isnothing(factor_sum)
        fill!(factor_sum, zero(eltype(factor_sum)))
    end

    lincs_position_rhs!(K, n_chunks, data.atom1, data.atom2, old_coords, coords, boundary,
                        ws.B, ws.rhs, data.sdiag, data.lengths)
    lincs_coupling_coefs!(K, n_chunks, ws.B, coupling.range, coupling.neighbors,
                          coupling.coef, ws.blcc)

    copyto!(ws.sol, ws.rhs)
    lincs_solve!(coords, data, ws, unit_scale, factor_sum, n_threads)

    # Outer correction iterations (rotational lengthening)
    for _ in 1:data.n_iter
        lincs_correction_rhs!(K, n_chunks, data.atom1, data.atom2, coords, boundary,
                              data.lengths, data.sdiag, ws.rhs, T)
        copyto!(ws.sol, ws.rhs)
        lincs_solve!(coords, data, ws, unit_scale, factor_sum, n_threads)
    end

    accumulate_lincs_position_virial!(data, ws, context)
    return coords
end

function accumulate_lincs_position_virial!(data::LincsData, ws::LincsWorkspace,
                                           context)
    context.needs_virial || return context
    if !(context.kind isa PositionConstraintApplication)
        error("LINCS position virial accumulation requires a position constraint context")
    end
    if isnothing(context.buffers)
        error("LINCS position virial accumulation requires context.buffers")
    end

    @inbounds for i in eachindex(data.atom1)
        a1 = data.atom1[i]
        a2 = data.atom2[i]
        λ = constraint_virial_lambda(context.atoms, a1, a2)

        coeff = λ * (-data.lengths[i] * ws.factor_sum[i])
        B_i = ws.B[i]
        contribution = coeff * (B_i * transpose(B_i))
        accumulate_constraint_virial!(context.buffers, contribution, context)
    end
    return context
end

default_position_constraint_context() = ConstraintApplicationContext(
    kind=PositionConstraintApplication(),
    needs_virial=false,
)

# Unit bond vectors from the current (constrained) coordinates and the velocity residual
function lincs_velocity_rhs!(K::Integer, n_chunks::Integer, atom1, atom2, coords,
                             velocities, boundary, B, rhs, sdiag)
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        @inbounds for i in constraint_chunk_range(K, chunk_i, n_chunks)
            a1, a2 = atom1[i], atom2[i]
            diff = lincs_bond_vector(coords, a1, a2, boundary)
            B_i = diff * inv(sqrt(dot(diff, diff)))
            B[i] = B_i
            dv = ustrip.(velocities[a2] - velocities[a1])
            rhs[i] = -sdiag[i] * dot(B_i, dv)
        end
    end
    return nothing
end

# Velocity residual for the correction iterations, reusing the bond vectors already in `B`
function lincs_velocity_correction_rhs!(K::Integer, n_chunks::Integer, atom1, atom2,
                                        velocities, B, rhs, sdiag)
    @maybe_threads (n_chunks > 1) for chunk_i in 1:n_chunks
        @inbounds for i in constraint_chunk_range(K, chunk_i, n_chunks)
            a1, a2 = atom1[i], atom2[i]
            dv = ustrip.(velocities[a2] - velocities[a1])
            rhs[i] = -sdiag[i] * dot(B[i], dv)
        end
    end
    return nothing
end

function lincs_vel_apply!(velocities, coords, data::LincsData, ws::LincsWorkspace,
                          boundary, context, n_iter_velocity::Integer=data.n_iter,
                          n_threads::Integer=Threads.nthreads())
    coupling = data.coupling
    K = length(data.atom1)
    n_chunks = n_constraint_chunks(K, n_threads)
    unit_vel_scale = oneunit(eltype(eltype(velocities)))
    factor_sum = context.needs_virial ? ws.factor_sum : nothing
    if !isnothing(factor_sum)
        fill!(factor_sum, zero(eltype(factor_sum)))
    end

    lincs_velocity_rhs!(K, n_chunks, data.atom1, data.atom2, coords, velocities, boundary,
                        ws.B, ws.rhs, data.sdiag)
    lincs_coupling_coefs!(K, n_chunks, ws.B, coupling.range, coupling.neighbors,
                          coupling.coef, ws.blcc)

    copyto!(ws.sol, ws.rhs)
    lincs_solve!(velocities, data, ws, unit_vel_scale, factor_sum, n_threads)

    # Iterative correction: re-evaluate velocity residual and solve again
    for _ in 1:n_iter_velocity
        lincs_velocity_correction_rhs!(K, n_chunks, data.atom1, data.atom2, velocities,
                                       ws.B, ws.rhs, data.sdiag)
        copyto!(ws.sol, ws.rhs)
        lincs_solve!(velocities, data, ws, unit_vel_scale, factor_sum, n_threads)
    end

    accumulate_lincs_velocity_virial!(data, ws, context)
    return velocities
end

function accumulate_lincs_velocity_virial!(data::LincsData, ws::LincsWorkspace,
                                           context)
    context.needs_virial || return context

    if !(context.kind isa VelocityConstraintApplication)
        error("LINCS velocity virial accumulation requires a velocity constraint context")
    end
    if isnothing(context.buffers)
        error("LINCS velocity virial accumulation requires context.buffers")
    end

    @inbounds for i in eachindex(data.atom1)
        a1 = data.atom1[i]
        a2 = data.atom2[i]
        λ = constraint_virial_lambda(context.atoms, a1, a2)

        coeff = λ * (-data.lengths[i] * ws.factor_sum[i])
        B_i = ws.B[i]
        contribution = coeff * (B_i * transpose(B_i))
        accumulate_constraint_virial!(context.buffers, contribution, context)
    end


    return context
end

default_velocity_constraint_context() = ConstraintApplicationContext(
    kind=VelocityConstraintApplication(),
    needs_virial=false,
)

# --- GPU solve path ---

function accumulate_lincs_position_virial_gpu!(data, ws, context, backend, block_size)
    context.needs_virial || return context
    if !(context.kind isa PositionConstraintApplication)
        error("LINCS position virial accumulation requires a position constraint context")
    end
    if isnothing(context.buffers)
        error("LINCS position virial accumulation requires context.buffers")
    end
    virial_kern! = lincs_accumulate_virial_kernel!(backend, block_size)
    virial_scale = eltype(data.lengths)(ustrip(context.virial_scale))
    launch_kernel!(virial_kern!, context.buffers.constraint_virial_slots, ws.B, data.lengths,
                 data.sdiag, ws.factor_sum, data.atom1, data.atom2, context.atoms, virial_scale;
                 ndrange=length(data.atom1))
    fold_virial_slots!(context.buffers.constraint_virial_nounits,
                       context.buffers.constraint_virial_slots)
    return context
end

function accumulate_lincs_velocity_virial_gpu!(data, ws, context, backend, block_size)
    context.needs_virial || return context
    if !(context.kind isa VelocityConstraintApplication)
        error("LINCS velocity virial accumulation requires a velocity constraint context")
    end
    if isnothing(context.buffers)
        error("LINCS velocity virial accumulation requires context.buffers")
    end
    virial_kern! = lincs_accumulate_virial_kernel!(backend, block_size)
    virial_scale = eltype(data.lengths)(ustrip(context.virial_scale))
    launch_kernel!(virial_kern!, context.buffers.constraint_virial_slots, ws.B, data.lengths,
                 data.sdiag, ws.factor_sum, data.atom1, data.atom2, context.atoms, virial_scale;
                 ndrange=length(data.atom1))
    fold_virial_slots!(context.buffers.constraint_virial_nounits,
                       context.buffers.constraint_virial_slots)
    return context
end

function lincs_apply_gpu!(coords, old_coords, data, ws, boundary,
                          delta_buf, constrained_atoms, block_size, context)
    K_padded = length(data.atom1)
    backend = get_backend(coords)
    unit_scale = oneunit(eltype(eltype(coords)))
    coupling = data.coupling
    if context.needs_virial
        fill!(ws.factor_sum, zero(eltype(ws.factor_sum)))
    end

    # Fused solve: bond vectors + blcc + SpMV iterations + scatter, with each workgroup
    #   then moving its atoms, and the correction iterations (rotational lengthening)
    # The coefficients of the coupled constraints are kept in registers when there are few
    if use_lincs_shared_kernels(data)
        wg = data.scatter
        kern! = ka_kernel(lincs_shared_position_kernel!, backend, block_size,
                          Val(default_lincs_block_size))
        launch_kernel!(kern!, coords, old_coords, data.atom1, data.atom2, data.lengths,
                       data.invmass, data.sdiag, wg.coupled_local, coupling.coupled_coef,
                       coupling.n_coupled, wg.local_atoms, wg.gather, ws.B,
                       Val(coupling.max_coupled), Val(wg.max_deg), Val(block_size),
                       data.n_rec, data.n_iter, boundary, ws.factor_sum,
                       context.needs_virial, data.apply_mask, unit_scale; ndrange=K_padded)
    elseif coupling.max_coupled <= max_coupled_registers
        fused_kern! = ka_kernel(lincs_fused_position_correction_kernel!, backend, block_size,
                                Val(default_lincs_block_size))
        launch_kernel!(fused_kern!, delta_buf, ws.B, ws.rhs, ws.tmp, coords, old_coords,
                       data.atom1, data.atom2, data.lengths, data.invmass, data.sdiag,
                       coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                       Val(coupling.max_coupled), data.n_rec, data.n_iter, boundary,
                       ws.factor_sum, context.needs_virial, data.apply_mask, unit_scale;
                       ndrange=K_padded)
    else
        fused_kern! = lincs_fused_position_kernel!(backend, block_size)
        launch_kernel!(fused_kern!, delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                    coords, old_coords,
                    data.atom1, data.atom2, data.lengths, data.invmass, data.sdiag,
                    coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                    coupling.max_coupled, data.n_rec, boundary, ws.factor_sum,
                    context.needs_virial, data.apply_mask, unit_scale;
                    ndrange=K_padded)
        if data.n_iter > 0
            corr_kern! = lincs_fused_correction_kernel!(backend, block_size)
            for _ in 1:data.n_iter
                launch_kernel!(corr_kern!, delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                           coords,
                           data.atom1, data.atom2, data.lengths, data.invmass, data.sdiag,
                           coupling.coupled_indices, coupling.coupled_coef,
                           coupling.n_coupled, coupling.max_coupled, data.n_rec, boundary,
                           ws.factor_sum, context.needs_virial, data.apply_mask,
                           unit_scale; ndrange=K_padded)
            end
        end
    end

    accumulate_lincs_position_virial_gpu!(data, ws, context, backend, block_size)
    return coords
end

function lincs_vel_apply_gpu!(velocities, coords, data, ws, boundary,
                              delta_buf, constrained_atoms, block_size, context,
                              n_iter_velocity::Integer=data.n_iter)
    K_padded = length(data.atom1)
    backend = get_backend(velocities)
    unit_vel_scale = oneunit(eltype(eltype(velocities)))
    coupling = data.coupling
    if context.needs_virial
        fill!(ws.factor_sum, zero(eltype(ws.factor_sum)))
    end

    if use_lincs_shared_kernels(data)
        # The correction iterations are done in the same kernel
        wg = data.scatter
        kern! = ka_kernel(lincs_shared_velocity_kernel!, backend, block_size,
                          Val(default_lincs_block_size))
        launch_kernel!(kern!, coords, velocities, data.atom1, data.atom2, data.invmass,
                       data.sdiag, wg.coupled_local, coupling.coupled_coef,
                       coupling.n_coupled, wg.local_atoms, wg.gather, ws.B,
                       Val(coupling.max_coupled), Val(wg.max_deg), Val(block_size),
                       data.n_rec, n_iter_velocity, boundary, ws.factor_sum,
                       context.needs_virial, data.apply_mask, unit_vel_scale;
                       ndrange=K_padded)
        accumulate_lincs_velocity_virial_gpu!(data, ws, context, backend, block_size)
        return velocities
    end

    fused_kern! = ka_kernel(lincs_fused_velocity_kernel!, backend, block_size,
                            Val(default_lincs_block_size))
    launch_kernel!(fused_kern!, delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                coords, velocities,
                data.atom1, data.atom2, data.invmass, data.sdiag,
                coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                coupling.max_coupled, data.n_rec, boundary, ws.factor_sum,
                context.needs_virial, data.apply_mask, unit_vel_scale;
                ndrange=K_padded)

    # Iterative correction: re-evaluate velocity residual and solve again
    if n_iter_velocity > 0
        corr_kern! = lincs_fused_velocity_correction_kernel!(backend, block_size)
        for _ in 1:n_iter_velocity
            launch_kernel!(corr_kern!, delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                       velocities,
                       data.atom1, data.atom2, data.invmass, data.sdiag,
                       coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                       coupling.max_coupled, data.n_rec, ws.factor_sum,
                       context.needs_virial, data.apply_mask, unit_vel_scale;
                       ndrange=K_padded)
        end
    end

    accumulate_lincs_velocity_virial_gpu!(data, ws, context, backend, block_size)
    return velocities
end

# --- Molly interface ---

function apply_position_constraints!(sys::System, ca::LINCS, r_pre_unconstrained_update;
                                     context=nothing,
                                     n_threads::Integer=Threads.nthreads(), kwargs...)
    context = isnothing(context) ? default_position_constraint_context() : context
    if !isnothing(ca.delta_buf)
        lincs_apply_gpu!(sys.coords, r_pre_unconstrained_update,
                         ca.lincs_data, ca.workspace, sys.boundary,
                         ca.delta_buf, ca.constrained_atoms, ca.gpu_block_size,
                         context)
    else
        lincs_apply!(sys.coords, r_pre_unconstrained_update,
                     ca.lincs_data, ca.workspace, sys.boundary, context, n_threads)
    end
    return sys
end

function apply_velocity_constraints!(sys::System, ca::LINCS; context=nothing,
                                     n_threads::Integer=Threads.nthreads(), kwargs...)
    context = (isnothing(context) ? default_velocity_constraint_context() : context)
    n_iter_velocity = (ca.iter_vel_correction ? ca.lincs_data.n_iter : 0)
    if !isnothing(ca.delta_buf)
        lincs_vel_apply_gpu!(sys.velocities, sys.coords,
                             ca.lincs_data, ca.workspace, sys.boundary,
                             ca.delta_buf, ca.constrained_atoms, ca.gpu_block_size, context,
                             n_iter_velocity)
    else
        lincs_vel_apply!(sys.velocities, sys.coords,
                         ca.lincs_data, ca.workspace, sys.boundary, context,
                         n_iter_velocity, n_threads)
    end
    return sys
end

function check_position_constraints(sys::System{<:Any, <:Any, FT}, ca::LINCS) where FT
    err_unit = unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.dist_tolerance)
        throw(ArgumentError("distance tolerance units in LINCS ($(unit(ca.dist_tolerance))) " *
                            "are inconsistent with system coordinate units ($err_unit)"))
    end

    max_err = typemin(FT)
    for dc in ca.dist_constraints
        dr = vector(sys.coords[dc.i], sys.coords[dc.j], sys.boundary)
        r = sqrt(sum(abs2, dr))
        err = ustrip(abs(r - dc.dist))
        max_err = max(err, max_err)
    end
    return max_err < ustrip(ca.dist_tolerance)
end

function check_position_constraints(sys::System{<:Any, <:AbstractGPUArray, FT}, ca::LINCS) where FT
    err_unit = unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.dist_tolerance)
        throw(ArgumentError("distance tolerance units in LINCS ($(unit(ca.dist_tolerance))) " *
                            "are inconsistent with system coordinate units ($err_unit)"))
    end

    # Use CPU dist_constraints to avoid issues with padded GPU arrays
    unit_len = oneunit(eltype(eltype(sys.coords)))
    coords_cpu = Array(sys.coords)
    max_err = typemin(FT)
    for dc in ca.dist_constraints
        dr = vector(coords_cpu[dc.i], coords_cpu[dc.j], sys.boundary)
        r = sqrt(sum(abs2, dr))
        err = ustrip(abs(r - dc.dist))
        max_err = max(err, max_err)
    end
    return max_err < ustrip(ca.dist_tolerance)
end

function check_velocity_constraints(sys::System{<:Any, <:Any, FT}, ca::LINCS) where FT
    err_unit = unit(eltype(eltype(sys.velocities))) * unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.vel_tolerance)
        throw(ArgumentError("velocity tolerance units in LINCS ($(unit(ca.vel_tolerance))) " *
                            "are inconsistent with system velocity and coordinate units ($err_unit)"))
    end

    max_err = typemin(FT)
    for dc in ca.dist_constraints
        dr = vector(sys.coords[dc.i], sys.coords[dc.j], sys.boundary)
        v_diff = sys.velocities[dc.j] .- sys.velocities[dc.i]
        err = ustrip(abs(dot(dr, v_diff)))
        max_err = max(err, max_err)
    end
    return max_err < ustrip(ca.vel_tolerance)
end

function check_velocity_constraints(sys::System{<:Any, <:AbstractGPUArray, FT}, ca::LINCS) where FT
    err_unit = unit(eltype(eltype(sys.velocities))) * unit(eltype(eltype(sys.coords)))
    if err_unit != unit(ca.vel_tolerance)
        throw(ArgumentError("velocity tolerance units in LINCS ($(unit(ca.vel_tolerance))) " *
                            "are inconsistent with system velocity and coordinate units ($err_unit)"))
    end

    # Use CPU dist_constraints to avoid issues with padded GPU arrays
    coords_cpu = Array(sys.coords)
    vels_cpu = Array(sys.velocities)
    max_err = typemin(FT)
    for dc in ca.dist_constraints
        dr = vector(coords_cpu[dc.i], coords_cpu[dc.j], sys.boundary)
        v_diff = vels_cpu[dc.j] .- vels_cpu[dc.i]
        err = ustrip(abs(dot(dr, v_diff)))
        max_err = max(err, max_err)
    end
    return max_err < ustrip(ca.vel_tolerance)
end

# --- GPU data transfer ---

function move_lincs_to_gpu(data::LincsData, ws, arr_type, n_atoms, block_size)
    T = eltype(data.lengths)
    K = length(data.atom1)

    # Group coupled constraints into thread blocks for cache-friendly atomics
    perm = group_constraints_for_gpu(data.atom1, data.atom2, block_size)
    K_padded = length(perm)

    # Reorder + pad constraint arrays according to grouping
    atom1_padded = Vector{Int}(undef, K_padded)
    atom2_padded = Vector{Int}(undef, K_padded)
    lengths_padded = Vector{T}(undef, K_padded)
    sdiag_padded = Vector{T}(undef, K_padded)

    for (new_i, old_c) in enumerate(perm)
        if old_c != 0
            atom1_padded[new_i] = data.atom1[old_c]
            atom2_padded[new_i] = data.atom2[old_c]
            lengths_padded[new_i] = data.lengths[old_c]
            sdiag_padded[new_i] = data.sdiag[old_c]
        else
            # Dummy: valid atom indices, zero sdiag ensures no effect
            atom1_padded[new_i] = 1
            atom2_padded[new_i] = min(2, n_atoms)
            lengths_padded[new_i] = zero(T)
            sdiag_padded[new_i] = zero(T)
        end
    end

    # Build dense coupling layout
    dense_coupling = build_dense_coupling(perm, data.coupling, T)

    # Each constrained atom is applied by the first constraint it is in, see
    #   `lincs_apply_owned!`, which is in the same workgroup as every other constraint it
    #   is in. The dummy constraints apply nothing.
    apply_mask = zeros(UInt8, K_padded)
    owned = falses(n_atoms)
    for (new_i, old_c) in enumerate(perm)
        old_c == 0 && continue
        for (bit, a) in ((0x01, atom1_padded[new_i]), (0x02, atom2_padded[new_i]))
            if !owned[a]
                owned[a] = true
                apply_mask[new_i] |= bit
            end
        end
    end

    # Transfer to GPU
    # 32-bit indices halve the index data each kernel reads
    atom1_gpu = arr_type(Int32.(atom1_padded))
    atom2_gpu = arr_type(Int32.(atom2_padded))
    lengths_gpu = arr_type(lengths_padded)
    invmass_gpu = arr_type(data.invmass)
    sdiag_gpu = arr_type(sdiag_padded)

    coupled_indices_gpu = arr_type(dense_coupling.coupled_indices)
    coupled_coef_gpu = arr_type(dense_coupling.coupled_coef)
    n_coupled_gpu = arr_type(Int32.(dense_coupling.n_coupled))
    coupling_gpu = LincsCouplingDense(coupled_indices_gpu, coupled_coef_gpu,
                                      n_coupled_gpu, dense_coupling.max_coupled)

    # The GPU path reorders and pads the constraints, so the CPU atom mapping does not
    #   apply and the atoms of each workgroup are used instead
    wg_atoms = build_lincs_workgroup_atoms(perm, atom1_padded, atom2_padded, apply_mask,
                                           dense_coupling.coupled_indices,
                                           dense_coupling.n_coupled, block_size)
    wg_atoms_gpu = LincsWorkgroupAtoms(arr_type(wg_atoms.local_atoms),
                                       arr_type(wg_atoms.gather),
                                       arr_type(wg_atoms.coupled_local), wg_atoms.max_deg)
    data_gpu = LincsData(atom1_gpu, atom2_gpu, lengths_gpu, invmass_gpu,
                         sdiag_gpu, coupling_gpu, data.n_rec, data.n_iter, wg_atoms_gpu,
                         arr_type(apply_mask))

    # Workspace sized for padded constraint count
    backend = get_backend(atom1_gpu)
    B_gpu = KernelAbstractions.zeros(backend, SVector{3, T}, K_padded)
    rhs_gpu = KernelAbstractions.zeros(backend, T, K_padded)
    sol_gpu = KernelAbstractions.zeros(backend, T, K_padded)
    tmp_gpu = KernelAbstractions.zeros(backend, T, K_padded)
    blcc_gpu = KernelAbstractions.zeros(backend, T, 1)  # unused in fused kernels
    factor_sum_gpu = KernelAbstractions.zeros(backend, T, K_padded)
    ws_gpu = LincsWorkspace(B_gpu, rhs_gpu, sol_gpu, tmp_gpu, blcc_gpu,
                            factor_sum_gpu)

    delta_buf = KernelAbstractions.zeros(backend, T, 3, n_atoms)

    return data_gpu, ws_gpu, delta_buf
end

function setup_constraints!(lincs::LINCS, neighbor_finder, arr_type)
    if !isnothing(neighbor_finder)
        disable_constrained_interactions!(neighbor_finder, lincs.clusters)
    end

    return move_constraints_to_device(lincs, arr_type)
end

function move_constraints_to_device(lincs::LINCS, ::Type{AT}) where {AT <: AbstractGPUArray}
    lincs.lincs_data.atom1 isa AT && return lincs

    n_atoms = length(lincs.lincs_data.invmass)
    data_gpu, ws_gpu, delta_buf = move_lincs_to_gpu(
        lincs.lincs_data, lincs.workspace, AT, n_atoms, lincs.gpu_block_size)

    ca_indices = sort!(unique!(vcat(lincs.lincs_data.atom1, lincs.lincs_data.atom2)))
    ca_gpu = AT(ca_indices)

    clusters_gpu = replace_storage(AT, lincs.clusters)

    return LINCS(clusters_gpu, data_gpu, ws_gpu, lincs.dist_constraints,
                 lincs.angle_constraints, lincs.dist_tolerance, lincs.vel_tolerance,
                 lincs.iter_vel_correction, lincs.gpu_block_size, delta_buf, ca_gpu)
end

# Conversion to another float type, units or device, see `System(sys; ...)`
function convert_component(c::SystemConversion, lincs::LINCS)
    # The distance constraints from the angle constraints are added at the end
    n_angle = (isnothing(lincs.angle_constraints) ? 0 : length(lincs.angle_constraints))
    n_dist = length(lincs.dist_constraints) - 3 * n_angle
    dist_constraints = [convert_component(c, dc) for dc in lincs.dist_constraints[1:n_dist]]
    if n_angle > 0
        angle_constraints = [convert_component(c, ac) for ac in lincs.angle_constraints]
    else
        angle_constraints = nothing
    end
    return LINCS(
        masses=mass.(from_device(c.atoms)),
        dist_tolerance=convert_component(c, lincs.dist_tolerance),
        vel_tolerance=convert_component(c, lincs.vel_tolerance),
        dist_constraints=dist_constraints,
        angle_constraints=angle_constraints,
        n_rec=lincs.lincs_data.n_rec,
        n_iter=lincs.lincs_data.n_iter,
        iter_vel_correction=lincs.iter_vel_correction,
        gpu_block_size=lincs.gpu_block_size,
    )
end
