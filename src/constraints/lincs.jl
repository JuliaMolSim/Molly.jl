export LINCS

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

struct LincsData{A1, A2, L, IM, SD, CM}
    atom1::A1
    atom2::A2
    lengths::L
    invmass::IM
    sdiag::SD
    coupling::CM
    nrec::Int
    niter::Int
end

struct LincsWorkspace{BV, R, S, TM, BL}
    B::BV
    rhs::R
    sol::S
    tmp::TM
    blcc::BL
end

"""
    LINCS(masses, dist_tolerance=1e-8u"nm", vel_tolerance=1e-8u"nm^2 * ps^-1";
          dist_constraints=nothing, angle_constraints=nothing, nrec=4, niter=1,
          iter_vel_correction=false, gpu_block_size=128)

Constrain bond distances during a simulation using the LINCS (LINear Constraint Solver)
algorithm.

LINCS is a non-iterative constraint algorithm that uses matrix expansion to approximate
the inverse of the constraint coupling matrix. It is typically faster than
[`SHAKE_RATTLE`](@ref) for large systems but is approximate for ring topologies.

Velocity constraints are applied implicitly through position constraint correction.
See [Hess et al. 1997](https://doi.org/10.1002/(SICI)1096-987X(199709)18:12<1463::AID-JCC4>3.0.CO;2-H)
for the original LINCS paper.

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
- `nrec=4`: order of the matrix expansion for coupling matrix inversion. Higher values
    improve accuracy for coupled constraints at the cost of performance.
- `niter=1`: number of outer correction iterations for rotational lengthening. Higher
    values improve accuracy for strongly perturbed bonds.
- `iter_vel_correction=false`: whether to use iterative velocity constraint solving.
    When `false` (the default), velocity correction uses the simple one-step approach
    `v += Δx/dt` as in GROMACS (for the Verlet simulator only, otherwise velocities are
    not constrained). When `true`, a full iterative velocity constraint projection
    is performed.
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

        for (idx2, ac2) in enumerate(angle_constraints)
            idx2 == idx && continue
            ac2_atoms = (ac2.i, ac2.j, ac2.k)
            for a in ac_atoms
                if a in ac2_atoms
                    throw(ArgumentError(
                        "angle constraint $idx (atoms $(ac.i)/$(ac.j)/$(ac.k)) shares " *
                        "atom $a with angle constraint $idx2 (atoms $(ac2.i)/$(ac2.j)/$(ac2.k)); " *
                        "LINCS requires angle constraints to be isolated"))
                end
            end
        end
    end
end

function LINCS(;masses,
               dist_tolerance=1e-8u"nm",
               vel_tolerance=1e-8u"nm^2 * ps^-1",
               dist_constraints=nothing,
               angle_constraints=nothing,
               nrec::Integer=4,
               niter::Integer=1,
               iter_vel_correction::Bool=false,
               gpu_block_size::Integer=128)
    ustrip(dist_tolerance) > 0 || throw(ArgumentError("dist_tolerance must be greater than zero"))
    ustrip(vel_tolerance)  > 0 || throw(ArgumentError("vel_tolerance must be greater than zero"))

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

    lincs_data = build_lincs_data(all_dist_constraints, masses; nrec=Int(nrec), niter=Int(niter))
    workspace = create_lincs_workspace(lincs_data)

    stored_angle_constraints = ac_present ? collect(angle_constraints) : nothing

    return LINCS(clusters, lincs_data, workspace, all_dist_constraints,
                 stored_angle_constraints, dist_tolerance, vel_tolerance,
                 iter_vel_correction, Int(gpu_block_size), nothing, nothing)
end

function Base.show(io::IO, lincs::LINCS)
    n_ac = isnothing(lincs.angle_constraints) ? 0 : length(lincs.angle_constraints)
    n_dc = length(lincs.dist_constraints) - 3 * n_ac # avoid counting angle constraints as distance constraints
    print(io, "LINCS with ", n_dc, " distance and ", n_ac, " angle constraints (nrec=",
          lincs.lincs_data.nrec, ", niter=", lincs.lincs_data.niter,
          ", iter_vel_correction=", lincs.iter_vel_correction, ")")
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
                          nrec::Int=4, niter::Int=1)
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
    return LincsData(atom1, atom2, lengths, invmass, sdiag, coupling, nrec, niter)
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
    return LincsWorkspace(B, rhs, sol, tmp, blcc)
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
                "LINCS: connected component of $(length(comp)) coupled constraints exceeds " *
                "gpu_block_size=$block_size. Increase gpu_block_size in the LINCS constructor " *
                "to at least $(length(comp)), or use CPU constraints for this system.",
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
# These fused kernels rely on @synchronize (workgroup barrier) between matrix expansion
# iterations. Correctness requires that all coupled constraints within a connected
# component reside in the same workgroup. This is ensured by group_constraints_for_gpu,
# which packs connected components into blocks, and by passing gpu_block_size as the
# workgroup size to the KernelAbstractions kernel constructor.

@kernel inbounds=true function lincs_fused_position_kernel!(
        delta_buf,
        B, rhs, sol, tmp,
        @Const(coords), @Const(old_coords),
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, nrec, boundary)
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

    @synchronize

    nc = n_coupled_arr[i]
    for rec in 1:nrec
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
        @synchronize
    end

    factor = sdiag[i] * sol[i]
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end
end

@kernel inbounds=true function lincs_fused_correction_kernel!(
        delta_buf,
        @Const(B), rhs, sol, tmp,
        @Const(coords),
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, nrec, boundary)
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

    @synchronize

    nc = n_coupled_arr[i]
    for rec in 1:nrec
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
        @synchronize
    end

    factor = sdiag[i] * sol[i]
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end
end

@kernel inbounds=true function lincs_fused_velocity_kernel!(
        delta_buf,
        B, rhs, sol, tmp,
        @Const(coords), @Const(velocities),
        @Const(atom1), @Const(atom2), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, nrec, boundary)
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

    @synchronize

    nc = n_coupled_arr[i]
    for rec in 1:nrec
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
        @synchronize
    end

    factor = sdiag[i] * sol[i]
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end
end

@kernel inbounds=true function lincs_fused_velocity_correction_kernel!(
        delta_buf,
        @Const(B), rhs, sol, tmp,
        @Const(velocities),
        @Const(atom1), @Const(atom2), @Const(invmass), @Const(sdiag),
        @Const(coupled_indices), @Const(coupled_coef), @Const(n_coupled_arr),
        max_coupled, nrec)
    i = @index(Global, Linear)
    @uniform T = eltype(sdiag)

    a1, a2 = atom1[i], atom2[i]

    B_i = B[i]
    dv = ustrip.(velocities[a2] - velocities[a1])
    val = -sdiag[i] * dot(B_i, dv)
    rhs[i] = val
    sol[i] = val

    @synchronize

    nc = n_coupled_arr[i]
    for rec in 1:nrec
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
        @synchronize
    end

    factor = sdiag[i] * sol[i]
    delta = B_i * factor
    for dim in 1:3
        d = delta[dim]
        Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
        Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
    end
end

@kernel inbounds=true function lincs_apply_deltas_kernel!(
        coords, delta_buf,
        @Const(constrained_atoms), unit_scale)
    idx = @index(Global, Linear)
    if idx <= length(constrained_atoms)
        a = constrained_atoms[idx]
        coords[a] += SVector(delta_buf[1, a], delta_buf[2, a], delta_buf[3, a]) .* unit_scale
        delta_buf[1, a] = zero(eltype(delta_buf))
        delta_buf[2, a] = zero(eltype(delta_buf))
        delta_buf[3, a] = zero(eltype(delta_buf))
    end
end

# --- CPU solve path ---

function lincs_solve!(coords, data::LincsData, ws::LincsWorkspace, unit_scale)
    T = eltype(data.lengths)
    coupling = data.coupling
    rhs = ws.rhs
    tmp = ws.tmp

    # Matrix expansion: nrec iterations
    for rec in 1:data.nrec
        @inbounds for i in eachindex(data.atom1)
            mvb = zero(T)
            for n in coupling.range[i]:(coupling.range[i+1] - 1)
                mvb += ws.blcc[n] * rhs[coupling.neighbors[n]]
            end
            tmp[i] = mvb
            ws.sol[i] += mvb
        end
        # Ping-pong: swap local bindings so next iteration reads from what was just written.
        # ws.rhs/ws.tmp fields still point to original arrays; callers reassign ws.rhs before reuse.
        rhs, tmp = tmp, rhs
    end

    # Position update
    @inbounds for i in eachindex(data.atom1)
        a1, a2 = data.atom1[i], data.atom2[i]
        factor = data.sdiag[i] * ws.sol[i]
        delta = ws.B[i] * factor
        coords[a1] -= (data.invmass[a1] * delta) .* unit_scale
        coords[a2] += (data.invmass[a2] * delta) .* unit_scale
    end
end

function lincs_apply!(coords, old_coords, data::LincsData, ws::LincsWorkspace,
                      boundary)
    T = eltype(data.lengths)
    K = length(data.atom1)
    coupling = data.coupling

    unit_scale = oneunit(eltype(eltype(coords)))

    # Compute unit bond vectors and initial RHS
    @inbounds for i in 1:K
        diff_old = lincs_bond_vector(old_coords, data.atom1[i], data.atom2[i], boundary)
        inv_len = inv(sqrt(dot(diff_old, diff_old)))
        B_i = diff_old * inv_len
        ws.B[i] = B_i

        diff_new = lincs_bond_vector(coords, data.atom1[i], data.atom2[i], boundary)
        proj = dot(B_i, diff_new)
        ws.rhs[i] = data.sdiag[i] * (proj - data.lengths[i])
    end

    # Compute runtime coupling coefficients: blcc = coef * dot(B[i], B[neighbor])
    @inbounds for i in 1:K
        for n in coupling.range[i]:(coupling.range[i+1] - 1)
            j = coupling.neighbors[n]
            ws.blcc[n] = coupling.coef[n] * dot(ws.B[i], ws.B[j])
        end
    end

    copyto!(ws.sol, ws.rhs)
    lincs_solve!(coords, data, ws, unit_scale)

    # Outer correction iterations (rotational lengthening)
    for _ in 1:data.niter
        @inbounds for i in 1:K
            diff = lincs_bond_vector(coords, data.atom1[i], data.atom2[i], boundary)
            dlen2 = 2 * data.lengths[i]^2 - dot(diff, diff)
            if dlen2 < zero(T)
                @warn "LINCS correction: bond $(data.atom1[i])-$(data.atom2[i]) stretched " *
                      "beyond sqrt(2) * target length, constraint may be unreliable" maxlog=1
            end
            p = sqrt(max(dlen2, zero(T)))
            ws.rhs[i] = data.sdiag[i] * (data.lengths[i] - p)
        end
        copyto!(ws.sol, ws.rhs)
        lincs_solve!(coords, data, ws, unit_scale)
    end

    return coords
end

function lincs_vel_apply!(velocities, coords, data::LincsData, ws::LincsWorkspace, boundary)
    K = length(data.atom1)
    coupling = data.coupling
    unit_vel_scale = oneunit(eltype(eltype(velocities)))

    # Bond vectors from current (constrained) coords + velocity RHS
    @inbounds for i in 1:K
        a1, a2 = data.atom1[i], data.atom2[i]
        diff = lincs_bond_vector(coords, a1, a2, boundary)
        inv_len = inv(sqrt(dot(diff, diff)))
        ws.B[i] = diff * inv_len
        dv = ustrip.(velocities[a2] - velocities[a1])
        ws.rhs[i] = -data.sdiag[i] * dot(ws.B[i], dv)
    end

    # Recompute coupling coefficients using current B vectors
    @inbounds for i in 1:K
        for n in coupling.range[i]:(coupling.range[i+1] - 1)
            ws.blcc[n] = coupling.coef[n] * dot(ws.B[i], ws.B[coupling.neighbors[n]])
        end
    end

    copyto!(ws.sol, ws.rhs)
    lincs_solve!(velocities, data, ws, unit_vel_scale)

    # Iterative correction: re-evaluate velocity residual and solve again
    for _ in 1:data.niter
        @inbounds for i in 1:K
            a1, a2 = data.atom1[i], data.atom2[i]
            dv = ustrip.(velocities[a2] - velocities[a1])
            ws.rhs[i] = -data.sdiag[i] * dot(ws.B[i], dv)
        end
        copyto!(ws.sol, ws.rhs)
        lincs_solve!(velocities, data, ws, unit_vel_scale)
    end
end

# --- GPU solve path ---

function lincs_apply_gpu!(coords, old_coords, data, ws, boundary,
                          delta_buf, constrained_atoms, block_size)
    K_padded = length(data.atom1)
    backend = get_backend(coords)
    unit_scale = oneunit(eltype(eltype(coords)))
    n_ca = length(constrained_atoms)
    coupling = data.coupling

    # Fused solve: bond vectors + blcc + SpMV iterations + scatter
    fused_kern! = lincs_fused_position_kernel!(backend, block_size)
    fused_kern!(delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                coords, old_coords,
                data.atom1, data.atom2, data.lengths, data.invmass, data.sdiag,
                coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                coupling.max_coupled, data.nrec, boundary;
                ndrange=K_padded)

    apply_kern! = lincs_apply_deltas_kernel!(backend, block_size)
    apply_kern!(coords, delta_buf, constrained_atoms, unit_scale;
                ndrange=n_ca)

    # Correction iterations (rotational lengthening)
    for _ in 1:data.niter
        corr_kern! = lincs_fused_correction_kernel!(backend, block_size)
        corr_kern!(delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                   coords,
                   data.atom1, data.atom2, data.lengths, data.invmass, data.sdiag,
                   coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                   coupling.max_coupled, data.nrec, boundary;
                   ndrange=K_padded)
        apply_kern!(coords, delta_buf, constrained_atoms, unit_scale;
                    ndrange=n_ca)
    end

    KernelAbstractions.synchronize(backend)
    return coords
end

function lincs_vel_apply_gpu!(velocities, coords, data, ws, boundary,
                               delta_buf, constrained_atoms, block_size)
    K_padded = length(data.atom1)
    backend = get_backend(velocities)
    unit_vel_scale = oneunit(eltype(eltype(velocities)))
    n_ca = length(constrained_atoms)
    coupling = data.coupling

    fused_kern! = lincs_fused_velocity_kernel!(backend, block_size)
    fused_kern!(delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                coords, velocities,
                data.atom1, data.atom2, data.invmass, data.sdiag,
                coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                coupling.max_coupled, data.nrec, boundary;
                ndrange=K_padded)

    apply_kern! = lincs_apply_deltas_kernel!(backend, block_size)
    apply_kern!(velocities, delta_buf, constrained_atoms, unit_vel_scale;
                ndrange=n_ca)

    # Iterative correction: re-evaluate velocity residual and solve again
    for _ in 1:data.niter
        corr_kern! = lincs_fused_velocity_correction_kernel!(backend, block_size)
        corr_kern!(delta_buf, ws.B, ws.rhs, ws.sol, ws.tmp,
                   velocities,
                   data.atom1, data.atom2, data.invmass, data.sdiag,
                   coupling.coupled_indices, coupling.coupled_coef, coupling.n_coupled,
                   coupling.max_coupled, data.nrec;
                   ndrange=K_padded)
        apply_kern!(velocities, delta_buf, constrained_atoms, unit_vel_scale;
                    ndrange=n_ca)
    end

    KernelAbstractions.synchronize(backend)
    return velocities
end

# --- Molly interface ---

function apply_position_constraints!(sys::System, ca::LINCS, r_pre_unconstrained_update;
                                     kwargs...)
    if !isnothing(ca.delta_buf)
        lincs_apply_gpu!(sys.coords, r_pre_unconstrained_update,
                         ca.lincs_data, ca.workspace, sys.boundary,
                         ca.delta_buf, ca.constrained_atoms, ca.gpu_block_size)
    else
        lincs_apply!(sys.coords, r_pre_unconstrained_update,
                     ca.lincs_data, ca.workspace, sys.boundary)
    end
    return sys
end

function apply_velocity_constraints!(sys::System, ca::LINCS; kwargs...)
    ca.iter_vel_correction || return sys
    if !isnothing(ca.delta_buf)
        lincs_vel_apply_gpu!(sys.velocities, sys.coords,
                             ca.lincs_data, ca.workspace, sys.boundary,
                             ca.delta_buf, ca.constrained_atoms, ca.gpu_block_size)
    else
        lincs_vel_apply!(sys.velocities, sys.coords,
                         ca.lincs_data, ca.workspace, sys.boundary)
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
        err = ustrip(abs(norm(dr) - dc.dist))
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
        err = ustrip(abs(norm(dr) - dc.dist))
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

    # Transfer to GPU
    atom1_gpu = arr_type(atom1_padded)
    atom2_gpu = arr_type(atom2_padded)
    lengths_gpu = arr_type(lengths_padded)
    invmass_gpu = arr_type(data.invmass)
    sdiag_gpu = arr_type(sdiag_padded)

    coupled_indices_gpu = arr_type(dense_coupling.coupled_indices)
    coupled_coef_gpu = arr_type(dense_coupling.coupled_coef)
    n_coupled_gpu = arr_type(dense_coupling.n_coupled)
    coupling_gpu = LincsCouplingDense(coupled_indices_gpu, coupled_coef_gpu,
                                      n_coupled_gpu, dense_coupling.max_coupled)

    data_gpu = LincsData(atom1_gpu, atom2_gpu, lengths_gpu, invmass_gpu,
                         sdiag_gpu, coupling_gpu, data.nrec, data.niter)

    # Workspace sized for padded constraint count
    backend = get_backend(atom1_gpu)
    B_gpu = KernelAbstractions.zeros(backend, SVector{3, T}, K_padded)
    rhs_gpu = KernelAbstractions.zeros(backend, T, K_padded)
    sol_gpu = KernelAbstractions.zeros(backend, T, K_padded)
    tmp_gpu = KernelAbstractions.zeros(backend, T, K_padded)
    blcc_gpu = KernelAbstractions.zeros(backend, T, 1)  # unused in fused kernels
    ws_gpu = LincsWorkspace(B_gpu, rhs_gpu, sol_gpu, tmp_gpu, blcc_gpu)

    delta_buf = KernelAbstractions.zeros(backend, T, 3, n_atoms)

    return data_gpu, ws_gpu, delta_buf
end

function setup_constraints!(lincs::LINCS, neighbor_finder, arr_type)
    if !(neighbor_finder isa NoNeighborFinder)
        disable_constrained_interactions!(neighbor_finder, lincs.clusters)
    end

    if arr_type <: AbstractGPUArray
        n_atoms = length(lincs.lincs_data.invmass)
        data_gpu, ws_gpu, delta_buf = move_lincs_to_gpu(
            lincs.lincs_data, lincs.workspace, arr_type, n_atoms, lincs.gpu_block_size)

        ca_indices = sort!(unique!(vcat(lincs.lincs_data.atom1, lincs.lincs_data.atom2)))
        ca_gpu = arr_type(ca_indices)

        clusters_gpu = replace_storage(arr_type, lincs.clusters)

        lincs = LINCS(clusters_gpu, data_gpu, ws_gpu, lincs.dist_constraints,
                      lincs.angle_constraints, lincs.dist_tolerance, lincs.vel_tolerance,
                      lincs.iter_vel_correction, lincs.gpu_block_size, delta_buf, ca_gpu)
    end

    return lincs
end
