export LINCS

# Internal types for LINCS algorithm

struct LincsCouplingMatrix{R, N, C}
    range::R       # length K+1, row pointers (CSR format)
    neighbors::N   # coupled constraint indices
    coef::C        # mass-weighted coupling coefficients
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
          gpu_block_size=128)

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
                 Int(gpu_block_size), nothing, nothing)
end

function Base.show(io::IO, lincs::LINCS)
    n_dc = length(lincs.dist_constraints)
    n_ac = isnothing(lincs.angle_constraints) ? 0 : length(lincs.angle_constraints)
    print(io, "LINCS with ", n_dc, " distance and ", n_ac, " angle constraints (nrec=",
          lincs.lincs_data.nrec, ", niter=", lincs.lincs_data.niter, ")")
end

function constrained_atom_inds(lincs::LINCS)
    atom_inds = Int[]
    for dc in lincs.dist_constraints
        push!(atom_inds, dc.i, dc.j)
    end
    return atom_inds
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

# --- Core algorithm ---

@inline function lincs_bond_vector(coords, a1, a2, boundary)
    return ustrip.(vector(coords[a2], coords[a1], boundary))
end

@inline function lincs_bond_vector(coords, a1, a2, ::Nothing)
    return ustrip.(coords[a1] - coords[a2])
end

# --- GPU kernels ---

@kernel inbounds=true function lincs_bond_vectors_and_rhs_kernel!(
        B, rhs, sol,
        @Const(coords), @Const(old_coords),
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(sdiag),
        boundary)
    i = @index(Global, Linear)
    if i <= length(atom1)
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
    end
end

@kernel inbounds=true function lincs_blcc_kernel!(
        blcc,
        @Const(B),
        @Const(csr_range), @Const(csr_neighbors), @Const(csr_coef))
    i = @index(Global, Linear)
    if i <= length(csr_range) - 1
        for n in csr_range[i]:(csr_range[i+1] - 1)
            j = csr_neighbors[n]
            blcc[n] = csr_coef[n] * dot(B[i], B[j])
        end
    end
end

@kernel inbounds=true function lincs_spmv_kernel!(
        sol, dst,
        @Const(src), @Const(blcc),
        @Const(csr_range), @Const(csr_neighbors))
    i = @index(Global, Linear)
    @uniform T = eltype(sol)
    if i <= length(sol)
        mvb = zero(T)
        for n in csr_range[i]:(csr_range[i+1] - 1)
            mvb += blcc[n] * src[csr_neighbors[n]]
        end
        dst[i] = mvb
        sol[i] += mvb
    end
end

@kernel inbounds=true function lincs_position_scatter_kernel!(
        delta_buf,
        @Const(B), @Const(sol), @Const(sdiag),
        @Const(atom1), @Const(atom2), @Const(invmass))
    i = @index(Global, Linear)
    if i <= length(atom1)
        a1, a2 = atom1[i], atom2[i]
        factor = sdiag[i] * sol[i]
        delta = B[i] * factor
        for dim in 1:3
            d = delta[dim]
            Atomix.@atomic delta_buf[dim, a1] -= invmass[a1] * d
            Atomix.@atomic delta_buf[dim, a2] += invmass[a2] * d
        end
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

@kernel inbounds=true function lincs_correction_rhs_kernel!(
        rhs, sol,
        @Const(coords),
        @Const(atom1), @Const(atom2), @Const(lengths), @Const(sdiag),
        boundary)
    i = @index(Global, Linear)
    @uniform T = eltype(lengths)
    if i <= length(atom1)
        a1, a2 = atom1[i], atom2[i]
        diff = lincs_bond_vector(coords, a1, a2, boundary)
        dlen2 = 2 * lengths[i]^2 - dot(diff, diff)
        p = sqrt(max(dlen2, zero(T)))
        val = sdiag[i] * (lengths[i] - p)
        rhs[i] = val
        sol[i] = val
    end
end

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
            p = sqrt(max(dlen2, zero(T)))
            ws.rhs[i] = data.sdiag[i] * (data.lengths[i] - p)
        end
        copyto!(ws.sol, ws.rhs)
        lincs_solve!(coords, data, ws, unit_scale)
    end

    return coords
end

# --- GPU solve path ---

function lincs_gpu_solve!(coords, data, ws, delta_buf, constrained_atoms,
                          block_size, backend, unit_scale)
    K = length(data.atom1)
    coupling = data.coupling
    n_ca = length(constrained_atoms)

    # Coupling coefficients
    blcc_kern! = lincs_blcc_kernel!(backend, block_size)
    blcc_kern!(ws.blcc, ws.B, coupling.range, coupling.neighbors, coupling.coef;
               ndrange=K)

    # SpMV iterations (ping-pong between rhs and tmp)
    spmv_kern! = lincs_spmv_kernel!(backend, block_size)
    src, dst = ws.rhs, ws.tmp
    for _ in 1:data.nrec
        spmv_kern!(ws.sol, dst, src, ws.blcc, coupling.range, coupling.neighbors;
                   ndrange=K)
        src, dst = dst, src
    end

    # Position scatter with atomics
    scatter_kern! = lincs_position_scatter_kernel!(backend, block_size)
    scatter_kern!(delta_buf, ws.B, ws.sol, data.sdiag, data.atom1, data.atom2, data.invmass;
                  ndrange=K)

    # Apply accumulated deltas
    apply_kern! = lincs_apply_deltas_kernel!(backend, block_size)
    apply_kern!(coords, delta_buf, constrained_atoms, unit_scale;
                ndrange=n_ca)
end

function lincs_apply_gpu!(coords, old_coords, data, ws, boundary,
                          delta_buf, constrained_atoms, block_size)
    K = length(data.atom1)
    backend = get_backend(coords)
    unit_scale = oneunit(eltype(eltype(coords)))

    # Bond vectors + initial RHS (fused)
    bv_kern! = lincs_bond_vectors_and_rhs_kernel!(backend, block_size)
    bv_kern!(ws.B, ws.rhs, ws.sol, coords, old_coords,
             data.atom1, data.atom2, data.lengths, data.sdiag, boundary;
             ndrange=K)

    lincs_gpu_solve!(coords, data, ws, delta_buf, constrained_atoms,
                     block_size, backend, unit_scale)

    # Outer correction iterations (rotational lengthening)
    corr_kern! = lincs_correction_rhs_kernel!(backend, block_size)
    for _ in 1:data.niter
        corr_kern!(ws.rhs, ws.sol, coords,
                   data.atom1, data.atom2, data.lengths, data.sdiag, boundary;
                   ndrange=K)

        lincs_gpu_solve!(coords, data, ws, delta_buf, constrained_atoms,
                         block_size, backend, unit_scale)
    end

    KernelAbstractions.synchronize(backend)
    return coords
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
    # LINCS handles velocities implicitly through position constraint correction
    # via the vel_storage mechanism in the combined apply_position_constraints! method.
    # No separate velocity projection (RATTLE) step is needed.
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

    # Use GPU lincs_data from the system to avoid scalar GPU array indexing
    gpu_ca = only(c for c in sys.constraints if c isa LINCS)
    data = gpu_ca.lincs_data
    unit_len = oneunit(eltype(eltype(sys.coords)))

    coords_a1 = sys.coords[data.atom1]
    coords_a2 = sys.coords[data.atom2]
    target_lengths = data.lengths .* unit_len
    dr_norms = norm.(vector.(coords_a1, coords_a2, (sys.boundary,)))
    max_err = maximum(ustrip.(abs.(dr_norms .- target_lengths)))

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

function move_lincs_to_gpu(data::LincsData, ws, arr_type, n_atoms)
    T = eltype(data.lengths)
    K = length(data.atom1)

    atom1_gpu = arr_type(data.atom1)
    atom2_gpu = arr_type(data.atom2)
    lengths_gpu = arr_type(data.lengths)
    invmass_gpu = arr_type(data.invmass)
    sdiag_gpu = arr_type(data.sdiag)
    range_gpu = arr_type(data.coupling.range)
    neighbors_gpu = arr_type(data.coupling.neighbors)
    coef_gpu = arr_type(data.coupling.coef)
    coupling_gpu = LincsCouplingMatrix(range_gpu, neighbors_gpu, coef_gpu)

    data_gpu = LincsData(atom1_gpu, atom2_gpu, lengths_gpu, invmass_gpu,
                         sdiag_gpu, coupling_gpu, data.nrec, data.niter)

    ncc = length(data.coupling.neighbors)
    B_gpu = KernelAbstractions.zeros(get_backend(atom1_gpu), SVector{3, T}, K)
    rhs_gpu = KernelAbstractions.zeros(get_backend(atom1_gpu), T, K)
    sol_gpu = KernelAbstractions.zeros(get_backend(atom1_gpu), T, K)
    tmp_gpu = KernelAbstractions.zeros(get_backend(atom1_gpu), T, K)
    blcc_gpu = KernelAbstractions.zeros(get_backend(atom1_gpu), T, ncc)
    ws_gpu = LincsWorkspace(B_gpu, rhs_gpu, sol_gpu, tmp_gpu, blcc_gpu)

    delta_buf = KernelAbstractions.zeros(get_backend(atom1_gpu), T, 3, n_atoms)

    return data_gpu, ws_gpu, delta_buf
end

function setup_constraints!(lincs::LINCS, neighbor_finder, arr_type)
    if !(neighbor_finder isa NoNeighborFinder)
        disable_constrained_interactions!(neighbor_finder, lincs.clusters)
    end

    if arr_type <: AbstractGPUArray
        n_atoms = length(lincs.lincs_data.invmass)
        data_gpu, ws_gpu, delta_buf = move_lincs_to_gpu(
            lincs.lincs_data, lincs.workspace, arr_type, n_atoms)

        ca_indices = sort!(unique!(vcat(lincs.lincs_data.atom1, lincs.lincs_data.atom2)))
        ca_gpu = arr_type(ca_indices)

        clusters_gpu = replace_storage(arr_type, lincs.clusters)

        lincs = LINCS(clusters_gpu, data_gpu, ws_gpu, lincs.dist_constraints,
                      lincs.angle_constraints, lincs.dist_tolerance, lincs.vel_tolerance,
                      lincs.gpu_block_size, delta_buf, ca_gpu)
    end

    return lincs
end
