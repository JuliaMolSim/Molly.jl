export
    EEM

@inline scalar_getindex(a::Array, i...) = getindex(a, i...)
scalar_getindex(a::AbstractGPUArray, i...) = @allowscalar getindex(a, i...)

@inline scalar_setindex!(a::Array, v, i...) = setindex!(a, v, i...)
scalar_setindex!(a::AbstractGPUArray, v, i...) = @allowscalar setindex!(a, v, i...)

# Should be implemented by all general interactions requiring charge equilibration
# Holds interaction-specific parameters which are used in eem_hij()
abstract type AbstractEEMCoupling end

# This function should be implemented by all general interactions that wish to use charge equilibration
# Must return the coulomb energy of an off-diagonal pair (i,j) in eV
# `r` can be larger than the interaction cutoff due to skin distance: should implement a check for this
function eem_hij end

# Specifies the units for the distance r passed to eem_hij by QEq solver
function eem_length_unit(coupling::AbstractEEMCoupling)
    return u"nm"
end

# Mutable since H_nonzero size can change
@kwdef mutable struct EEMState{VQ, VI}
    q::VQ
    q_prev_1::VQ
    q_prev_2::VQ
    q_prev_3::VQ
    q_prev_4::VQ
    q_backup::VQ
    history_count::Int      # number of valid converged charge states, including q, maximum 5
    last_history_step::Int  # last step_n used for charge prediction

    # for off-diagonal part of H, represent using CSR
    H_nonzero::VQ
    rowptr::VI          # rowptr[i] is where row i begins in colidx
    colidx::VI          # column indices of non zero entries in CSR matrix
    row_counts::VI      # number of entries in each row in CSR matrix
    write_ptr::VI       # mutable cursor when modifying CSR arrays
    pair_ptr::VI        # pair_ptr[2*i] and pair_ptr[2*i - 1] locate the entries in colidx for pair i 

    H_diag::VQ
    H_diag_inv::VQ
    r::VQ       # residual
    d::VQ       # search direction
    Hq::VQ      # H*q
    Hd::VQ      # H*d
    z::VQ       # projected preconditioned residual
    mχ::VQ      # -χ , shape=(n_atoms,)

    buffers::VQ     # holds reduction scalars
end


"""
    EEM(coupling, type_ids, χ, η; charges=nothing, tolerance=1e-6, max_iterations=200)

Electronegativity equalization method (EEM) for calculating partial charges on atoms
([Mortier et al. 1986](https://doi.org/10.1021/ja00275a013) and [Rappé and Goddard 1991](https://doi.org/10.1021/j100161a070)).

`type_ids` contains a one-based type index for each atom. `χ` and `η` contain the
electronegativity and hardness of each type in eV, with charges
expressed in elementary-charge units.

`coupling` supplies any required off-diagonal Coulomb coefficients through `Molly.eem_hij`.

`χ` and `η` must have the same floating point precision, which also sets the overall solver precision.
All input arrays must be on the same device already, and `charges`, if
provided, must use the same precision also. `charges` is the initial per-atom
charge vector and is used as mutable storage for the solver. Its sum also
sets the total charge that is preserved during charge equilibration.

`tolerance` controls the convergence of the projected preconditioned residual, scaled
by the norm of the electronegativities vector. If the solver fails to converge within
`max_iterations`, it raises an error and restores the charges from before the solve.

Include this calculator in a system's `general_inters` interactions block to equilibrate charges before any
pairwise and specific interactions are evaluated. Energy and force evaluations
write the converged charges from the solve into `sys.atoms`. EEM tallies the charge equilibration self-energy into
the global energy, but the corresponding electrostatic pair energy and forces are supplied by another
interaction. For [`ReaxFF`](@ref), use `EEM(reaxff, atoms_data; ...)`.
"""
@kwdef struct EEM{C, VI, VQ, QT}
    coupling::C
    n_atoms::Int
    type_ids::VI    # Vector of Ints, shape=(n_atoms,) 
    χ::VQ           # Vector of Floats with eV units, shape=(n_types,) 
    η::VQ           # Vector of Floats with eV units, shape=(n_types,) 
    tolerance::QT
    squared_tolerance::QT   # χ_norm = norm(eem.state.mχ); squared_tolerance = (eem.tolerance * χ_norm)^2
    max_iterations::Int
    state::EEMState{VQ, VI}
end

evaluate_before_pairwise(::EEM) = true

save_trial_state(eem::EEM) = (
    q=copy(eem.state.q),
    q_prev_1=copy(eem.state.q_prev_1),
    q_prev_2=copy(eem.state.q_prev_2),
    q_prev_3=copy(eem.state.q_prev_3),
    q_prev_4=copy(eem.state.q_prev_4),
    q_backup=copy(eem.state.q_backup),
    history_count=eem.state.history_count,
    last_history_step=eem.state.last_history_step,
)

function restore_trial_state!(sys, eem::EEM, snapshot::NamedTuple)
    eem.state.q .= snapshot.q
    eem.state.q_prev_1 .= snapshot.q_prev_1
    eem.state.q_prev_2 .= snapshot.q_prev_2
    eem.state.q_prev_3 .= snapshot.q_prev_3
    eem.state.q_prev_4 .= snapshot.q_prev_4
    eem.state.q_backup .= snapshot.q_backup
    eem.state.history_count = snapshot.history_count
    eem.state.last_history_step = snapshot.last_history_step
    write_charges!(sys, eem.state.q, get_backend(sys.coords))
    return nothing
end


# Type_ids, χ, γ, charges should already be on device
function EEM(
    coupling, 
    type_ids, 
    χ::AbstractVector{QT}, 
    η::AbstractVector{QT}; 
    charges=nothing,
    tolerance::Real=1e-6,
    max_iterations=200
) where {QT<:AbstractFloat}
    n_atoms = length(type_ids)

    if isnothing(charges)
        q = similar(χ, n_atoms)
        fill!(q, zero(QT))
    else
        if length(charges) != n_atoms
            throw(ArgumentError("Length of charges array must match number of atoms"))
        end
        q = charges
    end

    H_nonzero = similar(q, 0)               # resized later according to number of CSR entries (2*n_pairs)
    fill!(H_nonzero, zero(QT))

    rowptr = similar(type_ids, n_atoms + 1)
    colidx = similar(type_ids, 0)           # resized later according to number of CSR entries (2*n_pairs)
    row_counts = similar(type_ids, n_atoms)
    write_ptr = similar(type_ids, n_atoms)
    pair_ptr = similar(type_ids, 0)         # resized later according to number of CSR entries (2*n_pairs)

    fill!(rowptr, 0)
    fill!(row_counts, 0)
    fill!(write_ptr, 0)

    H_diag = similar(q, n_atoms)
    fill!(H_diag, zero(QT))
    H_diag_inv = similar(q, n_atoms)
    fill!(H_diag_inv, zero(QT))

    # initialize charge history as current charges
    prev_q1 = copy(q)
    prev_q2 = copy(q)
    prev_q3 = copy(q)
    prev_q4 = copy(q)
    q_backup = copy(q)

    r = copy(H_diag)
    d = copy(H_diag)
    Hq = copy(H_diag)
    Hd = copy(H_diag)
    z = copy(H_diag)
    mχ = copy(H_diag)

    buffers = similar(q, 5)
    fill!(buffers, zero(QT))

    backend = get_backend(q)

    # write H_diag and H_diag_inv, which do not change
    # also write per-atom -χ vector
    kernel! = update_QEq_diagonal!(backend)
    kernel!(
        H_diag,
        H_diag_inv,
        η,
        χ,
        mχ,
        type_ids;
        ndrange=n_atoms
    )

    χ_norm = norm(mχ)
    squared_tolerance = (QT(tolerance) * χ_norm)^2

    state = EEMState(
        q=q,
        q_prev_1=prev_q1,
        q_prev_2=prev_q2,
        q_prev_3=prev_q3,
        q_prev_4=prev_q4,
        q_backup=q_backup,
        history_count=0,
        last_history_step=-1,
        H_nonzero=H_nonzero,
        rowptr=rowptr,
        colidx=colidx,
        row_counts=row_counts,
        write_ptr=write_ptr,
        pair_ptr=pair_ptr,
        H_diag=H_diag,
        H_diag_inv=H_diag_inv,
        r=r,
        d=d,
        Hq=Hq,
        Hd=Hd,
        z=z,
        mχ=mχ,
        buffers=buffers
    )

    return EEM(
        coupling=coupling,
        n_atoms=n_atoms,
        type_ids=type_ids,
        χ=χ,
        η=η,
        tolerance=QT(tolerance),
        squared_tolerance=squared_tolerance,
        max_iterations=max_iterations,
        state=state
    )
end


@kernel function update_QEq_diagonal!(
    H_diag,
    H_diag_inv,
    η,
    χ,
    mχ,
    type_ids
)
    ti = @index(Global, Linear)

    @inbounds begin
        atom_type = type_ids[ti]
        η_ti = η[atom_type]
        H_diag[ti] = η_ti
        H_diag_inv[ti] = inv(η_ti)
        mχ[ti] = -χ[atom_type]
    end
end


# count number of entries in each row
# not cumulative sum yet
@kernel function count_QEq_csr_rows!(
    row_counts,
    neighbors
)
    ti = @index(Global, Linear)

    @inbounds begin
        i, j, _ = neighbors[ti]
        
        Atomix.@atomic row_counts[i] += 1
        Atomix.@atomic row_counts[j] += 1
    end
end

# populates rowptr based on cumulative row counts
# initially write_ptr is duplicating rowptr, but will later be incremented during CSR construction
@kernel function create_csr_rowptr!(
    rowptr,
    write_ptr,
    cumulative_row_counts
)
    ti = @index(Global, Linear)

    @inbounds begin
        rowptr[ti + 1] = cumulative_row_counts[ti] + 1

        if ti == 1
            rowptr[ti] = 1
            write_ptr[ti] = 1
        else
            write_ptr[ti] = cumulative_row_counts[ti - 1] + 1
        end
    end
end


# fills in colidx and pair_ptr arrays from write_ptr
# column order within each row is nondeterministic since pairs fill rows concurrently
@kernel function create_csr_colidx!(
    colidx,
    pair_ptr,
    write_ptr,
    neighbors
)
    ti = @index(Global, Linear)

    @inbounds begin
        i, j, _ = neighbors[ti]

        # first extracts the old value, then increments (fetch and add operation)
        write_ptr_i = first(Atomix.@atomic(write_ptr[i] + 1))
        write_ptr_j = first(Atomix.@atomic(write_ptr[j] + 1))

        colidx[write_ptr_i] = j
        colidx[write_ptr_j] = i

        pair_ptr[2*ti - 1] = write_ptr_i
        pair_ptr[2*ti] = write_ptr_j
    end
end


# updates both symmetric off-diagonal CSR values for every neighbor pair
@kernel function update_csr_values!(
    H_nonzero,
    pair_ptr,
    coupling,
    neighbors,
    type_ids,
    coords,
    boundary,
    conversion_factor::QT
) where QT
    ti = @index(Global, Linear)

    @inbounds begin
        i, j, _ = neighbors[ti]

        dr = vector(coords[j], coords[i], boundary)    # System float type
        dr_converted = map(x -> QT(ustrip(x)) * conversion_factor, dr)  # QEq float type
        r = norm(dr_converted)
        type_i = type_ids[i]
        type_j = type_ids[j]
        
        hij = eem_hij(coupling, r, type_i, type_j)

        H_nonzero[pair_ptr[2*ti - 1]] = hij
        H_nonzero[pair_ptr[2*ti]] = hij
    end
end


# 32 threads per matrix row, 8 rows per workgroup
const CSR_THREADS_PER_ROW = 32
const CSR_ROWS_PER_GROUP = 8

# calculates y = H * x using CSR representation
# GPU only
@kernel unsafe_indices=true function sparse_mat_mul_csr!(
    y,
    H_diag,
    H_nonzero,
    rowptr,
    colidx,
    x,
)   
    # thread_idx is 1->32, row_idx is 1->8
    thread_idx, row_idx = @index(Local, NTuple)
    _, group_idx = @index(Group, NTuple)

    # matrix_row_idx is 1->n_atoms (padded)
    matrix_row_idx = (group_idx - 1) * CSR_ROWS_PER_GROUP + row_idx

    num_matrix_rows = length(y)

    # shared memory containing a partial sum per thread and row
    partial_sums = @localmem eltype(y) (CSR_THREADS_PER_ROW, CSR_ROWS_PER_GROUP)

    thread_partial_sum = zero(eltype(y))

    # some threads correspond to rows beyond end of matrix since lauch range is multiple of CSR_ROWS_PER_GROUP
    if matrix_row_idx <= num_matrix_rows
        @inbounds begin
            # range for this row is rowptr[row] -> rowptr[row+1]-1
            row_start = rowptr[matrix_row_idx]
            row_end = rowptr[matrix_row_idx + 1] - 1

            # distribute nonzero entries across CSR_THREADS_PER_ROW
            for nonzero_idx in (row_start + thread_idx - 1):CSR_THREADS_PER_ROW:row_end
                matrix_col_idx = colidx[nonzero_idx]
                thread_partial_sum += H_nonzero[nonzero_idx] * x[matrix_col_idx]
            end
        end
    end

    @inbounds partial_sums[thread_idx, row_idx] = thread_partial_sum
    @synchronize    # need all partial sums completed before reduction

    # parallel tree reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    if thread_idx <= 16
        @inbounds partial_sums[thread_idx, row_idx] += partial_sums[thread_idx + 16, row_idx]
    end
    @synchronize

    if thread_idx <= 8
        @inbounds partial_sums[thread_idx, row_idx] += partial_sums[thread_idx + 8, row_idx]
    end
    @synchronize

    if thread_idx <= 4
        @inbounds partial_sums[thread_idx, row_idx] += partial_sums[thread_idx + 4, row_idx]
    end
    @synchronize

    if thread_idx <= 2
        @inbounds partial_sums[thread_idx, row_idx] += partial_sums[thread_idx + 2, row_idx]
    end
    @synchronize

    if thread_idx == 1
        @inbounds begin
            partial_sums[1, row_idx] += partial_sums[2, row_idx]

            # add diagonal contribution: y[matrix_row] = H_diag[matrix_row] * x[matrix_row] + off-diag CSR
            if matrix_row_idx <= num_matrix_rows
                y[matrix_row_idx] = H_diag[matrix_row_idx] * x[matrix_row_idx] + partial_sums[1, row_idx]
            end
        end
    end
end


# calculates y = H * x using CSR representation
# one thread owns one matrix row
# works on CPU and GPU
@kernel function sparse_mat_mul_csr_serial!(
    y,
    H_diag,
    H_nonzero,
    rowptr,
    colidx,
    x,
)
    ti = @index(Global, Linear)

    @inbounds begin
        y_ti = H_diag[ti] * x[ti]       # diagonal contribution

        row_start = rowptr[ti]
        row_end = rowptr[ti + 1] - 1

        for idx in row_start:row_end    # off-diagonal contribution
            c = colidx[idx]
            y_ti += H_nonzero[idx] * x[c]
        end

        y[ti] = y_ti
    end
end


# dispatch on backend type: CPU only works using serial CSR matmul function
function sparse_mat_mul_kernel(backend, n_atoms)
    if backend isa KernelAbstractions.GPU
        workgroup_size = (CSR_THREADS_PER_ROW, CSR_ROWS_PER_GROUP)
        num_rows_cld = cld(n_atoms, CSR_ROWS_PER_GROUP) * CSR_ROWS_PER_GROUP
        ndrange = (CSR_THREADS_PER_ROW, num_rows_cld)

        return (
            sparse_mat_mul_csr!(backend, workgroup_size),
            ndrange
        )
    else
        return (
            sparse_mat_mul_csr_serial!(backend),
            n_atoms
        )
    end
end

const QEQ_THREADS_PER_GROUP = 256

# sums the target and predicted charges into the slots 1 and 2 of the EEMState scalar buffer
# allows the neutrality correction to remain entirely on the gpu
@kernel unsafe_indices=true function charge_sums!(
    buffers,
    q_target,
    q_predicted
)
    ti = @index(Global, Linear)     # global index indentifies which atom the thread processes
    li = @index(Local, Linear)      # local index identifies slot in shared local memory, 1 -> QEQ_THREADS_PER_GROUP
    n_atoms = length(q_target)

    q_target_partial_sums = @localmem eltype(q_target) (QEQ_THREADS_PER_GROUP,)
    q_predicted_partial_sums = @localmem eltype(q_predicted) (QEQ_THREADS_PER_GROUP,)

    @inbounds begin
        # one atom per thread, with threads beyond atom count padded with zero
        q_target_partial_sums[li] = ti <= n_atoms ? q_target[ti] : zero(eltype(q_target))
        q_predicted_partial_sums[li] = ti <= n_atoms ? q_predicted[ti] : zero(eltype(q_predicted))
    end

    # parallel tree reduction
    for stage in 1:trailing_zeros(QEQ_THREADS_PER_GROUP)
        @synchronize
        offset = QEQ_THREADS_PER_GROUP >> stage
        if li <= offset
            @inbounds begin
                q_target_partial_sums[li] += q_target_partial_sums[li + offset]
                q_predicted_partial_sums[li] += q_predicted_partial_sums[li + offset]
            end
        end
    end

    if li == 1
        Atomix.@atomic buffers[1] += q_target_partial_sums[1]
        Atomix.@atomic buffers[2] += q_predicted_partial_sums[1]
    end
end

# apply a uniform charge shift so the predicted charges have the same net charge as the target charges
@kernel function correct_predicted_charge!(
    q_predicted,
    buffers
)
    ti = @index(Global, Linear)

    @inbounds begin
        # buffers[1] contains sum of target charges, buffers[2] contains sum of predicted charges
        q_predicted[ti] += (buffers[1] - buffers[2]) / length(q_predicted)
    end
end


# calculate initial unprojected residual r_0 = -χ - Hq and calculate its sum into buffer slot 3
@kernel unsafe_indices=true function initial_residual_sum!(
    r,
    mχ,
    Hq,
    buffers
)
    ti = @index(Global, Linear)
    li = @index(Local, Linear)
    n_atoms = length(r)

    # shared storage for workgroup's residual sum reduction
    partial_sum = @localmem eltype(r) (QEQ_THREADS_PER_GROUP,)

    # if thread maps to padded indices, value is zero
    val = zero(eltype(r))

    if ti <= n_atoms
        @inbounds begin
            val = mχ[ti] - Hq[ti]
            r[ti] = val
        end
    end

    @inbounds partial_sum[li] = val

    # parallel tree reduction
    for stage in 1:trailing_zeros(QEQ_THREADS_PER_GROUP)
        @synchronize
        offset = QEQ_THREADS_PER_GROUP >> stage
        if li <= offset
            @inbounds partial_sum[li] += partial_sum[li + offset]
        end
    end

    if li == 1
        # buffers[3] now stores sum(eem.state.r)
        Atomix.@atomic buffers[3] += partial_sum[1]
    end
end


# projects the residual vector r to the zero-sum subspace using projection operator P = I - (1/N)I(I^T)
# then preconditions this initial projected residual using a Jacobi preconditioner: M^(-1)*r
# M = H_diag_inv and r_shift = (1/N)1(1^T)
# then calculates the sum and writes into the buffers slots
@kernel unsafe_indices=true function project_precondition_reduce!(
    r,
    z,
    buffers,
    H_diag_inv,
)
    ti = @index(Global, Linear)
    li = @index(Local, Linear)
    n_atoms = length(r)

    partial_sum = @localmem eltype(r) (QEQ_THREADS_PER_GROUP,)

    preconditioned_residual = zero(eltype(r))

    if ti <= n_atoms
        @inbounds begin
            r_shift = buffers[3] / n_atoms
            projected_residual = r[ti] - r_shift
            r[ti] = projected_residual
            preconditioned_residual = projected_residual * H_diag_inv[ti]
            z[ti] = preconditioned_residual     # this needs to be projected again after this kernel
        end
    end

    @inbounds partial_sum[li] = preconditioned_residual

    for stage in 1:trailing_zeros(QEQ_THREADS_PER_GROUP)
        @synchronize
        offset = QEQ_THREADS_PER_GROUP >> stage
        if li <= offset
            @inbounds partial_sum[li] += partial_sum[li + offset]
        end
    end

    if li == 1
        # buffers[1] stores sum(eem.state.z)
        Atomix.@atomic buffers[1] += partial_sum[1]
    end
end


# project initial preconditioned residual and set initial search direction to initial projected preconditioned residual
# calculate the scalar product dot(r, z) and store into buffers slot 2
@kernel unsafe_indices=true function initialize_direction_reduce!(
    r,
    z,
    d,
    buffers,
)
    ti = @index(Global, Linear)
    li = @index(Local, Linear)
    n_atoms = length(r)

    # stores dot(r, z)
    partial_sum = @localmem eltype(r) (QEQ_THREADS_PER_GROUP,)

    rz_value = zero(eltype(r))

    # remove z's mean, initialize d, and calculate dot-product term
    if ti <= n_atoms
        @inbounds begin
            z_value = z[ti] - buffers[1] / n_atoms
            z[ti] = z_value
            d[ti] = z_value
            rz_value = r[ti] * z_value
        end
    end

    @inbounds partial_sum[li] = rz_value

    for stage in 1:trailing_zeros(QEQ_THREADS_PER_GROUP)
        @synchronize
        offset = QEQ_THREADS_PER_GROUP >> stage
        if li <= offset
            @inbounds partial_sum[li] += partial_sum[li + offset]
        end
    end

    if li == 1
        # buffers[2] stores dot(r, z)
        Atomix.@atomic buffers[2] += partial_sum[1]
    end
end

# computes the dot product dot(a, b) and accumulates into buffers slot 4
@kernel unsafe_indices=true function reduce_dot!(
    buffers,
    a,
    b,
)
    ti = @index(Global, Linear)
    li = @index(Local, Linear)
    n = length(a)

    partial_sums = @localmem eltype(a) (QEQ_THREADS_PER_GROUP,)

    @inbounds partial_sums[li] = ti <= n ? a[ti] * b[ti] : zero(eltype(a))

    for stage in 1:trailing_zeros(QEQ_THREADS_PER_GROUP)
        @synchronize
        offset = QEQ_THREADS_PER_GROUP >> stage
        if li <= offset
            @inbounds partial_sums[li] += partial_sums[li + offset]
        end
    end

    if li == 1
        # dot product written into buffers[4]
        Atomix.@atomic buffers[4] += partial_sums[1]
    end
end


# validate dHd, then update charges and residual if the curvature is valid
# also reduce updated r sum into buffers slot 3
@kernel unsafe_indices=true function update_charge_residual_reduce!(
    q,
    r,
    d,
    Hd,
    buffers,
    rz,
)
    ti = @index(Global, Linear)
    li = @index(Local, Linear)
    n_atoms = length(q)

    partial_sums = @localmem eltype(q) (QEQ_THREADS_PER_GROUP,)

    # read previously calculated curvature and reject non-positive values
    dHd = buffers[4]
    valid_curvature = isfinite(dHd) && dHd > zero(dHd)

    r_value = zero(eltype(q))

    # apply CG step and retain updated residual for reduction
    if ti <= n_atoms
        @inbounds begin
            if valid_curvature
                α = rz / dHd
                q[ti] += α * d[ti]
                r[ti] -= α * Hd[ti]
            end

            r_value = r[ti]
        end
    end

    # thread 1 records a failure flag when curvature invalid
    if ti == 1 && !valid_curvature
        @inbounds buffers[5] = one(eltype(q))
    end

    @inbounds partial_sums[li] = r_value

    for stage in 1:trailing_zeros(QEQ_THREADS_PER_GROUP)
        @synchronize
        offset = QEQ_THREADS_PER_GROUP >> stage
        if li <= offset
            @inbounds partial_sums[li] += partial_sums[li + offset]
        end
    end

    if li == 1
        # buffers[3] holds sum(eem.state.r)
        Atomix.@atomic buffers[3] += partial_sums[1]
    end
end


# project z to zero sum and reduce dot(r, z) into buffers slot 2
@kernel unsafe_indices=true function shift_z_reduce_rz!(
    r,
    z,
    buffers,
)
    ti = @index(Global, Linear)
    li = @index(Local, Linear)
    n_atoms = length(r)

    partial_sums = @localmem eltype(r) (QEQ_THREADS_PER_GROUP,)

    failed = buffers[5] != zero(eltype(r))

    rz_value = zero(eltype(r))

    # if curvature was valid, remove z's mean and calculate dot(r, z) term
    if ti <= n_atoms && !failed
        @inbounds begin
            z_value = z[ti] - buffers[1] / n_atoms
            z[ti] = z_value
            rz_value = r[ti] * z_value
        end
    end

    @inbounds partial_sums[li] = rz_value

    for stage in 1:trailing_zeros(QEQ_THREADS_PER_GROUP)
        @synchronize
        offset = QEQ_THREADS_PER_GROUP >> stage
        if li <= offset
            @inbounds partial_sums[li] += partial_sums[li + offset]
        end
    end

    # write NaN on failure, otherwise accumulate dot(r, z) in slot 2
    if li == 1
        if buffers[5] != zero(eltype(r))
            Atomix.@atomic buffers[2] += eltype(r)(NaN)
        else
            Atomix.@atomic buffers[2] += partial_sums[1]
        end
    end
end


# calculate the next search direction using d = z + βd
@kernel function update_direction!(
    d,
    z,
    β,
)
    ti = @index(Global, Linear)

    @inbounds d[ti] = z[ti] + β * d[ti]
end


# rounds atom count up to whole number of reduction workgroups
function qeq_reduction_ndrange(n_atoms)
    return cld(n_atoms, QEQ_THREADS_PER_GROUP) * QEQ_THREADS_PER_GROUP
end


# returns scalar conversion factor (no units) from System length units to those required by the coupling in eem_hij() 
# required length unit defined by eem_length_unit(coupling) function
function eem_length_conversion_factor(sys, coupling::AbstractEEMCoupling, ::Type{T}) where T
    if length_type(sys.boundary) <: Unitful.Quantity
        coord_unit = unit(eltype(eltype(sys.coords)))    # System length unit
        return T(ustrip(uconvert(eem_length_unit(coupling), one(T) * coord_unit)))
    else
        # no units used: assume coordinates are in nm
        return T(ustrip(uconvert(eem_length_unit(coupling), one(T) * u"nm")))
    end
end


# predicts next time step's charges using Lagrange interpolation
# uses quartic charge prediction when there is sufficient history
# when there is insufficient history, uses fewer history vectors to avoid oscillation in predicted charges
# every coefficient set sums to one, preserving total charge
# only modifies q
@kernel function predict_charges!(
    q,
    q_prev_1,
    q_prev_2,
    q_prev_3,
    q_prev_4,
    history_count
)
    ti = @index(Global, Linear)

    @inbounds begin
        q0 = q[ti]
        q1 = q_prev_1[ti]
        q2 = q_prev_2[ti]
        q3 = q_prev_3[ti]
        q4 = q_prev_4[ti]

        QT = typeof(q0)

        if history_count <= 1
            q_pred = q0
        elseif history_count == 2
            q_pred = QT(2) * q0 - q1
        elseif history_count == 3
            q_pred = QT(3) * q0 - QT(3) * q1 + q2
        elseif history_count == 4
            q_pred = QT(4) * q0 - QT(6) * q1 + QT(4) * q2 - q3
        else
            q_pred = QT(5) * q0 - QT(10) * q1 + QT(10) * q2 - QT(5) * q3 + q4
        end

        q[ti] = q_pred
    end
end


function shift_history!(
    q_prev_1,
    q_prev_2,
    q_prev_3,
    q_prev_4,
    q_backup
)
    q_prev_4 .= q_prev_3
    q_prev_3 .= q_prev_2
    q_prev_2 .= q_prev_1
    q_prev_1 .= q_backup

    return nothing
end


function charge_equilibration!(
    eem::EEM{C, VI, VQ, QT},
    sys::System,
    neighbors::Union{NeighborList, GPUCellListNeighborList},    # far neighbors list
    step_n::Int,
    backend
) where {C, VI, VQ, QT}
    n_pairs = length(neighbors)
    n_atoms = eem.n_atoms

    num_csr_entries = n_pairs * 2
    num_csr_entries_buffered = ceil(Int, num_csr_entries * 1.2)
    
    # resize H_nonzero based on number of CSR entries, with buffer factor
    if length(eem.state.H_nonzero) < num_csr_entries
        eem.state.H_nonzero = similar(eem.state.H_nonzero, num_csr_entries_buffered)
        eem.state.colidx = similar(eem.state.colidx, num_csr_entries_buffered)
        eem.state.pair_ptr = similar(eem.state.pair_ptr, num_csr_entries_buffered)
    end

    requires_rebuild = true

    if requires_rebuild
        use_geometric_counts =
            neighbors isa GPUCellListNeighborList &&
            has_ragged_neighbors(neighbors) &&
            sys.neighbor_finder isa GPUCellListNeighborFinder &&
            sys.neighbor_finder.output === :geometric_pairs

        # count number of CSR entries in each row
        if use_geometric_counts
            eem.state.row_counts .= ragged_counts(neighbors)
        else
            fill!(eem.state.row_counts, 0)

            if n_pairs > 0
                count_csr_kernel! = count_QEq_csr_rows!(backend)
                count_csr_kernel!(
                    eem.state.row_counts,
                    neighbors.list;
                    ndrange=n_pairs
                )
            end
        end

        # turn raw counts into cumulative counts
        cumsum!(eem.state.row_counts, eem.state.row_counts)

        # populate rowptr and initialize write_ptr
        create_rowptr_kernel! = create_csr_rowptr!(backend)
        create_rowptr_kernel!(
            eem.state.rowptr,
            eem.state.write_ptr,
            eem.state.row_counts;
            ndrange=n_atoms
        )

        # populate colidx and pair_ptr
        if n_pairs > 0
            create_colidx_kernel! = create_csr_colidx!(backend)
            create_colidx_kernel!(
                eem.state.colidx,
                eem.state.pair_ptr,
                eem.state.write_ptr,
                neighbors.list;
                ndrange=n_pairs
            )
        end
    end

    # length conversion factor from System length units (coords) to General Interaction length units (parameter-compatible)
    conversion_factor = eem_length_conversion_factor(sys, eem.coupling, QT)

    if n_pairs > 0
        # recalculate hij values every call since coordinates may have changed
        update_csr_kernel! = update_csr_values!(backend)
        update_csr_kernel!(
            eem.state.H_nonzero,
            eem.state.pair_ptr,
            eem.coupling,
            neighbors.list,
            eem.type_ids,
            sys.coords,
            sys.boundary,
            conversion_factor;
            ndrange=n_pairs
        )
    end

    eem.state.q_backup .= eem.state.q   # create backup in case of failure

    reduction_ndrange = qeq_reduction_ndrange(n_atoms)
    reduction_workgroup_size = (QEQ_THREADS_PER_GROUP,)

    # predict next time step's charges using charge history only if new MD step
    if step_n != eem.state.last_history_step
        predict_charges_kernel! = predict_charges!(backend)
        predict_charges_kernel!(
            eem.state.q,
            eem.state.q_prev_1,
            eem.state.q_prev_2,
            eem.state.q_prev_3,
            eem.state.q_prev_4,
            eem.state.history_count;
            ndrange=n_atoms
        )

        # correct any extrapolation roundoff to preserve net charge
        fill!(eem.state.buffers, zero(QT))
        charge_sums_kernel! = charge_sums!(backend, reduction_workgroup_size)
        charge_sums_kernel!(
            eem.state.buffers,
            eem.state.q_backup,
            eem.state.q;
            ndrange=reduction_ndrange
        )

        correct_charge_kernel! = correct_predicted_charge!(backend)
        correct_charge_kernel!(
            eem.state.q,
            eem.state.buffers;
            ndrange=n_atoms
        )
    end

    fill!(eem.state.buffers, zero(QT))

    # calculate H*q
    smm_kernel!, smm_kernel_ndrange = sparse_mat_mul_kernel(backend, n_atoms)
    smm_kernel!(
        eem.state.Hq,
        eem.state.H_diag,
        eem.state.H_nonzero,
        eem.state.rowptr,
        eem.state.colidx,
        eem.state.q;
        ndrange=smm_kernel_ndrange      # global range might be (32, 13200) but workgroup size is (32, 8)
    )

    # calculate initial unprojected residual r_0 = -χ - Hq and its sum
    residual_kernel! = initial_residual_sum!(backend, reduction_workgroup_size)
    residual_kernel!(
        eem.state.r,
        eem.state.mχ,
        eem.state.Hq,
        eem.state.buffers;
        ndrange=reduction_ndrange
    )

    # project initial residual and then precondition, and calculates its sum
    project_reduce_kernel! = project_precondition_reduce!(backend, reduction_workgroup_size)
    project_reduce_kernel!(
        eem.state.r,
        eem.state.z,
        eem.state.buffers,
        eem.state.H_diag_inv,
        ndrange=reduction_ndrange
    )

    # project initial preconditioned residual and set initial search direction to initial projected preconditioned residual
    # calculate the scalar product dot(r, z)
    initial_direction_kernel! = initialize_direction_reduce!(backend, reduction_workgroup_size)
    initial_direction_kernel!(
        eem.state.r,
        eem.state.z,
        eem.state.d,
        eem.state.buffers;
        ndrange=reduction_ndrange
    )

    # numerator of α (distance to move along search direction)
    rz = scalar_getindex(eem.state.buffers, 2)

    QEq_tolerance = eem.squared_tolerance

    # rz should be non-negative
    if !isfinite(rz) || rz < -QEq_tolerance
        eem.state.q .= eem.state.q_backup
        synchronize(backend)
        error("EEM QEq failed at step $step_n, iteration 0: rz=$rz, squared tolerance=$QEq_tolerance")
    end
    rz = max(rz, zero(QT))     # small negative value within tolerance assumed to be floating point roundoff and converged

    iter = 0

    dot_kernel! = reduce_dot!(backend, reduction_workgroup_size)
    update_qr_kernel! = update_charge_residual_reduce!(backend, reduction_workgroup_size)
    shift_z_kernel! = shift_z_reduce_rz!(backend, reduction_workgroup_size)
    direction_kernel! = update_direction!(backend)

    while iter < eem.max_iterations && rz > QEq_tolerance 
        fill!(eem.state.buffers, zero(QT))

        # calculate H*d
        smm_kernel!(
            eem.state.Hd,
            eem.state.H_diag,
            eem.state.H_nonzero,
            eem.state.rowptr,
            eem.state.colidx,
            eem.state.d;
            ndrange=smm_kernel_ndrange
        )
        
        # calculate dHd, the denominator of α
        dot_kernel!(
            eem.state.buffers,
            eem.state.d,
            eem.state.Hd;
            ndrange=reduction_ndrange
        )

        # check curvature, calculate α, update q and r 
        # calculate sum(r) and store in slot 3
        update_qr_kernel!(
            eem.state.q,
            eem.state.r,
            eem.state.d,
            eem.state.Hd,
            eem.state.buffers,
            rz;
            ndrange=reduction_ndrange
        )

        # project updated residual, precondition it
        # calculate sum(z) and store in slot 1
        project_reduce_kernel!(
            eem.state.r,
            eem.state.z,
            eem.state.buffers,
            eem.state.H_diag_inv,
            ndrange=reduction_ndrange
        )

        # project preconditioned updated residual
        # calculate dot(r, z) and store in slot 2
        shift_z_kernel!(
            eem.state.r,
            eem.state.z,
            eem.state.buffers;
            ndrange=reduction_ndrange
        )

        # calculate Gram-Schmidt coefficient for search direction update
        rz_old = rz
        rz = scalar_getindex(eem.state.buffers, 2)
        iter += 1

        # a NaN is written to slot 2 when dHd is invalid
        if !isfinite(rz)
            dHd = scalar_getindex(eem.state.buffers, 4)
            eem.state.q .= eem.state.q_backup
            synchronize(backend)
            error("EEM QEq failed at step $step_n, iteration $iter: rz=$rz, rz_old=$rz_old, dHd=$dHd, squared tolerance=$QEq_tolerance")
        end

        if rz < -QEq_tolerance
            eem.state.q .= eem.state.q_backup
            synchronize(backend)
            error("EEM QEq failed at step $step_n, iteration $iter: rz=$rz, squared tolerance=$QEq_tolerance")
        end

        rz = max(rz, zero(QT))  

        # only construct new direction if not converged
        if rz > QEq_tolerance
            β = rz / rz_old

            direction_kernel!(
                eem.state.d,
                eem.state.z,
                β;
                ndrange=n_atoms
            )
        end
    end

    if iter == eem.max_iterations && rz > QEq_tolerance
        eem.state.q .= eem.state.q_backup
        synchronize(backend)
        error("EEM QEq did not converge at step $step_n after $iter iterations: rz=$rz, squared tolerance=$QEq_tolerance")
    end

    # if reach here, CG converged

    if step_n != eem.state.last_history_step
        shift_history!(
            eem.state.q_prev_1,
            eem.state.q_prev_2,
            eem.state.q_prev_3,
            eem.state.q_prev_4,
            eem.state.q_backup
        )

        eem.state.history_count = min(eem.state.history_count + 1, 5)
        eem.state.last_history_step = step_n
    end

    return nothing
end


@inline function write_atom_charge(atom::Atom, q)
    return Atom(
        atom.index,
        atom.atom_type,
        atom.mass,
        convert(typeof(atom.charge), q),
        atom.σ,
        atom.ϵ,
        atom.λ,
        atom.alch_role
    )
end

function write_charges!(sys, q, backend)
    sys.atoms .= write_atom_charge.(sys.atoms, q)
    synchronize(backend)
    return nothing
end


# convert energy from eV to System energy units
# float type is set as System float precision
function eem_energy_conversion(energy_eV, sys, coupling::AbstractEEMCoupling)
    T = float_type(sys)
    energy_eV_mol = energy_eV * u"eV" * Unitful.Na
    target_energy_units = sys.energy_units

    if target_energy_units == NoUnits
        # assume default Molly energy unit is kJ/mol
        energy_sys = ustrip(uconvert(u"kJ/mol", energy_eV_mol))
    else
        energy_sys = uconvert(target_energy_units, energy_eV_mol)
    end

    return T(energy_sys)
end


# run charge equilibration using EEM, and return EEM self energy in System energy units and float precision
# charges are written into sys.atoms 
AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys, eem::EEM; neighbors, step_n::Integer=0, kwargs...)
    backend = get_backend(sys.coords)
    charge_equilibration!(eem, sys, neighbors, step_n, backend)
    write_charges!(sys, eem.state.q, backend)

    QT = typeof(eem.tolerance)

    # self energy in units of eV
    eem_self_energy = sum(
        -eem.state.mχ .* eem.state.q .+
        QT(0.5) .* eem.state.H_diag .* eem.state.q.^2
    )

    # convert from eV to System energy units and float type
    return eem_energy_conversion(eem_self_energy, sys, eem.coupling)
end

# run charge equilibration using EEM, returns unchanged fs array
# the EEM self energy does not contribute to the force if the parameters do not depend on geometry
AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs, sys, eem::EEM; neighbors, step_n::Integer=0, n_threads::Integer=Threads.nthreads(), kwargs...)
    backend = get_backend(sys.coords)
    charge_equilibration!(eem, sys, neighbors, step_n, backend)
    write_charges!(sys, eem.state.q, backend)

    return fs
end
