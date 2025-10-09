export SHAKE_RATTLE

"""
    SHAKE_RATTLE(n_atoms, dist_tolerance, vel_tolerance; dist_constraints=nothing,
                 angle_constraints=nothing, gpu_block_size=128, max_iters=25)

Constrain distances during a simulation using the SHAKE and RATTLE algorithms.
Either or both of `dist_constraints` and `angle_constraints` must be passed.

Velocity constraints will be imposed for simulators that integrate velocities such as
[`VelocityVerlet`](@ref).
See [Ryckaert et al. 1977](https://doi.org/10.1016/0021-9991(77)90098-5) for SHAKE,
[Andersen 1983](https://doi.org/10.1016/0021-9991(83)90014-1) for RATTLE,
[Elber et al. 2011](https://doi.org/10.1140/epjst/e2011-01525-9) for a derivation
of the linear system solved to satisfy the RATTLE algorithm and
[Krautler et al. 2000](https://doi.org/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V)
for the M-SHAKE algorithm.

# Arguments
- `n_atoms`: number of atoms in the system.
- `dist_tolerance=1e-8u"nm"`: the tolerance used to end the iterative procedure when calculating
    position constraints, should have the same units as the coordinates.
- `vel_tolerance=1e-8u"nm^2 * ps^-1"`: the tolerance used to end the iterative procedure when
    calculating velocity constraints, should have the same units as the velocities times the
    coordinates.
- `dist_constraints=nothing`: a vector of [`DistanceConstraint`](@ref) objects that define the
    distance constraints to be applied. If `nothing`, no distance constraints are applied.
- `angle_constraints=nothing`: a vector of [`AngleConstraint`](@ref) objects that define the
    angle constraints to be applied. If `nothing`, no angle constraints are applied.
- `gpu_block_size=128`: the number of threads per block to use for GPU calculations.
- `max_iters=25`: the maximum number of iterations to perform when doing SHAKE. If this
    number if iterations is reached, some constraints may not be satisfied.
"""
struct SHAKE_RATTLE{A, B, C, D, E, F, I <: Integer}
    clusters12::A
    clusters23::B
    clusters34::C
    angle_clusters::D
    dist_tolerance::E
    vel_tolerance::F
    gpu_block_size::I
    max_iters::I
end

function SHAKE_RATTLE(n_atoms,
                      dist_tolerance=1e-8u"nm",
                      vel_tolerance=1e-8u"nm^2 * ps^-1";
                      dist_constraints=nothing,
                      angle_constraints=nothing,
                      gpu_block_size=128,
                      max_iters=25)
    ustrip(dist_tolerance) > 0 || throw(ArgumentError("dist_tolerance must be greater than zero"))
    ustrip(vel_tolerance ) > 0 || throw(ArgumentError("vel_tolerance must be greater than zero" ))

    dc_present = !isnothing(dist_constraints ) && length(dist_constraints ) > 0
    ac_present = !isnothing(angle_constraints) && length(angle_constraints) > 0
    if !(dc_present || ac_present)
        throw(ArgumentError("at least one of dist_constraints or angle_constraints must be " *
                            "provided to SHAKE_RATTLE"))
    end

    if typeof(ustrip(dist_tolerance)) == Float32 && ustrip(dist_tolerance) < Float32(1e-6)
        @warn "Using Float32 with a SHAKE dist_tolerance less than 1e-6, this may lead " *
              "to convergence issues"
    end

    if typeof(ustrip(vel_tolerance)) == Float32 && ustrip(vel_tolerance) < Float32(1e-6)
        @warn "Using Float32 with a RATTLE vel_tolerance less than 1e-6, this may lead " *
              "to convergence issues"
    end

    if isa(dist_constraints, AbstractGPUArray) || isa(angle_constraints, AbstractGPUArray)
        throw(ArgumentError("constraints should be passd to SHAKE_RATTLE on CPU, data will " *
                            "be moved to GPU later"))
    end

    clusters12, clusters23, clusters34, angle_clusters = build_clusters(n_atoms, dist_constraints,
                                                                        angle_constraints)

    return SHAKE_RATTLE(clusters12, clusters23, clusters34, angle_clusters, dist_tolerance,
                        vel_tolerance, gpu_block_size, max_iters)
end

function SHAKE_RATTLE(sr::SHAKE_RATTLE, clusters12, clusters23, clusters34, angle_clusters)
    return SHAKE_RATTLE(clusters12, clusters23, clusters34, angle_clusters, sr.dist_tolerance,
                        sr.vel_tolerance, sr.gpu_block_size, sr.max_iters)
end

function Base.show(io::IO, sr::SHAKE_RATTLE)
    print(io, "SHAKE_RATTLE with ", length(sr.clusters12), " 2-atom clusters, ",
          length(sr.clusters23), " 3-atom clusters, ", length(sr.clusters34),
          " 4-atom clusters and ", length(sr.angle_clusters), " angle clusters")
end

cluster_keys(::SHAKE_RATTLE) = (:clusters12, :clusters23, :clusters34, :angle_clusters)

function setup_constraints!(sr::SHAKE_RATTLE, neighbor_finder, arr_type)
    # Disable Neighbor interactions that are constrained
    if typeof(neighbor_finder) != NoNeighborFinder
        disable_constrained_interactions!(neighbor_finder, sr.clusters12)
        disable_constrained_interactions!(neighbor_finder, sr.clusters23)
        disable_constrained_interactions!(neighbor_finder, sr.clusters34)
        disable_constrained_interactions!(neighbor_finder, sr.angle_clusters)
    end

    # Move to proper backend, if CPU do nothing
    if arr_type <: AbstractGPUArray
        clusters12_gpu, clusters23_gpu, clusters34_gpu = [], [], []
        angle_clusters_gpu = []

        if length(sr.clusters12) > 0
            clusters12_gpu = replace_storage(arr_type, sr.clusters12)
        end
        if length(sr.clusters23) > 0
            clusters23_gpu = replace_storage(arr_type, sr.clusters23)
        end
        if length(sr.clusters34) > 0
            clusters34_gpu = replace_storage(arr_type, sr.clusters34)
        end
        if length(sr.angle_clusters) > 0
            angle_clusters_gpu = replace_storage(arr_type, sr.angle_clusters)
        end

        sr = SHAKE_RATTLE(sr, clusters12_gpu, clusters23_gpu, clusters34_gpu, angle_clusters_gpu)
    end

    # neighbor_finder also modified
    return sr
end

# The RATTLE equations are modified from LAMMPS:
# https://github.com/lammps/lammps/blob/develop/src/RIGID/fix_rattle.cpp

@inline function solve_2x2_exactly(λ, A, C)
    determinant = (A[1, 1] * A[2, 2]) - (A[1, 2] * A[2, 1])

    if iszero(determinant)
        error("SHAKE determinant is zero, cannot solve")
    end
    inv_det = inv(determinant)

    λ[1] = inv_det * ((A[2, 2] * C[1]) - (A[1, 2] * C[2]))
    λ[2] = inv_det * ((A[1, 1] * C[2]) - (A[2, 1] * C[1]))
    return λ
end

@inline function solve_3x3_exactly!(λ, A, A′, C)
    determinant = A[1,1]*A[2,2]*A[3,3] + A[1,2]*A[2,3]*A[3,1] + A[1,3]*A[2,1]*A[3,2] -
                  A[1,1]*A[2,3]*A[3,2] - A[1,2]*A[2,1]*A[3,3] - A[1,3]*A[2,2]*A[3,1]

    if iszero(determinant)
        error("SHAKE determinant is zero, cannot solve")
    end
    inv_det = inv(determinant)

    A′[1,1] =  inv_det * (A[2,2]*A[3,3] - A[2,3]*A[3,2])
    A′[1,2] = -inv_det * (A[1,2]*A[3,3] - A[1,3]*A[3,2])
    A′[1,3] =  inv_det * (A[1,2]*A[2,3] - A[1,3]*A[2,2])
    A′[2,1] = -inv_det * (A[2,1]*A[3,3] - A[2,3]*A[3,1])
    A′[2,2] =  inv_det * (A[1,1]*A[3,3] - A[1,3]*A[3,1])
    A′[2,3] = -inv_det * (A[1,1]*A[2,3] - A[1,3]*A[2,1])
    A′[3,1] =  inv_det * (A[2,1]*A[3,2] - A[2,2]*A[3,1])
    A′[3,2] = -inv_det * (A[1,1]*A[3,2] - A[1,2]*A[3,1])
    A′[3,3] =  inv_det * (A[1,1]*A[2,2] - A[1,2]*A[2,1])

    # Assumes λ is set to zeros initially
    for i in 1:3
        λ[i] += (A′[i,1] * C[1]) + (A′[i,2] * C[2]) + (A′[i,3] * C[3])
    end
    return λ
end

# 2 atoms, 1 constraint
@kernel inbounds=true function rattle2_kernel!(@Const(k1s),
                                               @Const(k2s),
                                               @Const(r),
                                               v,
                                               @Const(ms),
                                               boundary)
    idx = @index(Global, Linear)

    if idx <= length(k1s)
        # Step 2: perform RATTLE, for a 2 atom cluster we
        # just re-arrange λ = A / c since they are all scalars
        k1 = k1s[idx]
        k2 = k2s[idx]

        v_k1 = v[k1] # Uncoalesced read
        v_k2 = v[k2] # Uncoalesced read

        m1_inv, m2_inv = inv(ms[k1]), inv(ms[k2]) # Uncoalesced read
        r_k1k2  = vector(r[k1], r[k2], boundary) # Uncoalesced read
        v_k1k2 = v_k2 .- v_k1

        λₖ = -dot(r_k1k2, v_k1k2) / (dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv))

        # Step 3: update velocities in global memory
        v[k1] -= m1_inv .* λₖ .* r_k1k2
        v[k2] += m2_inv .* λₖ .* r_k1k2
    end
end

# 3 atoms, 2 constraints
# Assumes first atom is central atom
@kernel inbounds=true function rattle3_kernel!(@Const(k1s),
                                               @Const(k2s),
                                               @Const(k3s),
                                               r::AbstractVector{<:AbstractVector{L}},
                                               v::AbstractVector{<:AbstractVector{V}},
                                               ms::AbstractVector{M},
                                               boundary) where {L, V, M}
    idx = @index(Global, Linear)
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform C_type = typeof(zero(V) * zero(L))
    @uniform L_type = typeof(zero(C_type) / zero(A_type))

    if idx <= length(k1s)
        A = @MMatrix zeros(A_type, 2, 2) # Units are L^2 / M
        C = @MVector zeros(C_type, 2) # Units are L^2 / T
        λ = @MVector zeros(L_type, 2) # Units are M / T

        k1 = k1s[idx] # Central atom
        k2 = k2s[idx]
        k3 = k3s[idx]
        r_k1 = r[k1]

        m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # Uncoalesced read
        r_k1k2 = vector(r_k1, r[k2], boundary)
        r_k1k3 = vector(r_k1, r[k3], boundary)

        v_k1 = v[k1] # Uncoalesced read
        v_k2 = v[k2] # Uncoalesced read
        v_k3 = v[k3] # Uncoalesced read

        v_k1k2 = v_k2 .- v_k1
        v_k1k3 = v_k3 .- v_k1

        A[1, 1] = dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv)
        A[1, 2] = dot(r_k1k2, r_k1k3) * (m1_inv)
        A[2, 1] = A[1, 2]
        A[2, 2] = dot(r_k1k3, r_k1k3) * (m1_inv + m3_inv)

        C[1] = -dot(r_k1k2, v_k1k2)
        C[2] = -dot(r_k1k3, v_k1k3)

        solve_2x2_exactly(λ, A, C)

        v[k1] -= m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        v[k2] -= m2_inv .* (-λ[1] .* r_k1k2)
        v[k3] -= m3_inv .* (-λ[2] .* r_k1k3)
    end
end

# 4 atoms, 3 constraints
# Assumes first atom is central atom
@kernel inbounds=true function rattle4_kernel!(@Const(k1s),
                                               @Const(k2s),
                                               @Const(k3s),
                                               @Const(k4s),
                                               r::AbstractVector{<:AbstractVector{L}},
                                               v::AbstractVector{<:AbstractVector{V}},
                                               ms::AbstractVector{M},
                                               boundary) where {L, V, M}
    idx = @index(Global, Linear)
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform A_tmp_type = typeof(zero(M) / (zero(L)*zero(L)))
    @uniform C_type = typeof(zero(V)*zero(L))
    @uniform L_type = typeof(zero(C_type) / zero(A_type))

    if idx <= length(k1s)
        A = @MMatrix zeros(A_type, 3, 3)
        A_tmp = @MMatrix zeros(A_tmp_type, 3, 3)
        C = @MVector zeros(C_type, 3)
        λ = @MVector zeros(L_type, 3)

        k1 = k1s[idx] # Central atom
        k2 = k2s[idx]
        k3 = k3s[idx]
        k4 = k4s[idx]
        r_k1 = r[k1] # Uncoalesced read

        m1_inv, m2_inv, m3_inv, m4_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]), inv(ms[k4]) # Uncoalesced read
        r_k1k2  = vector(r_k1, r[k2], boundary) # Uncoalesced read
        r_k1k3  = vector(r_k1, r[k3], boundary) # Uncoalesced read
        r_k1k4  = vector(r_k1, r[k4], boundary) # Uncoalesced read

        vk1 = v[k1] # Uncoalesced read
        vk2 = v[k2] # Uncoalesced read
        vk3 = v[k3] # Uncoalesced read
        vk4 = v[k4] # Uncoalesced read

        v_k1k2 = vk2 .- vk1
        v_k1k3 = vk3 .- vk1
        v_k1k4 = vk4 .- vk1

        A[1, 1] = dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv)
        A[1, 2] = dot(r_k1k2, r_k1k3) * (m1_inv)
        A[1, 3] = dot(r_k1k2, r_k1k4) * (m1_inv)
        A[2, 1] = A[1, 2]
        A[2, 2] = dot(r_k1k3, r_k1k3) * (m1_inv + m3_inv)
        A[2, 3] = dot(r_k1k3, r_k1k4) * (m1_inv)
        A[3, 1] = A[1, 3]
        A[3, 2] = A[2, 3]
        A[3, 3] = dot(r_k1k4, r_k1k4) * (m1_inv + m4_inv)

        C[1] = -dot(r_k1k2, v_k1k2)
        C[2] = -dot(r_k1k3, v_k1k3)
        C[3] = -dot(r_k1k4, v_k1k4)

        solve_3x3_exactly!(λ, A, A_tmp, C)

        v[k1] -= m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3) .+ λ[3] .* r_k1k4)
        v[k2] -= m2_inv .* (-λ[1] .* r_k1k2)
        v[k3] -= m3_inv .* (-λ[2] .* r_k1k3)
        v[k4] -= m4_inv .* (-λ[3] .* r_k1k4)
    end
end

# 3 atoms, 3 constraints
@kernel inbounds=true function rattle3_angle_kernel!(@Const(k1s),
                                                     @Const(k2s),
                                                     @Const(k3s),
                                                     r::AbstractVector{<:AbstractVector{L}},
                                                     v::AbstractVector{<:AbstractVector{V}},
                                                     ms::AbstractVector{M},
                                                     boundary) where {L, V, M}
    idx = @index(Global, Linear)
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform A_tmp_type = typeof(zero(M) / (zero(L) * zero(L)))
    @uniform C_type = typeof(zero(V) * zero(L))
    @uniform L_type = typeof(zero(C_type) / zero(A_type))

    if idx <= length(k1s)
        A = @MMatrix zeros(A_type, 3, 3)
        A_tmp = @MMatrix zeros(A_tmp_type, 3, 3)
        C = @MVector zeros(C_type, 3)
        λ = @MVector zeros(L_type, 3)

        k1 = k1s[idx] # Central atom
        k2 = k2s[idx]
        k3 = k3s[idx]

        r_k1, v_k1 = r[k1], v[k1]
        r_k2, v_k2 = r[k2], v[k2]
        r_k3, v_k3 = r[k3], v[k3]

        m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # Uncoalesced read
        r_k1k2  = vector(r_k1, r_k2, boundary) # Uncoalesced read
        r_k1k3  = vector(r_k1, r_k3, boundary) # Uncoalesced read
        r_k2k3  = vector(r_k2, r_k3, boundary) # Uncoalesced read

        v_k1k2 = v_k2 .- v_k1
        v_k1k3 = v_k3 .- v_k1
        v_k2k3 = v_k3 .- v_k2

        A[1, 1] = dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv)
        A[1, 2] = dot(r_k1k2, r_k1k3) * (m1_inv)
        A[1, 3] = dot(r_k1k2, r_k2k3) * (-m2_inv)
        A[2, 1] = A[1, 2]
        A[2, 2] = dot(r_k1k3, r_k1k3) * (m1_inv + m3_inv)
        A[2, 3] = dot(r_k1k3, r_k2k3) * (m3_inv)
        A[3, 1] = A[1, 3]
        A[3, 2] = A[2, 3]
        A[3, 3] = dot(r_k2k3, r_k2k3) * (m2_inv + m3_inv)

        C[1] = -dot(r_k1k2, v_k1k2)
        C[2] = -dot(r_k1k3, v_k1k3)
        C[3] = -dot(r_k2k3, v_k2k3)

        solve_3x3_exactly!(λ, A, A_tmp, C)

        v[k1] -= m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        v[k2] -= m2_inv .* ((-λ[1] .* r_k1k2) .+ (λ[3] .* r_k2k3))
        v[k3] -= m3_inv .* ((-λ[2] .* r_k1k3) .- (λ[3] .* r_k2k3))
    end
end

# 2 atoms, 1 constraint
@kernel inbounds=true function shake2_kernel!(@Const(k1s),
                                              @Const(k2s),
                                              @Const(dists),
                                              @Const(r_t1),
                                              r_t2::T,
                                              @Const(ms),
                                              boundary::AbstractBoundary{<:Any, FT}) where {T, FT}
    idx = @index(Global, Linear)

    if idx <= length(k1s)
        k1 = k1s[idx] # Central atom
        k2 = k2s[idx]
        distance = dists[idx]

        r_t2_k1 = r_t2[k1] # Uncoalesced read
        r_t2_k2 = r_t2[k2] # Uncoalesced read
        r_t1_k1 = r_t1[k1] # Uncoalesced read
        r_t1_k2 = r_t1[k2] # Uncoalesced read

        # Vector between the atoms after unconstrained update (s)
        s12 = vector(r_t2_k1, r_t2_k2, boundary)

        # Vector between the atoms before unconstrained update (r)
        r12 = vector(r_t1_k1, r_t1_k2, boundary)

        m1_inv, m2_inv = inv(ms[k1]), inv(ms[k2])
        a = (m1_inv + m2_inv)^2 * sum(abs2, r12)
        b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
        c = sum(abs2, s12) - (distance)^2
        D = b^2 - 4*a*c

        α1 = (-b + sqrt(D)) / (2*a)
        α2 = (-b - sqrt(D)) / (2*a)
        g = ifelse(α1 <= α2, α1, α2)

        r_t2[k1] += r12 .* (g*m1_inv)
        r_t2[k2] += r12 .* (-g*m2_inv)
    end
end

@kernel inbounds=true function shake_step!(@Const(active_idxs),
                                           still_active::AbstractVector{Bool},
                                           N_active,
                                           shake_fn,
                                           kernel_args...)
    tid = @index(Global, Linear)
    if tid <= N_active
        cluster_idx = active_idxs[tid]
        # Do one M-SHAKE iteration and check if it is converged or not
        is_active = shake_fn(
            cluster_idx,
            kernel_args...
        )
        still_active[cluster_idx] = is_active
    end
end

function shake_gpu!(clusters::StructArray{C},
                    max_iters,
                    gpu_block_size,
                    backend,
                    shake_kernel,
                    other_kernel_args...) where {C <: ConstraintKernelData}
    N_active_clusters = length(clusters)
    kern = shake_step!(backend, gpu_block_size)
    active_idxs = allocate(backend, Int32, N_active_clusters)
    active_idxs .= 1:N_active_clusters
    # Doesnt need to be initialized, kernel will do that
    still_active = allocate(backend, Bool, N_active_clusters)
    KernelAbstractions.pagelock!(backend, still_active)

    iter = 1
    while iter <= max_iters
        kern(
            active_idxs,
            still_active,
            N_active_clusters,
            shake_kernel,
            getproperty.(Ref(clusters), idx_keys(C))...,
            getproperty.(Ref(clusters), dist_keys(C))...,
            other_kernel_args...;
            ndrange=N_active_clusters,
        )

        # This compaction could be done on GPU with AcceleratedKernels.jl scan
        # and a scatter operation
        still_active_host = Array(still_active)
        active_idxs_host  = findall(still_active_host)
        isempty(active_idxs_host) && break

        N_active_clusters = length(active_idxs_host)
        # Move active indices to the start
        # Anything at the end is ignored by kernel as only N_active_clusters threads are launched
        @views copy!(active_idxs[1:N_active_clusters], Int32.(active_idxs_host))

        iter += 1
    end

    if iter == max_iters + 1
        @warn "SHAKE $(Symbol(shake_kernel)) did not converge after $max_iters iterations, " *
              "some constraints may not be satisfied"
    end
end

# 3 atoms, 2 constraints
# Constraints between 1-2 and 1-3
@inline function shake3_kernel!(cluster_idx,
                                k1s, k2s, k3s,
                                dist12s, dist13s,
                                r_t1::AbstractVector{<:AbstractVector{L}},
                                r_t2::AbstractVector{<:AbstractVector{L}},
                                ms::AbstractVector{M},
                                boundary::AbstractBoundary{<:Any, FT},
                                dist_tol::L) where {L, M, FT}
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform C_type = typeof(zero(L) * zero(L))

    A = @MMatrix zeros(A_type, 2, 2) # Units are L^2 / M
    C = @MVector zeros(C_type, 2) # Units are L^2
    λ = @MVector zeros(M, 2) # Units are M

    k1 = k1s[cluster_idx] # Central atom
    k2 = k2s[cluster_idx]
    k3 = k3s[cluster_idx]

    dist12 = dist12s[cluster_idx]
    dist13 = dist13s[cluster_idx]

    m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # Uncoalesced read

    r_t2_k1 = r_t2[k1] # Uncoalesced read
    r_t1_k1 = r_t1[k1] # Uncoalesced read
    r_t2_k2 = r_t2[k2] # Uncoalesced read
    r_t2_k3 = r_t2[k3] # Uncoalesced read

    r_k1k2  = vector(r_t1_k1, r_t1[k2], boundary)
    r_k1k3  = vector(r_t1_k1, r_t1[k3], boundary)

    # Distance vectors after unconstrained update
    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)

    # Matrix element i, j represents interaction of constraint i with constraint j
    A[1,1] = -2 * dot(r_k1k2, s_k1k2) * (m1_inv + m2_inv) # Set constraint 1 as between k1-k2
    A[2,2] = -2 * dot(r_k1k3, s_k1k3) * (m1_inv + m3_inv) # Set constraint 1 as between k1-k3
    A[1,2] = -2 * dot(r_k1k3, s_k1k2) * m1_inv
    A[2,1] = -2 * dot(r_k1k2, s_k1k3) * m1_inv

    C[1] = (dot(s_k1k2, s_k1k2) - (dist12*dist12))
    C[2] = (dot(s_k1k3, s_k1k3) - (dist13*dist13))

    solve_2x2_exactly(λ, A, C)

    Δ1 = m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
    Δ2 = m2_inv .* (-λ[1] .* r_k1k2)
    Δ3 = m3_inv .* (-λ[2] .* r_k1k3)

    r_t2[k1] -= Δ1
    r_t2[k2] -= Δ2
    r_t2[k3] -= Δ3

    r_t2_k1 -= Δ1
    r_t2_k2 -= Δ2
    r_t2_k3 -= Δ3

    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)

    tol12 = abs(norm(s_k1k2) - dist12)
    tol13 = abs(norm(s_k1k3) - dist13)

    # Constraint still active if either above tolerance
    return (tol12 > dist_tol) || (tol13 > dist_tol)
end

# 4 atoms, 3 constraints
# Constraints between 1-2, 1-3 and 1-4
@inline function shake4_kernel!(cluster_idx,
                                k1s, k2s, k3s, k4s,
                                dist12s, dist13s, dist14s,
                                r_t1::AbstractVector{<:AbstractVector{L}},
                                r_t2::AbstractVector{<:AbstractVector{L}},
                                ms::AbstractVector{M},
                                boundary::AbstractBoundary{<:Any, FT},
                                dist_tol::L) where {L, M, FT}
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform A_tmp_type = typeof(zero(M) / (zero(L) * zero(L)))
    @uniform C_type = typeof(zero(L) * zero(L))

    A = @MMatrix zeros(A_type, 3, 3)
    A_tmp = @MMatrix zeros(A_tmp_type, 3, 3)
    C = @MVector zeros(C_type, 3)
    λ = @MVector zeros(M, 3)

    k1 = k1s[cluster_idx] # Central atom
    k2 = k2s[cluster_idx]
    k3 = k3s[cluster_idx]
    k4 = k4s[cluster_idx]

    dist12 = dist12s[cluster_idx]
    dist13 = dist13s[cluster_idx]
    dist14 = dist14s[cluster_idx]

    m1_inv, m2_inv, m3_inv, m4_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]), inv(ms[k4]) # Uncoalesced read

    r_t1_k1 = r_t1[k1] # Uncoalesced read
    r_t2_k1 = r_t2[k1] # Uncoalesced read
    r_t2_k2 = r_t2[k2] # Uncoalesced read
    r_t2_k3 = r_t2[k3] # Uncoalesced read
    r_t2_k4 = r_t2[k4] # Uncoalesced read

    r_k1k2  = vector(r_t1_k1, r_t1[k2], boundary)
    r_k1k3  = vector(r_t1_k1, r_t1[k3], boundary)
    r_k1k4  = vector(r_t1_k1, r_t1[k4], boundary)

    # Distance vectors after unconstrainted update
    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)
    s_k1k4 = vector(r_t2_k1, r_t2_k4, boundary)

    # Matrix element i, j represents interaction of constraint i with constraint j
    A[1,1] = -2 * dot(r_k1k2, s_k1k2) * (m1_inv + m2_inv) # Set constraint 1 as between k1-k2
    A[2,2] = -2 * dot(r_k1k3, s_k1k3) * (m1_inv + m3_inv) # Set constraint 2 as between k1-k3
    A[3,3] = -2 * dot(r_k1k4, s_k1k4) * (m1_inv + m4_inv) # Set constraint 3 as between k1-k4
    A[1,2] = -2 * dot(r_k1k3, s_k1k2) * m1_inv
    A[2,1] = -2 * dot(r_k1k2, s_k1k3) * m1_inv
    A[1,3] = -2 * dot(r_k1k4, s_k1k2) * m1_inv
    A[3,1] = -2 * dot(r_k1k2, s_k1k4) * m1_inv
    A[2,3] = -2 * dot(r_k1k4, s_k1k3) * m1_inv
    A[3,2] = -2 * dot(r_k1k3, s_k1k4) * m1_inv

    C[1] = (dot(s_k1k2, s_k1k2) - (dist12*dist12))
    C[2] = (dot(s_k1k3, s_k1k3) - (dist13*dist13))
    C[3] = (dot(s_k1k4, s_k1k4) - (dist14*dist14))

    solve_3x3_exactly!(λ, A, A_tmp, C)

    Δ1 = m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3) .+ (λ[3] .* r_k1k4))
    Δ2 = m2_inv .* (-λ[1] .* r_k1k2)
    Δ3 = m3_inv .* (-λ[2] .* r_k1k3)
    Δ4 = m4_inv .* (-λ[3] .* r_k1k4)

    r_t2[k1] -= Δ1
    r_t2[k2] -= Δ2
    r_t2[k3] -= Δ3
    r_t2[k4] -= Δ4

    # Check tolerances, just re-compute instead of uncoalesced read
    r_t2_k1 -= Δ1
    r_t2_k2 -= Δ2
    r_t2_k3 -= Δ3
    r_t2_k4 -= Δ4

    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)
    s_k1k4 = vector(r_t2_k1, r_t2_k4, boundary)

    tol12 = abs(norm(s_k1k2) - dist12)
    tol13 = abs(norm(s_k1k3) - dist13)
    tol14 = abs(norm(s_k1k4) - dist14)

    # Constraint still active if either above tolerance
    return (tol12 > dist_tol) || (tol13 > dist_tol) || (tol14 > dist_tol)
end

# 3 atoms, 3 constraints
# Constraints between 1-2, 1-3 and 2-3
@inline function shake3_angle_kernel!(cluster_idx,
                                      k1s, k2s, k3s,
                                      dist12s, dist13s, dist23s,
                                      r_t1::AbstractVector{<:AbstractVector{L}},
                                      r_t2::AbstractVector{<:AbstractVector{L}},
                                      ms::AbstractVector{M},
                                      boundary::AbstractBoundary{<:Any, FT},
                                      dist_tol::L) where {L, M, FT}
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform A_tmp_type = typeof(zero(M) / (zero(L) * zero(L)))
    @uniform C_type = typeof(zero(L) * zero(L))

    A = @MMatrix zeros(A_type, 3, 3)
    A_tmp = @MMatrix zeros(A_tmp_type, 3, 3)
    C = @MVector zeros(C_type, 3)
    λ = @MVector zeros(M, 3)

    k1 = k1s[cluster_idx] # Central atom
    k2 = k2s[cluster_idx]
    k3 = k3s[cluster_idx]

    dist12 = dist12s[cluster_idx]
    dist13 = dist13s[cluster_idx]
    dist23 = dist23s[cluster_idx]

    m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # Uncoalesced read

    r_t2_k1 = r_t2[k1] # Uncoalesced read
    r_t2_k2 = r_t2[k2] # Uncoalesced read
    r_t2_k3 = r_t2[k3] # Uncoalesced read

    r_t1_k1 = r_t1[k1] # Uncoalesced read
    r_t1_k2 = r_t1[k2] # Uncoalesced read
    r_t1_k3 = r_t1[k3] # Uncoalesced read

    r_k1k2  = vector(r_t1_k1, r_t1_k2, boundary)
    r_k1k3  = vector(r_t1_k1, r_t1_k3, boundary)
    r_k2k3  = vector(r_t1_k2, r_t1_k3, boundary)

    # Distance vectors after unconstrained update
    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)
    s_k2k3 = vector(r_t2_k2, r_t2_k3, boundary)

    # Matrix element i, j represents interaction of constraint i with constraint j
    A[1,1] = -2 * dot(r_k1k2, s_k1k2) * (m1_inv + m2_inv) # Set constraint 1 as between k1-k2
    A[2,2] = -2 * dot(r_k1k3, s_k1k3) * (m1_inv + m3_inv) # Set constraint 2 as between k1-k3
    A[3,3] = -2 * dot(r_k2k3, s_k2k3) * (m2_inv + m3_inv) # Set constraint 3 as between k2-k3
    A[1,2] = -2 * dot(r_k1k3, s_k1k2) * m1_inv
    A[2,1] = -2 * dot(r_k1k2, s_k1k3) * m1_inv
    A[1,3] = -2 * dot(r_k2k3, s_k1k2) * (-m2_inv)
    A[3,1] = -2 * dot(r_k1k2, s_k1k3) * (-m2_inv)
    A[2,3] = -2 * dot(r_k2k3, s_k1k3) * m3_inv
    A[3,2] = -2 * dot(r_k1k3, s_k2k3) * m3_inv

    C[1] = (dot(s_k1k2, s_k1k2) - (dist12*dist12))
    C[2] = (dot(s_k1k3, s_k1k3) - (dist13*dist13))
    C[3] = (dot(s_k2k3, s_k2k3) - (dist23*dist23))

    solve_3x3_exactly!(λ, A, A_tmp, C)

    Δ1 = m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
    Δ2 = m2_inv .* ((-λ[1] .* r_k1k2) .+ (λ[3] .* r_k2k3))
    Δ3 = m3_inv .* ((-λ[2] .* r_k1k3) .- (λ[3] .* r_k2k3))

    r_t2[k1] -= Δ1
    r_t2[k2] -= Δ2
    r_t2[k3] -= Δ3

    # Check tolerances, just re-compute instead of uncoalesced read
    r_t2_k1 -= Δ1
    r_t2_k2 -= Δ2
    r_t2_k3 -= Δ3

    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)
    s_k2k3 = vector(r_t2_k2, r_t2_k3, boundary)

    tol12 = abs(norm(s_k1k2) - dist12)
    tol13 = abs(norm(s_k1k3) - dist13)
    tol23 = abs(norm(s_k2k3) - dist23)

    # Constraint still active if either above tolerance
    return (tol12 > dist_tol) || (tol13 > dist_tol) || (tol23 > dist_tol)
end

function apply_position_constraints!(sys::System,
                                     ca::SHAKE_RATTLE,
                                     r_pre_unconstrained_update;
                                     kwargs...)
    N12_clusters = length(ca.clusters12)
    N23_clusters = length(ca.clusters23)
    N34_clusters = length(ca.clusters34)
    N_angle_clusters = length(ca.angle_clusters)

    backend = get_backend(r_pre_unconstrained_update)
    KernelAbstractions.synchronize(backend)

    if N12_clusters > 0
        # 2 atom constraints are solved analytically, no need to iterate
        s2_kernel! = shake2_kernel!(backend, ca.gpu_block_size)
        s2_kernel!(
            ca.clusters12.k1,
            ca.clusters12.k2,
            ca.clusters12.dist12,
            r_pre_unconstrained_update,
            sys.coords,
            masses(sys),
            sys.boundary,
            ndrange=N12_clusters,
        )
    end

    if N23_clusters > 0
        shake_gpu!(
            ca.clusters23,
            ca.max_iters,
            ca.gpu_block_size,
            backend,
            shake3_kernel!,
            r_pre_unconstrained_update,
            sys.coords,
            masses(sys),
            sys.boundary,
            ca.dist_tolerance,
        )
    end

    if N34_clusters > 0
        shake_gpu!(
            ca.clusters34,
            ca.max_iters,
            ca.gpu_block_size,
            backend,
            shake4_kernel!,
            r_pre_unconstrained_update,
            sys.coords,
            masses(sys),
            sys.boundary,
            ca.dist_tolerance,
        )
    end

    if N_angle_clusters > 0
        shake_gpu!(
            ca.angle_clusters,
            ca.max_iters,
            ca.gpu_block_size,
            backend,
            shake3_angle_kernel!,
            r_pre_unconstrained_update,
            sys.coords,
            masses(sys),
            sys.boundary,
            ca.dist_tolerance,
        )
    end

    KernelAbstractions.synchronize(backend)
end

function apply_velocity_constraints!(sys::System, ca::SHAKE_RATTLE; kwargs...)
    N12_clusters = length(ca.clusters12)
    N23_clusters = length(ca.clusters23)
    N34_clusters = length(ca.clusters34)
    N_angle_clusters = length(ca.angle_clusters)

    backend = get_backend(sys.velocities)
    KernelAbstractions.synchronize(backend)

    if N12_clusters > 0
        N12_blocks = cld(N12_clusters, ca.gpu_block_size)
        r2_kernel! = rattle2_kernel!(backend, N12_blocks, N12_clusters)
        r2_kernel!(
            ca.clusters12.k1,
            ca.clusters12.k2,
            sys.coords,
            sys.velocities,
            masses(sys),
            sys.boundary,
            ndrange=N12_clusters,
        )
    end

    if N23_clusters > 0
        N23_blocks = cld(N23_clusters, ca.gpu_block_size)
        r3_kernel! = rattle3_kernel!(backend, N23_blocks, N23_clusters)
        r3_kernel!(
            ca.clusters23.k1,
            ca.clusters23.k2,
            ca.clusters23.k3,
            sys.coords,
            sys.velocities,
            masses(sys),
            sys.boundary,
            ndrange=N23_clusters,
        )
    end

    if N34_clusters > 0
        N34_blocks = cld(N34_clusters, ca.gpu_block_size)
        r4_kernel! = rattle4_kernel!(backend, N34_blocks, N34_clusters)
        r4_kernel!(
            ca.clusters34.k1,
            ca.clusters34.k2,
            ca.clusters34.k3,
            ca.clusters34.k4,
            sys.coords,
            sys.velocities,
            masses(sys),
            sys.boundary,
            ndrange=N34_clusters,
        )
    end

    if N_angle_clusters > 0
        N_angle_blocks = cld(N_angle_clusters, ca.gpu_block_size)
        r3_angle_kernel! = rattle3_angle_kernel!(backend, N_angle_blocks, N_angle_clusters)
        r3_angle_kernel!(
            ca.angle_clusters.k1,
            ca.angle_clusters.k2,
            ca.angle_clusters.k3,
            sys.coords,
            sys.velocities,
            masses(sys),
            sys.boundary,
            ndrange=N_angle_clusters,
        )
    end

    KernelAbstractions.synchronize(backend)
end
