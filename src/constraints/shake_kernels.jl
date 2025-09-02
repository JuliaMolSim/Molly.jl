# The RATTLE Equations are modified from LAMMPS:
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

    # Assumes lambda is set to zeros initially
    for i in 1:3
        λ[i] += (A′[i,1] * C[1]) + (A′[i,2] * C[2]) + (A′[i,3] * C[3])
    end
    return λ
end

# 2 atoms, 1 constraint
@kernel inbounds=true function rattle2_kernel!(
    @Const(k1s),
    @Const(k2s),
    @Const(r),
    v,
    @Const(ms),
    @Const(boundary))

    idx = @index(Global, Linear) # Global Constraint Idx

    if idx <= length(k1s)

        # Step 2 : Perform RATTLE, for a 2 atom cluster we
        # just re-arrange λ = A / c, since they are all scalars.
        k1 = k1s[idx]
        k2 = k2s[idx]

        v_k1 = v[k1] # uncoalesced read
        v_k2 = v[k2] # uncoalesced read
    
        m1_inv, m2_inv = inv(ms[k1]), inv(ms[k2]) # uncoalesced read
        r_k1k2  = vector(r[k1], r[k2], boundary) # uncoalesced read
        v_k1k2 = v_k2 .- v_k1

        λₖ = -dot(r_k1k2, v_k1k2) / (dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv))

        # Step 3: Update velocities in global memory
        v[k1] -= m1_inv .* λₖ .* r_k1k2
        v[k2] += m2_inv .* λₖ .* r_k1k2
    end
end

# 3 atoms, 2 constraints
# Assumes first atom is central atom
@kernel inbounds=true function rattle3_kernel!(
    @Const(k1s),
    @Const(k2s),
    @Const(k3s),
    @Const(r::AbstractVector{<:AbstractVector{L}}),
    v::AbstractVector{<:AbstractVector{V}},
    @Const(ms::AbstractVector{M}),
    @Const(boundary)) where {L, V, M}

    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform C_type = typeof(zero(V) * zero(L))
    @uniform L_type = typeof(zero(C_type) / zero(A_type))

    if idx <= length(k1s)        
        # Allocate thread-local memory
        A = @MMatrix zeros(A_type, 2, 2) # Units are L^2 / M
        C = @MVector zeros(C_type, 2) # Units are L^2 / T
        λ = @MVector zeros(L_type, 2) # Units are M / T

        k1 = k1s[idx] # central atom
        k2 = k2s[idx]
        k3 = k3s[idx]

        r_k1 = r[k1]

        m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # uncoalesced read
        r_k1k2 = vector(r_k1, r[k2], boundary)
        r_k1k3 = vector(r_k1, r[k3], boundary)

        v_k1 = v[k1] # uncoalesced read
        v_k2 = v[k2] # uncoalesced read
        v_k3 = v[k3] # uncoalesced read

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
@kernel inbounds=true function rattle4_kernel!(
    @Const(k1s),
    @Const(k2s),
    @Const(k3s),
    @Const(k4s),
    @Const(r::AbstractVector{<:AbstractVector{L}}),
    v::AbstractVector{<:AbstractVector{V}},
    @Const(ms::AbstractVector{M}),
    @Const(boundary)) where {L, V, M}

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

        k1 = k1s[idx] # central atom
        k2 = k2s[idx]
        k3 = k3s[idx]
        k4 = k4s[idx]

        r_k1 = r[k1] # uncoalesced read
     
        m1_inv, m2_inv, m3_inv, m4_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]), inv(ms[k4]) # uncoalesced read
        r_k1k2  = vector(r_k1, r[k2], boundary) # uncoalesced read
        r_k1k3  = vector(r_k1, r[k3], boundary) # uncoalesced read
        r_k1k4  = vector(r_k1, r[k4], boundary) # uncoalesced read

        vk1 = v[k1] # uncoalesced read
        vk2 = v[k2] # uncoalesced read
        vk3 = v[k3] # uncoalesced read
        vk4 = v[k4] # uncoalesced read

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
@kernel inbounds=true function rattle3_angle_kernel!(
    @Const(k1s),
    @Const(k2s),
    @Const(k3s),
    @Const(r::AbstractVector{<:AbstractVector{L}}),
    v::AbstractVector{<:AbstractVector{V}},
    @Const(ms::AbstractVector{M}),
    @Const(boundary)) where {L, V, M}

    idx = @index(Global, Linear)
    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform A_tmp_type = typeof(zero(M) / (zero(L) * zero(L)))
    @uniform C_type = typeof(zero(V) * zero(L))
    @uniform L_type = typeof(zero(C_type) / zero(A_type))

    if idx <= length(k1s)

        # Allocate thread-local memory
        A = @MMatrix zeros(A_type, 3, 3)
        A_tmp = @MMatrix zeros(A_tmp_type, 3, 3)
        C = @MVector zeros(C_type, 3)
        λ = @MVector zeros(L_type, 3)

        k1 = k1s[idx] # central atom
        k2 = k2s[idx]
        k3 = k3s[idx]

        r_k1 = r[k1]; v_k1 = v[k1]
        r_k2 = r[k2]; v_k2 = v[k2]
        r_k3 = r[k3]; v_k3 = v[k3]
     
        m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # uncoalesced read
        r_k1k2  = vector(r_k1, r_k2, boundary) # uncoalesced read
        r_k1k3  = vector(r_k1, r_k3, boundary) # uncoalesced read
        r_k2k3  = vector(r_k2, r_k3, boundary) # uncoalesced read

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
@kernel inbounds=true function shake2_kernel!(
    k1s,
    k2s,
    dists, 
    r_t1::T,
    r_t2::T,
    @Const(ms),
    @Const(boundary::AbstractBoundary{<:Any, FT})) where {T, FT}

    idx = @index(Global, Linear)

    if idx <= length(k1s)
        k1 = k1s[idx] # central atom
        k2 = k2s[idx]
        distance = dists[idx]

        r_t2_k1 = r_t2[k1] # uncoalesced read
        r_t2_k2 = r_t2[k2] # uncoalesced read
        r_t1_k1 = r_t1[k1] # uncoalesced read
        r_t1_k2 = r_t1[k2] # uncoalesced read
        
        # Vector between the atoms after unconstrained update (s)
        s12 = vector(r_t2_k1, r_t2_k2, boundary) 

        # Vector between the atoms before unconstrained update (r)
        r12 = vector(r_t1_k1, r_t1_k2, boundary)

        m1_inv, m2_inv = inv(ms[k1]), inv(ms[k2])
        a = (m1_inv + m2_inv)^2 * sum(abs2, r12)
        b = -FT(2.0) * (m1_inv + m2_inv) * dot(r12, s12)
        c = sum(abs2, s12) - (distance)^2
        D = b^2 - FT(4.0)*a*c
        
        # Just let the system blow up?? 
        # This usually happens when timestep too large or over constrained
        # if ustrip(D) < FT(0.0)
        #     error("SHAKE determinant negative: $(D)")
        # end

        α1 = (-b + sqrt(D)) / (2*a)
        α2 = (-b - sqrt(D)) / (2*a)
        g = ifelse(α1 <= α2, α1, α2)

        r_t2[k1] += r12 .* (g*m1_inv)
        r_t2[k2] += r12 .* (-g*m2_inv)
    end
end

@kernel inbounds=true function shake_step!( 
        @Const(active_idxs),    
        still_active::AbstractVector{Bool},
        @Const(N_active),
        @Const(shake_fn),
        kernel_args...;
    )

    tid = @index(Global, Linear)
        
    if tid <= N_active
        # Get cluster-idx this thread will work on
        cluster_idx = active_idxs[tid]

        # Do one M-SHAKE iteration and check if it is converged or not
        is_active = shake_fn(
            cluster_idx,
            kernel_args...
        )
        still_active[cluster_idx] = is_active
    end
end

function shake_gpu!(
        clusters::StructArray{C},
        max_iters,
        gpu_block_size,
        backend,
        shake_kernel,
        other_kernel_args...,
    ) where {C <: ConstraintKernelData}

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

        #* This compaction can be done ON GPU with 
        #* the scan imeplmented in AcceleratedKernels.jl + a scatter operation
        #* for now this is easier and (probably) faster for smaller systems
        #* On CUDA/AMD we could also pin this memory...
        still_active_host = Array(still_active) #! MOVING FROM DEVICE TO HOST
        active_idxs_host  = findall(still_active_host) 

        isempty(active_idxs_host) && break

        N_active_clusters = length(active_idxs_host)
        # Move active indices to the start. Anything at the end
        # is ignored by kernel as only N_active_clusters 
        # threads are launched.
        @views copy!(active_idxs[1:N_active_clusters], Int32.(active_idxs_host))#! MOVING FROM HOST TO DEVICE
    
        iter += 1
    end

    if iter == max_iters + 1
        @warn "SHAKE, $(Symbol(shake_kernel)), did not converge after $(max_iters) iterations. Some constraints may not be satisfied."
    end
end

# 3 atoms, 2 constraints
# Constraints between 1-2 and 1-3
@inline function shake3_kernel!(
        cluster_idx,
        k1s, k2s, k3s,
        dist12s, dist13s,
        r_t1::AbstractVector{<:AbstractVector{L}}, 
        r_t2::AbstractVector{<:AbstractVector{L}},
        ms::AbstractVector{M}, 
        boundary::AbstractBoundary{<:Any, FT},
        dist_tol::L
    ) where {L, M, FT}

    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform C_type = typeof(zero(L) * zero(L))

    # Allocate thread-local memory
    A = @MMatrix zeros(A_type, 2, 2) # Units are L^2 / M
    C = @MVector zeros(C_type, 2) # Units are L^2
    λ = @MVector zeros(M, 2) # Units are M

    k1 = k1s[cluster_idx] # central atom
    k2 = k2s[cluster_idx]
    k3 = k3s[cluster_idx]

    dist12 = dist12s[cluster_idx]
    dist13 = dist13s[cluster_idx]

    m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # uncoalesced read

    r_t2_k1 = r_t2[k1] # uncoalesced read
    r_t1_k1 = r_t1[k1] # uncoalesced read
    r_t2_k2 = r_t2[k2] # uncoalesced read
    r_t2_k3 = r_t2[k3] # uncoalesced read

    r_k1k2  = vector(r_t1_k1, r_t1[k2], boundary)
    r_k1k3  = vector(r_t1_k1, r_t1[k3], boundary)

    # Distance vectors after unconstrained update
    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)

    # Matrix element i, j represents interaction of constraint i with constraint j
    A[1,1] = -FT(2.0) * dot(r_k1k2, s_k1k2) * (m1_inv + m2_inv) # Set constraint 1 as between k1-k2
    A[2,2] = -FT(2.0) * dot(r_k1k3, s_k1k3) * (m1_inv + m3_inv) # Set constraint 1 as between k1-k3
    A[1,2] = -FT(2.0) * dot(r_k1k3, s_k1k2) * m1_inv
    A[2,1] = -FT(2.0) * dot(r_k1k2, s_k1k3) * m1_inv

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
@inline function shake4_kernel!(
        cluster_idx,
        k1s, k2s, k3s, k4s,
        dist12s, dist13s, dist14s,
        r_t1::AbstractVector{<:AbstractVector{L}},
        r_t2::AbstractVector{<:AbstractVector{L}},
        ms::AbstractVector{M}, 
        boundary::AbstractBoundary{<:Any, FT},
        dist_tol::L
    ) where {L, M, FT}   

    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform A_tmp_type = typeof(zero(M) / (zero(L) * zero(L)))
    @uniform C_type = typeof(zero(L) * zero(L))

    A = @MMatrix zeros(A_type, 3, 3)
    A_tmp = @MMatrix zeros(A_tmp_type, 3, 3)
    C = @MVector zeros(C_type, 3)
    λ = @MVector zeros(M, 3)

    k1 = k1s[cluster_idx] # central atom
    k2 = k2s[cluster_idx]
    k3 = k3s[cluster_idx]
    k4 = k4s[cluster_idx]

    dist12 = dist12s[cluster_idx]
    dist13 = dist13s[cluster_idx]
    dist14 = dist14s[cluster_idx]

    m1_inv, m2_inv, m3_inv, m4_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]), inv(ms[k4]) # uncoalesced read    

    r_t1_k1 = r_t1[k1] # uncoalesced read
    r_t2_k1 = r_t2[k1] # uncoalesced read
    r_t2_k2 = r_t2[k2] # uncoalesced read
    r_t2_k3 = r_t2[k3] # uncoalesced read
    r_t2_k4 = r_t2[k4] # uncoalesced read

    r_k1k2  = vector(r_t1_k1, r_t1[k2], boundary)
    r_k1k3  = vector(r_t1_k1, r_t1[k3], boundary)
    r_k1k4  = vector(r_t1_k1, r_t1[k4], boundary)

    # Distance vectors after unconstrainted update
    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)
    s_k1k4 = vector(r_t2_k1, r_t2_k4, boundary)

    # Matrix element i, j represents interaction of constraint i with constraint j
    A[1,1] = -FT(2.0) * dot(r_k1k2, s_k1k2) * (m1_inv + m2_inv) # Set constraint 1 as between k1-k2
    A[2,2] = -FT(2.0) * dot(r_k1k3, s_k1k3) * (m1_inv + m3_inv) # Set constraint 2 as between k1-k3
    A[3,3] = -FT(2.0) * dot(r_k1k4, s_k1k4) * (m1_inv + m4_inv) # Set constraint 3 as between k1-k4
    A[1,2] = -FT(2.0) * dot(r_k1k3, s_k1k2) * m1_inv
    A[2,1] = -FT(2.0) * dot(r_k1k2, s_k1k3) * m1_inv
    A[1,3] = -FT(2.0) * dot(r_k1k4, s_k1k2) * m1_inv
    A[3,1] = -FT(2.0) * dot(r_k1k2, s_k1k4) * m1_inv
    A[2,3] = -FT(2.0) * dot(r_k1k4, s_k1k3) * m1_inv
    A[3,2] = -FT(2.0) * dot(r_k1k3, s_k1k4) * m1_inv

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
@inline function shake3_angle_kernel!(
        cluster_idx,
        k1s, k2s, k3s,
        dist12s, dist13s, dist23s,
        r_t1::AbstractVector{<:AbstractVector{L}},
        r_t2::AbstractVector{<:AbstractVector{L}},
        ms::AbstractVector{M}, 
        boundary::AbstractBoundary{<:Any, FT},
        dist_tol::L
    ) where {L, M, FT}

    @uniform A_type = typeof(zero(L) * zero(L) / zero(M))
    @uniform A_tmp_type = typeof(zero(M) / (zero(L) * zero(L)))
    @uniform C_type = typeof(zero(L) * zero(L))

    A = @MMatrix zeros(A_type, 3, 3)
    A_tmp = @MMatrix zeros(A_tmp_type, 3, 3)
    C = @MVector zeros(C_type, 3)
    λ = @MVector zeros(M, 3)

    k1 = k1s[cluster_idx] # central atom
    k2 = k2s[cluster_idx]
    k3 = k3s[cluster_idx]

    dist12 = dist12s[cluster_idx]
    dist13 = dist13s[cluster_idx]
    dist23 = dist23s[cluster_idx]

    m1_inv, m2_inv, m3_inv = inv(ms[k1]), inv(ms[k2]), inv(ms[k3]) # uncoalesced read

    r_t2_k1 = r_t2[k1] # uncoalesced read
    r_t2_k2 = r_t2[k2] # uncoalesced read
    r_t2_k3 = r_t2[k3] # uncoalesced read

    r_t1_k1 = r_t1[k1] # uncoalesced read
    r_t1_k2 = r_t1[k2] # uncoalesced read
    r_t1_k3 = r_t1[k3] # uncoalesced read

    r_k1k2  = vector(r_t1_k1, r_t1_k2, boundary)
    r_k1k3  = vector(r_t1_k1, r_t1_k3, boundary)
    r_k2k3  = vector(r_t1_k2, r_t1_k3, boundary)

    # Distance vectors after unconstrained update
    s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
    s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)
    s_k2k3 = vector(r_t2_k2, r_t2_k3, boundary)

    # Matrix element i, j represents interaction of constraint i with constraint j
    A[1,1] = -FT(2.0) * dot(r_k1k2, s_k1k2) * (m1_inv + m2_inv) # Set constraint 1 as between k1-k2
    A[2,2] = -FT(2.0) * dot(r_k1k3, s_k1k3) * (m1_inv + m3_inv) # Set constraint 2 as between k1-k3
    A[3,3] = -FT(2.0) * dot(r_k2k3, s_k2k3) * (m2_inv + m3_inv) # Set constraint 3 as between k2-k3
    A[1,2] = -FT(2.0) * dot(r_k1k3, s_k1k2) * m1_inv
    A[2,1] = -FT(2.0) * dot(r_k1k2, s_k1k3) * m1_inv
    A[1,3] = -FT(2.0) * dot(r_k2k3, s_k1k2) * (-m2_inv)
    A[3,1] = -FT(2.0) * dot(r_k1k2, s_k1k3) * (-m2_inv)
    A[2,3] = -FT(2.0) * dot(r_k2k3, s_k1k3) * m3_inv
    A[3,2] = -FT(2.0) * dot(r_k1k3, s_k2k3) * m3_inv

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
 
function apply_position_constraints!(
        sys::System,
        ca::SHAKE_RATTLE, 
        r_pre_unconstrained_update;
        kwargs...
    )

    backend = get_backend(r_pre_unconstrained_update)

    N12_clusters = length(ca.clusters12)
    N23_clusters = length(ca.clusters23)
    N34_clusters = length(ca.clusters34)
    N_angle_clusters = length(ca.angle_clusters)

    KernelAbstractions.synchronize(backend)

    if N12_clusters > 0
        # 2 atom constraints are solved analytically, no need to iterate
        s2_kernel = shake2_kernel!(backend, ca.gpu_block_size)
        s2_kernel(
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
            ca.dist_tolerance
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
            ca.dist_tolerance
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
            ca.dist_tolerance
        )
    end

    KernelAbstractions.synchronize(backend)

end

function apply_velocity_constraints!(sys::System, ca::SHAKE_RATTLE; kwargs...)
    backend = get_backend(sys.velocities)

    N12_clusters = length(ca.clusters12)
    N23_clusters = length(ca.clusters23)
    N34_clusters = length(ca.clusters34)
    N_angle_clusters = length(ca.angle_clusters)

    KernelAbstractions.synchronize(backend)

    if N12_clusters > 0
        N12_blocks = cld(N12_clusters, ca.gpu_block_size)
        r2_kernel = rattle2_kernel!(backend, N12_blocks, N12_clusters)
        r2_kernel(
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
        r3_kernel = rattle3_kernel!(backend, N23_blocks, N23_clusters)
        r3_kernel(
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
        r4_kernel = rattle4_kernel!(backend, N34_blocks, N34_clusters)
        r4_kernel(
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
        r3_angle_kernel = rattle3_angle_kernel!(backend, N_angle_blocks, N_angle_clusters)
        r3_angle_kernel(
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
