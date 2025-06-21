# The RATTLE Equations are modified from LAMMPS:
# https://github.com/lammps/lammps/blob/develop/src/RIGID/fix_rattle.cpp


# Returns a if condition is true, b otherwise (without branching)
@inline function branchless_select(condition, a, b)
    return condition * a + (!condition) * b
end

@inline function branchless_min(a, b)
    return branchless_select(a <= b, a, b)
end


@inline function solve2x2exactly(λ, A, C)

    determinant = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]

    if determinant == 0.0
        error("SHAKE determinant is zero, cannot solve")
    end

    inv_det = 1.0 / determinant

    λ[1] = inv_det * ((A[2, 2] * C[1]) - (A[1, 2] * C[2]))
    λ[2] = inv_det * ((A[1, 1] * C[2]) - (A[2, 1] * C[1]))

    return λ
end

@inline function solve3x3exactly!(λ, A, A′, C)

    determinant = A[1,1]*A[2,2]*A[3,3] + A[1,2]*A[2,3]*A[3,1] + A[1,3]*A[2,1]*A[3,2] - 
                    A[1,1]*A[2,3]*A[3,2] - A[1,2]*A[2,1]*A[3,3] - A[1,3]*A[2,2]*A[3,1]

    if determinant == 0.0
        error("SHAKE determinant is zero, cannot solve")
    end

    inv_det = 1.0 / determinant

    A′[1,1] = inv_det * (A[2,2]*A[3,3] - A[2,3]*A[3,2])
    A′[1,2] = -inv_det * (A[1,2]*A[3,3] - A[1,3]*A[3,2])
    A′[1,3] = inv_det * (A[1,2]*A[2,3] - A[1,3]*A[2,2])
    A′[2,1] = -inv_det * (A[2,1]*A[3,3] - A[2,3]*A[3,1])
    A′[2,2] = inv_det * (A[1,1]*A[3,3] - A[1,3]*A[3,1])
    A′[2,3] = -inv_det * (A[1,1]*A[2,3] - A[1,3]*A[2,1])
    A′[3,1] = inv_det * (A[2,1]*A[3,2] - A[2,2]*A[3,1])
    A′[3,2] = -inv_det * (A[1,1]*A[3,2] - A[1,2]*A[3,1])
    A′[3,3] = inv_det * (A[1,1]*A[2,2] - A[1,2]*A[2,1])

    # Assumes lambda is set to zeros initially
    for i in 1:3
        λ[i] += A′[i,1] * C[1] + A′[i,2] * C[2] + A′[i,3] * C[3]
    end

    return λ

end

# 2 atoms, 1 constraint
@kernel inbounds=true function rattle2_kernel!(r, v, m, clusters, boundary)

    idx = @index(Global, Linear) # Global Constraint Idx

    if idx <= length(clusters)

        # Step 2 : Perform RATTLE, for a 2 atom cluster we
        # just re-arrange λ = A / c, since they are all scalars.

        k1 = clusters[idx].unique_atoms[1]
        k2 = clusters[idx].unique_atoms[2]

        v_k1 = v[k1] # uncoalesced read
        v_k2 = v[k2] # uncoalesced read
    
        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]  # uncoalesced read
        r_k1k2  = vector(r[k1], r[k2], boundary) # uncoalesced read
        v_k1k2 = v_k2 .- v_k1

        λₖ = -dot(r_k1k2, v_k1k2) / (dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv))

        # Step 3: Update velocities in global memory
        v[k1] -= m1_inv .* λₖ .* r_k1k2
        v[k2] += m2_inv .* λₖ .* r_k1k2
    end
end

# 3 atoms 2 constraints
# Assumes first atom is central atom
@kernel inbounds=true function rattle3_kernel!(r, v, m, clusters, boundary)

    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform NUM_CONSTRAINTS = 0x2


    if idx <= length(clusters)

        # Allocate thread-local memory
        A = zeros(eltype(r), NUM_CONSTRAINTS, NUM_CONSTRAINTS)
        C = zeros(eltype(r), NUM_CONSTRAINTS)
        λ = zeros(eltype(r), NUM_CONSTRAINTS)

        k1 = clusters[idx].unique_atoms[1] # this is assumed to be the central atom
        k2 = clusters[idx].unique_atoms[2]
        k3 = clusters[idx].unique_atoms[3]
     
        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]; m3_inv = 1 / m[k3] # uncoalesced read
        r_k1k2  = vector(r[k1], r[k2], boundary)
        r_k1k3  = vector(r[k1], r[k3], boundary)

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

        solve_2x2exactly!(λ, A, C)

        # Update global memory
        v[k1] -= m1_inv * ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        v[k2] -= m2_inv .* (-λ[1] .* r_k1k2)
        v[k3] -= m3_inv .* (-λ[2] .* r_k1k3)

    end
end

# 4 atoms 3 cosntraints
# Assumes first atom is central atom
@kernel inbounds=true function rattle4_kernel!(r, v, m, clusters, boundary)
    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform NUM_CONSTRAINTS = 0x3

    if idx <= length(clusters)

        # Allocate thread-local memory
        A = zeros(eltype(r), NUM_CONSTRAINTS, NUM_CONSTRAINTS)
        A_tmp = zeros(eltype(r), NUM_CONSTRAINTS, NUM_CONSTRAINTS)
        C = zeros(eltype(r), NUM_CONSTRAINTS)
        λ = zeros(eltype(r), NUM_CONSTRAINTS)

        k1 = clusters[idx].unique_atoms[1] # this is assumed to be the central atom
        k2 = clusters[idx].unique_atoms[2]
        k3 = clusters[idx].unique_atoms[3]
        k4 = clusters[idx].unique_atoms[4]
     
        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]; m3_inv = 1 / m[k3]; m4_inv = 1 / m[k4] # uncoalesced read
        r_k1k2  = vector(r_shared[k1], r_shared[k2], boundary) # uncoalesced read
        r_k1k3  = vector(r_shared[k1], r_shared[k3], boundary) # uncoalesced read
        r_k1k4  = vector(r_shared[k1], r_shared[k4], boundary) # uncoalesced read

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

        solve3x3exactly!(λ, A, A_tmp, C)

        # Update global memory
        v[k1] -= m1_inv * ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3) .+ λ[3] .* r_k1k4)
        v[k2] -= m2_inv .* (-λ[1] .* r_k1k2)
        v[k3] -= m3_inv .* (-λ[2] .* r_k1k3)
        v[k4] -= m4_inv .* (-λ[3] .* r_k1k4)

    end
end

# 3 atoms 3 constraints
@kernel inbounds=true function rattle3_angle_kernel!(r, v, m, clusters, boundary)

    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform NUM_CONSTRAINTS = 0x3

    if idx <= length(clusters)

        # Allocate thread-local memory
        A = zeros(eltype(r), NUM_CONSTRAINTS, NUM_CONSTRAINTS)
        A_tmp = zeros(eltype(r), NUM_CONSTRAINTS, NUM_CONSTRAINTS)
        C = zeros(eltype(r), NUM_CONSTRAINTS)
        λ = zeros(eltype(r), NUM_CONSTRAINTS)

        k1 = clusters[idx].unique_atoms[1] # this is assumed to be the central atom
        k2 = clusters[idx].unique_atoms[2]
        k3 = clusters[idx].unique_atoms[3]
     
        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]; m3_inv = 1 / m[k3] #uncoalesced read
        r_k1k2  = vector(r_shared[k1], r_shared[k2], boundary) # uncoalesced read
        r_k1k3  = vector(r_shared[k1], r_shared[k3], boundary) # uncoalesced read
        r_k2k3  = vector(r_shared[k2], r_shared[k3], boundary) # uncoalesced read

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

        solve_2x2exactly!(λ, A, A_tmp, C)

        # Update global memory
        v[k1] -= m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        v[k2] -= m2_inv .* ((-λ[1] .* r_k1k2) .+ (λ[3] .* r_k2k3))
        v[k3] -= m3_inv .* ((-λ[2] .* r_k1k3) .- (λ[3] .* r_k2k3))

    end
end

# 2 atoms, 1 constraint
@kernel inbounds=true function shake2_kernel!(clusters12, r_t1::T, r_t2::T, m, boundary) where T

    idx = @index(Global, Linear) # Global Constraint Idx
    
    if idx <= length(clusters12)

        k1 = clusters12[idx].unique_atoms[1]
        k2 = clusters12[idx].unique_atoms[2]
        distance = first(clusters12[idx].constraints.dist) # Only 1 cluster

        r_t2_k1 = r_t2[k1] # uncoalesced read
        r_t2_k2 = r_t2[k2] # uncoalesced read
        r_t1_k1 = r_t1[k1] # uncoalesced read
        r_t1_k2 = r_t1[k2] # uncoalesced read
        
        # Vector between the atoms after unconstrained update (s)
        s12 = vector(r_t2_k1, r_t2_k2, boundary) 

        # Vector between the atoms before unconstrained update (r)
        r12 = vector(r_t1_k1, r_t1_k2, boundary)

        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]
        a = (m1_inv + m2_inv)^2 * sum(abs2, r12)
        b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
        c = sum(abs2, s12) - (distance)^2
        D = b^2 - 4*a*c
        
        # Just let the system blow up?? 
        # This usually happens when timestep too larger or over constrained
        if ustrip(D) < 0.0
            error("SHAKE determinant negative: $(D)")
        end

        α1 = (-b + sqrt(D)) / (2*a)
        α2 = (-b - sqrt(D)) / (2*a)
        g = branchless_min(α1, α2)

        # Step 3: Update global memory
        r_t2[k1] += r12 .* (g*m1_inv)
        r_t2[k2] += r12 .* (-g*m2_inv)

    end

end

@inline function A_matrix_branchless!(
    A::AbstractMatrix{T},
    cc::ConstraintCluster,
    masses::AbstractVector{T},
    r::AbstractVector{SVector{3,T}},      # positions at time t
    r_uc::AbstractVector{SVector{3,T}},   # unconstrained pos at t+Δt
    boundary                              
  ) where T

  cons = cc.constraints
  M    = length(cons)

  is = cons.i
  js = cons.j

  @inbounds for a in 1:M
      ia = is[a]
      ja = js[a]

      inv_ia = one(T) / masses[ia]
      inv_ja = one(T) / masses[ja]

      # unconstrained bond‐vector r_uc[a] → SVector{3}
      uc_vec = vector(r_uc[ia], r_uc[ja], boundary)
      ux0, ux1, ux2 = uc_vec[1], uc_vec[2], uc_vec[3]

      @inbounds for b in 1:M
          ib = is[b]
          jb = js[b]

          # branchless Kronecker terms
          bracket = ((ia==ib) - (ia==jb)) * inv_ia +
                    ((ja==jb) - (ja==ib)) * inv_ja

          # real bond‐vector at t
          rt_vec = vector(r[ib], r[jb], boundary)
          rtb0, rtb1, rtb2 = rt_vec[1], rt_vec[2], rt_vec[3]

          # dot & write
          A[a,b] = bracket * (rtb0*ux0 + rtb1*ux1 + rtb2*ux2)
      end
  end

  return A
end

# 3 atoms, 2 constraints
# Constraints between 1-2 and 1-3
@kernel inbounds=true function shake3_kernel!(clusters, r_t1::T, r_t2::T, m, boundary, dt) where T

    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform NUM_CONSTRAINTS = 0x2

    if idx <= length(clusters)

        cluster = clusters[idx] # Type is StructArray{DistanceConstraint}

        # Allocate thread-local memory
        A = zeros(eltype(r), NUM_CONSTRAINTS, NUM_CONSTRAINTS)
        C = zeros(eltype(r), NUM_CONSTRAINTS)
        λ = zeros(eltype(r), NUM_CONSTRAINTS)

        k1 = c.unique_atoms[1] # central atom
        k2 = c.unique_atoms[2]
        k3 = c.unique_atoms[3]

        # distances are ordered in cluster creation
        dist12, dist13 = c.constraints.dist

        r_t2_k1 = r_t2[k1] # uncoalesced read
        r_t2_k2 = r_t2[k2] # uncoalesced read
        r_t2_k3 = r_t2[k3] # uncoalesced read

        r_t1_k1 = r_t1[k1] # uncoalesced read
        r_t1_k2 = r_t1[k2] # uncoalesced read
        r_t1_k3 = r_t1[k3] # uncoalesced read

        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]; m3_inv = 1 / m[k3] # uncoalesced read

        r_k1k2  = vector(r_t1_k1, r_t1_k2, boundary)
        r_k1k3  = vector(r_t1_k1, r_t1_k3, boundary)

        s_k1k2 = vector(r_t2_k1, r_t2_k2, boundary)
        s_k1k3 = vector(r_t2_k1, r_t2_k3, boundary)

        # A matrix element (i,j) represents interaction of constraint i with constraint j.
        A[1,1] = dot(r_k1k2, s_k1k2) * (m1_inv + m2_inv) # this sets constraint 1 as between k1-k2
        A[2,2] = dot(r_k1k3, s_k1k3) * (m1_inv + m3_inv) # this sets constraint 1 as between k1-k3
        A[1,2] = dot(r_k1k3, s_k1k2) * m1_inv
        A[2,1] = dot(r_k1k2, s_k1k3) * m1_inv

        A = A_matrix_branchless!(A, cluster, m, r_t1, r_t2, boundary)

        denom = 4*dt*dt
        C[1] = dot(s_k1k2, s_k1k2) - (dist12*dist12) / denom
        C[2] = dot(s_k1k3, s_k1k3) - (dist13*dist13) / denom

        solve_2x2exactly!(λ, A, C)

        # Step 3: Update global memory
        r_t2[k1] -= m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        r_t2[k2] -= m2_inv .* (-λ[1] .* r_k1k2)
        r_t2[k3] -= m3_inv .* (-λ[2] .* r_k1k3)
    end
    

end

# 4 atoms, 3 constraints
@kernel inbounds=true function shake4_kernel!(clusters)

    tid = @index(Global) # constraint index
    
    #* Iterate inside kernel?

end

# 3 atoms, 3 constraints
@kernel inbounds=true function shake3_angle_kernel!()

end
 
function apply_position_constraints!(
        sys::System,
        ca::SHAKE_RATTLE, 
        r_pre_unconstrained_update
    )

    backend = get_backend(r_pre_unconstrained_update)
    N_clusters = length(ca.clusters12)
    N_blocks = cld(N_clusters, ca.gpu_block_size)
    s2_kernel = shake2_kernel!(backend, N_blocks, N_clusters)
    N_clusters > 0 && s2_kernel(ca.clusters12, r_pre_unconstrained_update, sys.coords, sys.masses, sys.boundary, ndrange = N_clusters)


    #* TODO LAUNCH ON SEPARATE STREAMS/TASKS
    # shake3_kernel!()
    # shake4_kernel!()

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

function apply_velocity_constraints!(sys::System, ca::SHAKE_RATTLE)

    backend = get_backend(sys.velocities)

    N12_clusters = length(ca.clusters12)
    N12_blocks = cld(N12_clusters, ca.gpu_block_size)
    r2_kernel = rattle2_kernel!(backend, N12_blocks, N12_clusters)

    N23_clusters = length(ca.clusters23)
    N23_blocks = cld(N23_clusters, ca.gpu_block_size)
    r3_kernel = rattle3_kernel!(backend, N23_blocks, N23_clusters)

    N34_clusters = length(ca.clusters34)
    N34_blocks = cld(N34_clusters, ca.gpu_block_size)
    r4_kernel = rattle4_kernel!(backend, N34_blocks, N34_clusters)

    N_angle_clusters = length(ca.angle_clusters)
    N_angle_blocks = cld(N_angle_clusters, ca.gpu_block_size)
    r3_angle_kernel = rattle3_angle_kernel!(backend, N_angle_blocks, N_angle_clusters)


    #* TODO LAUNCH ON SEPARATE STREAMS/TASKS
    N12_clusters > 0 && r2_kernel(sys.coords, sys.velocities, sys.masses, ca.clusters12, sys.boundary, ndrange = N12_clusters)
    N23_clusters > 0 && r3_kernel(sys.coords, sys.velocities, sys.masses, ca.clusters23, sys.boundary, ndrange = N23_clusters)
    N34_clusters > 0 && r4_kernel(sys.coords, sys.velocities, sys.masses, ca.clusters34, sys.boundary, ndrange = N34_clusters)
    N_angle_clusters > 0 && r3_angle_kernel(sys.coords, sys.velocities, sys.masses, ca.angle_clusters, sys.boundary, ndrange = N_angle_clusters)
end
