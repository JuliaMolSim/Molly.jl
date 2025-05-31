
# Returns a if condition is true, b otherwise (without branching)
@inline function branchless_select(condition, a, b)
    return condition * a + (!condition) * b
end

@inline function branchless_min(a, b)
    return branchless_select(a <= b, a, b)
end

# From LAMMPS: https://github.com/lammps/lammps/blob/2744647c75f065f259845c3636182600c9ce00c9/src/RIGID/fix_rattle.cpp#L517
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

#From LAMMPS: https://github.com/lammps/lammps/blob/2744647c75f065f259845c3636182600c9ce00c9/src/RIGID/fix_rattle.cpp#L539
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
@kernel inbounds=true function rattle2_kernel!(r, v, m, boundary, N_constraints)

    idx = @index(Global, Linear) # Global Constraint Idx
    tidx = @index(Local, Linear)
    @uniform CONSTRAINTS_PER_BLOCK = @groupsize()[1]
    @uniform ATOMS_PER_CLUSTER = 0x2 
    @uniform D = n_dimensions(boundary)

    r_shared = @localmem eltype(r) (ATOMS_PER_CLUSTER * CONSTRAINTS_PER_BLOCK, D)
    v_shared = @localmem eltype(v) (ATOMS_PER_CLUSTER * CONSTRAINTS_PER_BLOCK, D)

    # Step 1: Move velocities to shared memory to avoid uncoalesced access
    # Positions are pre-sorted such that atoms in 1-constraint clusters are
    # at the start. Furthermore, the atoms are sorted within that sector
    # such that the 2 atoms in constraint 1 are the first two elements and so on

    for i in 1:ATOMS_PER_CLUSTER
        global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
        shared_idx = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + i 
        r_shared[shared_idx] .= r[global_atom_idx]  #* does this work with .= syntax?
        v_shared[shared_idx] .= v[global_atom_idx]
    end

     # Ensure all data is loaded into shared memory
     @synchronize()

    if idx <= N_constraints

        # Step 2 : Perform RATTLE, for a 2 atom cluster we
        # just re-arrange λ = A / c, since they are all scalars.

        k1 = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + 0x1
        k2 = k1 + 0x1
     
        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]
        r_k1k2  = vector(r_shared[k1], r_shared[k2], boundary)
        v_k1k2 = v_shared[k2] .- v_shared[k1]

        λₖ = -dot(r_k1k2, v_k1k2) / (dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv))
        v_shared[k1] -= m1_inv .* λₖ .* r_k1k2
        v_shared[k2] += m2_inv .* λₖ .* r_k1k2

        # Step 3: Write positions back to global memory
        for i in 1:ATOMS_PER_CLUSTER
            global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
            shared_idx = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + i 
            v[global_atom_idx] .= v_shared[shared_idx]
        end

    end

end

# 3 atoms 2 constraints
@kernel inbounds=true function rattle3_kernel!()
    idx = @index(Global, Linear) # Global Constraint Idx
    tidx = @index(Local, Linear)
    @uniform CONSTRAINTS_PER_BLOCK = @groupsize()[1]
    @uniform NUM_CONSTRAINTS = 0x2
    @uniform ATOMS_PER_CLUSTER = 0x3
    @uniform D = n_dimensions(boundary)

    r_shared = @localmem eltype(r) (ATOMS_PER_CLUSTER * CONSTRAINTS_PER_BLOCK, D)
    v_shared = @localmem eltype(v) (ATOMS_PER_CLUSTER * CONSTRAINTS_PER_BLOCK, D)

    # Step 1: Move velocities to shared memory to avoid uncoalesced access
    # Positions are pre-sorted such that atoms in 1-constraint clusters are
    # at the start. Furthermore, the atoms are sorted within that sector
    # such that the 2 atoms in constraint 1 are the first two elements and so on

    for i in 1:ATOMS_PER_CLUSTER
        global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
        shared_idx = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + i 
        r_shared[shared_idx] .= r[global_atom_idx]  #* does this work with .= syntax?
        v_shared[shared_idx] .= v[global_atom_idx]
    end

     # Ensure all data is loaded into shared memory
     @synchronize()

    if idx <= N_constraints

        # Allocate thread-local memory
        A = zeros(eltype(r), NUM_CONSTRAINTS, NUM_CONSTRAINTS)
        C = zeros(eltype(r), NUM_CONSTRAINTS)
        λ = zeros(eltype(r), NUM_CONSTRAINTS)

        k1 = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + 0x1
        k2 = k1 + 0x1
        k3 = k1 + 0x2
     
        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]; m3_inv = 1 / m[k3]
        r_k1k2  = vector(r_shared[k1], r_shared[k2], boundary)
        r_k1k3  = vector(r_shared[k1], r_shared[k3], boundary)

        v_k1k2 = v_shared[k2] .- v_shared[k1]
        v_k1k3 = v_shared[k3] .- v_shared[k1]

        A[1, 1] = dot(r_k1k2, r_k1k2) * (m1_inv + m2_inv)
        A[1, 2] = dot(r_k1k2, r_k1k3) * (m1_inv)
        A[2, 1] = A[1, 2]
        A[2, 2] = dot(r_k1k3, r_k1k3) * (m1_inv + m3_inv)

        #* HOW TO ALLOCATE LAMBDA AND C LOCALLY PER THREAD??
        C[1] = -dot(r_k1k2, v_k1k2)
        C[2] = -dot(r_k1k3, v_k1k3)

        solve_2x2exactly!(λ, A[tid], C)

        v_shared[k1] -= m1_inv * ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        v_shared[k2] -= m2_inv .* (λ[1] .* r_k1k2)
        v_shared[k3] -= m3_inv .* (λ[2] .* r_k1k3)

        # Step 3: Write positions back to global memory
        for i in 1:ATOMS_PER_CLUSTER
            global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
            shared_idx = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + i 
            v[global_atom_idx] .= v_shared[shared_idx]
        end

    end
end

# 4 atoms 3 cosntraints
@kernel inbounds=true function rattle4_kernel!()
    
end

# 3 atoms 3 constraints
@kernel inbounds=true function rattle3_angle_kernel!()

end

# 2 atoms, 1 constraint
@kernel inbounds=true function shake2_kernel!(distances, r_t1::T, r_t2::T, m, boundary)

    idx = @index(Global, Linear) # Global Constraint Idx
    tidx = @index(Local, Linear)
    @uniform CONSTRAINTS_PER_BLOCK = @groupsize()[1]
    @uniform CONSTRAINT_SIZE = 0x2
    D = n_dimensions(boundary)
     
    r_t1_shared = @localmem eltype(r_t1) (CONSTRAINT_SIZE * CONSTRAINTS_PER_BLOCK, D)
    r_t2_shared = @localmem eltype(r_t1) (CONSTRAINT_SIZE * CONSTRAINTS_PER_BLOCK, D)

    if idx <= length(distances)

        # Step 1: Move positions to shared memory to avoid uncoalesced access
        # Positions are pre-sorted such that atoms in 1-constraint clusters are
        # at the start. Furthermore, the atoms are sorted within that sector
        # such that the 2 atoms in constraint 1 are the first two elements and so on

        for i in 1:CONSTRAINT_SIZE
            global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
            shared_idx = (CONSTRAINT_SIZE * (tidx - 0x1)) + i 
            r_t1_shared[shared_idx] .= r_t1[global_atom_idx]  #* does this work with .= syntax?
            r_t2_shared[shared_idx] .= r_t2[global_atom_idx]
        end

        # Ensure all data is loaded into shared memory
        @synchronize()

        # Step 2 : Perform SHAKE, for a 2 atom cluster we can use 
        # the quadratic formula to get an analytical solution

        k1 = (CONSTRAINT_SIZE * (tidx - 0x1)) + 0x1
        k2 = k1 + 0x1
        
        # Vector between the atoms after unconstrained update (s)
        s12 = vector(r_t2_shared[k1], r_t2_shared[k2], boundary) #* WILL JULIA UNDERSTAND TO ALLOCATE VECTORS LIKE THIS??

        # Vector between the atoms before unconstrained update (r)
        r12 = vector(r_t1_shared[k1], r_t1_shared[k2], boundary)

        m1_inv = 1 / m[k1]; m2_inv = 1 / m[k2]
        a = (m1_inv + m2_inv)^2 * sum(abs2, r12)
        b = -2 * (m1_inv + m2_inv) * dot3(r12, s12)
        c = sum(abs2, s12) - (distances[idx])^2
        D = b^2 - 4*a*c

        # Just let the system blow up?? 
        # This usually happens when timestep too larger or over constrained
        if D < 0.0
            @error "SHAKE determinant negative"
        end

        α1 = (-b + sqrt(D))
        α2 = (-b - sqrt(D))
        g = branchless_min(α1, α2) / (2*a)

        r_t2_shared[k1] += r12 .* (-g*m1_inv)
        r_t2_shared[k2] += r12 .* (g*m2_inv)

        # Step 3: Write positions back to global memory
        for i in 1:CONSTRAINT_SIZE
            global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
            shared_idx = (CONSTRAINT_SIZE * (tidx - 0x1)) + i 
            r_t2[global_atom_idx] .= r_t2_shared[shared_idx]
        end
    end

end

# 3 atoms, 2 constraints
@kernel inbounds=true function shake3_kernel!(three_atom_clusters::AbstractVector{ConstraintCluster{3}})

    tid = @index(Global) # constraint index


    #* Still need to pass clusters to know which atoms to update
    #* as constraints could be between (1,2) and (2,3) or (1,3) and (2,3) etc.
    

end

# 4 atoms, 3 constraints
@kernel inbounds=true function shake4_kernel!(four_atom_clusters::AbstractVector{ConstraintCluster{4}})

    tid = @index(Global) # constraint index
    
    #* this will require extra shared memory for analytical matrix inversion

end

# 3 atoms, 3 constraints
@kernel inbounds=true function shake3_angle_kernel!()

end

function apply_position_cosntraints!(sys, ca::SHAKE_RATTLE, r_pre_unconstrained_update)


    # Organize coordinates according to constraints
    # this enables coalesced memory access on GPU.
    # Sorting is about 3x faster than doing data[ordering]
    # This faster on CPU for systems with fewer than ~200,000 atoms...
    # but its slow to move things to GPU.
    #* MAKE SURE I DO NOT MODIFY ca.coord_ordering!!!
    AK.merge_sort_by_key!(ca.coord_ordering, sys.coords, ca.buffers.keys, ca.buffers.coords)
    AK.merge_sort_by_key!(ca.coord_ordering, r_pre_unconstrained_update, ca.buffers.keys, ca.buffers.coords)

    shake2_kernel!(ca.clusters1.dist, r_pre_unconstrained_update, sys.coords, sys.masses, sys.boundary)

    #* TODO LAUNCH ON SEPARATE STREAMS/TASKS
    # shake3_kernel!()
    # shake4_kernel!()

    # Re-vert coordinate ordering
    AK.merge_sort_by_key!(ca.inv_ordering, sys.coords, ca.buffers.keys, ca.buffers.coords)

end

function apply_velocity_constraints!(sys, ca::SHAKE_RATTLE)

end