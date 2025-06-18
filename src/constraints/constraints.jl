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
@kernel inbounds=true function rattle2_kernel!(r, v, m, clusters, N_constraints)

    idx = @index(Global, Linear) # Global Constraint Idx

    if idx <= N_constraints

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
        v_k1 -= m1_inv .* λₖ .* r_k1k2
        v_k2 += m2_inv .* λₖ .* r_k1k2

        # Step 3: Write positions back to global memory
        v[k1] .= v_k1 # uncoalesced write
        v[k2] .= v_k2 # uncoalesced write
    end
end

# 3 atoms 2 constraints
# Assumes first atom is central atom
@kernel inbounds=true function rattle3_kernel!(r, v, m, clusters, N_constraints)

    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform NUM_CONSTRAINTS = 0x2


    if idx <= N_constraints

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

        solve_2x2exactly!(λ, A[tid], C)

        v_k1 -= m1_inv * ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        v_k2 -= m2_inv .* (-λ[1] .* r_k1k2)
        v_k3 -= m3_inv .* (-λ[2] .* r_k1k3)

        # Write positions back to global memory
        v[k1] = v_k1 # uncoalesced write
        v[k2] = v_k2 # uncoalesced write
        v[k3] = v_k3 # uncoalesced write

    end
end

# 4 atoms 3 cosntraints
# Assumes first atom is central atom
@kernel inbounds=true function rattle4_kernel!(r, v, m, clusters, N_constraints)
    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform NUM_CONSTRAINTS = 0x3

    if idx <= N_constraints

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

        vk1 -= m1_inv * ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3) .+ λ[3] .* r_k1k4)
        vk2 -= m2_inv .* (-λ[1] .* r_k1k2)
        vk3 -= m3_inv .* (-λ[2] .* r_k1k3)
        vk4 -= m4_inv .* (-λ[3] .* r_k1k4)

        v[k1] = vk1 # uncoalesced write
        v[k2] = vk2 # uncoalesced write
        v[k3] = vk3 # uncoalesced write
        v[k4] = vk4 # uncoalesced write

    end
end

# 3 atoms 3 constraints
@kernel inbounds=true function rattle3_angle_kernel!(r, v, m, clusters, N_constraints)

    idx = @index(Global, Linear) # Global Constraint Idx
    @uniform NUM_CONSTRAINTS = 0x3

    if idx <= N_constraints

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

        v_k1 -= m1_inv .* ((λ[1] .* r_k1k2) .+ (λ[2] .* r_k1k3))
        v_k2 -= m2_inv .* ((-λ[1] .* r_k1k2) .+ (λ[3] .* r_k2k3))
        v_k3 -= m3_inv .* ((-λ[2] .* r_k1k3) .- (λ[3] .* r_k2k3))

        # Write positions back to global memory
        v[k1] = k1
        v[k2] = k3
        v[k3] = k3

    end
end

# 2 atoms, 1 constraint
@kernel inbounds=true function shake2_kernel!(distances, r_t1::T, r_t2::T, m, D) where T

    idx = @index(Global, Linear) # Global Constraint Idx
    tidx = @index(Local, Linear)
    @uniform CONSTRAINTS_PER_BLOCK = @groupsize()[1]
    @uniform ATOMS_PER_CLUSTER = 0x2
     
    r_t1_shared = @localmem eltype(r_t1) (ATOMS_PER_CLUSTER * CONSTRAINTS_PER_BLOCK, D)
    r_t2_shared = @localmem eltype(r_t2) (ATOMS_PER_CLUSTER * CONSTRAINTS_PER_BLOCK, D)

    if idx <= length(distances)

        # Step 1: Move positions to shared memory to avoid uncoalesced access
        # Positions are pre-sorted such that atoms in 1-constraint clusters are
        # at the start. Furthermore, the atoms are sorted within that sector
        # such that the 2 atoms in constraint 1 are the first two elements and so on

        for i in 1:ATOMS_PER_CLUSTER
            global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
            shared_idx = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + i 
            r_t1_shared[shared_idx] .= r_t1[global_atom_idx]  #* does this work with .= syntax?
            r_t2_shared[shared_idx] .= r_t2[global_atom_idx]
        end

        # Ensure all data is loaded into shared memory
        @synchronize()

        # Step 2 : Perform SHAKE, for a 2 atom cluster we can use 
        # the quadratic formula to get an analytical solution

        k1 = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + 0x1
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
        for i in 1:ATOMS_PER_CLUSTER
            global_atom_idx = idx + ((i - 0x1) * CONSTRAINTS_PER_BLOCK)
            shared_idx = (ATOMS_PER_CLUSTER * (tidx - 0x1)) + i 
            r_t2[global_atom_idx] .= r_t2_shared[shared_idx]
        end
    end

end

# 3 atoms, 2 constraints
@kernel inbounds=true function shake3_kernel!(clusters)

    tid = @index(Global) # constraint index


    #* Iterate inside kernel?
    

end

# 4 atoms, 3 constraints
@kernel inbounds=true function shake4_kernel!(clusters)

    tid = @index(Global) # constraint index
    
    #* Iterate inside kernel?

end

# 3 atoms, 3 constraints
@kernel inbounds=true function shake3_angle_kernel!()

end

function apply_position_constraints!(sys::System, ca::SHAKE_RATTLE, r_pre_unconstrained_update)



    shake2_kernel!(ca.clusters1.dist, r_pre_unconstrained_update, sys.coords, sys.masses, sys.boundary)

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
    #* TODO LAUNCH ON SEPARATE STREAMS/TASKS
    rattle2_kernel!(sys.coords, sys.velocities, sys.masses, ca.clusters1, length(ca.clusters1))
    rattle3_kernel!(sys.coords, sys.velocities, sys.masses, ca.clusters2, length(ca.clusters2))
    rattle4_kernel!(sys.coords, sys.velocities, sys.masses, ca.clusters3, length(ca.clusters3))
    rattle3_angle_kernel!(sys.coords, sys.velocities, sys.masses, ca.angle_clusters, length(ca.angle_clusters))
end
