
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

@kernel inbounds=true function shake2_kernel!(distances, r_t1::T, r_t2::T, boundary)

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

@kernel inbounds=true function shake3_kernel!(three_atom_clusters::AbstractVector{ConstraintCluster{3}})

    tid = @index(Global) # constraint index
    
    

end

@kernel inbounds=true function shake4_kernel!(four_atom_clusters::AbstractVector{ConstraintCluster{4}})

    tid = @index(Global) # constraint index
    
    #* this will require extra shared memory for analytical matrix inversion

end


#* Launch SHAKE AND RATTLE SIMULTANEOUSLY IF POSSIBLE?

#* IMPLE
function apply_position_cosntraints!(sys, ca::SHAKE_RATTLE, coord_storage)

    #* TODO LAUNCH ON SEPARATE STREAMS/TASKS
    shake2_kernel!()
    shake3_kernel!()
    shake4_kernel!()

end