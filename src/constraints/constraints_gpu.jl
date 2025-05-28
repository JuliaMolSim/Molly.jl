
# Returns a if condition is true, b otherwise (without branching)
@inline function branchless_select(condition, a, b)
    return condition * a + (!condition) * b
end

@inline function branchless_max(a, b)
    return branchless_select(a > b, a, b)
end

@inline function dot3(a, b)
    return a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

@kernel function shake_kernel!(constraints::AbstractVector{DistanceConstraint}, num_active_constraints)

    idx = @index(Global) # Index of constraint in the global context
    
    if idx > num_active_constraints return end


    
end


#* Launch SHAKE AND RATTLE SIMULTANEOUSLY IF POSSIBLE?

#* IMPLE
function apply_position_cosntraints!(sys, ca::SHAKE_RATTLE, coord_storage)


    # 1. Launch one thread per constraint
    # 2. Move r(t) , r(t + dt) to shared memory, allocate s in shared memory. 
    # 2. Run SHAKE, update positions atomically
    # 3. Check convergence
    # 4. Global?? compaction of positions, re-order so active constraints are at the front 
        # to reduce warp divergence
    # 5. Repeat until all converged

    #* STILL NEED TO DO THE RE-ORDERING OF THE POSITIONS AND VELOCITIES
    constrained_coords = @views sys.coords[1:ca.n_constrainted_atoms]

    shake_kernel!()

end