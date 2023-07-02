export SHAKE

"""
    SHAKE(coords, tolerance)

Constrains a set of bonds to defined distances.
"""
struct SHAKE{UC, T} <: ConstraintAlgorithm
    coord_storage::UC #Used as storage to avoid re-allocating arrays
    tolerance::T
end

function SHAKE(coord_storage; tolerance=1e-4)
    return SHAKE{typeof(coord_storage), typeof(tolerance)}(
        coord_storage, tolerance)
end

save_positions!(constraint_algo::SHAKE, c) = (constraint_algo.coord_storage .= c)
save_velocities!(constraint_algo::SHAKE, v) = constraint_algo

apply_velocity_constraint!(sys, constraint_algo::SHAKE, constraint_cluster::ConstraintCluster) = nothing

function apply_position_constraint!(sys, constraint_algo::SHAKE, 
    constraint_cluster::ConstraintCluster{1}, accels, dt)

    SHAKE_update!(sys, constraint_algo, constraint_cluster, accels, dt)

end

#Assumes NVE update
function unconstrained_update!(constraint_algo, sys, accels, dt)

    #Move system coordinates into temporary storage
    save_positions!(constraint_algo, sys.coords)

    #Unconstrained update on stored coordinates
    constraint_algo.coord_storage .+= (sys.velocities .* dt) .+ (accel_remove_mol.(accels) .* dt ^ 2) ./ 2

    return constraint_algo
end

#TODO: I do not think we actually need to iterate here its analytical solution
#TODO: Modify forces instead of positions?
function SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{1}, accels, dt)

    constraint = cluster.constraints[1]

    # Index of atoms in bond k
    k1, k2 = constraint.atom_idxs

    # Distance vector after unconstrained update (s)
    s01 = vector(ca.coord_storage[k2], ca.coord_storage[k1], sys.boundary)

    # Distance vector between the atoms before unconstrained update (r)
    r01 = vector(sys.coords[k2], sys.coords[k1], sys.boundary)


    if ustrip(abs(norm(s01) - constraint.dist)) > ca.tolerance

        m1 = mass(sys.atoms[k1])
        m2 = mass(sys.atoms[k2])
        a = (1/m1 + 1/m2)^2 * norm(r01)^2
        b = 2 * (1/m1 + 1/m2) * dot(r01, s01)
        c = norm(s01)^2 - ((constraint.dist)^2)
        D = (b^2 - 4*a*c)
        
        if ustrip(D) < 0.0
            @warn "SHAKE determinant negative: $D, setting to 0.0"
            # throw(error())
            D = zero(D)
        end

        # Quadratic solution for g = 2*λ*dt^2
        α1 = (-b + sqrt(D)) / (2*a)
        α2 = (-b - sqrt(D)) / (2*a)

        g = abs(α1) <= abs(α2) ? α1 : α2


        lambda = g/(2*(dt^2))
        # println(lambda)
        # println(unit((g*r01/m1)[1]))
        # println(unit(accels[k1][1]))
        # println(unit(((lambda/m1).*r01)[1]))
        # println(uconvert.(unit(accels[k1][1]), ((lambda/m1).*r01)))
        #TODO get unitful to play nice
        accels[k1] += ustrip.((1/418.4).*(lambda/m1).*r01) * unit(accels[k1][1])
        accels[k2] -= ustrip.((1/418.4).*(lambda/m2).*r01) * unit(accels[k2][1])

        # println(abs(norm(vector(sys.coords[k2], sys.coords[k1], sys.boundary)) - constraint.dist))
    end

end

# TODO: Manually implement matrix inversion
SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{2}) = nothing
SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{3}) = nothing
SHAKE_update!(sys, ca::SHAKE, cluster::ConstraintCluster{4}) = nothing

#Implement later, see:
# https://onlinelibrary.wiley.com/doi/abs/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/
#Currently code is setup for independent constraints, but M-SHAKE does not care about that
# SHAKE_update!(sys, cluster::ConstraintClusterP{D}) where {D >= 5} = nothing
