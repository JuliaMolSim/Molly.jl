export SHAKE

"""
    SHAKE(coords, tolerance, init_posn_tol)

A constraint algorithm to set bonds distances in a simulation.

# Arguments
- coords: An empty array with same type and size as coords in sys, `similar(sys.coords)` is best
- tolerance: Tolerance used to end iterative procedure when calculating constraint forces. This
    is not a tolerance on the error in positions or velocities, but a lower `tolerance` should
    result in smaller error. Default is `1e-4`.
- init_posn_tol: Tolerance used when checking if system initial positions satisfy position constraints. 
    Default is `nothing.`
"""
struct SHAKE{UC, T, I} <: PositionConstraintAlgorithm
    coord_storage::UC #Used as storage to avoid re-allocating arrays
    tolerance::T
    init_posn_tol::Union{I,Nothing}
end

function SHAKE(coord_storage; tolerance=1e-4, init_posn_tol = nothing)
    return SHAKE{typeof(coord_storage), typeof(tolerance), typeof(init_posn_tol)}(
        coord_storage, tolerance, init_posn_tol)
end

save_positions!(constraint_algo::SHAKE, c) = (constraint_algo.coord_storage .= c)
save_velocities!(constraint_algo::SHAKE, v) = constraint_algo

apply_velocity_constraint!(sys::System, constraint_algo::SHAKE, constraint_cluster::ConstraintCluster) = nothing

function apply_position_constraint!(sys::System, constraint_algo::SHAKE, 
    constraint_cluster::ConstraintCluster{1})

    SHAKE_update!(sys, constraint_algo, constraint_cluster)

end

# Done in simulators now
# function unconstrained_position_update!(constraint_algo, sys, accels, dt)

#     #Move system coordinates into temporary storage
#     save_positions!(constraint_algo, sys.coords)

#     #Unconstrained update on stored coordinates
#     constraint_algo.coord_storage .+= (sys.velocities .* dt) .+ (accels .* dt ^ 2) ./ 2

#     return constraint_algo
# end

#TODO: I do not think we actually need to iterate here its analytical solution
#TODO: Modify forces instead of positions?
function SHAKE_update!(sys, ca::Union{SHAKE,RATTLE}, cluster::ConstraintCluster{1})

    constraint = cluster.constraints[1]

    # Index of atoms in bond k
    k1, k2 = constraint.atom_idxs

    # Distance vector after unconstrained update (s)
    s12 = vector(sys.coords[k2], sys.coords[k1], sys.boundary) #& extra allocation

    # Distance vector between the atoms before unconstrained update (r)
    r12 = vector(ca.coord_storage[k2], ca.coord_storage[k1], sys.boundary) #& extra allocation


    m1_inv = 1/mass(sys.atoms[k1])
    m2_inv = 1/mass(sys.atoms[k2])
    a = (m1_inv + m2_inv)^2 * norm(r12)^2 #* can remove sqrt in norm here
    b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
    c = norm(s12)^2 - ((constraint.dist)^2) #* can remove sqrt in norm here
    D = (b^2 - 4*a*c)
    
    if ustrip(D) < 0.0
        @warn "SHAKE determinant negative: $D, setting to 0.0"
        D = zero(D)
    end

    # Quadratic solution for g = 2*λ*dt^2
    α1 = (-b + sqrt(D)) / (2*a)
    α2 = (-b - sqrt(D)) / (2*a)

    g = abs(α1) <= abs(α2) ? α1 : α2 #* why take smaller one?

    # Update positions
    δri1 = r12 .* (-g*m1_inv)
    δri2 = r12 .* (g*m2_inv)

    sys.coords[k1] += δri1
    sys.coords[k2] += δri2


    #Version that modifies accelerations
    # lambda = g/(2*(dt^2))

    # accels[k1] += (lambda/m1).*r12
    # accels[k2] -= (lambda/m2).*r12

end

# TODO: Manually implement matrix inversion when doing this
SHAKE_update!(sys, ca::Union{SHAKE,RATTLE}, cluster::ConstraintCluster{2}) = nothing
SHAKE_update!(sys, ca::Union{SHAKE,RATTLE}, cluster::ConstraintCluster{3}) = nothing
SHAKE_update!(sys, ca::Union{SHAKE,RATTLE}, cluster::ConstraintCluster{4}) = nothing

#Implement later, see:
# https://onlinelibrary.wiley.com/doi/abs/10.1002/1096-987X(20010415)22:5%3C501::AID-JCC1021%3E3.0.CO;2-V
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3285512/
#Currently code is setup for independent constraints, but M-SHAKE does not care about that
# SHAKE_update!(sys, ca::Union{SHAKE,RATTLE}, cluster::ConstraintCluster{D}) where {D >= 5} = nothing
