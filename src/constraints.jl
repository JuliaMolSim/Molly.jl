# Bond and angle constraints

export BondConstraint,
    CCMAConstraints,
    constraincoordinates!,
    constrainvelocities!

"""
    BondConstraint(i, j, distance, reduced_mass)

A constraint that required two atoms to be a certain distance apart.
"""
struct BondConstraint{D, M}
    i::Int
    j::Int
    distance::D
    reduced_mass::M
end

"""
    CCMAConstraints(atoms, atoms_data, bonds, angles, coords, box_size)
    CCMAConstraints(bond_constraints, inv_K)

A set of bond and angle constraints to be constrained using the Constraint
Matrix Approximation method.
See Eastman and Pande 2010.
"""
struct CCMAConstraints{B, M}
    bond_constraints::Vector{B}
    inv_K::M
end

function CCMAConstraints(atoms, atoms_data, bonds, angles, coords, box_size; cutoff=0.01)
    T = Float64
    elements = [at.element for at in atoms_data]
    masses = mass.(atoms)

    bond_constraints = BondConstraint[]
    angle_constraints = Tuple{Int, Int, Int}[]

    for bond in bonds
        if elements[bond.i] == "H" || elements[bond.j] == "H"
            i, j = sort([bond.i, bond.j])
            dist = norm(vector(coords[i], coords[j], box_size))
            reduced_mass = inv(2 * (inv(masses[i]) + inv(masses[j])))
            push!(bond_constraints, BondConstraint(i, j, dist, reduced_mass))
        end
    end

    for angle in angles
        n_hydrogen = 0
        if elements[angle.i] == "H"
            n_hydrogen += 1
        end
        if elements[angle.k] == "H"
            n_hydrogen += 1
        end
        if n_hydrogen == 2 || (n_hydrogen == 1 && elements[angle.j] == "O")
            i, k = sort([angle.i, angle.k])
            push!(angle_constraints, (i, angle.j, k))
        end
    end

    n_constraints = length(bond_constraints)
    K = spzeros(T, n_constraints, n_constraints)
    for ci in 1:n_constraints
        for cj in 1:n_constraints
            cj_inds = (bond_constraints[cj].i, bond_constraints[cj].j)
            if ci == cj
                K[ci, ci] = one(T)
            elseif bond_constraints[ci].i in cj_inds || bond_constraints[ci].j in cj_inds
                if bond_constraints[ci].i in cj_inds
                    shared_ind = bond_constraints[ci].i
                    other_ind_i = bond_constraints[ci].j
                else
                    shared_ind = bond_constraints[ci].j
                    other_ind_i = bond_constraints[ci].i
                end
                if shared_ind == bond_constraints[cj].i
                    other_ind_j = bond_constraints[cj].j
                else
                    other_ind_j = bond_constraints[cj].i
                end
                sort_other_ind_1, sort_other_ind_2 = sort([other_ind_i, other_ind_j])
                angle_constraint_ind = findfirst(isequal((sort_other_ind_1, shared_ind, sort_other_ind_2)), angle_constraints)
                if isnothing(angle_constraint_ind)
                    # If the angle is unconstrained, use the equilibrium angle of the harmonic force term
                    angle_force_ind = findfirst(angles) do a
                        (sort_other_ind_1, shared_ind, sort_other_ind_2) in ((a.i, a.j, a.k), (a.k, a.j, a.i))
                    end
                    isnothing(angle_force_ind) && error("No angle term found for atoms ", (sort_other_ind_1, shared_ind, sort_other_ind_2))
                    cos_θ = cos(angles[angle_force_ind].th0)
                else
                    # If the angle is constrained, use the actual constrained angle
                    ba = vector(coords[shared_ind], coords[other_ind_i], box_size)
                    bc = vector(coords[shared_ind], coords[other_ind_j], box_size)
                    cos_θ = dot(ba, bc) / (norm(ba) * norm(bc))
                end
                mass_term = inv(masses[shared_ind]) / (inv(masses[shared_ind]) + inv(masses[other_ind_i]))
                K[ci, cj] = mass_term * cos_θ
            end
        end
    end

    bond_constraints = [bond_constraints...]
    inv_K = inv(Array(K)) # Could do this with QR
    inv_K = sparse((abs.(inv_K) .> cutoff) .* inv_K)
    return CCMAConstraints(bond_constraints, inv_K)
end

#
constraincoordinates!(coords, coords_prev, constraints, atoms, box_size; kwargs...) = applyconstraints!(
    coords, coords_prev, constraints, atoms, box_size, false; kwargs...)

#
constrainvelocities!( vels  , coords_prev, constraints, atoms, box_size; kwargs...) = applyconstraints!(
    vels  , coords_prev, constraints, atoms, box_size, true ; kwargs...)

function applyconstraints!(coords_or_vels,
                            coords_prev,
                            constraints,
                            atoms,
                            box_size,
                            constraining_velocities;
                            tolerance=1e-5,
                            max_n_iters=150)
    bond_constraints = constraints.bond_constraints
    inv_K = constraints.inv_K
    n_constraints = length(bond_constraints)
    vecs_start = [vector(coords_prev[bc.j], coords_prev[bc.i], box_size) for bc in bond_constraints]
    r2s_start = sum.(abs2, vecs_start)

    lower_tol = 1.0 - 2 * tolerance + tolerance ^ 2
    upper_tol = 1.0 + 2 * tolerance + tolerance ^ 2
    T = typeof(mass(first(atoms)) * first(first(coords_or_vels)) / first(first(coords_prev)))

    for iter_i in 1:max_n_iters
        n_converged = 0
        deltas = T[]
        for (bc, vec_start, r2_start) in zip(bond_constraints, vecs_start, r2s_start)
            if constraining_velocities
                dr = coords_or_vels[bc.i] - coords_or_vels[bc.j]
                rrpr = dot(dr, vec_start)
                delta = -2 * bc.reduced_mass * rrpr / r2_start
                push!(deltas, delta)
                if abs(delta) <= (tolerance)u"u * ps^-1"
                    n_converged += 1
                end
            else
                dr = vector(coords_or_vels[bc.j], coords_or_vels[bc.i], box_size)
                rrpr = dot(dr, vec_start)
                r2 = sum(abs2, dr)
                diff = bc.distance ^ 2 - r2
                push!(deltas, bc.reduced_mass * diff / rrpr)
                if r2 >= (lower_tol * bc.distance ^ 2) && r2 <= (upper_tol * bc.distance ^ 2)
                    n_converged += 1
                end
            end
        end

        if n_converged == n_constraints
            break
        end

        deltas = inv_K * deltas

        for (bc, vec_start, delta) in zip(bond_constraints, vecs_start, deltas)
            dr = vec_start * delta
            coords_or_vels[bc.i] += dr / mass(atoms[bc.i])
            coords_or_vels[bc.j] -= dr / mass(atoms[bc.j])
        end
    end

    return coords_or_vels
end
