# Bond and angle constraints

export DistanceConstraint,
    AngleConstraint,
    CCMAConstraints,
    constraincoordinates!,
    constrainvelocities!

"""
    DistanceConstraint(i, j, distance, reduced_mass)

A constraint that requires two atoms to be a certain distance apart.
"""
struct DistanceConstraint{D, M}
    i::Int
    j::Int
    distance::D
    reduced_mass::M
end

"""
    AngleConstraint(i, j, k, angle)

A constraint that requires three atoms to have a certain bond angle.
"""
struct AngleConstraint{T}
    i::Int
    j::Int
    k::Int
    angle::T
end

"""
    CCMAConstraints(atoms, atoms_data, bonds, angles, coords, box_size)
    CCMAConstraints(bond_constraints, angle_constraints,
                    combined_constraints, inv_K)

A set of bond and angle constraints to be constrained using the Constraint
Matrix Approximation method.
See Eastman and Pande 2010.
"""
struct CCMAConstraints{B, A, M}
    bond_constraints::Vector{B}
    angle_constraints::Vector{A}
    combined_constraints::Vector{B}
    inv_K::M
end

function CCMAConstraints(atoms, atoms_data, bonds, angles, coords, box_size; cutoff=0.01)
    T = Float64
    elements = [at.element for at in atoms_data]
    masses = mass.(atoms)

    bond_constraints = DistanceConstraint[]
    angle_constraints = AngleConstraint[]

    for bond in bonds
        if elements[bond.i] == "H" || elements[bond.j] == "H"
            i, j = sort([bond.i, bond.j])
            dist = norm(vector(coords[i], coords[j], box_size))
            reduced_mass = inv(2 * (inv(masses[i]) + inv(masses[j])))
            push!(bond_constraints, DistanceConstraint(i, j, dist, reduced_mass))
        end
    end

    combined_constraints = [bond_constraints...]

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
            d1 = norm(vector(coords[angle.j], coords[i], box_size))
            d2 = norm(vector(coords[angle.j], coords[k], box_size))
            dist = norm(vector(coords[i], coords[k], box_size))
            θ = acos((d1 * d1 + d2 * d2 - dist * dist) / (2 * d1 * d2))
            push!(angle_constraints, AngleConstraint(i, angle.j, k, θ))
            reduced_mass = inv(2 * (inv(masses[i]) + inv(masses[k])))
            push!(combined_constraints, DistanceConstraint(i, k, dist, reduced_mass))
        end
    end

    # Remove a constraint from atoms bonded to 3 or more others
    # This is a quick way to avoid a non-singular inverse matrix due to an over-constrained system
    constraints_per_atom = zeros(Int, length(atoms))
    for (ai, ad) in enumerate(atoms_data)
        # Doesn't converge for larger angles so exclude these for now
        if (ad.atom_name in ("NH1", "NH2") && ad.res_name == "ARG") || (ad.atom_name == "ND2" && ad.res_name == "ASN")
            constraints_per_atom[ai] += 1
        end
    end
    for c in bond_constraints
        constraints_per_atom[c.i] += 1
        constraints_per_atom[c.j] += 1
    end
    constraints_to_remove = falses(length(combined_constraints))
    for (ci, c) in enumerate(combined_constraints)
        if constraints_per_atom[c.i] >= 3
            constraints_to_remove[ci] = true
            constraints_per_atom[c.i] = 0
        end
        if constraints_per_atom[c.j] >= 3
            constraints_to_remove[ci] = true
            constraints_per_atom[c.j] = 0
        end
    end
    combined_constraints = [combined_constraints[ci] for ci in 1:length(combined_constraints) if !constraints_to_remove[ci]]

    n_constraints = length(combined_constraints)
    K = spzeros(T, n_constraints, n_constraints)
    for ci in 1:n_constraints
        for cj in 1:n_constraints
            cj_inds = (combined_constraints[cj].i, combined_constraints[cj].j)
            if ci == cj
                K[ci, ci] = one(T)
            elseif combined_constraints[ci].i in cj_inds || combined_constraints[ci].j in cj_inds
                if combined_constraints[ci].i in cj_inds
                    shared_ind = combined_constraints[ci].i
                    other_ind_i = combined_constraints[ci].j
                else
                    shared_ind = combined_constraints[ci].j
                    other_ind_i = combined_constraints[ci].i
                end
                if shared_ind == combined_constraints[cj].i
                    other_ind_j = combined_constraints[cj].j
                else
                    other_ind_j = combined_constraints[cj].i
                end
                sort_other_ind_1, sort_other_ind_2 = sort([other_ind_i, other_ind_j])
                triangle_ind = findfirst(combined_constraints) do c
                    c.i == sort_other_ind_1 && c.j == sort_other_ind_2
                end                                            
                angle_force_ind = findfirst(angles) do a
                    (sort_other_ind_1, shared_ind, sort_other_ind_2) in ((a.i, a.j, a.k), (a.k, a.j, a.i))
                end
                if isnothing(triangle_ind)
                    if !isnothing(angle_force_ind)
                        cos_θ = cos(angles[angle_force_ind].th0)
                    else
                        d1 = combined_constraints[ci].distance
                        d2 = combined_constraints[cj].distance
                        d3 = norm(vector(coords[sort_other_ind_1], coords[sort_other_ind_2], box_size))
                        cos_θ = (d1 * d1 + d2 * d2 - d3 * d3) / (2 * d1 * d2)
                    end
                else
                    d1 = combined_constraints[ci].distance
                    d2 = combined_constraints[cj].distance
                    d3 = combined_constraints[triangle_ind].distance
                    cos_θ = (d1 * d1 + d2 * d2 - d3 * d3) / (2 * d1 * d2)
                end
                mass_term = inv(masses[shared_ind]) / (inv(masses[shared_ind]) + inv(masses[other_ind_i]))
                K[ci, cj] = mass_term * cos_θ
            end
        end
    end

    inv_K = inv(Array(K))
    inv_K = sparse((abs.(inv_K) .> cutoff) .* inv_K)
    return CCMAConstraints([bond_constraints...], [angle_constraints...], combined_constraints, inv_K)
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
                            vel_tolerance=1e-5u"u * ps^-1",
                            max_n_iters=150)
    dist_constraints = constraints.combined_constraints
    inv_K = constraints.inv_K
    n_constraints = length(dist_constraints)
    vecs_start = [vector(coords_prev[dc.j], coords_prev[dc.i], box_size) for dc in dist_constraints]
    r2s_start = sum.(abs2, vecs_start)

    lower_tol = 1 - 2 * tolerance + tolerance ^ 2
    upper_tol = 1 + 2 * tolerance + tolerance ^ 2
    T = typeof(mass(first(atoms)) * first(first(coords_or_vels)) / first(first(coords_prev)))

    for iter_i in 1:max_n_iters
        n_converged = 0
        deltas = T[]
        for (dc, vec_start, r2_start) in zip(dist_constraints, vecs_start, r2s_start)
            if constraining_velocities
                dr = coords_or_vels[dc.i] - coords_or_vels[dc.j]
                rrpr = dot(dr, vec_start)
                delta = -2 * dc.reduced_mass * rrpr / r2_start
                push!(deltas, delta)
                if abs(delta) <= vel_tolerance
                    n_converged += 1
                end
            else
                dr = vector(coords_or_vels[dc.j], coords_or_vels[dc.i], box_size)
                rrpr = dot(dr, vec_start)
                r2 = sum(abs2, dr)
                diff = dc.distance ^ 2 - r2
                push!(deltas, dc.reduced_mass * diff / rrpr)
                if r2 >= (lower_tol * dc.distance ^ 2) && r2 <= (upper_tol * dc.distance ^ 2)
                    n_converged += 1
                end
            end
        end

        if n_converged == n_constraints
            break
        end

        deltas = inv_K * deltas

        for (dc, vec_start, delta) in zip(dist_constraints, vecs_start, deltas)
            dr = vec_start * delta
            coords_or_vels[dc.i] += dr / mass(atoms[dc.i])
            coords_or_vels[dc.j] -= dr / mass(atoms[dc.j])
        end
    end

    return coords_or_vels
end
