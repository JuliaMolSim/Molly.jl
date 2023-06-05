export
    apply_constraints!,
    SHAKE

"""
    apply_constraints!(system, old_coords, dt)

Applies all the bond and angle constraints associated with the [`System`](@ref).
"""
function apply_constraints!(sys, old_values, dt)
    for constraint in sys.constraints
        apply_constraints!(sys, constraint, old_values, dt)
    end
end

"""
    SHAKE(dists, is, js)

Constrains a set of bonds to defined distances.
"""
struct SHAKE{D, B, E}
    dists::D
    is::B
    js::B
    tolerance::E
end

SHAKE(dists, is, js, tolerance=1e-10u"nm") = SHAKE{typeof(dists), typeof(tolerance)}(dists, is, js, tolerance)

"""
    apply_constraints!(sys, constraint, old_coords, dt)

Updates the coordinates and/or velocities of a [`System`](@ref) based on the constraints.
"""
function apply_constraints!(sys, constraint::SHAKE, old_coords, dt)
    converged = false

    while !converged
        for r in eachindex(constraint.is)
            # Atoms that are part of the bond
            i0 = constraint.is[r]
            i1 = constraint.js[r]

            # Distance vector between the atoms before unconstrained update
            r01 = vector(old_coords[i1], old_coords[i0], sys.boundary)

            # Distance vector after unconstrained update
            s01 = vector(sys.coords[i1], sys.coords[i0], sys.boundary)

            if abs(norm(s01) - constraint.dists[r]) > constraint.tolerance
                m0 = mass(sys.atoms[i0])
                m1 = mass(sys.atoms[i1])
                a = (1/m0 + 1/m1)^2 * norm(r01)^2
                b = 2 * (1/m0 + 1/m1) * dot(r01, s01)
                c = norm(s01)^2 - ((constraint.dists[r])^2)
                D = (b^2 - 4*a*c)
                
                if ustrip(D) < 0.0
                    @warn "SHAKE determinant negative, setting to 0.0"
                    D = zero(D)
                end

                # Quadratic solution for g
                α1 = (-b + sqrt(D)) / (2*a)
                α2 = (-b - sqrt(D)) / (2*a)

                g = abs(α1) <= abs(α2) ? α1 : α2

                # Update positions
                δri0 = r01 .* ( g/m0)
                δri1 = r01 .* (-g/m1)

                sys.coords[i0] += δri0
                sys.coords[i1] += δri1
            end
    
        end

        lengths = [abs(norm(vector(sys.coords[constraint.is[r]], sys.coords[constraint.js[r]], sys.boundary)) - constraint.dists[r]) for r in eachindex(constraint.is)]

        if maximum(lengths) < constraint.tolerance
            converged = true
        end
    end
end


"""
    RATTLE(dists, is, js)

Constrains a set of bonds to defined distances in a way that the velocities also satisfy the constraints.
"""
struct RATTLE{D,B,E}
    dists::D
    is::B
    js::B
    tolerance::E
end

#This doesn't really fit in with the current interface in simulators.jl....I think that will need to change.
#Do not see a general way to apply constraint algos
function apply_constraints!(sys, constraint::RATTLE)

    converged = zeros(length(constraint.dists))

    while !all(converged)
        for r in eachindex(constraint.is)
            d_ij_sq = (constraint.dists[r])^2
            m0 = mass(sys.atoms[i0])
            m1 = mass(sys.atoms[i1])

            # Atoms that are part of the bond
            i0 = constraint.is[r]
            i1 = constraint.js[r]

            # Distance vector between the atoms after SHAKE constraint
            r01 = vector(sys.coords[i1], sys.coords[i0], sys.boundary)

            # Current velocity vector between atoms i,j
            v01 = sys.velocities[i1] .- sys.velocities[i0]

            #should there be abs() around dot???
            if dot(r01, v01) > constraint.tolerance
                k = r01 * (sys.velocities[i1] .- sys.velocities[i0])/(d_ij_sq*(1/m0 + 1/m1))
                sys.velocities[i0] += k*r01/m0
                sys.velocities[i1] -= k*r01/m1
            else
                converged[r] = true
            end

        end

    end

end