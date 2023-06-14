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

        # case for SHAKE w/o RATTLE (does it need velocity changes still?)
        # case for RATTLE which will call SHAKE
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
    tolerance::E``
end

function SHAKE(dists, is, js, tolerance=1e-10u"nm")
    @assert (length(is) == length(js)) && (length(dists) == length(js)) "Constraint lengths do not match"
    return SHAKE{typeof(dists), typeof(tolerance)}(dists, is, js, tolerance)
end

"""
    apply_constraints!(sys, constraint, old_coords, dt)

Updates the coordinates and/or velocities of a [`System`](@ref) based on the constraints.
"""
function apply_position_constraints!(sys, constraint::SHAKE, old_coords, dt)
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

                # g needs to be divided by dt^2???

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
struct RATTLE{D,B,T,C,E}
    A::D
    b::B
    dt::T
    is::C
    js::C
    tolerance::E
end

function RATTLE(is, js, dt, tolerance=1e-10)
    @assert length(is) == length(js) "Constraint lengths do not match"
    #Allocate storage for linear system
    A = zeros(length(is), lengths(is))
    b = zeros(length(is))
    return RATTLE{typeof(dists), typeof(is), typeof(tolerance)}(A, b, dt, is, js, tolerance = tolerance)
end

# Find isolated 2, 3 & 4 atom clusters
function RATTLE_setup(constraint::RATTLE)

    rattle2_pairs = []
    # rattle3_pairs = []
    # rattle4_pairs = []

    #Exhasutive search through constraints to find conflicts -- will not scale to 3,4 bond clusters
    for r1 in eachindex(constraint.is)
        # Atoms that are part of the bond
        i_r1 = constraint.is[r1]
        j_r1 = constraint.js[r1]

        is_isolated = i_r1 ∉ constraint.is[constraint.is .!= r1] &&
                      i_r1 ∉ constraint.js &&
                      j_r1 ∉ constraint.js[constraint.js .!= r1] &&
                      j_r1 ∉ constraints.is

        if is_isolated
            push!(rattle2_pairs, r1)
        end

    end


end

function apply_velocity_constraints!(sys, constraint::RATTLE)


end

# IS THIS JUST FOR A HALF STEP?

# Rattle for a single distance constraint between atoms i and j
# Atoms i and j do NOT participate in any other constraints
function RATTLE2(sys, k, constraint::RATTLE)

    # Index of atoms in bond k
    k1 = constraint.is[k]
    k2 = constraint.js[k]

    # Inverse of masses of atoms in bond k
    inv_m1 = 1/mass(sys.atoms[k1])
    inv_m2 = 1/mass(sys.atoms[k2])

    # Distance vector between the atoms after SHAKE constraint
    r_k1k2 = vector(sys.coords[k2], sys.coords[k1], sys.boundary)

    # Difference between unconstrainted velocities
    v_k1k2 = sys.velocities[k2] .- sys.velocities[k1]

    # Re-arrange constraint equation to solve for Lagrange multiplier
    # Technically this has a factor of dt which cancels out in the velocity update
    λₖ = -dot(r_k1k2,v_k1k2)/(dot(r_k1k2,r_k1k2)*(inv_m1 + inv_m2))

    # Correct velocities
    sys.velocities[k1] .-= (inv_m1 .* λₖ .* r_k1k2)
    sys.coords[k2] .+= (inv_m1 .* λₖ .* r_k1k2)

end