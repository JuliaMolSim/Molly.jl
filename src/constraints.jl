export
    run_constraints!,
    SHAKE,
    apply_constraint!


"""
    run_constraints!(system, old_coords, dt)

Applies all the bond and angle constraints associated with the system.
"""

function run_constraints!(sys, old_values, dt)
    for constraint in sys.constraints
        apply_constraint!(sys, old_values, dt, constraint)
    end
end

"""
    SHAKE(d, bond_list)

Constrains a set of bonds to a particular distance d.
"""
struct SHAKE{D, B}
    d::D
    bond_list::B
end

"""
    apply_constraint!(sys, old_values, dt, constraint)

Updates the system coordinates and/or velocities based on the constraint.
"""
function apply_constraint!(sys, old_coords, dt, constraint::SHAKE)

    for r in 1:length(constraint.bond_list.is)
        # Atoms that are part of the bond
        i0 = constraint.bond_list.is[r]
        i1 = constraint.bond_list.js[r]

        # Distance vector between the atoms before unconstraied update
        r01 = vector(old_coords[i1], old_coords[i0], sys.boundary)

        # Distance vector after unconstrained update
        s01 = vector(sys.coords[i1], sys.coords[i0], sys.boundary)

        m0 = masses(sys)[i0]
        m1 = masses(sys)[i1]
        a = (1/m0 + 1/m1)^2 * norm(r01)^2
        b = 2.0 * (1/m0 + 1/m1) * dot(r01, s01)
        c = norm(s01)^2 - constr.d^2
        D = (b^2 - 4*a*c)
        

        if (ustrip(D) < 0.0)
            @warn "SHAKE determinant negative. Setting to 0.0"
            D = 0.0u"nm^4/u^2"
        end

        # Quadratic solution for g
        α1 = (-b + sqrt(D))/(2*a)
        α2 = (-b - sqrt(D))/(2*a)

        g = ((abs(α1) <= abs(α2)) ? α1 : α2)

        # update positions

        δri0 = r01.*(g/m0)
        δri1 = r01.*(-g/m1)

        sys.coords[i0] += δri0
        sys.coords[i1] += δri1
    end
end
