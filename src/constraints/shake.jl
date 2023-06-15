export SHAKE

"""
    SHAKE(dists, is, js)

Constrains a set of bonds to defined distances.
"""
struct SHAKE{D, B, E} <: PositionConstraintAlgorithm
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
    apply_constraint!(sys, constraint, unconstrained_coords)

Updates the coordinates and/or velocities of a [`System`](@ref) based on the constraints.
"""
function apply_constraints!(sys, constraint::SHAKE, constraint_cluster::SmallConstraintCluster, unconstrained_coords)

    #IDENTIFY TYPE OF CLUSTER AND APPLY CORRECT VERSION OF SHAKE

    converged = false

    while !converged
        for r in eachindex(constraint.is)
            # Atoms that are part of the bond
            i0 = constraint.is[r]
            i1 = constraint.js[r]

            # Distance vector between the atoms before unconstrained update
            r01 = vector(unconstrained_coords[i1], unconstrained_coords[i0], sys.boundary)

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


function apply_constraints!(sys, constraint::SHAKE, constraint_cluster::LargeConstraintCluster, unconstrained_coords)

    #implement M Shake??
end