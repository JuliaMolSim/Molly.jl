

function apply_position_constraints!(sys, ca::SHAKE_RATTLE, coord_storage;
                                     n_threads::Integer=Threads.nthreads())
    # SHAKE updates
    converged = false

    while !converged
        Threads.@threads for cluster in ca.clusters
            for constraint in cluster.constraints
                k1, k2 = constraint.i, constraint.j

                # Vector between the atoms after unconstrained update (s)
                s12 = vector(sys.coords[k1], sys.coords[k2], sys.boundary)

                # Vector between the atoms before unconstrained update (r)
                r12 = vector(coord_storage[k1], coord_storage[k2], sys.boundary)

                if abs(norm(s12) - constraint.dist) > ca.dist_tolerance
                    m1_inv = inv(mass(sys.atoms[k1]))
                    m2_inv = inv(mass(sys.atoms[k2]))
                    a = (m1_inv + m2_inv)^2 * sum(abs2, r12)
                    b = -2 * (m1_inv + m2_inv) * dot(r12, s12)
                    c = sum(abs2, s12) - (constraint.dist)^2
                    D = b^2 - 4*a*c

                    if ustrip(D) < 0.0
                        @warn "SHAKE determinant negative, setting to 0.0"
                        D = zero(D)
                    end

                    # Quadratic solution for g
                    α1 = (-b + sqrt(D)) / (2*a)
                    α2 = (-b - sqrt(D)) / (2*a)

                    g = abs(α1) <= abs(α2) ? α1 : α2

                    # Update positions
                    δri1 = r12 .* (g*m1_inv)
                    δri2 = r12 .* (-g*m2_inv)

                    sys.coords[k1] += δri1
                    sys.coords[k2] += δri2
                end
            end
        end

        converged = check_position_constraints(sys, ca)
    end
    return sys
end