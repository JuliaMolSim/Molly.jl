export CMAPTorsion

"""
    CMAPTorsion(index, size)

Torsional correction map (CMAP) for sets of five atoms, for example protein ϕ and ψ
backbone torsion angles.

The CMAP data is stored in the `data` field of the associated [`InteractionList5Atoms`](@ref).

Only compatible with 3D systems.
"""
struct CMAPTorsion
    index::Int
    size::Int
end

Base.zero(::CMAPTorsion) = CMAPTorsion(0, 0)

Base.:+(c1::CMAPTorsion, c2::CMAPTorsion) = c1

function cmap_coefficients(n, mp::Vector{E}) where E
    c = cmap_map_derivatives(n, mp)
    coeff_matrix = Matrix{E}(undef, n*n*4, 4)
    for j in 1:(n*n)
        coeff_matrix[(j-1)*4+1, :] .= c[j, 1:4]
        coeff_matrix[(j-1)*4+2, :] .= c[j, 5:8]
        coeff_matrix[(j-1)*4+3, :] .= c[j, 9:12]
        coeff_matrix[(j-1)*4+4, :] .= c[j, 13:16]
    end
    return coeff_matrix
end

function cmap_map_derivatives(n, energy::Vector{E}) where E
    T = typeof(ustrip(one(E)))
    x = [(i * 2 * T(π) / n) for i in 0:n]
    y     = zeros(E, n+1)
    deriv = zeros(E, n+1)
    d1    = zeros(E, n*n)
    d2    = zeros(E, n*n)
    d12   = zeros(E, n*n)

    for i in 1:n
        for j in 1:n
            y[j] = energy[j+n*(i-1)] 
        end
        y[n+1] = energy[n*(i-1)+1]  
        deriv = create_periodic_spline(x,y,deriv)
        for j in 1:n
            d1[j+n*(i-1)] = evaluate_spline_derivative(x, y, deriv, x[j])
        end
    end

    for i in 1:n
        for j in 1:n
            y[j] = energy[i+n*(j-1)]  
        end
        y[n+1] = energy[i]  

        deriv = create_periodic_spline(x,y,deriv)
        for j in 1:n
            d2[i+n*(j - 1)] = evaluate_spline_derivative(x, y, deriv, x[j])
        end
    end

    for i in 1:n
        for j in 1:n
            y[j] = d2[j+n*(i-1)]
        end
        y[n+1] = d2[n*(i-1)+1]
        deriv = create_periodic_spline(x,y,deriv)
        for j in 1:n
            d12[j+n*(i-1)] = evaluate_spline_derivative(x, y, deriv, x[j])
        end
    end

    wt = [
            1, 0, -3, 2, 0, 0, 0, 0, -3, 0, 9, -6, 2, 0, -6, 4,
            0, 0, 0, 0, 0, 0, 0, 0, 3, 0, -9, 6, -2, 0, 6, -4,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -6, 0, 0, -6, 4,
            0, 0, 3, -2, 0, 0, 0, 0, 0, 0, -9, 6, 0, 0, 6, -4,
            0, 0, 0, 0, 1, 0, -3, 2, -2, 0, 6, -4, 1, 0, -3, 2,
            0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 3, -2, 1, 0, -3, 2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 2, 0, 0, 3, -2,
            0, 0, 0, 0, 0, 0, 3, -2, 0, 0, -6, 4, 0, 0, 3, -2,
            0, 1, -2, 1, 0, 0, 0, 0, 0, -3, 6, -3, 0, 2, -4, 2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -6, 3, 0, -2, 4, -2,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, 2, -2,
            0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 3, -3, 0, 0, -2, 2,
            0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0, 1, -2, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, -1, 1,
            0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 2, -2, 0, 0, -1, 1
        ]

    rhs = zeros(E, 16)
    delta = 2 * T(π) / n
    c = zeros(E, n * n, 16)

    for i in 1:n
        for j in 1:n
            nexti = (i % n) + 1
            nextj = (j % n) + 1
            e = [energy[i+n*(j-1)],energy[nexti+n*(j-1)], energy[nexti+n*(nextj-1)], energy[i+n*(nextj-1)]]
            e1 = [d1[i+n*(j-1)], d1[nexti+n*(j-1)], d1[nexti+n*(nextj-1)], d1[i+n*(nextj-1)]]
            e2 = [d2[i+n*(j-1)], d2[nexti+n*(j-1)], d2[nexti+n*(nextj-1)], d2[i+n*(nextj-1)]]
            e12 = [d12[i+n*(j-1)], d12[nexti+n*(j-1)], d12[nexti+n*(nextj-1)], d12[i+n*(nextj-1)]]

            for k in 1:4
                rhs[k] = e[k]
                rhs[k+4] = e1[k]*delta
                rhs[k+8] = e2[k]*delta
                rhs[k+12] = e12[k]*delta*delta
            end
            for k in 1:16
                s = zero(E)
                for m in 1:16
                    s += wt[k+16*(m-1)]*rhs[m]
                end
                c[i+n*(j-1),k] = s
            end
        end
    end
    return c
end

function create_periodic_spline(x::AbstractVector{X}, y, deriv::AbstractVector{D}) where {X, D}
    n = length(x)
    if length(y) != n
        throw(ArgumentError("x and y must have equal length for create_periodic_spline"))
    end
    if n < 3
        throw(ArgumentError("x must have a length of at least 3 for create_periodic_spline"))
    end
    if y[1] != y[end]
        throw(ArgumentError("y[1] must equal y[end] for create_periodic_spline"))
    end
    # Create the system of equations to solve
    a = zeros(X, n-1)
    b = zeros(X, n-1)
    c = zeros(X, n-1)
    rhs = zeros(D, n-1)
    a[1] = x[n]-x[n-1]
    b[1] = 2*(x[2]-x[1]+x[n]-x[n-1])
    c0 = x[2]-x[1]
    rhs[1] =  6*((y[2]-y[1])/(x[2]-x[1]) - (y[n]-y[n-1])/(x[n]-x[n-1]))
    for i in 2:n-1
        a[i] = x[i]-x[i-1]
        b[i] = 2*(x[i+1]-x[i-1])
        c[i] = x[i+1]-x[i]
        rhs[i] = 6*((y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]))
    end
    beta = a[1]
    alpha = c[n-1]
    gamma = -b[1]

    # This is a cyclic tridiagonal matrix. We solve it using the Sherman-Morrison method,
    # which involves solving two tridiagonal systems

    n -= 1
    b[1] -= gamma
    b[n] -= alpha*beta/gamma
    deriv = solve_tridiagonal_matrix(a, b, c, rhs, deriv)  
    u = zeros(X, n)
    z = zeros(X, n)
    u[1] = gamma
    u[n] = alpha
    z = solve_tridiagonal_matrix(a, b, c, u, z)
    scale = (deriv[1]+beta*deriv[n]/gamma)/(1+z[1]+beta*z[n]/gamma)
    for i in 1:n
        deriv[i] -= scale*z[i]
    end
    deriv[n+1] = deriv[1]
    return deriv
end

function solve_tridiagonal_matrix(a, b, c, rhs, deriv)
    n = length(a)
    gamma = zero(a)

    # Decompose the matrix
    deriv[1] = rhs[1] / b[1]
    beta = b[1]
    for i in 2:n
        gamma[i] = c[i-1]/beta
        beta = b[i]-a[i]*gamma[i]
        deriv[i] = (rhs[i]-a[i]*deriv[i-1])/beta
    end

    # Perform backsubstitution
    for i in n-1:-1:1
        deriv[i] -= gamma[i+1]*deriv[i+1]
    end
    return deriv
end

function evaluate_spline_derivative(x, y, deriv, t)
    n = length(x)
    if t < x[1] || t > x[n]
        error()
    end

    lower = 1
    upper = n
    while (upper-lower) > 1
        middle = round(Int,(upper+lower)/2)
        if (x[middle] > t)
            upper = Int(middle)
        else
            lower = Int(middle)
        end
    end

    dx = x[upper] - x[lower]
    a = (x[upper]-t)/dx
    b = one(a)-a
    dadx = -one(dx)/dx
    return dadx*y[lower]-dadx*y[upper] + ((1-3*a*a)*deriv[lower] + ((3*b*b)-1)*deriv[upper])*dx/6
end

function cmap_angles(inter, coords_i, coords_j, coords_k, coords_l, coords_m, boundary)
    # First angle
    v0a = vector(coords_j, coords_i, boundary)
    v1a = vector(coords_j, coords_k, boundary)
    v2a = vector(coords_l, coords_k, boundary)
    cp0a = cross(v0a, v1a)
    cp1a = cross(v1a, v2a)
    cosangle = dot(cp0a/norm(cp0a), cp1a/norm(cp1a))
    T = typeof(cosangle)
    if cosangle > T(0.99) || cosangle < T(-0.99)
        # Close to the singularity in acos, so take the cross product and use asin instead
        cross_prod = cross(cp0a, cp1a)
        scale = dot(cp0a,cp0a) * dot(cp1a,cp1a)
        angleA = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < zero(T)
            angleA = T(π) - angleA
        end
    else
        angleA = acos(cosangle)
    end
    dot_v0a_cp1a = dot(v0a, cp1a)
    angleA = (dot_v0a_cp1a >= zero(dot_v0a_cp1a) ? angleA : -angleA)
    angleA = mod(angleA + 2*T(π), 2*T(π))

    # Second angle
    v0b = vector(coords_k, coords_j, boundary)
    v1b = vector(coords_k, coords_l, boundary)
    v2b = vector(coords_m, coords_l, boundary)
    cp0b = cross(v0b, v1b)
    cp1b = cross(v1b, v2b)
    cosangle = dot(cp0b/norm(cp0b), cp1b/norm(cp1b))
    if cosangle > T(0.99) || cosangle < T(-0.99)
        cross_prod = cross(cp0b, cp1b)
        scale = dot(cp0b,cp0b) * dot(cp1b,cp1b)
        angleB = asin(sqrt(dot(cross_prod,cross_prod)/scale))
        if cosangle < zero(T)
            angleB = T(π) - angleB
        end
    else
        angleB = acos(cosangle)
    end
    dot_v0b_cp1b = dot(v0b, cp1b)
    angleB = (dot_v0b_cp1b >= zero(dot_v0b_cp1b) ? angleB : -angleB)
    angleB = mod(angleB + 2*T(π), 2*T(π))

    # Identify patch
    delta = 2*T(π) / inter.size
    s = Int(trunc(min(angleA/delta, inter.size-1)))
    t = Int(trunc(min(angleB/delta, inter.size-1)))
    idx = inter.index+(4*(s+inter.size*t))+1
    da = angleA/delta - s
    db = angleB/delta - t

    return v0a, v1a, v2a, cp0a, cp1a, v0b, v1b, v2b, cp0b, cp1b, delta, idx, da, db
end

@inline function force(inter::CMAPTorsion, coords_i, coords_j, coords_k, coords_l, 
                       coords_m, boundary, atoms_i, atoms_j, atoms_k, atoms_l, 
                       atoms_m, force_units, velocities_i, velocities_j,
                       velocities_k, velocities_l, velocities_m, step_n, data)
    v0a, v1a, v2a, cp0a, cp1a, v0b, v1b, v2b, cp0b, cp1b, delta, idx, da, db = cmap_angles(
                        inter, coords_i, coords_j, coords_k, coords_l, coords_m, boundary)
    
    # Evaluate the spline to determine the energy and gradients
    dEdA = (3*data[idx+3,4]*da + 2*data[idx+2,4])*da + data[idx+1,4]
    dEdB = (3*data[idx+3,4]*db + 2*data[idx+3,3])*db + data[idx+3,2]
    dEdA = db*dEdA + (3*data[idx+3,3]*da + 2*data[idx+2,3])*da + data[idx+1,3]
    dEdB = da*dEdB + (3*data[idx+2,4]*db + 2*data[idx+2,3])*db + data[idx+2,2]
    dEdA = db*dEdA + (3*data[idx+3,2]*da + 2*data[idx+2,2])*da + data[idx+1,2]
    dEdB = da*dEdB + (3*data[idx+1,4]*db + 2*data[idx+1,3])*db + data[idx+1,2]
    dEdA = db*dEdA + (3*data[idx+3,1]*da + 2*data[idx+2,1])*da + data[idx+1,1]
    dEdB = da*dEdB + (3*data[idx,4]*db + 2*data[idx,3])*db + data[idx,2]
    dEdA /= delta
    dEdB /= delta

    # Calculate the force to the first torsion
    normCross1 = dot(cp0a, cp0a)
    normSqrBC = dot(v1a, v1a)
    normBC = sqrt(normSqrBC)
    normCross2 = dot(cp1a, cp1a)
    dp = inv(normSqrBC)
    ff = ((-dEdA*normBC)/normCross1, dot(v0a, v1a)*dp, dot(v2a, v1a)*dp, (dEdA*normBC)/normCross2)
    force1 = ff[1]*cp0a
    force4 = ff[4]*cp1a
    d = ff[2]*force1 - ff[3]*force4
    force2 =  d-force1
    force3 = -d-force4

    # Calculate the force to the second torsion
    normCross1 = dot(cp0b, cp0b)
    normSqrBC = dot(v1b, v1b)
    normBC = sqrt(normSqrBC)
    normCross2 = dot(cp1b, cp1b)
    dp = inv(normSqrBC)
    ff = ((-dEdB*normBC)/normCross1, dot(v0b, v1b)*dp, dot(v2b, v1b)*dp, (dEdB*normBC)/normCross2)
    force5 = ff[1]*cp0b
    force8 = ff[4]*cp1b
    d = ff[2]*force5 - ff[3]*force8
    force6 =  d-force5
    force7 = -d-force8

    # Apply the forces to the atoms
    fi = force1
    fj = force2 + force5
    fk = force3 + force6
    fl = force4 + force7
    fm =          force8
    return SpecificForce5Atoms(fi, fj, fk, fl, fm)
end

@inline function potential_energy(inter::CMAPTorsion, coords_i, coords_j, coords_k, coords_l, 
                    coords_m, boundary, atoms_i, atoms_j, atoms_k, atoms_l, atoms_m, energy_units,
                    velocities_i, velocities_j, velocities_k, velocities_l, velocities_m,
                    step_n, data)
    v0a, v1a, v2a, cp0a, cp1a, v0b, v1b, v2b, cp0b, cp1b, delta, idx, da, db = cmap_angles(
                        inter, coords_i, coords_j, coords_k, coords_l, coords_m, boundary)

    # Spline with coefficients
    energy = ((data[idx+3,4]*db + data[idx+3,3])*db + data[idx+3,2])*db + data[idx+3,1]
    energy = da*energy + ((data[idx+2,4]*db + data[idx+2,3])*db + data[idx+2,2])*db + data[idx+2,1]
    energy = da*energy + ((data[idx+1,4]*db + data[idx+1,3])*db + data[idx+1,2])*db + data[idx+1,1]
    energy = da*energy + ((data[idx,4]*db + data[idx,3])*db + data[idx,2])*db + data[idx,1]
    return energy
end
