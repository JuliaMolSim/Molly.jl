
# Long range electrostatic summation methods
# Based on the OpenMM source code

export
    Ewald,
    PME

abstract type AbstractEwald end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                            inter::AbstractEwald; kwargs...)
    return ewald_pe_forces(sys, inter)[1]
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(sys,
                                                            inter::AbstractEwald; kwargs...)
    return ewald_pe_forces(sys, inter)[2]
end

function AtomsCalculators.energy_forces(sys, inter::AbstractEwald; kwargs...)
    E, Fs = ewald_pe_forces(sys, inter)
    return (energy=E, forces=Fs)
end

function find_excluded_pairs(eligible)
    excluded_pairs = Tuple{Int32, Int32}[]
    if !isnothing(eligible)
        n_atoms = size(eligible, 1)
        eligible_cpu = Array(eligible)
        for i in 1:n_atoms
            for j in (i+1):n_atoms
                if !eligible_cpu[i, j]
                    push!(excluded_pairs, (Int32(i), Int32(j)))
                end
            end
        end
    end
    return excluded_pairs
end

"""
    Ewald(; dist_cutoff, error_tol=0.0005, eligible=nothing)

Ewald summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator.

Should be used alongside the [`CoulombEwald`](@ref) pairwise interaction,
which provide the short range term.
The `dist_cutoff` and `error_tol` should match.

`dist_cutoff` is the cutoff distance for short range interactions.
`eligible` indicates pairs eligible for short range interaction, and can
be a matrix like the neighbor list or `nothing` to indicate that all pairs
are eligible.

This algorithm is O(N^2) and in general [`PME`](@ref) should be used instead.
Only compatible with 3D systems and [`CubicBoundary`](@ref).
Runs on the CPU, even for GPU systems.
"""
struct Ewald{T, D} <: AbstractEwald
    dist_cutoff::D
    error_tol::T
    excluded_pairs::Vector{Tuple{Int32, Int32}}
end

function Ewald(; dist_cutoff, error_tol=0.0005, eligible=nothing, special=nothing)
    return Ewald(dist_cutoff, error_tol, find_excluded_pairs(eligible))
end

function ewald_error(αr::T, target, guess) where T
    t = guess * T(π) / αr
    return target - T(0.05) * sqrt(αr) * guess * exp(-t^2)
end

function find_zero(αr::T, target, initial_guess) where T
    guess = initial_guess
    x = ewald_error(αr, target, guess)
    if x > zero(T)
        while x > zero(T) && guess > 0
            guess -= 1
            x = ewald_error(αr, target, guess)
        end
        return guess + 1
    else
        while x < zero(T)
            guess += 1
            x = ewald_error(αr, target, guess)
        end
        return guess
    end
end

function ewald_params(side_length, α, error_tol)
    k = find_zero(α * side_length, error_tol, 10)
    if iszero(k % 2)
        k += 1
    end
    return k
end

function excluded_interactions!(Fs, excluded_pairs, partial_charges::Vector{T}, coords,
                                boundary, α, f, energy_units) where T
    sqrt_π = sqrt(T(π))
    exclusion_E = zero(T) * energy_units
    for (i, j) in excluded_pairs
        charge_ij = partial_charges[i] * partial_charges[j]
        vec_ij = vector(coords[i], coords[j], boundary)
        r = norm(vec_ij)
        αr = α * r
        erf_αr = erf(αr)
        if erf_αr > T(1e-6)
            inv_r = inv(r)
            exclusion_E -= f * charge_ij * inv_r * erf_αr
            dE_dr = f * charge_ij * inv_r^3 * (erf_αr - 2 * αr * exp(-αr^2) / sqrt_π)
            F = dE_dr * vec_ij
            Fs[i] += F
            Fs[j] -= F
        else
            exclusion_E -= α * 2 * f * charge_ij / sqrt_π
        end
    end
    return exclusion_E
end

function ewald_pe_forces(sys::System{3, AT}, inter::Ewald{T}) where {AT, T}
    n_atoms = length(sys)
    coords, boundary, energy_units = Array(sys.coords), sys.boundary, sys.energy_units
    dist_cutoff, error_tol = inter.dist_cutoff, inter.error_tol
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    nrx, nry, nrz = ewald_params.(boundary.side_lengths, α, error_tol)
    kmax = maximum((nrx, nry, nrz))
    if kmax < 1
        error("kmax for Ewald summation is $kmax, should be at least 1")
    end
    partial_charges = Array(charges(sys))
    V = volume(boundary)
    f = T(Molly.coulomb_const)
    Fs = zeros(SVector{3, typeof(zero(T) * sys.force_units)}, n_atoms)

    exclusion_E = excluded_interactions!(Fs, inter.excluded_pairs, partial_charges, coords,
                                         boundary, α, f, energy_units)

    recip_box_size = (2 * T(π)) ./ boundary.side_lengths
    eir = zeros(Complex{T}, kmax * n_atoms * 3)
    tab_xy = zeros(Complex{T}, n_atoms)
    tab_qxyz = zeros(Complex{T}, n_atoms)
    factor_ewald = -inv(4 * α^2)
    recip_coeff = f * 4 * T(π) / V
    reciprocal_space_E = zero(T) * energy_units

    for i in 1:n_atoms
        for m in 1:3
            eir[3*(i-1) + m] = Complex(one(T), zero(T))
            eir[n_atoms*3 + 3*(i-1) + m] = Complex(cos(coords[i][m]*recip_box_size[m]),
                                                   sin(coords[i][m]*recip_box_size[m]))
        end
        for j in 2:(kmax-1)
            for m in 1:3
                eir[j*n_atoms*3 + 3*(i-1) + m] = eir[(j-1)*n_atoms*3 + 3*(i-1) + m] *
                                                        eir[n_atoms*3 + 3*(i-1) + m]
            end
        end
    end

    lowry = 0
    lowrz = 1
    for rx in 0:(nrx-1)
        kx = rx * recip_box_size[1]
        for ry in lowry:(nry-1)
            ky = ry * recip_box_size[2]
            if ry >= 0
                for n in 1:n_atoms
                    tab_xy[n] = eir[rx*n_atoms*3 + 3*(n-1) + 1] * eir[ry*n_atoms*3 + 3*(n-1) + 2]
                end
            else
                for n in 1:n_atoms
                    tab_xy[n] = eir[rx*n_atoms*3 + 3*(n-1) + 1] *
                                        conj(eir[-ry*n_atoms*3 + 3*(n-1) + 2])
                end
            end
            for rz in lowrz:(nrz-1)
                if rz >= 0
                    for n in 1:n_atoms
                        tab_qxyz[n] = partial_charges[n] * tab_xy[n] *
                                            eir[rz*n_atoms*3 + 3*(n-1) + 3]
                    end
                else
                    for n in 1:n_atoms
                        tab_qxyz[n] = partial_charges[n] * tab_xy[n] *
                                            conj(eir[-rz*n_atoms*3 + 3*(n-1) + 3])
                    end
                end
                cs = sum(real, tab_qxyz)
                ss = sum(imag, tab_qxyz)
                kz = rz * recip_box_size[3]
                k2 = kx * kx + ky * ky + kz * kz
                ak = exp(k2 * factor_ewald) / k2
                for n in 1:n_atoms
                    F = ak * (cs * imag(tab_qxyz[n]) - ss * real(tab_qxyz[n]))
                    Fs[n] += 2 .* recip_coeff .* F .* SVector(kx, ky, kz)
                end
                reciprocal_space_E += recip_coeff * ak * (cs * cs + ss * ss)
                lowrz = 1 - nrz
            end
            lowry = 1 - nry
        end
    end

    charge_E = -f * T(π) * sum(partial_charges)^2 / (2 * V * α^2)
    self_E = f * -sum(abs2, partial_charges) * α / sqrt(T(π)) + charge_E
    total_E = reciprocal_space_E + self_E + exclusion_E
    return total_E, AT(Fs)
end

"""
    PME(boundary, n_atoms; dist_cutoff, error_tol=0.0005, order=5,
        ϵr=1.0, eligible=nothing)

Particle mesh Ewald summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator.

Should be used alongside the [`CoulombEwald`](@ref) pairwise interaction,
which provide the short range term.
The `dist_cutoff` and `error_tol` should match.

`dist_cutoff` is the cutoff distance for short range interactions.
`eligible` indicates pairs eligible for short range interaction, and can
be a matrix like the neighbor list or `nothing` to indicate that all pairs
are eligible.

Only compatible with 3D systems.
"""
struct PME{T, D, A} <: AbstractEwald
    dist_cutoff::D
    error_tol::T
    order::Int
    ϵr::T
    excluded_pairs::Vector{Tuple{Int32, Int32}}
    α::A
    mesh_dims::SVector{3, Int}
    grid_indices::Matrix{Int}
    grid_fractions::Matrix{T}
    bsplines_θ::Matrix{T}
    bsplines_dθ::Matrix{T}
    charge_grid::Array{Complex{T}, 3}
    bsplines_moduli::Vector{Vector{T}}
end

function PME(boundary, n_atoms; dist_cutoff, error_tol::T=0.0005, order=5,
             ϵr=one(error_tol), eligible=nothing, special=nothing) where T
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    mesh_dims = pme_params.(box_sides(boundary), α, error_tol)
    grid_indices, grid_fractions = zeros(Int, 3, n_atoms), zeros(T, 3, n_atoms)
    bsplines_θ, bsplines_dθ = zeros(T, order*n_atoms, 3), zeros(T, order*n_atoms, 3)
    # Ordered z/y/x for better memory access
    charge_grid = zeros(Complex{T}, mesh_dims[3], mesh_dims[2], mesh_dims[1])
    excluded_pairs = find_excluded_pairs(eligible)

    bsplines_moduli = [zeros(T, mesh_dims[1]), zeros(T, mesh_dims[2]), zeros(T, mesh_dims[3])]
    nmax = maximum(mesh_dims)
    data, ddata = zeros(T, order), zeros(T, order)
    bsplines_data = zeros(T, nmax)
    data[1] = one(T)
    for k in 3:(order-1)
        d = inv(k - one(T))
        data[k] = zero(T)
        for l in 1:(k-2)
            data[k-l] = d * (l * data[k-l-1] + (k-l) * data[k-l])
        end
        data[1] *= d
    end

    ddata[1] = -data[1]
    for k in 1:(order-1)
        ddata[k+1] = data[k] - data[k+1]
    end
    d = inv(order - one(T))
    data[order] = zero(T)

    for l in 1:(order-2)
        data[order-l] = d * (l * data[order-l-1] + (order-l) * data[order-l])
    end
    data[1] *= d

    for i in 1:order
        bsplines_data[i+1] = data[i]
    end

    for (d, ndata) in enumerate(mesh_dims)
        for i in 1:ndata
            sc, ss = zero(T), zero(T)
            for j in 1:ndata
                arg = 2 * T(π) * (i-1) * (j-1) / ndata
                sc += bsplines_data[j] * cos(arg)
                ss += bsplines_data[j] * sin(arg)
            end
            bsplines_moduli[d][i] = sc^2 + ss^2
        end
        for i in 1:ndata
            if bsplines_moduli[d][i] < T(1e-7)
                bsplines_moduli[d][i] = (bsplines_moduli[d][((i-2+ndata)%ndata)+1] +
                                         bsplines_moduli[d][(i%ndata)+1]) / 2
            end
        end
    end

    return PME(dist_cutoff, error_tol, order, ϵr, excluded_pairs, α, mesh_dims,
               grid_indices, grid_fractions, bsplines_θ, bsplines_dθ,
               charge_grid, bsplines_moduli)
end

function pme_params(side_length, α, error_tol::T) where T
    s = ceil(Int, 2α * side_length / (3 * error_tol^T(0.2)))
    return max(s, 6)
end

function grid_placement!(grid_indices, grid_fractions, coords, recip_box, mesh_dims)
    for i in eachindex(coords)
        for d in 1:3
            t = sum(coords[i] .* SVector(recip_box[1][d], recip_box[2][d], recip_box[3][d]))
            t = (t - floor(t)) * mesh_dims[d]
            ti = floor(Int, t)
            grid_fractions[d, i] = t - ti
            grid_indices[d, i] = ti % mesh_dims[d]
        end
    end
    return grid_indices, grid_fractions
end

function update_bsplines!(bsplines_θ::Matrix{T}, bsplines_dθ, grid_fractions, order) where T
    for i in axes(grid_fractions, 2)
        offset = (i - 1) * order
        for j in 1:3
            dr = grid_fractions[j, i]
            bsplines_θ[offset + order, j] = zero(T)
            bsplines_θ[offset + 2, j]     = dr
            bsplines_θ[offset + 1, j]     = 1 - dr
            for k in 3:(order-1)
                d = inv(k - one(T))
                bsplines_θ[offset + k, j] = d * dr * bsplines_θ[offset + k - 1, j]
                for l in 1:(k-2)
                    bsplines_θ[offset + k - l, j] = d * (
                            (dr + l) * bsplines_θ[offset + k - l - 1, j] +
                            (k - l - dr) * bsplines_θ[offset + k - l, j]
                        )
                end
                bsplines_θ[offset + 1, j] *= d * (1 - dr)
            end

            bsplines_dθ[offset + 1, j] = -bsplines_θ[offset + 1, j]
            for k in 1:(order-1)
                bsplines_dθ[offset + k + 1, j] = bsplines_θ[offset + k, j] -
                                                        bsplines_θ[offset + k + 1, j]
            end
            d = inv(order - one(T))
            bsplines_θ[offset + order, j] = d * dr * bsplines_θ[offset + order - 1, j]
            for l in 1:(order-2)
                bsplines_θ[offset + order - l, j] = d * (
                        (dr + l) * bsplines_θ[offset + order - l - 1, j] +
                        (order - l - dr) * bsplines_θ[offset + order - l, j]
                    )
            end
            bsplines_θ[offset + 1, j] *= d * (1 - dr)
        end
    end
    return bsplines_θ, bsplines_dθ
end

function spread_charge!(charge_grid::Array{Complex{T}, 3}, grid_indices, bsplines_θ, mesh_dims,
                        order, atoms) where T
    charge_grid .= Complex(zero(T), zero(T))
    for i in eachindex(atoms)
        q = charge(atoms[i])
        x0index, y0index, z0index = grid_indices[1, i], grid_indices[2, i], grid_indices[3, i]
        for ix in 0:(order-1)
            xindex = (x0index + ix) % mesh_dims[1]
            for iy in 0:(order-1)
                yindex = (y0index + iy) % mesh_dims[2]
                for iz in 0:(order-1)
                    zindex = (z0index + iz) % mesh_dims[3]
                    cb = q * bsplines_θ[(i-1)*order+ix+1, 1] *
                                bsplines_θ[(i-1)*order+iy+1, 2] * bsplines_θ[(i-1)*order+iz+1, 3]
                    charge_grid[zindex+1, yindex+1, xindex+1] += Complex(cb, zero(T))
                end
            end
        end
    end
    return charge_grid
end

function recip_conv!(charge_grid::Array{Complex{T}, 3}, bsplines_moduli, recip_box,
                     ϵr, α, mesh_dims, boundary, energy_units) where T
    f = T(Molly.coulomb_const) / ϵr
    factor = T(π)^2 / α^2
    nx, ny, nz = mesh_dims
    maxkx, maxky, maxkz = T(0.5)*(nx+1), T(0.5)*(ny+1), T(0.5)*(nz+1)
    boxfactor = T(π) * volume(boundary)
    esum = zero(T) * energy_units
    for kx in 0:(nx-1)
        mx = (kx < maxkx ? kx : kx - nx)
        mhx = mx * recip_box[1][1]
        bx = boxfactor * bsplines_moduli[1][kx+1]
        for ky in 0:(ny-1)
            my = (ky < maxky ? ky : ky - ny)
            mhy = mx * recip_box[2][1] + my * recip_box[2][2]
            by = bsplines_moduli[2][ky+1]
            for kz in 0:(nz-1)
                if iszero(kx) && iszero(ky) && iszero(kz)
                    continue
                end
                mz = (kz < maxkz ? kz : kz - nz)
                mhz = mx * recip_box[3][1] + my * recip_box[3][2] + mz * recip_box[3][3]
                d1, d2 = reim(charge_grid[kz+1, ky+1, kx+1])
                m2 = mhx^2 + mhy^2 + mhz^2
                bz = bsplines_moduli[3][kz+1]
                denom = m2 * bx * by * bz
                eterm = f * exp(-factor * m2) / denom
                eterm_nou = ustrip(energy_units, eterm)
                charge_grid[kz+1, ky+1, kx+1] = Complex(d1*eterm_nou, d2*eterm_nou)
                struct2 = d1^2 + d2^2
                esum += eterm * struct2
            end
        end
    end
    return esum / 2
end

function interpolate_force!(Fs, charge_grid::Array{Complex{T}, 3}, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms) where T
    nx, ny, nz = mesh_dims
    for i in eachindex(atoms)
        fx, fy, fz = zero(T), zero(T), zero(T)
        q = charge(atoms[i])
        x0index, y0index, z0index = grid_indices[1, i], grid_indices[2, i], grid_indices[3, i]
        for ix in 0:(order-1)
            xindex = (x0index + ix) % mesh_dims[1]
            tx, dtx = bsplines_θ[(i-1)*order+ix+1, 1], bsplines_dθ[(i-1)*order+ix+1, 1]
            for iy in 0:(order-1)
                yindex = (y0index + iy) % mesh_dims[2]
                ty, dty = bsplines_θ[(i-1)*order+iy+1, 2], bsplines_dθ[(i-1)*order+iy+1, 2]
                for iz in 0:(order-1)
                    zindex = (z0index + iz) % mesh_dims[3]
                    tz, dtz = bsplines_θ[(i-1)*order+iz+1, 3], bsplines_dθ[(i-1)*order+iz+1, 3]
                    gridvalue = real(charge_grid[zindex+1, yindex+1, xindex+1])
                    fx += dtx * ty * tz * gridvalue
                    fy += tx * dty * tz * gridvalue
                    fz += tx * ty * dtz * gridvalue
                end
            end
        end
        Fs[i] -= SVector(
            q * (fx*nx*recip_box[1][1]),
            q * (fx*nx*recip_box[2][1] + fy*ny*recip_box[2][2]),
            q * (fx*nx*recip_box[3][1] + fy*ny*recip_box[3][2] + fz*nz*recip_box[3][3]),
        ) * energy_units
    end
    return Fs
end

function ewald_pe_forces(sys::System{3, AT}, inter::PME{T}) where {AT, T}
    atoms, coords, boundary, energy_units = sys.atoms, sys.coords, sys.boundary, sys.energy_units
    order, ϵr, α, mesh_dims = inter.order, inter.ϵr, inter.α, inter.mesh_dims
    partial_charges = Array(charges(sys))
    V = volume(boundary)
    f = T(Molly.coulomb_const)
    Fs = zeros(SVector{3, typeof(zero(T) * sys.force_units)}, length(sys))

    exclusion_E = excluded_interactions!(Fs, inter.excluded_pairs, partial_charges, coords,
                                         boundary, α, f, energy_units)

    recip_box = invert_box_vectors(boundary)
    grid_placement!(inter.grid_indices, inter.grid_fractions, coords, recip_box, mesh_dims)
    update_bsplines!(inter.bsplines_θ, inter.bsplines_dθ, inter.grid_fractions, order)
    spread_charge!(inter.charge_grid, inter.grid_indices, inter.bsplines_θ, mesh_dims,
                   order, atoms)
    fft!(inter.charge_grid)
    reciprocal_space_E = recip_conv!(inter.charge_grid, inter.bsplines_moduli, recip_box, ϵr,
                    α, mesh_dims, boundary, energy_units)
    bfft!(inter.charge_grid)
    interpolate_force!(Fs, inter.charge_grid, inter.grid_indices, inter.bsplines_θ,
                       inter.bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms)

    charge_E = -f * T(π) * sum(partial_charges)^2 / (2 * V * α^2)
    self_E = f * -sum(abs2, partial_charges) * α / sqrt(T(π)) + charge_E
    total_E = reciprocal_space_E + self_E + exclusion_E
    return total_E, Fs
end
