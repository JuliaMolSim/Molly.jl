
# Long range electrostatic summation methods
# Based on the OpenMM source code

export Ewald

"""
    Ewald(; error_tol=0.0005, dist_cutoff=1.0u"nm", eligible=nothing,
            energy_units=u"kJ * mol^-1", length_units=u"nm")

Ewald summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator.

Calculates both the short and long range interactions, so should replace
[`Coulomb`](@ref) rather than being used alongside it.
`dist_cutoff` is the cutoff distance for short range interactions.
`eligible` indicates pairs eligible for short range interaction, and can
be a matrix like the neighbor list or `nothing` to indicate that all pairs
are eligible.

Only compatible with 3D systems and [`CubicBoundary`](@ref).
Runs on the CPU, even for GPU systems.
"""
@kwdef struct Ewald{T, D, E, EU, LU}
    error_tol::T = 0.0005
    dist_cutoff::D = 1.0u"nm"
    eligible::E = nothing
    energy_units::EU = u"kJ * mol^-1"
    length_units::LU = u"nm"
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

function ewald_pe_forces(sys::System{3, AT}, inter::Ewald{T}) where {AT, T}
    n_atoms = length(sys)
    coords, boundary = Array(sys.coords), sys.boundary
    dist_cutoff, error_tol = inter.dist_cutoff, inter.error_tol
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    nrx, nry, nrz = ewald_params.(boundary.side_lengths, α, error_tol)
    kmax = maximum((nrx, nry, nrz))
    if kmax < 1
        error("kmax for Ewald summation is $kmax, should be at least 1")
    end
    eligible = (isnothing(inter.eligible) ? nothing : Array(inter.eligible))
    partial_charges = Array(charges(sys))
    V = volume(boundary)
    sq_dist_cutoff = dist_cutoff^2
    f = T(Molly.coulomb_const)
    sqrt_π = sqrt(T(π))
    real_space_E = zero(T) * sys.energy_units
    exclusion_E = zero(real_space_E)
    Fs = zeros(SVector{3, typeof(zero(T) * sys.force_units)}, n_atoms)

    for i in 1:n_atoms
        charge_i = partial_charges[i]
        for j in (i + 1):n_atoms
            charge_ij = charge_i * partial_charges[j]
            if isnothing(eligible) || eligible[i, j]
                vec_ij = vector(coords[i], coords[j], boundary)
                r2 = sum(abs2, vec_ij)
                if r2 <= sq_dist_cutoff
                    r = sqrt(r2)
                    inv_r = inv(r)
                    αr = α * r
                    erfc_αr = erfc(αr)
                    real_space_E += f * charge_ij * erfc_αr * inv_r
                    dE_dr = f * charge_ij * inv_r^3 * (erfc_αr + 2 * αr * exp(-αr^2) / sqrt_π)
                    F = dE_dr * vec_ij
                    Fs[i] -= F
                    Fs[j] += F
                end
            else
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
        end
    end

    recip_box_size = (2 * T(π)) ./ boundary.side_lengths
    eir = zeros(Complex{T}, kmax * n_atoms * 3)
    tab_xy = zeros(Complex{T}, n_atoms)
    tab_qxyz = zeros(Complex{T}, n_atoms)
    factor_ewald = -inv(4 * α^2)
    recip_coeff = f * 4 * T(π) / V
    reciprocal_space_E = zero(real_space_E)

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
    self_E = f * -sum(abs2, partial_charges) * α / sqrt_π + charge_E
    total_E = real_space_E + reciprocal_space_E + self_E + exclusion_E
    return total_E, AT(Fs)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                                inter::Ewald; kwargs...)
    return ewald_pe_forces(sys, inter)[1]
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(sys,
                                                                inter::Ewald; kwargs...)
    return ewald_pe_forces(sys, inter)[2]
end

function AtomsCalculators.energy_forces(sys, inter::Ewald; kwargs...)
    E, Fs = ewald_pe_forces(sys, inter)
    return (energy=E, forces=Fs)
end

AtomsCalculators.energy_unit(inter::Ewald) = inter.energy_units
AtomsCalculators.length_unit(inter::Ewald) = inter.length_units
