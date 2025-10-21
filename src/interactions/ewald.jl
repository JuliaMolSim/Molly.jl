# Long range electrostatic summation methods
# Based on the OpenMM source code

export
    Ewald,
    PME

abstract type AbstractEwald end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                            inter::AbstractEwald;
                                            n_threads::Integer=Threads.nthreads(),
                                            kwargs...)
    pe = ewald_pe_forces!(nothing, nothing, sys, inter, Val(false); n_threads=n_threads)
    return pe
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs,
                                            sys::System,
                                            inter::AbstractEwald;
                                            n_threads::Integer=Threads.nthreads(),
                                            buffers=nothing,
                                            needs_vir=false,
                                            kwargs...)
    vir = (needs_vir ? buffers.virial : nothing)
    pe = ewald_pe_forces!(fs, vir, sys, inter, Val(needs_vir); n_threads=n_threads)
    return fs
end

function AtomsCalculators.energy_forces!(fs::AbstractVector, # Required for disambiguation
                                         sys::System,
                                         inter::AbstractEwald;
                                         n_threads::Integer=Threads.nthreads(),
                                         kwargs...)
    pe = ewald_pe_forces!(fs, nothing, sys, inter, Val(false); n_threads=n_threads)
    return (energy=pe, forces=fs)
end

function AtomsCalculators.energy_forces(sys::System,
                                        inter::AbstractEwald;
                                        n_threads::Integer=Threads.nthreads(),
                                        kwargs...)
    fs = zero_forces(sys)
    pe = ewald_pe_forces!(fs, nothing, sys, inter, Val(false); n_threads=n_threads)
    return (energy=pe, forces=fs)
end

function find_excluded_pairs(eligible, special)
    excluded_pairs = Tuple{Int32, Int32}[]
    if !(isnothing(eligible) && isnothing(special))
        n_atoms = (isnothing(eligible) ? size(special, 1) : size(eligible, 1))
        eligible_cpu = (isnothing(eligible) ? trues( n_atoms, n_atoms) : from_device(eligible))
        special_cpu  = (isnothing(special ) ? falses(n_atoms, n_atoms) : from_device(special ))
        for i in 1:n_atoms
            for j in (i+1):n_atoms
                if !eligible_cpu[i, j] || special_cpu[i, j]
                    push!(excluded_pairs, (Int32(i), Int32(j)))
                end
            end
        end
    end
    return excluded_pairs
end

function excluded_interactions_inner!(Fs, vir, atoms, coords, boundary, α, f, i, j,
                            ::Val{T}, ::Val{calculate_forces}, ::Val{atomic},
                            ::Val{needs_vir}) where {T, calculate_forces, atomic, needs_vir}
    sqrt_π = sqrt(T(π))
    charge_ij = charge(atoms[i]) * charge(atoms[j])
    vec_ij = vector(coords[i], coords[j], boundary)
    r = norm(vec_ij)
    αr = α * r
    erf_αr = erf(αr)
    if erf_αr > T(1e-6)
        inv_r = inv(r)
        exclusion_E = -f * charge_ij * inv_r * erf_αr
        if calculate_forces
            dE_dr = f * charge_ij * inv_r^3 * (erf_αr - 2 * αr * exp(-αr^2) / sqrt_π)
            F = dE_dr * vec_ij
            if atomic
                for dim in 1:3
                    fval = ustrip(F[dim])
                    Atomix.@atomic Fs[dim, i] +=  fval
                    Atomix.@atomic Fs[dim, j] += -fval
                    if needs_vir
                        for alpha in 1:3
                            Atomix.@atomic vir[alpha, dim] += ustrip(vec_ij[alpha]) * fval
                        end
                    end
                end
            else
                Fs[i] += F
                Fs[j] -= F
                if needs_vir
                    vir .+= vec_ij * transpose(F)
                end
            end
        end
    else
        exclusion_E = -α * 2 * f * charge_ij / sqrt_π
    end
    return exclusion_E
end

function excluded_interactions!(Fs, vir, buffer_Fs, virial_buffer, buffer_Es, excluded_pairs, atoms,
                                coords::Vector, boundary, α, f, force_units, energy_units,
                                calculate_forces, ::Val{T},
                                ::Val{needs_vir}) where {T, needs_vir}
    exclusion_E = zero(T) * energy_units
    for (i, j) in excluded_pairs
        E = excluded_interactions_inner!(Fs, vir, atoms, coords, boundary, α, f,
                            i, j, Val(T), Val(calculate_forces), Val(false), Val(needs_vir))
        exclusion_E += E
    end
    return exclusion_E
end

function excluded_interactions!(Fs, vir, buffer_Fs, virial_buffer, buffer_Es, excluded_pairs, atoms,
                                coords::AbstractVector{SVector{D, C}}, boundary, α, f, force_units,
                                energy_units, calculate_forces, ::Val{T},
                                ::Val{needs_vir}) where {D, C, T, needs_vir}
    if calculate_forces
        buffer_Fs .= zero(T)
        if needs_vir
            virial_buffer .= zero(T)
        end
    end
    backend = get_backend(atoms)
    n_threads_gpu = 128
    kernel! = excluded_interactions_kernel!(backend, n_threads_gpu)
    kernel!(buffer_Fs, virial_buffer, buffer_Es, excluded_pairs, atoms, coords, boundary, α, f,
            energy_units, Val(T), Val(calculate_forces), Val(needs_vir);
            ndrange=length(excluded_pairs))

    if calculate_forces
        Fs .+= reinterpret(SVector{D, T}, vec(buffer_Fs)) .* force_units
        if needs_vir
            vir .+= from_device(virial_buffer) .* energy_units
        end
    end
    return sum(buffer_Es) * energy_units
end

@kernel function excluded_interactions_kernel!(Fs_mat, vir, exclusion_Es, @Const(excluded_pairs),
                            @Const(atoms), @Const(coords), boundary, α, f, energy_units,
                            ::Val{T}, ::Val{calculate_forces},
                            ::Val{needs_vir}) where {T, calculate_forces, needs_vir}
    ei = @index(Global, Linear)
    if ei <= length(excluded_pairs)
        i, j = excluded_pairs[ei]
        E = excluded_interactions_inner!(Fs_mat, vir, atoms, coords, boundary, α, f,
                                i, j, Val(T), Val(calculate_forces), Val(true), Val(needs_vir))
        exclusion_Es[ei] = ustrip(energy_units, E)
    end
end

"""
    Ewald(dist_cutoff; error_tol=0.0005, eligible=nothing, special=nothing)

Ewald summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator.

Should be used alongside the [`CoulombEwald`](@ref) pairwise interaction,
which provide the short range term.
`dist_cutoff` and `error_tol` should match [`CoulombEwald`](@ref).

`dist_cutoff` is the cutoff distance for short range interactions.
`eligible` indicates pairs eligible for short range interaction, and can
be a matrix like the neighbor list or `nothing` to indicate that all pairs
are eligible.
`special` should also be given where relevant, as these interactions are
excluded from long range calculation.

This algorithm is O(N^2) and in general [`PME`](@ref) should be used instead.
Only compatible with 3D systems and [`CubicBoundary`](@ref).
Runs on the CPU, even for GPU systems.
"""
struct Ewald{T, D} <: AbstractEwald
    dist_cutoff::D
    error_tol::T
    excluded_pairs::Vector{Tuple{Int32, Int32}}
end

function Ewald(dist_cutoff; error_tol=0.0005, eligible=nothing, special=nothing)
    T = typeof(ustrip(dist_cutoff))
    excluded_pairs = find_excluded_pairs(eligible, special)
    return Ewald(dist_cutoff, T(error_tol), excluded_pairs)
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

function ewald_pe_forces!(Fs, vir, sys::System{3}, inter::AbstractEwald, ::Val{needs_vir};
                          n_threads::Integer=Threads.nthreads()) where needs_vir
    calculate_forces = !isnothing(Fs)
    return ewald_pe_forces!(Fs, vir, inter, sys.atoms, sys.coords, sys.boundary, sys.force_units,
                            sys.energy_units, Val(needs_vir), calculate_forces;
                            n_threads=n_threads)
end

function ewald_pe_forces!(Fs, vir, inter::Ewald{T}, atoms, coords, boundary, force_units,
                          energy_units, ::Val{needs_vir}, calculate_forces=true;
                          n_threads::Integer=Threads.nthreads()) where {T, needs_vir}
    AT = array_type(atoms)
    n_atoms = length(atoms)
    atoms_cpu, coords_cpu = from_device(atoms), from_device(coords)
    dist_cutoff, error_tol = inter.dist_cutoff, inter.error_tol
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol))
    nrx, nry, nrz = ewald_params.(boundary.side_lengths, α, error_tol)
    kmax = maximum((nrx, nry, nrz))
    if kmax < 1
        error("kmax for Ewald summation is $kmax, should be at least 1")
    end
    partial_charges_cpu = charge.(atoms_cpu)
    V = volume(boundary)
    f = (energy_units == NoUnits ? ustrip(T(Molly.coulomb_const)) : T(Molly.coulomb_const))
    if AT <: AbstractGPUArray && calculate_forces
        Fs_cpu = zeros(SVector{3, typeof(zero(T) * force_units)}, n_atoms)
    else
        Fs_cpu = Fs
    end

    exclusion_E = excluded_interactions!(Fs_cpu, vir, nothing, nothing, nothing,
                        inter.excluded_pairs, atoms_cpu, coords_cpu, boundary, α, f, force_units,
                        energy_units, calculate_forces, Val(T), Val(needs_vir))

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
            eir[n_atoms*3 + 3*(i-1) + m] = Complex(cos(coords_cpu[i][m]*recip_box_size[m]),
                                                   sin(coords_cpu[i][m]*recip_box_size[m]))
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
                        tab_qxyz[n] = partial_charges_cpu[n] * tab_xy[n] *
                                            eir[rz*n_atoms*3 + 3*(n-1) + 3]
                    end
                else
                    for n in 1:n_atoms
                        tab_qxyz[n] = partial_charges_cpu[n] * tab_xy[n] *
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
                    if calculate_forces
                        Fs_cpu[n] += 2 .* recip_coeff .* F .* SVector(kx, ky, kz)
                    end
                end

                if needs_vir
                    Ek = recip_coeff * ak * (cs*cs + ss*ss) # E_k
                    invk2 = one(T)/k2
                    cfac  = 2*(one(T) + (-factor_ewald)*k2) * invk2 # 2*(1 + k^2/(4α^2))/k^2
                    gxx = 1 - cfac*kx*kx
                    gxy =   - cfac*kx*ky
                    gxz =   - cfac*kx*kz
                    gyy = 1 - cfac*ky*ky
                    gyz =   - cfac*ky*kz
                    gzz = 1 - cfac*kz*kz
                    vir .+= Ek .* SMatrix{3,3,T}(gxx,gxy,gxz, gxy,gyy,gyz, gxz,gyz,gzz)
                end

                reciprocal_space_E += recip_coeff * ak * (cs * cs + ss * ss)
                lowrz = 1 - nrz
            end
            lowry = 1 - nry
        end
    end

    charge_E = -f * T(π) * sum(partial_charges_cpu)^2 / (2 * V * α^2)
    self_E = f * -sum(abs2, partial_charges_cpu) * α / sqrt(T(π)) + charge_E
    total_E = reciprocal_space_E + self_E + exclusion_E

    if needs_vir
        # E_charge = -A/V with A = f*π*Q^2/(2α^2) ⇒ W_charge = -E_charge * I
        vir .+= (-charge_E) .* I(3)
    end

    if calculate_forces && AT <: AbstractGPUArray
        Fs .+= to_device(Fs_cpu, AT)
    end
    return total_E
end

"""
    PME(dist_cutoff, atoms, boundary; error_tol=0.0005, order=5,
        ϵr=1.0, fixed_charges=true, eligible=nothing, special=nothing,
        grad_safe=false, n_threads=Threads.nthreads())

Particle mesh Ewald summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator.

Should be used alongside the [`CoulombEwald`](@ref) pairwise interaction,
which provide the short range term.
`dist_cutoff` and `error_tol` should match [`CoulombEwald`](@ref).

`dist_cutoff` is the cutoff distance for short range interactions.
`eligible` indicates pairs eligible for short range interaction, and can
be a matrix like the neighbor list or `nothing` to indicate that all pairs
are eligible.
`special` should also be given where relevant, as these interactions are
excluded from long range calculation.
`fixed_charges` should be set to `false` if the partial charges can change,
for example when using a polarizable force field.
`grad_safe` should be set to `true` if gradients are going to be calculated
with Enzyme.jl.
`n_threads` is used to pre-allocate memory on CPU.

This implementation is based on the implementation in OpenMM, which
is based on the smooth PME algorithm from
[Essmann et al. 1995](https://doi.org/10.1063/1.470117).

Only compatible with 3D systems.
"""
struct PME{T, D, E, A, I, M, BM, C, CB, FB, EB, RB, VB, P, F, B} <: AbstractEwald
    dist_cutoff::D
    error_tol::T
    order::Int
    ϵr::T
    excluded_pairs::E
    α::A
    mesh_dims::SVector{3, Int}
    grid_indices::I
    grid_fractions::M
    bsplines_θ::M
    bsplines_dθ::M
    bsplines_moduli_x::BM
    bsplines_moduli_y::BM
    bsplines_moduli_z::BM
    charge_grid::C
    charge_grid_buffer::CB
    excluded_buffer_Fs::FB
    excluded_buffer_Es::EB
    recip_conv_buffer::RB
    virial_buffer::VB
    pc_sum::P
    pc_abs2_sum::P
    fft_plan::F
    bfft_plan::B
    grad_safe::Bool
end

function PME(dist_cutoff, atoms, boundary; error_tol=0.0005, order=5,
             ϵr=1.0, fixed_charges=true, eligible=nothing, special=nothing, grad_safe=false,
             n_threads::Integer=Threads.nthreads())
    T = typeof(ustrip(dist_cutoff))
    AT = array_type(atoms)
    n_atoms = length(atoms)
    error_tol_T = T(error_tol)
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol_T))
    mesh_dims = pme_params.(box_sides(boundary), α, error_tol_T)
    grid_indices = to_device(zeros(Int, 3, n_atoms), AT)
    grid_fractions = to_device(zeros(T, 3, n_atoms), AT)
    bsplines_θ = to_device(zeros(T, order * n_atoms, 3), AT)
    bsplines_dθ = zero(bsplines_θ)
    # Ordered z/y/x for better memory access
    charge_grid = to_device(zeros(Complex{T}, mesh_dims[3], mesh_dims[2], mesh_dims[1]), AT)
    excluded_pairs = to_device(find_excluded_pairs(eligible, special), AT)

    bsplines_moduli = (zeros(T, mesh_dims[1]), zeros(T, mesh_dims[2]), zeros(T, mesh_dims[3]))
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

    if AT <: AbstractGPUArray
        charge_grid_buffer = to_device(zeros(T, size(charge_grid)), AT)
        recip_conv_buffer  = to_device(zeros(T, mesh_dims...), AT)
        excluded_buffer_Fs = to_device(zeros(T, 3, n_atoms), AT)
        excluded_buffer_Es = to_device(zeros(T, length(excluded_pairs)), AT)
        virial_buffer      = to_device(zeros(T, 3, 3), AT)
    elseif n_threads > 1
        charge_grid_buffer = [zero(charge_grid) for _ in 1:n_threads]
        recip_conv_buffer = zeros(T, n_threads)
        excluded_buffer_Fs, excluded_buffer_Es = nothing, nothing
        virial_buffer = [zeros(T, 3, 3) for _ in 1:n_threads]
    else
        charge_grid_buffer, recip_conv_buffer = nothing, nothing
        excluded_buffer_Fs, excluded_buffer_Es = nothing, nothing
        virial_buffer = [zeros(T, 3, 3)]
    end

    if fixed_charges && !grad_safe
        partial_charges = charge.(atoms)
        pc_sum = sum(partial_charges)
        pc_abs2_sum = sum(abs2, partial_charges)
    else
        pc_sum, pc_abs2_sum = nothing, nothing
    end

    fft_plan  = plan_fft!(charge_grid)
    bfft_plan = plan_bfft!(charge_grid)
    bsm_x = to_device(bsplines_moduli[1], AT)
    bsm_y = to_device(bsplines_moduli[2], AT)
    bsm_z = to_device(bsplines_moduli[3], AT)

    return PME(dist_cutoff, error_tol_T, order, T(ϵr), excluded_pairs, α, mesh_dims,
               grid_indices, grid_fractions, bsplines_θ, bsplines_dθ, bsm_x, bsm_y, bsm_z,
               charge_grid, charge_grid_buffer, excluded_buffer_Fs, excluded_buffer_Es,
               recip_conv_buffer, virial_buffer, pc_sum, pc_abs2_sum, fft_plan,
               bfft_plan, grad_safe)
end

zero_or_nothing(x) = zero(x)
zero_or_nothing(x::Nothing) = nothing
zero_or_nothing(x::Vector{Matrix{T}}) where {T} = zero.(x) # Required for Julia 1.10

function Base.zero(pme::PME)
    if pme.charge_grid_buffer isa Vector
        charge_grid_buffer = zero.(pme.charge_grid_buffer)
    else
        charge_grid_buffer = zero_or_nothing(pme.charge_grid_buffer)
    end
    return PME(
        zero(pme.dist_cutoff),
        zero(pme.error_tol),
        pme.order,
        zero(pme.ϵr),
        pme.excluded_pairs,
        zero(pme.α),
        pme.mesh_dims,
        zero(pme.grid_indices),
        zero(pme.grid_fractions),
        zero(pme.bsplines_θ),
        zero(pme.bsplines_dθ),
        zero(pme.bsplines_moduli_x),
        zero(pme.bsplines_moduli_y),
        zero(pme.bsplines_moduli_z),
        zero(pme.charge_grid),
        charge_grid_buffer,
        zero_or_nothing(pme.excluded_buffer_Fs),
        zero_or_nothing(pme.excluded_buffer_Es),
        zero_or_nothing(pme.recip_conv_buffer),
        zero_or_nothing(pme.virial_buffer),
        zero_or_nothing(pme.pc_sum),
        zero_or_nothing(pme.pc_abs2_sum),
        pme.fft_plan,
        pme.bfft_plan,
        pme.grad_safe,
    )
end

function pme_params(side_length, α, error_tol::T) where T
    s = ceil(Int, 2α * side_length / (3 * error_tol^T(0.2)))
    return max(s, 6)
end

function grid_placement_inner!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, i)
    @inbounds for d in 1:3
        t = sum(coords[i] .* SVector(recip_box[1][d], recip_box[2][d], recip_box[3][d]))
        t = (t - floor(t)) * mesh_dims[d]
        ti = floor(Int, t)
        grid_fractions[d, i] = t - ti
        grid_indices[d, i] = ti % mesh_dims[d]
    end
    return grid_indices, grid_fractions
end

function grid_placement!(grid_indices::Matrix, grid_fractions, coords, recip_box, mesh_dims)
    for i in eachindex(coords)
        grid_placement_inner!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, i)
    end
    return grid_indices, grid_fractions
end

function grid_placement!(grid_indices, grid_fractions, coords, recip_box, mesh_dims)
    backend = get_backend(grid_indices)
    n_threads_gpu = 128
    kernel! = grid_placement_kernel!(backend, n_threads_gpu)
    kernel!(grid_indices, grid_fractions, coords, recip_box, mesh_dims; ndrange=length(coords))
    return grid_indices, grid_fractions
end

@kernel function grid_placement_kernel!(grid_indices, grid_fractions, @Const(coords),
                                        recip_box, mesh_dims)
    i = @index(Global, Linear)
    if i <= length(coords)
        grid_placement_inner!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, i)
    end
end

function update_bsplines_inner!(bsplines_θ::AbstractArray{T, 2}, bsplines_dθ, grid_fractions,
                                order, i) where T
    offset = (i - 1) * order
    @inbounds for j in 1:3
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
    return bsplines_θ, bsplines_dθ
end

function update_bsplines!(bsplines_θ::Matrix, bsplines_dθ, grid_fractions, order,
                          n_threads)
    n_atoms = size(grid_fractions, 2)
    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:n_atoms
            update_bsplines_inner!(bsplines_θ, bsplines_dθ, grid_fractions,
                                   order, i)
        end
    end
    return bsplines_θ, bsplines_dθ
end

function update_bsplines!(bsplines_θ, bsplines_dθ, grid_fractions, order,
                          n_threads)
    n_atoms = size(grid_fractions, 2)
    backend = get_backend(bsplines_θ)
    n_threads_gpu = 128
    kernel! = update_bsplines_kernel!(backend, n_threads_gpu)
    kernel!(bsplines_θ, bsplines_dθ, grid_fractions, order; ndrange=n_atoms)
    return bsplines_θ, bsplines_dθ
end

@kernel function update_bsplines_kernel!(bsplines_θ, bsplines_dθ, @Const(grid_fractions),
                                         order)
    i = @index(Global, Linear)
    n_atoms = size(grid_fractions, 2)
    if i <= n_atoms
        update_bsplines_inner!(bsplines_θ, bsplines_dθ, grid_fractions, order, i)
    end
end

@inline function spread_charge_inner!(charge_grid, grid_indices, bsplines_θ,
                              mesh_dims, order, atoms, i, ::Val{atomic}) where atomic
    q = charge(atoms[i])
    @inbounds x0index, y0index, z0index = grid_indices[1, i], grid_indices[2, i], grid_indices[3, i]
    @inbounds for ix in 0:(order-1)
        xindex = (x0index + ix) % mesh_dims[1]
        for iy in 0:(order-1)
            yindex = (y0index + iy) % mesh_dims[2]
            for iz in 0:(order-1)
                zindex = (z0index + iz) % mesh_dims[3]
                cb = q * bsplines_θ[(i-1)*order+ix+1, 1] *
                            bsplines_θ[(i-1)*order+iy+1, 2] * bsplines_θ[(i-1)*order+iz+1, 3]
                if atomic
                    # Atomic doesn't work for complex numbers, this is just the real part
                    Atomix.@atomic charge_grid[zindex+1, yindex+1, xindex+1] += cb
                else
                    charge_grid[zindex+1, yindex+1, xindex+1] += Complex(cb, zero(cb))
                end
            end
        end
    end
    return charge_grid
end

function spread_charge!(charge_grid::Array{Complex{T}, 3}, buffer, grid_indices,
                        bsplines_θ, mesh_dims, order, atoms, ::Val{1}) where T
    charge_grid .= zero(Complex{T})
    for i in eachindex(atoms)
        spread_charge_inner!(charge_grid, grid_indices, bsplines_θ, mesh_dims,
                             order, atoms, i, Val(false))
    end
    return charge_grid, buffer
end

function spread_charge!(charge_grid::Array{Complex{T}, 3}, buffer, grid_indices, bsplines_θ,
                        mesh_dims, order, atoms, ::Val{n_threads}) where {T, n_threads}
    Threads.@threads for chunk_i in 1:n_threads
        buffer[chunk_i] .= zero(Complex{T})
        for i in chunk_i:n_threads:length(atoms)
            spread_charge_inner!(buffer[chunk_i], grid_indices, bsplines_θ,
                                 mesh_dims, order, atoms, i, Val(false))
        end
    end
    charge_grid .= buffer[1]
    for chunk_i in 2:n_threads
        charge_grid .+= buffer[chunk_i]
    end
    return charge_grid, buffer
end

function spread_charge!(charge_grid::AbstractArray{Complex{T}, 3}, buffer, grid_indices,
                        bsplines_θ, mesh_dims, order, atoms, n_threads_val) where T
    backend = get_backend(charge_grid)
    n_threads_gpu = 128
    kernel! = spread_charge_kernel!(backend, n_threads_gpu)
    buffer .= zero(T)
    kernel!(buffer, grid_indices, bsplines_θ, mesh_dims, order, atoms; ndrange=length(atoms))
    charge_grid .= Complex.(buffer, zero(T))
    return charge_grid, buffer
end

@kernel function spread_charge_kernel!(charge_grid_real, @Const(grid_indices), @Const(bsplines_θ),
                                       mesh_dims, order, atoms)
    i = @index(Global, Linear)
    if i <= length(atoms)
        spread_charge_inner!(charge_grid_real, grid_indices, bsplines_θ, mesh_dims, order, atoms,
                             i, Val(true))
    end
end

function recip_conv_inner!(vir_nou, charge_grid::AbstractArray{Complex{T}, 3}, bsm_x, bsm_y, bsm_z,
                           recip_box, mesh_dims, energy_units, f_div_ϵr, factor, boxfactor,
                           kx, ky, kz, ::Val{needs_vir},
                           ::Val{atomic}) where {T, needs_vir, atomic}
    if iszero(kx) && iszero(ky) && iszero(kz)
        return zero(T) * energy_units
    end
    nx, ny, nz = mesh_dims
    maxkx, maxky, maxkz = T(0.5)*(nx+1), T(0.5)*(ny+1), T(0.5)*(nz+1)
    @inbounds begin
        mx = (kx < maxkx ? kx : kx - nx)
        mhx = mx * recip_box[1][1]
        bx = boxfactor * bsm_x[kx+1]
        my = (ky < maxky ? ky : ky - ny)
        mhy = mx * recip_box[2][1] + my * recip_box[2][2]
        by = bsm_y[ky+1]
        mz = (kz < maxkz ? kz : kz - nz)
        mhz = mx * recip_box[3][1] + my * recip_box[3][2] + mz * recip_box[3][3]
        d1, d2 = reim(charge_grid[kz+1, ky+1, kx+1])
        m2 = mhx^2 + mhy^2 + mhz^2
        bz = bsm_z[kz+1]
        denom = m2 * bx * by * bz
        c  = exp(-factor * m2)
        eterm = f_div_ϵr * c / denom
        eterm_nou = ustrip(energy_units, eterm)
        charge_grid[kz+1, ky+1, kx+1] = Complex(d1*eterm_nou, d2*eterm_nou)
        struct2 = d1^2 + d2^2

        if needs_vir
            # V*P_k = E_k * [I - 2(1 + factor*m2) * (m ⊗ m) / m2], symmetric by construction.
            Ek = eterm * struct2
            invm2 = one(T) / m2
            coeff = 2*one(T) * (one(T) + factor*m2) * invm2
            gxx = 1 - coeff*mhx*mhx
            gxy =   - coeff*mhx*mhy
            gxz =   - coeff*mhx*mhz
            gyy = 1 - coeff*mhy*mhy
            gyz =   - coeff*mhy*mhz
            gzz = 1 - coeff*mhz*mhz
            G = SMatrix{3, 3, T}(gxx, gxy, gxz,
                                 gxy, gyy, gyz,
                                 gxz, gyz, gzz)
            Ek_nou = ustrip(energy_units, Ek)
            if atomic
                for d1 in 1:3
                    for d2 in 1:3
                        Atomix.@atomic vir_nou[d1, d2] += Ek_nou * G[d1, d2]
                    end
                end
            else
                vir_nou .+= Ek_nou .* G
            end
        end
    end
    return eterm * struct2
end

function recip_conv!(vir, buffer_virial, charge_grid::Array{Complex{T}, 3}, buffer,
                     bsm_x, bsm_y, bsm_z, recip_box, f_div_ϵr, α, mesh_dims, boundary,
                     energy_units, ::Val{1}, ::Val{needs_vir}) where {T, needs_vir}
    if needs_vir
        buffer_virial[1] .= zero(T)
    end
    factor = T(π)^2 / α^2
    boxfactor = T(π) * volume(boundary)
    esum = zero(T) * energy_units
    for kx in 0:(mesh_dims[1]-1), ky in 0:(mesh_dims[2]-1), kz in 0:(mesh_dims[3]-1)
        esum_val = recip_conv_inner!(buffer_virial[1], charge_grid, bsm_x, bsm_y, bsm_z, recip_box,
                            mesh_dims, energy_units, f_div_ϵr, factor, boxfactor, kx, ky, kz,
                            Val(needs_vir), Val(false))
        esum += esum_val
    end
    if needs_vir
        vir .+= buffer_virial[1] .* energy_units
    end
    return esum / 2
end

function recip_conv!(vir, buffer_virial, charge_grid::Array{Complex{T}, 3}, buffer,
                     bsm_x, bsm_y, bsm_z, recip_box, f_div_ϵr, α, mesh_dims, boundary, energy_units,
                     ::Val{n_threads}, ::Val{needs_vir}) where {T, n_threads, needs_vir}
    factor = T(π)^2 / α^2
    boxfactor = T(π) * volume(boundary)
    buffer .= zero(T)
    Threads.@threads for chunk_i in 1:n_threads
        if needs_vir
            buffer_virial[chunk_i] .= zero(T)
        end
        for kx in (chunk_i-1):n_threads:(mesh_dims[1]-1)
            for ky in 0:(mesh_dims[2]-1), kz in 0:(mesh_dims[3]-1)
                esum_val = recip_conv_inner!(buffer_virial[chunk_i], charge_grid, bsm_x, bsm_y,
                            bsm_z, recip_box, mesh_dims, energy_units, f_div_ϵr, factor, boxfactor,
                            kx, ky, kz, Val(needs_vir), Val(false))
                buffer[chunk_i] += ustrip(energy_units, esum_val)
            end
        end
    end
    esum = sum(buffer) * energy_units
    if needs_vir
        for chunk_i in 1:n_threads
            vir .+= buffer_virial[chunk_i] .* energy_units
        end
    end
    return esum / 2
end

function recip_conv!(vir, buffer_virial, charge_grid::AbstractArray{Complex{T}, 3}, buffer, bsm_x,
                     bsm_y, bsm_z, recip_box, f_div_ϵr, α, mesh_dims, boundary, energy_units,
                     n_threads_val, ::Val{needs_vir}) where {T, needs_vir}
    if needs_vir
        buffer_virial .= zero(T)
    end
    ndrange = Tuple(mesh_dims)
    factor = T(π)^2 / α^2
    boxfactor = T(π) * volume(boundary)
    backend = get_backend(charge_grid)
    n_threads_gpu = 16
    kernel! = recip_conv_kernel!(backend, n_threads_gpu)
    kernel!(buffer_virial, buffer, charge_grid, bsm_x, bsm_y, bsm_z, recip_box, mesh_dims,
            energy_units, f_div_ϵr, factor, boxfactor, Val(needs_vir); ndrange=ndrange)
    if needs_vir
        vir .+= from_device(buffer_virial) .* energy_units
    end
    return sum(buffer) * energy_units / 2
end

@kernel function recip_conv_kernel!(vir, esum_arr, charge_grid, @Const(bsm_x), @Const(bsm_y),
                                    @Const(bsm_z), recip_box, mesh_dims, energy_units,
                                    f_div_ϵr, factor, boxfactor,
                                    ::Val{needs_vir}) where needs_vir
    kxp1, kyp1, kzp1 = @index(Global, NTuple)
    if kxp1 <= mesh_dims[1] && kyp1 <= mesh_dims[2] && kzp1 <= mesh_dims[3]
        esum = recip_conv_inner!(vir, charge_grid, bsm_x, bsm_y, bsm_z, recip_box, mesh_dims,
                                 energy_units, f_div_ϵr, factor, boxfactor,
                                 kxp1-1, kyp1-1, kzp1-1, Val(needs_vir), Val(true))
        esum_arr[kxp1, kyp1, kzp1] = ustrip(energy_units, esum)
    end
end

function interpolate_force_inner!(Fs, charge_grid, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            ::Val{T}, i) where T
    nx, ny, nz = mesh_dims
    fx, fy, fz = zero(T), zero(T), zero(T)
    @inbounds begin
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
        f = SVector(
            q * (fx*nx*recip_box[1][1]),
            q * (fx*nx*recip_box[2][1] + fy*ny*recip_box[2][2]),
            q * (fx*nx*recip_box[3][1] + fy*ny*recip_box[3][2] + fz*nz*recip_box[3][3]),
        ) * energy_units
        Fs[i] -= f
    end
    return Fs
end

function interpolate_force!(Fs, charge_grid::Array{Complex{T}, 3}, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            n_threads) where T
    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:length(atoms)
            interpolate_force_inner!(Fs, charge_grid, grid_indices, bsplines_θ,
                        bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                        Val(T), i)
        end
    end
    return Fs
end

function interpolate_force!(Fs, charge_grid::AbstractArray{T, 3}, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            n_threads) where T
    backend = get_backend(Fs)
    n_threads_gpu = 128
    kernel! = interpolate_force_kernel!(backend, n_threads_gpu)
    kernel!(Fs, charge_grid, grid_indices, bsplines_θ, bsplines_dθ, recip_box,
            mesh_dims, order, energy_units, atoms, Val(T); ndrange=length(atoms))
    return Fs
end

@kernel function interpolate_force_kernel!(Fs, @Const(charge_grid), @Const(grid_indices),
                        @Const(bsplines_θ), @Const(bsplines_dθ), recip_box, mesh_dims, order,
                        energy_units, @Const(atoms), ::Val{T}) where T
    i = @index(Global, Linear)
    if i <= length(atoms)
        interpolate_force_inner!(Fs, charge_grid, grid_indices, bsplines_θ,
                    bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms, Val(T), i)
    end
end

grad_safe_fft!( charge_grid, fft_plan ) = fft_plan  * charge_grid
grad_safe_bfft!(charge_grid, bfft_plan) = bfft_plan * charge_grid

function ewald_pe_forces!(Fs, vir, inter::PME{T}, atoms, coords, boundary, force_units,
                          energy_units, ::Val{needs_vir}, calculate_forces=true;
                          n_threads::Integer=Threads.nthreads()) where {T, needs_vir}
    n_thr = (inter.grad_safe ? 1 : n_threads) # Enzyme error with multiple threads
    order, ϵr, α, mesh_dims = inter.order, inter.ϵr, inter.α, inter.mesh_dims
    V = volume(boundary)
    f = (energy_units == NoUnits ? ustrip(T(Molly.coulomb_const)) : T(Molly.coulomb_const))

    exclusion_E = excluded_interactions!(Fs, vir, inter.excluded_buffer_Fs, inter.virial_buffer, 
                    inter.excluded_buffer_Es, inter.excluded_pairs, atoms, coords, boundary, α, f,
                    force_units, energy_units, calculate_forces, Val(T), Val(needs_vir))

    recip_box = invert_box_vectors(boundary)
    grid_placement!(inter.grid_indices, inter.grid_fractions, coords, recip_box, mesh_dims)
    update_bsplines!(inter.bsplines_θ, inter.bsplines_dθ, inter.grid_fractions, order, n_thr)
    spread_charge!(inter.charge_grid, inter.charge_grid_buffer, inter.grid_indices,
                   inter.bsplines_θ, mesh_dims, order, atoms, Val(n_thr))
    grad_safe_fft!(inter.charge_grid, inter.fft_plan)
    reciprocal_space_E = recip_conv!(vir, inter.virial_buffer, inter.charge_grid,
                    inter.recip_conv_buffer, inter.bsplines_moduli_x, inter.bsplines_moduli_y,
                    inter.bsplines_moduli_z, recip_box, f / ϵr, α, mesh_dims, boundary,
                    energy_units, Val(n_thr), Val(needs_vir))
    grad_safe_bfft!(inter.charge_grid, inter.bfft_plan)
    if calculate_forces
        interpolate_force!(Fs, inter.charge_grid, inter.grid_indices, inter.bsplines_θ,
                        inter.bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms, 
                        n_thr)
    end

    if isnothing(inter.pc_sum) || inter.grad_safe
        partial_charges = charge.(atoms)
        pc_sum = sum(partial_charges)
        pc_abs2_sum = sum(abs2, partial_charges)
    else
        pc_sum, pc_abs2_sum = inter.pc_sum, inter.pc_abs2_sum
    end
    charge_E = -f * T(π) * pc_sum^2 / (2 * V * α^2)
    self_E = f * -pc_abs2_sum * α / sqrt(T(π)) + charge_E
    total_E = reciprocal_space_E + self_E + exclusion_E
    return total_E
end
