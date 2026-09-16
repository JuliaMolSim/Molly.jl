# Long range electrostatic summation methods
# Based on the OpenMM source code
import Base: ==, hash

export
    Ewald,
    SetupEwald,
    PME,
    SetupPME,
    EwaldExclusion

abstract type AbstractEwald end

const default_ewald_error_tol = 0.0005

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                            inter::AbstractEwald;
                                            n_threads::Integer=Threads.nthreads(),
                                            kwargs...)
    pe = ewald_pe_forces!(nothing, nothing, sys, inter, Val(false), Val(true);
                          n_threads=n_threads)
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
    ewald_pe_forces!(fs, vir, sys, inter, Val(needs_vir), Val(false); n_threads=n_threads)
    return fs
end

function AtomsCalculators.energy_forces!(fs::AbstractVector, # Required for disambiguation
                                         sys::System,
                                         inter::AbstractEwald;
                                         n_threads::Integer=Threads.nthreads(),
                                         kwargs...)
    pe = ewald_pe_forces!(fs, nothing, sys, inter, Val(false), Val(true); n_threads=n_threads)
    return (energy=pe, forces=fs)
end

function AtomsCalculators.energy_forces(sys::System,
                                        inter::AbstractEwald;
                                        n_threads::Integer=Threads.nthreads(),
                                        kwargs...)
    fs = zero_forces(sys)
    pe = ewald_pe_forces!(fs, nothing, sys, inter, Val(false), Val(true); n_threads=n_threads)
    return (energy=pe, forces=fs)
end

@inline function effective_charge(atom::Atom, scheduler, ::Val{T}) where T
    dual_val = scheduler.dual ? Val(true) : Val(false)
    λ, λR, λ_params = scale_elec(scheduler, T(atom.λ), atom.alch_role, dual_val)
    if scheduler.dual
        return λ*atom.charge
    else
        return params_mixing(λ_params, atom.charge)
    end
end


@inline function effective_charge(atom::Atom, scheduler, ::Val{T}, global_λ) where T
    dual_val = scheduler.dual ? Val(true) : Val(false)
    λ, λR, λ_params = scale_elec(scheduler, T(global_λ), atom.alch_role, dual_val)
    if scheduler.dual
        return λ*atom.charge
    else
        return params_mixing(λ_params, atom.charge)
    end
end

"""
    Ewald(dist_cutoff; error_tol=0.0005, scheduler=DefaultLambdaScheduler())

Ewald summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator.

Should be used alongside the [`CoulombEwald`](@ref) pairwise interaction,
which provides the short range term, and the [`EwaldExclusion`](@ref) specific
interaction, which provides the exclusions for bonded atoms.
`dist_cutoff` and `error_tol` should match these interactions.
[`SetupEwald`](@ref) provides Ewald parameters when setting up a system from a file.

`dist_cutoff` is the cutoff distance for short range interactions.
This algorithm is O(N^2) and in general [`PME`](@ref) should be used instead.
Only compatible with 3D systems and [`CubicBoundary`](@ref).
Not compatible with infinite boundaries.
Runs on the CPU, even for GPU systems.
"""
struct Ewald{T, D, SCH} <: AbstractEwald
    dist_cutoff::D
    error_tol::T
    scheduler::SCH
end

function Ewald(dist_cutoff; error_tol=default_ewald_error_tol, scheduler=DefaultLambdaScheduler())
    T = typeof(ustrip(dist_cutoff))
    return Ewald(dist_cutoff, T(error_tol), scheduler)
end

Base.zero(inter::Ewald{T, D}) where {T, D} = Ewald(zero(D), zero(T), inter.scheduler)

function Base.:+(i1::Ewald, i2::Ewald)
    return Ewald(i1.dist_cutoff + i2.dist_cutoff, i1.error_tol + i2.error_tol, i1.scheduler)
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

function ewald_pe_forces!(Fs, vir, sys::System{3, <:Any, <:Any, TH}, inter::AbstractEwald,
                          ::Val{needs_vir}, ::Val{needs_pe}=Val(true);
                          n_threads::Integer=Threads.nthreads()) where {TH, needs_vir, needs_pe}
    calculate_forces = !isnothing(Fs)
    return ewald_pe_forces!(Fs, vir, inter, sys.atoms, sys.coords, sys.boundary, sys.force_units,
                            sys.energy_units, Val(needs_vir), calculate_forces, Val(needs_pe),
                            Val(TH); n_threads=n_threads)
end

@inline sum_float_type(f, ::Type{T}, v::AbstractVector{T}) where {T} = sum(f, v)
@inline sum_float_type(f, ::Type{T}, v) where {T} = sum(f ∘ T, v)

# The Ewald sum shares its loop between the energy and the forces, so `needs_pe` is ignored
function ewald_pe_forces!(Fs, vir, inter::Ewald{T}, atoms, coords, boundary, force_units,
                          energy_units, ::Val{needs_vir}, calculate_forces=true, ::Val=Val(true),
                          ::Val{TH}=Val(Float64);
                          n_threads::Integer=Threads.nthreads()) where {T, TH, needs_vir}
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
    partial_charges_cpu = [effective_charge(atom, inter.scheduler, Val(T)) for atom in atoms_cpu]
    V = volume(boundary)
    f = (energy_units == NoUnits ? ustrip(T(Molly.coulomb_const)) : T(Molly.coulomb_const))
    if AT <: AbstractGPUArray && calculate_forces
        Fs_cpu = zeros(SVector{3, typeof(zero(T) * force_units)}, n_atoms)
    else
        Fs_cpu = Fs
    end

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

    f_h, α_h, V_h = TH(f), TH(α), TH(V)
    pc_sum      = sum_float_type(identity, TH, partial_charges_cpu)
    pc_abs2_sum = sum_float_type(abs2    , TH, partial_charges_cpu)
    charge_E = -f_h * TH(π) * pc_sum^2 / (2 * V_h * α_h^2)
    self_E = -f_h * pc_abs2_sum * α_h / sqrt(TH(π)) + charge_E
    total_E = reciprocal_space_E + self_E

    if needs_vir
        # Since charge_E = -A/V, affine box differentiation gives W = charge_E * I
        vir .+= charge_E .* I(3)
    end

    if calculate_forces && AT <: AbstractGPUArray
        Fs .+= to_device(Fs_cpu, AT)
    end
    return total_E
end

function ==(a::Ewald, b::Ewald)
    return a.dist_cutoff == b.dist_cutoff &&
           a.error_tol   == b.error_tol   &&
           a.scheduler == b.scheduler
end

function hash(a::Ewald, h::UInt)
    v = hash(a.dist_cutoff, h)
    v = hash(a.error_tol, v)
    return hash(a.scheduler, v)
end

abstract type AbstractSetupEwald end

"""
    SetupEwald(; error_tol=0.0005, approximate_erfc=true,
               coulomb_const=138.93545764u"kJ * mol^-1 * nm")

Set up Ewald summation for long range electrostatics.

Passed to the [`System`](@ref) constructor from files, where it creates a [`Ewald`](@ref)
general interaction, a [`CoulombEwald`](@ref) pairwise interaction and a
[`EwaldExclusion`](@ref) specific interaction.

`error_tol` is the error tolerance for Ewald summation.
`approximate_erfc` determines whether to use a fast approximation to the erfc function.
"""
struct SetupEwald{T, C} <: AbstractSetupEwald
    error_tol::T
    approximate_erfc::Bool
    coulomb_const::C
end

function SetupEwald(; error_tol=default_ewald_error_tol, approximate_erfc::Bool=true,
                    coulomb_const=coulomb_const)
    if error_tol <= zero(error_tol)
        throw(ArgumentError("error_tol must be greater than zero, found $error_tol"))
    end
    return SetupEwald(error_tol, approximate_erfc, coulomb_const)
end

function setup_coulomb_pairwise(se::AbstractSetupEwald, dist_cutoff, weight_special,
                                use_neighbors, units, T)
    return CoulombEwald(
        dist_cutoff=T(dist_cutoff),
        error_tol=T(se.error_tol),
        use_neighbors=use_neighbors,
        weight_special=weight_special,
        coulomb_const=convert_setup_quantity(se.coulomb_const, units, T),
        approximate_erfc=se.approximate_erfc,
    )
end

function setup_coulomb_general(se::SetupEwald, atoms, boundary, dist_cutoff, n_threads,
                               grad_safe, units, T)
    return Ewald(T(dist_cutoff); error_tol=T(se.error_tol))
end

"""
    PME(dist_cutoff, atoms, boundary; error_tol=0.0005, order=5,
        ϵr=1.0, fixed_charges=true, mesh_dims=nothing,
        scheduler=DefaultLambdaScheduler(), float_type_high=Float64,
        grad_safe=false, n_threads=Threads.nthreads())

Particle mesh Ewald (PME) summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator.

Should be used alongside the [`CoulombEwald`](@ref) pairwise interaction,
which provides the short range term, and the [`EwaldExclusion`](@ref) specific
interaction, which provides the exclusions for bonded atoms.
`dist_cutoff` and `error_tol` should match these interactions.
[`SetupPME`](@ref) provides PME parameters when setting up a system from a file.

`dist_cutoff` is the cutoff distance for short range interactions.
`fixed_charges` should be set to `false` if the partial charges can change,
for example when using a polarizable force field.
`mesh_dims` gives the number of grid points in each dimension, overriding the
value chosen from `error_tol`.
`grad_safe` should be set to `true` if gradients are going to be calculated
with Enzyme.jl.
`n_threads` is used to pre-allocate memory on CPU and plan the FFTs.

This implementation is based on the implementation in OpenMM, which
is based on the smooth PME algorithm from
[Essmann et al. 1995](https://doi.org/10.1063/1.470117).

Only compatible with 3D systems.
Not compatible with infinite boundaries.
"""
struct PME{T, D, A, I, M, BM, C, RG, CB, RB, VB, PB, P, F, B, SCH} <: AbstractEwald
    dist_cutoff::D
    error_tol::T
    order::Int
    ϵr::T
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
    recip_grid::RG
    charge_grid_buffer::CB
    recip_conv_buffer::RB
    virial_buffer::VB
    partial_charge_buffer::PB
    pc_sum::P
    pc_abs2_sum::P
    fft_plan::F
    bfft_plan::B
    scheduler::SCH
    grad_safe::Bool
end

# The charge is spread into one grid per thread and the grids are then summed, so past a
# certain number of threads the sum costs more than the extra parallelism is worth
n_spread_threads(n_threads) = min(n_threads, 16)

function pme_bspline_moduli(::Type{T}, order, mesh_dims) where {T}
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

    return bsplines_moduli
end

function find_excluded_pairs(eligible, special)
    excluded_pairs = Tuple{Int32, Int32}[]
    if !(isnothing(eligible) && isnothing(special))
        n_atoms = (isnothing(eligible) ? size(special, 1) : size(eligible, 1))
        eligible_cpu = (isnothing(eligible) ? nothing : to_bitmatrix(from_device(eligible)))
        special_cpu  = (isnothing(special ) ? nothing : to_bitmatrix(from_device(special )))
        # Only a small fraction of the n_atoms^2 entries are excluded, so scan the mask
        #   64 entries at a time and skip the chunks with nothing set
        n_entries = n_atoms * n_atoms
        n_chunks = cld(n_entries, 64)
        eligible_chunks = (isnothing(eligible_cpu) ? nothing : eligible_cpu.chunks)
        special_chunks  = (isnothing(special_cpu ) ? nothing : special_cpu.chunks )
        # Bits past the end of the last chunk are unset in a BitArray but are set by the
        #   negation below, so mask them off
        end_mask = ~zero(UInt64) >>> ((-n_entries) & 63)
        for ci in 1:n_chunks
            # A missing eligible matrix means every pair is eligible, a missing special
            #   matrix means no pair is special, so neither excludes anything
            chunk = zero(UInt64)
            if !isnothing(eligible_chunks)
                chunk = ~eligible_chunks[ci]
            end
            if !isnothing(special_chunks)
                chunk |= special_chunks[ci]
            end
            if ci == n_chunks
                chunk &= end_mask
            end
            while !iszero(chunk)
                # Column-major linear index of the set bit, zero-based
                li = (ci - 1) * 64 + trailing_zeros(chunk)
                j, i = divrem(li, n_atoms)
                if i < j
                    push!(excluded_pairs, (Int32(i + 1), Int32(j + 1)))
                end
                chunk &= chunk - one(UInt64)
            end
        end
        # The scan runs down the columns, sort to give the same order as looping over
        #   i and then j
        sort!(excluded_pairs)
    end
    return excluded_pairs
end

function PME(dist_cutoff, atoms, boundary; error_tol=default_ewald_error_tol, order=5,
             ϵr=1.0, fixed_charges=true, mesh_dims=nothing, eligible=nothing, special=nothing,
             scheduler=DefaultLambdaScheduler(), float_type_high=Float64, grad_safe=false,
             n_threads::Integer=Threads.nthreads())
    T = typeof(ustrip(dist_cutoff))
    TH = float_type_high
    AT = array_type(atoms)
    n_atoms = length(atoms)
    error_tol_T = T(error_tol)
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol_T))
    if isnothing(mesh_dims)
        mesh_dims = pme_params.(box_sides(boundary), α, error_tol_T)
    else
        if length(mesh_dims) != 3
            throw(ArgumentError("mesh_dims should have 3 entries, one for each dimension, " *
                                "found $mesh_dims"))
        end
        mesh_dims = SVector{3, Int}(mesh_dims)
        if any(<(order), mesh_dims)
            throw(ArgumentError("every entry of mesh_dims should be at least the B-spline " *
                                "order ($order), found $(Tuple(mesh_dims))"))
        end
    end
    # The three B-spline dimensions are flattened into one axis to keep these 2D. The atom
    # index goes last on CPU, so that the values belonging to an atom share a cache line,
    # and first on GPU, so that neighbouring threads touch neighbouring elements. See
    # `atom_last`, which hides the difference from the code that uses them.
    if AT <: AbstractGPUArray
        grid_indices = to_device(zeros(Int, n_atoms, 3), AT)
        grid_fractions = to_device(zeros(T, n_atoms, 3), AT)
        bsplines_θ = to_device(zeros(T, n_atoms, order * 3), AT)
    else
        grid_indices = zeros(Int, 3, n_atoms)
        grid_fractions = zeros(T, 3, n_atoms)
        bsplines_θ = zeros(T, order * 3, n_atoms)
    end
    bsplines_dθ = zero(bsplines_θ)
    # Ordered z/y/x for better memory access. The charge grid is real, so a real to complex
    # transform is used, which halves the work of the FFTs and of everything else that
    # touches the reciprocal space grid.
    charge_grid = to_device(zeros(T, mesh_dims[3], mesh_dims[2], mesh_dims[1]), AT)
    recip_grid = to_device(zeros(Complex{T}, mesh_dims[3] ÷ 2 + 1, mesh_dims[2], mesh_dims[1]),
                           AT)
    excluded_pairs = to_device(find_excluded_pairs(eligible, special), AT)

    bsplines_moduli = pme_bspline_moduli(T, order, mesh_dims)

    if AT <: AbstractGPUArray
        # The charge is added to the real grid atomically, so no per-thread grid is needed
        charge_grid_buffer = nothing
        recip_conv_buffer  = to_device(zeros(T, size(recip_grid)), AT)
        virial_buffer      = to_device(zeros(T, 3, 3), AT)
        partial_charge_buffer = to_device(zeros(T, n_atoms), AT)
    elseif n_threads > 1
        charge_grid_buffer = [zeros(T, size(charge_grid)) for _ in 1:n_spread_threads(n_threads)]
        recip_conv_buffer = zeros(T, n_threads)
        virial_buffer = [zeros(T, 3, 3) for _ in 1:n_threads]
        partial_charge_buffer = zeros(T, n_atoms)
    else
        charge_grid_buffer = nothing
        recip_conv_buffer = zeros(T, 1)
        virial_buffer = [zeros(T, 3, 3)]
        partial_charge_buffer = zeros(T, n_atoms)
    end

    fixed_charges = false
    if fixed_charges && !grad_safe
        atoms_cpu = from_device(atoms)
        partial_charges = effective_charge.(atoms_cpu, Ref(scheduler), Val(T))
        pc_sum = sum(TH, partial_charges)
        pc_abs2_sum = sum(abs2 ∘ TH, partial_charges)
    else
        pc_sum, pc_abs2_sum = nothing, nothing
    end

    if AT <: AbstractGPUArray
        fft_plan  = plan_rfft(charge_grid)
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3])
    else
        fft_plan  = plan_rfft( charge_grid, 1:3; flags=FFTW.MEASURE, num_threads=n_threads)
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3], 1:3; flags=FFTW.MEASURE,
                               num_threads=n_threads)
    end
    charge_grid .= zero(T) # Can be overwritten by FFTW.MEASURE
    recip_grid .= zero(Complex{T})

    bsm_x = to_device(bsplines_moduli[1], AT)
    bsm_y = to_device(bsplines_moduli[2], AT)
    bsm_z = to_device(bsplines_moduli[3], AT)

    return PME(dist_cutoff, error_tol_T, order, T(ϵr), α, mesh_dims, grid_indices, grid_fractions,
               bsplines_θ, bsplines_dθ, bsm_x, bsm_y, bsm_z, charge_grid, recip_grid, charge_grid_buffer,
               recip_conv_buffer, virial_buffer, partial_charge_buffer, pc_sum, pc_abs2_sum, fft_plan, bfft_plan,
               scheduler, grad_safe)
end

Base.deepcopy_internal(pme::PME, dict::IdDict) = deepcopy_registered(pme, dict)

function Base.deepcopy(pme::PME)
    # 1. Deepcopy standard immutable parameters and CPU buffers
    dist_cutoff    = pme.dist_cutoff
    error_tol      = pme.error_tol
    order          = pme.order
    ϵr             = pme.ϵr
    α              = pme.α
    mesh_dims      = pme.mesh_dims
    pc_sum         = deepcopy(pme.pc_sum)
    pc_abs2_sum    = deepcopy(pme.pc_abs2_sum)
    scheduler      = pme.scheduler
    grad_safe      = pme.grad_safe

    # 2. Duplicate GPU/CPU Arrays cleanly via deepcopy or copy
    grid_indices        = deepcopy(pme.grid_indices)
    grid_fractions      = deepcopy(pme.grid_fractions)
    bsplines_θ          = deepcopy(pme.bsplines_θ)
    bsplines_dθ         = deepcopy(pme.bsplines_dθ)
    bsplines_moduli_x   = deepcopy(pme.bsplines_moduli_x)
    bsplines_moduli_y   = deepcopy(pme.bsplines_moduli_y)
    bsplines_moduli_z   = deepcopy(pme.bsplines_moduli_z)
    partial_charge_buffer = deepcopy(pme.partial_charge_buffer)

    # Critical allocations: Main charge grid and working buffers
    charge_grid         = deepcopy(pme.charge_grid)
    charge_grid_buffer  = deepcopy(pme.charge_grid_buffer)
    recip_grid          = deepcopy(pme.recip_grid)
    recip_conv_buffer   = deepcopy(pme.recip_conv_buffer)
    virial_buffer       = deepcopy(pme.virial_buffer)

    # 3. CRITICAL FIX: Re-plan the FFTs on the NEW charge_grid buffer
    # Do NOT copy the old fft_plan; it contains shared handles/streams.
    AT = array_type(charge_grid)
    if AT <: AbstractGPUArray
        fft_plan  = plan_rfft(charge_grid)
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3])
    else
        fft_plan  = plan_rfft( charge_grid, 1:3; flags=FFTW.MEASURE, num_threads=Threads.nthreads())
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3], 1:3; flags=FFTW.MEASURE,
                               num_threads=Threads.nthreads())
    end

    # 4. Return the brand new, isolated PME struct
    return PME(
        dist_cutoff, error_tol, order, ϵr, α, mesh_dims,
        grid_indices, grid_fractions, bsplines_θ, bsplines_dθ, 
        bsplines_moduli_x, bsplines_moduli_y, bsplines_moduli_z,
        charge_grid, recip_grid, charge_grid_buffer,
        recip_conv_buffer, virial_buffer, partial_charge_buffer, pc_sum, pc_abs2_sum, 
        fft_plan, bfft_plan, scheduler, grad_safe
    )
end

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
        zero(pme.recip_grid),
        charge_grid_buffer,
        zero_or_nothing(pme.recip_conv_buffer),
        zero_or_nothing(pme.virial_buffer),
        zero_or_nothing(pme.partial_charge_buffer),
        zero_or_nothing(pme.pc_sum),
        zero_or_nothing(pme.pc_abs2_sum),
        pme.fft_plan,
        pme.bfft_plan,
        pme.scheduler,
        pme.grad_safe,
    )
end

function ==(a::PME, b::PME)
    return a.dist_cutoff    == b.dist_cutoff    &&
           a.error_tol      == b.error_tol      &&
           a.order          == b.order          &&
           a.ϵr             == b.ϵr             &&
           a.α              == b.α              &&
           a.mesh_dims      == b.mesh_dims      &&
           a.scheduler      == b.scheduler      &&
           a.grad_safe      == b.grad_safe
end

function hash(a::PME, h::UInt)
    v = hash(a.dist_cutoff, h)
    v = hash(a.error_tol, v)
    v = hash(a.order, v)
    v = hash(a.ϵr, v)
    v = hash(a.α, v)
    v = hash(a.mesh_dims, v)
    v = hash(a.scheduler, v)
    v = hash(a.grad_safe, v)
    return v
end

# Round up to the next size whose prime factors are all below 8
# FFT libraries only have fast paths for these sizes
function legal_fft_dim(minimum_size)
    n = minimum_size
    while true
        unfactored = n
        for factor in 2:7
            while unfactored > 1 && iszero(unfactored % factor)
                unfactored ÷= factor
            end
        end
        isone(unfactored) && return n
        n += 1
    end
end

function pme_params(side_length, α, error_tol::T) where T
    s = ceil(Int, 2α * side_length / (3 * error_tol^T(0.2)))
    return legal_fft_dim(max(s, 6))
end

@inline function grid_placement_inner!(grid_indices, grid_fractions, coords, recip_box,
                                       mesh_dims, i)
    @inbounds for d in 1:3
        t = sum(coords[i] .* SVector(recip_box[1][d], recip_box[2][d], recip_box[3][d]))
        t = (t - floor(t)) * mesh_dims[d]
        ti = floor(Int, t)
        grid_fractions[d, i] = t - ti
        # `t` is below the mesh length, so rounding is the only way `ti` can reach it and
        # the wrap can be done with a subtraction rather than an integer division
        grid_indices[d, i] = wrap_grid_index(ti, mesh_dims[d])
    end
    return grid_indices, grid_fractions
end

function grid_placement!(grid_indices::Matrix, grid_fractions, coords, recip_box, mesh_dims,
                         n_threads)
    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:length(coords)
            grid_placement_inner!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, i)
        end
    end
    return grid_indices, grid_fractions
end

function grid_placement!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, n_threads)
    # Allow Enzyme rule
    grid_placement_gpu!(grid_indices, grid_fractions, coords, recip_box, mesh_dims)
    return grid_indices, grid_fractions
end

function grid_placement_gpu!(grid_indices, grid_fractions, coords, recip_box, mesh_dims)
    backend = get_backend(parent(grid_indices))
    n_threads_gpu = 128
    kernel! = grid_placement_kernel!(backend, n_threads_gpu)
    kernel!(grid_indices, grid_fractions, coords, recip_box, mesh_dims; ndrange=length(coords))
    return nothing
end

@kernel function grid_placement_kernel!(grid_indices, grid_fractions, @Const(coords),
                                        recip_box, mesh_dims)
    i = @index(Global, Linear)
    if i <= length(coords)
        grid_placement_inner!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, i)
    end
end

@inline function update_bsplines_inner!(bsplines_θ::AbstractArray{T, 2}, bsplines_dθ,
                                        grid_fractions, order, i) where T
    @inbounds for j in 1:3
        o = (j - 1) * order
        dr = grid_fractions[j, i]
        bsplines_θ[o + order, i] = zero(T)
        bsplines_θ[o + 2, i]     = dr
        bsplines_θ[o + 1, i]     = 1 - dr
        for k in 3:(order-1)
            d = inv(k - one(T))
            bsplines_θ[o + k, i] = d * dr * bsplines_θ[o + k - 1, i]
            for l in 1:(k-2)
                bsplines_θ[o + k - l, i] = d * (
                        (dr + l) * bsplines_θ[o + k - l - 1, i] +
                        (k - l - dr) * bsplines_θ[o + k - l, i]
                    )
            end
            bsplines_θ[o + 1, i] *= d * (1 - dr)
        end

        bsplines_dθ[o + 1, i] = -bsplines_θ[o + 1, i]
        for k in 1:(order-1)
            bsplines_dθ[o + k + 1, i] = bsplines_θ[o + k, i] - bsplines_θ[o + k + 1, i]
        end
        d = inv(order - one(T))
        bsplines_θ[o + order, i] = d * dr * bsplines_θ[o + order - 1, i]
        for l in 1:(order-2)
            bsplines_θ[o + order - l, i] = d * (
                    (dr + l) * bsplines_θ[o + order - l - 1, i] +
                    (order - l - dr) * bsplines_θ[o + order - l, i]
                )
        end
        bsplines_θ[o + 1, i] *= d * (1 - dr)
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
    # Allow Enzyme rule
    update_bsplines_gpu!(bsplines_θ, bsplines_dθ, grid_fractions, order)
    return bsplines_θ, bsplines_dθ
end

function update_bsplines_gpu!(bsplines_θ, bsplines_dθ, grid_fractions, order)
    n_atoms = size(grid_fractions, 2)
    backend = get_backend(parent(bsplines_θ))
    n_threads_gpu = 128
    kernel! = update_bsplines_kernel!(backend, n_threads_gpu)
    kernel!(bsplines_θ, bsplines_dθ, grid_fractions, order; ndrange=n_atoms)
    return nothing
end

@kernel function update_bsplines_kernel!(bsplines_θ, bsplines_dθ, @Const(grid_fractions),
                                         order)
    i = @index(Global, Linear)
    n_atoms = size(grid_fractions, 2)
    if i <= n_atoms
        update_bsplines_inner!(bsplines_θ, bsplines_dθ, grid_fractions, order, i)
    end
end

# CPU case, each thread has its own grid so the addition does not have to be atomic
@inline function add_charge_grid!(charge_grid, li, cb, ::Val{false})
    @inbounds charge_grid[li] += cb
    return charge_grid
end

# GPU case, where all the threads share one grid
@inline function add_charge_grid!(charge_grid, li, cb, ::Val{true})
    @inbounds Atomix.@atomic charge_grid[li] += cb
    return charge_grid
end

# The per-atom B-spline and grid index arrays are stored atom index last on CPU and atom
# index first on GPU, see the `PME` constructor. Transposing the GPU arrays lets both be
# indexed as [value, atom] everywhere else.
@inline atom_last(A::Matrix) = A
@inline atom_last(A) = transpose(A)

# A grid index is only ever one mesh length past the end, so it can be wrapped with a
# subtraction rather than a modulo, and the strides of the outer loops hoisted out
@inline function wrap_grid_index(index, mesh_dim)
    return index - ifelse(index >= mesh_dim, mesh_dim, zero(index))
end

@inline function spread_charge_inner!(charge_grid, grid_indices, bsplines_θ,
                              mesh_dims, order, atoms, scheduler, i, ::Val{T},
                              ::Val{atomic}) where {T, atomic}
    q = effective_charge(atoms[i], scheduler, Val(T))
    nx, ny, nz = mesh_dims[1], mesh_dims[2], mesh_dims[3]
    @inbounds x0index, y0index, z0index = grid_indices[1, i], grid_indices[2, i], grid_indices[3, i]
    @inbounds for ix in 0:(order-1)
        xbase = wrap_grid_index(x0index + ix, nx) * ny * nz
        θx = bsplines_θ[ix+1, i]
        qx = q * θx
        for iy in 0:(order-1)
            ybase = xbase + wrap_grid_index(y0index + iy, ny) * nz
            θy = bsplines_θ[order+iy+1, i]
            qxy = qx * θy
            for iz in 0:(order-1)
                zindex = wrap_grid_index(z0index + iz, nz)
                θz = bsplines_θ[2*order+iz+1, i]
                cb = qxy * θz
                add_charge_grid!(charge_grid, ybase + zindex + 1, cb, Val(atomic))
            end
        end
    end
    return charge_grid
end

# GPU version, one thread per (atom, z slice) pair. `order` threads cooperate on each
# atom, which gives `order` times the parallelism of one thread per atom and makes the
# threads of an atom write neighbouring grid points, as in the OpenMM implementation
@inline function spread_charge_slice!(charge_grid, grid_indices, bsplines_θ, mesh_dims, order,
                                      atoms, scheduler, i, iz, ::Val{T}) where T
    q = effective_charge(atoms[i], scheduler, Val(T))
    nx, ny, nz = mesh_dims[1], mesh_dims[2], mesh_dims[3]
    @inbounds begin
        x0index, y0index, z0index = grid_indices[1, i], grid_indices[2, i], grid_indices[3, i]
        zindex = wrap_grid_index(z0index + iz, nz)
        qz = q * bsplines_θ[2*order+iz+1, i]
        for ix in 0:(order-1)
            xbase = wrap_grid_index(x0index + ix, nx) * ny * nz
            qzx = qz * bsplines_θ[ix+1, i]
            for iy in 0:(order-1)
                ybase = xbase + wrap_grid_index(y0index + iy, ny) * nz
                cb = qzx * bsplines_θ[order+iy+1, i]
                add_charge_grid!(charge_grid, ybase + zindex + 1, cb, Val(true))
            end
        end
    end
    return charge_grid
end

function spread_charge!(charge_grid::Array{T, 3}, buffer, grid_indices, bsplines_θ,
                        mesh_dims, order, atoms, scheduler, n_threads) where T
    if n_threads == 1
        charge_grid .= zero(T)
        for i in eachindex(atoms)
            spread_charge_inner!(charge_grid, grid_indices, bsplines_θ, mesh_dims,
                                 order, atoms, scheduler, i, Val(T), Val(false))
        end
        return charge_grid
    end
    Threads.@threads for chunk_i in 1:n_threads
        buffer[chunk_i] .= zero(T)
        for i in chunk_i:n_threads:length(atoms)
            spread_charge_inner!(buffer[chunk_i], grid_indices, bsplines_θ,
                                 mesh_dims, order, atoms, scheduler, i, Val(T), Val(false))
        end
    end
    return reduce_charge_grids!(charge_grid, buffer, Val(n_threads))
end

# Sum the per-thread grids in one parallel pass, as reduce_force_chunks! does for the
# forces, rather than one serial pass over the whole grid per thread. The number of grids
# is a type parameter so that the inner sum can be unrolled.
function reduce_charge_grids!(charge_grid::Array{T, 3}, buffer,
                              ::Val{n_threads}) where {T, n_threads}
    @inbounds Threads.@threads for li in eachindex(charge_grid)
        c = zero(T)
        for chunk_i in 1:n_threads
            c += buffer[chunk_i][li]
        end
        charge_grid[li] = c
    end
    return charge_grid
end

function spread_charge!(charge_grid::AbstractArray{T, 3}, buffer, grid_indices,
                        bsplines_θ, mesh_dims, order, atoms, scheduler, n_threads) where T
    # Allow Enzyme rule
    spread_charge_gpu!(charge_grid, grid_indices, bsplines_θ, mesh_dims, order, atoms,
                       scheduler, Val(T))
    return charge_grid
end

function spread_charge_gpu!(charge_grid, grid_indices, bsplines_θ, mesh_dims, order, atoms,
                            scheduler, ::Val{T}) where T
    backend = get_backend(charge_grid)
    n_threads_gpu = 128
    kernel! = spread_charge_kernel!(backend, n_threads_gpu)
    charge_grid .= zero(T)
    kernel!(charge_grid, grid_indices, bsplines_θ, mesh_dims, order, atoms, scheduler, Val(T);
            ndrange=length(atoms)*order)
    return nothing
end

@kernel function spread_charge_kernel!(charge_grid_real, @Const(grid_indices), @Const(bsplines_θ),
                                       mesh_dims, order, atoms, scheduler, ::Val{T}) where T
    ti = @index(Global, Linear)
    if ti <= length(atoms)*order
        i, iz1 = fldmod1(ti, order)
        spread_charge_slice!(charge_grid_real, grid_indices, bsplines_θ, mesh_dims, order, atoms,
                             scheduler, i, iz1-1, Val(T))
    end
end

@inline function recip_conv_terms(bsm_x, bsm_y, bsm_z, recip_box, mesh_dims, energy_units,
                                  f_div_ϵr, factor, boxfactor, kx, ky, kz, ::Val{T}) where {T}
    nx, ny, nz = mesh_dims
    maxkx, maxky, maxkz = T(0.5)*(nx+1), T(0.5)*(ny+1), T(0.5)*(nz+1)
    # The real to complex transform only keeps the modes with kz up to nz/2, and each of
    # them stands for both k and -k of the full mesh apart from the two, or one when nz is
    # odd, that are their own conjugate
    weight = (iszero(kz) || 2*kz == nz ? one(T) : T(2))
    @inbounds begin
        mx = (kx < maxkx ? kx : kx - nx)
        mhx = mx * recip_box[1][1]
        bx = boxfactor * bsm_x[kx+1]
        my = (ky < maxky ? ky : ky - ny)
        mhy = mx * recip_box[2][1] + my * recip_box[2][2]
        by = bsm_y[ky+1]
        mz = (kz < maxkz ? kz : kz - nz)
        mhz = mx * recip_box[3][1] + my * recip_box[3][2] + mz * recip_box[3][3]
        m2 = mhx^2 + mhy^2 + mhz^2
        bz = bsm_z[kz+1]
        denom = m2 * bx * by * bz
        c  = exp(-factor * m2)
        eterm = f_div_ϵr * c / denom
        eterm_nou = ustrip(energy_units, eterm)
    end
    return weight, eterm, eterm_nou, mhx, mhy, mhz, m2
end

@inline function recip_conv_inner!(vir_nou, recip_grid::AbstractArray{Complex{T}, 3}, bsm_x,
                           bsm_y, bsm_z, recip_box, mesh_dims, energy_units, f_div_ϵr, factor,
                           boxfactor, kx, ky, kz, ::Val{needs_vir},
                           ::Val{atomic}) where {T, needs_vir, atomic}
    if iszero(kx) && iszero(ky) && iszero(kz)
        return zero(T) * energy_units
    end
    weight, eterm, eterm_nou, mhx, mhy, mhz, m2 = recip_conv_terms(bsm_x, bsm_y, bsm_z,
                recip_box, mesh_dims, energy_units, f_div_ϵr, factor, boxfactor,
                kx, ky, kz, Val(T))
    @inbounds begin
        d1, d2 = reim(recip_grid[kz+1, ky+1, kx+1])
        recip_grid[kz+1, ky+1, kx+1] = Complex(d1*eterm_nou, d2*eterm_nou)
        struct2 = weight * (d1^2 + d2^2)

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

function recip_conv!(vir, buffer_virial, recip_grid::Array{Complex{T}, 3}, buffer,
                     bsm_x, bsm_y, bsm_z, recip_box, f_div_ϵr, α, mesh_dims, boundary,
                     energy_units, n_threads, ::Val{needs_vir},
                     ::Val{needs_pe}=Val(true)) where {T, needs_vir, needs_pe}
    factor = T(π)^2 / α^2
    boxfactor = T(π) * volume(boundary)
    n_columns = mesh_dims[1] * mesh_dims[2]
    nzh = size(recip_grid, 1)
    # The threads take whole (kx, ky) columns, of which there are many more than there are
    # threads, so they get an even share of the mesh whatever its dimensions are
    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        if needs_vir
            buffer_virial[chunk_i] .= zero(T)
        end
        # The energy is summed into a local variable rather than into `buffer`, where the
        # threads would be writing to the same cache line on every grid point
        esum = zero(T)
        for column in chunk_i:n_threads:n_columns
            kx, ky = fldmod(column - 1, mesh_dims[2])
            for kz in 0:(nzh-1)
                esum_val = recip_conv_inner!(buffer_virial[chunk_i], recip_grid, bsm_x, bsm_y,
                            bsm_z, recip_box, mesh_dims, energy_units, f_div_ϵr, factor, boxfactor,
                            kx, ky, kz, Val(needs_vir), Val(false))
                if needs_pe
                    esum += ustrip(energy_units, esum_val)
                end
            end
        end
        buffer[chunk_i] = esum
    end
    if needs_vir
        for chunk_i in 1:n_threads
            # The mesh sums both k and -k, so the virial needs the same 1/2 as the energy.
            vir .+= buffer_virial[chunk_i] .* energy_units / 2
        end
    end
    needs_pe || return zero(T) * energy_units
    # `buffer` is sized for the threads the PME was created with, which is not necessarily
    # how many are in use here, so only the entries written above are summed
    return sum(@view buffer[1:n_threads]) * energy_units / 2
end

function recip_conv!(vir, buffer_virial, recip_grid::AbstractArray{Complex{T}, 3}, buffer, bsm_x,
                     bsm_y, bsm_z, recip_box, f_div_ϵr, α, mesh_dims, boundary, energy_units,
                     n_threads, ::Val{needs_vir}, ::Val{needs_pe}) where {T, needs_vir, needs_pe}
    if needs_vir
        buffer_virial .= zero(T)
    end
    factor = T(π)^2 / α^2
    boxfactor = T(π) * volume(boundary)
    recip_conv_gpu!(buffer_virial, buffer, recip_grid, bsm_x, bsm_y, bsm_z, recip_box,
                    mesh_dims, energy_units, f_div_ϵr, factor, boxfactor, Val(needs_vir),
                    Val(needs_pe))
    if needs_vir
        # The mesh sums both k and -k, so the virial needs the same 1/2 as the energy.
        vir .+= from_device(buffer_virial) .* energy_units / 2
    end
    # The energy is discarded when only forces are wanted, in which case the reduction
    # over the whole mesh, and the device synchronisation it forces, can be skipped
    needs_pe || return zero(T) * energy_units
    return sum(buffer) * energy_units / 2
end

function recip_conv_gpu!(buffer_virial, buffer, recip_grid, bsm_x, bsm_y, bsm_z, recip_box,
                         mesh_dims, energy_units, f_div_ϵr, factor, boxfactor,
                         ::Val{needs_vir}, ::Val{needs_pe}) where {needs_vir, needs_pe}
    backend = get_backend(recip_grid)
    n_threads_gpu = 256
    kernel! = recip_conv_kernel!(backend, n_threads_gpu)
    kernel!(buffer_virial, buffer, recip_grid, bsm_x, bsm_y, bsm_z, recip_box, mesh_dims,
            energy_units, f_div_ϵr, factor, boxfactor, Val(needs_vir), Val(needs_pe);
            ndrange=length(recip_grid))
    return nothing
end

# One thread per grid point, indexed so that neighbouring threads touch neighbouring grid
# points. `recip_grid` is stored z fastest, so z has to be the fastest varying index of
# the launch as well.
@kernel function recip_conv_kernel!(vir, esum_arr, recip_grid, @Const(bsm_x), @Const(bsm_y),
                                    @Const(bsm_z), recip_box, mesh_dims, energy_units,
                                    f_div_ϵr, factor, boxfactor, ::Val{needs_vir},
                                    ::Val{needs_pe}) where {needs_vir, needs_pe}
    li = @index(Global, Linear)
    if li <= length(recip_grid)
        nzh = size(recip_grid, 1)
        i0 = li - 1
        kz, r = i0 % nzh, i0 ÷ nzh
        ky, kx = r % mesh_dims[2], r ÷ mesh_dims[2]
        esum = recip_conv_inner!(vir, recip_grid, bsm_x, bsm_y, bsm_z, recip_box, mesh_dims,
                                 energy_units, f_div_ϵr, factor, boxfactor,
                                 kx, ky, kz, Val(needs_vir), Val(true))
        if needs_pe
            @inbounds esum_arr[li] = ustrip(energy_units, esum)
        end
    end
end

@inline function interpolate_force_inner!(Fs, charge_grid, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            scheduler, ::Val{T}, i) where T
    nx, ny, nz = mesh_dims
    fx, fy, fz = zero(T), zero(T), zero(T)
    @inbounds begin
        q = effective_charge(atoms[i], scheduler, Val(T))
        x0index, y0index, z0index = grid_indices[1, i], grid_indices[2, i], grid_indices[3, i]
        for ix in 0:(order-1)
            xbase = wrap_grid_index(x0index + ix, nx) * ny * nz
            tx, dtx = bsplines_θ[ix+1, i], bsplines_dθ[ix+1, i]
            for iy in 0:(order-1)
                ybase = xbase + wrap_grid_index(y0index + iy, ny) * nz
                ty, dty = bsplines_θ[order+iy+1, i], bsplines_dθ[order+iy+1, i]
                dtx_ty = dtx * ty
                tx_dty = tx * dty
                txy = tx * ty
                for iz in 0:(order-1)
                    zindex = wrap_grid_index(z0index + iz, nz)
                    tz, dtz = bsplines_θ[2*order+iz+1, i], bsplines_dθ[2*order+iz+1, i]
                    gridvalue = charge_grid[ybase + zindex + 1]
                    fx += dtx_ty * tz * gridvalue
                    fy += tx_dty * tz * gridvalue
                    fz += txy * dtz * gridvalue
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

function interpolate_force!(Fs, charge_grid::Array{T, 3}, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            scheduler, n_threads) where T
    @maybe_threads (n_threads > 1) for chunk_i in 1:n_threads
        for i in chunk_i:n_threads:length(atoms)
            interpolate_force_inner!(Fs, charge_grid, grid_indices, bsplines_θ,
                        bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                        scheduler, Val(T), i)
        end
    end
    return Fs
end

# GPU version, one thread per (atom, z slice) pair as for the charge spreading. Each of
# the `order` threads of an atom accumulates a partial force over its own z slice and
# adds it atomically, which gives `order` times the parallelism of one thread per atom
# and makes the threads of an atom read neighbouring grid points.
@inline function interpolate_force_slice!(Fs_flat, charge_grid, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, unit_scale, atoms,
                            scheduler, ::Val{T}, i, iz) where T
    nx, ny, nz = mesh_dims
    fx, fy, fz = zero(T), zero(T), zero(T)
    @inbounds begin
        q = effective_charge(atoms[i], scheduler, Val(T))
        x0index, y0index, z0index = grid_indices[1, i], grid_indices[2, i], grid_indices[3, i]
        zindex = wrap_grid_index(z0index + iz, nz)
        tz, dtz = bsplines_θ[2*order+iz+1, i], bsplines_dθ[2*order+iz+1, i]
        for ix in 0:(order-1)
            xbase = wrap_grid_index(x0index + ix, nx) * ny * nz
            tx, dtx = bsplines_θ[ix+1, i], bsplines_dθ[ix+1, i]
            for iy in 0:(order-1)
                ybase = xbase + wrap_grid_index(y0index + iy, ny) * nz
                ty, dty = bsplines_θ[order+iy+1, i], bsplines_dθ[order+iy+1, i]
                gridvalue = charge_grid[ybase + zindex + 1]
                fx += dtx * ty * tz * gridvalue
                fy += tx * dty * tz * gridvalue
                fz += tx * ty * dtz * gridvalue
            end
        end
        # `Fs_flat` reinterprets the force vectors as raw numbers in the force units, since
        # atomics do not work on the unitful static vectors. `recip_box` is stripped of its
        # units on the host for the same reason, with `unit_scale` putting them back.
        f1 = q * (fx*nx*recip_box[1][1])
        f2 = q * (fx*nx*recip_box[2][1] + fy*ny*recip_box[2][2])
        f3 = q * (fx*nx*recip_box[3][1] + fy*ny*recip_box[3][2] + fz*nz*recip_box[3][3])
        Atomix.@atomic Fs_flat[3*(i-1)+1] -= unit_scale * f1
        Atomix.@atomic Fs_flat[3*(i-1)+2] -= unit_scale * f2
        Atomix.@atomic Fs_flat[3*(i-1)+3] -= unit_scale * f3
    end
    return Fs_flat
end

function interpolate_force!(Fs, charge_grid::AbstractArray{T, 3}, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            scheduler, n_threads) where T
    backend = get_backend(Fs)
    n_threads_gpu = 128
    force_units = unit(zero(eltype(eltype(Fs))))
    recip_box_nou = map(v -> ustrip.(v), recip_box)
    unit_scale = T(ustrip(force_units,
                          oneunit(T) * unit(eltype(eltype(recip_box))) * energy_units))
    interpolate_force_gpu!(Fs, charge_grid, grid_indices, bsplines_θ, bsplines_dθ,
                           recip_box_nou, mesh_dims, order, unit_scale, atoms, scheduler,
                           Val(T))
    return Fs
end

function interpolate_force_gpu!(Fs, charge_grid, grid_indices, bsplines_θ, bsplines_dθ,
                                recip_box, mesh_dims, order, unit_scale, atoms, scheduler,
                                ::Val{T}) where T
    backend = get_backend(Fs)
    n_threads_gpu = 128
    Fs_flat = reinterpret(T, Fs)
    kernel! = interpolate_force_kernel!(backend, n_threads_gpu)
    kernel!(Fs_flat, charge_grid, grid_indices, bsplines_θ, bsplines_dθ, recip_box,
            mesh_dims, order, unit_scale, atoms, scheduler, Val(T);
            ndrange=length(atoms)*order)
    return nothing
end

@kernel function interpolate_force_kernel!(Fs_flat, @Const(charge_grid), @Const(grid_indices),
                        @Const(bsplines_θ), @Const(bsplines_dθ), recip_box, mesh_dims, order,
                        unit_scale, @Const(atoms), scheduler, ::Val{T}) where T
    ti = @index(Global, Linear)
    if ti <= length(atoms)*order
        i, iz1 = fldmod1(ti, order)
        interpolate_force_slice!(Fs_flat, charge_grid, grid_indices, bsplines_θ,
                    bsplines_dθ, recip_box, mesh_dims, order, unit_scale, atoms, scheduler,
                    Val(T), i, iz1-1)
    end
end

# Enzyme rules defined in extension
grad_safe_fft!( charge_grid, recip_grid, fft_plan ) = mul!(recip_grid, fft_plan, charge_grid)
grad_safe_bfft!(charge_grid, recip_grid, bfft_plan) = mul!(charge_grid, bfft_plan, recip_grid)

function ewald_pe_forces!(Fs, vir, inter::PME{T}, atoms, coords, boundary, force_units,
                          energy_units, ::Val{needs_vir}, calculate_forces=true,
                          ::Val{needs_pe}=Val(true), ::Val{TH}=Val(Float64);
                          n_threads::Integer=Threads.nthreads()) where {T, TH, needs_vir, needs_pe}
    if !is_on_gpu(coords) && n_threads > 1 &&
            (isnothing(inter.charge_grid_buffer) || length(inter.virial_buffer) != n_threads)
        ntc = (isnothing(inter.charge_grid_buffer) ? 1 : length(inter.virial_buffer))
        error("PME was created with n_threads $ntc but called with n_threads $n_threads")
    end
    n_thr = (inter.grad_safe ? 1 : n_threads) # Enzyme error with multiple threads
    order, ϵr, α, mesh_dims = inter.order, inter.ϵr, inter.α, inter.mesh_dims
    V = volume(boundary)
    f = (energy_units == NoUnits ? ustrip(T(Molly.coulomb_const)) : T(Molly.coulomb_const))
    f_div_ϵr = f / ϵr

    recip_box = invert_box_vectors(boundary)
    grid_indices, grid_fractions = atom_last(inter.grid_indices), atom_last(inter.grid_fractions)
    bsplines_θ, bsplines_dθ = atom_last(inter.bsplines_θ), atom_last(inter.bsplines_dθ)
    grid_placement!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, n_thr)
    update_bsplines!(bsplines_θ, bsplines_dθ, grid_fractions, order, n_thr)
    spread_charge!(inter.charge_grid, inter.charge_grid_buffer, grid_indices,
                   bsplines_θ, mesh_dims, order, atoms, inter.scheduler,
                   n_spread_threads(n_thr))
    grad_safe_fft!(inter.charge_grid, inter.recip_grid, inter.fft_plan)
    reciprocal_space_E = recip_conv!(vir, inter.virial_buffer, inter.recip_grid,
                    inter.recip_conv_buffer, inter.bsplines_moduli_x, inter.bsplines_moduli_y,
                    inter.bsplines_moduli_z, recip_box, f_div_ϵr, α, mesh_dims, boundary,
                    energy_units, n_thr, Val(needs_vir), Val(needs_pe))
    grad_safe_bfft!(inter.charge_grid, inter.recip_grid, inter.bfft_plan)
    if calculate_forces
        interpolate_force!(Fs, inter.charge_grid, grid_indices, bsplines_θ,
                           bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                           inter.scheduler, n_thr)
    end

    if needs_pe || needs_vir
        if isnothing(inter.pc_sum) || inter.grad_safe
            partial_charges = effective_charge.(from_device(atoms), Ref(inter.scheduler), Val(T))
            pc_sum      = sum_float_type(identity, TH, partial_charges)
            pc_abs2_sum = sum_float_type(abs2    , TH, partial_charges)
        else
            pc_sum, pc_abs2_sum = TH(inter.pc_sum), TH(inter.pc_abs2_sum)
        end
        f_h, α_h, V_h = TH(f_div_ϵr), TH(α), TH(V)
        charge_E = -f_h * TH(π) * pc_sum^2 / (2 * V_h * α_h^2)
        self_E = -f_h * pc_abs2_sum * α_h / sqrt(TH(π)) + charge_E
        if needs_vir
            # Since charge_E = -A/V, affine box differentiation gives W = charge_E * I
            vir .+= charge_E .* I(3)
        end
        if needs_pe
            total_E = reciprocal_space_E + self_E
            return total_E
        end
    end
    return nothing
end

"""
    SetupPME(; error_tol=0.0005, approximate_erfc=true, mesh_dims=nothing,
             coulomb_const=138.93545764u"kJ * mol^-1 * nm")

Set up the particle mesh Ewald (PME) summation for long range electrostatics.

Passed to the [`System`](@ref) constructor from files, where it creates a [`PME`](@ref)
general interaction, a [`CoulombEwald`](@ref) pairwise interaction and a
[`EwaldExclusion`](@ref) specific interaction.

`error_tol` is the error tolerance for Ewald summation.
`approximate_erfc` determines whether to use a fast approximation to the erfc function.
`mesh_dims` determines the number of PME grid points in each dimension and defaults
to a value chosen from `error_tol`.
"""
struct SetupPME{T, C, M} <: AbstractSetupEwald
    error_tol::T
    approximate_erfc::Bool
    coulomb_const::C
    mesh_dims::M
end

function SetupPME(; error_tol=default_ewald_error_tol, approximate_erfc::Bool=true,
                  coulomb_const=coulomb_const, mesh_dims=nothing)
    if error_tol <= zero(error_tol)
        throw(ArgumentError("error_tol must be greater than zero, found $error_tol"))
    end
    return SetupPME(error_tol, approximate_erfc, coulomb_const, mesh_dims)
end

function setup_coulomb_general(se::SetupPME, atoms, boundary, dist_cutoff, n_threads,
                               grad_safe, units, T)
    return PME(
        T(dist_cutoff),
        atoms,
        boundary;
        error_tol=T(se.error_tol),
        mesh_dims=se.mesh_dims,
        grad_safe=grad_safe,
        n_threads=n_threads,
    )
end

"""
    ewald_pair_qq(scheduler, atom_i, atom_j, Val(T))

The charge product a pair contributes under the reciprocal sum that `scheduler` selects.

Used by the short range Ewald interactions and by [`EwaldExclusion`](@ref) so that both
agree with the mesh by construction. Note that no role mixing appears here: the per atom
effective charges already carry everything the mesh knows, so routing this through
`mix_roles` could only introduce disagreement.
"""
@inline function ewald_pair_qq(scheduler, atom_i, atom_j, ::Val{T}; kwargs) where T
    return effective_charge(atom_i, scheduler, Val(T)) *
           effective_charge(atom_j, scheduler, Val(T))
end

@inline function ewald_pair_qq(scheduler::OpenFEScheduler, atom_i, atom_j, ::Val{T}; special=false) where T
    # OpenFE/OpenMM: charges scaled per atom on one grid
    λ_glob = T(λ_mixing(MinimumMixing(), (atom_i.λ, atom_j.λ)))
    pair_role = mix_roles(scheduler, (atom_i.alch_role, atom_j.alch_role); type="coulomb")
    λ, λR, λ_params = scale_elec_dual(scheduler, λ_glob, pair_role)
    if scheduler.dual
        qij = atom_i.charge * atom_j.charge
    elseif special
        qij = atom_i.charge .* atom_j.charge
        qij = params_mixing(λ_params, qij)
    else
        qi = effective_charge(atom_i, scheduler, Val(T))
        qj = effective_charge(atom_j, scheduler, Val(T))
        qij = qi*qj
    end
    return qij
end

@inline function ewald_pair_qq(scheduler::Union{GROMACSLambdaABFEScheduler,GROMACSLambdaRBFEScheduler}, atom_i, atom_j,
                               ::Val{T}; kwargs) where T
    # GROMACS: two grids mixed with weights (1 - λ, λ), with λ the electrostatic coupling —
    # the same weight `pme_lambda_mesh_weight` gives the mesh, so the two agree by
    # construction. `InsertRole` stands in for any role because the scheduler is required to
    # scale Insert and Delete as mirror images.
    λ_glob = T(λ_mixing(MinimumMixing(), (atom_i.λ, atom_j.λ)))
    λ, λR, λ_params = scale_elec_dual(scheduler, λ_glob, InsertRole)
    qA = effective_charge(atom_i, scheduler, Val(T), zero(T)) *
         effective_charge(atom_j, scheduler, Val(T), zero(T))
    qB = effective_charge(atom_i, scheduler, Val(T), one(T)) *
         effective_charge(atom_j, scheduler, Val(T), one(T))
    return (one(T) - λ) * qA + λ * qB
end

"""
    EwaldExclusion()

Exclusions for bonded interactions for long range electrostatics.

Should be used alongside the [`Ewald`](@ref) or [`PME`](@ref) general interaction,
which provide the long-range term, and the [`CoulombEwald`](@ref) pairwise interaction,
which provides the short range term.

Since the properties of the interaction are the same for all pairs, they are stored once
in the `data` field of the [`InteractionList2Atoms`](@ref) with `EwaldExclusionData`.
`dist_cutoff` and `error_tol` for `EwaldExclusionData` should match the above interactions.

Only compatible with 3D systems.
"""
@kwdef struct EwaldExclusion null::UInt8 = 0 end
# Due to a CuArray error with empty structs (https://github.com/JuliaGPU/CUDA.jl/issues/3181)

Base.zero(::Type{EwaldExclusion}) = EwaldExclusion()
Base.zero(e::EwaldExclusion) = zero(typeof(e))
Base.:+(::EwaldExclusion, ::EwaldExclusion) = EwaldExclusion()

struct EwaldExclusionData{T, D, A, F, S, LM}
    dist_cutoff::D
    error_tol::T
    ϵr::T
    α::A
    f_div_ϵr::F
    scheduler::S
    λ_mixing::LM
end

function EwaldExclusionData(dist_cutoff; error_tol=default_ewald_error_tol, ϵr=1.0,
                            scheduler=DefaultLambdaScheduler(), λ_mix=MinimumMixing())
    T = typeof(ustrip(dist_cutoff))
    error_tol_T = T(error_tol)
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol_T))
    f = (unit(dist_cutoff) == NoUnits ? ustrip(T(Molly.coulomb_const)) : T(Molly.coulomb_const))
    ϵr_T = T(ϵr)
    f_div_ϵr = f / ϵr_T
    return EwaldExclusionData(dist_cutoff, error_tol_T, ϵr_T, α, f_div_ϵr, scheduler, λ_mix)
end

function Base.zero(inter::EwaldExclusionData{T, D, A, F}) where {T, D, A, F}
    return EwaldExclusionData(zero(D), zero(T), zero(T), zero(A), zero(F), inter.scheduler,
                              inter.λ_mixing)
end

function Base.:+(i1::EwaldExclusionData, i2::EwaldExclusionData)
    return EwaldExclusionData(
        i1.dist_cutoff + i2.dist_cutoff,
        i1.error_tol + i2.error_tol,
        i1.ϵr + i2.ϵr,
        i1.α + i2.α,
        i1.f_div_ϵr + i2.f_div_ϵr,
        i1.scheduler,
        i1.λ_mixing
    )
end

to_device(x::EwaldExclusionData, ::Type{AT}) where AT = x

@inline function force(::EwaldExclusion, coord_i, coord_j, boundary, atom_i, atom_j,
                       force_units, velocities_i, velocities_j, step_n,
                       data::EwaldExclusionData{T}) where T
    vec_ij = vector(coord_i, coord_j, boundary)
    r = sqrt(sum(abs2, vec_ij))
    scheduler, α, f_div_ϵr = data.scheduler, data.α, data.f_div_ϵr
    qq = ewald_pair_qq(scheduler, atom_i, atom_j, Val(T))

    αr = α * r
    erf_αr = erf(αr)
    if erf_αr > T(1e-6)
        inv_r = inv(r)
        F = f_div_ϵr * qq * inv_r^3 * (erf_αr - 2 * αr * exp(-αr^2) / sqrt(T(π))) * vec_ij
        return SpecificForce2Atoms(F, -F)
    else
        zf = zero(SVector{3, T}) * force_units
        return SpecificForce2Atoms(zf, zf)
    end
end

@inline function potential_energy(inter::EwaldExclusion, coord_i, coord_j, boundary, atom_i, atom_j,
                                  energy_units, velocities_i, velocities_j, step_n,
                                  data::EwaldExclusionData{T}) where T
    vec_ij = vector(coord_i, coord_j, boundary)
    r = sqrt(sum(abs2, vec_ij))
    scheduler, α, f_div_ϵr = data.scheduler, data.α, data.f_div_ϵr
    qq = ewald_pair_qq(scheduler, atom_i, atom_j, Val(T))

    erf_αr = erf(α * r)
    if erf_αr > T(1e-6)
        E = -f_div_ϵr * qq * inv(r) * erf_αr
    else
        E = -α * 2 * f_div_ϵr * qq / sqrt(T(π))
    end
    return E
end

"""
    PME_λ(dist_cutoff, atoms, boundary; error_tol=0.0005, order=5,
        ϵr=1.0, fixed_charges=true, mesh_dims=nothing,
        scheduler=LinearLambdaScheduler(dual=true), grad_safe=false,
        n_threads=Threads.nthreads())

Particle mesh Ewald summation for long range electrostatics implemented as an
AtomsCalculators.jl calculator for GROMACS-based free energy calculations.

Should be used alongside any alchemical version of [`CoulombEwald`](@ref) pairwise interaction,
which provides the short range term, and the [`EwaldExclusion`](@ref) specific
interaction, which provides the exclusions for bonded atoms.
`dist_cutoff` and `error_tol` should match these interactions.

`dist_cutoff` is the cutoff distance for short range interactions.
`fixed_charges` should be set to `false` if the partial charges can change,
for example when using a polarizable force field.
`mesh_dims` gives the number of grid points in each dimension, overriding the
value chosen from `error_tol`.
`scheduler` scheduler used for λ-interpolation, this PME works only with GROMACSLambdaABFEScheduler or
    GROMACSLambdaRBFEScheduler.
`λ` the global λ assigned to the system.
`grad_safe` should be set to `true` if gradients are going to be calculated
with Enzyme.jl.
`n_threads` is used to pre-allocate memory on CPU and plan the FFTs.

This implementation is based on the implementation in GROMACS.

Only compatible with 3D systems.
Not compatible with infinite boundaries.
"""
struct PME_λ{T, D, A, I, M, BM, C, RG, CB, RB, VB, PB, P, F, B, SCH} <: AbstractEwald
    dist_cutoff::D
    error_tol::T
    order::Int
    ϵr::T
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
    recip_grid::RG
    charge_grid_buffer::CB
    recip_conv_buffer::RB
    virial_buffer::VB
    partial_charge_buffer::PB
    pc_sum::P
    pc_abs2_sum::P
    fft_plan::F
    bfft_plan::B
    scheduler::SCH
    grad_safe::Bool
    λ::T
end

"""
    pme_lambda_mesh_weight(scheduler, global_λ, Val(T))

The weight the two grid reciprocal sum mixes its end state grids with:
`E = (1 - w)·E[q(0)] + w·E[q(1)]`.

`w` is the electrostatic coupling `s_I(global_λ)`, **not** the global λ. Under a staged
schedule they differ everywhere in between: `GROMACSLambdaABFEScheduler` has the charges fully
off by `global_λ = 0.5`, while the sterics have not started. Taking `w` from the coupling is
what lets electrostatics and sterics be staged independently — the direct, reciprocal, self
and exclusion terms are all electrostatic and all read this one weight, while the LJ curve
touches none of them.

One weight can only serve both roles when `s_I(λ) == 1 - s_D(λ)`, which is checked here.
"""
function pme_lambda_mesh_weight(scheduler::Union{GROMACSLambdaABFEScheduler,GROMACSLambdaRBFEScheduler}, 
                                global_λ, ::Val{T}) where T
    for l in 0.0:0.001:1.0
        l0, _, _ = scale_elec_dual(scheduler, l, InsertRole)
        l1, _, _ = scale_elec_dual(scheduler, l, DeleteRole)
        if abs(l0 - (1-l1)) > 1e-6
            error("PME_λ needs InsertRole and DeleteRole to be mirror images, " *
                  "s_I(λ) == 1 - s_D(λ), so that one weight serves both. $(scheduler) " *
                  "breaks that at λ=$l.")
        end
    end
    λ, _, _ = scale_elec_dual(scheduler, global_λ, InsertRole)
    return λ
end

function PME_λ(dist_cutoff, atoms, boundary; error_tol=0.0005, order=5,
             ϵr=1.0, fixed_charges=true, mesh_dims=nothing, eligible=nothing, special=nothing,
             scheduler=GROMACSLambdaABFEScheduler(dual=true), λ=0.0, grad_safe=false,
             n_threads::Integer=Threads.nthreads())
    T = typeof(ustrip(dist_cutoff))
    AT = array_type(atoms)
    n_atoms = length(atoms)
    error_tol_T = T(error_tol)
    mesh_w = pme_lambda_mesh_weight(scheduler, T(λ), Val(T))
    α = inv(dist_cutoff) * sqrt(-log(2 * error_tol_T))
    if isnothing(mesh_dims)
        mesh_dims = pme_params.(box_sides(boundary), α, error_tol_T)
    else
        mesh_dims = SVector{3, Int}(mesh_dims)
    end

    if !scheduler.dual
        error(" $(scheduler) has dual=false which results in single topology." *
             "PME_λ only works for dual topology. Use a dual GROMACS scheduler, " *
             "or `PME` with different lambda scheduler.")
    end

    # Unlike `PME`, the per-atom arrays keep the same (state, value, atom) layout on
    # both CPU and GPU. The batched kernels below are KernelAbstractions kernels that
    # run on either backend, so there is no CPU-specific memory layout to match.
    grid_indices = to_device(zeros(Int, 2, 3, n_atoms), AT)
    grid_fractions = to_device(zeros(T, 2, 3, n_atoms), AT)
    bsplines_θ = to_device(zeros(T, 2, order * 3, n_atoms), AT)
    bsplines_dθ = zero(bsplines_θ)
    # Ordered z/y/x for better memory access. The charge grid is real, so a real to complex
    # transform is used, which halves the work of the FFTs and of everything else that
    # touches the reciprocal space grid.
    charge_grid = to_device(zeros(T, mesh_dims[3], mesh_dims[2], mesh_dims[1], 2), AT)
    recip_grid = to_device(zeros(Complex{T}, mesh_dims[3] ÷ 2 + 1, mesh_dims[2], mesh_dims[1], 2), AT)
    excluded_pairs = to_device(find_excluded_pairs(eligible, special), AT)

    bsplines_moduli = pme_bspline_moduli(T, order, mesh_dims)

    # The charge is added to the real grid atomically, so no per-thread grid is needed
    charge_grid_buffer    = nothing
    recip_conv_buffer     = to_device(zeros(T, size(recip_grid)), AT)
    virial_buffer         = to_device(zeros(T, 2, 3, 3), AT)
    partial_charge_buffer = to_device(zeros(T, 2, n_atoms), AT)

    # The end state charges depend on the scheduler and on each atom's role, so they are
    # recomputed on the fly rather than cached; see `ewald_pe_forces!` below.
    pc_sum, pc_abs2_sum = nothing, nothing

    # The grids are stored with the two end states last, which is the layout both FFT libraries
    # want: transform dims leading and contiguous, batch stride `nz*ny*nx`. Storing them the
    # other way round is why this used to permute in and out of FFT layout on every evaluation
    # — and planning over the batch-first layout directly is not an option, since cuFFT's R2C
    # advanced data layout cannot express `istride=2, idist=1`.
    if AT <: AbstractGPUArray
        fft_plan  = plan_rfft(charge_grid, 1:3)
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3], 1:3)
    else
        fft_plan  = plan_rfft(charge_grid, 1:3; flags=FFTW.MEASURE, num_threads=n_threads)
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3], 1:3; flags=FFTW.MEASURE,
                               num_threads=n_threads)
    end

    charge_grid .= zero(T) # Can be overwritten by FFTW.MEASURE
    recip_grid .= zero(Complex{T})

    bsm_x = to_device(bsplines_moduli[1], AT)
    bsm_y = to_device(bsplines_moduli[2], AT)
    bsm_z = to_device(bsplines_moduli[3], AT)

    return PME_λ(dist_cutoff, error_tol_T, order, T(ϵr), α, mesh_dims, grid_indices, grid_fractions,
               bsplines_θ, bsplines_dθ, bsm_x, bsm_y, bsm_z, charge_grid, recip_grid, charge_grid_buffer,
               recip_conv_buffer, virial_buffer, partial_charge_buffer, pc_sum, pc_abs2_sum, fft_plan, bfft_plan,
               scheduler, grad_safe, mesh_w)
end

Base.deepcopy_internal(pme::PME_λ, dict::IdDict) = deepcopy_registered(pme, dict)

function Base.deepcopy(pme::PME_λ)
    # 1. Deepcopy standard immutable parameters and CPU buffers
    dist_cutoff    = pme.dist_cutoff
    error_tol      = pme.error_tol
    order          = pme.order
    ϵr             = pme.ϵr
    α              = pme.α
    mesh_dims      = pme.mesh_dims
    pc_sum         = deepcopy(pme.pc_sum)
    pc_abs2_sum    = deepcopy(pme.pc_abs2_sum)
    scheduler      = pme.scheduler
    grad_safe      = pme.grad_safe
    λ              = pme.λ

    # 2. Duplicate GPU/CPU Arrays cleanly via deepcopy or copy
    grid_indices        = deepcopy(pme.grid_indices)
    grid_fractions      = deepcopy(pme.grid_fractions)
    bsplines_θ          = deepcopy(pme.bsplines_θ)
    bsplines_dθ         = deepcopy(pme.bsplines_dθ)
    bsplines_moduli_x   = deepcopy(pme.bsplines_moduli_x)
    bsplines_moduli_y   = deepcopy(pme.bsplines_moduli_y)
    bsplines_moduli_z   = deepcopy(pme.bsplines_moduli_z)
    partial_charge_buffer = deepcopy(pme.partial_charge_buffer)

    # Critical allocations: Main charge grid and working buffers
    charge_grid         = deepcopy(pme.charge_grid)
    charge_grid_buffer  = deepcopy(pme.charge_grid_buffer)
    recip_grid          = deepcopy(pme.recip_grid)
    recip_conv_buffer   = deepcopy(pme.recip_conv_buffer)
    virial_buffer       = deepcopy(pme.virial_buffer)

    if charge_grid isa AbstractGPUArray
        fft_plan  = plan_rfft(charge_grid, 1:3)
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3], 1:3)
    else
        n_threads = Threads.nthreads()
        fft_plan  = plan_rfft(charge_grid, 1:3; flags=FFTW.MEASURE, num_threads=n_threads)
        bfft_plan = plan_brfft(recip_grid, mesh_dims[3], 1:3; flags=FFTW.MEASURE,
                               num_threads=n_threads)
    end
    # FFTW.MEASURE overwrites its input while planning; both grids are scratch, rewritten at the
    # start of every evaluation, but zero them so a copy is never observed holding junk.
    charge_grid .= zero(eltype(charge_grid))
    recip_grid  .= zero(eltype(recip_grid))

    # 4. Return the brand new, isolated PME_λ struct
    return PME_λ(
        dist_cutoff, error_tol, order, ϵr, α, mesh_dims,
        grid_indices, grid_fractions, bsplines_θ, bsplines_dθ, 
        bsplines_moduli_x, bsplines_moduli_y, bsplines_moduli_z,
        charge_grid, recip_grid, charge_grid_buffer,
        recip_conv_buffer, virial_buffer, partial_charge_buffer, pc_sum, pc_abs2_sum, 
        fft_plan, bfft_plan, scheduler, grad_safe, λ
    )
end

function Base.zero(pme::PME_λ)
    if pme.charge_grid_buffer isa Vector
        charge_grid_buffer = zero.(pme.charge_grid_buffer)
    else
        charge_grid_buffer = zero_or_nothing(pme.charge_grid_buffer)
    end
    return PME_λ(
        zero(pme.dist_cutoff),
        zero(pme.error_tol),
        pme.order,
        zero(pme.ϵr),
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
        zero(pme.recip_grid),
        charge_grid_buffer,
        zero_or_nothing(pme.recip_conv_buffer),
        zero_or_nothing(pme.virial_buffer),
        zero_or_nothing(pme.partial_charge_buffer),
        zero_or_nothing(pme.pc_sum),
        zero_or_nothing(pme.pc_abs2_sum),
        pme.fft_plan,
        pme.bfft_plan,
        pme.scheduler,
        pme.grad_safe,
        zero(pme.λ),
    )
end

function ==(a::PME_λ, b::PME_λ)
    return a.dist_cutoff    == b.dist_cutoff    &&
           a.error_tol      == b.error_tol      &&
           a.order          == b.order          &&
           a.ϵr             == b.ϵr             &&
           a.α              == b.α              &&
           a.mesh_dims      == b.mesh_dims      &&
           a.scheduler      == b.scheduler      &&
           a.grad_safe      == b.grad_safe      &&
           a.λ              == b.λ
end

function hash(a::PME_λ, h::UInt)
    v = hash(a.dist_cutoff, h)
    v = hash(a.error_tol, v)
    v = hash(a.order, v)
    v = hash(a.ϵr, v)
    v = hash(a.α, v)
    v = hash(a.mesh_dims, v)
    v = hash(a.scheduler, v)
    v = hash(a.grad_safe, v)
    v = hash(a.λ, v)
    return v
end

@inline function grid_placement_inner_batch!(grid_indices, grid_fractions, coords, recip_box,
                                       mesh_dims, b, a)
    @inbounds for d in 1:3
        t = sum(coords[a] .* SVector(recip_box[1][d], recip_box[2][d], recip_box[3][d]))
        t = (t - floor(t)) * mesh_dims[d]
        ti = floor(Int, t)
        grid_fractions[b, d, a] = t - ti
        # `t` is below the mesh length, so rounding is the only way `ti` can reach it and
        # the wrap can be done with a subtraction rather than an integer division
        grid_indices[b, d, a] = wrap_grid_index(ti, mesh_dims[d])
    end
    return grid_indices, grid_fractions
end

function grid_placement_batch!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, n_threads)
    backend = get_backend(parent(grid_indices))
    n_threads_gpu = 128

    n_batches = size(grid_indices, 1)
    n_atoms   = size(coords, 1)

    kernel! = grid_placement_kernel_batch!(backend, n_threads_gpu)
    kernel!(grid_indices, grid_fractions, coords, recip_box, mesh_dims; ndrange=(n_batches,n_atoms))
    KernelAbstractions.synchronize(backend)
    return grid_indices, grid_fractions
end

@kernel function grid_placement_kernel_batch!(grid_indices, grid_fractions, @Const(coords),
                                        recip_box, mesh_dims)
    i = @index(Global, Linear)
    b, a = @index(Global, NTuple)
    
    n_batches = size(grid_indices, 1)
    n_atoms   = size(coords, 1)

    if b <= n_batches && a <= n_atoms
        grid_placement_inner_batch!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, b,a)
    end
end

@inline function update_bsplines_inner_batch!(bsplines_θ::AbstractArray{T, 3}, bsplines_dθ,
                                        grid_fractions, order, b, a) where T
    # The first entry of each dimension's block is built up multiplicatively. Keeping it in a
    # local accumulator rather than repeatedly reading and writing the same array element is
    # what makes this safe under KernelAbstractions' CPU backend, which vectorises the
    # workitem loop and otherwise drops all but the last of those updates.
    @inbounds for j in 1:3
        o = (j - 1) * order
        dr = grid_fractions[b, j, a]
        θ1 = one(T) - dr
        bsplines_θ[b, o + order, a] = zero(T)
        bsplines_θ[b, o + 2, a]     = dr
        bsplines_θ[b, o + 1, a]     = θ1
        for k in 3:(order-1)
            d = inv(k - one(T))
            bsplines_θ[b, o + k, a] = d * dr * bsplines_θ[b, o + k - 1, a]
            for l in 1:(k-2)
                bsplines_θ[b, o + k - l, a] = d * (
                        (dr + l) * bsplines_θ[b, o + k - l - 1, a] +
                        (k - l - dr) * bsplines_θ[b, o + k - l, a]
                    )
            end
            θ1 *= d * (1 - dr)
            bsplines_θ[b, o + 1, a] = θ1
        end

        bsplines_dθ[b, o + 1, a] = -θ1
        for k in 1:(order-1)
            bsplines_dθ[b, o + k + 1, a] = bsplines_θ[b, o + k, a] - bsplines_θ[b, o + k + 1, a]
        end
        d = inv(order - one(T))
        bsplines_θ[b, o + order, a] = d * dr * bsplines_θ[b, o + order - 1, a]
        for l in 1:(order-2)
            bsplines_θ[b, o + order - l, a] = d * (
                    (dr + l) * bsplines_θ[b, o + order - l - 1, a] +
                    (order - l - dr) * bsplines_θ[b, o + order - l, a]
                )
        end
        θ1 *= d * (1 - dr)
        bsplines_θ[b, o + 1, a] = θ1
    end
    return bsplines_θ, bsplines_dθ
end

function update_bsplines_batch!(bsplines_θ, bsplines_dθ, grid_fractions, order,
                          n_threads)
    backend = get_backend(parent(bsplines_θ))
    n_threads_gpu = 128

    n_batches = size(bsplines_θ, 1)
    n_atoms   = size(grid_fractions, 3)

    kernel! = update_bsplines_kernel_batch!(backend, n_threads_gpu)
    kernel!(bsplines_θ, bsplines_dθ, grid_fractions, order; ndrange=(n_batches,n_atoms))
    KernelAbstractions.synchronize(backend)
    return bsplines_θ, bsplines_dθ
end

@kernel function update_bsplines_kernel_batch!(bsplines_θ, bsplines_dθ, @Const(grid_fractions),
                                         order)
    b, a = @index(Global, NTuple)
    n_batches = size(bsplines_θ, 1)
    n_atoms   = size(grid_fractions, 3)

    if b <= n_batches && a <= n_atoms
        update_bsplines_inner_batch!(bsplines_θ, bsplines_dθ, grid_fractions, order, b, a)
    end
end

# The state-first grids are addressed by a state `b` plus a linear index `li` over the three
# mesh axes. Julia removed partial linear indexing, so `A[b, li]` on a 4D array is not a
# valid two-subscript access: under `@inbounds` it silently computes an unchecked index, and
# `Atomix.@atomic` bounds checks through its own array wrapper even inside `@inbounds`, which
# is how the CPU charge grid ended up corrupted. Build the full linear index explicitly.
@inline batch_linear_index(A, b, li) = li + (b - 1) * size(A, 1) * size(A, 2) * size(A, 3)

# CPU case, each thread has its own grid so the addition does not have to be atomic
@inline function add_charge_grid_batch!(charge_grid, b, li, cb, ::Val{false})
    @inbounds charge_grid[batch_linear_index(charge_grid, b, li)] += cb
    return charge_grid
end

# GPU case, where all the threads share one grid
@inline function add_charge_grid_batch!(charge_grid, b, li, cb, ::Val{true})
    @inbounds Atomix.@atomic charge_grid[batch_linear_index(charge_grid, b, li)] += cb
    return charge_grid
end

# The per-atom arrays are stored (state, value, atom) on both backends, which is the layout
# the kernels index, so nothing has to be permuted. A lazy `PermutedDimsArray` cannot be used
# here: Adapt rebuilds it inside the kernel and its constructor is not GPU-compilable.
@inline atom_last_batch(A) = A

@inline function spread_charge_inner_batch!(charge_grid, grid_indices, bsplines_θ,
                              mesh_dims, order, atoms, scheduler, b, i, ::Val{T},
                              ::Val{atomic}) where {T, atomic}
    q = effective_charge(atoms[i], scheduler, Val(T), (b-1))
    nx, ny, nz = mesh_dims[1], mesh_dims[2], mesh_dims[3]
    @inbounds x0index, y0index, z0index = grid_indices[b, 1, i], grid_indices[b, 2, i], grid_indices[b, 3, i]
    @inbounds for ix in 0:(order-1)
        xbase = wrap_grid_index(x0index + ix, nx) * ny * nz
        θx = bsplines_θ[b, ix+1, i]
        qx = q * θx
        for iy in 0:(order-1)
            ybase = xbase + wrap_grid_index(y0index + iy, ny) * nz
            θy = bsplines_θ[b, order+iy+1, i]
            qxy = qx * θy
            for iz in 0:(order-1)
                zindex = wrap_grid_index(z0index + iz, nz)
                θz = bsplines_θ[b, 2*order+iz+1, i]
                cb = qxy * θz
                add_charge_grid_batch!(charge_grid, b, ybase + zindex + 1, cb, Val(atomic))
            end
        end
    end
    return charge_grid
end

# GPU version, one thread per (atom, z slice) pair. `order` threads cooperate on each
# atom, which gives `order` times the parallelism of one thread per atom and makes the
# threads of an atom write neighbouring grid points, as in the OpenMM implementation
@inline function spread_charge_slice_batch!(charge_grid, grid_indices, bsplines_θ, mesh_dims, order,
                                      atoms, scheduler, b, i, iz, ::Val{T}) where T
    q = effective_charge(atoms[i], scheduler, Val(T), T(b-1))
    nx, ny, nz = mesh_dims[1], mesh_dims[2], mesh_dims[3]
    @inbounds begin
        x0index, y0index, z0index = grid_indices[b, 1, i], grid_indices[b, 2, i], grid_indices[b, 3, i]
        zindex = wrap_grid_index(z0index + iz, nz)
        qz = q * bsplines_θ[b, 2*order+iz+1, i]
        for ix in 0:(order-1)
            xbase = wrap_grid_index(x0index + ix, nx) * ny * nz
            qzx = qz * bsplines_θ[b, ix+1, i]
            for iy in 0:(order-1)
                ybase = xbase + wrap_grid_index(y0index + iy, ny) * nz
                cb = qzx * bsplines_θ[b, order+iy+1, i]
                add_charge_grid_batch!(charge_grid, b, ybase + zindex + 1, cb, Val(true))
            end
        end
    end
    return charge_grid
end

function spread_charge_batch!(charge_grid::AbstractArray{T, 4}, buffer, grid_indices,
                        bsplines_θ, mesh_dims, order, atoms, scheduler, n_threads) where T
    backend = get_backend(charge_grid)
    n_threads_gpu = 128

    n_atoms = length(atoms)*order
    n_batches = size(charge_grid, 4)

    kernel! = spread_charge_kernel_batch!(backend, n_threads_gpu)
    charge_grid .= zero(T)
    kernel!(charge_grid, grid_indices, bsplines_θ, mesh_dims, order, atoms, scheduler, Val(T);
            ndrange=(n_batches,n_atoms))
    KernelAbstractions.synchronize(backend)
    return charge_grid
end

@kernel function spread_charge_kernel_batch!(charge_grid_real, @Const(grid_indices), @Const(bsplines_θ),
                                       mesh_dims, order, atoms, scheduler, ::Val{T}) where T
    b, a = @index(Global, NTuple)
    n_batches = size(charge_grid_real, 4)
    n_atoms = length(atoms)*order

    if b <= n_batches && a <= n_atoms
        i, iz1 = fldmod1(a, order)
        spread_charge_slice_batch!(charge_grid_real, grid_indices, bsplines_θ, mesh_dims, order, atoms,
                             scheduler, b, i, iz1-1, Val(T))
    end
end

@inline function recip_conv_inner_batch!(vir_nou, recip_grid::AbstractArray{Complex{T}, 4}, bsm_x,
                           bsm_y, bsm_z, recip_box, mesh_dims, energy_units, f_div_ϵr, factor,
                           boxfactor, b, kx, ky, kz, ::Val{needs_vir},
                           ::Val{atomic}) where {T, needs_vir, atomic}
    if iszero(kx) && iszero(ky) && iszero(kz)
        return zero(T) * energy_units
    end
    nx, ny, nz = mesh_dims
    maxkx, maxky, maxkz = T(0.5)*(nx+1), T(0.5)*(ny+1), T(0.5)*(nz+1)
    # The real to complex transform only keeps the modes with kz up to nz/2, and each of
    # them stands for both k and -k of the full mesh apart from the two, or one when nz is
    # odd, that are their own conjugate
    weight = (iszero(kz) || 2*kz == nz ? one(T) : T(2))
    @inbounds begin
        mx = (kx < maxkx ? kx : kx - nx)
        mhx = mx * recip_box[1][1]
        bx = boxfactor * bsm_x[kx+1]
        my = (ky < maxky ? ky : ky - ny)
        mhy = mx * recip_box[2][1] + my * recip_box[2][2]
        by = bsm_y[ky+1]
        mz = (kz < maxkz ? kz : kz - nz)
        mhz = mx * recip_box[3][1] + my * recip_box[3][2] + mz * recip_box[3][3]
        d1, d2 = reim(recip_grid[kz+1, ky+1, kx+1, b])
        m2 = mhx^2 + mhy^2 + mhz^2
        bz = bsm_z[kz+1]
        denom = m2 * bx * by * bz
        c  = exp(-factor * m2)
        eterm = f_div_ϵr * c / denom
        eterm_nou = ustrip(energy_units, eterm)
        recip_grid[kz+1, ky+1, kx+1, b] = Complex(d1*eterm_nou, d2*eterm_nou)
        struct2 = weight * (d1^2 + d2^2)

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
                        Atomix.@atomic vir_nou[b, d1, d2] += Ek_nou * G[d1, d2]
                    end
                end
            else
                for d1 in 1:3, d2 in 1:3
                    vir_nou[b, d1, d2] += Ek_nou * G[d1, d2]
                end
            end
        end
    end
    return eterm * struct2
end

function recip_conv_batch!(vir, buffer_virial, recip_grid::AbstractArray{Complex{T}, 4}, buffer, bsm_x,
                     bsm_y, bsm_z, recip_box, f_div_ϵr, α, mesh_dims, boundary, energy_units,
                     n_threads, ::Val{needs_vir}, ::Val{needs_pe}, λ) where {T, needs_vir, needs_pe}
    if needs_vir
        buffer_virial .= zero(T)
    end
    factor = T(π)^2 / α^2
    boxfactor = T(π) * volume(boundary)
    backend = get_backend(recip_grid)
    n_threads_gpu = 256

    n_batches = size(recip_grid, 4)
    len_recipg = Int(length(recip_grid)/n_batches)

    kernel! = recip_conv_kernel_batch!(backend, n_threads_gpu)
    kernel!(buffer_virial, buffer, recip_grid, bsm_x, bsm_y, bsm_z, recip_box, mesh_dims,
            energy_units, f_div_ϵr, factor, boxfactor, Val(needs_vir), Val(needs_pe);
            ndrange=(n_batches, len_recipg))
    KernelAbstractions.synchronize(backend)
    if needs_vir
        # The mesh sums both k and -k, so the virial needs the same 1/2 as the energy.
        vir .+= dropdims(sum((from_device(buffer_virial) .* energy_units / 2) .* (1-λ,λ), dims=1), dims=1)
    end
    # The energy is discarded when only forces are wanted, in which case the reduction
    # over the whole mesh, and the device synchronisation it forces, can be skipped
    needs_pe || return zero(T) * energy_units
    return sum(vec(sum(buffer, dims=[1,2,3])) * energy_units / 2 .* (1-λ,λ))
end

# One thread per grid point, indexed so that neighbouring threads touch neighbouring grid
# points. `recip_grid` is stored z fastest, so z has to be the fastest varying index of
# the launch as well.
@kernel function recip_conv_kernel_batch!(vir, esum_arr, recip_grid, @Const(bsm_x), @Const(bsm_y),
                                    @Const(bsm_z), recip_box, mesh_dims, energy_units,
                                    f_div_ϵr, factor, boxfactor, ::Val{needs_vir},
                                    ::Val{needs_pe}) where {needs_vir, needs_pe}
    b, li = @index(Global, NTuple)

    n_batches = size(recip_grid, 4)
    len_recipg = Int(length(recip_grid)/n_batches)

    if b <= n_batches && li <= len_recipg
        nzh = size(recip_grid, 1)
        i0 = li - 1
        kz, r = i0 % nzh, i0 ÷ nzh
        ky, kx = r % mesh_dims[2], r ÷ mesh_dims[2]
        esum = recip_conv_inner_batch!(vir, recip_grid, bsm_x, bsm_y, bsm_z, recip_box, mesh_dims,
                                 energy_units, f_div_ϵr, factor, boxfactor, b,
                                 kx, ky, kz, Val(needs_vir), Val(true))
        if needs_pe
            @inbounds esum_arr[batch_linear_index(esum_arr, b, li)] = ustrip(energy_units, esum)
        end
    end
end

# GPU version, one thread per (atom, z slice) pair as for the charge spreading. Each of
# the `order` threads of an atom accumulates a partial force over its own z slice and
# adds it atomically, which gives `order` times the parallelism of one thread per atom
# and makes the threads of an atom read neighbouring grid points.
@inline function interpolate_force_slice_batch!(Fs_flat, charge_grid, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, unit_scale, atoms,
                            scheduler, ::Val{T}, b, i, iz) where T
    nx, ny, nz = mesh_dims
    fx, fy, fz = zero(T), zero(T), zero(T)
    @inbounds begin
        q = effective_charge(atoms[i], scheduler, Val(T), T(b-1))
        x0index, y0index, z0index = grid_indices[b, 1, i], grid_indices[b, 2, i], grid_indices[b, 3, i]
        zindex = wrap_grid_index(z0index + iz, nz)
        tz, dtz = bsplines_θ[b, 2*order+iz+1, i], bsplines_dθ[b, 2*order+iz+1, i]
        for ix in 0:(order-1)
            xbase = wrap_grid_index(x0index + ix, nx) * ny * nz
            tx, dtx = bsplines_θ[b, ix+1, i], bsplines_dθ[b, ix+1, i]
            for iy in 0:(order-1)
                ybase = xbase + wrap_grid_index(y0index + iy, ny) * nz
                ty, dty = bsplines_θ[b, order+iy+1, i], bsplines_dθ[b, order+iy+1, i]
                gridvalue = charge_grid[batch_linear_index(charge_grid, b, ybase + zindex + 1)]
                fx += dtx * ty * tz * gridvalue
                fy += tx * dty * tz * gridvalue
                fz += tx * ty * dtz * gridvalue
            end
        end
        # `Fs_flat` reinterprets the force vectors as raw numbers in the force units, since
        # atomics do not work on the unitful static vectors. `recip_box` is stripped of its
        # units on the host for the same reason, with `unit_scale` putting them back.
        f1 = q * (fx*nx*recip_box[1][1])
        f2 = q * (fx*nx*recip_box[2][1] + fy*ny*recip_box[2][2])
        f3 = q * (fx*nx*recip_box[3][1] + fy*ny*recip_box[3][2] + fz*nz*recip_box[3][3])
        Atomix.@atomic Fs_flat[b, 3*(i-1)+1] -= unit_scale * f1
        Atomix.@atomic Fs_flat[b, 3*(i-1)+2] -= unit_scale * f2
        Atomix.@atomic Fs_flat[b, 3*(i-1)+3] -= unit_scale * f3
    end
    return Fs_flat
end

function interpolate_force_batch!(Fs, charge_grid::AbstractArray{T, 4}, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            scheduler, n_threads, λ) where T
    backend = get_backend(Fs)
    AT = array_type(atoms)
    n_threads_gpu = 128
    force_units = unit(zero(eltype(eltype(Fs))))
    recip_box_nou = map(v -> ustrip.(v), recip_box)
    unit_scale = T(ustrip(force_units,
                          oneunit(T) * unit(eltype(eltype(recip_box))) * energy_units))

    n_atoms = length(atoms)*order
    n_batches = size(charge_grid, 4)
    Fs_flat = to_device(zeros(T, n_batches, length(atoms)*3), AT)

    kernel! = interpolate_force_kernel_batch!(backend, n_threads_gpu)
    kernel!(Fs_flat, charge_grid, grid_indices, bsplines_θ, bsplines_dθ, recip_box_nou,
            mesh_dims, order, unit_scale, atoms, scheduler, Val(T);
            ndrange=(n_batches, n_atoms))
    KernelAbstractions.synchronize(backend)
    
    f_sum = dropdims(sum(Fs_flat .* (1-λ, λ), dims=1),dims=1)
    Fs .+= reinterpret(SVector{3, T}, f_sum) .* force_units

    return Fs
end

@kernel function interpolate_force_kernel_batch!(Fs_flat, @Const(charge_grid), @Const(grid_indices),
                        @Const(bsplines_θ), @Const(bsplines_dθ), recip_box, mesh_dims, order,
                        unit_scale, @Const(atoms), scheduler, ::Val{T}) where T
    b, ti = @index(Global, NTuple)
    n_batches = size(charge_grid, 4)
    n_atoms = length(atoms)*order

    if b <= n_batches && ti <= n_atoms
        i, iz1 = fldmod1(ti, order)
        interpolate_force_slice_batch!(Fs_flat, charge_grid, grid_indices, bsplines_θ,
                    bsplines_dθ, recip_box, mesh_dims, order, unit_scale, atoms, scheduler,
                    Val(T), b, i, iz1-1)
    end
end

function ewald_pe_forces!(Fs, vir, inter::PME_λ{T}, atoms, coords, boundary, force_units,
                          energy_units, ::Val{needs_vir}, calculate_forces=true,
                          ::Val{needs_pe}=Val(true);
                          n_threads::Integer=Threads.nthreads()) where {T, needs_vir, needs_pe}
    n_thr = (inter.grad_safe ? 1 : n_threads) # Enzyme error with multiple threads
    order, ϵr, α, mesh_dims, λ = inter.order, inter.ϵr, inter.α, inter.mesh_dims, inter.λ
    V = volume(boundary)
    f = (energy_units == NoUnits ? ustrip(T(Molly.coulomb_const)) : T(Molly.coulomb_const))
    f_div_ϵr = f / ϵr

    recip_box = invert_box_vectors(boundary)
    grid_indices, grid_fractions = atom_last_batch(inter.grid_indices), atom_last_batch(inter.grid_fractions)
    bsplines_θ, bsplines_dθ = atom_last_batch(inter.bsplines_θ), atom_last_batch(inter.bsplines_dθ)
    charge_grid, recip_grid = inter.charge_grid, inter.recip_grid
    grid_placement_batch!(grid_indices, grid_fractions, coords, recip_box, mesh_dims, n_thr)
    update_bsplines_batch!(bsplines_θ, bsplines_dθ, grid_fractions, order, n_thr)
    spread_charge_batch!(charge_grid, inter.charge_grid_buffer, grid_indices,
                    bsplines_θ, mesh_dims, order, atoms, inter.scheduler,
                    Molly.n_spread_threads(n_thr))

    Molly.grad_safe_fft!(charge_grid, recip_grid, inter.fft_plan)

    reciprocal_space_E = Molly.recip_conv_batch!(vir, inter.virial_buffer, recip_grid,
                    inter.recip_conv_buffer, inter.bsplines_moduli_x, inter.bsplines_moduli_y,
                    inter.bsplines_moduli_z, recip_box, f_div_ϵr, α, mesh_dims, boundary,
                    energy_units, n_thr, Val(needs_vir), Val(needs_pe), λ)

    Molly.grad_safe_bfft!(charge_grid, recip_grid, inter.bfft_plan)

    if calculate_forces
        Molly.interpolate_force_batch!(Fs, charge_grid, grid_indices, bsplines_θ,
                            bsplines_dθ, recip_box, mesh_dims, order, energy_units, atoms,
                            inter.scheduler, n_thr, λ)
    end

    if needs_pe || needs_vir
        if isnothing(inter.pc_sum) || inter.grad_safe
            cA = effective_charge.(atoms, Ref(inter.scheduler), Val(T), T(0))
            cB = effective_charge.(atoms, Ref(inter.scheduler), Val(T), T(1))
            pc_sum      = (sum(cA), sum(cB))
            pc_abs2_sum = (sum(abs2,cA), sum(abs2,cB))
        else
            pc_sum, pc_abs2_sum = inter.pc_sum, inter.pc_abs2_sum
        end
        factor1 = -f_div_ϵr * T(π) / (2 * V * α^2)
        factor2 = -f_div_ϵr * α / sqrt(T(π))
        charge_E = factor1 .* (pc_sum .^ 2)
        self_E   = (factor2 .* pc_abs2_sum) .+ charge_E
        λ_weights = (1 - λ, λ)
        self_E_scalar = self_E ⋅ λ_weights

        if needs_vir
            # Since charge_E = -A/V, affine box differentiation gives W = charge_E * I.
            charge_E_scalar = charge_E ⋅ λ_weights
            vir .+= charge_E_scalar .* I(3)
        end
        if needs_pe
            return reciprocal_space_E + self_E_scalar
        end
    end
    return nothing
end