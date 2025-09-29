# Temperature and pressure coupling methods

export
    apply_coupling!,
    NoCoupling,
    ImmediateThermostat,
    VelocityRescaleThermostat,
    AndersenThermostat,
    BerendsenThermostat,
    BerendsenBarostat,
    CRescaleBarostat,
    MonteCarloBarostat

"""
    apply_coupling!(system, coupling, simulator, neighbors=nothing, step_n=0;
                    n_threads=Threads.nthreads(), rng=Random.default_rng())

Apply a coupler to modify a simulation.

Returns whether the coupling has invalidated the currently stored forces, for
example by changing the coordinates.
This information is useful for some simulators.
If `coupling` is a tuple or named tuple then each coupler will be applied in turn.
Custom couplers should implement this function.
"""
function apply_coupling!(sys, couplers::Union{Tuple, NamedTuple}, sim, neighbors,
                         step_n; kwargs...)
    recompute_forces = false
    for coupler in couplers
        rf = apply_coupling!(sys, coupler, sim, neighbors, step_n; kwargs...)
        if rf
            recompute_forces = true
        end
    end
    return recompute_forces
end

"""
    NoCoupling()

Placeholder coupler that does nothing.
"""
struct NoCoupling end

apply_coupling!(sys, ::NoCoupling, sim, neighbors, step_n; kwargs...) = false

needs_virial(c::NoCoupling) = (truth = false, steps = Inf)

@doc raw"""
    ImmediateThermostat(temperature)

The immediate velocity rescaling thermostat for controlling temperature.

Velocities are immediately rescaled to match a target temperature.
The scaling factor for the velocities each step is
```math
\lambda = \sqrt{\frac{T_0}{T}}
```

This thermostat should be used with caution as it can lead to simulation
artifacts.
"""
struct ImmediateThermostat{T}
    temperature::T
end

function apply_coupling!(sys, thermostat::ImmediateThermostat, sim, neighbors=nothing,
                         step_n::Integer=0; kwargs...)
    sys.velocities .*= sqrt(thermostat.temperature / temperature(sys))
    return false
end

needs_virial(c::ImmediateThermostat) = (truth = false, steps = Inf)

@doc raw"""
    VelocityRescaleThermostat(temperature, coupling_const; n_steps = 1, seed = 42)

The stochastic velocity rescaling thermostat. See:

[Bussi, Donadio & Parrinello (2007)](https://doi.org/10.1063/1.2408420)

In brief, acts like the [`BerendsenThermostat`](@ref) but adds an 
stochastic term, allowing correct sampling of isothermal ensembles.

Let `Δt` be the simulation timestep, `Nf` the kinetic DOFs used to calculate the instantaneous
temperature of the system. Then, ``K = \frac{1}{2} \cdot \sum m \cdot v^2`` is the current
kinetic energy, and ``K̄ = \frac{1}{2} Nf k_B T_0`` is the target kinetic energy for
a reference temperature `T_0`.

Define ``c = e^{-Δt/τ}``. Draw ``R \sim 𝒩(0,1)`` and ``S \sim \chi^{2}_{Nf-1}``. Then

```math
\lambda^2 = c + (1-c) \cdot \frac{\bar K}{N_f K} \cdot (R^2 + S)\;+\;
            2 \cdot \sqrt{c(1-c) \frac{\bar K}{N_f K}} \cdot R,
\qquad v' = \lambda\,v .
```

"""
struct VelocityRescaleThermostat{T, C, N, S}
    temperature::T
    coupling_const::C
    n_steps::N
    seed::S
end

function VelocityRescaleThermostat(temperature, coupling_const; n_steps = 1, seed = 42)
    return VelocityRescaleThermostat(temperature, coupling_const, n_steps, seed)
end

function apply_coupling!(sys::System{<:Any, AT}, thermostat::VelocityRescaleThermostat, sim, neighbors, step_n;
                         n_threads::Integer=Threads.nthreads(), rng=Random.default_rng()) where {AT}

    if step_n % thermostat.n_steps != 0
        return false
    end

    # DOFs and current kinetic energy
    Nf  = sys.df
    Nf  > 0 || return false
    vels = from_device(sys.velocities)

    K = kinetic_energy(sys)
    ustrip(K) > 0 || return false

    # Target kinetic energy
    Kbar = 0.5 * Nf * sys.k * thermostat.temperature  # unit-consistent with K
    
    # Scalars
    dt  = sim.dt * thermostat.n_steps
    τ   = uconvert(unit(dt), thermostat.coupling_const)
    c   = exp(-dt / τ)              # e^{-Δt/τ}
    A   = Kbar / (Nf * K)           # = (K̄/(Nf*K)), dimensionless

    # Deterministic per-step RNG
    rrng = MersenneTwister(UInt(thermostat.seed) ⊻ UInt(step_n))

    # Draw R1 ~ N(0,1), and χ²_{Nf-1} via sum of squares
    R1   = randn(rrng)
    rsum = zero(eltype(R1))
    @inbounds for _ in 1:(Nf-1)
        x = randn(rrng); rsum += x*x
    end

    # λ² (Appendix A7). Guard tiny negatives from roundoff.
    lam2 = c + (1 - c) * A * (R1*R1 + rsum) + 2 * sqrt(c * (1 - c) * A) * R1
    lam2 = max(lam2, zero(lam2) + eps(Float64))
    λ    = sqrt(lam2)

    # Uniform rescale (preserves constraints; COM unchanged)
    @inbounds for i in eachindex(vels)
        vels[i] = λ * vels[i]
    end
    sys.velocities .= to_device(vels, AT)

    return false  # no force recompute needed
end

needs_virial(c::VelocityRescaleThermostat) = (truth = false, steps = Inf)

"""
    AndersenThermostat(temperature, coupling_const)

The Andersen thermostat for controlling temperature.

The velocity of each atom is randomly changed each time step with probability
`dt / coupling_const` to a velocity drawn from the Maxwell-Boltzmann distribution.
See [Andersen 1980](https://doi.org/10.1063/1.439486).
"""
struct AndersenThermostat{T, C}
    temperature::T
    coupling_const::C
end

function apply_coupling!(sys::System, thermostat::AndersenThermostat, sim,
                         neighbors=nothing, step_n::Integer=0;
                         n_threads::Integer=Threads.nthreads(),
                         rng=Random.default_rng())
    for i in eachindex(sys)
        if rand(rng) < (sim.dt / thermostat.coupling_const)
            sys.velocities[i] = random_velocity(mass(sys.atoms[i]), thermostat.temperature, sys.k;
                                                dims=AtomsBase.n_dimensions(sys), rng=rng)
        end
    end
    return false
end

function apply_coupling!(sys::System{<:Any, AT, T}, thermostat::AndersenThermostat, sim,
                         neighbors=nothing, step_n::Integer=0;
                         n_threads::Integer=Threads.nthreads(),
                         rng=Random.default_rng()) where {AT <: AbstractGPUArray, T}
    atoms_to_bump = T.(rand(rng, length(sys)) .< (sim.dt / thermostat.coupling_const))
    atoms_to_leave = one(T) .- atoms_to_bump
    atoms_to_bump_dev = to_device(atoms_to_bump, AT)
    atoms_to_leave_dev = to_device(atoms_to_leave, AT)
    vs = random_velocities(sys, thermostat.temperature; rng=rng)
    sys.velocities .= sys.velocities .* atoms_to_leave_dev .+ vs .* atoms_to_bump_dev
    return false
end

needs_virial(c::AndersenThermostat) = (truth = false, steps = Inf)

@doc raw"""
    BerendsenThermostat(temperature, coupling_const)

The Berendsen thermostat for controlling temperature.

The scaling factor for the velocities each step is
```math
\lambda^2 = 1 + \frac{\delta t}{\tau} \left( \frac{T_0}{T} - 1 \right)
```

This thermostat should be used with caution as it can lead to simulation
artifacts.
"""
struct BerendsenThermostat{T, C}
    temperature::T
    coupling_const::C
end

function apply_coupling!(sys, thermostat::BerendsenThermostat, sim, neighbors=nothing,
                         step_n::Integer=0; kwargs...)
    λ2 = 1 + (sim.dt / thermostat.coupling_const) * ((thermostat.temperature / temperature(sys)) - 1)
    sys.velocities .*= sqrt(λ2)
    return false
end

needs_virial(c::BerendsenThermostat) = (truth = false, steps = Inf)

@doc raw"""
    BerendsenBarostat(pressure, coupling_const; 
                      coupling_type=:isotropic,
                      compressibility=4.6e-5u"bar^-1",
                      max_scale_frac=0.1, n_steps=1)

The Berendsen barostat for controlling pressure.

The scaling factor for the box every `n_steps` steps is
```math
\mu_{ij} = \delta_{ij} - \frac{\Delta t}{3\, \tau_p} \kappa_{ij} \left(P_{0ij} - P_{ij}(t) \right).
```
with the fractional change limited to `max_scale_frac`.

The scaling factor ``\mu`` is a matrix and ``\delta`` represents a Kronecker delta,
allowing for non-isotropic pressure control. Available options are `:isotropic`,
`:semiisotropic` and `:anisotropic`.

This barostat should be used with caution as it known not to properly sample
isobaric ensembles and therefore can lead to simulation artifacts.
"""
struct BerendsenBarostat{P, C, S, IC, T}
    pressure::P
    coupling_const::C
    coupling_type::S
    compressibility::IC
    max_scale_frac::T
    n_steps::Int
end

_isbar(x) = try
    x isa Number && unit(x) == NoUnits ? true : (uconvert(u"bar", x); true)
catch
    false
end

_isibar(x) = try
    x isa Number && unit(x) == NoUnits ? true : (uconvert(u"bar^-1", x); true)
catch
    false
end

function BerendsenBarostat(pressure::Union{PT, AbstractArray{PT}}, coupling_const;
                           coupling_type=:isotropic, compressibility=4.6e-5u"bar^-1",
                           max_scale_frac=0.1, n_steps=1) where {PT}

    if !(coupling_type ∈ (:isotropic, :semiisotropic, :anisotropic))
        throw(ArgumentError(ArgumentError("coupling_type must be :isotropic, :semiisotropic, or :anisotropic")))
    end

    if coupling_type == :isotropic

        if pressure isa AbstractArray
            throw(ArgumentError("isotropic: pressure must be a scalar"))
        end
        
        if compressibility isa AbstractArray
            throw(ArgumentError("isotropic: compressibility must be a scalar"))
        end
        
        if !_isbar(pressure)
            throw(ArgumentError("isotropic: pressure must have pressure units"))
        end
        
        if !_isibar(compressibility)
            throw(ArgumentError("isotropic: compressibility must have 1/pressure units"))
        end

        
        # Use the caller’s units, but convert internal scalars consistently
        P_units = unit(pressure)
        K_units = unit(compressibility)
        p = ustrip(uconvert(P_units, pressure))
        κs = ustrip(uconvert(K_units, compressibility))

        FT = typeof(p)

        P = SMatrix{3,3,FT}(p,0,0, 0,p,0, 0,0,p) .* P_units
        κ = SMatrix{3,3,FT}(κs,0,0, 0,κs,0, 0,0,κs) .* K_units

        return BerendsenBarostat(P, coupling_const, coupling_type, κ, FT(max_scale_frac), n_steps)
    end

    if coupling_type == :semiisotropic

        if !(pressure isa AbstractArray && length(pressure) == 2)
            throw(ArgumentError("semiisotropic: pressure must be a 2-vector (xy, z)"))
        end
        
        if !(compressibility isa AbstractArray && length(compressibility) == 2) 
            throw(ArgumentError("semiisotropic: compressibility must be a 2-vector (xy, z)"))
        end

        if !all(_isbar.(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end
        
        if !all(_isibar.(compressibility))
            throw(ArgumentError("semiisotropic: compressibility must have 1/pressure units"))
        end

        P_units = unit(pressure[1])
        K_units = unit(compressibility[1])

        p_xy = ustrip(uconvert(P_units, pressure[1]))
        p_z  = ustrip(uconvert(P_units, pressure[2]))
        κ_xy = ustrip(uconvert(K_units, compressibility[1]))
        κ_z  = ustrip(uconvert(K_units, compressibility[2]))

        FT = promote_type(typeof(p_xy), typeof(p_z))

        P = SMatrix{3,3,FT}(p_xy,0,0, 0,p_xy,0, 0,0,p_z) .* P_units
        κ = SMatrix{3,3,FT}(κ_xy,0,0, 0,κ_xy,0, 0,0,κ_z) .* K_units

        return BerendsenBarostat(P, coupling_const, coupling_type, κ, FT(max_scale_frac), n_steps)
    end
        
    if coupling_type == :anisotropic

        if !(pressure isa  AbstractArray && length(pressure) == 6)
            throw(ArgumentError("semiisotropic: pressure must be a 6-vector (x, y, z, xy/yx, xz/zx, yz/zy)"))
        end
        
        if !(compressibility isa  AbstractArray && length(pressure) == 6) 
            throw(ArgumentError("semiisotropic: compressibility must be a 6-vector (x, y, z, xy/yx, xz/zx, yz/zy)"))
        end
        
        if !all(_isbar.(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end
        
        if !all(_isibar.(compressibility))
            throw(ArgumentError("semiisotropic: compressibility must have 1/pressure units"))
        end


        P_units = unit(pressure[1])
        K_units = unit(compressibility[1])

        px  = ustrip(uconvert(P_units, pressure[1]))
        py  = ustrip(uconvert(P_units, pressure[2]))
        pz  = ustrip(uconvert(P_units, pressure[3]))
        pxy = ustrip(uconvert(P_units, pressure[4]))
        pxz = ustrip(uconvert(P_units, pressure[5]))
        pyz = ustrip(uconvert(P_units, pressure[6]))

        κx  = ustrip(uconvert(K_units, compressibility[1]))
        κy  = ustrip(uconvert(K_units, compressibility[2]))
        κz  = ustrip(uconvert(K_units, compressibility[3]))
        κxy = ustrip(uconvert(K_units, compressibility[4]))
        κxz = ustrip(uconvert(K_units, compressibility[5]))
        κyz = ustrip(uconvert(K_units, compressibility[6]))

        FT = promote_type(typeof(px), typeof(py), typeof(pz), typeof(pxy), typeof(pxz), typeof(pyz))

        P = SMatrix{3,3,FT}(px,  pxy, pxz,
                            pxy, py,  pyz,
                            pxz, pyz, pz) .* P_units

        κ = SMatrix{3,3,FT}(κx,  κxy, κxz,
                            κxy, κy,  κyz,
                            κxz, κyz, κz) .* K_units

        return BerendsenBarostat(P, coupling_const, coupling_type, κ, FT(max_scale_frac), n_steps)
    end
end

function apply_coupling!(sys, barostat::BerendsenBarostat{PT, CT, ST, ICT, FT}, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...) where {PT, CT, ST, ICT, FT} # Avoids uconvert calls in method
    if step_n % barostat.n_steps != 0; return false; end
    
    # pressure in barostat units
    P = pressure(sys, neighbors, step_n; n_threads=n_threads)

    D   = size(P,1)
    τp  = barostat.coupling_const
    dt  = sim.dt * barostat.n_steps
    μ   = Matrix{FT}(I, D, D)

    Pavg = tr(P)/FT(D)
    Pxy  = D == 3 ? (P[1,1] + P[2,2]) / FT(2) : Pavg

    
    if barostat.coupling_type == :isotropic
        for d in 1:D
            α = (barostat.compressibility[d,d] * dt) / (D*τp)
            s = 1 + α * (Pavg - barostat.pressure[d,d])
            μ[d,d] = clamp(s, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac)
        end

    elseif barostat.coupling_type == :semiisotropic
        @assert D == 3 "Cannot apply semi-isotropic in 2D."
        for d in 1:2
            α = (barostat.compressibility[d,d] * dt) / (FT(2)*τp)
            s = 1 + α * (Pxy - barostat.pressure[d,d])
            μ[d,d] = clamp(s, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac)
        end
        αz = (barostat.compressibility[3,3] * dt) / τp
        sz = 1 + αz * (P[3,3] - barostat.pressure[3,3])
        μ[3,3] = clamp(sz, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac)

    elseif barostat.coupling_type == :anisotropic
        # diagonals multiplicative
        for d in 1:D
            α = (barostat.compressibility[d,d] * dt) / τp
            s = 1 + α * (P[d,d] - barostat.pressure[d,d])
            μ[d,d] = clamp(s, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac)
        end
        # shear increments (dimensionless), then project to triclinic
        for i in 1:D, j in 1:D
            i == j && continue
            αij = (barostat.compressibility[i,j] * dt) / τp
            Δ   = αij * (P[i,j] - barostat.pressure[i,j])
            if isfinite(barostat.max_scale_frac)
                Δ = clamp(Δ, -barostat.max_scale_frac, barostat.max_scale_frac)
            end
            μ[i,j] = Δ
        end

    else
        error("Unsupported coupling_type=$(barostat.coupling_type)")
    end

    # Triclinic projector (move lower into upper, zero lower) before left-multiplying B' = μ B
    if D == 3
        μ[1,2] += μ[2,1]
        μ[1,3] += μ[3,1]
        μ[2,3] += μ[3,2]
        μ[2,1] = 0; μ[3,1] = 0; μ[3,2] = 0
    end

    scale_coords!(sys, SMatrix{D,D,FT}(μ))
    return true
end

needs_virial(c::BerendsenBarostat) = (truth = true, steps = c.n_steps)

@doc raw"""
    CRescaleBarostat(pressure, coupling_const; 
                      coupling_type=:isotropic,
                      compressibility=4.6e-5u"bar^-1",
                      max_scale_frac=0.1, n_steps=1)

The Stochastic Cell Rescale barostat, see:

[Bernetti & Bussi (2020)] (https://doi.org/10.1063/5.0020514)

and 

[Del Tatto, Raiteri, Bernetti & Bussi (2022)] (https://doi.org/10.3390/app12031139)

I brief, this is an extension of the Berendsen barostat that includes an 
stochastic term to the scaling matrix. This allows to properly
sample isobaric ensembles.

```math
\mu = \rm{exp}\left[ \frac{-\kappa_T \cdot \Delta t}{\tau_P} \cdot (P(t) - P_0) + \sqrt{\frac{2 \cdot k_BT \cdot \kappa_T \cdot dt}{V(t) \cdot \tau_P}} \cdot dW \right]
```

Where ``\kappa_T`` is the isothermal compressibility, ``\tau_P`` is the barostat coupling constant,
and ``\rm{dW}`` represents a Wiener process.

The scaling factor ``\mu`` is a matrix, allowing for non-isotropic
pressure control. Available options are `:isotropic`, `:semiisotropic` and `:anisotropic`

"""
struct CRescaleBarostat{P, C, S, IC, T}
    pressure::P
    coupling_const::C
    coupling_type::S
    compressibility::IC
    max_scale_frac::T
    seed::Int
    n_steps::Int
end

function CRescaleBarostat(pressure::Union{PT, AbstractArray{PT}}, coupling_const;
                           coupling_type=:isotropic, compressibility=4.6e-5u"bar^-1",
                           max_scale_frac=0.1, seed = 42, n_steps=1) where {PT}

    if !(coupling_type ∈ (:isotropic, :semiisotropic, :anisotropic))
        throw(ArgumentError(ArgumentError("coupling_type must be :isotropic, :semiisotropic, or :anisotropic")))
    end

    if coupling_type == :isotropic

        if pressure isa AbstractArray
            throw(ArgumentError("isotropic: pressure must be a scalar"))
        end
        
        if compressibility isa AbstractArray
            throw(ArgumentError("isotropic: compressibility must be a scalar"))
        end
        
        if !_isbar(pressure)
            throw(ArgumentError("isotropic: pressure must have pressure units"))
        end
        
        if !_isibar(compressibility)
            throw(ArgumentError("isotropic: compressibility must have 1/pressure units"))
        end

        
        # Use the caller’s units, but convert internal scalars consistently
        P_units = unit(pressure)
        K_units = unit(compressibility)
        p = ustrip(uconvert(P_units, pressure))
        κs = ustrip(uconvert(K_units, compressibility))

        FT = typeof(p)

        P = SMatrix{3,3,FT}(p,0,0, 0,p,0, 0,0,p) .* P_units
        κ = SMatrix{3,3,FT}(κs,0,0, 0,κs,0, 0,0,κs) .* K_units

        return CRescaleBarostat(P, coupling_const, coupling_type, κ, FT(max_scale_frac), seed, n_steps)
    end

    if coupling_type == :semiisotropic

        if !(pressure isa AbstractArray && length(pressure) == 2)
            throw(ArgumentError("semiisotropic: pressure must be a 2-vector (xy, z)"))
        end
        
        if !(compressibility isa AbstractArray && length(compressibility) == 2) 
            throw(ArgumentError("semiisotropic: compressibility must be a 2-vector (xy, z)"))
        end

        if !all(_isbar.(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end
        
        if !all(_isibar.(compressibility))
            throw(ArgumentError("semiisotropic: compressibility must have 1/pressure units"))
        end

        P_units = unit(pressure[1])
        K_units = unit(compressibility[1])

        p_xy = ustrip(uconvert(P_units, pressure[1]))
        p_z  = ustrip(uconvert(P_units, pressure[2]))
        κ_xy = ustrip(uconvert(K_units, compressibility[1]))
        κ_z  = ustrip(uconvert(K_units, compressibility[2]))

        FT = promote_type(typeof(p_xy), typeof(p_z))

        P = SMatrix{3,3,FT}(p_xy,0,0, 0,p_xy,0, 0,0,p_z) .* P_units
        κ = SMatrix{3,3,FT}(κ_xy,0,0, 0,κ_xy,0, 0,0,κ_z) .* K_units

        return CRescaleBarostat(P, coupling_const, coupling_type, κ, FT(max_scale_frac), seed, n_steps)
    end
        
    if coupling_type == :anisotropic

        if !(pressure isa  AbstractArray && length(pressure) == 6)
            throw(ArgumentError("semiisotropic: pressure must be a 6-vector (x, y, z, xy/yx, xz/zx, yz/zy)"))
        end
        
        if !(compressibility isa  AbstractArray && length(pressure) == 6) 
            throw(ArgumentError("semiisotropic: compressibility must be a 6-vector (x, y, z, xy/yx, xz/zx, yz/zy)"))
        end
        
        if !all(_isbar.(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end
        
        if !all(_isibar.(compressibility))
            throw(ArgumentError("semiisotropic: compressibility must have 1/pressure units"))
        end


        P_units = unit(pressure[1])
        K_units = unit(compressibility[1])

        px  = ustrip(uconvert(P_units, pressure[1]))
        py  = ustrip(uconvert(P_units, pressure[2]))
        pz  = ustrip(uconvert(P_units, pressure[3]))
        pxy = ustrip(uconvert(P_units, pressure[4]))
        pxz = ustrip(uconvert(P_units, pressure[5]))
        pyz = ustrip(uconvert(P_units, pressure[6]))

        κx  = ustrip(uconvert(K_units, compressibility[1]))
        κy  = ustrip(uconvert(K_units, compressibility[2]))
        κz  = ustrip(uconvert(K_units, compressibility[3]))
        κxy = ustrip(uconvert(K_units, compressibility[4]))
        κxz = ustrip(uconvert(K_units, compressibility[5]))
        κyz = ustrip(uconvert(K_units, compressibility[6]))

        FT = promote_type(typeof(px), typeof(py), typeof(pz), typeof(pxy), typeof(pxz), typeof(pyz))

        P = SMatrix{3,3,FT}(px, pxy, pxz,
                            pxy, py,  pyz,
                            pxz, pyz, pz) .* P_units

        κ = SMatrix{3,3,FT}(κx, κxy, κxz,
                            κxy, κy,  κyz,
                            κxz, κyz, κz) .* K_units

        return CRescaleBarostat(P, coupling_const, coupling_type, κ, FT(max_scale_frac), seed, n_steps)
    end
end

function apply_coupling!(sys::System{D, AT}, barostat::CRescaleBarostat{PT, CT, ST, ICT, FT}, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...) where {D, AT, PT, CT, ST, ICT, FT}
    if step_n % barostat.n_steps != 0
        return false
    end

    # Pressure tensor in barostat units
    P = pressure(sys, neighbors, step_n; n_threads=n_threads)

    # Thermo factors
    V         = volume(sys.boundary)
    τp, dt    = barostat.coupling_const, sim.dt * barostat.n_steps
    kT_energy = Unitful.k * temperature(sys)
    kT_pv     = uconvert(unit(P[1,1]) * unit(V), kT_energy)

    scalarP(P) = (P[1,1] + P[2,2] + P[3,3]) / D
    xyP(P)     = (P[1,1] + P[2,2]) / 2

    # RNG
    rng  = MersenneTwister(UInt(barostat.seed) ⊻ UInt(step_n))
    ξ()  = FT(randn(rng))

    μ = zeros(FT, D, D)

    if barostat.coupling_type == :isotropic
        P̄ = scalarP(P); g = ξ()
        for d in 1:D
            α = (barostat.compressibility[d,d] * dt) / τp
            det_term = -α * (barostat.pressure[d,d] - P̄) / D
            stoch    = sqrt(2*kT_pv*α / V) * (g / D)
            s = exp(det_term + stoch)
            if isfinite(barostat.max_scale_frac); s = clamp(s, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac); end
            μ[d,d] = s
        end

    elseif barostat.coupling_type == :semiisotropic
        @assert size(P,1) == 3
        Pxy = xyP(P); gxy, gz = ξ(), ξ()
        for d in 1:2
            α = (barostat.compressibility[d,d] * dt) / τp
            det_term = -α * (barostat.pressure[d,d] - Pxy) / D
            stoch    = sqrt((D-1) * 2*kT_pv*α / (V*D)) * (gxy / (D-1))
            s = exp(det_term + stoch)
            if isfinite(barostat.max_scale_frac); s = clamp(s, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac); end
            μ[d,d] = s
        end
        αz = (barostat.compressibility[3,3] * dt) / τp
        det_term_z = -αz * (barostat.pressure[3,3] - P[3,3]) / D
        stoch_z    = sqrt(2*kT_pv*αz / (V*D)) * gz
        sz = exp(det_term_z + stoch_z)
        if isfinite(barostat.max_scale_frac); sz = clamp(sz, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac); end
        μ[3,3] = sz

    elseif barostat.coupling_type == :anisotropic
        # Diagonals (exp map)
        gx, gy, gz = ξ(), ξ(), ξ()
        for d in 1:3
            α = (barostat.compressibility[d,d] * dt) / τp
            det_term = -α * (barostat.pressure[d,d] - P[d,d]) / D
            stoch    = sqrt(2*kT_pv*α / (V*D)) * (d==1 ? gx : d==2 ? gy : gz)
            s = exp(det_term + stoch)
            if isfinite(barostat.max_scale_frac); s = clamp(s, 1 - barostat.max_scale_frac, 1 + barostat.max_scale_frac); end
            μ[d,d] = s
        end
        # Shear increments (lower triangle)
        for i in 2:3, j in 1:i-1
            αij = (barostat.compressibility[i,j] * dt) / τp
            gij = ξ()
            det_ij   = -αij * (barostat.pressure[i,j] - P[i,j]) / D
            stoch_ij = sqrt(2*kT_pv*abs(αij) / (V*D)) * gij
            Δ = det_ij + stoch_ij
            if isfinite(barostat.max_scale_frac); Δ = clamp(Δ, -barostat.max_scale_frac, barostat.max_scale_frac); end
            μ[i,j] = Δ
        end

    else
        error("Unsupported coupling_type=$(barostat.coupling_type)")
    end

    # Triclinic projector (move lower into upper, zero lower) before left-multiplying B' = μ B
    if D == 3
        μ[1,2] += μ[2,1]
        μ[1,3] += μ[3,1]
        μ[2,3] += μ[3,2]
        μ[2,1] = 0; μ[3,1] = 0; μ[3,2] = 0
    end

    scale_coords!(sys, SMatrix{3,3,FT}(μ); scale_velocities = true)
    return true
end

needs_virial(c::CRescaleBarostat) = (truth = true, steps = c.n_steps)

@doc raw"""
    MonteCarloBarostat(pressure, temperature, boundary; coupling_type = :isotropic,
                       n_steps=30, n_iterations=1,
                       scale_factor=0.01, scale_increment=1.1, max_volume_frac=0.3,
                       trial_find_neighbors=false)

The Monte Carlo barostat for controlling pressure.

See [Chow and Ferguson 1995](https://doi.org/10.1016/0010-4655(95)00059-O),
[Åqvist et al. 2004](https://doi.org/10.1016/j.cplett.2003.12.039) and the OpenMM
source code.
At regular intervals a Monte Carlo step is attempted by scaling the coordinates and
the bounding box by a randomly chosen amount.
The step is accepted or rejected based on
```math
\Delta G = \Delta E + \Delta W - N k_B T \ln \left( \frac{V + \Delta V}{V} \right)
```
where `ΔE` is the change in potential energy, `ΔV` is the change in volume, `N` is the 
number of molecules in the system, `T` is the equilibrium temperature and `V` is the system
volume. `ΔW` is the work done by scaling the simulation box and its specific form 
changes depending on the type of scaling applied. In general and in the absence of shear stress:

```math
\Delta W = (V + \Delta V) \cdot \sum w_i \cdot  P_{i,i} \cdot \ln \left ( {\frac{V + \Delta V}{V}} \right )
```

where `w_i` is the proportional scaling along a specific box axis.

If `ΔG ≤ 0` the step is always accepted, if `ΔG > 0` the step is accepted with
probability `exp(-ΔG/kT)`.

The scale factor is modified over time to maintain an acceptance rate of around half.
If the topology of the system is set then molecules are moved as a unit so properties
such as bond lengths do not change.

The barostat assumes that the simulation is being run at a constant temperature but
does not actively control the temperature.
It should be used alongside a temperature coupling method such as the [`Langevin`](@ref)
simulator or [`AndersenThermostat`](@ref) coupling.
The neighbor list is not updated when making trial moves or after accepted moves.
Note that the barostat can change the bounding box of the system. Does not currently 
work with shear stresses, the anisotropic variant only applies independent linear scaling
of the box vectors. If shear deformation is required the [`BerendsenBarostat`](@ref) or,
preferrably, the [`CRescaleBarostat`](@ref) should be used instead. 
"""
mutable struct MonteCarloBarostat{T, P, K, V}
    pressure::P
    temperature::K
    coupling_type::Symbol
    n_steps::Int
    n_iterations::Int
    volume_scale::V
    scale_increment::T
    max_volume_frac::T
    trial_find_neighbors::Bool
    n_attempted::Int
    n_accepted::Int
end

function MonteCarloBarostat(pressure::Union{PT, AbstractArray{PT}}, temp, boundary::AbstractBoundary{<:Any, FT}; coupling_type::Symbol = :isotropic,
                            n_steps=30, n_iterations=1, scale_factor=0.01,
                            scale_increment=1.1, max_volume_frac=0.3, trial_find_neighbors=false) where {PT, FT}
    
    volume_scale = volume(boundary) * FT(scale_factor)

    if !(coupling_type ∈ (:isotropic, :semiisotropic, :anisotropic))
        throw(ArgumentError(ArgumentError("coupling_type must be :isotropic, :semiisotropic, or :anisotropic")))
    end

    if coupling_type == :isotropic

        if pressure isa AbstractArray
            throw(ArgumentError("isotropic: pressure must be a scalar"))
        end
        
        if !_isbar(pressure)
            throw(ArgumentError("isotropic: pressure must have pressure units"))
        end
        
        # Use the caller’s units, but convert internal scalars consistently
        P_units = unit(pressure)
        p = ustrip(uconvert(P_units, pressure))

        P = SMatrix{3,3,FT}(p,0,0, 0,p,0, 0,0,p) .* P_units

        return MonteCarloBarostat(P, temp, coupling_type,
                                  n_steps, n_iterations, 
                                  volume_scale, scale_increment,
                                  max_volume_frac,
                                  trial_find_neighbors, 0, 0)

    end

    if coupling_type == :semiisotropic

        if !(pressure isa AbstractArray && length(pressure) == 2)
            throw(ArgumentError("semiisotropic: pressure must be a 2-vector (xy, z)"))
        end

        if !all(_isbar.(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end

        P_units = unit(pressure[1])

        p_xy = ustrip(uconvert(P_units, pressure[1]))
        p_z  = ustrip(uconvert(P_units, pressure[2]))

        P = SMatrix{3,3,FT}(p_xy,0,0, 0,p_xy,0, 0,0,p_z) .* P_units

        return MonteCarloBarostat(P, temp, coupling_type,
                                  n_steps, n_iterations, 
                                  volume_scale, scale_increment,
                                  max_volume_frac,
                                  trial_find_neighbors, 0, 0)

    end
        
    if coupling_type == :anisotropic

        if !(pressure isa  AbstractArray && length(pressure) == 6)
            throw(ArgumentError("semiisotropic: pressure must be a 6-vector (x, y, z, xy/yx, xz/zx, yz/zy)"))
        end
        
        if !all(_isbar.(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end

        P_units = unit(pressure[1])

        px  = ustrip(uconvert(P_units, pressure[1]))
        py  = ustrip(uconvert(P_units, pressure[2]))
        pz  = ustrip(uconvert(P_units, pressure[3]))
        pxy = ustrip(uconvert(P_units, pressure[4]))
        pxz = ustrip(uconvert(P_units, pressure[5]))
        pyz = ustrip(uconvert(P_units, pressure[6]))

        P = SMatrix{3,3,FT}(px, pxy, pxz,
                            pxy, py,  pyz,
                            pxz, pyz, pz) .* P_units

        return MonteCarloBarostat(P, temp, coupling_type,
                                  n_steps, n_iterations, 
                                  volume_scale, scale_increment,
                                  max_volume_frac,
                                  trial_find_neighbors, 0, 0)

    end
end

function apply_coupling!(sys::System{D, AT, T}, barostat::MonteCarloBarostat, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads(),
                         rng=Random.default_rng()) where {D, AT, T}

    if !iszero(step_n % barostat.n_steps)
        return false
    end

    Pxx = barostat.pressure[1,1]; Pyy = barostat.pressure[2,2]; Pzz = barostat.pressure[3,3]

    kT = energy_remove_mol(sys.k * barostat.temperature)
    n_molecules = isnothing(sys.topology) ? length(sys) : length(sys.topology.molecule_atom_counts)
    recompute_forces = false
    old_coords = similar(sys.coords)

    if barostat.coupling_type == :isotropic

        for attempt_n in 1:barostat.n_iterations
        
            E  = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
            V  = volume(sys.boundary)
            dV = barostat.volume_scale * (2 * rand(rng, T) - 1)

            v_scale = (V + dV)/V

            l_scale = cbrt(v_scale)

            old_coords  .= sys.coords
            old_boundary = sys.boundary
            scale_coords!(sys, SMatrix{D, D, T}([l_scale 0 0;
                                                 0 l_scale 0;
                                                 0 0 l_scale]))

            if barostat.trial_find_neighbors
                neighbors_trial = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, true;
                                                 n_threads=n_threads)
            else
                # Assume neighbors are unchanged by the change in coordinates
                # This may not be valid for larger changes
                neighbors_trial = neighbors
            end

            E_trial = potential_energy(sys, neighbors_trial, step_n; n_threads=n_threads)
            dE = energy_remove_mol(E_trial - E)

            dW = dE + uconvert(unit(dE), (1/3)*tr(barostat.pressure) * dV) - n_molecules * kT * log(v_scale)

            if dW <= zero(dW) || rand(rng, T) < exp(-dW / kT)
                recompute_forces = true
                barostat.n_accepted += 1
            else
                sys.coords .= old_coords
                sys.boundary = old_boundary
            end
            barostat.n_attempted += 1

            # Modify size of volume change to keep accept/reject ratio roughly equal
            if barostat.n_attempted >= 10
                if barostat.n_accepted < 0.25 * barostat.n_attempted
                    barostat.volume_scale /= barostat.scale_increment
                    barostat.n_attempted = 0
                    barostat.n_accepted = 0
                elseif barostat.n_accepted > 0.75 * barostat.n_attempted
                    barostat.volume_scale = min(barostat.volume_scale * barostat.scale_increment,
                                                V * barostat.max_volume_frac)
                    barostat.n_attempted = 0
                    barostat.n_accepted = 0
                end
            end
        end

    
    elseif barostat.coupling_type == :semiisotropic

        for attempt_n in 1:barostat.n_iterations
        
            E         = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
            V         = volume(sys.boundary)
            dV        = barostat.volume_scale * (2 * rand(rng, T) - 1)
            V_plus_dV = V + dV

            v_scale = V_plus_dV/V

            w1, w2 = rand(rng, T), rand(rng, T)
            
            s = w1+w2
            
            w1 = w1/s
            w2 = w2/s

            l_scale_xy = v_scale^w1
            l_scale_z  = v_scale^w2

            old_coords  .= sys.coords
            old_boundary = sys.boundary
            scale_coords!(sys, SMatrix{D, D, T}([l_scale_xy 0 0;
                                                 0 l_scale_xy 0;
                                                 0 0 l_scale_z]))

            if barostat.trial_find_neighbors
                neighbors_trial = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, true;
                                                 n_threads=n_threads)
            else
                # Assume neighbors are unchanged by the change in coordinates
                # This may not be valid for larger changes
                neighbors_trial = neighbors
            end

            E_trial = potential_energy(sys, neighbors_trial, step_n; n_threads=n_threads)
            dE = energy_remove_mol(E_trial - E)

            work = ((w1/2)*Pxx + (w1/2)*Pyy + w2*Pzz) * V_plus_dV * log(v_scale)
            
            dW = dE + uconvert(unit(dE), work) - n_molecules * kT * log(v_scale)

            if dW <= zero(dW) || rand(rng, T) < exp(-dW / kT)
                recompute_forces = true
                barostat.n_accepted += 1
            else
                sys.coords .= old_coords
                sys.boundary = old_boundary
            end
            barostat.n_attempted += 1

            # Modify size of volume change to keep accept/reject ratio roughly equal
            if barostat.n_attempted >= 10
                if barostat.n_accepted < 0.25 * barostat.n_attempted
                    barostat.volume_scale /= barostat.scale_increment
                    barostat.n_attempted = 0
                    barostat.n_accepted = 0
                elseif barostat.n_accepted > 0.75 * barostat.n_attempted
                    barostat.volume_scale = min(barostat.volume_scale * barostat.scale_increment,
                                                V * barostat.max_volume_frac)
                    barostat.n_attempted = 0
                    barostat.n_accepted = 0
                end
            end
        end
        
    
    elseif barostat.coupling_type == :anisotropic

        for attempt_n in 1:barostat.n_iterations
        
            E         = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
            V         = volume(sys.boundary)
            dV        = barostat.volume_scale * (2 * rand(rng, T) - 1)
            V_plus_dV = V + dV

            v_scale = V_plus_dV / V

            w1, w2, w3 = rand(rng, T), rand(rng, T), rand(rng, T)
            
            s = w1+w2+w3
            
            w1 = w1/s
            w2 = w2/s
            w3 = w3/s

            l_scale_x = v_scale^w1
            l_scale_y = v_scale^w2
            l_scale_z = v_scale^w3

            old_coords  .= sys.coords
            old_boundary = sys.boundary
            scale_coords!(sys, SMatrix{D, D, T}([l_scale_x 0 0;
                                                 0 l_scale_y 0;
                                                 0 0 l_scale_z]))

            if barostat.trial_find_neighbors
                neighbors_trial = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, true;
                                                 n_threads=n_threads)
            else
                # Assume neighbors are unchanged by the change in coordinates
                # This may not be valid for larger changes
                neighbors_trial = neighbors
            end

            E_trial = potential_energy(sys, neighbors_trial, step_n; n_threads=n_threads)
            dE = energy_remove_mol(E_trial - E)

            work = (w1*Pxx + w2*Pyy + w3*Pzz) * V_plus_dV * log(v_scale)
            dW = dE + uconvert(unit(dE), work) - n_molecules * kT * log(v_scale)

            if dW <= zero(dW) || rand(rng, T) < exp(-dW / kT)
                recompute_forces = true
                barostat.n_accepted += 1
            else
                sys.coords .= old_coords
                sys.boundary = old_boundary
            end
            barostat.n_attempted += 1

            # Modify size of volume change to keep accept/reject ratio roughly equal
            if barostat.n_attempted >= 10
                if barostat.n_accepted < 0.25 * barostat.n_attempted
                    barostat.volume_scale /= barostat.scale_increment
                    barostat.n_attempted = 0
                    barostat.n_accepted = 0
                elseif barostat.n_accepted > 0.75 * barostat.n_attempted
                    barostat.volume_scale = min(barostat.volume_scale * barostat.scale_increment,
                                                V * barostat.max_volume_frac)
                    barostat.n_attempted = 0
                    barostat.n_accepted = 0
                end
            end
        end
    
    end

    return recompute_forces
                    
end

needs_virial(c::MonteCarloBarostat) = (truth = false, steps = Inf)