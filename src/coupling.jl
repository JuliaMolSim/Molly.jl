# Temperature and pressure coupling methods

export
    apply_coupling!,
    NoCoupling,
    ImmediateThermostat,
    AndersenThermostat,
    BerendsenThermostat,
    BerendsenBarostat,
    MonteCarloBarostat,
    MonteCarloAnisotropicBarostat,
    MonteCarloMembraneBarostat

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

@doc raw"""
    BerendsenBarostat(pressure, coupling_const; 
                      coupling_type=:isotropic,
                      compressibility=4.6e-5u"bar^-1",
                      max_scale_frac=0.1, n_steps=1)

The Berendsen barostat for controlling pressure.

The scaling factor for the box every `n_steps` steps is
```math
\mu = 1 - \frac{\kappa_T \delta t}{3 \tau} ( P_0 - P )
```
with the fractional change limited to `max_scale_frac`.

The scaling factor ``\mu`` is a matrix, allowing for non-isotropic
pressure control. Available options are :isotropic, :semiisotropic and :anisotropic

This barostat should be used with caution as it can lead to simulation
artifacts.
"""
struct BerendsenBarostat{P, C, S, IC, T}
    pressure::P
    coupling_const::C
    coupling_type::S
    compressibility::IC
    max_scale_frac::T
    n_steps::Int
end

_isbar(x) = try;  uconvert(u"bar", x);      true; catch; false; end
_isibar(x)= try;  uconvert(u"bar^-1", x);   true; catch; false; end

function BerendsenBarostat(pressure::Union{PT, SVector{D, PT}}, coupling_const;
                           coupling_type=:isotropic, compressibility=4.6e-5u"bar^-1",
                           max_scale_frac=0.1, n_steps=1) where {PT, D}

    if !(coupling_type ∈ (:isotropic, :semiisotropic, :anisotropic))
        throw(ArgumentError(ArgumentError("coupling_type must be :isotropic, :semiisotropic, or :anisotropic")))
    end

    if coupling_type == :isotropic

        if pressure isa SVector
            throw(ArgumentError("isotropic: pressure must be a scalar"))
        end
        
        if compressibility isa SVector 
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

        return BerendsenBarostat(P, coupling_const, :isotropic, κ, FT(max_scale_frac), n_steps)
    end

    if coupling_type == :semiisotropic

        if !(pressure isa SVector{2})
            throw(ArgumentError("semiisotropic: pressure must be a 2-vector (xy, z)"))
        end
        
        if !(compressibility isa SVector{2}) 
            throw(ArgumentError("semiisotropic: compressibility must be a 2-vector (xy, z)"))
        end
        
        if !all(_isbar(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end
        
        if !all(_isibar(compressibility))
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

        return BerendsenBarostat(P, coupling_const, :semiisotropic, κ, FT(max_scale_frac), n_steps)
    end
        
    if coupling_type == :anisotropic

        if !(pressure isa SVector{6})
            throw(ArgumentError("semiisotropic: pressure must be a 6-vector (x, y, z, xy/yx, xz/zx, yz/zy)"))
        end
        
        if !(compressibility isa SVector{6}) 
            throw(ArgumentError("semiisotropic: compressibility must be a 6-vector (x, y, z, xy/yx, xz/zx, yz/zy)"))
        end
        
        if !all(_isbar(pressure))
            throw(ArgumentError("semiisotropic: pressure must have pressure units"))
        end
        
        if !all(_isibar(compressibility))
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

        return BerendsenBarostat(P, coupling_const, :anisotropic, κ, FT(max_scale_frac), n_steps)
    end
end

function apply_coupling!(sys, barostat::BerendsenBarostat, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...)
    if step_n % barostat.n_steps != 0; return false; end
    FT = typeof(ustrip(sim.dt))

    # pressure in barostat units
    Pmeas = pressure(sys, neighbors, step_n; n_threads=n_threads)
    tgt_unit = unit(barostat.pressure[1,1])
    P = uconvert.(tgt_unit, Pmeas)

    D   = size(P,1)
    τp  = barostat.coupling_const
    dt  = sim.dt
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
        if D == 3
            # move upper triangle into lower (first-order) and zero upper
            μ[2,1] += μ[1,2]
            μ[3,1] += μ[1,3]
            μ[3,2] += μ[2,3]
            μ[1,2] = 0; μ[1,3] = 0; μ[2,3] = 0
        end
    else
        error("Unsupported coupling_type=$(barostat.coupling_type)")
    end

    scale_coords!(sys, SMatrix{D,D,FT}(μ))
    return true
end

@doc raw"""
    MonteCarloBarostat(pressure, temperature, boundary; n_steps=30, n_iterations=1,
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
\Delta W = \Delta E + P \Delta V - N k_B T \ln \left( \frac{V + \Delta V}{V} \right)
```
where `ΔE` is the change in potential energy, `P` is the equilibrium pressure, `ΔV` is
the change in volume, `N` is the number of molecules in the system, `T` is the equilibrium
temperature and `V` is the system volume.
If `ΔW ≤ 0` the step is always accepted, if `ΔW > 0` the step is accepted with probability
`exp(-ΔW/kT)`.

The scale factor is modified over time to maintain an acceptance rate of around half.
If the topology of the system is set then molecules are moved as a unit so properties
such as bond lengths do not change.

The barostat assumes that the simulation is being run at a constant temperature but
does not actively control the temperature.
It should be used alongside a temperature coupling method such as the [`Langevin`](@ref)
simulator or [`AndersenThermostat`](@ref) coupling.
The neighbor list is not updated when making trial moves or after accepted moves.
Note that the barostat can change the bounding box of the system.
"""
mutable struct MonteCarloBarostat{T, P, K, V}
    pressure::P
    temperature::K
    n_steps::Int
    n_iterations::Int
    volume_scale::V
    scale_increment::T
    max_volume_frac::T
    trial_find_neighbors::Bool
    n_attempted::Int
    n_accepted::Int
end

function MonteCarloBarostat(P, temp, boundary; n_steps=30, n_iterations=1, scale_factor=0.01,
                            scale_increment=1.1, max_volume_frac=0.3, trial_find_neighbors=false)
    T = float_type(boundary)
    volume_scale = volume(boundary) * T(scale_factor)
    return MonteCarloBarostat(P, temp, n_steps, n_iterations, volume_scale, T(scale_increment),
                              T(max_volume_frac), trial_find_neighbors, 0, 0)
end

function apply_coupling!(sys::System{D, AT, T}, barostat::MonteCarloBarostat, sim, neighbors=nothing,
                         step_n::Integer=0; n_threads::Integer=Threads.nthreads(),
                         rng=Random.default_rng()) where {D, AT, T}
    if !iszero(step_n % barostat.n_steps)
        return false
    end

    kT = energy_remove_mol(sys.k * barostat.temperature)
    n_molecules = isnothing(sys.topology) ? length(sys) : length(sys.topology.molecule_atom_counts)
    recompute_forces = false
    old_coords = similar(sys.coords)

    for attempt_n in 1:barostat.n_iterations
        E = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
        V = volume(sys.boundary)
        dV = barostat.volume_scale * (2 * rand(rng, T) - 1)
        v_scale = (V + dV) / V
        l_scale = (D == 2 ? sqrt(v_scale) : cbrt(v_scale))
        old_coords .= sys.coords
        old_boundary = sys.boundary
        scale_coords!(sys, l_scale)

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
        dW = dE + uconvert(unit(dE), barostat.pressure * dV) - n_molecules * kT * log(v_scale)
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
    return recompute_forces
end

@doc raw"""
    MonteCarloAnisotropicBarostat(pressure, temperature, boundary; n_steps=30,
                       n_iterations=1, scale_factor=0.01, scale_increment=1.1,
                       max_volume_frac=0.3, trial_find_neighbors=false)

The Monte Carlo anisotropic barostat for controlling pressure.

For 3D systems, `pressure` is a `SVector` of length 3 with components pressX, pressY,
and pressZ representing the target pressure in each axis.
For 2D systems, `pressure` is a `SVector` of length 2 with components pressX and pressY.
To keep an axis fixed, set the corresponding pressure to `nothing`.

See [Chow and Ferguson 1995](https://doi.org/10.1016/0010-4655(95)00059-O),
[Åqvist et al. 2004](https://doi.org/10.1016/j.cplett.2003.12.039) and the OpenMM
source code.
At regular intervals a Monte Carlo step is attempted by scaling the coordinates and
the bounding box by a randomly chosen amount in a randomly selected axis.
The step is accepted or rejected based on
```math
\Delta W = \Delta E + P \Delta V - N k_B T \ln \left( \frac{V + \Delta V}{V} \right)
```
where `ΔE` is the change in potential energy, `P` is the equilibrium pressure along the
selected axis, `ΔV` is the change in volume, `N` is the number of molecules in the system,
`T` is the equilibrium temperature and `V` is the system volume.
If `ΔW ≤ 0` the step is always accepted, if `ΔW > 0` the step is accepted with probability
`exp(-ΔW/kT)`.

The scale factor is modified over time to maintain an acceptance rate of around half.
If the topology of the system is set then molecules are moved as a unit so properties
such as bond lengths do not change.

The barostat assumes that the simulation is being run at a constant temperature but
does not actively control the temperature.
It should be used alongside a temperature coupling method such as the [`Langevin`](@ref)
simulator or [`AndersenThermostat`](@ref) coupling.
The neighbor list is not updated when making trial moves or after accepted moves.
Note that the barostat can change the bounding box of the system.
"""
mutable struct MonteCarloAnisotropicBarostat{D, T, P, K, V}
    pressure::SVector{D, P}
    temperature::K
    n_steps::Int
    n_iterations::Int
    volume_scale::V
    scale_increment::T
    max_volume_frac::T
    trial_find_neighbors::Bool
    n_attempted::Vector{Int}
    n_accepted::Vector{Int}
end

function MonteCarloAnisotropicBarostat(P::SVector{D},
                                       temp,
                                       boundary;
                                       n_steps=30,
                                       n_iterations=1,
                                       scale_factor=0.01,
                                       scale_increment=1.1,
                                       max_volume_frac=0.3,
                                       trial_find_neighbors=false) where D
    T = float_type(boundary)
    volume_scale_factor = volume(boundary) * T(scale_factor)
    volume_scale = fill(volume_scale_factor, D)
    if AtomsBase.n_dimensions(boundary) != D
        throw(ArgumentError("pressure vector length ($(D)) must match boundary " *
                            "dimensionality ($(AtomsBase.n_dimensions(boundary)))"))
    end

    return MonteCarloAnisotropicBarostat(
        P,
        temp,
        n_steps,
        n_iterations,
        volume_scale,
        T(scale_increment),
        T(max_volume_frac),
        trial_find_neighbors,
        zeros(Int, D),
        zeros(Int, D),
    )
end

function apply_coupling!(sys::System{D, AT, T},
                         barostat::MonteCarloAnisotropicBarostat{D},
                         sim,
                         neighbors=nothing,
                         step_n::Integer=0;
                         n_threads::Integer=Threads.nthreads(),
                         rng=Random.default_rng()) where {D, AT, T}
    !iszero(step_n % barostat.n_steps) && return false
    all(isnothing, barostat.pressure) && return false

    kT = energy_remove_mol(sys.k * barostat.temperature)
    n_molecules = isnothing(sys.topology) ? length(sys) : length(sys.topology.molecule_atom_counts)
    recompute_forces = false
    old_coords = similar(sys.coords)

    for attempt_n in 1:barostat.n_iterations
        axis = 0
        while true
            axis = rand(rng, 1:D)
            !isnothing(barostat.pressure[axis]) && break
        end
        mask1 = falses(D)
        mask2 = trues(D)
        mask1[axis] = true
        mask2[axis] = false

        E = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
        V = volume(sys.boundary)
        dV = barostat.volume_scale[axis] * (2 * rand(rng, T) - 1)
        v_scale = (V + dV) / V
        l_scale = SVector{D}(mask1 * v_scale + mask2)
        old_coords .= sys.coords
        old_boundary = sys.boundary
        scale_coords!(sys, l_scale)

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
        dW = dE + uconvert(unit(dE), barostat.pressure[axis] * dV) - n_molecules * kT * log(v_scale)
        if dW <= zero(dW) || rand(rng, T) < exp(-dW / kT)
            recompute_forces = true
            barostat.n_accepted[axis] += 1
        else
            sys.coords .= old_coords
            sys.boundary = old_boundary
        end
        barostat.n_attempted[axis] += 1

        # Modify size of volume change to keep accept/reject ratio roughly equal
        if barostat.n_attempted[axis] >= 10
            if barostat.n_accepted[axis] < 0.25 * barostat.n_attempted[axis]
                barostat.volume_scale[axis] /= barostat.scale_increment
                barostat.n_attempted[axis] = 0
                barostat.n_accepted[axis] = 0
            elseif barostat.n_accepted[axis] > 0.75 * barostat.n_attempted[axis]
                barostat.volume_scale[axis] = min(barostat.volume_scale[axis] * barostat.scale_increment,
                                                  V * barostat.max_volume_frac)
                barostat.n_attempted[axis] = 0
                barostat.n_accepted[axis] = 0
            end
        end
    end
    return recompute_forces
end

@doc raw"""
    MonteCarloMembraneBarostat(pressure, tension, temperature, boundary; n_steps=30,
                       n_iterations=1, scale_factor=0.01, scale_increment=1.1,
                       max_volume_frac=0.3, trial_find_neighbors=false,
                       xy_isotropy=false, z_axis_fixed=false, constant_volume=false)

The Monte Carlo membrane barostat for controlling pressure.

Set the `xy_isotropy` flag to `true` to scale the x and y axes isotropically.
Set the `z_axis_fixed` flag to `true` to uncouple the z-axis and keep it fixed.
Set the `constant_volume` flag to `true` to keep the system volume constant by
scaling the z-axis accordingly.
The `z_axis_fixed` and `constant_volume` flags cannot be `true` simultaneously.

See [Chow and Ferguson 1995](https://doi.org/10.1016/0010-4655(95)00059-O),
[Åqvist et al. 2004](https://doi.org/10.1016/j.cplett.2003.12.039) and the OpenMM
source code.
At regular intervals a Monte Carlo step is attempted by scaling the coordinates and
the bounding box by a randomly chosen amount in a randomly selected axis.
The step is accepted or rejected based on
```math
\Delta W = \Delta E + P \Delta V - \gamma \Delta A - N k_B T \ln \left( \frac{V + \Delta V}{V} \right)
```
where `ΔE` is the change in potential energy, `P` is the equilibrium pressure along the
selected axis, `ΔV` is the change in volume, `γ` is the surface tension, `ΔA` is the change
in surface area, `N` is the number of molecules in the system, `T` is the equilibrium
temperature and `V` is the system volume.
If `ΔW ≤ 0` the step is always accepted, if `ΔW > 0` the step is accepted with probability
`exp(-ΔW/kT)`.

The scale factor is modified over time to maintain an acceptance rate of around half.
If the topology of the system is set then molecules are moved as a unit so properties
such as bond lengths do not change.

The barostat assumes that the simulation is being run at a constant temperature but
does not actively control the temperature.
It should be used alongside a temperature coupling method such as the [`Langevin`](@ref)
simulator or [`AndersenThermostat`](@ref) coupling.
The neighbor list is not updated when making trial moves or after accepted moves.
Note that the barostat can change the bounding box of the system.

This barostat is only available for 3D systems.
"""
mutable struct MonteCarloMembraneBarostat{T, P, K, V, S}
    pressure::SVector{3, P}
    tension::S
    temperature::K
    n_steps::Int
    n_iterations::Int
    volume_scale::V
    scale_increment::T
    max_volume_frac::T
    trial_find_neighbors::Bool
    n_attempted::Vector{Int}
    n_accepted::Vector{Int}
    xy_isotropy::Bool
    constant_volume::Bool
end

function MonteCarloMembraneBarostat(P,
                                    tension,
                                    temp,
                                    boundary;
                                    n_steps=30,
                                    n_iterations=1,
                                    scale_factor=0.01,
                                    scale_increment=1.1,
                                    max_volume_frac=0.3,
                                    trial_find_neighbors=false,
                                    xy_isotropy=false,
                                    z_axis_fixed=false,
                                    constant_volume=false)
    T = float_type(boundary)
    volume_scale_factor = volume(boundary) * T(scale_factor)
    volume_scale = fill(volume_scale_factor, 3)

    if AtomsBase.n_dimensions(boundary) != 3
        throw(ArgumentError("boundary dimensionality ($(AtomsBase.n_dimensions(boundary))) must be 3"))
    end
    if z_axis_fixed && constant_volume
        throw(ArgumentError("cannot keep z-axis fixed whilst keeping the volume constant"))
    end

    pressX = P
    pressY = P
    pressZ = ((z_axis_fixed || constant_volume) ? nothing : P)

    return MonteCarloMembraneBarostat(
        SVector(pressX, pressY, pressZ),
        tension,
        temp,
        n_steps,
        n_iterations,
        volume_scale,
        T(scale_increment),
        T(max_volume_frac),
        trial_find_neighbors,
        zeros(Int, 3),
        zeros(Int, 3),
        xy_isotropy,
        constant_volume,
    )
end

function apply_coupling!(sys::System{D, AT, T},
                         barostat::MonteCarloMembraneBarostat,
                         sim,
                         neighbors=nothing,
                         step_n::Integer=0;
                         n_threads::Integer=Threads.nthreads(),
                         rng=Random.default_rng()) where {D, AT, T}
    !iszero(step_n % barostat.n_steps) && return false

    kT = energy_remove_mol(sys.k * barostat.temperature)
    n_molecules = isnothing(sys.topology) ? length(sys) : length(sys.topology.molecule_atom_counts)
    recompute_forces = false
    old_coords = similar(sys.coords)

    for attempt_n in 1:barostat.n_iterations
        axis = 0
        while true
            axis = rand(rng, 1:D)
            !isnothing(barostat.pressure[axis]) && break
        end
        if barostat.xy_isotropy && axis == 2
            axis = 1
        end

        E = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
        V = volume(sys.boundary)
        dV = barostat.volume_scale[axis] * (2 * rand(rng, T) - 1)
        v_scale = (V + dV) / V
        l_scale = SVector{D, T}(one(T), one(T), one(T))
        if (axis == 1 || axis == 2) && barostat.xy_isotropy
            xy_scale = sqrt(v_scale)
            l_scale = SVector{D}(xy_scale, xy_scale, one(T))
        else
            mask1 = falses(D)
            mask2 = trues(D)
            mask1[axis] = true
            mask2[axis] = false
            l_scale = SVector{D}(mask1 * v_scale + mask2)
        end

        if barostat.constant_volume
            l_scale = SVector{D}(l_scale[1], l_scale[2], inv(l_scale[1] * l_scale[2]))
            v_scale = one(T)
            dV = zero(dV)
        end

        dA = sys.boundary[1] * sys.boundary[2] * (l_scale[1] * l_scale[2] - one(T))

        old_coords .= sys.coords
        old_boundary = sys.boundary
        scale_coords!(sys, l_scale)

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
        PdV = uconvert(unit(dE), barostat.pressure[axis] * dV)
        γdA = uconvert(unit(dE), barostat.tension * dA)
        dW = dE + PdV - γdA - n_molecules * kT * log(v_scale)
        if dW <= zero(dW) || rand(rng, T) < exp(-dW / kT)
            recompute_forces = true
            barostat.n_accepted[axis] += 1
        else
            sys.coords .= old_coords
            sys.boundary = old_boundary
        end
        barostat.n_attempted[axis] += 1

        # Modify size of volume change to keep accept/reject ratio roughly equal
        if barostat.n_attempted[axis] >= 10
          if barostat.n_accepted[axis] < 0.25 * barostat.n_attempted[axis]
            barostat.volume_scale[axis] /= barostat.scale_increment
                barostat.n_attempted[axis] = 0
                barostat.n_accepted[axis] = 0
              elseif barostat.n_accepted[axis] > 0.75 * barostat.n_attempted[axis]
                barostat.volume_scale[axis] = min(barostat.volume_scale[axis] * barostat.scale_increment,
                                            V * barostat.max_volume_frac)
                barostat.n_attempted[axis] = 0
                barostat.n_accepted[axis] = 0
            end
        end
    end
    return recompute_forces
end
