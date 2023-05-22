# Different ways to simulate molecules

export
    SteepestDescentMinimizer,
    simulate!,
    VelocityVerlet,
    Verlet,
    StormerVerlet,
    Langevin,
    LangevinSplitting,
    NoseHoover,
    TemperatureREMD,
    remd_exchange!,
    HamiltonianREMD,
    simulate_remd!,
    MetropolisMonteCarlo,
    random_uniform_translation!,
    random_normal_translation!

"""
    SteepestDescentMinimizer(; <keyword arguments>)

Steepest descent energy minimization.

Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `step_size::D=0.01u"nm"`: the initial maximum displacement.
- `max_steps::Int=1000`: the maximum number of steps.
- `tol::F=1000.0u"kJ * mol^-1 * nm^-1"`: the maximum force below which to
    finish minimization.
- `run_loggers::Bool=false`: whether to run the loggers during minimization.
- `log_stream::L=devnull`: stream to print minimization progress to.
"""
struct SteepestDescentMinimizer{D, F, L}
    step_size::D
    max_steps::Int
    tol::F
    run_loggers::Bool
    log_stream::L
end

function SteepestDescentMinimizer(;
                                    step_size=0.01u"nm",
                                    max_steps=1_000,
                                    tol=1000.0u"kJ * mol^-1 * nm^-1",
                                    run_loggers=false,
                                    log_stream=devnull)
    return SteepestDescentMinimizer(step_size, max_steps, tol, run_loggers, log_stream)
end

"""
    simulate!(system, simulator, n_steps; n_threads=Threads.nthreads())
    simulate!(system, simulator; n_threads=Threads.nthreads())

Run a simulation on a system according to the rules of the given simulator.

Custom simulators should implement this function.
"""
function simulate!(sys,
                    sim::SteepestDescentMinimizer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    sim.run_loggers && run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    E = potential_energy(sys, neighbors; n_threads=n_threads)
    println(sim.log_stream, "Step 0 - potential energy ",
            E, " - max force N/A - N/A")
    hn = sim.step_size

    for step_n in 1:sim.max_steps
        F = forces(sys, neighbors; n_threads=n_threads)
        max_force = maximum(norm.(F))

        coords_copy = sys.coords
        sys.coords += hn * F ./ max_force
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

        neighbors_copy = neighbors
        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                    n_threads=n_threads)
        E_trial = potential_energy(sys, neighbors; n_threads=n_threads)
        if E_trial < E
            hn = 6 * hn / 5
            E = E_trial
            println(sim.log_stream, "Step ", step_n, " - potential energy ",
                    E_trial, " - max force ", max_force, " - accepted")
        else
            sys.coords = coords_copy
            neighbors = neighbors_copy
            hn = hn / 5
            println(sim.log_stream, "Step ", step_n, " - potential energy ",
                    E_trial, " - max force ", max_force, " - rejected")
        end

        sim.run_loggers && run_loggers!(sys, neighbors, step_n;
                                        n_threads=n_threads)

        if max_force < sim.tol
            break
        end
    end
    return sys
end

"""
    VelocityVerlet(; <keyword arguments>)

The velocity Verlet integrator.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct VelocityVerlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Int
    run_loggers::Bool
end

function VelocityVerlet(; dt, coupling=NoCoupling(), run_loggers = true, remove_CM_motion=1)
    return VelocityVerlet(dt, coupling, Int(remove_CM_motion))
end

function simulate!(sys,
                    sim::VelocityVerlet,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    sim.run_loggers && run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    accels_t = accelerations(sys, neighbors; n_threads=n_threads)
    accels_t_dt = zero(accels_t)

    for step_n in 1:n_steps
        old_coords = copy(sys.coords)
        sys.coords += sys.velocities .* sim.dt .+ (accel_remove_mol.(accels_t) .* sim.dt ^ 2) ./ 2

        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

        accels_t_dt = accelerations(sys, neighbors; n_threads=n_threads)

        sys.velocities += accel_remove_mol.(accels_t .+ accels_t_dt) .* sim.dt / 2

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)
        if recompute_forces
            accels_t = accelerations(sys, neighbors; n_threads=n_threads)
        else
            accels_t = accels_t_dt
        end

        sim.run_loggers && run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
    end
    return sys
end

"""
    Verlet(; <keyword arguments>)

The leapfrog Verlet integrator.

This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct Verlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Int
end

function Verlet(; dt, coupling=NoCoupling(), remove_CM_motion=1)
    return Verlet(dt, coupling, Int(remove_CM_motion))
end

function simulate!(sys,
                    sim::Verlet,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; n_threads=n_threads)

        sys.velocities += accel_remove_mol.(accels_t) .* sim.dt

        old_coords = copy(sys.coords)
        sys.coords += sys.velocities .* sim.dt
        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)

        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
    end
    return sys
end

"""
    StormerVerlet(; <keyword arguments>)

The Störmer-Verlet integrator.

The velocity calculation is accurate to O(dt).

Does not currently work with coupling methods that alter the velocity.
Does not currently remove the center of mass motion.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
"""
struct StormerVerlet{T, C}
    dt::T
    coupling::C
end

StormerVerlet(; dt, coupling=NoCoupling()) = StormerVerlet(dt, coupling)

function simulate!(sys,
                    sim::StormerVerlet,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    coords_last = sys.coords

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; n_threads=n_threads)

        coords_copy = sys.coords
        if step_n == 1
            # Use the velocities at the first step since there is only one set of coordinates
            sys.coords += sys.velocities .* sim.dt .+
                                        (accel_remove_mol.(accels_t) .* sim.dt ^ 2) ./ 2
        else
            sys.coords += vector.(coords_last, sys.coords, (sys.boundary,)) .+
                                        accel_remove_mol.(accels_t) .* sim.dt ^ 2
        end
        
        apply_constraints!(sys, coords_copy, sim.dt)

        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        # This is accurate to O(dt)
        sys.velocities = vector.(coords_copy, sys.coords, (sys.boundary,)) ./ sim.dt

        recompute_forces = apply_coupling!(sys, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)
        coords_last = coords_copy

        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
    end
    return sys
end

"""
    Langevin(; <keyword arguments>)

The Langevin integrator, based on the Langevin Middle Integrator in OpenMM.

This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::S`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `friction::F`: the friction coefficient of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct Langevin{S, K, F, C, T}
    dt::S
    temperature::K
    friction::F
    coupling::C
    remove_CM_motion::Int
    vel_scale::T
    noise_scale::T
end

function Langevin(; dt, temperature, friction, coupling=NoCoupling(), remove_CM_motion=1)
    vel_scale = exp(-dt * friction)
    noise_scale = sqrt(1 - vel_scale^2)
    return Langevin(dt, temperature, friction, coupling, Int(remove_CM_motion),
                    vel_scale, noise_scale)
end

function simulate!(sys,
                    sim::Langevin,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    rng=Random.GLOBAL_RNG)
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; n_threads=n_threads)

        sys.velocities += accel_remove_mol.(accels_t) .* sim.dt

        old_coords = copy(sys.coords)
        sys.coords += sys.velocities .* sim.dt / 2
        noise = random_velocities(sys, sim.temperature; rng=rng)
        sys.velocities = sys.velocities .* sim.vel_scale .+ noise .* sim.noise_scale

        sys.coords += sys.velocities .* sim.dt / 2

        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        recompute_forces = apply_coupling!(sys, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)

        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
    end
    return sys
end

"""
    LangevinSplitting(; <keyword arguments>)

The Langevin simulator using a general splitting scheme.

This consists of a succession of **A**, **B** and **O** steps, corresponding
respectively to updates in position, velocity for the potential part, and velocity
for the thermal fluctuation-dissipation part.
The [`Langevin`](@ref) and [`VelocityVerlet`](@ref) simulators without coupling
correspond to the **BAOA** and **BAB** schemes respectively.
For more information on the sampling properties of splitting schemes, see
[Fass et al. 2018](https://doi.org/10.3390/e20050318).

Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `dt::S`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `friction::F`: the friction coefficient. If units are used, it should have a
    dimensionality of mass per time.
- `splitting::W`: the splitting specifier. Should be a string consisting of the
    characters `A`, `B` and `O`. Strings with no `O`s reduce to deterministic
    symplectic schemes.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct LangevinSplitting{S, K, F, W}
    dt::S
    temperature::K
    friction::F
    splitting::W
    remove_CM_motion::Int
end

function LangevinSplitting(; dt, temperature, friction, splitting, remove_CM_motion=1)
    LangevinSplitting{typeof(dt), typeof(temperature), typeof(friction), typeof(splitting)}(
        dt, temperature, friction, splitting, Int(remove_CM_motion))
end

function simulate!(sys,
                    sim::LangevinSplitting,
                    n_steps::Integer;
                    n_threads::Integer=Threads.nthreads(),
                    rng=Random.GLOBAL_RNG)
    M_inv = inv.(masses(sys))
    α_eff = exp.(-sim.friction * sim.dt .* M_inv / count('O', sim.splitting))
    σ_eff = sqrt.((1 * unit(eltype(α_eff))) .- (α_eff .^ 2))

    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    accels_t = accelerations(sys, neighbors; n_threads=n_threads)

    effective_dts = [sim.dt / count(c, sim.splitting) for c in sim.splitting]

    # Determine the need to recompute accelerations before B steps
    forces_known = !occursin(r"^.*B[^B]*A[^B]*$", sim.splitting)

    force_computation_steps = map(collect(sim.splitting)) do op
        if op == 'O'
            return false
        elseif op == 'A'
            forces_known = false
            return false
        elseif op == 'B'
            if forces_known
                return false
            else
                forces_known = true
                return true
            end
        end
    end

    step_arg_pairs = map(enumerate(sim.splitting)) do (j, op)
        if op == 'A'
            return (A_step!, (sys, effective_dts[j]))
        elseif op == 'B'
            return (B_step!, (sys, effective_dts[j], accels_t, force_computation_steps[j], n_threads))
        elseif op == 'O'
            return (O_step!, (sys, α_eff, σ_eff, rng, sim.temperature))
        end
    end

    for step_n in 1:n_steps
        old_coords = copy(sys.coords)
        for (step!, args) in step_arg_pairs
            step!(args..., neighbors)
        end

        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                   n_threads=n_threads)

        run_loggers!(sys, neighbors, step_n)
    end
    return sys
end

function O_step!(sys, α_eff, σ_eff, rng, temperature, neighbors)
    noise = random_velocities(sys, temperature; rng=rng)
    sys.velocities = α_eff .* sys.velocities + σ_eff .* noise
    return sys
end

function A_step!(sys, dt_eff, neighbors)
    sys.coords += sys.velocities * dt_eff
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    return sys
end

function B_step!(sys, dt_eff, acceleration_vector, compute_forces::Bool,
                 n_threads::Integer, neighbors)
    if compute_forces
        acceleration_vector .= accelerations(sys, neighbors; n_threads=n_threads)
    end
    sys.velocities += dt_eff * accel_remove_mol.(acceleration_vector)
    return sys
end

"""
    NoseHoover(; <keyword arguments>)

The Nosé-Hoover integrator, a NVT simulator that extends velocity Verlet to control the
temperature of the system.

See [Evans and Holian 1985](https://doi.org/10.1063/1.449071).

# Arguments
- `dt::T`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `damping::D=100*dt`: the temperature damping time scale.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct NoseHoover{T, K, D, C}
    dt::T
    temperature::K
    damping::D
    coupling::C
    remove_CM_motion::Int
end

function NoseHoover(; dt, temperature, damping=100*dt, coupling=NoCoupling(), remove_CM_motion=1)
    return NoseHoover(dt, temperature, damping, coupling, Int(remove_CM_motion))
end

function simulate!(sys, sim::NoseHoover, n_steps::Integer; n_threads::Integer=Threads.nthreads())
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0; n_threads=n_threads)
    accels_t = accelerations(sys, neighbors; n_threads=n_threads)
    accels_t_dt = zero(accels_t)

    v_half = zero(sys.velocities)
    zeta = zero(inv(sim.dt))
    df = 3 * length(sys) - 3

    for step_n in 1:n_steps
        v_half = sys.velocities .+ (accel_remove_mol.(accels_t) .- (sys.velocities .* zeta)) .* (sim.dt / 2)
        old_coords = copy(sys.coords)
        sys.coords += v_half .* sim.dt

        zeta_half = zeta + (sim.dt / (2 * (sim.damping^2))) * ((temperature(sys) / sim.temperature) - 1)
        KE_half = sum(masses(sys) .* sum.(abs2, v_half)) / 2
        T_half = uconvert(u"K", 2 * KE_half / (df * sys.k))
        zeta = zeta_half + (sim.dt / (2 * (sim.damping^2))) * ((T_half / sim.temperature) - 1)

        apply_constraints!(sys, old_coords, sim.dt)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

        accels_t_dt = accelerations(sys, neighbors; n_threads=n_threads)

        sys.velocities = (v_half .+ accel_remove_mol.(accels_t_dt) .* (sim.dt / 2)) ./
                         (1 + (zeta * sim.dt / 2))

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                    n_threads=n_threads)
        if recompute_forces
            accels_t = accelerations(sys, neighbors; n_threads=n_threads)
        else
            accels_t = accels_t_dt
        end

        run_loggers!(sys, neighbors, step_n; n_threads=n_threads)
    end
    return sys
end

"""
    TemperatureREMD(; <keyword arguments>)

A simulator for a parallel temperature replica exchange MD (T-REMD) simulation on a
[`ReplicaSystem`](@ref).

See [Sugita and Okamoto 1999](https://doi.org/10.1016/S0009-2614(99)01123-9).
The corresponding [`ReplicaSystem`](@ref) should have the same number of replicas as
the number of temperatures in the simulator.
When calling [`simulate!`](@ref), the `assign_velocities` keyword argument determines
whether to assign random velocities at the appropriate temperature for each replica.

Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `dt::DT`: the time step of the simulation.
- `temperatures::TP`: the temperatures corresponding to the replicas.
- `simulators::ST`: individual simulators for simulating each replica.
- `exchange_time::ET`: the time interval between replica exchange attempts.
"""
struct TemperatureREMD{N, T, DT, TP, ST, ET}
    dt::DT
    temperatures::TP
    simulators::ST
    exchange_time::ET
end

function TemperatureREMD(;
                         dt,
                         temperatures,
                         simulators,
                         exchange_time)
    T = eltype(temperatures)
    N = length(temperatures)
    DT = typeof(dt)
    TP = typeof(temperatures)
    ET = typeof(exchange_time)

    if length(simulators) != length(temperatures)
        throw(ArgumentError("number of temperatures ($(length(temperatures))) must match " *
                            "number of simulators ($(length(simulators)))"))
    end
    if exchange_time <= dt
        throw(ArgumentError("exchange time ($exchange_time) must be greater than the time step ($dt)"))
    end

    simulators = Tuple(simulators[i] for i in 1:N)
    ST = typeof(simulators)
    
    return TemperatureREMD{N, T, DT, TP, ST, ET}(dt, temperatures, simulators, exchange_time)
end

function simulate!(sys::ReplicaSystem,
                    sim::TemperatureREMD,
                    n_steps::Integer;
                    assign_velocities::Bool=false,
                    rng=Random.GLOBAL_RNG,
                    n_threads::Integer=Threads.nthreads())
    if sys.n_replicas != length(sim.simulators)
        throw(ArgumentError("number of replicas in ReplicaSystem ($(length(sys.n_replicas))) " *
                "and simulators in TemperatureREMD ($(length(sim.simulators))) do not match"))
    end

    if assign_velocities
        for i in eachindex(sys.replicas)
            random_velocities!(sys.replicas[i], sim.temperatures[i]; rng=rng)
        end
    end

    return simulate_remd!(sys, sim, n_steps; rng=rng, n_threads=n_threads)
end

"""
    remd_exchange!(sys, sim, n, m; rng=Random.GLOBAL_RNG, n_threads=Threads.nthreads())

Attempt an exchange of replicas `n` and `m` in a [`ReplicaSystem`](@ref) during a REMD simulation.

Successful exchanges should exchange coordinates and velocities as appropriate.
Returns acceptance quantity `Δ` and a `Bool` indicating whether the exchange was successful.
"""
function remd_exchange!(sys::ReplicaSystem{D, G, T},
                        sim::TemperatureREMD,
                        n::Integer,
                        m::Integer;
                        n_threads::Integer=Threads.nthreads(),
                        rng=Random.GLOBAL_RNG) where {D, G, T}
    k_b = energy_add_mol(sys.k, sys.energy_units)
    T_n, T_m = sim.temperatures[n], sim.temperatures[m]
    β_n, β_m = inv(k_b * T_n), inv(k_b * T_m)
    neighbors_n = find_neighbors(sys.replicas[n], sys.replicas[n].neighbor_finder;
                                    n_threads=n_threads)
    neighbors_m = find_neighbors(sys.replicas[m], sys.replicas[m].neighbor_finder;
                                    n_threads=n_threads)
    V_n = potential_energy(sys.replicas[n], neighbors_n; n_threads=n_threads)
    V_m = potential_energy(sys.replicas[m], neighbors_m; n_threads=n_threads)
    Δ = (β_m - β_n) * (V_n - V_m)
    should_exchange = Δ <= 0 || rand(rng) < exp(-Δ)

    if should_exchange
        # Exchange coordinates and velocities
        sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
        sys.replicas[n].velocities, sys.replicas[m].velocities = sys.replicas[m].velocities, sys.replicas[n].velocities
        # Scale velocities
        sys.replicas[n].velocities .*= sqrt(T_n / T_m)
        sys.replicas[m].velocities .*= sqrt(T_m / T_n)
    end

    return Δ, should_exchange
end

"""
    HamiltonianREMD(; <keyword arguments>)

A simulator for a parallel Hamiltonian replica exchange MD (H-REMD) simulation on a
[`ReplicaSystem`](@ref).

The replicas are expected to have different Hamiltonians, i.e. different interactions.
When calling [`simulate!`](@ref), the `assign_velocities` keyword argument determines
whether to assign random velocities at the appropriate temperature for each replica.

Not currently compatible with automatic differentiation using Zygote.

# Arguments
- `dt::DT`: the time step of the simulation.
- `temperature::T`: the temperatures of the simulation.
- `simulators::ST`: individual simulators for simulating each replica.
- `exchange_time::ET`: the time interval between replica exchange attempts.
"""
struct HamiltonianREMD{N, T, DT, ST, ET}
    dt::DT
    temperature::T
    simulators::ST
    exchange_time::ET
end

function HamiltonianREMD(;
                         dt,
                         temperature,
                         simulators,
                         exchange_time)
    N = length(simulators)
    DT = typeof(dt)
    T = typeof(temperature)
    ST = typeof(simulators)
    ET = typeof(exchange_time)

    if exchange_time <= dt
        throw(ArgumentError("exchange time ($exchange_time) must be greater than the time step ($dt)"))
    end
    
    return HamiltonianREMD{N, T, DT, ST, ET}(dt, temperature, simulators, exchange_time)
end

function simulate!(sys::ReplicaSystem,
                    sim::HamiltonianREMD,
                    n_steps::Integer;
                    assign_velocities::Bool=false,
                    rng=Random.GLOBAL_RNG,
                    n_threads::Integer=Threads.nthreads())
    if sys.n_replicas != length(sim.simulators)
        throw(ArgumentError("number of replicas in ReplicaSystem ($(length(sys.n_replicas))) " *
                "and simulators in HamiltonianREMD ($(length(sim.simulators))) do not match"))
    end

    if assign_velocities
        for i in eachindex(sys.replicas)
            random_velocities!(sys.replicas[i], sim.temperature; rng=rng)
        end
    end
    
    return simulate_remd!(sys, sim, n_steps; rng=rng, n_threads=n_threads)
end

function remd_exchange!(sys::ReplicaSystem{D, G, T},
                        sim::HamiltonianREMD,
                        n::Integer,
                        m::Integer;
                        n_threads::Integer=Threads.nthreads(),
                        rng=Random.GLOBAL_RNG) where {D, G, T}
    k_b = energy_add_mol(sys.k, sys.energy_units)
    T_sim = sim.temperature
    β_sim = inv(k_b * T_sim)
    neighbors_n = find_neighbors(sys.replicas[n], sys.replicas[n].neighbor_finder;
                                    n_threads=n_threads)
    neighbors_m = find_neighbors(sys.replicas[m], sys.replicas[m].neighbor_finder;
                                    n_threads=n_threads)
    V_n_i = potential_energy(sys.replicas[n], neighbors_n; n_threads=n_threads)
    V_m_i = potential_energy(sys.replicas[m], neighbors_m; n_threads=n_threads)

    sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
    V_n_f = potential_energy(sys.replicas[n], neighbors_m; n_threads=n_threads) # Use already calculated neighbors
    V_m_f = potential_energy(sys.replicas[m], neighbors_n; n_threads=n_threads)

    Δ = β_sim * (V_n_f - V_n_i + V_m_f - V_m_i)
    should_exchange = Δ <= 0 || rand(rng) < exp(-Δ)

    if should_exchange
        # Exchange velocities
        sys.replicas[n].velocities, sys.replicas[m].velocities = sys.replicas[m].velocities, sys.replicas[n].velocities
    else
        # Revert coordinate exchange
        sys.replicas[n].coords, sys.replicas[m].coords = sys.replicas[m].coords, sys.replicas[n].coords
    end

    return Δ, should_exchange
end

"""
    simulate_remd!(sys, remd_sim, n_steps; rng=Random.GLOBAL_RNG, n_threads=Threads.nthreads())

Run a REMD simulation on a [`ReplicaSystem`](@ref) using a REMD simulator.
"""
function simulate_remd!(sys::ReplicaSystem,
                        remd_sim,
                        n_steps::Integer;
                        rng=Random.GLOBAL_RNG,
                        n_threads::Integer=Threads.nthreads())
    if sys.n_replicas != length(remd_sim.simulators)
        throw(ArgumentError("number of replicas in ReplicaSystem ($(length(sys.n_replicas))) " *
            "and simulators in the REMD simulator ($(length(remd_sim.simulators))) do not match"))
    end

    if n_threads > sys.n_replicas
        thread_div = equal_parts(n_threads, sys.n_replicas)
    else
        # Use 1 thread per replica
        thread_div = equal_parts(sys.n_replicas, sys.n_replicas)
    end

    n_cycles = convert(Int, (n_steps * remd_sim.dt) ÷ remd_sim.exchange_time)
    cycle_length = n_cycles > 0 ? n_steps ÷ n_cycles : 0
    remaining_steps = n_cycles > 0 ? n_steps % n_cycles : n_steps
    n_attempts = 0

    for cycle in 1:n_cycles
        @sync for idx in eachindex(remd_sim.simulators)
            Threads.@spawn simulate!(sys.replicas[idx], remd_sim.simulators[idx], cycle_length;
                                     n_threads=thread_div[idx])
        end

        # Alternate checking even pairs 2-3/4-5/6-7/... and odd pairs 1-2/3-4/5-6/...
        cycle_parity = cycle % 2
        for n in (1 + cycle_parity):2:(sys.n_replicas - 1)
            n_attempts += 1
            m = n + 1
            Δ, exchanged = remd_exchange!(sys, remd_sim, n, m; rng=rng, n_threads=n_threads)
            if exchanged && !isnothing(sys.exchange_logger)
                log_property!(sys.exchange_logger, sys, nothing, cycle * cycle_length;
                                    indices=(n, m), delta=Δ, n_threads=n_threads)
            end
        end
    end

    if remaining_steps > 0
        @sync for idx in eachindex(remd_sim.simulators)
            Threads.@spawn simulate!(sys.replicas[idx], remd_sim.simulators[idx], remaining_steps;
                                     n_threads=thread_div[idx])
        end
    end

    if !isnothing(sys.exchange_logger)
        finish_logs!(sys.exchange_logger; n_steps=n_steps, n_attempts=n_attempts)
    end

    return sys
end

# Calculate k almost equal patitions of n
@inline function equal_parts(n, k)
    ndiv = n ÷ k
    nrem = n % k
    n_parts = ntuple(i -> (i <= nrem) ? ndiv + 1 : ndiv, k)
    return n_parts
end

"""
    MetropolisMonteCarlo(; <keyword arguments>)

A Monte Carlo simulator that uses the Metropolis algorithm to sample the configuration space.

`simulate!` for this simulator accepts an optional keyword argument `log_states::Bool=true` which 
determines whether to run the loggers or not (for example, during equilibration).

# Arguments
- `temperature::T`: the temperature of the system.
- `trial_moves::M`: a function that performs the trial moves.
- `trial_args::Dict`: a dictionary of arguments to be passed to the trial move function.
"""
struct MetropolisMonteCarlo{T, M}
    temperature::T
    trial_moves::M
    trial_args::Dict
end

function MetropolisMonteCarlo(; temperature, trial_moves, trial_args=Dict())
    return MetropolisMonteCarlo(temperature, trial_moves, trial_args)
end

function simulate!(sys::System{D, G, T},
                   sim::MetropolisMonteCarlo,
                   n_steps::Integer;
                   n_threads::Integer=Threads.nthreads(),
                   log_states::Bool=true) where {D, G, T}
    k_b = energy_add_mol(sys.k, sys.energy_units)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    E_old = potential_energy(sys, neighbors; n_threads=n_threads)
    for i in 1:n_steps
        coords_old = copy(sys.coords)
        sim.trial_moves(sys; sim.trial_args...) # Changes the coordinates of the system
        neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        E_new = potential_energy(sys, neighbors; n_threads=n_threads)

        ΔE = E_new - E_old
        δ = ΔE / (k_b * sim.temperature)
        if δ < 0 || rand() < exp(-δ)
            log_states && run_loggers!(sys, neighbors, i; n_threads=n_threads, success=true,
                                       energy_rate=E_new / (k_b * sim.temperature))
            E_old = E_new
        else
            sys.coords = coords_old
            log_states && run_loggers!(sys, neighbors, i; n_threads=n_threads, success=false,
                                       energy_rate=E_old / (k_b * sim.temperature))
        end
    end
    return sys
end

"""
    random_uniform_translation!(sys::System; shift_size=oneunit(eltype(eltype(sys.coords))))

Performs a random translation of the coordinates of a randomly selected atom in a [`System`](@ref).

The translation is generated using a uniformly selected direction and uniformly selected length
in range [0, 1) scaled by `shift_size` which should have appropriate length units.
"""
function random_uniform_translation!(sys::System{D, G, T};
                                     shift_size=oneunit(eltype(eltype(sys.coords)))) where {D, G, T}
    rand_idx = rand(eachindex(sys))
    direction = random_unit_vector(T, D)
    magnitude = rand(T) * shift_size
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
    return sys
end

"""
    random_normal_translation!(sys::System; shift_size=oneunit(eltype(eltype(sys.coords))))

Performs a random translation of the coordinates of a randomly selected atom in a [`System`](@ref).

The translation is generated using a uniformly chosen direction and length selected from
the standard normal distribution i.e. with mean 0 and standard deviation 1, scaled by `shift_size`
which should have appropriate length units.
"""
function random_normal_translation!(sys::System{D, G, T};
                                    shift_size=oneunit(eltype(eltype(sys.coords)))) where {D, G, T}
    rand_idx = rand(eachindex(sys))
    direction = random_unit_vector(T, D)
    magnitude = randn(T) * shift_size
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
    return sys
end

function random_unit_vector(float_type, dims)
    vec = randn(float_type, dims)
    return vec / norm(vec)
end
