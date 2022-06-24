# Different ways to simulate molecules

export
    SteepestDescentMinimizer,
    simulate!,
    VelocityVerlet,
    Verlet,
    StormerVerlet,
    Langevin,
    LangevinSplitting

"""
    SteepestDescentMinimizer(; <keyword arguments>)

Steepest descent energy minimization.

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
    return SteepestDescentMinimizer(step_size, max_steps, tol,
                                    run_loggers, log_stream)
end

"""
    simulate!(system, simulator, n_steps; parallel=true)
    simulate!(system, simulator; parallel=true)

Run a simulation on a system according to the rules of the given simulator.
Custom simulators should implement this function.
"""
function simulate!(sys,
                    sim::SteepestDescentMinimizer;
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    sim.run_loggers && run_loggers!(sys, neighbors, 0; parallel=parallel)
    E = potential_energy(sys, neighbors)
    println(sim.log_stream, "Step 0 - potential energy ",
            E, " - max force N/A - N/A")
    hn = sim.step_size

    for step_n in 1:sim.max_steps
        F = forces(sys, neighbors; parallel=parallel)
        max_force = maximum(norm.(F))

        coords_copy = sys.coords
        sys.coords += hn * F ./ max_force
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))

        neighbors_copy = neighbors
        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                    parallel=parallel)
        E_trial = potential_energy(sys, neighbors)
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
                                        parallel=parallel)

        if max_force < sim.tol
            break
        end
    end
    return sys
end

# Forces are often expressed per mol but this dimension needs removing for use in the integrator
function remove_molar(x)
    fx = first(x)
    if dimension(fx) == u"𝐋 * 𝐍^-1 * 𝐓^-2"
        T = typeof(ustrip(fx))
        return x / T(Unitful.Na)
    else
        return x
    end
end

"""
    VelocityVerlet(; <keyword arguments>)

The velocity Verlet integrator.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=NoCoupling()`: the coupling which applies during the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the centre of mass motion
    every time step.
"""
struct VelocityVerlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Bool
end

function VelocityVerlet(; dt, coupling=NoCoupling(), remove_CM_motion=true)
    return VelocityVerlet(dt, coupling, remove_CM_motion)
end

function simulate!(sys,
                    sim::VelocityVerlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    run_loggers!(sys, neighbors, 0; parallel=parallel)
    accels_t = accelerations(sys, neighbors; parallel=parallel)
    accels_t_dt = zero(accels_t)
    sim.remove_CM_motion && remove_CM_motion!(sys)

    for step_n in 1:n_steps
        sys.coords += sys.velocities .* sim.dt .+ (remove_molar.(accels_t) .* sim.dt ^ 2) ./ 2
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))

        accels_t_dt = accelerations(sys, neighbors; parallel=parallel)

        sys.velocities += remove_molar.(accels_t .+ accels_t_dt) .* sim.dt / 2

        sim.remove_CM_motion && remove_CM_motion!(sys)
        apply_coupling!(sys, sim, sim.coupling)

        run_loggers!(sys, neighbors, step_n; parallel=parallel)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        parallel=parallel)
            accels_t = accels_t_dt
        end
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
- `remove_CM_motion::Bool=true`: whether to remove the centre of mass motion
    every time step.
"""
struct Verlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Bool
end

function Verlet(; dt, coupling=NoCoupling(), remove_CM_motion=true)
    return Verlet(dt, coupling, remove_CM_motion)
end

function simulate!(sys,
                    sim::Verlet,
                    n_steps::Integer;
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    run_loggers!(sys, neighbors, 0; parallel=parallel)
    sim.remove_CM_motion && remove_CM_motion!(sys)

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; parallel=parallel)

        sys.velocities += remove_molar.(accels_t) .* sim.dt

        sys.coords += sys.velocities .* sim.dt
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))

        sim.remove_CM_motion && remove_CM_motion!(sys)
        apply_coupling!(sys, sim, sim.coupling)

        run_loggers!(sys, neighbors, step_n; parallel=parallel)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        parallel=parallel)
        end
    end
    return sys
end

"""
    StormerVerlet(; <keyword arguments>)

The Störmer-Verlet integrator.
Does not currently work with coupling methods that alter the velocity.

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
                    parallel::Bool=true)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    run_loggers!(sys, neighbors, 0; parallel=parallel)
    coords_last = sys.coords

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; parallel=parallel)

        coords_copy = sys.coords
        if step_n == 1
            # Use the velocities at the first step since there is only one set of coordinates
            sys.coords += sys.velocities .* sim.dt .+ (remove_molar.(accels_t) .* sim.dt ^ 2) ./ 2
        else
            sys.coords += vector.(coords_last, sys.coords, (sys.box_size,)) .+ remove_molar.(accels_t) .* sim.dt ^ 2
        end
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))

        # This is accurate to O(dt)
        sys.velocities = vector.(coords_copy, sys.coords, (sys.box_size,)) ./ sim.dt

        apply_coupling!(sys, sim, sim.coupling)

        run_loggers!(sys, neighbors, step_n; parallel=parallel)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        parallel=parallel)
            coords_last = coords_copy
        end
    end
    return sys
end

"""
    Langevin(; <keyword arguments>)

The Langevin integrator, based on the Langevin Middle Integrator in OpenMM.
This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::T`: the time step of the simulation.
- `temperature::K`: the temperature of the simulation.
- `friction::F`: the friction coefficient of the simulation.
- `remove_CM_motion::Bool=true`: whether to remove the centre of mass motion
    every time step.
"""
struct Langevin{S, K, F, T}
    dt::S
    temperature::K
    friction::F
    remove_CM_motion::Bool
    vel_scale::T
    noise_scale::T
end

function Langevin(; dt, temperature, friction, remove_CM_motion=true)
    vel_scale = exp(-dt * friction)
    noise_scale = sqrt(1 - vel_scale^2)
    return Langevin(dt, temperature, friction, remove_CM_motion, vel_scale, noise_scale)
end

function simulate!(sys,
                    sim::Langevin,
                    n_steps::Integer;
                    parallel::Bool=true,
                    rng=Random.GLOBAL_RNG)
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel=parallel)
    run_loggers!(sys, neighbors, 0; parallel=parallel)
    sim.remove_CM_motion && remove_CM_motion!(sys)

    for step_n in 1:n_steps
        accels_t = accelerations(sys, neighbors; parallel=parallel)

        sys.velocities += remove_molar.(accels_t) .* sim.dt

        sys.coords += sys.velocities .* sim.dt / 2
        noise = random_velocities(sys, sim.temperature; rng=rng)
        sys.velocities = sys.velocities .* sim.vel_scale .+ noise .* sim.noise_scale

        sys.coords += sys.velocities .* sim.dt / 2
        sys.coords = wrap_coords_vec.(sys.coords, (sys.box_size,))
        sim.remove_CM_motion && remove_CM_motion!(sys)

        run_loggers!(sys, neighbors, step_n; parallel=parallel)

        if step_n != n_steps
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                        parallel=parallel)
        end
    end
    return sys
end

""" `LangevinSplitting(; <keyword arguments>)`
A Langevin simulator using a general splitting scheme, consisting of a succession of
**A**, **B** and **O** steps, corresponding respectively to updates in position, velocity for the potential part,
 and velocity for the thermal fluctuation-dissipation part.
The `Langevin` and `VelocityVerlet` simulators without coupling correspond to the **BAOA** and **BAB** schemes respectively.
For more information on the sampling properties of splitting schemes, see this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6208357/pdf/entropy-20-00318.pdf) for a general introduction.

# Arguments
- `dt::dtType`: The timestep for the simulation
- `friction::frictionType`: The friction coefficient. If units are used, it should have a dimensionality of mass per time.
- `temperature::temperatureType`: The equilibrium temperature.
- `splitting::splittingType`: The splitting specifier. Should be a string consisting of the characters `A`,`B` and `O`. Strings with no `O`s reduce to deterministic symplectic schemes.
- `remove_CM_motion::Bool=true`: Whether to remove the centre of mass motion at each simulation iteration.
"""
struct LangevinSplitting{S,F,K,W}
    dt::S
    friction::F
    temperature::K
    splitting::W
    remove_CM_motion::Bool
end
function LangevinSplitting(; dt, friction, temperature, splitting,remove_CM_motion=true)
    LangevinSplitting{typeof(dt),typeof(friction),typeof(temperature),typeof(splitting)}(dt, friction, temperature, splitting,remove_CM_motion)
end

function simulate!(sys,sim::LangevinSplitting,n_steps::Integer;parallel::Bool=true,rng=Random.GLOBAL_RNG)
    M_inv = inv.(mass.(sys.atoms))
    α_eff = exp.(-sim.friction * sim.dt .* M_inv / count('O', sim.splitting))
    σ_eff = sqrt.( (1 * unit(eltype(α_eff))) .- (α_eff .^ 2))
    neighbors = find_neighbors(sys, sys.neighbor_finder; parallel = parallel)
    accels_t = accelerations(sys, neighbors; parallel=parallel)

    effective_dts = [sim.dt / count(c, sim.splitting) for c in sim.splitting]

    forces_known = true
    force_computation_steps = Bool[]

    occursin(r"^.*B[^B]*A[^B]*$", sim.splitting) && (forces_known = false) #determine the need to recompute accelerations before B steps

    for op in sim.splitting
        if op == 'O'
            push!(force_computation_steps, false)
        elseif op == 'A'
            push!(force_computation_steps, false)
            forces_known = false
        elseif op == 'B'
            if forces_known
                push!(force_computation_steps, false)
            else
                push!(force_computation_steps, true)
                forces_known = true
            end
        end
    end

    steps = []
    arguments = []

    for (j, op) in enumerate(sim.splitting)
        if op == 'A'
            push!(steps, A_step!)
            push!(arguments, (sys, effective_dts[j]))
        elseif op == 'B'
            push!(steps, B_step!)
            push!(arguments, (sys, effective_dts[j], accels_t, neighbors, force_computation_steps[j], parallel))
        elseif op == 'O'
            push!(steps, O_step!)
            push!(arguments, (sys, α_eff, σ_eff, rng, sim.temperature))
        end
    end

    step_arg_pairs = zip(steps, arguments)

    run_loggers!(sys, neighbors, 0; parallel=parallel)
    sim.remove_CM_motion && remove_CM_motion!(sys)

    for step_n = 1:n_steps

        
        for (step!, args) = step_arg_pairs
            step!(args...)
        end
        
        run_loggers!(sys, neighbors, step_n)
        sim.remove_CM_motion && remove_CM_motion!(sys)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; parallel=parallel)
    end
end

function O_step!(s::System, α_eff::V, σ_eff::V, rng::R, temperature::T) where {V,R<:AbstractRNG,T}
    noise = random_velocities(s, temperature; rng = rng)
    s.velocities = α_eff .* s.velocities + σ_eff .* noise
end

function A_step!(s::System, dt_eff::T) where {T}
    s.coords += s.velocities * dt_eff
    s.coords = wrap_coords_vec.(s.coords, (s.box_size,))
end

function B_step!(s::System, dt_eff::T, acceleration_vector::A, neighbors, compute_forces::Bool, parallel::Bool) where {T,A}
    compute_forces && (acceleration_vector .= accelerations(s, neighbors, parallel = parallel))
    s.velocities += dt_eff * remove_molar.(acceleration_vector)
end