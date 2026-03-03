# Different ways to simulate molecules

export
    SteepestDescentMinimizer,
    simulate!,
    VelocityVerlet,
    Verlet,
    StormerVerlet,
    Langevin,
    LangevinSplitting,
    OverdampedLangevin,
    NoseHoover,
    ReplicaExchangeMD,
    simulate_remd!,
    remd_exchange!,
    MetropolisMonteCarlo,
    random_uniform_translation!,
    random_normal_translation!

shortcut_sim(::Nothing, args...; kwargs...) = false

function default_show_progress()
    if haskey(ENV, "MOLLY_SHOW_PROGRESS")
        return parse(Bool, lowercase(ENV["MOLLY_SHOW_PROGRESS"]))
    else
        # true in interactive contexts, false otherwise
        return isdefined(Base, :active_repl) || isdefined(Main, :IJulia) ||
                                                        isdefined(Main, :PlutoRunner)
    end
end

function setup_progress(n_steps, show_progress)
    if show_progress
        return Progress(
            n_steps;
            enabled=true,
            showspeed=true,
            desc="Simulating:",
            color=:green,
            barglyphs=BarGlyphs('|', '█', ['▁', '▂', '▃', '▄', '▅', '▆', '▇'], ' ', '|'),
        )
    else
        return nothing
    end
end

function setup_progress_minimizer(threshold, show_progress)
    if show_progress
        return ProgressThresh(
            threshold;
            enabled=true,
            showspeed=true,
            desc="Minimizing:",
            color=:green,
        )
    else
        return nothing
    end
end

# Marked as inactive for Enzyme
next_nograd!(progress) = next!(progress)
next_nograd!(::Nothing) = nothing
update_nograd!(progress, val) = ProgressMeter.update!(progress, val)
update_nograd!(::Nothing, val) = nothing

"""
    SteepestDescentMinimizer(; <keyword arguments>)

Steepest descent energy minimization.

# Arguments
- `step_size::D=0.01u"nm"`: the initial maximum displacement.
- `max_steps::Int=1000`: the maximum number of steps.
- `tol::F=1000.0u"kJ * mol^-1 * nm^-1"`: the maximum force below which to
    finish minimization.
- `log_stream::L=devnull`: stream to print minimization progress to.
"""
@kwdef struct SteepestDescentMinimizer{D, F, L}
    step_size::D = 0.01u"nm"
    max_steps::Int = 1_000
    tol::F = 1000.0u"kJ * mol^-1 * nm^-1"
    log_stream::L = devnull
end

"""
    simulate!(system, simulator, n_steps; <keyword arguments>)
    simulate!(system, simulator; <keyword arguments>)
    simulate!(awh_sim::AWHSimulation, n_steps::Int)

Run a simulation on a system according to the rules of the given simulator.

Custom simulators should implement this function.
Constraints are applied during minimization, which can lead to issues.

# Arguments
- `n_threads=Threads.nthreads()`: the number of threads to run the simulation on, only
    relevant when running on CPU.
- `run_loggers`: whether to run the loggers during the simulation. Can be `true`, `false`
    or `:skipzero`, in which case the loggers are not run before the first step. `run_loggers`
    is `true` by default except for [`SteepestDescentMinimizer`](@ref), where it is `false`.
- `shortcut=nothing`: when to stop the simulation early. A struct with the `shortcut_sim`
    method defined can be provided. Unused for REMD simulations.
- `show_progress`: whether to show a progress bar for the simulation. `true` by default in
    the REPL/IJulia/Pluto, otherwise `false` by default. Can be set globally with the
    environmental variable `MOLLY_SHOW_PROGRESS`.
- `rng=Random.default_rng()`: the random number generator used for the simulation. Setting
    this allows reproducible stochastic simulations.

Alternatively:

Run an AWH simulation for a given number of molecular dynamics steps.

The total number of AWH iterations is automatically determined by dividing `n_steps` 
by the `num_md_steps` defined in the `AWHSimulation` struct.

# Arguments
- `awh_sim::AWHSimulation`: The [`AWHSimulation`](@ref) struct defining the AWH parameters and state.
- `n_steps::Int`: The total number of molecular dynamics steps to perform.
"""
@inline function simulate!(sys,
                           sim::SteepestDescentMinimizer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=false,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    # @inline needed to avoid Enzyme error
    needs_vir = false
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    buffers = init_buffers!(sys, n_threads)
    E = potential_energy(sys, neighbors, buffers; n_threads=n_threads)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads,
                   current_potential_energy=E)
    using_constraints = (length(sys.constraints) > 0)
    println(sim.log_stream, "Step 0 - potential energy ", E, " - max force N/A - N/A")
    hn = sim.step_size
    coords_copy = zero(sys.coords)
    F = zero_forces(sys)

    progress = setup_progress_minimizer(ustrip(sim.tol), show_progress)
    for step_n in 1:sim.max_steps
        forces!(F, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
        max_force = maximum(norm.(F))

        coords_copy .= sys.coords
        sys.coords .+= hn .* F ./ max_force
        using_constraints && apply_position_constraints!(sys, coords_copy; n_threads=n_threads)
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        neighbors_copy = neighbors
        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                    n_threads=n_threads)
        E_trial = potential_energy(sys, neighbors, buffers, step_n; n_threads=n_threads)
        if E_trial < E
            hn = 6 * hn / 5
            E = E_trial
            println(sim.log_stream, "Step ", step_n, " - potential energy ",
                    E_trial, " - max force ", max_force, " - accepted")
        else
            sys.coords .= coords_copy
            neighbors = neighbors_copy
            hn = hn / 5
            println(sim.log_stream, "Step ", step_n, " - potential energy ",
                    E_trial, " - max force ", max_force, " - rejected")
        end

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads,
                       current_potential_energy=E)

        if max_force < sim.tol
            break
        end
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads,
                        current_potential_energy=E)
            break
        end
        update_nograd!(progress, ustrip(max_force))
    end
    return sys
end

"""
    VelocityVerlet(; <keyword arguments>)

The velocity Verlet integrator.

# Arguments
- `dt::T`: the time step of the simulation.
- `coupling::C=nothing`: the coupling which applies during the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct VelocityVerlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Int
end

function VelocityVerlet(; dt, coupling=nothing, remove_CM_motion=1)
    return VelocityVerlet(dt, coupling, Int(remove_CM_motion))
end

@inline function simulate!(sys,
                           sim::VelocityVerlet,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    forces_t, forces_t_dt = zero_forces(sys), zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    forces!(forces_t, sys, neighbors, buffers, Val(needs_vir), 0; n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    accels_t_dt = zero(accels_t)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads, current_forces=forces_t)
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints
        cons_coord_storage = zero(sys.coords)
        cons_vel_storage = zero(sys.velocities)
    end
    dt_div2 = sim.dt / 2
    dt_sq_div2 = sim.dt^2 / 2

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        if using_constraints
            cons_coord_storage .= sys.coords
        end
        needs_vir = (step_n % needs_vir_steps == 0)

        sys.coords .+= sys.velocities .* sim.dt .+ accels_t .* dt_sq_div2
        using_constraints && apply_position_constraints!(sys, cons_coord_storage, cons_vel_storage,
                                                         sim.dt; n_threads=n_threads)
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        forces!(forces_t_dt, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
        accels_t_dt .= calc_accels.(forces_t_dt, masses(sys))

        sys.velocities .+= (accels_t .+ accels_t_dt) .* dt_div2
        using_constraints && apply_velocity_constraints!(sys; n_threads=n_threads)

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)
        if recompute_forces
            forces!(forces_t_dt, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
            forces_t .= forces_t_dt
            accels_t .= calc_accels.(forces_t, masses(sys))
        else
            forces_t .= forces_t_dt
            accels_t .= accels_t_dt
        end

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads,
                       current_forces=forces_t)
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads,
                        current_forces=forces_t)
            break
        end
        next_nograd!(progress)
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
- `coupling::C=nothing`: the coupling which applies during the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct Verlet{T, C}
    dt::T
    coupling::C
    remove_CM_motion::Int
end

function Verlet(; dt, coupling=nothing, remove_CM_motion=1)
    return Verlet(dt, coupling, Int(remove_CM_motion))
end

@inline function simulate!(sys,
                           sim::Verlet,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints
        cons_coord_storage = zero(sys.coords)
    end

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        needs_vir = (step_n % needs_vir_steps == 0)
        forces!(forces_t, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        sys.velocities .+= accels_t .* sim.dt

        if using_constraints
            cons_coord_storage .= sys.coords
        end
        sys.coords .+= sys.velocities .* sim.dt
        using_constraints && apply_position_constraints!(sys, cons_coord_storage;
                                                         n_threads=n_threads)

        if using_constraints
            sys.velocities .= (sys.coords .- cons_coord_storage) ./ sim.dt
        end

        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads,
                       current_forces=forces_t)
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads,
                        current_forces=forces_t)
            break
        end
        next_nograd!(progress)
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
- `coupling::C=nothing`: the coupling which applies during the simulation.
"""
@kwdef struct StormerVerlet{T, C}
    dt::T
    coupling::C = nothing
end

@inline function simulate!(sys,
                           sim::StormerVerlet,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads)
    coords_last, coords_copy = zero(sys.coords), zero(sys.coords)
    accels_t = calc_accels.(forces_t, masses(sys))
    using_constraints = (length(sys.constraints) > 0)
    dt_sq = sim.dt^2

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        needs_vir = (step_n % needs_vir_steps == 0)
        forces!(forces_t, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        coords_copy .= sys.coords
        if step_n == 1
            # Use the velocities at the first step since there is only one set of coordinates
            sys.coords .+= sys.velocities .* sim.dt .+ (accels_t .* dt_sq) ./ 2
        else
            sys.coords .+= vector.(coords_last, sys.coords, (sys.boundary,)) .+ accels_t .* dt_sq
        end

        using_constraints && apply_position_constraints!(sys, coords_copy; n_threads=n_threads)

        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)
        # This is accurate to O(dt)
        sys.velocities .= zero_vs_velocity.(
            vector.(coords_copy, sys.coords, (sys.boundary,)) ./ sim.dt,
            sys.virtual_site_flags,
        )

        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)
        coords_last .= coords_copy

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads,
                       current_forces=forces_t)
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads,
                        current_forces=forces_t)
            break
        end
        next_nograd!(progress)
    end
    return sys
end

"""
    Langevin(; <keyword arguments>)

The Langevin integrator, based on the Langevin Middle Integrator in OpenMM.

See [Zhang et al. 2019](https://doi.org/10.1021/acs.jpca.9b02771).
This is a leapfrog integrator, so the velocities are offset by half a time step
behind the positions.

# Arguments
- `dt::S`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `friction::F`: the friction coefficient of the simulation.
- `coupling::C=nothing`: the coupling which applies during the simulation.
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

function Langevin(; dt, temperature, friction, coupling=nothing, remove_CM_motion=1)
    vel_scale = exp(-dt * friction)
    noise_scale = sqrt(1 - vel_scale^2)
    return Langevin(dt, temperature, friction, coupling, Int(remove_CM_motion),
                    vel_scale, noise_scale)
end

@inline function simulate!(sys,
                           sim::Langevin,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    noise = zero(sys.velocities)
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints
        cons_coord_storage = zero(sys.coords)
        cons_vel_storage = zero(sys.velocities)
    end
    dt_div2 = sim.dt / 2

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        needs_vir = (step_n % needs_vir_steps == 0)
        forces!(forces_t, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        sys.velocities .+= accels_t .* sim.dt
        apply_velocity_constraints!(sys; n_threads=n_threads)

        if using_constraints
            cons_coord_storage .= sys.coords
        end
        sys.coords .+= sys.velocities .* dt_div2

        random_velocities!(noise, sys, sim.temperature; rng=rng)
        sys.velocities .= sys.velocities .* sim.vel_scale .+ noise .* sim.noise_scale

        sys.coords .+= sys.velocities .* dt_div2

        using_constraints && apply_position_constraints!(sys, cons_coord_storage, cons_vel_storage,
                                                         sim.dt; n_threads=n_threads)
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads,
                       current_forces=forces_t)
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads,
                        current_forces=forces_t)
            break
        end
        next_nograd!(progress)
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

Not currently compatible with constraints, will print a warning and continue
without applying constraints.

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

@inline function simulate!(sys,
                           sim::LangevinSplitting,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    if length(sys.constraints) > 0
        @warn "LangevinSplitting is not currently compatible with constraints, " *
              "constraints will be ignored"
    end
    M_inv = inv.(masses(sys))
    α_eff = exp.(-sim.friction * sim.dt .* M_inv / count('O', sim.splitting))
    σ_eff = sqrt.((1 * unit(eltype(α_eff))) .- (α_eff .^ 2))

    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads)
    forces!(forces_t, sys, neighbors, buffers, Val(false), 0; n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    noise = zero(sys.velocities)

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
            return (B_step!, (sys, forces_t, buffers, accels_t, effective_dts[j],
                              force_computation_steps[j], n_threads))
        elseif op == 'O'
            return (O_step!, (sys, noise, α_eff, σ_eff, rng, sim.temperature))
        end
    end

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        for (step!, args) in step_arg_pairs
            step!(args..., neighbors, step_n)
        end

        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                   n_threads=n_threads)

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads)
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads)
            break
        end
        next_nograd!(progress)
    end
    return sys
end

function A_step!(sys, dt_eff, neighbors, step_n)
    sys.coords .+= sys.velocities .* dt_eff
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    return sys
end

function B_step!(sys, forces_t, buffers, accels_t, dt_eff,
                 compute_forces::Bool, n_threads::Integer, neighbors, step_n::Integer)
    if compute_forces
        forces!(forces_t, sys, neighbors, buffers, Val(false), step_n; n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))
    end
    sys.velocities .+= dt_eff .* accels_t
    return sys
end

function O_step!(sys, noise, α_eff, σ_eff, rng, temperature, neighbors, step_n)
    random_velocities!(noise, sys, temperature; rng=rng)
    sys.velocities .= α_eff .* sys.velocities .+ σ_eff .* noise
    return sys
end

"""
    OverdampedLangevin(; <keyword arguments>)

Simulates the overdamped Langevin equation using the Euler-Maruyama method.

Not currently compatible with constraints, will print a warning and continue
without applying constraints.

# Arguments
- `dt::S`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `friction::F`: the friction coefficient of the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct OverdampedLangevin{S, K, F}
    dt::S
    temperature::K
    friction::F
    remove_CM_motion::Int
end

function OverdampedLangevin(; dt, temperature, friction, remove_CM_motion=1)
    return OverdampedLangevin(dt, temperature, friction, Int(remove_CM_motion))
end

@inline function simulate!(sys,
                           sim::OverdampedLangevin,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    if length(sys.constraints) > 0
        @warn "OverdampedLangevin is not currently compatible with constraints, " *
              "constraints will be ignored"
    end
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    noise = zero(sys.velocities)
    noise_prefac = sqrt((2 / sim.friction) * sim.dt)

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        forces!(forces_t, sys, neighbors, buffers, Val(false), step_n; n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        random_velocities!(noise, sys, sim.temperature; rng=rng)
        sys.coords .+= (accels_t ./ sim.friction) .* sim.dt .+ noise_prefac .* noise
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                   n_threads=n_threads)

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads,
                       current_forces=forces_t)
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads,
                        current_forces=forces_t)
            break
        end
        next_nograd!(progress)
    end
    return sys
end

"""
    NoseHoover(; <keyword arguments>)

The Nosé-Hoover integrator, a NVT simulator that extends velocity Verlet to control the
temperature of the system.

See [Evans and Holian 1985](https://doi.org/10.1063/1.449071).
The current implementation is limited to ergodic systems.

Not currently compatible with constraints, will print a warning and continue
without applying constraints.

# Arguments
- `dt::T`: the time step of the simulation.
- `temperature::K`: the equilibrium temperature of the simulation.
- `damping::D=100*dt`: the temperature damping time scale.
- `coupling::C=nothing`: the coupling which applies during the simulation.
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

function NoseHoover(; dt, temperature, damping=100*dt, coupling=nothing, remove_CM_motion=1)
    return NoseHoover(dt, temperature, damping, coupling, Int(remove_CM_motion))
end

@inline function simulate!(sys,
                           sim::NoseHoover,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    if length(sys.constraints) > 0
        @warn "NoseHoover is not currently compatible with constraints, " *
              "constraints will be ignored"
    end
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    forces_t, forces_t_dt = zero_forces(sys), zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    forces!(forces_t, sys, neighbors, buffers, Val(true), 0; n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    accels_t_dt = zero(accels_t)
    apply_loggers!(sys, buffers, neighbors, 0, run_loggers; n_threads=n_threads, current_forces=forces_t)
    v_half = zero(sys.velocities)
    zeta = zero(inv(sim.dt))
    dt_div2 = sim.dt / 2

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        needs_vir = (step_n % needs_vir_steps == 0)
        v_half .= sys.velocities .+ (accels_t .- (sys.velocities .* zeta)) .* dt_div2

        sys.coords .+= v_half .* sim.dt
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        zeta_half = zeta + (sim.dt / (2 * (sim.damping^2))) *
                        ((temperature(sys; kin_tensor=buffers.kin_tensor) / sim.temperature) - 1)
        KE_half = sum(masses(sys) .* sum.(abs2, v_half)) / 2
        T_half = uconvert(unit(sim.temperature), 2 * KE_half / (sys.df * sys.k))
        zeta = zeta_half + (sim.dt / (2 * (sim.damping^2))) * ((T_half / sim.temperature) - 1)

        forces!(forces_t_dt, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
        accels_t_dt .= calc_accels.(forces_t_dt, masses(sys))

        sys.velocities .= (v_half .+ accels_t_dt .* dt_div2) ./
                          (1 + (zeta * dt_div2))

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                    n_threads=n_threads)
        if recompute_forces
            forces!(forces_t_dt, sys, neighbors, buffers, Val(needs_vir), step_n; n_threads=n_threads)
            forces_t .= forces_t_dt
            accels_t .= calc_accels.(forces_t, masses(sys))
        else
            forces_t .= forces_t_dt
            accels_t .= accels_t_dt
        end

        apply_loggers!(sys, buffers, neighbors, step_n, run_loggers; n_threads=n_threads,
                       current_forces=forces_t)
        if shortcut_sim(shortcut, sys, buffers, neighbors, step_n; n_threads=n_threads,
                        current_forces=forces_t)
            break
        end
        next_nograd!(progress)
    end
    return sys
end

@doc raw"""
    ReplicaExchangeMD(; dt, exchange_time)

A generalized simulator for Replica Exchange Molecular Dynamics (REMD). 
Handles Temperature REMD, Hamiltonian REMD, or multi-dimensional combinations 
automatically based on the definitions in the `ThermoState`s of the `ReplicaSystem`.

# Arguments
- `dt::DT`: the time step of the simulation.
- `exchange_time::ET`: the time interval between replica exchange attempts.
"""
struct ReplicaExchangeMD{DT, ET}
    dt::DT
    exchange_time::ET
end

function ReplicaExchangeMD(; dt, exchange_time)
    if exchange_time <= dt
        throw(ArgumentError("exchange time ($exchange_time) must be greater than the time step ($dt)"))
    end
    return ReplicaExchangeMD(dt, exchange_time)
end

function simulate!(sys::ReplicaSystem,
                   sim::ReplicaExchangeMD,
                   n_steps::Integer;
                   assign_velocities::Bool=false,
                   n_threads::Integer=Threads.nthreads(),
                   run_loggers=true,
                   shortcut=nothing,
                   show_progress=default_show_progress(),
                   rng=Random.default_rng())
    
    if assign_velocities
        master_sys = sys.partition.master_sys
        k_B = master_sys.k
        e_unit = master_sys.energy_units
        
        # Extract the raw float value of the Boltzmann constant in internal units
        # If the system does not use units, k_B is already a raw float
        k_B_val = e_unit == NoUnits ? k_B : ustrip(uconvert(e_unit / u"K", k_B))
        
        for i in 1:sys.n_replicas
            state_idx = sys.state_indices[i]
            beta = sys.betas[state_idx]
            
            # Derive target temperature from internal beta: T = 1 / (k_B * beta)
            T_val = 1 / (k_B_val * beta)
            T_target = e_unit == NoUnits ? T_val : (T_val * u"K")
            
            # Assign random velocities directly to the replica's array
            random_velocities!(sys.replica_velocities[i], master_sys, T_target; rng=rng)
        end
    end

    return simulate_remd!(sys, sim, n_steps; n_threads=n_threads, run_loggers=run_loggers,
                          shortcut=shortcut, show_progress=show_progress, rng=rng)
end

@doc raw"""
    remd_exchange!(sys::ReplicaSystem, sim::ReplicaExchangeMD, i::Integer, j::Integer; <keyword arguments>)

Attempt a generalized replica exchange between physical replicas `i` and `j`. 
"""
function remd_exchange!(sys::ReplicaSystem,
                        sim::ReplicaExchangeMD,
                        i::Integer,
                        j::Integer;
                        rng=Random.default_rng())
    
    # Identify the current thermodynamic states (m and n) assigned to physical replicas i and j
    m = sys.state_indices[i]
    n = sys.state_indices[j]
    
    # Retrieve inverse temperatures directly from the betas array
    beta_m = sys.betas[m]
    beta_n = sys.betas[n]
    
    coords_i = sys.replica_coords[i]
    coords_j = sys.replica_coords[j]
    bound_i  = sys.replica_boundaries[i]
    bound_j  = sys.replica_boundaries[j]
    
    # Evaluate energies via AlchemicalPartition API
    U_m_xi = evaluate_energy!(sys.partition, coords_i, bound_i, m; force_recompute=false)
    U_n_xi = evaluate_energy!(sys.partition, coords_i, bound_i, n; force_recompute=false)
    
    U_n_xj = evaluate_energy!(sys.partition, coords_j, bound_j, n; force_recompute=false)
    U_m_xj = evaluate_energy!(sys.partition, coords_j, bound_j, m; force_recompute=false)
    
    # Strip units for Metropolis math
    e_unit = sys.partition.master_sys.energy_units
    U_m_xi_val = ustrip(e_unit, U_m_xi)
    U_n_xi_val = ustrip(e_unit, U_n_xi)
    U_n_xj_val = ustrip(e_unit, U_n_xj)
    U_m_xj_val = ustrip(e_unit, U_m_xj)
    
    # Generalized Metropolis Criterion
    delta = beta_n * U_n_xi_val - beta_m * U_m_xi_val + beta_m * U_m_xj_val - beta_n * U_n_xj_val
    
    should_exchange = delta <= 0 || rand(rng) < exp(-delta)
    
    if should_exchange
        # Swap the state assignments pointers
        sys.state_indices[i] = n
        sys.state_indices[j] = m
        
        # Rescale velocities to obey equipartition if the exchange involves a temperature differential
        if beta_m != beta_n
            sys.replica_velocities[i] .*= sqrt(beta_m / beta_n)
            sys.replica_velocities[j] .*= sqrt(beta_n / beta_m)
        end
    end
    
    return delta, should_exchange
end

@doc raw"""
    simulate_remd!(sys::ReplicaSystem, remd_sim::ReplicaExchangeMD, n_steps::Integer; <keyword arguments>)

Run a Replica Exchange Molecular Dynamics (REMD) simulation on a multiple-replica system.

The simulation divides the total `n_steps` into cycles based on the time step and exchange time specified in the `ReplicaExchangeMD` simulator. Within each cycle, standard molecular dynamics propagation is independently executed for each replica. At the end of every cycle, replica exchange attempts are made between neighboring states. Any remaining steps that do not fit evenly into the exchange cycles are executed at the end of the run.

# Arguments
- `sys::ReplicaSystem`: the partitioned system containing the replicas and thermodynamic states.
- `remd_sim::ReplicaExchangeMD`: the simulator containing the specific time step and exchange time interval.
- `n_steps::Integer`: the total number of MD steps to simulate for each replica.
- `n_threads::Integer=Threads.nthreads()`: the total number of threads to use, which are equally partitioned among the individual replicas.
- `run_loggers=true`: whether to run the loggers during the simulation, including the exchange logger.
- `rng=Random.default_rng()`: the random number generator used for the exchange accept/reject criteria and any stochastic dynamics.
"""
function simulate_remd!(sys::ReplicaSystem,
                        remd_sim::ReplicaExchangeMD,
                        n_steps::Integer;
                        n_threads::Integer=Threads.nthreads(),
                        run_loggers=true,
                        shortcut=nothing, # Unused
                        show_progress=default_show_progress(),
                        rng=Random.default_rng())
    
    thread_div = equal_parts(n_threads, sys.n_replicas)

    n_cycles = convert(Int, (n_steps * remd_sim.dt) ÷ remd_sim.exchange_time)
    cycle_length = n_cycles > 0 ? n_steps ÷ n_cycles : 0
    remaining_steps = n_cycles > 0 ? n_steps % n_cycles : n_steps
    n_attempts = 0

    progress = setup_progress(n_steps, show_progress)
    for cycle in 1:n_cycles
        @sync for i in 1:sys.n_replicas
            state_idx = sys.state_indices[i]
            integrator = sys.integrators[state_idx]
            
            # Construct active_sys with the FULL interaction lists for standard MD forces
            active_sys = System(sys.partition.master_sys;
                coords = sys.replica_coords[i],
                velocities = sys.replica_velocities[i],
                boundary = sys.replica_boundaries[i],
                atoms = sys.partition.λ_atoms[state_idx],
                pairwise_inters = sys.state_pairwise_inters[state_idx],
                specific_inter_lists = sys.state_specific_inter_lists[state_idx],
                general_inters = sys.state_general_inters[state_idx],
                neighbor_finder = sys.replica_neighbor_finders[i],
                loggers = sys.replica_loggers[i]
            )
            
            # Enforce n_threads >= 1 to prevent buffer chunk crashes
            Threads.@spawn simulate!(active_sys, integrator, cycle_length;
                                     n_threads=max(1, thread_div[i]), run_loggers=run_loggers, rng=rng)
        end

        cycle_parity = cycle % 2
        for n in (1 + cycle_parity):2:(sys.n_replicas - 1)
            n_attempts += 1
            m = n + 1
            Δ, exchanged = remd_exchange!(sys, remd_sim, n, m; rng=rng)
            
            if run_loggers != false && exchanged && !isnothing(sys.exchange_logger)
                log_property!(sys.exchange_logger, sys, nothing, nothing, cycle * cycle_length;
                              indices=(n, m), delta=Δ, n_threads=n_threads)
            end
        end
        next_nograd!(progress)
    end

    if remaining_steps > 0
        @sync for i in 1:sys.n_replicas
            state_idx = sys.state_indices[i]
            integrator = sys.integrators[state_idx]
            
            active_sys = System(sys.partition.master_sys;
                coords = sys.replica_coords[i],
                velocities = sys.replica_velocities[i],
                boundary = sys.replica_boundaries[i],
                atoms = sys.partition.λ_atoms[state_idx],
                pairwise_inters = sys.state_pairwise_inters[state_idx],
                specific_inter_lists = sys.state_specific_inter_lists[state_idx],
                general_inters = sys.state_general_inters[state_idx],
                neighbor_finder = sys.replica_neighbor_finders[i],
                loggers = sys.replica_loggers[i]
            )
            
            Threads.@spawn simulate!(active_sys, integrator, remaining_steps;
                                     n_threads=max(1, thread_div[i]), run_loggers=run_loggers, rng=rng)
        end
    end

    if run_loggers != false && !isnothing(sys.exchange_logger)
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

@inline function simulate!(sys::System,
                           sim::MetropolisMonteCarlo,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng())
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    buffers = init_buffers!(sys, n_threads)
    E_old = potential_energy(sys, neighbors, buffers; n_threads=n_threads)
    coords_old = zero(sys.coords)

    progress = setup_progress(n_steps, show_progress)
    for step_n in 1:n_steps
        coords_old .= sys.coords
        sim.trial_moves(sys; sim.trial_args...) # Changes the coordinates of the system
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)
        neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        E_new = potential_energy(sys, neighbors, buffers, step_n; n_threads=n_threads)

        ΔE = E_new - E_old
        δ = ΔE / (sys.k * sim.temperature)
        if δ < 0 || (rand(rng) < exp(-δ))
            apply_loggers!(sys, nothing, neighbors, step_n, run_loggers; n_threads=n_threads,
                           current_potential_energy=E_new, success=true,
                           energy_rate=(E_new / (sys.k * sim.temperature)))
            E_old = E_new
        else
            sys.coords .= coords_old
            apply_loggers!(sys, nothing, neighbors, step_n, run_loggers; n_threads=n_threads,
                           current_potential_energy=E_old, success=false,
                           energy_rate=(E_old / (sys.k * sim.temperature)))
        end
        if shortcut_sim(shortcut, sys, nothing, neighbors, step_n; n_threads=n_threads)
            break
        end
        next_nograd!(progress)
    end

    return sys
end

"""
    random_uniform_translation!(sys::System; shift_size=oneunit(eltype(eltype(sys.coords))),
                                rng=Random.default_rng())

Performs a random translation of the coordinates of a randomly selected atom in a [`System`](@ref).

The translation is generated using a uniformly selected direction and uniformly selected length
in range [0, 1) scaled by `shift_size` which should have appropriate length units.
"""
function random_uniform_translation!(sys::System{D, <:Any, T};
                                     shift_size=oneunit(eltype(eltype(sys.coords))),
                                     rng=Random.default_rng()) where {D, T}
    rand_idx = pick_non_virtual_site(rng, sys)
    direction = random_unit_vector(T, D, rng)
    magnitude = rand(rng, T) * shift_size
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
    return sys
end

"""
    random_normal_translation!(sys::System; shift_size=oneunit(eltype(eltype(sys.coords))),
                               rng=Random.default_rng())

Performs a random translation of the coordinates of a randomly selected atom in a [`System`](@ref).

The translation is generated using a uniformly chosen direction and length selected from
the standard normal distribution i.e. with mean 0 and standard deviation 1, scaled by `shift_size`
which should have appropriate length units.
"""
function random_normal_translation!(sys::System{D, <:Any, T};
                                    shift_size=oneunit(eltype(eltype(sys.coords))),
                                    rng=Random.default_rng()) where {D, T}
    rand_idx = pick_non_virtual_site(rng, sys)
    direction = random_unit_vector(T, D, rng)
    magnitude = randn(rng, T) * shift_size
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
    return sys
end

function random_unit_vector(T, dims, rng=Random.default_rng())
    vec = randn(rng, T, dims)
    return vec / norm(vec)
end
