# Different ways to simulate molecules

export
    SteepestDescentMinimizer,
    simulate!,
    VelocityVerlet,
    DPDVelocityVerlet,
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

calc_n_steps(n_steps::Integer, dt) = n_steps

function calc_n_steps(sim_time::Number, dt)
    if dimension(sim_time) != dimension(dt)
        throw(ArgumentError("simulation time ($sim_time) and simulator time step ($dt) " *
                            "must have the same dimension, you may have meant to pass an " *
                            "integer number of steps instead of a non-integer time step"))
    end
    return Int(cld(sim_time, dt))
end

function check_init_step(init_step::Integer)
    init_step >= 0 || throw(ArgumentError("init_step must be non-negative"))
    return Int(init_step)
end

initial_logger_mode(run_loggers, log_initial_state::Bool) =
    log_initial_state ? run_loggers : false

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
- `max_steps::Int=1_000`: the maximum number of steps.
- `tol::F=1_000.0u"kJ * mol^-1 * nm^-1"`: the maximum force below which to
    finish minimization.
- `constraint_bond_constant::K=500_000.0u"kJ * mol^-1 * nm^-2"`: the force constant
    for the harmonic bonds that are used instead of constraints during
    minimisation. Set to `nothing` to not use harmonic bonds and ignore
    constraints. Unused if the system does not have constraints.
- `log_stream::L=devnull`: stream to print minimization progress to.
"""
@kwdef struct SteepestDescentMinimizer{D, F, K, L}
    step_size::D = 0.01u"nm"
    max_steps::Int = 1_000
    tol::F = 1_000.0u"kJ * mol^-1 * nm^-1"
    constraint_bond_constant::K = 500_000.0u"kJ * mol^-1 * nm^-2"
    log_stream::L = devnull
end

"""
    simulate!(system, simulator, n_steps::Integer; <keyword arguments>)
    simulate!(system, simulator, sim_time; <keyword arguments>)
    simulate!(system, simulator; <keyword arguments>)
    simulate!(awh_sim::AWHSimulation, n_steps)

Run a simulation on a system according to the rules of the given simulator.

For simulators that run for a given period of time, the third argument can either be an
`Integer` number of steps or a simulation time (e.g. `10.0u"ns"` or `10.0` if not using units).
Custom simulators should implement this function.
Constraints are applied during minimization, which can lead to issues.

# Arguments
- `n_threads=Threads.nthreads()`: the number of threads to run the simulation on, only
    relevant when running on CPU.
- `run_loggers`: whether to run the loggers during the simulation. Can be `true`, `false`
    or `:skipzero`, in which case the loggers are not run before the first step. `run_loggers`
    is `true` by default except for [`SteepestDescentMinimizer`](@ref), where it is `false`.
- `shortcut=nothing`: when to stop the simulation early. A struct with the `shortcut_sim`
    method defined can be provided. `shortcut_sim` is checked at the end of each step.
    Unused for REMD simulations.
- `init_step=0`: the step number before the first step is taken, useful for time-dependent
    potentials.
- `log_initial_state=true`: whether to run loggers for the state at `init_step`. Set this
    to `false` when continuing a simulation in blocks to avoid duplicate boundary records.
- `show_progress`: whether to show a progress bar for the simulation. `true` by default in
    the REPL/IJulia/Pluto, otherwise `false` by default. Can be set globally with the
    environmental variable `MOLLY_SHOW_PROGRESS`.
- `rng=Random.default_rng()`: the random number generator used for the simulation. Setting
    this allows reproducible stochastic simulations.
- `strictness=:warn`: determines behavior when encountering possible problems,
    options are `:warn` to emit warnings, `:nowarn` to suppress warnings or
    `:error` to error.

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
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    # @inline needed to avoid Enzyme error
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints && !isnothing(sim.constraint_bond_constant)
        constraint_bonds = constraints_to_bonds(sys, sim.constraint_bond_constant)
        if length(constraint_bonds) > 0
            sis = (sys.specific_inter_lists..., constraint_bonds)
        else
            constraint_bonds = constraints_to_bonds(sys, sim.constraint_bond_constant)
            if length(constraint_bonds) > 0
                sis = (sys.specific_inter_lists..., constraint_bonds)
            else
                sis = sys.specific_inter_lists
            end
        end
    else
        sis = sys.specific_inter_lists
    end

    needs_vir = false
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    buffers = init_buffers!(sys, n_threads)
    E = potential_energy(sys, neighbors, init_step, buffers; n_threads=n_threads,
                         specific_inter_lists=sis)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads,
                   current_potential_energy=E)
    println(sim.log_stream, "Step ", init_step, " - potential energy ", E,
            " - max force N/A - N/A")
    hn = sim.step_size
    coords_copy = zero(sys.coords)
    F = zero_forces(sys)

    progress = setup_progress_minimizer(ustrip(sim.tol), show_progress)
    for step_n in (init_step + 1):(init_step + sim.max_steps)
        forces!(F, sys, neighbors, step_n, buffers, Val(needs_vir); n_threads=n_threads,
                                                    specific_inter_lists=sis)
        max_force = maximum(norm.(F))

        coords_copy .= sys.coords
        sys.coords .+= hn .* F ./ max_force
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        neighbors_copy = neighbors
        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                    n_threads=n_threads)
        E_trial = potential_energy(sys, neighbors, step_n, buffers; n_threads=n_threads,
                                                            specific_inter_lists=sis)
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

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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

function virial_schedule_from_steps(steps)
    isempty(steps) && return false, Inf
    smin = minimum(steps)
    for s in steps
        if s % smin != 0
            throw(ArgumentError("incompatible virial step interval $steps, all must be " *
                                "multiples of the minimum interval $smin"))
        end
    end
    return true, smin
end

function logger_due_on_step(logger_interval, step_n::Integer, run_loggers)
    run_loggers == false && return false
    run_loggers == :skipzero && step_n == 0 && return false
    return !isinf(logger_interval) && step_n % logger_interval == 0
end

function virial_logger_due_on_step(loggers, step_n::Integer, run_loggers)
    return any(logger -> logger_due_on_step(logger_virial_interval(logger), step_n,
                                            run_loggers), loggers)
end

function pressure_logger_due_on_step(loggers, step_n::Integer, run_loggers)
    return any(logger -> logger_due_on_step(logger_pressure_interval(logger), step_n,
                                            run_loggers), loggers)
end

function needs_virial_schedule(coupling, loggers, run_loggers)
    steps = Int[]
    coupling_needs_virial, coupling_steps = needs_virial_schedule(coupling)
    if coupling_needs_virial
        push!(steps, Int(coupling_steps))
    end
    if run_loggers != false
        for logger in loggers
            logger_steps = logger_virial_interval(logger)
            if !isinf(logger_steps)
                push!(steps, Int(logger_steps))
            end
        end
    end
    return virial_schedule_from_steps(steps)
end

needs_virial_on_step(needs_virial::Bool, steps, step_n::Integer) =
    needs_virial && step_n % steps == 0

may_recompute_forces_after_coupling(coupler) = true
may_recompute_forces_after_coupling(::Nothing) = false
may_recompute_forces_after_coupling(couplers::Union{Tuple, NamedTuple}) =
    any(may_recompute_forces_after_coupling, couplers)

function coupling_may_recompute_on_step(coupler, step_n::Integer)
    may_recompute_forces_after_coupling(coupler) || return false
    if hasproperty(coupler, :n_steps)
        return step_n % getproperty(coupler, :n_steps) == 0
    end
    return true
end

coupling_may_recompute_on_step(couplers::Union{Tuple, NamedTuple}, step_n::Integer) =
    any(coupler -> coupling_may_recompute_on_step(coupler, step_n), couplers)
coupling_may_recompute_on_step(::Nothing, step_n::Integer) = false

function save_pre_coupling_virial_for_loggers!(buffers, sys, coupling, step_n::Integer,
                                               pressure_kin_tensor, run_loggers)
    coupling_may_recompute_on_step(coupling, step_n) || return buffers
    length(sys.constraints) > 0 || return buffers
    has_total_virial(buffers, step_n) || return buffers
    virial_logger_due_on_step(values(sys.loggers), step_n, run_loggers) || return buffers

    save_pre_coupling_virial!(buffers, step_n)
    if pressure_logger_due_on_step(values(sys.loggers), step_n, run_loggers)
        save_pre_coupling_pressure!(buffers, sys, step_n, pressure_kin_tensor)
    end
    return buffers
end

constraint_virial_integrator_factor(sim) = 1
constraint_virial_integrator_factor(sim::VelocityVerlet) = 2

function position_constraint_virial_scale(sys, buffers, dt)
    e_unit = unit(eltype(buffers.constraint_virial))
    if e_unit == NoUnits
        return inv(dt^2)
    else
        m_unit = unit(eltype(masses(sys)))
        x_unit = unit(eltype(eltype(sys.coords)))
        raw_unit = m_unit * x_unit^2
        return uconvert(e_unit, raw_unit * inv(dt^2))
    end
end

position_constraint_virial_scale(sys::System{D, AT, T}, buffers, dt, sim) where {D, AT, T} =
    T(constraint_virial_integrator_factor(sim)) * position_constraint_virial_scale(sys, buffers, dt)

function velocity_constraint_virial_scale(sys, buffers, dt)
    e_unit = unit(eltype(buffers.constraint_virial))
    if e_unit == NoUnits
        return inv(dt)
    else
        m_unit = unit(eltype(masses(sys)))
        x_unit = unit(eltype(eltype(sys.coords)))
        v_unit = unit(eltype(eltype(sys.velocities)))
        raw_unit = m_unit * x_unit * v_unit
        return uconvert(e_unit, raw_unit * inv(dt))
    end
end

velocity_constraint_virial_scale(sys::System{D, AT, T}, buffers, dt, sim) where {D, AT, T} =
    T(constraint_virial_integrator_factor(sim)) * velocity_constraint_virial_scale(sys, buffers, dt)

function position_constraint_context(buffers, sys, step_n::Integer, dt, needs_virial::Bool,
                                     sim=nothing)
    return ConstraintApplicationContext(
        kind=PositionConstraintApplication(),
        needs_virial=needs_virial,
        step_n=Int(step_n),
        atoms=sys.atoms,
        dt=dt,
        virial_scale=position_constraint_virial_scale(sys, buffers, dt, sim),
        buffers=buffers,
        coords_buffer=buffers.constraint_coords_buffer,
    )
end

function velocity_constraint_context(buffers, sys, step_n::Integer, dt, needs_virial::Bool,
                                     sim=nothing)
    return ConstraintApplicationContext(
        kind=VelocityConstraintApplication(),
        needs_virial=needs_virial,
        step_n=Int(step_n),
        atoms=sys.atoms,
        dt=dt,
        virial_scale=velocity_constraint_virial_scale(sys, buffers, dt, sim),
        buffers=buffers,
        velocities_buffer=buffers.constraint_velocities_buffer,
    )
end

function prepare_constraint_virial!(buffers, sys, step_n::Integer, needs_virial::Bool)
    if needs_virial && length(sys.constraints) > 0
        clear_constraint_virial!(buffers, sys, step_n)
    end
    return buffers
end

function merge_constraint_virial_if_needed!(buffers, sys, step_n::Integer,
                                            needs_virial::Bool)
    if needs_virial && length(sys.constraints) > 0
        merge_constraint_virial!(buffers, sys, step_n)
    end
    return buffers
end

function default_constraint_preview_dt(sys)
    T = typeof(ustrip(oneunit(eltype(eltype(sys.coords)))))
    return sys.energy_units == NoUnits ? T(0.0005) : T(0.0005)u"ps"
end

function merge_initial_constraint_virial!(buffers, sys, step_n::Integer, needs_virial::Bool,
                                          current_forces; n_threads::Integer=Threads.nthreads(),
                                          dt=default_constraint_preview_dt(sys))
    if needs_virial && length(sys.constraints) > 0
        coords = copyto_constraint_scratch!(buffers.constraint_preview_coords_buffer, sys.coords)
        velocities = copyto_constraint_scratch!(buffers.constraint_velocities_buffer, sys.velocities)
        accels = calc_accels.(current_forces, masses(sys))

        clear_constraint_virial!(buffers, sys, step_n)
        sys.coords .+= sys.velocities .* dt .+ (accels .* dt^2) ./ 2
        pos_context = position_constraint_context(buffers, sys, step_n, dt, true)
        apply_position_constraints!(sys, coords; context=pos_context, n_threads=n_threads)
        merge_constraint_virial!(buffers, sys, step_n)

        sys.coords .= coords
        sys.velocities .= velocities
    end
    return buffers
end

function compute_initial_total_virial!(buffers, sys, neighbors, step_n::Integer;
                                       n_threads::Integer=Threads.nthreads(), kwargs...)
    forces_t = zero_forces(sys)
    forces!(forces_t, sys, neighbors, step_n, buffers, Val(true);
            n_threads=n_threads, kwargs...)
    merge_initial_constraint_virial!(buffers, sys, step_n, true, forces_t;
                                     n_threads=n_threads)
    return forces_t, buffers
end

function recompute_forces_after_coupling!(forces_out, sys, neighbors, buffers, step_n::Integer,
                                          needs_virial::Bool;
                                          n_threads::Integer=Threads.nthreads())
    needs_current_virial = needs_virial && length(sys.constraints) == 0
    forces!(forces_out, sys, neighbors, step_n, buffers, Val(needs_current_virial);
            n_threads=n_threads)
    return forces_out
end

@inline function simulate!(sys,
                           sim::VelocityVerlet,
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling, sys.loggers, run_loggers)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    init_step == 0 && !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t, forces_t_dt = zero_forces(sys), zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    needs_vir_init = needs_virial_on_step(needs_vir, needs_vir_steps, init_step)
    forces!(forces_t, sys, neighbors, init_step, buffers, Val(needs_vir_init);
            n_threads=n_threads)
    merge_initial_constraint_virial!(buffers, sys, init_step, needs_vir_init, forces_t;
                                     n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    accels_t_dt = zero(accels_t)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads,
                   current_forces=forces_t)
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints
        cons_coord_storage = zero(sys.coords)
        cons_vel_storage = zero(sys.velocities)
    end
    dt_div2 = sim.dt / 2
    pressure_kin_tensor = zero(buffers.kin_tensor)

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
        needs_vir_step = needs_virial_on_step(needs_vir, needs_vir_steps, step_n)
        fill!(pressure_kin_tensor, zero(eltype(pressure_kin_tensor)))
        pressure_kin_tensor_valid = false

        sys.velocities .+= accels_t .* dt_div2
        if using_constraints
            vel_context = velocity_constraint_context(buffers, sys, step_n, sim.dt,
                                                      false, sim)
            apply_velocity_constraints!(sys; context=vel_context, n_threads=n_threads)
            cons_coord_storage .= sys.coords
        end

        sys.coords .+= sys.velocities .* sim.dt
        if using_constraints
            pos_context = position_constraint_context(buffers, sys, step_n, sim.dt,
                                                      false, sim)
            apply_position_constraints!(sys, cons_coord_storage, cons_vel_storage, sim.dt;
                                        context=pos_context, n_threads=n_threads)
        end
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        forces!(forces_t_dt, sys, neighbors, step_n, buffers, Val(needs_vir_step);
                n_threads=n_threads)
        accels_t_dt .= calc_accels.(forces_t_dt, masses(sys))

        sys.velocities .+= accels_t_dt .* dt_div2
        if using_constraints
            prepare_constraint_virial!(buffers, sys, step_n, needs_vir_step)
            vel_context = velocity_constraint_context(buffers, sys, step_n, sim.dt,
                                                      needs_vir_step, sim)
            apply_velocity_constraints!(sys; context=vel_context, n_threads=n_threads)
            merge_constraint_virial_if_needed!(buffers, sys, step_n, needs_vir_step)
        end

        # Remove drift after the final velocity constraints/virial accumulation
        # and before the kinetic tensor snapshot used for pressure coupling.
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        if using_constraints && needs_vir_step
            kinetic_energy_tensor!(buffers.kin_tensor, sys)
            pressure_kin_tensor .= buffers.kin_tensor
            pressure_kin_tensor_valid = true
        end

        save_pre_coupling_virial_for_loggers!(
            buffers, sys, sim.coupling, step_n,
            pressure_kin_tensor_valid ? pressure_kin_tensor : nothing, run_loggers)
        recompute_forces = apply_coupling_with_pressure_kin_tensor!(
            sys, buffers, sim.coupling, sim, neighbors, step_n,
            pressure_kin_tensor_valid ? pressure_kin_tensor : nothing; n_threads=n_threads,
            rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)
        if recompute_forces
            recompute_forces_after_coupling!(forces_t_dt, sys, neighbors, buffers, step_n,
                                             needs_vir_step; n_threads=n_threads)
            forces_t .= forces_t_dt
            accels_t .= calc_accels.(forces_t, masses(sys))
        else
            forces_t .= forces_t_dt
            accels_t .= accels_t_dt
        end

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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
    DPDVelocityVerlet(; <keyword arguments>)

The modified velocity Verlet integrator for dissipative particle dynamics.

Implements the Groot-Warren modified velocity-Verlet (MVV) algorithm from
[Groot and Warren 1997](https://doi.org/10.1063/1.474784).

Because DPD dissipative forces depend on particle velocities, a velocity
prediction step is used before recomputing forces at the new positions.
The `λ` parameter controls this prediction: `v_predicted = v(t) + λ * dt * a(t)`.
A value of 0.65 is commonly used.

Should be used with [`DPDInteraction`](@ref) as the pairwise interaction.

# Arguments
- `dt::T`: the time step of the simulation.
- `λ::L=0.65`: the velocity prediction parameter (typically 0.5–0.65).
- `coupling::C=nothing`: the coupling which applies during the simulation.
- `remove_CM_motion=1`: remove the center of mass motion every this number of steps,
    set to `false` or `0` to not remove center of mass motion.
"""
struct DPDVelocityVerlet{T, L, C}
    dt::T
    λ::L
    coupling::C
    remove_CM_motion::Int
end

function DPDVelocityVerlet(; dt, λ=0.65, coupling=nothing, remove_CM_motion=1)
    return DPDVelocityVerlet(dt, λ, coupling, Int(remove_CM_motion))
end

constraint_virial_integrator_factor(sim::DPDVelocityVerlet) = 2

@inline function simulate!(sys,
                           sim::DPDVelocityVerlet,
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling, sys.loggers, run_loggers)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    init_step == 0 && !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t, forces_t_dt = zero_forces(sys), zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    needs_vir_init = needs_virial_on_step(needs_vir, needs_vir_steps, init_step)
    forces!(forces_t, sys, neighbors, init_step, buffers, Val(needs_vir_init);
            n_threads=n_threads)
    merge_initial_constraint_virial!(buffers, sys, init_step, needs_vir_init, forces_t;
                                     n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    accels_t_dt = zero(accels_t)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads,
                   current_forces=forces_t)
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints
        cons_coord_storage = zero(sys.coords)
        cons_vel_storage = zero(sys.velocities)
    end
    velocities_half = zero(sys.velocities)
    dt_div2 = sim.dt / 2
    λ_shift_dt = (sim.λ - 1//2) * sim.dt
    pressure_kin_tensor = zero(buffers.kin_tensor)

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
        needs_vir_step = needs_virial_on_step(needs_vir, needs_vir_steps, step_n)
        fill!(pressure_kin_tensor, zero(eltype(pressure_kin_tensor)))
        pressure_kin_tensor_valid = false

        sys.velocities .+= accels_t .* dt_div2
        if using_constraints
            vel_context = velocity_constraint_context(buffers, sys, step_n, sim.dt,
                                                      false, sim)
            apply_velocity_constraints!(sys; context=vel_context, n_threads=n_threads)
            cons_coord_storage .= sys.coords
        end

        sys.coords .+= sys.velocities .* sim.dt
        if using_constraints
            pos_context = position_constraint_context(buffers, sys, step_n, sim.dt,
                                                      false, sim)
            apply_position_constraints!(sys, cons_coord_storage, cons_vel_storage, sim.dt;
                                        context=pos_context, n_threads=n_threads)
        end
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        velocities_half .= sys.velocities

        # DPD dissipative forces depend on velocity. Temporarily use the
        # Groot-Warren predicted velocity for the force evaluation, then restore
        # the real half-step velocity below for the final VV update.
        sys.velocities .= velocities_half .+ accels_t .* λ_shift_dt
        if using_constraints
            vel_context = velocity_constraint_context(buffers, sys, step_n, sim.dt,
                                                      false, sim)
            apply_velocity_constraints!(sys; context=vel_context, n_threads=n_threads)
        end
        forces!(forces_t_dt, sys, neighbors, step_n, buffers, Val(needs_vir_step);
                n_threads=n_threads)
        accels_t_dt .= calc_accels.(forces_t_dt, masses(sys))

        sys.velocities .= velocities_half .+ accels_t_dt .* dt_div2
        if using_constraints
            prepare_constraint_virial!(buffers, sys, step_n, needs_vir_step)
            vel_context = velocity_constraint_context(buffers, sys, step_n, sim.dt,
                                                      needs_vir_step, sim)
            apply_velocity_constraints!(sys; context=vel_context, n_threads=n_threads)
            merge_constraint_virial_if_needed!(buffers, sys, step_n, needs_vir_step)
        end

        # Remove drift after the final velocity constraints/virial accumulation
        # and before the kinetic tensor snapshot used for pressure coupling.
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        if using_constraints && needs_vir_step
            kinetic_energy_tensor!(buffers.kin_tensor, sys)
            pressure_kin_tensor .= buffers.kin_tensor
            pressure_kin_tensor_valid = true
        end

        save_pre_coupling_virial_for_loggers!(
            buffers, sys, sim.coupling, step_n,
            pressure_kin_tensor_valid ? pressure_kin_tensor : nothing, run_loggers)
        recompute_forces = apply_coupling_with_pressure_kin_tensor!(
            sys, buffers, sim.coupling, sim, neighbors, step_n,
            pressure_kin_tensor_valid ? pressure_kin_tensor : nothing; n_threads=n_threads,
            rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)
        if recompute_forces
            recompute_forces_after_coupling!(forces_t_dt, sys, neighbors, buffers, step_n,
                                             needs_vir_step; n_threads=n_threads)
            forces_t .= forces_t_dt
            accels_t .= calc_accels.(forces_t, masses(sys))
        else
            forces_t .= forces_t_dt
            accels_t .= accels_t_dt
        end

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling, sys.loggers, run_loggers)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    init_step == 0 && !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    needs_vir_init = needs_virial_on_step(needs_vir, needs_vir_steps, init_step)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    if needs_vir_init
        forces!(forces_t, sys, neighbors, init_step, buffers, Val(true); n_threads=n_threads)
        merge_initial_constraint_virial!(buffers, sys, init_step, true, forces_t;
                                         n_threads=n_threads)
        apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads,
                       current_forces=forces_t)
    else
        apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads)
    end
    accels_t = calc_accels.(forces_t, masses(sys))
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints
        cons_coord_storage = zero(sys.coords)
    end

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
        needs_vir_step = needs_virial_on_step(needs_vir, needs_vir_steps, step_n)
        forces!(forces_t, sys, neighbors, step_n, buffers, Val(needs_vir_step);
                n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        sys.velocities .+= accels_t .* sim.dt

        if using_constraints
            cons_coord_storage .= sys.coords
            prepare_constraint_virial!(buffers, sys, step_n, needs_vir_step)
        end
        sys.coords .+= sys.velocities .* sim.dt
        if using_constraints
            pos_context = position_constraint_context(buffers, sys, step_n, sim.dt,
                                                      needs_vir_step, sim)
            apply_position_constraints!(sys, cons_coord_storage; context=pos_context,
                                        n_threads=n_threads)
        end

        if using_constraints
            apply_position_constraints!(sys, cons_coord_storage; n_threads=n_threads)
            sys.velocities .= (sys.coords .- cons_coord_storage) ./ sim.dt
            merge_constraint_virial_if_needed!(buffers, sys, step_n, needs_vir_step)
        end

        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        # Remove drift after the step velocity is finalized and before
        # coupling/loggers observe the state.
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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

Position constraints are supported. Coupling methods are intentionally
unsupported.

# Arguments
- `dt::T`: the time step of the simulation.
"""
@kwdef struct StormerVerlet{T}
    dt::T
end

@inline function simulate!(sys,
                           sim::StormerVerlet,
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    needs_vir, needs_vir_steps = needs_virial_schedule(nothing, sys.loggers, run_loggers)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    needs_vir_init = needs_virial_on_step(needs_vir, needs_vir_steps, init_step)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    if needs_vir_init
        forces!(forces_t, sys, neighbors, init_step, buffers, Val(true); n_threads=n_threads)
        merge_initial_constraint_virial!(buffers, sys, init_step, true, forces_t;
                                         n_threads=n_threads)
        apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads,
                       current_forces=forces_t)
    else
        apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads)
    end
    coords_last, coords_copy = zero(sys.coords), zero(sys.coords)
    accels_t = calc_accels.(forces_t, masses(sys))
    using_constraints = (length(sys.constraints) > 0)
    dt_sq = sim.dt^2

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
        needs_vir_step = needs_virial_on_step(needs_vir, needs_vir_steps, step_n)
        forces!(forces_t, sys, neighbors, step_n, buffers, Val(needs_vir_step);
                n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        coords_copy .= sys.coords
        prepare_constraint_virial!(buffers, sys, step_n, needs_vir_step)
        if step_n == init_step + 1
            # Use the velocities at the first step since there is only one set of coordinates.
            sys.coords .+= sys.velocities .* sim.dt .+ (accels_t .* dt_sq) ./ 2
        else
            # After the first step, coordinates are advanced from the previous
            # and current coordinate arrays rather than primary velocity state.
            sys.coords .+= vector.(coords_last, sys.coords, (sys.boundary,)) .+ accels_t .* dt_sq
        end

        if using_constraints
            pos_context = position_constraint_context(buffers, sys, step_n, sim.dt,
                                                      needs_vir_step, sim)
            apply_position_constraints!(sys, coords_copy; context=pos_context,
                                        n_threads=n_threads)
            merge_constraint_virial_if_needed!(buffers, sys, step_n, needs_vir_step)
        end

        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)
        # This is accurate to O(dt)
        sys.velocities .= zero_vs_velocity.(
            vector.(coords_copy, sys.coords, (sys.boundary,)) ./ sim.dt,
            sys.virtual_site_flags,
        )

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, false;
                                   n_threads=n_threads)
        coords_last .= coords_copy

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling, sys.loggers, run_loggers)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    init_step == 0 && !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    needs_vir_init = needs_virial_on_step(needs_vir, needs_vir_steps, init_step)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    if needs_vir_init
        forces!(forces_t, sys, neighbors, init_step, buffers, Val(true); n_threads=n_threads)
        merge_initial_constraint_virial!(buffers, sys, init_step, true, forces_t;
                                         n_threads=n_threads)
        apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads,
                       current_forces=forces_t)
    else
        apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads)
    end
    accels_t = calc_accels.(forces_t, masses(sys))
    noise = zero(sys.velocities)
    vel_zero = zero(eltype(eltype(sys.velocities)))
    kT = sim.temperature*sys.k
    noise_scale = sim.noise_scale
    # precompute to avoid the sqrt and division at each step
    # capture value-typed locals (not the velocity element Type, which is not
    # isbits) so this map compiles into a GPU kernel
    noise_scales = map(sys.masses, sys.virtual_site_flags) do m, vsf
        ifelse(vsf, vel_zero, oftype(vel_zero, noise_scale * sqrt(kT/m)))
    end
    # Seed the per step noise
    philox_key = rand(rng, UInt64)
    philox_ctr1 = rand(rng, UInt64)
    using_constraints = (length(sys.constraints) > 0)
    if using_constraints
        cons_coord_storage = zero(sys.coords)
        cons_vel_storage = zero(sys.velocities)
    end
    dt_div2 = sim.dt / 2

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
        needs_vir_step = needs_virial_on_step(needs_vir, needs_vir_steps, step_n)
        forces!(forces_t, sys, neighbors, step_n, buffers, Val(needs_vir_step);
                n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        sys.velocities .+= accels_t .* sim.dt
        if using_constraints
            prepare_constraint_virial!(buffers, sys, step_n, needs_vir_step)
            vel_context = velocity_constraint_context(buffers, sys, step_n, sim.dt,
                                                      needs_vir_step, sim)
            apply_velocity_constraints!(sys; context=vel_context, n_threads=n_threads)
        end

        if using_constraints
            cons_coord_storage .= sys.coords
        end
        # On GPU fuse the thread per atom work into one kernel
        # coords[i] = coords[i] + vels[i]*dt/2
        # vels[i] = vels[i]*vel_scale + noise[i]*noise_scales[i]
        # coords[i] = coords[i] + vels[i]*dt/2
        langevin_per_atom_inner!(
            sys.coords,
            sys.velocities,
            dt_div2,
            sim.vel_scale,
            noise_scales,
            philox_ctr1,
            philox_key,
            float_type(sys),
        )
        philox_ctr1 += UInt64(1)

        if using_constraints
            pos_context = position_constraint_context(buffers, sys, step_n, sim.dt,
                                                      needs_vir_step, sim)
            apply_position_constraints!(sys, cons_coord_storage, cons_vel_storage, sim.dt;
                                        context=pos_context, n_threads=n_threads)
            merge_constraint_virial_if_needed!(buffers, sys, step_n, needs_vir_step)
        end
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        # Remove drift after the step velocity is finalized and before
        # coupling/loggers observe the state.
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                   n_threads=n_threads)

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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
    return LangevinSplitting{typeof(dt), typeof(temperature), typeof(friction), typeof(splitting)}(
                    dt, temperature, friction, splitting, Int(remove_CM_motion))
end

@inline function simulate!(sys,
                           sim::LangevinSplitting,
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    if length(sys.constraints) > 0
        err_str = "LangevinSplitting is not currently compatible with constraints, " *
                  "constraints will be ignored"
        report_issue(err_str, strictness)
    end
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    M_inv = inv.(masses(sys))
    α_eff = exp.(-sim.friction * sim.dt .* M_inv / count('O', sim.splitting))
    σ_eff = sqrt.((1 * unit(eltype(α_eff))) .- (α_eff .^ 2))

    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    init_step == 0 && !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads)
    forces!(forces_t, sys, neighbors, init_step, buffers, Val(false); n_threads=n_threads)
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
    for step_n in (init_step + 1):(init_step + n_steps)
        for (step!, args) in step_arg_pairs
            step!(args..., neighbors, step_n)
        end

        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)
        # Remove drift after all splitting substeps and before loggers observe
        # the state.
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                   n_threads=n_threads)

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads)
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
        forces!(forces_t, sys, neighbors, step_n, buffers, Val(false); n_threads=n_threads)
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
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    if length(sys.constraints) > 0
        err_str = "OverdampedLangevin is not currently compatible with constraints, " *
                  "constraints will be ignored"
        report_issue(err_str, strictness)
    end
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    init_step == 0 && !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t = zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    noise = zero(sys.velocities)
    noise_prefac = sqrt((2 / sim.friction) * sim.dt)

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
        forces!(forces_t, sys, neighbors, step_n, buffers, Val(false); n_threads=n_threads)
        accels_t .= calc_accels.(forces_t, masses(sys))

        random_velocities!(noise, sys, sim.temperature; rng=rng)
        sys.coords .+= (accels_t ./ sim.friction) .* sim.dt .+ noise_prefac .* noise
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)

        # Overdamped dynamics advance coordinates directly; removing velocity
        # drift here only affects the velocity state seen by loggers.
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                                   n_threads=n_threads)

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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
                           n_steps_or_time;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true,
                           shortcut=nothing,
                           init_step::Integer=0,
                           log_initial_state::Bool=true,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    if length(sys.constraints) > 0
        err_str = "NoseHoover is not currently compatible with constraints, " *
                  "constraints will be ignored"
        report_issue(err_str, strictness)
    end
    n_steps = calc_n_steps(n_steps_or_time, sim.dt)
    needs_vir, needs_vir_steps = needs_virial_schedule(sim.coupling)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    init_step == 0 && !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    forces_t, forces_t_dt = zero_forces(sys), zero_forces(sys)
    buffers = init_buffers!(sys, n_threads)
    forces!(forces_t, sys, neighbors, init_step, buffers, Val(true); n_threads=n_threads)
    accels_t = calc_accels.(forces_t, masses(sys))
    accels_t_dt = zero(accels_t)
    initial_loggers = initial_logger_mode(run_loggers, log_initial_state)
    apply_loggers!(sys, neighbors, init_step, buffers, initial_loggers; n_threads=n_threads,
                   current_forces=forces_t)
    v_half = zero(sys.velocities)
    zeta = zero(inv(sim.dt))
    dt_div2 = sim.dt / 2

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
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

        forces!(forces_t_dt, sys, neighbors, step_n, buffers, Val(needs_vir); n_threads=n_threads)
        accels_t_dt .= calc_accels.(forces_t_dt, masses(sys))

        sys.velocities .= (v_half .+ accels_t_dt .* dt_div2) ./
                          (1 + (zeta * dt_div2))

        # Remove drift after the Nose-Hoover velocity update and before
        # coupling/loggers observe the state.
        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end
        recompute_forces = apply_coupling!(sys, buffers, sim.coupling, sim, neighbors, step_n;
                                           n_threads=n_threads, rng=rng)

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n, recompute_forces;
                                    n_threads=n_threads)
        if recompute_forces
            forces!(forces_t_dt, sys, neighbors, step_n, buffers, Val(needs_vir);
                    n_threads=n_threads)
            forces_t .= forces_t_dt
            accels_t .= calc_accels.(forces_t, masses(sys))
        else
            forces_t .= forces_t_dt
            accels_t .= accels_t_dt
        end

        apply_loggers!(sys, neighbors, step_n, buffers, run_loggers; n_threads=n_threads,
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
                   n_steps_or_time;
                   assign_velocities::Bool=false,
                   n_threads::Integer=Threads.nthreads(),
                   run_loggers=true,
                   shortcut=nothing,
                   init_step::Integer=sys.current_step,
                   show_progress=default_show_progress(),
                   rng=Random.default_rng(),
                   strictness=default_strictness())
    check_strictness(strictness)
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

    return simulate_remd!(sys, sim, n_steps_or_time; n_threads=n_threads, run_loggers=run_loggers,
                          shortcut=shortcut, init_step=init_step, show_progress=show_progress,
                          rng=rng, strictness=strictness)
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
    simulate_remd!(sys::ReplicaSystem, remd_sim::ReplicaExchangeMD, n_steps::Integer;
                   <keyword arguments>)
    simulate_remd!(sys::ReplicaSystem, remd_sim::ReplicaExchangeMD, sim_time;
                   <keyword arguments>)

Run a Replica Exchange Molecular Dynamics (REMD) simulation on a multiple-replica system.

The simulation divides the total `n_steps` into cycles based on the time step and exchange time specified in the `ReplicaExchangeMD` simulator. Within each cycle, standard molecular dynamics propagation is independently executed for each replica. At the end of every cycle, replica exchange attempts are made between neighboring states. Any remaining steps that do not fit evenly into the exchange cycles are executed at the end of the run.

# Arguments
- `sys::ReplicaSystem`: the partitioned system containing the replicas and thermodynamic states.
- `remd_sim::ReplicaExchangeMD`: the simulator containing the specific time step and exchange time interval.
- `n_steps::Integer` or `sim_time`: the total number of steps or time to simulate for each replica.
- `n_threads::Integer=Threads.nthreads()`: the total number of threads to use, which are equally partitioned among the individual replicas.
- `run_loggers=true`: whether to run the loggers during the simulation, including the exchange logger.
- `init_step=sys.current_step`: absolute step before the first MD step. By default a repeated
    call resumes from the step stored in `sys`.
- `show_progress`: whether to show a progress bar for the simulation. `true` by default in
    the REPL/IJulia/Pluto, otherwise `false` by default. Can be set globally with the
    environmental variable `MOLLY_SHOW_PROGRESS`.
- `rng=Random.default_rng()`: the random number generator used for the exchange accept/reject criteria and any stochastic dynamics.
- `strictness=:warn`: determines behavior when encountering possible problems,
    options are `:warn` to emit warnings, `:nowarn` to suppress warnings or
    `:error` to error.
"""
function simulate_remd!(sys::ReplicaSystem,
                        remd_sim::ReplicaExchangeMD,
                        n_steps_or_time;
                        n_threads::Integer=Threads.nthreads(),
                        run_loggers=true,
                        shortcut=nothing, # Unused
                        init_step::Integer=sys.current_step,
                        show_progress=default_show_progress(),
                        rng=Random.default_rng(),
                        strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    sys.current_step = init_step
    n_steps = calc_n_steps(n_steps_or_time, remd_sim.dt)
    thread_div = equal_parts(n_threads, sys.n_replicas)

    n_cycles = convert(Int, (n_steps * remd_sim.dt) ÷ remd_sim.exchange_time)
    cycle_length = n_cycles > 0 ? n_steps ÷ n_cycles : 0
    remaining_steps = n_cycles > 0 ? n_steps % n_cycles : n_steps
    n_attempts = 0

    progress = setup_progress(n_steps, show_progress)
    for cycle in 1:n_cycles
        cycle_start_step = init_step + (cycle - 1) * cycle_length
        log_initial_state = sys.initial_log_pending
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
                                     n_threads=max(1, thread_div[i]), run_loggers=run_loggers,
                                     init_step=cycle_start_step,
                                     log_initial_state=log_initial_state,
                                     rng=rng, strictness=strictness)
        end
        sys.initial_log_pending = false

        cycle_parity = cycle % 2
        for n in (1 + cycle_parity):2:(sys.n_replicas - 1)
            n_attempts += 1
            m = n + 1
            Δ, exchanged = remd_exchange!(sys, remd_sim, n, m; rng=rng)
            
            if run_loggers != false && exchanged && !isnothing(sys.exchange_logger)
                log_property!(sys.exchange_logger, sys, nothing,
                              init_step + cycle * cycle_length, nothing;
                              indices=(n, m), delta=Δ, n_threads=n_threads)
            end
        end
        next_nograd!(progress)
    end

    if remaining_steps > 0
        remainder_start_step = init_step + n_cycles * cycle_length
        log_initial_state = sys.initial_log_pending
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
                                     n_threads=max(1, thread_div[i]), run_loggers=run_loggers,
                                     init_step=remainder_start_step,
                                     log_initial_state=log_initial_state,
                                     rng=rng, strictness=strictness)
        end
        sys.initial_log_pending = false
    end

    if run_loggers != false && !isnothing(sys.exchange_logger)
        if sys.exchange_logger isa ReplicaExchangeLogger
            finish_logs!(
                sys.exchange_logger;
                n_steps=n_steps,
                n_attempts=n_attempts,
                end_step=init_step + n_steps,
            )
        else
            finish_logs!(sys.exchange_logger; n_steps=n_steps, n_attempts=n_attempts)
        end
    end
    sys.current_step = init_step + n_steps

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
                           init_step::Integer=0,
                           show_progress=default_show_progress(),
                           rng=Random.default_rng(),
                           strictness=default_strictness())
    check_strictness(strictness)
    init_step = check_init_step(init_step)
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    place_virtual_sites!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder, nothing, init_step, true;
                               n_threads=n_threads)
    buffers = init_buffers!(sys, n_threads)
    E_old = potential_energy(sys, neighbors, init_step, buffers; n_threads=n_threads)
    coords_old = zero(sys.coords)

    progress = setup_progress(n_steps, show_progress)
    for step_n in (init_step + 1):(init_step + n_steps)
        coords_old .= sys.coords
        sim.trial_moves(sys; sim.trial_args...) # Changes the coordinates of the system
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
        place_virtual_sites!(sys)
        neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        E_new = potential_energy(sys, neighbors, step_n, buffers; n_threads=n_threads)

        ΔE = E_new - E_old
        δ = ΔE / (sys.k * sim.temperature)
        if δ < 0 || (rand(rng) < exp(-δ))
            apply_loggers!(sys, neighbors, step_n, nothing, run_loggers; n_threads=n_threads,
                           current_potential_energy=E_new, success=true,
                           energy_rate=(E_new / (sys.k * sim.temperature)))
            E_old = E_new
        else
            sys.coords .= coords_old
            apply_loggers!(sys, neighbors, step_n, nothing, run_loggers; n_threads=n_threads,
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
    rand_idx = pick_non_virtual_site(sys, rng)
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
    rand_idx = pick_non_virtual_site(sys, rng)
    direction = random_unit_vector(T, D, rng)
    magnitude = randn(rng, T) * shift_size
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
    return sys
end

function random_unit_vector(T, dims, rng=Random.default_rng())
    vec = randn(rng, T, dims)
    return vec / norm(vec)
end
