export MetropolisMonteCarlo,
    random_uniform_translation!,
    random_normal_translation!

"""
    MetropolisMonteCarlo(; <keyword arguments>)

A Monte Carlo simulator for [`System`](@ref) that uses the Metropolis algorithm to sample the configuration space.

# Arguments
- `temperature::T`: The temperature of the system.
- `log_interval::Int`: The interval at which to log the system state.
- `trial_moves::UP`: A function that performs the trial moves.
"""
struct MetropolisMonteCarlo{T, M}
    temperature::T
    trial_moves::M
    trial_args::Dict
end

function MetropolisMonteCarlo(;temperature::T,
                                trial_moves::M,
                                trial_args::Dict=Dict()) where {T, M}
    return MetropolisMonteCarlo{T, M}(temperature, trial_moves, trial_args)
end

function simulate!(sys::System{D, G, T},
                    sim::MetropolisMonteCarlo,
                    n_steps::Int;
                    n_threads::Int=Threads.nthreads()) where {D, G, T}
    if dimension(sys.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        k_b = sys.k * T(Unitful.Na) 
    else
        k_b = sys.k
    end

    for i in 1:n_steps
        E_old = potential_energy(sys)
        coords_old = copy(sys.coords)
        sim.trial_moves(sys; sim.trial_args...)  # changes the coordinates of the system
        E_new = potential_energy(sys)
        ΔE = E_new - E_old
        δ = ΔE / (k_b * sim.temperature)
        if δ > 0 && rand() > exp(-δ)
            sys.coords = coords_old
        end
        run_loggers!(sys, nothing, i; n_threads=n_threads)
    end
end

"""
    random_uniform_translation!(sys; shift_scaling=1.0, length_units=unit(sys.coords[1][1]))

Performs a random translation of the coordinates of a random atom in `sys`. The translation for each coordinate is independent and uniform in range [-0.5, 0.5] with units `length_units` and scaled by `shift_scaling`.
"""
function random_uniform_translation!(sys; shift_scaling=1.0, length_units=unit(sys.coords[1][1]))
    natoms = length(sys)
    rand_idx = rand(1:natoms)
    delta = shift_scaling * (rand(float_type(sys), 3) .- 0.5) * length_units
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ delta, sys.boundary)
end

"""
    random_normal_translation!(sys; shift_scaling=1.0, length_units=unit(sys.coords[1][1]))

Performs a random translation of the coordinates of a random atom in `sys`. The translation for each coordinatr independent and is generated from the standard normal distribution (i.e. mean 0 and standard deviation 1) with units `length_units` and scaled by `shift_scaling`.
"""
function random_normal_translation!(sys; shift_scaling=1.0, length_units=unit(sys.coords[1][1]))
    natoms = length(sys)
    rand_idx = rand(1:natoms)
    delta = shift_scaling * (randn(float_type(sys), 3) .- 0.5) * length_units
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ delta, sys.boundary)
end