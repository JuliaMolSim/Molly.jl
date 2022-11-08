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

    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    E_old = potential_energy(sys, neighbors)
    for i in 1:n_steps
        coords_old = copy(sys.coords)
        sim.trial_moves(sys; sim.trial_args...)  # changes the coordinates of the system
        neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        E_new = potential_energy(sys, neighbors)

        ΔE = E_new - E_old
        δ = ΔE / (k_b * sim.temperature)
        if δ < 0 || rand() < exp(-δ)
            E_old = E_new
        else
            sys.coords = coords_old
        end
        run_loggers!(sys, neighbors, i; n_threads=n_threads)
    end
end

"""
	random_uniform_translation!(sys::System; scaling=1.0, length_units=unit(sys.coords[1][1]))

Performs a random translation of the coordinates of a randomly selected atom in `sys`. 
The translation is generated using a uniformly selected direction and uniformly selected length 
in range [-0.5, 0.5] scaled by `scaling` and with units `length_units`.
"""
function random_uniform_translation!(sys::System{D, G, T}; scaling::T=one(T),
                                    length_units=unit(sys.coords[1][1])) where {D, G, T}
	natoms = length(sys)
	rand_idx = rand(1:natoms)
    direction = random_unit_vector(T, D)
    magnitude = rand(T) * scaling * length_units
	sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
end

"""
	random_normal_translation!(sys::System; shift_scaling=1.0, length_units=unit(sys.coords[1][1]))

Performs a random translation of the coordinates of a randomly selected atom in `sys`. 
The translation is generated using a uniformly choosen direction and legth selected from 
the standard normal distribution i.e. ``\mathrm{Normal}(0, 1)``, scaled by `scaling` 
and with units `length_units`.
"""
function random_normal_translation!(sys::System{D, G, T}; scaling::T=one(T),
                                    length_units=unit(sys.coords[1][1])) where {D, G, T}
	natoms = length(sys)
	rand_idx = rand(1:natoms)
    direction = random_unit_vector(T, D)
    magnitude = randn(T) * scaling * length_units
	sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
end

"""
    random_unit_vector(float_type::Type, dims::Int)

Returns a random unit vector of in of length `dims` with elements of type `float_type`.
"""
function random_unit_vector(float_type, dims)
    vec = randn(float_type, dims)
    return vec / norm(vec)
end