export MetropolisMonteCarlo,
    random_uniform_translation!,
    random_normal_translation!

"""
    MetropolisMonteCarlo(; <keyword arguments>)

A Monte Carlo simulator for [`System`](@ref) that uses the Metropolis algorithm to sample the configuration space.
The method of `simulate!` for this simulator accepts an optional keyword argument `log_states::Bool=true` which 
tells whether to run the loggers or not (for example, during equilibration).

# Arguments
- `temperature::T`: The temperature of the system.
- `trial_moves::UP`: A function that performs the trial moves.
- `trial_args::Dict`: A dictionary of arguments to be passed to the trial moves.
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
                    n_threads::Int=Threads.nthreads(),
                    log_states::Bool=true) where {D, G, T}
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
            log_states && run_loggers!(sys, neighbors, i; n_threads=n_threads, success=true,
                        energy_rate=E_new / (k_b * sim.temperature))
            E_old = E_new
        else
            sys.coords = coords_old
            log_states && run_loggers!(sys, neighbors, i; n_threads=n_threads, success=false,
                        energy_rate=E_old / (k_b * sim.temperature))
        end
    end
end

"""
	random_uniform_translation!(sys::System; shift_size=1.0*unit(sys.coords[1][1]))

Performs a random translation of the coordinates of a randomly selected atom in `sys`. 
The translation is generated using a uniformly selected direction and uniformly selected length 
in range [0, 1) scaled by `shift_size` which has appropriate legth units.
"""
function random_uniform_translation!(sys::System{D, G, T};
                                        shift_size=one(T)*unit(sys.coords[1][1])) where {D, G, T}
	natoms = length(sys)
	rand_idx = rand(1:natoms)
    direction = random_unit_vector(T, D)
    magnitude = rand(T) * shift_size
	sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
end

"""
	random_normal_translation!(sys::System; shift_size=1.0*unit(sys.coords[1][1]))

Performs a random translation of the coordinates of a randomly selected atom in `sys`. 
The translation is generated using a uniformly choosen direction and legth selected from 
the standard normal distribution i.e. with mean 0 and standard deviation 1, scaled by `shift_size` 
which has appropriate length units.
"""
function random_normal_translation!(sys::System{D, G, T};
                                        shift_size=one(T)*unit(sys.coords[1][1])) where {D, G, T}
	natoms = length(sys)
	rand_idx = rand(1:natoms)
    direction = random_unit_vector(T, D)
    magnitude = randn(T) * shift_size
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