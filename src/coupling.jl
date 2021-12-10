# Temperature and pressure coupling

export
    NoCoupling,
    apply_coupling!,
    AndersenThermostat,
    maxwellboltzmann,
    temperature

"""
    NoCoupling()

Placeholder coupler that does nothing.
"""
struct NoCoupling <: AbstractCoupler end

"""
    apply_coupling!(system, simulator, coupling)

Apply a coupler to modify a simulation.
Custom couplers should implement this function.
"""
function apply_coupling!(sys::System, simulator, ::NoCoupling)
    return sys
end

"""
    AndersenThermostat(temperature, coupling_const)

Rescale random velocities according to the Andersen thermostat.
"""
struct AndersenThermostat{T, C} <: AbstractCoupler
    temperature::T
    coupling_const::C
end

function apply_coupling!(sys::System{D}, sim, thermostat::AndersenThermostat) where D
    for i in 1:length(sys)
        if rand() < (sim.dt / thermostat.coupling_const)
            mass = sys.atoms[i].mass
            sys.velocities[i] = velocity(mass, thermostat.temperature; dims=D)
        end
    end
    return sys
end

"""
    velocity(mass, temperature; dims=3)

Generate a random velocity from the Maxwell-Boltzmann distribution.
"""
function AtomsBase.velocity(mass, temp; dims::Integer=3)
    return SVector([maxwellboltzmann(mass, temp) for i in 1:dims]...)
end

"""
    maxwellboltzmann(mass, temperature)

Draw a speed along one dimension in accordance with the Maxwell-Boltzmann distribution.
"""
function maxwellboltzmann(mass, temp)
    T = typeof(convert(AbstractFloat, ustrip(temp)))
    k = unit(temp) == NoUnits ? one(T) : uconvert(u"u * nm^2 * ps^-2 * K^-1", T(Unitful.k))
    σ = sqrt(k * temp / mass)
    return rand(Normal(zero(T), T(ustrip(σ)))) * unit(σ)
end

"""
    temperature(system)

Calculate the temperature of a system from the kinetic energy of the atoms.
"""
function temperature(s::System{D, S, false}) where {D, S}
    ke = sum([a.mass * dot(s.velocities[i], s.velocities[i]) for (i, a) in enumerate(s.atoms)]) / 2
    df = 3 * length(s) - 3
    T = typeof(ustrip(ke))
    k = unit(ke) == NoUnits ? one(T) : uconvert(u"K^-1" * unit(ke), T(Unitful.k))
    return 2 * ke / (df * k)
end

function temperature(s::System{D, S, true}) where {D, S}
    ke = sum(mass.(s.atoms) .* sum.(abs2, s.velocities)) / 2
    df = 3 * length(s) - 3
    T = typeof(ustrip(ke))
    k = unit(ke) == NoUnits ? one(T) : uconvert(u"K^-1" * unit(ke), T(Unitful.k))
    return 2 * ke / (df * k)
end
