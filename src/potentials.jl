@inbounds function potential_energy(inter::LennardJones,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    T = eltype(s.atoms[i]) # this is not Unitful compatible
    i == j && return zero(T)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    r2 > sqdist_cutoff_nb && return zero(T)

    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
        return zero(T)
    end
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)

    invr2 = inv(r2)
    six_term = (σ ^ 2 * invr2) ^ 3

    return 4ϵ * (2 * six_term ^ 2 - six_term) - cutoff_energy(inter, σ, ϵ)
end

function cutoff_energy(inter, σ, ϵ)
    invr2 = inter.inv_sqdist_cutoff
    six_term = (σ ^ 2 * invr2) ^ 3
    
    4ϵ * (2 * six_term ^ 2 - six_term)
end