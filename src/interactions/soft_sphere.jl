"""
    SoftSphere(nl_only)

The soft-sphere potential.
"""
struct SoftSphere <: GeneralInteraction
    nl_only::Bool
end

@fastmath @inbounds function force!(forces,
                                    inter::SoftSphere,
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer)
    if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ) || i == j
        return
    end
    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    r2 > sqdist_cutoff_nb && return
    invr2 = inv(r2)
    six_term = (σ ^ 2 * invr2) ^ 3
    # Limit this to 100 as a fudge to stop it exploding
    f = min((24ϵ * invr2) * 2 * six_term ^ 2, 100)
    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end
