"""
    Coulomb(nl_only)

The Coulomb electrostatic interaction.
"""
struct Coulomb{C, T} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
    coulomb_const::T
end

Coulomb(cutoff, nl_only) = Coulomb(
    cutoff,
    nl_only,
    138.935458 / 70.0 # Treat ϵr as 70 for now
)

Coulomb(nl_only=false) = Coulomb(
    NoCutoff(),
    nl_only,
    138.935458 / 70.0 # Treat ϵr as 70 for now
)

@inline @inbounds function force!(forces,
                                    inter::Coulomb{C},
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer) where C
    i == j && return
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    cutoff = inter.cutoff
    coulomb_const = inter.coulomb_const
    qi, qj = s.atoms[i].charge, s.atoms[j].charge

    params = (coulomb_const, qi, qj)
    
    if @generated
        if cutoff_points(C) == 0
            quote
                f = force(inter, r2, inv(r2), params)
            end
        elseif cutoff_points(C) == 1
            quote
                sqdist_cutoff = cutoff.sqdist_cutoff
                r2 > sqdist_cutoff && return

                f = force_cutoff(cutoff, r2, inter, params)
            end
        elseif cutoff_points(C) == 2
            quote
                sqdist_cutoff = cutoff.sqdist_cutoff
                activation_dist = cutoff.activation_dist

                r2 > sqdist_cutoff && return

                if r2 < activation_dist
                    f = force(inter, r2, inv(r2), params)
                else
                    f = force_cutoff(cutoff, r2, inter, params)
                end
            end
        end
    else
        if cutoff_points(C) == 0
            f = force(inter, r2, inv(r2), params)
        elseif cutoff_points(C) == 1
            sqdist_cutoff = cutoff.sqdist_cutoff
            r2 > sqdist_cutoff && return

            f = force_cutoff(cutoff, r2, inter, params)
        elseif cutoff_points(C) == 2
            sqdist_cutoff = cutoff.sqdist_cutoff
            activation_dist = cutoff.activation_dist

            r2 > sqdist_cutoff && return
            
            if r2 < activation_dist
                f = force(inter, r2, inv(r2), params)
            else
                f = force_cutoff(cutoff, r2, inter, params)
            end
        end
    end

    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

@fastmath function force(::Coulomb, r2, invr2, (coulomb_const, qi, qj))
    (coulomb_const * qi * qj) / √(r2 ^ 3)
end

@inline @inbounds function potential_energy(inter::Coulomb{C},
                                    s::Simulation,
                                    i::Integer,
                                    j::Integer) where C
    U = eltype(s.coords[i]) # this is not Unitful compatible
    i == j && return zero(U)

    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)
    
    cutoff = inter.cutoff
    sqdist_cutoff = cutoff.sqdist_cutoff
    coulomb_const = inter.coulomb_const
    qi = s.atoms[i].charge
    qj = s.atoms[j].charge
    params = (coulomb_const, qi, qj)

    if @generated
        if cutoff_points(C) == 0
            quote
                potential(inter, r2, inv(r2), params)
            end
        elseif cutoff_points(C) == 1
            quote
                sqdist_cutoff = cutoff.sqdist_cutoff * σ2
                r2 > sqdist_cutoff && return zero(U)

                potential_cutoff(cutoff, r2, inter, params)
            end
        elseif cutoff_points(C) == 2
            quote
                r2 > sqdist_cutoff && return zero(U)
            
                if r2 < activation_dist
                    potential(inter, r2, inv(r2), params)
                else
                    potential_cutoff(cutoff, r2, inter, params)
                end
            end
        end
    else
        if cutoff_points(C) == 0
            potential(inter, r2, inv(r2), params)
        elseif cutoff_points(C) == 1
            sqdist_cutoff = cutoff.sqdist_cutoff * σ2
            r2 > sqdist_cutoff && return zero(U)

            potential_cutoff(cutoff, r2, inter, params)
        elseif cutoff_points(C) == 2
            r2 > sqdist_cutoff && return zero(U)
            
            if r2 < activation_dist
                potential(inter, r2, inv(r2), params)
            else
                potential_cutoff(cutoff, r2, inter, params)
            end
        end
    end
end

@fastmath function potential(::Coulomb, r2, invr2, (oulomb_const, qi, qj))
    (coulomb_const * qi * qj) * √invr2
end
