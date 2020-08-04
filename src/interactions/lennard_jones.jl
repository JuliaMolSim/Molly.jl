"""
    LennardJones(nl_only)

The Lennard-Jones 6-12 interaction.
"""
struct LennardJones{T, C, S} <: GeneralInteraction
    cutoff::C
    nl_only::Bool
    sqdist_cutoff_coeff::T
    inv_sqdist_cutoff::T
end

LennardJones{S}(cutoff, nl_only, sqdist_cutoff_coeff=9.0) where S = 
    LennardJones{typeof(sqdist_cutoff_coeff), typeof(cutoff), S}(
        cutoff,
        nl_only,
        sqdist_cutoff_coeff,
        inv(sqdist_cutoff_coeff)
)

LennardJones(cutoff, nl_only, sqdist_cutoff_coeff=9.0) =
    LennardJones{typeof(sqdist_cutoff_coeff), typeof(cutoff), false}(
        cutoff,
        nl_only,
        sqdist_cutoff_coeff,
        inv(sqdist_cutoff_coeff)
)

LennardJones(nl_only=false) =
    LennardJones{Float64, ShiftedPotentialCutoff{Int}, false}(
        ShiftedPotentialCutoff(100),
                              nl_only,
                              9.0,
                              1/9.0
)

@inline @inbounds function force!(forces,
                                 inter::LennardJones{T, C, S},
                                 s::Simulation,
                                 i::Integer,
                                 j::Integer) where {T, C, S}

    i == j && return    # TODO: get rid of this check
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    if @generated
        if !S
            quote
                if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
                    return
                end
            end
        end
    else
        if !S && iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
            return
        end
    end

    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)

    σ2 = σ^2

    if @generated
        if cutoff_points(C) == 1
            quote
                sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
                r2 > sqdist_cutoff && return

                f = force_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
            end
        elseif cutoff_points(C) == 2
            quote
                sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
                activation_dist = inter.activation_dist_coeff * σ2

                r2 > sqdist_cutoff && return

                if r2 < activation_dist
                    f = force(inter, inv(r2), (σ2, ϵ))
                else
                    f = force_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
                end
            end
        end
    else
        if cutoff_points(C) == 1
            sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
            r2 > sqdist_cutoff && return

            f = force_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
        elseif cutoff_points(C) == 2
            sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
            activation_dist = inter.activation_dist_coeff * σ2

            r2 > sqdist_cutoff && return
            
            if r2 < activation_dist
                f = force(inter, inv(r2), (σ2, ϵ))
            else
                f = force_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
            end
        end
    end

    fdr = f * dr
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

@fastmath function force(::LennardJones, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    
    return (24ϵ * invr2) * (2 * six_term ^ 2 - six_term)
end

@inline @inbounds function potential_energy(inter::LennardJones{T, C, S},
                                            s::Simulation,
                                            i::Integer,
                                            j::Integer) where {T, C, S}
    U = eltype(s.coords[i]) # this is not Unitful compatible
    i == j && return zero(T)

    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r2 = sum(abs2, dr)

    if @generated
        if !S
            quote
                if iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
                    return zero(U)
                end
            end
        end
    else
        if !S && iszero(s.atoms[i].σ) || iszero(s.atoms[j].σ)
            return zero(U)
        end
    end

    σ = sqrt(s.atoms[i].σ * s.atoms[j].σ)
    ϵ = sqrt(s.atoms[i].ϵ * s.atoms[j].ϵ)
    
    σ2 = σ^2

    if @generated
        if cutoff_points(C) == 1
            quote
                sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
                r2 > sqdist_cutoff && return zero(U)

                potential_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
            end
        elseif cutoff_points(C) == 2
            quote
                r2 > sqdist_cutoff && return zero(U)
            
                if r2 < activation_dist
                    potential(inter, inv(r2), (σ2, ϵ))
                else
                    potential_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
                end
            end
        end
    else
        if cutoff_points(C) == 1
            sqdist_cutoff = inter.sqdist_cutoff_coeff * σ2
            r2 > sqdist_cutoff && return zero(U)

            potential_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
        elseif cutoff_points(C) == 2
            r2 > sqdist_cutoff && return zero(U)
            
            if r2 < activation_dist
                potential(inter, inv(r2), (σ2, ϵ))
            else
                potential_cutoff(inter.cutoff, r2, inter, (σ2, ϵ))
            end
        end
    end
end

@fastmath function potential(::LennardJones, invr2, (σ2, ϵ))
    six_term = (σ2 * invr2) ^ 3
    
    return 4ϵ * (six_term ^ 2 - six_term)
end

@fastmath function potential_cutoff(::ShiftedForceCutoff{false}, r2, inter::LennardJones, (σ2, ϵ))
    invr2 = inv(r2)
    r = √r2
    rc = √(inter.sqdist_cutoff_coeff * σ2)
    fc = force(inter, inter.inv_sqdist_cutoff, (σ2, ϵ))

    potential(inter, invr2, (σ2, ϵ)) - (r - rc) * fc -
        potential(inter, inter.inv_sqdist_cutoff, (σ2, ϵ))
end
