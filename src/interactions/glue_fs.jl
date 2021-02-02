"""
A glue interaction that will apply to all atom pairs.
Finnis-Sinclair and similar interactions should sub-type this type.
This type should be a GeneralInteraction type. But due to the special
nature of glue interactions and the restriction to pair interactions
of the GeneralInteraction type, glue interactions are for now a sub-type of 
SpecificInteraction.
"""
abstract type GlueInteraction <: SpecificInteraction end

"""
    FinnisSinclairInteraction(nl_only,element_pair_map,params)

The Finnis-Sinclair interaction.
"""
struct FinnisSinclair <: GlueInteraction
    nl_only::Bool
    element_pair_map::Dict
    params::DataFrame
end

"""
    glue_potential(r,β,d)

The core component of the Finnis-Sinclair type GlueInteraction. Used to calculate contribution to the glue value of an atom, based on its neighbouring atoms.
"""
function glue_potential(r::T, β::T, d::T)::T where T<:Real
    return r > d ? 0 : (r-d)^2 + β*(r-d)^3/d
end

"""
"""
function pair_potential_derivative(r::T, c::T, c₀::T, c₁::T, c₂::T)::T where T<:Real
    return (r > c) ? 0 : 2 * (r - c) * (c₀ + c₁*r + c₂*r^2) + (r - c)^2 * (c₁ + 2*c₂*r)
end

"""
"""
function glue_energy_derivative(ρ::Float64, A::Float64)::Float64
   return A/2 * ρ^-1.5 
end

"""
"""
function glue_potential_derivative(r::T, β::T, d::T)::T where T<:Real
    return r > d ? 0 : 2*(r-d) + 3*β*(r-d)^2/d
end

function get_pair_params(element1::String, element2::String, inter::FinnisSinclair)
    pair = string(sort([element1, element2])...)
    return inter.params[inter.element_pair_map[pair],:]
end

@inline @inbounds function force(
        inter::FinnisSinclair, 
        coords, 
        s::Simulation
    )
    # computing the embedding densities
    n_atoms = length(s.coords)
    ρs = zeros(n_atoms)
    rs = zeros(length(s.neighbours))
    r_vec_norms = zeros(length(s.neighbours),3)
    
    for (n,(i,j)) in enumerate(s.neighbours)
        element_i = s.atoms[i].name
        element_j = s.atoms[j].name
        element_pair = string(sort([element_i, element_j])...)
        pi = get_pair_params(element_i,element_i,inter) # inter.params[inter.element_map[element_i],:]
        pj = get_pair_params(element_j,element_j,inter) # inter.params[inter.element_map[element_j],:]
        pij = get_pair_params(element_i,element_j,inter) # inter.params[inter.element_map[element_pair],:]
        
        r_vec = vector(s.coords[i], s.coords[j], s.box_size)
        r2 = sum(abs2, r_vec)
        r = sqrt(r2)
        # storing distance (vectors) so we don't need to recompute
        rs[n] = r
        r_vec_norms[[n],:] = r_vec / r
        # storing glue densities
        ρs[i] += glue_potential(r, pj.β, pj.d)
        ρs[j] += glue_potential(r, pi.β, pi.d)
    end
    
    fs = [zeros(1,3) for _ in 1:n_atoms]
    for (n,(i,j)) in enumerate(s.neighbours)
        element_i = s.atoms[i].name
        element_j = s.atoms[j].name
        element_pair = string(sort([element_i, element_j])...)
        pi = get_pair_params(element_i,element_i,inter) # inter.params[inter.element_map[element_i],:]
        pj = get_pair_params(element_j,element_j,inter) # inter.params[inter.element_map[element_j],:]
        pij = get_pair_params(element_i,element_j,inter) # inter.params[inter.element_map[element_pair],:]
        
        r = rs[n]
        r2 = r^2
        r_vec_norm = r_vec_norms[[n],:]
        
        # pair contribution
        dpairdR_i = r_vec_norm * pair_potential_derivative(r, pij.c, pij.c₀, pij.c₁, pij.c₂)
        dpairdR_j = - dpairdR_i
        
        # glue contribution
        dudρ_i = glue_energy_derivative(ρs[i], pi.A)
        dudρ_j = glue_energy_derivative(ρs[j], pj.A)
        dΦdr_i = glue_potential_derivative(r, pi.β, pi.d)
        dΦdr_j = glue_potential_derivative(r, pj.β, pj.d)
        
        ## density change by moving the current atom
        dgluedR_i_curr = r_vec_norm * dudρ_i * dΦdr_j
        dgluedR_j_curr = r_vec_norm * dudρ_j * dΦdr_i
        ## density change by moving a neighbouring atom
        dgluedR_i_neigh = - r_vec_norm * dudρ_j * dΦdr_i
        dgluedR_j_neigh = - r_vec_norm * dudρ_i * dΦdr_j
        
        # updating the forces
        f_i = (dpairdR_i + dgluedR_i_curr + dgluedR_i_neigh)
        f_j = (dpairdR_j + dgluedR_j_curr + dgluedR_j_neigh)
        fs[i] += f_i
        fs[j] += f_j
    end
    
    return collect(1:n_atoms), fs
end

"""
"""
function pair_potential(r::T, c::T, c₀::T, c₁::T, c₂::T)::T where T<:Real
    return (r > c) ? 0 : (r - c)^2 * (c₀ + c₁*r + c₂*r^2)
end

"""
"""
function glue_energy(ρ::Float64, A::Float64)::Float64
   return -A * √ρ 
end

@inline @inbounds function potential_energy(inter::FinnisSinclair, s::Simulation)
    #computing the potential energy combining glue and pair components.
    
    e_pair = 0.
    e_glue = 0.
    n_atoms = length(s.coords)
    ρs = zeros(n_atoms)
    for (i,j) in s.neighbours
        element_i = s.atoms[i].name
        element_j = s.atoms[j].name
        element_pair = string(sort([element_i, element_j])...)
        pi = get_pair_params(element_i,element_i,inter) # inter.params[inter.element_map[element_i],:]
        pj = get_pair_params(element_j,element_j,inter) # inter.params[inter.element_map[element_j],:]
        pij = get_pair_params(element_i,element_j,inter) # inter.params[inter.element_map[element_pair],:]
        
        r_vec = vector(s.coords[i], s.coords[j], s.box_size)
        r2 = sum(abs2, r_vec)
        r = sqrt(r2)
        
        e_pair += pair_potential(r, pij.c, pij.c₀, pij.c₁, pij.c₂)
        
        ρs[i] += glue_potential(r, pj.β, pj.d)
        ρs[j] += glue_potential(r, pi.β, pi.d)
    end
    
    es_glue = zeros(n_atoms)
    for (i, atom) in enumerate(s.atoms)
        A = get_pair_params(atom.name, atom.name, inter).A
        es_glue[i] = glue_energy(ρs[i], A)
    end
    e_glue = sum(es_glue)
    return e_pair + e_glue 
end