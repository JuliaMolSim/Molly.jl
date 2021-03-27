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

The Finnis-Sinclair interaction. This interaction expects units to be of 
these https://lammps.sandia.gov/doc/units.html units (eV, Å, K, ps and so on).
"""
struct FinnisSinclair <: GlueInteraction
    nl_only::Bool
    element_pair_map::Dict
    params::DataFrame
    kb::Real
end


"""
    get_finnissinclair1984(nl_only)

Finnis and Sinclair 1984 parameterization: https://doi.org/10.1080/01418618408244210
"""
function get_finnissinclair1984(nl_only::Bool)
    
    elements = ["V", "Nb", "Ta", "Cr", "Mo", "W", "Fe"]
    element_pairings = [string(el, el) for el in elements]
    element_pair_map = Dict(pair => i for (i,pair) in enumerate(element_pairings))
    
    df = DataFrame(
        element_pair = element_pairings,
        d = [3.692767, 3.915354, 4.076980, 3.915720, 
            4.114825, 4.400224, 3.699579],
        # (Å)
        A = [2.010637, 3.013789, 2.591061, 1.453418, 
            1.887117, 1.896373, 1.889846],
        # (eV)
        β = [0, 0, 0, 1.8, 0, 0, 1.8],
        # (1)
        c = [3.8, 4.2, 4.2, 2.9, 3.25, 3.25, 3.4],
        # (Å)
        c₀ = [-0.8816318, -1.5640104, 1.2157373, 29.1429813, 
            43.4475218, 47.1346499, 1.2110601],
        # (1)
        c₁ = [1.4907756, 2.0055779, 0.0271471, -23.3975027, 
            -31.9332978, -33.7665655, -0.7510840],
        # (1)
        c₂ = [-0.3976370, -0.4663764, -0.1217350, 4.7578297, 
            6.0804249, 6.2541999, 0.1380773],
        # (1)
    )

    masses = Dict("V" => 50.9415, "Nb" => 92.9064, "Ta" => 180.9479,
                  "Cr" => 51.996, "Mo" => 95.94, "W" => 183.85,
                  "Fe" => 55.847) # g/mole

    # Å
    bcc_lattice_constants = Dict(
        "V" => 3.0399, "Nb" => 3.3008, "Ta" => 3.3058, 
        "Cr" => 2.8845, "Mo" => 3.1472, "W" => 3.1652, 
        "Fe" => 2.8665
    )

    kb = 8.617333262145e-5 # eV/K
    fs84 = FinnisSinclair(nl_only, element_pair_map, df, kb)
    
    reference_energies = DataFrame(
        element_pair = element_pairings,
        u = [5.31, 7.57, 8.1, 4.1, 6.82, 8.9, 4.28],
        u_vac = [1.92, 2.64, 3.13, 1.97, 2.58, 3.71, 1.77]
    )
    return fs84, elements, masses, bcc_lattice_constants, reference_energies 
end

"""
    glue(r,β,d; f=10.)

The core component of the Finnis-Sinclair type GlueInteraction. 
Used to calculate contribution to the glue value of an atom, based on 
its neighbouring atoms.
f is a fudge factor to help translate a Å model to a nm model.
"""
function glue(r, β, d)
    return r > d ? 0 : (r-d)^2 + β*(r-d)^3/d
end

"""
    ∂glue_∂r(r, β, d; f=10.)

Derivative of the glue density function.
"""
∂glue_∂r(r, β, d) = 2*(r-d) + 3*β*(r-d)^2/d

"""
    Uglue(ρ, A)

Energy based on the glue density .
"""
function Uglue(ρ, A)
   return -A * √ρ 
end


"""
    ∂Uglue_∂ρ(ρ, A)

Energy derivative given glue density.
"""
∂Uglue_∂ρ(ρ,A) = - A / (2 * √ρ)


"""
    Upair(r, c, c₀, c₁, c₂; f=10.)

Energy contribution directly from atom pair distances. f is a fudge factor to help translate a Å model to a nm model. 
"""
function Upair(r, c, c₀, c₁, c₂) 
    return (r > c) ? 0 : (r - c)^2 * (c₀ + c₁*r + c₂*r^2)
end

"""
    ∂Upair_∂r(r, c, c₀, c₁, c₂, f=10.)

Derivative of the pair potential.
f is a fudge factor to help translate a Å model to a nm model.
"""
∂Upair_∂r(r, c, c₀,c₁, c₂) = ForwardDiff.derivative(r -> Upair(r,c,c₀,c₁,c₂), r)

"""
    get_pair_params(element1, element2, inter)

Convenience function to generate element pair and return relevant model parameters. 
"""
function get_pair_params(element1::String, element2::String, inter::FinnisSinclair)
    pair = string(sort([element1, element2])...)
    return inter.params[inter.element_pair_map[pair],:]
end

"""
    update_glue_densities!(inter, coords, s, parallel)

Convenience function to update the densities before the forces are computed in serial/parallel.
"""
function update_glue_densities!(
        inter::FinnisSinclair, 
        coords, 
        s::Simulation;
        parallel::Bool=false
    )
    n_atoms = length(s.coords)
    
    # wiping old glue densities
    for i in 1:n_atoms
        s.glue_densities[i] = 0
    end
    
    # updating glue densities
    for (i,j) in s.neighbours

        # collecting parameters
        el_i = s.atoms[i].name
        el_j = s.atoms[j].name
        pi = get_pair_params(el_i,el_i,inter) 
        pj = get_pair_params(el_j,el_j,inter) 
        pij = get_pair_params(el_i,el_j,inter) 
        
        # computing distance
        dr = vector(s.coords[i], s.coords[j], s.box_size)
        r = norm(dr)
        
        # updating glue densities
        s.glue_densities[i] += glue(r, pj.β, pj.d)
        s.glue_densities[j] += glue(r, pi.β, pi.d)
    end
    return nothing
end

@inline @inbounds function force!(forces, inter::FinnisSinclair, s::Simulation, i::Integer, j::Integer)
    fdr = force(inter, s, i, j)
    forces[i] -= fdr
    forces[j] += fdr
    return nothing
end

@inline @inbounds function force(inter::FinnisSinclair, s::Simulation, i::Integer, j::Integer)
    
    # parameters
    element_i = s.atoms[i].name
    element_j = s.atoms[j].name
    element_pair = string(sort([element_i, element_j])...)
    pi = get_pair_params(element_i, element_i, inter) 
    pj = get_pair_params(element_j, element_j, inter)
    pij = get_pair_params(element_i, element_j, inter)

    # distance i -> j
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r = norm(dr)
    dr = normalize(dr)

    # pair contribution
    f_pair = - ∂Upair_∂r(r, pij.c, pij.c₀, pij.c₁, pij.c₂)

    # glue contribution
    dudρ_i = ∂Uglue_∂ρ(s.glue_densities[i], pi.A)
    dudρ_j = ∂Uglue_∂ρ(s.glue_densities[j], pj.A)
    dΦdr_i = ∂glue_∂r(r, pi.β, pi.d)
    dΦdr_j = ∂glue_∂r(r, pj.β, pj.d)
    f_glue = - (dudρ_j * dΦdr_i + dudρ_i * dΦdr_j)

    # total force contribution
    f = f_pair + f_glue
    return f * dr
end

@inline @inbounds function potential_energy(inter::FinnisSinclair, s::Simulation, i)
    # logger - general inters - computes the glue energy part only for a single atom
    # note: assumes that the glue densities are up to date, only calcutes them when inter.glue_densities are all 0    
    
    # check if densities are zero, if so calculate current, otherwise assume they are current
    no_glue = all(isapprox.(s.glue_densities, zeros(length(s.glue_densities)), atol=1e-4))
    if no_glue
        update_glue_densities!(inter, s.coords, s, parallel=false)
    end
        
    A = get_pair_params(s.atoms[i].name, s.atoms[i].name, inter).A
    return Uglue(s.glue_densities[i], A)
end

@inline @inbounds function potential_energy(inter::FinnisSinclair, s::Simulation, i, j)
    # logger - general inters - computes the pair energy part only for a single atom pair
    dr = vector(s.coords[i], s.coords[j], s.box_size)
    r = norm(dr)
    pij = get_pair_params(s.atoms[i].name, s.atoms[j].name, inter)
    return Upair(r, pij.c, pij.c₀, pij.c₁, pij.c₂)
end
