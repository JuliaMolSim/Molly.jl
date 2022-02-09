export ImplicitSolventOBC2

"""
    ImplicitSolventOBC2(atoms, atoms_data)

Onufriev-Bashford-Case GBSA model using the GBOBCII parameters.
"""
struct ImplicitSolventOBC2{R, T}
    offset_radii::Vector{R}
    scaled_offset_radii::Vector{R}
    solvent_dielectric::T
    solute_dielectric::T
    offset::R
    cutoff::R
    use_ACE::Bool
end

# Default solvent dielectric is 78.5 for consistency with AMBER
# Elsewhere it is 78.3
function ImplicitSolventOBC2(atoms,
                                atoms_data,
                                bonds;
                                solvent_dielectric=78.5,
                                solute_dielectric=1.0,
                                offset=0.009,
                                cutoff=0.0,
                                use_ACE=true)
    # See OpenMM source code
    default_radius = 0.15 # in nm
    element_to_radius = Dict(
        "N"  => 0.155,
        "O"  => 0.15 ,
        "F"  => 0.15 ,
        "Si" => 0.21 ,
        "P"  => 0.185,
        "S"  => 0.18 ,
        "Cl" => 0.17 ,
        "C"  => 0.17 ,
    )
    default_screen = 0.8
    element_to_screen = Dict(
        "H" => 0.85,
        "C" => 0.72,
        "N" => 0.79,
        "O" => 0.85,
        "F" => 0.88,
        "P" => 0.86,
        "S" => 0.96,
    )

    # Find atoms bonded to nitrogen
    atoms_bonded_to_N = falses(length(atoms))
    for (i, j) in zip(bonds.is, bonds.js)
        if atoms_data[i].element == "N"
            atoms_bonded_to_N[j] = true
        end
        if atoms_data[j].element == "N"
            atoms_bonded_to_N[i] = true
        end
    end

    T = typeof(first(atoms).charge)
    offset_radii = T[]
    scaled_offset_radii = T[]
    for (at, at_data, bonded_to_N) in zip(atoms, atoms_data, atoms_bonded_to_N)
        if at_data.element in ("H", "D")
            radius = bonded_to_N ? 0.13 : 0.12
        else
            radius = get(element_to_radius, at_data.element, default_radius)
        end
        offset_radius = radius - offset
        screen = get(element_to_screen, at_data.element, default_screen)
        push!(offset_radii, offset_radius)
        push!(scaled_offset_radii, screen * offset_radius)
    end
    return ImplicitSolventOBC2{T, T}(offset_radii, scaled_offset_radii, solvent_dielectric,
                                        solute_dielectric, offset, cutoff, use_ACE)
end

function forces(inter::ImplicitSolventOBC2, sys, neighbors)
    return ustrip_vec.(zero(sys.coords))
end

function potential_energy(inter::ImplicitSolventOBC2{R, T}, sys, neighbors) where {R, T}
    n_atoms = length(sys)
    coords, atoms, box_size = sys.coords, sys.atoms, sys.box_size

    Is = T[]
    for i in 1:n_atoms
        I = zero(T)
        for j in 1:n_atoms
            i == j && continue
            r = norm(vector(coords[i], coords[j], box_size))
            sr2 = inter.scaled_offset_radii[j]
            D = abs(r - sr2)
            or1 = inter.offset_radii[i]
            L = max(or1, D)
            U = r + sr2
            if U >= or1
                I += (1/L - 1/U + (r - (sr2^2)/r)*(1/(U^2) - 1/(L^2))/4 + log(L/U)/(2*r)) / 2
            end
        end
        push!(Is, I)
    end

    Bs = T[]
    for i in 1:n_atoms
        or1 = inter.offset_radii[i]
        radius = or1 + inter.offset
        psi = Is[i] * or1
        B = 1 / (1/or1 - tanh(psi - T(0.8)*(psi^2)+T(4.85)*(psi^3)) / radius)
        push!(Bs, B)
    end

    E = zero(T)
    factor = -138.935485 * (1/inter.solute_dielectric - 1/inter.solvent_dielectric)
    for i in 1:n_atoms
        charge_i = atoms[i].charge
        Bi = Bs[i]
        E += factor * (charge_i^2) / (2*Bi)
        if inter.use_ACE
            radius = inter.offset_radii[i] + inter.offset
            E += T(28.3919551) * (radius + T(0.14))^2 * (radius / Bi)^6
        end
        for j in (i + 1):n_atoms
            Bj = Bs[j]
            r2 = square_distance(i, j, coords, box_size)
            f = sqrt(r2 + Bi*Bj*exp(-r2/(4*Bi*Bj)))
            if iszero(inter.cutoff)
                f_cutoff = 1/f
            else
                f_cutoff = (1/f - 1/inter.cutoff)
            end
            E += factor * charge_i * atoms[j].charge * f_cutoff
        end
    end

    return E
end
