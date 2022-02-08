export ImplicitSolventOBC2

"""
    ImplicitSolventOBC2(atoms, atoms_data)

Onufriev-Bashford-Case GBSA model using the GBOBCII parameters.
"""
struct ImplicitSolventOBC2{T}
    offset_radii::Vector{T}
    scaled_offset_radii::Vector{T}
end

function ImplicitSolventOBC2(atoms, atoms_data)
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

    T = typeof(first(atoms).charge)
    offset_radii = T[]
    scaled_offset_radii = T[]
    for (at, at_data) in zip(atoms, atoms_data)
        if at_data.element in ("H", "D")
            offset_radius = 0.12 # TODO check N
        else
            offset_radius = get(element_to_radius, at_data.element, default_radius)
        end
        push!(offset_radii, offset_radius)
        push!(scaled_offset_radii, get(element_to_screen, at_data.element, default_screen))
    end
    return ImplicitSolventOBC2{T}(offset_radii, scaled_offset_radii)
end

function forces(inter::ImplicitSolventOBC2, sys, neighbors)
    return ustrip_vec.(zero(sys.coords))
end

function potential_energy(inter::ImplicitSolventOBC2, sys, neighbors)
    return 0.0
end
