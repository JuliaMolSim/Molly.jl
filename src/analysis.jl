# Analysis tools

export
    visualize,
    displacements,
    distances,
    rdf,
    energy

"""
    visualize(coord_logger, box_size, out_filepath; <keyword arguments>)

Visualize a simulation as an animation.
This function is only available when Makie is imported.
It can take a while to run, depending on the length and size of the simulation.

# Arguments
- `connections=Tuple{Int, Int}[]`: pairs of atoms indices to link with bonds.
- `connection_frames`: the frames in which bonds are shown. Is a list of the
    same length as the number of frames, where each item is a list of `Bool`s of
    the same length as `connections`. Defaults to always `true`.
- `trails::Integer=0`: the number of preceding frames to show as transparent
    trails.
- `framerate::Integer=30`: the frame rate of the animation.
- `color=:purple`: the color of the atoms. Can be a single color or a list of
    colors of the same length as the number of atoms.
- `connection_color=:orange`: the color of the bonds. Can be a single color or a
    list of colors of the same length as `connections`.
- `markersize=0.1`: the size of the atom markers.
- `linewidth=2.0`: the width of the bond lines.
- `transparency=true`: whether transparency is active on the plot.
- `kwargs...`: other keyword arguments are passed to the plotting function.
"""
function visualize end

"""
    displacements(coords, box_size)

Get the pairwise vector displacements of a set of coordinates, accounting for
the periodic boundary conditions.
"""
function displacements(coords, box_size)
    n_atoms = length(coords)
    coords_rep = repeat(reshape(coords, n_atoms, 1), 1, n_atoms)
    diffs = vector.(coords_rep, permutedims(coords_rep, (2, 1)), box_size)
    return diffs
end

"""
    distances(coords, box_size)

Get the pairwise distances of a set of coordinates, accounting for the periodic
boundary conditions.
"""
distances(coords, box_size) = norm.(displacements(coords, box_size))

"""
    rdf(coords, box_size; npoints=200)

Get the radial distribution function of a set of coordinates.
This describes how density varies as a function of distance from each atom.
Returns a list of distance bin centres and a list of the corresponding
densities.
"""
function rdf(coords, box_size::Real; npoints::Integer=200)
    n_atoms = length(coords)
    dists = distances(coords, box_size)
    dists_vec = [dists[i, j] for i in 1:n_atoms, j in 1:n_atoms if j > i]
    kd = kde(dists_vec, npoints=npoints)
    density_weighted = kd.density ./ (4π .* step(kd.x) .* kd.x .^ 2)
    return collect(kd.x), density_weighted
end

"""
    energy(s)

Compute the total energy of the system.
"""
energy(s) = kinetic_energy(s) + potential_energy(s)

function kinetic_energy(s::Simulation)
    ke = sum(i -> s.atoms[i].mass * dot(s.velocities[i], s.velocities[i]) / 2, axes(s.atoms, 1))
    # Convert energy to per mol if required
    if dimension(s.energy_units) == u"𝐋^2 * 𝐌 * 𝐍^-1 * 𝐓^-2"
        return uconvert(s.energy_units, ke * Unitful.Na)
    else
        return uconvert(s.energy_units, ke)
    end
end

function potential_energy(s::Simulation)
    n_atoms = length(s.coords)
    potential = zero(ustrip(s.timestep)) * s.energy_units

    for inter in values(s.general_inters)
        if inter.nl_only
            neighbors = s.neighbors
            @inbounds for ni in 1:length(neighbors)
                i, j = neighbors[ni]
                potential += potential_energy(inter, s, i, j)
            end
        else
            for i in 1:n_atoms
                for j in i+1:n_atoms
                    potential += potential_energy(inter, s, i, j)
                end
            end
        end
    end

    for inter_list in values(s.specific_inter_lists)
        for inter in inter_list
            potential += potential_energy(inter, s)
        end
    end

    return potential
end
