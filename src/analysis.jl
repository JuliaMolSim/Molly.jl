# Analysis tools

export
    visualize,
    displacements,
    distances,
    rdf,
    velocity_autocorr

"""
    visualize(coord_logger, box_size, out_filepath; <keyword arguments>)

Visualize a simulation as an animation.
This function is only available when GLMakie is imported.
GLMakie v0.5 or later should be used.
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
- `markersize=0.05`: the size of the atom markers, in the units of the data.
- `linewidth=2.0`: the width of the bond lines.
- `transparency=true`: whether transparency is active on the plot.
- `kwargs...`: other keyword arguments are passed to the point plotting
    function.
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
    diffs = vector.(coords_rep, permutedims(coords_rep, (2, 1)), (box_size,))
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
function rdf(coords, box_size; npoints::Integer=200)
    n_atoms = length(coords)
    dims = length(first(coords))
    dists = distances(coords, box_size)
    dists_vec = [dists[i, j] for i in 1:n_atoms, j in 1:n_atoms if j > i]
    dist_unit = unit(first(dists_vec))
    kd = kde(ustrip.(dists_vec), npoints=npoints)
    ρ = n_atoms / reduce(*, box_size)
    if dims == 3
        normalizing_factor = 4π .* ρ .* step(kd.x) .* kd.x .^ 2 .* dist_unit .^ 3
    elseif dims == 2
        normalizing_factor = 2π .* ρ .* step(kd.x) .* kd.x .* dist_unit .^ 2
    end
    bin_centres = collect(kd.x) .* dist_unit
    density_weighted = kd.density ./ normalizing_factor
    return bin_centres, density_weighted
end

"""
    velocity_autocorr(vl, first_ind, last_ind)

Calculates the autocorrelation function of velocity from the velocity logger. 
This helps characterize the similarity between velocities observed at different
time instances.
"""
function velocity_autocorr(vl::VelocityLogger, first_ind::Integer=1, last_ind::Integer=length(vl.velocities))
    n_atoms = length(first(vl.velocities))
    return dot(vl.velocities[first_ind], vl.velocities[last_ind]) / n_atoms
end
