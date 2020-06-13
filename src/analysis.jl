# Analysis tools

export
    displacements,
    distances,
    rdf,
    visualize

"""
    displacements(coords, box_size)

Get the pairwise vector displacements of a set of coordinates, accounting for
the periodic boundary conditions.
"""
function displacements(coords, box_size::Real)
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
distances(coords, box_size::Real) = norm.(displacements(coords, box_size))

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

"Visualize a simulation."
function visualize end
