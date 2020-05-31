# Analysis tools

export
    displacements,
    distances,
    rdf

function displacements(coords)
    n_atoms = length(coords)
    coords_rep = repeat(reshape(coords, 1, n_atoms), n_atoms, 1)
    diffs = coords_rep .- permutedims(coords_rep, (2, 1))
    return diffs
end

distances(coords) = norm.(displacements(coords))

function rdf(coords; npoints::Integer=200)
    n_atoms = length(coords)
    dists = distances(coords)
    dists_vec = [dists[i, j] for i in 1:n_atoms, j in 1:n_atoms if j > i]
    kd = kde(dists_vec, npoints=npoints)
    density_weighted = kd.density ./ (4π .* step(kd.x) .* kd.x .^ 2)
    return collect(kd.x), density_weighted
end
