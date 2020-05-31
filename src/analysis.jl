# Analysis tools

export
    rdf

function rdf(coords; npoints::Integer=200)
    n_atoms = length(coords)
    coords_rep = repeat(reshape(coords, n_atoms, 1), 1, n_atoms)
    diffs = coords_rep .- permutedims(coords_rep, (2, 1))
    dists = norm.(diffs)
    dists_vec = [dists[i, j] for i in 1:n_atoms, j in 1:n_atoms if j > i]
    kd = kde(dists_vec, npoints=npoints)
    density_weighted = kd.density ./ (4π .* step(kd.x) .* kd.x .^ 2)
    return collect(kd.x), density_weighted
end
