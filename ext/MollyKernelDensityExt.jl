# Radial distribution function
# This file is only loaded when KernelDensity is imported

module MollyKernelDensityExt

using Molly
using KernelDensity
using LinearAlgebra

function Molly.rdf(coords, boundary::Molly.AbstractBoundary{D, T};
                   npoints::Integer=200) where {D, T}
    n_atoms = length(coords)
    coords_cpu = Molly.from_device(coords)
    # The distances of the pairs i < j
    dists_vec = [norm(Molly.vector(coords_cpu[i], coords_cpu[j], boundary))
                 for j in 1:n_atoms for i in 1:(j - 1)]
    dist_unit = unit(first(dists_vec))
    kd = kde(ustrip.(dists_vec); npoints=npoints)
    ρ = n_atoms / volume(boundary)
    # kd.density is the probability density of the pair distances, so the number
    #   density of neighbors at distance r around a given atom is
    #   (n_atoms - 1) * p(r) / shell_area(r), and g(r) is that divided by ρ
    if D == 3
        normalizing_factor = 4 .* T(π) .* ρ .* kd.x .^ 2 .* dist_unit .^ 3 ./ (n_atoms - 1)
    elseif D == 2
        normalizing_factor = 2 .* T(π) .* ρ .* kd.x .* dist_unit .^ 2 ./ (n_atoms - 1)
    end
    bin_centers = collect(kd.x) .* dist_unit
    density_weighted = kd.density ./ normalizing_factor
    return bin_centers, density_weighted
end

end
