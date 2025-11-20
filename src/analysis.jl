# Analysis tools

export
    displacements,
    distances,
    kabsch,
    rmsd,
    radius_gyration,
    hydrodynamic_radius,
    visualize,
    rdf

"""
    displacements(coords, boundary)

Calculate the pairwise vector displacements of a set of coordinates, accounting
for the periodic boundary conditions.
"""
function displacements(coords, boundary)
    n_atoms = length(coords)
    coords_rep = repeat(reshape(coords, n_atoms, 1), 1, n_atoms)
    vec_2_arg(c1, c2) = vector(c1, c2, boundary)
    diffs = vec_2_arg.(coords_rep, permutedims(coords_rep, (2, 1)))
    return diffs
end

"""
    distances(coords, boundary)

Calculate the pairwise distances of a set of coordinates, accounting for the
periodic boundary conditions.
"""
distances(coords, boundary) = norm.(displacements(coords, boundary))

"""
    kabsch(coords_1, coords_2)

Superimpose two sets of 3D coordinates using the Kabsch algorithm. 

Coodinates are superimposed by rotating and translating the first set
of input coordinates and translating the second set of input coordinates. 

Assumes the coordinates do not cross the bounding box, i.e. all
coordinates in each set correspond to the same periodic image.
"""
function kabsch(coords_1::AbstractArray{SVector{D, T}},
                coords_2::AbstractArray{SVector{D, T}}) where {D, T}
    
    n_atoms = length(coords_1)
    trans_1 = mean(coords_1)
    trans_2 = mean(coords_2)

    p = Molly.from_device(reshape(reinterpret(T, coords_1), D, n_atoms)) .- repeat(reinterpret(T, trans_1), 1, n_atoms)
    q = Molly.from_device(reshape(reinterpret(T, coords_2), D, n_atoms)) .- repeat(reinterpret(T, trans_2), 1, n_atoms)
    
    cov = p * transpose(q)
    svd_res = svd(ustrip.(cov))
    Ut = transpose(svd_res.U)
    d = sign(det(svd_res.V * Ut))
    dmat = [1 0 0; 0 1 0; 0 0 d]
    rot = svd_res.V * dmat * Ut
    
    p_rot = rot * p
    
    p_rot_reshaped = SArray[SVector{D,T}(p_rot[i:i+2]) for i=1:3:length(p_rot)-2]
    q_reshaped = SArray[SVector{D,T}(q[i:i+2]) for i=1:3:length(q)-2]

    return p_rot_reshaped, q_reshaped # return p centered and rotated, q centered
end

"""
    ref_kabsch(coords_1, coords_2)

Wrapper function to return only the translated and rotated coordinates
of coords_1 after superimposition of coords_1 and coords_2 by the Kabsch algorithm.

Assumes the coordinates do not cross the bounding box, i.e. all
coordinates in each set correspond to the same periodic image.
"""
function ref_kabsch(coords_1::AbstractArray{SVector{D, T}},
                    coords_2::AbstractArray{SVector{D, T}}) where {D, T}
    p_rot, _ = kabsch(coords_1, coords_2)
    return p_rot
end 

"""
    rmsd(coords_1, coords_2)

Calculate the root-mean-square deviation (RMSD) of two sets of
3D coordinates after superimposition by the Kabsch algorithm.

Assumes the coordinates do not cross the bounding box, i.e. all
coordinates in each set correspond to the same periodic image.
"""
function rmsd(coords_1::AbstractArray{SVector{D, T}},
              coords_2::AbstractArray{SVector{D, T}}) where {D, T}
    p_rot, q = kabsch(coords_1, coords_2) 
    diffs = p_rot - q
    msd = mean(norm.(diffs).^2)
    return sqrt(msd)
end

sum_abs2(x) = sum(abs2, x)

"""
    radius_gyration(coords, atoms)

Calculate the radius of gyration of a set of coordinates.

Assumes the coordinates do not cross the bounding box, i.e. all
coordinates correspond to the same periodic image.
"""
function radius_gyration(coords, atoms)
    center = mean(coords)
    vecs_to_center = coords .- (center,)
    atom_masses = mass.(atoms)
    I = sum(sum_abs2.(vecs_to_center) .* atom_masses)
    return sqrt(I / sum(atom_masses))
end

@doc raw"""
    hydrodynamic_radius(coords, boundary)

Calculate the hydrodynamic radius of a set of coordinates.

``R_{hyd}`` is defined by
```math
\frac{1}{R_{hyd}} = \frac{1}{2N^2}\sum_{i \neq j} \frac{1}{r_{ij}}
```
"""
function hydrodynamic_radius(coords::AbstractArray{SVector{D, T}}, boundary) where {D, T}
    n_atoms = length(coords)
    diag  = array_type(coords)(Diagonal(ones(T, n_atoms)))
    dists = distances(coords, boundary) .+ diag
    sum_inv_dists = sum(inv.(dists)) - sum(inv(diag))
    inv_R_hyd = sum_inv_dists / (2 * n_atoms^2)
    return inv(inv_R_hyd)
end

function axis_limits(boundary_conv, coord_logger, dim)
    lim = boundary_conv[dim]
    if isinf(lim)
        # Find coordinate limits in given dimension
        low  = ustrip(minimum(cs -> minimum(c -> c[dim], cs), values(coord_logger)))
        high = ustrip(maximum(cs -> maximum(c -> c[dim], cs), values(coord_logger)))
        return low, high
    else
        return 0.0, lim
    end
end

"""
    visualize(coord_logger, boundary, out_filepath; <keyword arguments>)

Visualize a simulation as an animation.

This function is only available when GLMakie is imported.
It can take a while to run, depending on the length of the simulation and the
number of atoms.

# Arguments
- `connections=Tuple{Int, Int}[]`: pairs of atoms indices to link with bonds.
- `connection_frames`: the frames in which bonds are shown. Should be a list of
    the same length as the number of frames, where each item is a list of
    `Bool`s of the same length as `connections`. Defaults to always `true`.
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
- `show_boundary::Bool=true`: whether to show the bounding box as lines.
- `boundary_linewidth=2.0`: the width of the boundary lines.
- `boundary_color=:black`: the color of the boundary lines.
- `kwargs...`: other keyword arguments are passed to the point plotting
    function.
"""
function visualize end

"""
    rdf(coords, boundary; npoints=200)

Calculate the radial distribution function of a set of coordinates.

This function is only available when KernelDensity is imported.
This describes how density varies as a function of distance from each atom.
Returns a list of distance bin centers and a list of the corresponding
densities.
"""
function rdf end
