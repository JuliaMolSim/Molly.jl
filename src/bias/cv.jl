# Calculate collective variables

export
    CalcMinDist,
    CalcMaxDist,
    CalcCMDist,
    CalcSingleDist,
    CalcDist,
    calculate_cv,
    CalcRg,
    CalcRMSD

# Does not account for periodic boundary conditions
function centre_of_mass(coords, atoms)
    masses = mass.(atoms)
    com = sum(masses .* coords; dims=1) ./ sum(masses)
    return only(com)
end

#=  pairwise_distance_matrix(coords_1, coords_2, calc_type, boundary)

Calculate all pairwise distances between two sets of coordinates.

When `calc_type` is `:closest`, periodic boundary conditions are accounted for in the
distance calculations. To calculate raw distances instead, `calc_type` should be `:raw`.=#
function pairwise_distance_matrix(coords_1::AbstractArray{SVector{D, T}},
                                  coords_2::AbstractArray{SVector{D, T}},
                                  calc_type,
                                  boundary) where {D, T}
    dist_matrix = zeros(T, length(coords_1), length(coords_2))
    if calc_type == :closest
        for i in eachindex(coords_1), j in eachindex(coords_2)
            dist_matrix[i, j] = norm(vector(coords_1[i], coords_2[j], boundary))
        end
    else
        for i in eachindex(coords_1), j in eachindex(coords_2)
            dist_matrix[i, j] = norm(coords_1[i] - coords_2[j])
        end
    end
    return dist_matrix
end

"""
    CalcMinDist

Bias the minimum distance between two groups of atoms.

(If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box sidelength of eachother to ensure correct results.)

# Arguments
- `calc_type::Symbol=:closest`: distances are calculated between closest periodic images. Should be set to `:raw` to calculate raw distances ignoring PBCs.
"""
@kwdef struct CalcMinDist
    calc_type::Symbol=:closest
end

function dist_between_groups(md::CalcMinDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_matrix = pairwise_distance_matrix(coords_1, coords_2, md.calc_type, boundary)
    min_dist, min_dist_pair = findmin(dist_matrix)
    return min_dist
end

"""
    CalcMaxDist

Bias the maximum distance between two groups of atoms.

(If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box sidelength of eachother to ensure correct results.)

# Arguments
- `calc_type::Symbol=:closest`: distances are calculated between closest periodic images. Should be set to `:raw` to calculate raw distances ignoring PBCs.
"""
@kwdef struct CalcMaxDist
    calc_type::Symbol=:closest
end

function dist_between_groups(md::CalcMaxDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_matrix = pairwise_distance_matrix(coords_1, coords_2, md.calc_type, boundary)
    max_dist, max_dist_pair = findmax(dist_matrix)
    return max_dist
end

"""
    CalcCMDist

Bias the distance between the centres of mass of two groups of atoms.

(If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box sidelength of eachother to ensure correct results.)

# Arguments
- `calc_type::Symbol=:closest`: distances are calculated between closest periodic images. Should be set to `:raw` to calculate raw distances ignoring PBCs.
"""
@kwdef struct CalcCMDist
    calc_type::Symbol=:closest
end

function dist_between_groups(cd::CalcCMDist, coords_1, coords_2, boundary, atoms_1, atoms_2, args...; kwargs...)
    com_1 = centre_of_mass(coords_1,atoms_1)
    com_2 = centre_of_mass(coords_2,atoms_2)
    if cd.calc_type == :closest
        com_dist_val = norm(vector(com_1, com_2, boundary))
    else
        com_dist_val = norm(com_1 - com_2)
    end
    return com_dist_val
end

"""
    CalcSingleDist

Bias the distance between two atoms.

(If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box sidelength of eachother to ensure correct results.)

# Arguments
- `calc_type::Symbol=:closest`: distances are calculated between closest periodic images. Should be set to `:raw` to calculate raw distances ignoring PBCs.
"""
@kwdef struct CalcSingleDist
    calc_type::Symbol=:closest
end

function dist_between_groups(sd::CalcSingleDist, coords_1::AbstractArray{SVector{D, T}}, coords_2::AbstractArray{SVector{D, T}}, boundary, args...; kwargs...) where {D, T}
    coords_1 = coords_1[1] # get single set of coords from vector
    coords_2 = coords_2[1] # get single set of coords from vector
    if sd.calc_type == :closest
        dist_val = norm(vector(coords_1, coords_2, boundary))
    else
        dist_val = norm(coords_1 - coords_2)
    end
    return dist_val
end

"""
    CalcDist(atom_idx_1, atom_idx_2, correction, dist_type)

Bias the distance between two atoms or two groups of atoms in a system.
A bias is applied to the type of distance specified by `dist_type`.

# Arguments
- `atom_inds_1::Vector{Int}`: indices of atom(s) in first group.
- `atom_inds_2::Vector{Int}`: indices of atom(s) in second group.
- `correction::Symbol=:wrap`: the system is wrapped in the unit cell. Should be `:pbc` to unwrap molecules prior to the distance calculation.
- `dist_type::DT`: type of distance to calculate, e.g. MinDist() or ComDist().
"""
@kwdef struct CalcDist{DT}
    atom_inds_1::Vector{Int}
    atom_inds_2::Vector{Int}
    correction::Symbol=:wrap
    dist_type::DT
end

"""
...
"""
function calculate_cv(cv::CalcDist, coords, atoms, boundary, args...; kwargs...)

    coords_1 = @view coords[cv.atom_inds_1]
    coords_2 = @view coords[cv.atom_inds_2]

    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]

    dist_val = dist_between_groups(cv.dist_type, coords_1, coords_2, boundary, atoms_1, atoms_2)

    return dist_val
end

"""
    CalcRg(atom_inds, correction)

Bias the radius of gyration.

# Arguments
- `atom_inds::Vector{Int}`: indices of atoms for which the radius of gyration should be calculated.
- `correction::Symbol=:pbc`: molecules are unwrapped prior to the calculation. Should be `:wrap` to keep system wrapped in the unit cell.
"""
@kwdef struct CalcRg
    atom_inds::Vector{Int} = Int[]
    correction::Symbol = :pbc
end

function calculate_cv(cv::CalcRg, coords, atoms, args...; kwargs...)
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    coords_used = @view coords[atom_inds_used]
    atoms_used = @view atoms[atom_inds_used]
    rg_val = radius_gyration(coords_used, atoms_used)
    return rg_val
end

"""
    CalcRMSD(atom_inds, ref_atom_inds, ref_coords, correction)

Bias the root-mean-square-deviation (RMSD) between the coordinates or a set of the coordinates of a system and a reference coordinate set.
The two sets of coordinates are superimposed using the Kabsch algorithm.

Superimposed coordinates are here obtained through translation of system coordinates and translation and rotation of reference coordinates.
The RMSD can be differentiated with respect to the system coordinates without differentiating through the Kabsch algorithm.

# Arguments
- `atom_inds::Vector{Int}`: indices of system (i.e., non-reference) coordinates to use in the RMSD calculation.
- `ref_atom_inds::Vector{Int}`: indices of reference coordinates to use in the RMSD calculation.
- `ref_coords::RC`: reference coordinate set.
- `correction::Symbol=:pbc`: molecules are unwrapped prior to the RMSD calculation. Should be `:wrap` to keep system wrapped in the unit cell.
"""
@kwdef struct CalcRMSD{RC}
    atom_inds::Vector{Int} = Int[]
    ref_atom_inds::Vector{Int} = Int[]
    ref_coords::RC
    correction::Symbol = :pbc
end

function calculate_cv(cv::CalcRMSD, coords, args...; kwargs...)

    # Say can be zero
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    coords_used = coords[atom_inds_used]

    ref_atom_inds_used = (iszero(length(cv.ref_atom_inds)) ? eachindex(cv.ref_coords) : cv.ref_atom_inds)
    ref_coords_used = cv.ref_coords[ref_atom_inds_used]

    p_rot = ref_kabsch(ref_coords_used, coords_used)        # translate and rotate ref coordinates
    trans = mean(coords_used)                               # mean coords
    trans_matrix = [trans for i in 1:length(coords_used)]   # mean coords matrix
    q = coords_used - trans_matrix                          # center coords
    diffs = p_rot - q
    rmsd_val = sqrt(mean(norm.(diffs).^2))

    return rmsd_val
end

function calculate_cv_ustrip!(unit_arr, args...)
    cv = calculate_cv(args...)
    # Enzyme requires a unitless value to be returned, so we strip the unit and store it
    unit_arr[1] = unit(cv)
    return ustrip(cv)
end

# Calculate the gradient of a CV with respect to the input coordinates
# When Enzyme is imported this defaults to using AD, but an explicit method
#   can be provided for a given CV type
# The AD approach should work with and without units
function cv_gradient end
