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

# Does not account for periodic boundary conditions, assumes appropriate unwrapping
function center_of_mass(coords, atoms)
    masses = mass.(atoms)
    com = sum(masses .* coords; dims=1) ./ sum(masses)
    return only(com)
end

function pairwise_distance_matrix(coords_1::AbstractArray{SVector{D, C}},
                                  coords_2::AbstractArray{SVector{D, C}},
                                  calc_type,
                                  boundary) where {D, C}
    dist_matrix = zeros(C, length(coords_1), length(coords_2))
    if calc_type == :closest
        for i in eachindex(coords_1), j in eachindex(coords_2)
            dist_matrix[i, j] = norm(vector(coords_1[i], coords_2[j], boundary))
        end
    else
        for i in eachindex(coords_1), j in eachindex(coords_2)
            dist_matrix[i, j] = norm(coords_2[i] - coords_1[j])
        end
    end
    return dist_matrix
end

function check_calc_type(calc_type)
    if !(calc_type in (:closest, :raw))
        throw(ArgumentError("calc_type argument must be :closest or :raw, found $calc_type"))
    end
end

"""
    CalcMinDist(calc_type=:closest)

Bias the minimum distance between two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcMinDist
    calc_type::Symbol

    function CalcMinDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(md::CalcMinDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_matrix = pairwise_distance_matrix(coords_1, coords_2, md.calc_type, boundary)
    return minimum(dist_matrix)
end

"""
    CalcMaxDist(calc_type=:closest)

Bias the maximum distance between two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcMaxDist
    calc_type::Symbol

    function CalcMaxDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(md::CalcMaxDist, coords_1, coords_2, boundary, args...; kwargs...)
    dist_matrix = pairwise_distance_matrix(coords_1, coords_2, md.calc_type, boundary)
    return maximum(dist_matrix)
end

"""
    CalcCMDist(calc_type=:closest)

Bias the distance between the centers of mass of two groups of atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

Should generally be used with molecule unwrapping since it assumes that the atoms
within each group are in the same periodic box.
If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcCMDist
    calc_type::Symbol

    function CalcCMDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(cd::CalcCMDist, coords_1, coords_2, boundary,
                             atoms_1, atoms_2, args...; kwargs...)
    com_1 = center_of_mass(coords_1, atoms_1)
    com_2 = center_of_mass(coords_2, atoms_2)
    if cd.calc_type == :closest
        com_dist_val = norm(vector(com_1, com_2, boundary))
    else
        com_dist_val = norm(com_2 - com_1)
    end
    return com_dist_val
end

"""
    CalcSingleDist(calc_type=:closest)

Bias the distance between two atoms.

Given as an argument to [`CalcDist`](@ref).
By default, distances are calculated between the closest periodic images.
Setting `calc_type=:raw` means that distances are calculated ignoring PBCs.

If distances are evaluated using the minimum image convention on an unwrapped system,
raw coordinates must be within a distance of 1.5x the box length of each other to
ensure correct results.
"""
struct CalcSingleDist
    calc_type::Symbol

    function CalcSingleDist(calc_type=:closest)
        check_calc_type(calc_type)
        new(calc_type)
    end
end

function dist_between_groups(sd::CalcSingleDist, coords_1, coords_2, boundary, args...; kwargs...)
    if length(coords_1) > 1 || length(coords_2) > 1
        throw(ArgumentError("CalcSingleDist can only be used with atom groups containing one atom"))
    end
    c1, c2 = only(coords_1), only(coords_2)
    if sd.calc_type == :closest
        dist_val = norm(vector(c1, c2, boundary))
    else
        dist_val = norm(c2 - c1)
    end
    return dist_val
end

"""
    CalcDist(atom_inds_1, atom_inds_2, dist_type=CalcMinDist(), correction=:pbc)

Bias the distance between two atoms or groups of atoms.

Given as an argument to [`BiasPotential`](@ref).

# Arguments
- `atom_inds_1`: indices of the atom(s) in the first group.
- `atom_inds_2`: indices of the atom(s) in the second group.
- `dist_type=CalcMinDist()`: type of distance to calculate.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. If using multiple atoms in
    a group, they should generally be in the same molecule and `:pbc` should be used.
"""
struct CalcDist{DT}
    atom_inds_1::Vector{Int}
    atom_inds_2::Vector{Int}
    dist_type::DT
    correction::Symbol

    function CalcDist(atom_inds_1, atom_inds_2, dist_type::DT=CalcMinDist(),
                      correction=:pbc) where DT
        check_correction_arg(correction)
        return new{DT}(atom_inds_1, atom_inds_2, dist_type, correction)
    end
end

"""
    calculate_cv(cv, coords, atoms, boundary, velocities; kwargs...)

Calculate the value of a collective variable (CV) with the current system state.

New CV types should implement this function.
This function does not apply the molecule correction over the boundaries; if
required, `coords` can be obtained from [`unwrap_molecules`](@ref) first.
The gradient of this function with respect to coordinates, used to calculate forces,
is by default calculated with automatic differentiation when Enzyme is imported.
Alternatively, the `cv_gradient` function can be defined for a new CV type.
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
    CalcRg(atom_inds=[], correction=:pbc)

Bias the radius of gyration of a group of atoms.

Given as an argument to [`BiasPotential`](@ref).

# Arguments
- `atom_inds=[]`: indices of the atoms in the group, `[]` uses all atoms.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. Generally atoms in a group
    should be in the same molecule and `:pbc` should be used.
"""
struct CalcRg
    atom_inds::Vector{Int}
    correction::Symbol

    function CalcRg(atom_inds=[], correction=:pbc)
        check_correction_arg(correction)
        return new(atom_inds, correction)
    end
end

function calculate_cv(cv::CalcRg, coords, atoms, args...; kwargs...)
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    coords_used = @view coords[atom_inds_used]
    atoms_used = @view atoms[atom_inds_used]
    rg_val = radius_gyration(coords_used, atoms_used)
    return rg_val
end

"""
    CalcRMSD(ref_coords, atom_inds=[], ref_atom_inds=[], correction=:pbc)

Bias the root-mean-square deviation (RMSD) between the coordinates of a group of atoms
and a set of reference coordinates.

Given as an argument to [`BiasPotential`](@ref).
The two sets of coordinates are superimposed using the Kabsch algorithm.

# Arguments
- `ref_coords`: reference coordinates.
- `atom_inds=[]`: indices of the atoms in the group, `[]` uses all atoms.
- `ref_atom_inds=[]`: indices of the reference coordinates to use, `[]` uses all coordinates.
- `correction=:pbc`: the correction to be applied to the molecules. `:pbc` keeps molecules
    whole, `:wrap` wraps all atoms inside the simulation box. Generally atoms in a group
    should be in the same molecule and `:pbc` should be used.
"""
struct CalcRMSD{RC}
    ref_coords::RC
    atom_inds::Vector{Int}
    ref_atom_inds::Vector{Int}
    correction::Symbol

    function CalcRMSD(ref_coords, atom_inds=[], ref_atom_inds=[], correction=:pbc)
        check_correction_arg(correction)
        ref_coords_cpu = from_device(ref_coords)
        RC = typeof(ref_coords_cpu)
        new{RC}(ref_coords_cpu, atom_inds, ref_atom_inds, correction)
    end
end

function calculate_cv(cv::CalcRMSD, coords, args...; kwargs...)
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    coords_used = coords[atom_inds_used]
    ref_atom_inds_used = (iszero(length(cv.ref_atom_inds)) ? eachindex(cv.ref_coords)
                                                           : cv.ref_atom_inds)
    ref_coords_used = cv.ref_coords[ref_atom_inds_used]

    # RMSD can be differentiated with respect to the system coordinates without
    #   differentiating through the Kabsch algorithm
    p_rot = kabsch_nograd(ref_coords_used, coords_used)
    trans = mean(coords_used)
    q = coords_used .- (trans,)
    diffs = p_rot .- q
    return sqrt(mean(sum_abs2, diffs))
end

function calculate_cv_ustrip!(unit_arr, args...)
    cv = calculate_cv(args...)
    # Enzyme requires a unitless value to be returned
    # We strip the unit, store it and add it back on later
    unit_arr[1] = unit(cv)
    return ustrip(cv)
end

# Calculate the gradient of a CV with respect to the input coordinates
# When Enzyme is imported this defaults to using AD, but an explicit method
#   can be provided for a given CV type
# The AD approach should work with and without units
# Returns gradient and CV value
function cv_gradient end
