# Calculate collective variables

export
    CalcMinDist,
    CalcMaxDist,
    CalcComDist,
    CalcSingleDist,
    CalcDist,
    CalcRg,
    CalcRMSD,
    calculate_cv

@doc raw"""
    centre_of_mass(coords, atoms)

Calculate the centre of mass for a set of coordinates 
without accounting for periodic boundary conditions.
"""
function centre_of_mass(coords, atoms)
    masses = mass.(atoms)
    com = sum(masses .* coords, dims=1) ./ sum(masses)
    com = com[1]
    return com 
end

@doc raw"""
    pairwise_distance_matrix(coords_1, coords_2, calc_type, boundary)

Calculate all pairwise distances between two sets of coordinates.

When `calc_type` is `:closest`, periodic boundary conditions are accounted for in the 
distance calculations. To calculate raw distances instead, `calc_type` should be `:raw`.
"""
function pairwise_distance_matrix( 
        coords_1::AbstractArray{SVector{D, T}},
        coords_2::AbstractArray{SVector{D, T}},
        calc_type,
        boundary) where {D, T} 

    dist_matrix = zeros(T, (length(coords_1), length(coords_2)))	
    u = unit(coords_1[1][1]) 
    dist_matrix[:,:] .= NaN * u

    if calc_type == :closest
        for i=1:length(coords_1), j=1:length(coords_2)
            dist_matrix[i,j] = norm(vector(coords_1[i],coords_2[j],boundary)) 
        end
    else
        for i=1:length(coords_1), j=1:length(coords_2)
            dist_matrix[i,j] = norm(coords_1[i]-coords_2[j]) 
        end
    end 
        
    return dist_matrix
end

@doc raw"""
    CalcMinDist

Bias the minimum distance between two groups of atoms. 

# Arguments
- `calc_type::Symbol`: Should be `:closest` to calculate distances between closest periodic images and `:raw` otherwise. 
"""
struct CalcMinDist
    calc_type::Symbol 
end

function dist_between_groups(md::CalcMinDist, coords_1, coords_2, boundary, args...; kwargs...)    
    dist_matrix = pairwise_distance_matrix(coords_1, coords_2, md.calc_type, boundary) 
    min_dist, min_dist_pair = findmin(dist_matrix) 
    return min_dist 
end

@doc raw"""
    CalcMaxDist

Bias the maximum distance between two groups of atoms. 

# Arguments
- `calc_type::Symbol`: Should be `:closest` to calculate distances between closest periodic images and `:raw` otherwise. 
"""
struct CalcMaxDist 
    calc_type::Symbol 
end

function dist_between_groups(md::CalcMaxDist, coords_1, coords_2, boundary, args...; kwargs...)     
    dist_matrix = pairwise_distance_matrix(coords_1, coords_2, md.calc_type, boundary)  
    max_dist, max_dist_pair = findmax(dist_matrix) 
    return max_dist 
end

@doc raw"""
    CalcComDist

Bias the distance between the centres of mass of two groups of atoms. 

# Arguments
- `calc_type::Symbol`: Should be `:closest` to calculate distances between closest periodic images and `:raw` otherwise. 
"""
struct CalcComDist  
    calc_type::Symbol
end

function dist_between_groups(cd::CalcComDist, coords_1, coords_2, boundary, atoms_1, atoms_2, args...; kwargs...)   
    com_1 = centre_of_mass(coords_1,atoms_1)
    com_2 = centre_of_mass(coords_2,atoms_2)
    if cd.calc_type == :closest
        com_dist_val = norm(vector(com_1, com_2, boundary))  
    else
        com_dist_val = norm(com_1 - com_2)
    end
    return com_dist_val 
end

@doc raw"""
    CalcSingleDist

Bias the distance between two atoms.

# Arguments
- `calc_type::Symbol`: Should be `:closest` to calculate distances between closest periodic images and `:raw` otherwise. 
"""
struct CalcSingleDist  
    calc_type::Symbol
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

@doc raw"""
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

function calculate_cv(cv::CalcDist, coords, atoms, boundary, args...; kwargs...) 

    coords_1 = @view coords[cv.atom_inds_1]  
    coords_2 = @view coords[cv.atom_inds_2]

    atoms_1 = @view atoms[cv.atom_inds_1]
    atoms_2 = @view atoms[cv.atom_inds_2]

    dist_val = dist_between_groups(cv.dist_type, coords_1, coords_2, boundary, atoms_1, atoms_2)
    
    return dist_val
end

@doc raw"""
    CalcRg(atom_inds, correction)

Bias the radius of gyration. 

# Arguments
- `atom_inds::Vector{Int}`: indices of atoms for which the radius of gyration should be calculated.
- `correction::Symbol=:pbc`: molecules are unwrapped prior to the calculation. Should be `:wrap` to keep system wrapped in the unit cell. 
"""
@kwdef struct CalcRg 
    atom_inds::Vector{Int}
    correction::Symbol = :pbc
end

function calculate_cv(cv::CalcRg, coords, atoms, args...; kwargs...) 
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)  
    coords_used = @view coords[atom_inds_used]
    atoms_used = @view atoms[atom_inds_used]
    rg_val = radius_gyration(coords_used, atoms_used)
    return rg_val
end

@doc raw"""
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
    atom_inds::Vector{Int}
    ref_atom_inds::Vector{Int}
    ref_coords::RC       
    correction::Symbol = :pbc
end

function calculate_cv(cv::CalcRMSD, coords, args...; kwargs...) 

    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds) 
    coords_used = coords[atom_inds_used]

    ref_atom_inds_used = (iszero(length(cv.ref_atom_inds)) ? eachindex(cv.ref_coords) : cv.ref_atom_inds) 
    ref_coords_used = cv.ref_coords[ref_atom_inds_used]

    p_rot = Molly.ref_kabsch(ref_coords_used, coords_used)  # translate and rotate ref coordinates   
    trans = mean(coords_used)                               # mean coords
    trans_matrix = [trans for i in 1:length(coords_used)]   # mean coords matrix
    q = coords_used - trans_matrix                          # center coords
    diffs = p_rot - q                                
    rmsd_val = sqrt(mean(norm.(diffs).^2))

    return rmsd_val
end

#@doc raw"""
#    calculate_cv_ustrip!(unit_arr, args...)
#
#Calculate the value of a collective variable of an input system with `calculate_cv(cv::cv_type, args...)` and return the value stripped of its unit. 
#"""#
function calculate_cv_ustrip!(unit_arr, args...)
    cv = calculate_cv(args...)  # calculcate cv
    unit_arr[1] = unit(cv)      # infer units of cv
    return ustrip(cv)           # return unitless cv
end

#@doc raw"""
#    cv_gradient(cv_type, coords, atoms, boundary, velocities)
#
#Calculate the gradient of a collective variable of type `cv_type` with respect to the input coordinates. 
#
#Gradients of collective variables are generally calculated with automatic differentiation (AD), unless a `cv_gradient` method that does not rely on AD is defined for the input `cv_type`. 
#
#This method can be run on either unitful or unitless inputs, since `calculate_cv_ustrip!` is called prior to the AD call. 
#"""
function cv_gradient end