# Calculate collective variables

export
    CalcDist,
    CalcRg,
    CalcRMSD,
    calculate_cv
    #calculate_cv_ustrip!, # dont export
    #cv_gradient #dont export

@doc raw"""
    CalcDist(atom_idx_1, atom_idx_2, correction)

Bias the distance between two atoms in a system. 

# Arguments
- `atom_idx_1::Int`: index of the first atom.
- `atom_idx_2::Int`: index of the second atom.
- `dist_type::Symbol=:pbc_dist`: distances are evaluated taking periodic boundary conditions into account. Should be `:dist` if PBCs should not be considered.
- `correction::Symbol=:wrap`: the system is wrapped in the unit cell. Should be `:pbc` to unwrap molecules prior to the distance calculation. 
"""
@kwdef struct CalcDist   
    atom_idx_1::Int     
    atom_idx_2::Int
    dist_type::Symbol=:pbc_dist   
    correction::Symbol=:wrap
end

function calculate_cv(cv::CalcDist, coords, atoms, boundary, args...; kwargs...) 
    coords_1 = coords[cv.atom_idx_1]
    coords_2 = coords[cv.atom_idx_2]
    if cv.dist_type == :pbc_dist
        dist_val = distances([coords_1,coords_2], boundary)[2]
    else
        dist_val = norm(coords_1 - coords_2)
    end
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

    p_rot_reshaped, _ = kabsch(ref_coords_used, coords_used)  # translate and rotate ref coordinates   
    trans = mean(coords_used)                                 # mean coords
    trans_matrix = [trans for i in 1:length(coords_used)]     # mean coords matrix
    q = coords_used - trans_matrix                            # center coords
    diffs = p_rot_reshaped - q                                
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