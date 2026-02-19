# Calculate collective variables

export
    CalcMinDist,
    CalcMaxDist,
    CalcCMDist,
    CalcSingleDist,
    CalcDist,
    calculate_cv,
    cv_gradient,
    CalcRg,
    CalcRMSD,
    CalcTorsion

# Does not account for periodic boundary conditions, assumes appropriate unwrapping
function center_of_mass(coords, atoms)
    masses = mass.(atoms)
    com = sum(masses .* coords; dims=1) ./ sum(masses)
    return only(com)
end

function calculate_virial(cv, args...; kwargs...) end

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
    has_virial::Bool

    function CalcDist(atom_inds_1, atom_inds_2, dist_type::DT=CalcMinDist(),
                      correction=:pbc, has_virial = true) where DT
        check_correction_arg(correction)
        return new{DT}(atom_inds_1, atom_inds_2, dist_type, correction, has_virial)
    end
end

"""
    calculate_cv(cv, coords, atoms, boundary, velocities; kwargs...)

Calculate the value of a collective variable (CV) with the current system state.

New CV types should implement this function.
This function does not apply the molecule correction over the boundaries; if
required, `coords` can be obtained from `unwrap_molecules` first.
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

@doc raw"""
    cv_gradient(cv::CalcDist{CalcSingleDist}, coords, atoms, boundary, velocities; kwargs...)

Computes the analytical gradient of the distance between two atoms.

### Mathematics
Let the coordinates of the two atoms be $\mathbf{r}_i$ and $\mathbf{r}_j$.
The minimum image vector from atom $i$ to atom $j$ is $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$.
The distance is given by $d = |\mathbf{r}_{ij}|$.
The gradients with respect to the atomic coordinates are:
```math
\nabla_{\mathbf{r}_i} d = -\frac{\mathbf{r}_{ij}}{d}, \quad \nabla_{\mathbf{r}_j} d = \frac{\mathbf{r}_{ij}}{d}
```
"""
function cv_gradient(cv::CalcDist{CalcSingleDist}, coords, atoms, boundary, velocities; kwargs...)
    i, j = cv.atom_inds_1[1], cv.atom_inds_2[1]
    c1, c2 = coords[i], coords[j]

    if cv.dist_type.calc_type == :closest
        r_ij = vector(c1, c2, boundary)
    else
        r_ij = c2 - c1
    end

    d = norm(r_ij)
    grad = ustrip.(zero(coords))

    if d > 0u"nm"
        dir = r_ij / d
        grad[i] = -dir
        grad[j] = dir
    end

    return grad, d
end

@doc raw"""
    cv_gradient(cv::CalcDist{CalcMinDist}, coords, atoms, boundary, velocities; kwargs...)

Computes the analytical gradient of the minimum distance between two groups of atoms.

### Mathematics
Let A and B be two sets of atoms. The minimum distance is defined by the specific pair 
$\left( i^*, j^* \right)$ ∈  A x B that minimizes $d_{i,j}​ = ∣ \mathbf{r}_{i,j}​∣$.
The gradient evaluates to zero for all atoms except $i^∗$ and $j^∗$, for which it reduces to
the single distance gradient:
```math
\nabla_{\mathbf{r}_{i^*}} d = -\frac{\mathbf{r}_{i^*j^*}}{d}, \quad \nabla_{\mathbf{r}_{j^*}} d = \frac{\mathbf{r}_{i^*j^*}}{d}
```
"""
function cv_gradient(cv::CalcDist{CalcMinDist}, coords, atoms, boundary, velocities; kwargs...)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]

    min_d2 = typemax(eltype(eltype(coords)))u"nm"
    min_idx = (1, 1)

    if cv.dist_type.calc_type == :closest
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, vector(p1, p2, boundary))
            if d2 < min_d2
                min_d2 = d2
                min_idx = (i, j)
            end
        end
        r_ij = vector(c1[min_idx[1]], c2[min_idx[2]], boundary)
    else
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, p2 - p1)
            if d2 < min_d2
                min_d2 = d2
                min_idx = (i, j)
            end
        end
        r_ij = c2[min_idx[2]] - c1[min_idx[1]]
    end

    d = sqrt(min_d2)
    grad = ustrip.(zero(coords))

    if d > 0u"nm"
        dir = r_ij / d
        grad[cv.atom_inds_1[min_idx[1]]] = -dir
        grad[cv.atom_inds_2[min_idx[2]]] = dir
    end

    return grad, d

end

@doc raw"""
    cv_gradient(cv::CalcDist{CalcMaxDist}, coords, atoms, boundary, velocities; kwargs...)

Computes the analytical gradient of the maximum distance between two groups of atoms.
Mathematics

### Mathematics
Let A and B be two sets of atoms. The minimum distance is defined by the specific pair 
$\left( i^*, j^* \right)$ ∈  A x B that mmaximizes $d_{i,j}​ = ∣ \mathbf{r}_{i,j}​∣$.
The gradient is equivalent to the single distance gradient applied exclusively to this maximizing pair.
"""
function cv_gradient(cv::CalcDist{CalcMaxDist}, coords, atoms, boundary, velocities; kwargs...)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]

    max_d2 = typemin(eltype(eltype(coords)))u"nm"
    max_idx = (1, 1)

    if cv.dist_type.calc_type == :closest
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, vector(p1, p2, boundary))
            if d2 > max_d2
                max_d2 = d2
                max_idx = (i, j)
            end
        end
        r_ij = vector(c1[max_idx[1]], c2[max_idx[2]], boundary)
    else
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, p2 - p1)
            if d2 > max_d2
                max_d2 = d2
                max_idx = (i, j)
            end
        end
        r_ij = c2[max_idx[2]] - c1[max_idx[1]]
    end

    d = sqrt(max_d2)
    grad = ustrip.(zero(coords))

    if d > 0u"nm"
        dir = r_ij / d
        grad[cv.atom_inds_1[max_idx[1]]] = -dir
        grad[cv.atom_inds_2[max_idx[2]]] = dir
    end

    return grad, d

end

@doc raw"""
    cv_gradient(cv::CalcDist{CalcCMDist}, coords, atoms, boundary, velocities; kwargs...)

Computes the analytical gradient of the center-of-mass distance between two groups of atoms.

### Mathematics
Let $M_{\mathrm{A}}$ and $M_{\mathrm{B}}$ be the total masses of groups A and B.
Let $\mathbf{R}_{\mathrm{A}}$​ and $\mathbf{R}_{\mathrm{B}}$​ be their respective centers of mass, 
and $D = ∣\mathbf{R}_{\mathrm{B}} − \mathbf{R}_{\mathrm{A}}​∣$.

Applying the chain rule through the center of mass definition, the gradients for individual atoms are
proportional to their fractional mass:

```math
\nabla_{\mathbf{r}_i} D = -\frac{m_i}{M_A} \frac{\mathbf{R}_{AB}}{D} \quad \forall i \in A
```

and

```math
\nabla_{\mathbf{r}_j} D = \frac{m_j}{M_B} \frac{\mathbf{R}_{AB}}{D} \quad \forall j \in B
```
"""
function cv_gradient(cv::CalcDist{CalcCMDist}, coords, atoms, boundary, velocities; kwargs...)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    a1 = @view atoms[cv.atom_inds_1]
    a2 = @view atoms[cv.atom_inds_2]

    com1 = center_of_mass(c1, a1)
    com2 = center_of_mass(c2, a2)

    if cv.dist_type.calc_type == :closest
        r_12 = vector(com1, com2, boundary)
    else
        r_12 = com2 - com1
    end

    d = norm(r_12)
    grad = ustrip.(zero(coords))

    if d > 0u"nm"
        dir = r_12 / d
        
        m1 = mass.(a1)
        m2 = mass.(a2)
        M1 = sum(m1)
        M2 = sum(m2)
        
        for (idx, i) in enumerate(cv.atom_inds_1)
            grad[i] += -dir * (m1[idx] / M1)
        end
        for (idx, j) in enumerate(cv.atom_inds_2)
            grad[j] += dir * (m2[idx] / M2)
        end
    end

    return grad, d

end

function calculate_virial!(virial_buff, cv::CalcDist, coords, forces, atoms, boundary)
    calculate_virial_dist!(virial_buff, cv.dist_type, cv, coords, forces, atoms, boundary)
end

function calculate_virial_dist!(virial_buff, dt::CalcSingleDist, cv, coords, forces, atoms, boundary)
    i = cv.atom_inds_1[1]
    j = cv.atom_inds_2[1]
    f_i = forces[i]

    if dt.calc_type == :closest
        r_ji = vector(coords[j], coords[i], boundary)
    else
        r_ji = coords[i] - coords[j]
    end

    virial_buff .+= r_ji * transpose(f_i)
end

function calculate_virial_dist!(virial_buff, dt::CalcMinDist, cv, coords, forces, atoms, boundary)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    
    min_d2 = typemax(eltype(eltype(coords)))
    min_idx = (1, 1)
    
    # Find the pair minimizing the distance to get the correct vector
    if dt.calc_type == :closest
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, vector(p1, p2, boundary))
            if d2 < min_d2
                min_d2 = d2
                min_idx = (i, j)
            end
        end
        r_ji = vector(c2[min_idx[2]], c1[min_idx[1]], boundary)
    else
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, p2 - p1)
            if d2 < min_d2
                min_d2 = d2
                min_idx = (i, j)
            end
        end
        r_ji = c1[min_idx[1]] - c2[min_idx[2]]
    end
    
    # The total force on group 1 is the sum of forces on its atoms.
    # For MinDist, this is effectively the force on the closest atom.
    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_ji * transpose(f_sum)
end

function calculate_virial_dist!(virial_buff, dt::CalcMaxDist, cv, coords, forces, atoms, boundary)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    
    max_d2 = typemin(eltype(eltype(coords)))
    max_idx = (1, 1)
    
    # Find the pair maximizing the distance
    if dt.calc_type == :closest
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, vector(p1, p2, boundary))
            if d2 > max_d2
                max_d2 = d2
                max_idx = (i, j)
            end
        end
        r_ji = vector(c2[max_idx[2]], c1[max_idx[1]], boundary)
    else
        for (i, p1) in enumerate(c1), (j, p2) in enumerate(c2)
            d2 = sum(abs2, p2 - p1)
            if d2 > max_d2
                max_d2 = d2
                max_idx = (i, j)
            end
        end
        r_ji = c1[max_idx[1]] - c2[max_idx[2]]
    end
    
    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_ji * transpose(f_sum)
end

function calculate_virial_dist!(virial_buff, dt::CalcCMDist, cv, coords, forces, atoms, boundary)
    c1 = @view coords[cv.atom_inds_1]
    c2 = @view coords[cv.atom_inds_2]
    a1 = @view atoms[cv.atom_inds_1]
    a2 = @view atoms[cv.atom_inds_2]

    com1 = center_of_mass(c1, a1)
    com2 = center_of_mass(c2, a2)

    if dt.calc_type == :closest
        r_12 = vector(com2, com1, boundary)
    else
        r_12 = com1 - com2
    end

    f_sum = sum(forces[cv.atom_inds_1])
    virial_buff .+= r_12 * transpose(f_sum)
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
    has_virial::Bool

    function CalcRg(atom_inds=[], correction=:pbc, has_virial = true)
        check_correction_arg(correction)
        return new(atom_inds, correction, has_virial)
    end
end

function calculate_cv(cv::CalcRg, coords, atoms, args...; kwargs...)
    atom_inds_used = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    coords_used = @view coords[atom_inds_used]
    atoms_used = @view atoms[atom_inds_used]
    rg_val = radius_gyration(coords_used, atoms_used)
    return rg_val
end

@doc raw"""
    cv_gradient(cv::CalcRg, coords, atoms, boundary, velocities; kwargs...)

Computes the analytical gradient of the radius of gyration, as computed by
[`radius_gyration`](@ref).

### Mathematics
The mass-weighted radius of gyration is:

```math
R_{g} = \sqrt{\frac{1}{M} \sum_{k} m_{k} | \mathbf{r}_{k} - \mathbf{R}_{COM} |^{2}}
```

Differentiating with respect to the coordinates of atom $k$ yields:

```math
\nabla_{\mathbf{r}_k} R_g = \frac{m_k}{M \cdot R_g} (\mathbf{r}_k - \mathbf{R}_{COM})
```

Note: The derivative of the center of mass $R_{COM}$​ with respect to rk​ cancels out
in the summation due to the definition of the center of mass.

"""
function cv_gradient(cv::CalcRg, coords, atoms, boundary, velocities; kwargs...)
    atom_inds_used = iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds
    c_used = @view coords[atom_inds_used]
    a_used = @view atoms[atom_inds_used]

    com = center_of_mass(c_used, a_used)
    m_used = mass.(a_used)
    M_total = sum(m_used)

    rg_sq = zero(eltype(eltype(coords)))u"nm * g * mol^-1"
    for (idx, c) in enumerate(c_used)
        r_ic = vector(com, c, boundary)
        rg_sq += m_used[idx] * sum(abs2, r_ic)
    end
    rg_sq /= M_total
    rg = sqrt(rg_sq)

    grad = ustrip.(zero(coords))

    if rg > 0u"nm"
        factor = 1 / (M_total * rg)
        for (idx, i) in enumerate(atom_inds_used)
            r_ic = vector(com, c_used[idx], boundary)
            grad[i] += factor * m_used[idx] * r_ic
        end
    end

    return grad, rg

end

# For Rg and also for the RMSD the forces applied to the atoms 
# are dependent only on the relative configuration of said
# atoms, making them translationally invariant. Therefore:
#
# Σ F_i = 0
#
# We can exploit this fact to obtain the virial by computing 
#
# Ξ = Σ (r_i - r_COM) ⊗ F_i; 
#
# rearranging:
#
# Ξ = Σ ( r_i ⊗ F_i ) - r_COM ⊗ Σ F_i = Σ r_i ⊗ F_i
#
# which is equivalent to the standard definition of the virial!
# Note: we cannot just compute Σ r_i ⊗ F_i as this will give 
# different results depending on the choice of origin of coordinates.

function calculate_virial!(virial_buff, cv::CalcRg, coords, forces, atoms, boundary)
    # Select the relevant atoms/coordinates
    ids = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used = @view coords[ids]
    f_used = @view forces[ids]
    a_used = @view atoms[ids]
    
    # Calculate Center of Mass of the group to define relative coordinates
    com = center_of_mass(c_used, a_used)

    # Accumulate sum( (r_i - r_com) * F_i^T )
    for (c, f) in zip(c_used, f_used)
        # Vector from COM to atom i (r_i - r_com), handling PBC
        r_ic = vector(com, c, boundary) 
        virial_buff .+= r_ic * transpose(f)
    end
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
    has_virial::Bool

    function CalcRMSD(ref_coords, atom_inds=[], ref_atom_inds=[], correction=:pbc, has_virial = true)
        check_correction_arg(correction)
        ref_coords_cpu = from_device(ref_coords)
        RC = typeof(ref_coords_cpu)
        new{RC}(ref_coords_cpu, atom_inds, ref_atom_inds, correction, has_virial)
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

@doc raw"""
    cv_gradient(cv::CalcRMSD, coords, atoms, boundary, velocities; kwargs...)

Computes the analytical gradient of the optimal Root-Mean-Square Deviation (RMSD)
using Kabsch alignment.

### Mathematics
Let $\mathbf{r}_{k}^{sys}$ be the current system coordinates and $\mathbf{R}_{COM}^{sys}$
be their centroid. Let $\mathbf{r}_{k}^{ref}$ be the centered reference coordinates.
The optimally aligned RMSD distance is:

```math
d_{RMSD} = \sqrt{ \frac{1}{N} \sum_{k=1}^N \left| (\mathbf{r}_k^{\text{sys}} - \mathbf{R}_{COM}^{\text{sys}}) - Q \mathbf{r}_k^{\text{ref}} \right|^2 }
```

Because the rotation matrix $Q$ optimally minimizes the distance, the derivative of $Q$ with
respect to coordinates vanishes. The exact analytical gradient for an evaluated atom $k$ simplifies to:

```math
\nabla_{\mathbf{r}_k} d_{RMSD} = \frac{1}{N \cdot d_{RMSD}} \left( (\mathbf{r}_k^{\text{sys}} - \mathbf{R}_{COM}^{\text{sys}}) - Q \mathbf{r}_k^{\text{ref}} \right)
```
"""
function cv_gradient(cv::CalcRMSD, coords, atoms, boundary, velocities; kwargs...)
    atom_inds_used = iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds
    ref_atom_inds_used = iszero(length(cv.ref_atom_inds)) ? eachindex(cv.ref_coords) : cv.ref_atom_inds

    c_used = coords[atom_inds_used]
    ref_c_used = cv.ref_coords[ref_atom_inds_used]
    N = length(c_used)

    p_rot = kabsch_nograd(ref_c_used, c_used)
    centroid = mean(c_used)
    q = c_used .- (centroid,)
    diffs = p_rot .- q
    rmsd_val = sqrt(mean(sum_abs2, diffs))

    grad = ustrip.(zero(coords))

    if rmsd_val > 0u"nm"
        factor = 1 / (N * rmsd_val)
        for (idx, i) in enumerate(atom_inds_used)
            grad[i] += factor * -diffs[idx] # diffs is (p_rot - q), we need (q - p_rot)
        end
    end

    return grad, rmsd_val
end


function calculate_virial!(virial_buff, cv::CalcRMSD, coords, forces, atoms, boundary)
    # Select the relevant atoms/coordinates
    ids = (iszero(length(cv.atom_inds)) ? eachindex(coords) : cv.atom_inds)
    c_used = @view coords[ids]
    f_used = @view forces[ids]
    
    # RMSD with centering is translationally invariant.
    # We use the centroid of the current configuration as the reference point.
    com = mean(c_used)

    # Accumulate sum( (r_i - r_centroid) * F_i^T )
    for (c, f) in zip(c_used, f_used)
        # Vector from centroid to atom i
        r_ic = vector(com, c, boundary)
        virial_buff .+= r_ic * transpose(f)
    end
end

"""
    CalcTorsion(atom_inds::AbstractVector{Int}; correction=:pbc, has_virial::Bool=true)

A collective variable that calculates the torsion angle (dihedral) defined by four atoms.

The angle is defined by the intersection of the planes formed by atoms (i, j, k) and (j, k, l), where the indices are given by `atom_inds`.

# Fields
- `atom_inds::AbstractVector{Int}`: The indices of the four atoms (i, j, k, l) defining the torsion.
- `correction::Symbol`: The method used to handle periodic boundary conditions. Defaults to `:pbc`.
- `has_virial::Bool`: Whether the virial contribution should be calculated for this collective variable. Defaults to `true`.
"""
struct CalcTorsion
    atom_inds::Vector{Int}
    correction::Symbol
    has_virial::Bool

    function CalcTorsion(atom_inds=[],correction=:pbc, has_virial = true)
        check_correction_arg(correction)
        return new(atom_inds, correction, has_virial)
    end
end

function calculate_cv(cv::CalcTorsion, coords, atoms, boundary, args...; kwargs...)
    c = @view coords[collect(cv.atom_inds)]
    return  torsion_angle(c[1], c[2], c[3], c[4], boundary)
end

@doc raw"""
    cv_gradient(cv::CalcTorsion, coords, atoms, boundary, velocities; kwargs...)

Computes the analytical gradient of the torsion (dihedral) angle defined by four atoms.

### Mathematics
Let the four atoms be $i, j, k, l$. Define bond vectors: $\mathbf{b}_{1} = \mathbf{r}_{j} - \mathbf{r}_{i}$,
$\mathbf{b}_{2} = \mathbf{r}_{k} - \mathbf{r}_{j}$, $\mathbf{b}_{3} = \mathbf{r}_{l} - \mathbf{r}_{k}$.
Define normal vectors to the planes: $\mathbf{m} = \mathbf{b}_{1}$ x $\mathbf{b}_{2}$, 
$\mathbf{n} = \mathbf{b}_{2}$ x $\mathbf{b}_{3}$. The gradients are evaluated via the chain rule on:

```math
\phi = \mathrm{atan2}\left( |\mathbf{b}_{2}|\mathbf{b}_{1} \cdot \mathbf{n}, \mathbf{m} \cdot \mathbf{n} \right)
```

we get:

```math
\nabla_{\mathbf{r}_i} \phi =  \frac{|\mathbf{b}_2|}{|\mathbf{m}|^2} \mathbf{m}
```

```math
\nabla_{\mathbf{r}_l} \phi = -\frac{|\mathbf{b}_2|}{|\mathbf{n}|^2} \mathbf{n}
```

```math
\nabla_{\mathbf{r}_j} \phi = - \left( 1 + \frac{\mathbf{b}_1 \cdot \mathbf{b}_2}{|\mathbf{b}_2|^2} \right) \nabla_{\mathbf{r}_i} \phi + \left( \frac{\mathbf{b}_2 \cdot \mathbf{b}_3}{|\mathbf{b}_2|^2} \right) \nabla_{\mathbf{r}_l} \phi
```

```math
\nabla_{\mathbf{r}_k} \phi = \left( \frac{\mathbf{b}_1 \cdot \mathbf{b}_2}{|\mathbf{b}_2|^2} \right) \nabla_{\mathbf{r}_i} \phi - \left( 1 + \frac{\mathbf{b}_2 \cdot \mathbf{b}_3}{|\mathbf{b}_2|^2} \right) \nabla_{\mathbf{r}_l} \phi
```

"""
function cv_gradient(cv::CalcTorsion, coords, atoms, boundary, velocities; kwargs...)
    i, j, k, l = cv.atom_inds
    ri, rj, rk, rl = coords[i], coords[j], coords[k], coords[l]
    
    b1 = vector(ri, rj, boundary)
    b2 = vector(rj, rk, boundary)
    b3 = vector(rk, rl, boundary)
    
    m = cross(b1, b2)
    n = cross(b2, b3)
    
    m_sq = sum(abs2, m)
    n_sq = sum(abs2, n)
    b2_norm = norm(b2)
    b2_sq = b2_norm^2
    
    grad = ustrip.(zero(coords))u"nm^-1"
    phi = torsion_angle(ri, rj, rk, rl, boundary)
    
    if m_sq > 0u"nm^4" && n_sq > 0u"nm^4" && b2_sq > 0u"nm^2"
        grad_i =  (b2_norm / m_sq) * m
        grad_l = -(b2_norm / n_sq) * n
        
        b1_dot_b2 = dot(b1, b2)
        b3_dot_b2 = dot(b3, b2)
        
        grad_j = -(1 + b1_dot_b2 / b2_sq) * grad_i + (b3_dot_b2 / b2_sq) * grad_l
        grad_k = (b1_dot_b2 / b2_sq) * grad_i - (1 + b3_dot_b2 / b2_sq) * grad_l
        
        grad[i] = grad_i
        grad[j] = grad_j
        grad[k] = grad_k
        grad[l] = grad_l
    end
    
    return -grad, phi
end

function calculate_virial!(virial_buff, cv::CalcTorsion, coords, forces, atoms, boundary)
    ids = collect(cv.atom_inds)
    c = @view coords[ids]
    f = @view forces[ids]
    r_ji = vector(c[2], c[1], boundary) # r_i - r_j
    r_jk = vector(c[2], c[3], boundary) # r_k - r_j
    r_jl = vector(c[2], c[4], boundary) # r_l - r_j

    virial_buff .+= r_ji * transpose(f[1]) +
                    r_jk * transpose(f[3]) +
                    r_jl * transpose(f[4])
end

# Calculate the gradient of a CV with respect to the input coordinates
# When Enzyme is imported this defaults to using AD, but an explicit method
#   can be provided for a given CV type
# The AD approach should work with and without units
# Returns gradient and CV value
function cv_gradient end
