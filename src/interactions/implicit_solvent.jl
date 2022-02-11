export ImplicitSolventOBC

"""
    ImplicitSolventOBC(atoms, atoms_data, bonds)

Onufriev-Bashford-Case GBSA model.
"""
struct ImplicitSolventOBC{T, R, I}
    offset_radii::Vector{R}
    scaled_offset_radii::Vector{R}
    solvent_dielectric::T
    solute_dielectric::T
    offset::R
    cutoff::R
    use_ACE::Bool
    α::T
    β::T
    γ::T
    is::I
    js::I
end

# Default solvent dielectric is 78.5 for consistency with AMBER
# Elsewhere it is 78.3
function ImplicitSolventOBC(atoms,
                            atoms_data,
                            bonds;
                            solvent_dielectric=78.5,
                            solute_dielectric=1.0,
                            offset=0.009,
                            cutoff=0.0,
                            use_ACE=true,
                            use_OBC2=false)
    # See OpenMM source code
    default_radius = 0.15 # in nm
    element_to_radius = Dict(
        "N"  => 0.155,
        "O"  => 0.15 ,
        "F"  => 0.15 ,
        "Si" => 0.21 ,
        "P"  => 0.185,
        "S"  => 0.18 ,
        "Cl" => 0.17 ,
        "C"  => 0.17 ,
    )
    default_screen = 0.8
    element_to_screen = Dict(
        "H" => 0.85,
        "C" => 0.72,
        "N" => 0.79,
        "O" => 0.85,
        "F" => 0.88,
        "P" => 0.86,
        "S" => 0.96,
    )

    # Find atoms bonded to nitrogen
    n_atoms = length(atoms)
    atoms_bonded_to_N = falses(n_atoms)
    for (i, j) in zip(bonds.is, bonds.js)
        if atoms_data[i].element == "N"
            atoms_bonded_to_N[j] = true
        end
        if atoms_data[j].element == "N"
            atoms_bonded_to_N[i] = true
        end
    end

    T = typeof(first(atoms).charge)
    offset_radii = T[]
    scaled_offset_radii = T[]
    for (at, at_data, bonded_to_N) in zip(atoms, atoms_data, atoms_bonded_to_N)
        if at_data.element in ("H", "D")
            radius = bonded_to_N ? 0.13 : 0.12
        else
            radius = get(element_to_radius, at_data.element, default_radius)
        end
        offset_radius = radius - offset
        screen = get(element_to_screen, at_data.element, default_screen)
        push!(offset_radii, offset_radius)
        push!(scaled_offset_radii, screen * offset_radius)
    end

    if use_OBC2
        # GBOBCII parameters
        α, β, γ = T(1.0), T(0.8), T(4.85)
    else
        # GBOBCI parameters
        α, β, γ = T(0.8), T(0.0), T(2.909125)
    end

    is = hcat([collect(1:n_atoms) for i in 1:n_atoms]...)
    js = permutedims(is, (2, 1))
    return ImplicitSolventOBC{T, T, typeof(is)}(offset_radii, scaled_offset_radii,
                solvent_dielectric, solute_dielectric, offset, cutoff, use_ACE,
                α, β, γ, is, js)
end

# Calculate Born radii and gradients with respect to atomic distance
function born_radii_and_grad(inter::ImplicitSolventOBC{T}, coords, box_size) where T
    coords_i = @view coords[inter.is]
    coords_j = @view coords[inter.js]
    oris = @view inter.offset_radii[inter.is]
    srjs = @view inter.scaled_offset_radii[inter.js]
    Is_2D = broadcast(coords_i, coords_j, oris, srjs, (box_size,)) do coord_i, coord_j, ori, srj, bs
        I = zero(T)
        r = norm(vector(coord_i, coord_j, bs))
        if iszero(r) || (!iszero(inter.cutoff) && r > inter.cutoff)
            return I
        end
        U = r + srj
        if ori < U
            D = abs(r - srj)
            L = max(ori, D)
            I += (1/L - 1/U + (r - (srj^2)/r)*(1/(U^2) - 1/(L^2))/4 + log(L/U)/(2*r)) / 2
            if ori < (srj - r)
                I += 2 * (1/ori - 1/L)
            end
        end
        return I
    end
    Is = sum(Is_2D; dims=2)

    n_atoms = length(coords)
    Bs, B_grads = T[], T[]
    α, β, γ = inter.α, inter.β, inter.γ
    for i in 1:n_atoms
        ori = inter.offset_radii[i]
        radius_i = ori + inter.offset
        psi = Is[i] * ori
        psi2 = psi^2
        tanh_sum = tanh(α*psi - β*psi2 + γ*psi2*psi)
        B = 1 / (1/ori - tanh_sum/radius_i)
        grad_term = ori * (α - 2*β*psi + 3*γ*psi2)
        B_grad = (1 - tanh_sum^2) * grad_term / radius_i
        push!(Bs, B)
        push!(B_grads, B_grad)
    end

    ori = inter.offset_radii
    radii = ori .+ inter.offset
    ψs = Is .* ori
    ψs2 = ψs .^ 2
    α, β, γ = inter.α, inter.β, inter.γ
    tanh_sums = tanh.(α .* ψs .- β .* ψs2 .+ γ .* ψs2 .* ψs)
    Bs = 1 ./ (1 ./ ori .- tanh_sums ./ radii)
    grad_terms = ori .* (α .- 2 .* β .* ψs .+ 3 .* γ .* ψs2)
    B_grads = (1 .- tanh_sums .^ 2) .* grad_terms ./ radii

    return Bs, B_grads
end

function forces(inter::ImplicitSolventOBC{T}, sys, neighbors) where T
    n_atoms = length(sys)
    coords, atoms, box_size = sys.coords, sys.atoms, sys.box_size
    Bs, B_grads = born_radii_and_grad(inter, coords, box_size)

    born_forces = zeros(T, n_atoms)
    if inter.use_ACE
        for i in 1:n_atoms
            Bi = Bs[i]
            if Bi > 0
                radius_i = inter.offset_radii[i] + inter.offset
                probe_radius = T(0.14)
                sa_term = T(28.3919551) * (radius_i + probe_radius)^2 * (radius_i / Bi)^6
                born_forces[i] -= 6 * sa_term / Bi
            end
        end
    end

    fs = ustrip_vec.(zero(coords))
    if !iszero(inter.solute_dielectric) && !iszero(inter.solvent_dielectric)
        pre_factor = T(-138.935485) * (1/inter.solute_dielectric - 1/inter.solvent_dielectric)
    else
        pre_factor = zero(T)
    end
    for i in 1:n_atoms
        Bi = Bs[i]
        charge_i_fac = pre_factor * atoms[i].charge
        for j in i:n_atoms
            Bj = Bs[j]
            dr = vector(coords[i], coords[j], box_size)
            r2 = sum(abs2, dr)
            if !iszero(inter.cutoff) && r2 > inter.cutoff^2
                continue
            end
            alpha2_ij = Bi * Bj
            D_ij = r2 / (4 * alpha2_ij)
            exp_term = exp(-D_ij)
            denominator2 = r2 + alpha2_ij * exp_term
            denominator = sqrt(denominator2)
            Gpol = (charge_i_fac * atoms[j].charge) / denominator
            dGpol_dr = -Gpol * (1 - exp_term/4) / denominator2
            dGpol_dalpha2_ij = -Gpol * exp_term * (1 + D_ij) / (2 * denominator2)
            if i != j
                born_forces[j] += dGpol_dalpha2_ij * Bi
                fdr = dr * dGpol_dr
                fs[i] += fdr
                fs[j] -= fdr
            end
            born_forces[i] += dGpol_dalpha2_ij * Bj
        end
    end

    for i in 1:n_atoms
        born_forces[i] *= Bs[i]^2 * B_grads[i]
    end

    for i in 1:n_atoms
        ori = inter.offset_radii[i]
        for j in 1:n_atoms
            i == j && continue
            dr = vector(coords[i], coords[j], box_size)
            r = norm(dr)
            if !iszero(inter.cutoff) && r > inter.cutoff
                continue
            end
            srj = inter.scaled_offset_radii[j]
            rsrj = r + srj
            if ori < rsrj
                D = abs(r - srj)
                L = inv(max(ori, D))
                U = inv(rsrj)
                rinv = inv(r)
                r2inv = rinv^2
                t3 = (1 + (srj^2)*r2inv)*(L^2 - U^2)/8 + log(U/L)*r2inv/4
                de = born_forces[i] * t3 * rinv
                fdr = dr * de
                fs[i] -= fdr
                fs[j] += fdr
            end
        end
    end

    return fs
end

function potential_energy(inter::ImplicitSolventOBC{T}, sys, neighbors) where T
    n_atoms = length(sys)
    coords, atoms, box_size = sys.coords, sys.atoms, sys.box_size
    Bs, B_grads = born_radii_and_grad(inter, coords, box_size)

    E = zero(T)
    if !iszero(inter.solute_dielectric) && !iszero(inter.solvent_dielectric)
        pre_factor = T(-138.935485) * (1/inter.solute_dielectric - 1/inter.solvent_dielectric)
    else
        pre_factor = zero(T)
    end
    for i in 1:n_atoms
        charge_i = atoms[i].charge
        Bi = Bs[i]
        E += pre_factor * (charge_i^2) / (2*Bi)
        if inter.use_ACE
            if Bi > 0
                radius_i = inter.offset_radii[i] + inter.offset
                E += T(28.3919551) * (radius_i + T(0.14))^2 * (radius_i / Bi)^6
            end
        end
        for j in (i + 1):n_atoms
            Bj = Bs[j]
            r2 = square_distance(i, j, coords, box_size)
            if !iszero(inter.cutoff) && r2 > inter.cutoff^2
                continue
            end
            f = sqrt(r2 + Bi*Bj*exp(-r2/(4*Bi*Bj)))
            if iszero(inter.cutoff)
                f_cutoff = 1/f
            else
                f_cutoff = (1/f - 1/inter.cutoff)
            end
            E += pre_factor * charge_i * atoms[j].charge * f_cutoff
        end
    end

    return E
end
