# Implicit solvent models
# Based on the OpenMM source code

export
    AbstractGBSA,
    ImplicitSolventOBC,
    born_radii_and_grad

"""
Generalized Born (GB) implicit solvent models augmented with the
hydrophobic solvent accessible surface area (SA) term.
Custom GBSA methods should sub-type this type.
"""
abstract type AbstractGBSA{T, D} end

"""
    ImplicitSolventOBC(atoms, atoms_data, bonds)

Onufriev-Bashford-Case GBSA model.
Should be used along with a Coulomb or CoulombReactionField interaction.
"""
struct ImplicitSolventOBC{T, D, V, S, F, I} <: AbstractGBSA{T, D}
    offset_radii::V
    scaled_offset_radii::V
    solvent_dielectric::T
    solute_dielectric::T
    offset::D
    cutoff::D
    use_ACE::Bool
    α::T
    β::T
    γ::T
    probe_radius::D
    sa_factor::S
    pre_factor::F
    is::I
    js::I
end

# Default solvent dielectric is 78.5 for consistency with AMBER
# Elsewhere it is 78.3
function ImplicitSolventOBC(atoms::AbstractArray{Atom{T, M, D, E}},
                            atoms_data,
                            bonds;
                            solvent_dielectric=78.5,
                            solute_dielectric=1.0,
                            offset=0.009u"nm",
                            cutoff=0.0u"nm",
                            probe_radius=0.14u"nm",
                            sa_factor=28.3919551u"kJ * mol^-1 * nm^-2",
                            use_ACE=true,
                            use_OBC2=false) where {T, M, D, E}
    default_radius = 0.15u"nm"
    element_to_radius = Dict(
        "N"  => 0.155u"nm",
        "O"  => 0.15u"nm" ,
        "F"  => 0.15u"nm" ,
        "Si" => 0.21u"nm" ,
        "P"  => 0.185u"nm",
        "S"  => 0.18u"nm" ,
        "Cl" => 0.17u"nm" ,
        "C"  => 0.17u"nm" ,
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

    offset_radii = D[]
    scaled_offset_radii = D[]
    for (at_data, bonded_to_N) in zip(atoms_data, atoms_bonded_to_N)
        if at_data.element in ("H", "D")
            radius = bonded_to_N ? 0.13u"nm" : 0.12u"nm"
        else
            radius = get(element_to_radius, at_data.element, default_radius)
        end
        offset_radius = radius - offset
        screen = get(element_to_screen, at_data.element, default_screen)
        if dimension(D) == u"𝐋"
            push!(offset_radii, offset_radius)
            push!(scaled_offset_radii, screen * offset_radius)
        else
            push!(offset_radii, ustrip(offset_radius))
            push!(scaled_offset_radii, ustrip(screen * offset_radius))
        end
    end

    if use_OBC2
        # GBOBCII parameters
        α, β, γ = T(1.0), T(0.8), T(4.85)
    else
        # GBOBCI parameters
        α, β, γ = T(0.8), T(0.0), T(2.909125)
    end

    inds_i = hcat([collect(1:n_atoms) for i in 1:n_atoms]...)
    inds_j = permutedims(inds_i, (2, 1))

    if !iszero(solute_dielectric) && !iszero(solvent_dielectric)
        pre_factor = T(-coulombconst) * (1/T(solute_dielectric) - 1/T(solvent_dielectric))
    else
        pre_factor = zero(T(coulombconst))
    end

    if isa(atoms, CuArray)
        or = CuArray(offset_radii) # cu converts to Float32
        sor = CuArray(scaled_offset_radii)
        is, js = cu(inds_i), cu(inds_j)
    else
        or = offset_radii
        sor = scaled_offset_radii
        is, js = inds_i, inds_j
    end

    if dimension(D) == u"𝐋"
        return ImplicitSolventOBC{T, D, typeof(or), typeof(T(sa_factor)), typeof(pre_factor), typeof(is)}(
                    or, sor, solvent_dielectric, solute_dielectric, offset, cutoff,
                    use_ACE, α, β, γ, probe_radius, T(sa_factor), pre_factor, is, js)
    else
        return ImplicitSolventOBC{T, T, typeof(or), T, T, typeof(is)}(
                    or, sor, solvent_dielectric, solute_dielectric, ustrip(offset), ustrip(cutoff),
                    use_ACE, α, β, γ, ustrip(probe_radius), ustrip(sa_factor), ustrip(pre_factor), is, js)
    end
end

function born_radii_loop(coord_i::SVector{D, T}, coord_j, ori, srj, cutoff, box_size) where {D, T}
    I = zero(coord_i[1] / unit(T)^2)
    r = norm(vector(coord_i, coord_j, box_size))
    if iszero(r) || (!iszero(cutoff) && r > cutoff)
        return I
    end
    U = r + srj
    if ori < U
        D_ij = abs(r - srj)
        L = max(ori, D_ij)
        I += (1/L - 1/U + (r - (srj^2)/r)*(1/(U^2) - 1/(L^2))/4 + log(L/U)/(2*r)) / 2
        if ori < (srj - r)
            I += 2 * (1/ori - 1/L)
        end
    end
    return I
end

"""
    born_radii_and_grad(inter, coords, box_size)

Calculate Born radii and gradients of Born radii with respect to atomic distance.
Custom GBSA methods should implement this function.
"""
function born_radii_and_grad(inter::ImplicitSolventOBC{T}, coords, box_size) where T
    coords_i = @view coords[inter.is]
    coords_j = @view coords[inter.js]
    oris = @view inter.offset_radii[inter.is]
    srjs = @view inter.scaled_offset_radii[inter.js]
    Is = sum(born_radii_loop.(coords_i, coords_j, oris, srjs, (inter.cutoff,), (box_size,)); dims=2)

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

# Store the results of the ij broadcasts during force calculation
struct ForceLoopResult1{T, V}
    bi::T
    bj::T
    fi::V
    fj::V
end

struct ForceLoopResult2{V}
    fi::V
    fj::V
end

function gb_force_loop_1(coord_i, coord_j, i, j, charge_i, charge_j,
                            Bi, Bj, cutoff, pre_factor, box_size)
    if j < i
        zero_force = zero(pre_factor ./ coord_i .^ 2)
        return ForceLoopResult1(zero_force[1], zero_force[1], zero_force, zero_force)
    end
    dr = vector(coord_i, coord_j, box_size)
    r2 = sum(abs2, dr)
    if !iszero(cutoff) && r2 > cutoff^2
        zero_force = zero(pre_factor ./ coord_i .^ 2)
        return ForceLoopResult1(zero_force[1], zero_force[1], zero_force, zero_force)
    end
    alpha2_ij = Bi * Bj
    D = r2 / (4 * alpha2_ij)
    exp_term = exp(-D)
    denominator2 = r2 + alpha2_ij * exp_term
    denominator = sqrt(denominator2)
    Gpol = (pre_factor * charge_i * charge_j) / denominator
    dGpol_dr = -Gpol * (1 - exp_term/4) / denominator2
    dGpol_dalpha2_ij = -Gpol * exp_term * (1 + D) / (2 * denominator2)
    change_born_force_i = dGpol_dalpha2_ij * Bj
    if i != j
        change_born_force_j = dGpol_dalpha2_ij * Bi
        fdr = dr * dGpol_dr
        change_fs_i =  fdr
        change_fs_j = -fdr
        return ForceLoopResult1(change_born_force_i, change_born_force_j,
                                change_fs_i, change_fs_j)
    else
        zero_force = zero(pre_factor ./ coord_i .^ 2)
        return ForceLoopResult1(change_born_force_i, zero_force[1], zero_force, zero_force)
    end
end

function gb_force_loop_2(coord_i, coord_j, bi, ori, srj, cutoff, box_size)
    dr = vector(coord_i, coord_j, box_size)
    r = norm(dr)
    if iszero(r) || (!iszero(cutoff) && r > cutoff)
        zero_force = zero(bi ./ coord_i .^ 2)
        return ForceLoopResult2(zero_force, zero_force)
    end
    rsrj = r + srj
    if ori < rsrj
        D = abs(r - srj)
        L = inv(max(ori, D))
        U = inv(rsrj)
        rinv = inv(r)
        r2inv = rinv^2
        t3 = (1 + (srj^2)*r2inv)*(L^2 - U^2)/8 + log(U/L)*r2inv/4
        de = bi * t3 * rinv
        fdr = dr * de
        return ForceLoopResult2(-fdr, fdr)
    else
        zero_force = zero(bi ./ coord_i .^ 2)
        return ForceLoopResult2(zero_force, zero_force)
    end
end

function forces(inter::AbstractGBSA{T, D}, sys, neighbors=nothing) where {T, D}
    n_atoms = length(sys)
    coords, atoms, box_size = sys.coords, sys.atoms, sys.box_size
    Bs, B_grads = born_radii_and_grad(inter, coords, box_size)

    if inter.use_ACE
        radii = inter.offset_radii .+ inter.offset
        sa_terms = inter.sa_factor .* (radii .+ inter.probe_radius) .^ 2 .* (radii ./ Bs) .^ 6
        born_forces = (-6 .* sa_terms ./ Bs) .* (Bs .> zero(D))
    else
        born_forces = zeros(typeof(inter.sa_factor * inter.cutoff), n_atoms)
    end

    coords_i = @view coords[inter.is]
    coords_j = @view coords[inter.js]
    charges = charge.(atoms)
    charges_i = @view charges[inter.is]
    charges_j = @view charges[inter.js]
    Bsi = @view Bs[inter.is]
    Bsj = @view Bs[inter.js]
    loop_res_1 = gb_force_loop_1.(coords_i, coords_j, inter.is, inter.js, charges_i, charges_j,
                                    Bsi, Bsj, (inter.cutoff,), (inter.pre_factor,), (box_size,))
    born_forces = born_forces .+ sum(lr -> lr.bi, loop_res_1; dims=2)[:, 1]
    born_forces = born_forces .+ sum(lr -> lr.bj, loop_res_1; dims=1)[1, :]
    fs = sum(lr -> lr.fi, loop_res_1; dims=2)[:, 1] .+ sum(lr -> lr.fj, loop_res_1; dims=1)[1, :]

    born_forces_2 = born_forces .* (Bs .^ 2) .* B_grads

    bis = @view born_forces_2[inter.is]
    oris = @view inter.offset_radii[inter.is]
    srjs = @view inter.scaled_offset_radii[inter.js]
    loop_res_2 = gb_force_loop_2.(coords_i, coords_j, bis, oris, srjs, (inter.cutoff,), (box_size,))

    return fs .+ sum(lr -> lr.fi, loop_res_2; dims=2)[:, 1] .+ sum(lr -> lr.fj, loop_res_2; dims=1)[1, :]
end

function potential_energy(inter::AbstractGBSA{T, D}, sys, neighbors=nothing) where {T, D}
    n_atoms = length(sys)
    coords, atoms, box_size = sys.coords, sys.atoms, sys.box_size
    Bs, B_grads = born_radii_and_grad(inter, coords, box_size)

    E = zero(inter.pre_factor / inter.offset)
    for i in 1:n_atoms
        charge_i = atoms[i].charge
        Bi = Bs[i]
        E += inter.pre_factor * (charge_i^2) / (2*Bi)
        if inter.use_ACE
            if Bi > zero(D)
                radius_i = inter.offset_radii[i] + inter.offset
                E += inter.sa_factor * (radius_i + inter.probe_radius)^2 * (radius_i / Bi)^6
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
            E += inter.pre_factor * charge_i * atoms[j].charge * f_cutoff
        end
    end

    return E
end
