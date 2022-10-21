# Chain rules to allow differentiable simulations

@non_differentiable CUDA.zeros(args...)
@non_differentiable random_velocities(args...)
@non_differentiable random_velocities!(args...)
@non_differentiable check_force_units(args...)
@non_differentiable atoms_bonded_to_N(args...)
@non_differentiable lookup_table(args...)
@non_differentiable find_neighbors(args...)
@non_differentiable DistanceVecNeighborFinder(args...)
@non_differentiable run_loggers!(args...)
@non_differentiable visualize(args...)
@non_differentiable place_atoms(args...)
@non_differentiable place_diatomics(args...)
@non_differentiable OpenMMForceField(T::Type, ff_files::AbstractString...)
@non_differentiable OpenMMForceField(ff_files::AbstractString...)
@non_differentiable System(coord_file::AbstractString, force_field::OpenMMForceField)
@non_differentiable System(T::Type, coord_file::AbstractString, top_file::AbstractString)
@non_differentiable System(coord_file::AbstractString, top_file::AbstractString)

function ChainRulesCore.rrule(T::Type{<:SVector}, vs::Number...)
    Y = T(vs...)
    function SVector_pullback(Ȳ)
        return NoTangent(), Ȳ...
    end
    return Y, SVector_pullback
end

function ChainRulesCore.rrule(T::Type{<:Atom}, vs...)
    Y = T(vs...)
    function Atom_pullback(Ȳ)
        return NoTangent(), Ȳ.index, Ȳ.charge, Ȳ.mass, Ȳ.σ, Ȳ.ϵ, Ȳ.solute
    end
    return Y, Atom_pullback
end

function ChainRulesCore.rrule(T::Type{<:SpecificInteraction}, vs...)
    Y = T(vs...)
    function SpecificInteraction_pullback(Ȳ)
        return NoTangent(), Ȳ...
    end
    return Y, SpecificInteraction_pullback
end

function ChainRulesCore.rrule(T::Type{<:PairwiseInteraction}, vs...)
    Y = T(vs...)
    function PairwiseInteraction_pullback(Ȳ)
        return NoTangent(), getfield.((Ȳ,), fieldnames(T))...
    end
    return Y, PairwiseInteraction_pullback
end

function ChainRulesCore.rrule(T::Type{<:InteractionList1Atoms}, vs...)
    Y = T(vs...)
    function InteractionList1Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.inters
    end
    return Y, InteractionList1Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:InteractionList2Atoms}, vs...)
    Y = T(vs...)
    function InteractionList2Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters
    end
    return Y, InteractionList2Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:InteractionList3Atoms}, vs...)
    Y = T(vs...)
    function InteractionList3Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters
    end
    return Y, InteractionList3Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:InteractionList4Atoms}, vs...)
    Y = T(vs...)
    function InteractionList4Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
               Ȳ.inters
    end
    return Y, InteractionList4Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:SpecificForce1Atoms}, vs...)
    Y = T(vs...)
    function SpecificForce1Atoms_pullback(Ȳ)
        return NoTangent(), Ȳ.f1
    end
    return Y, SpecificForce1Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:SpecificForce2Atoms}, vs...)
    Y = T(vs...)
    function SpecificForce2Atoms_pullback(Ȳ)
        return NoTangent(), Ȳ.f1, Ȳ.f2
    end
    return Y, SpecificForce2Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:SpecificForce3Atoms}, vs...)
    Y = T(vs...)
    function SpecificForce3Atoms_pullback(Ȳ)
        return NoTangent(), Ȳ.f1, Ȳ.f2, Ȳ.f3
    end
    return Y, SpecificForce3Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:SpecificForce4Atoms}, vs...)
    Y = T(vs...)
    function SpecificForce4Atoms_pullback(Ȳ)
        return NoTangent(), Ȳ.f1, Ȳ.f2, Ȳ.f3, Ȳ.f4
    end
    return Y, SpecificForce4Atoms_pullback
end

function ChainRulesCore.rrule(::typeof(sparsevec), is, vs, l)
    Y = sparsevec(is, vs, l)
    @views function sparsevec_pullback(Ȳ)
        return NoTangent(), nothing, Ȳ[is], nothing
    end
    return Y, sparsevec_pullback
end

# Required for SVector gradients in RescaleThermostat
function ChainRulesCore.rrule(::typeof(sqrt), x::Real)
    Y = sqrt(x)
    function sqrt_pullback(Ȳ)
        return NoTangent(), sum(Ȳ * inv(2 * Y))
    end
    return Y, sqrt_pullback
end

function ChainRulesCore.rrule(::typeof(reinterpret),
                                ::Type{T},
                                arr::SVector{D, T}) where {D, T}
    Y = reinterpret(T, arr)
    function reinterpret_pullback(Ȳ::Vector{T})
        return NoTangent(), NoTangent(), SVector{D, T}(Ȳ)
    end
    return Y, reinterpret_pullback
end

function ChainRulesCore.rrule(::typeof(reinterpret),
                                ::Type{T},
                                arr::AbstractArray{SVector{D, T}}) where {D, T}
    Y = reinterpret(T, arr)
    function reinterpret_pullback(Ȳ::Vector{T})
        return NoTangent(), NoTangent(), reinterpret(SVector{D, T}, Ȳ)
    end
    return Y, reinterpret_pullback
end

function ChainRulesCore.rrule(::typeof(reinterpret),
                                ::Type{SVector{D, T}},
                                arr::AbstractVector{T}) where {D, T}
    Y = reinterpret(SVector{D, T}, arr)
    function reinterpret_pullback(Ȳ::AbstractArray{SVector{D, T}})
        return NoTangent(), NoTangent(), reinterpret(T, Ȳ)
    end
    return Y, reinterpret_pullback
end

function ChainRulesCore.rrule(::typeof(sum_svec), arr::AbstractArray{SVector{D, T}}) where {D, T}
    Y = sum_svec(arr)
    function sum_svec_pullback(Ȳ::SVector{D, T})
        return NoTangent(), zero(arr) .+ (Ȳ,)
    end
    return Y, sum_svec_pullback
end

function ChainRulesCore.rrule(::typeof(mean), arr::AbstractArray{SVector{D, T}}) where {D, T}
    Y = mean(arr)
    function mean_pullback(Ȳ::SVector{D, T})
        return NoTangent(), zero(arr) .+ (Ȳ ./ length(arr),)
    end
    return Y, mean_pullback
end

function ChainRulesCore.rrule(T::Type{<:HarmonicBond}, vs...)
    Y = T(vs...)
    function HarmonicBond_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.r0
    end
    return Y, HarmonicBond_pullback
end

function ChainRulesCore.rrule(T::Type{<:HarmonicAngle}, vs...)
    Y = T(vs...)
    function HarmonicAngle_pullback(Ȳ)
        return NoTangent(), Ȳ.k, Ȳ.θ0
    end
    return Y, HarmonicAngle_pullback
end

function ChainRulesCore.rrule(T::Type{<:PeriodicTorsion}, vs...)
    Y = T(vs...)
    function PeriodicTorsion_pullback(Ȳ)
        return NoTangent(), NoTangent(), Ȳ.phases, Ȳ.ks, NoTangent()
    end
    return Y, PeriodicTorsion_pullback
end

duplicated_if_present(x, dx) = length(x) > 0 ? Duplicated(x, dx) : Const(x)

function ChainRulesCore.rrule(::typeof(forces_pair_spec), sys::System{D, G, T}, neighbors,
                              n_threads) where {D, G, T}
    if sys.force_units != NoUnits
        error("Taking gradients through force calculation is not compatible with units, " *
              "system force units are $(sys.force_units)")
    end
    Y = forces_pair_spec(sys, neighbors, n_threads)

    function forces_pair_spec_pullback(d_forces)
        fs = zero(sys.coords)
        z = zero(T)
        d_coords = zero(sys.coords)
        d_atoms = [Atom(charge=z, mass=z, σ=z, ϵ=z) for _ in 1:length(sys)]
        pairwise_inters_nonl = filter(inter -> !inter.nl_only, values(sys.pairwise_inters))
        pairwise_inters_nl   = filter(inter ->  inter.nl_only, values(sys.pairwise_inters))
        d_pairwise_inters_nonl = zero.(pairwise_inters_nonl)
        d_pairwise_inters_nl   = zero.(pairwise_inters_nl  )
        sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(sys.specific_inter_lists))
        sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
        sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
        sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))
        d_sils_1_atoms = zero.(sils_1_atoms)
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        autodiff(
            forces_pair_spec!,
            Const,
            Duplicated(fs, d_forces),
            Duplicated(sys.coords, d_coords),
            Duplicated(sys.atoms, d_atoms),
            duplicated_if_present(pairwise_inters_nonl, d_pairwise_inters_nonl),
            duplicated_if_present(pairwise_inters_nl  , d_pairwise_inters_nl  ),
            duplicated_if_present(sils_1_atoms, d_sils_1_atoms),
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            Const(sys.boundary),
            Const(sys.force_units),
            Const(neighbors),
            Const(n_threads),
        )
        d_sys = Tangent{System}(
            atoms=d_atoms,
            pairwise_inters=(d_pairwise_inters_nonl..., d_pairwise_inters_nl...),
            specific_inter_lists=(d_sils_1_atoms..., d_sils_2_atoms..., d_sils_3_atoms...,
                                  d_sils_4_atoms...),
            coords=d_coords,
            boundary=CubicBoundary(z, z, z),
        )
        return NoTangent(), d_sys, NoTangent(), NoTangent()
    end

    return Y, forces_pair_spec_pullback
end

function ChainRulesCore.rrule(::typeof(potential_energy_pair_spec), sys::System{D, G, T}, neighbors,
                              n_threads) where {D, G, T}
    if sys.energy_units != NoUnits
        error("Taking gradients through potential energy calculation is not compatible with " *
              "units, system energy units are $(sys.energy_units)")
    end
    Y = potential_energy_pair_spec(sys, neighbors, n_threads)

    function potential_energy_pair_spec_pullback(d_forces)
        z = zero(T)
        d_coords = zero(sys.coords)
        d_atoms = [Atom(charge=z, mass=z, σ=z, ϵ=z) for _ in 1:length(sys)]
        pairwise_inters_nonl = filter(inter -> !inter.nl_only, values(sys.pairwise_inters))
        pairwise_inters_nl   = filter(inter ->  inter.nl_only, values(sys.pairwise_inters))
        d_pairwise_inters_nonl = zero.(pairwise_inters_nonl)
        d_pairwise_inters_nl   = zero.(pairwise_inters_nl  )
        sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(sys.specific_inter_lists))
        sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
        sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
        sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))
        d_sils_1_atoms = zero.(sils_1_atoms)
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        autodiff(
            potential_energy_pair_spec,
            Active,
            Duplicated(sys.coords, d_coords),
            Duplicated(sys.atoms, d_atoms),
            duplicated_if_present(pairwise_inters_nonl, d_pairwise_inters_nonl),
            duplicated_if_present(pairwise_inters_nl  , d_pairwise_inters_nl  ),
            duplicated_if_present(sils_1_atoms, d_sils_1_atoms),
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            Const(sys.boundary),
            Const(sys.energy_units),
            Const(neighbors),
            Const(n_threads),
            Const(Val(T)),
        )
        d_sys = Tangent{System}(
            atoms=d_atoms,
            pairwise_inters=(d_pairwise_inters_nonl..., d_pairwise_inters_nl...),
            specific_inter_lists=(d_sils_1_atoms..., d_sils_2_atoms..., d_sils_3_atoms...,
                                  d_sils_4_atoms...),
            coords=d_coords,
            boundary=CubicBoundary(z, z, z),
        )
        return NoTangent(), d_sys, NoTangent(), NoTangent()
    end

    return Y, potential_energy_pair_spec_pullback
end

function grad_pairwise_force_kernel!(fs_mat, d_fs_mat, virial, d_virial, coords, d_coords,
                                     atoms, d_atoms, boundary, inters, d_inters,
                                     neighbors, val_force_units)
    Enzyme.autodiff_deferred(
        pairwise_force_kernel!,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(virial, d_virial),
        Duplicated(coords, d_coords),
        Duplicated(atoms, d_atoms),
        Const(boundary),
        Duplicated(inters, d_inters),
        Const(neighbors),
        Const(val_force_units),
    )
    return nothing
end

function ChainRulesCore.rrule(::typeof(pairwise_force_gpu), virial,
                              coords::AbstractArray{SVector{D, C}}, atoms,
                              boundary, pairwise_inters, nbs, force_units,
                              val_ft::Val{T}) where {D, C, T}
    if force_units != NoUnits
        error("Taking gradients through force calculation is not compatible with units, " *
              "system force units are $force_units")
    end
    Y = pairwise_force_gpu(virial, coords, atoms, boundary, pairwise_inters,
                           nbs, force_units, val_ft)

    function pairwise_force_gpu_pullback(d_fs_mat)
        n_atoms = length(atoms)
        z = zero(T)
        fs_mat = CUDA.zeros(T, D, n_atoms)
        d_virial = zero(virial)
        d_coords = zero(coords)
        d_atoms = CuArray([Atom(charge=z, mass=z, σ=z, ϵ=z) for _ in 1:n_atoms])
        d_pairwise_inters = zero.(pairwise_inters)
        n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))

        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_pairwise_force_kernel!(fs_mat,
                d_fs_mat, virial, d_virial, coords, d_coords, atoms, d_atoms, boundary,
                pairwise_inters, d_pairwise_inters, nbs, Val(force_units))

        return NoTangent(), d_virial, d_coords, d_atoms, NoTangent(),
               d_pairwise_inters, NoTangent(), NoTangent(), NoTangent()
    end

    return Y, pairwise_force_gpu_pullback
end

function grad_pairwise_pe_kernel!(pe_vec::CuDeviceVector{T}, d_pe_vec, coords, d_coords, atoms,
                                  d_atoms, boundary, inters, d_inters, neighbors, val_energy_units,
                                  shared_mem_size::Val{M}) where {T, M}
    shared_pes = CuStaticSharedArray(T, M)
    d_shared_pes = CuStaticSharedArray(T, M)
    sync_threads()

    Enzyme.autodiff_deferred(
        pairwise_pe_kernel!,
        Duplicated(pe_vec, d_pe_vec),
        Duplicated(coords, d_coords),
        Duplicated(atoms, d_atoms),
        Const(boundary),
        Duplicated(inters, d_inters),
        Const(neighbors),
        Const(val_energy_units),
        Const(shared_mem_size),
        Duplicated(shared_pes, d_shared_pes),
    )
    return nothing
end

function ChainRulesCore.rrule(::typeof(pairwise_pe_gpu), coords::AbstractArray{SVector{D, C}},
                              atoms, boundary, pairwise_inters, nbs, energy_units,
                              val_ft::Val{T}) where {D, C, T}
    if energy_units != NoUnits
        error("Taking gradients through potential energy calculation is not compatible with " *
              "units, system energy units are $energy_units")
    end
    Y = pairwise_pe_gpu(coords, atoms, boundary, pairwise_inters, nbs, energy_units, val_ft)

    function pairwise_pe_gpu_pullback(d_pe_vec_arg)
        n_atoms = length(atoms)
        z = zero(T)
        pe_vec = CUDA.zeros(T, 1)
        d_pe_vec = CuArray([d_pe_vec_arg[1]])
        d_coords = zero(coords)
        d_atoms = CuArray([Atom(charge=z, mass=z, σ=z, ϵ=z) for _ in 1:n_atoms])
        d_pairwise_inters = zero.(pairwise_inters)
        n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))

        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_pairwise_pe_kernel!(pe_vec,
                d_pe_vec, coords, d_coords, atoms, d_atoms, boundary,
                pairwise_inters, d_pairwise_inters, nbs, Val(energy_units),
                Val(n_threads_gpu))

        return NoTangent(), d_coords, d_atoms, NoTangent(), d_pairwise_inters, NoTangent(),
               NoTangent(), NoTangent()
    end

    return Y, pairwise_pe_gpu_pullback
end

function grad_specific_force_1_atoms_kernel!(fs_mat::CuDeviceMatrix, d_fs_mat, coords, d_coords,
                                             boundary, is, inters, d_inters,
                                             val_force_units)
    Enzyme.autodiff_deferred(
        specific_force_1_atoms_kernel!,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Duplicated(inters, d_inters),
        Const(val_force_units),
    )
    return nothing
end

function grad_specific_force_2_atoms_kernel!(fs_mat::CuDeviceMatrix, d_fs_mat, coords, d_coords,
                                             boundary, is, js, inters, d_inters,
                                             val_force_units)
    Enzyme.autodiff_deferred(
        specific_force_2_atoms_kernel!,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Duplicated(inters, d_inters),
        Const(val_force_units),
    )
    return nothing
end

function grad_specific_force_3_atoms_kernel!(fs_mat::CuDeviceMatrix, d_fs_mat, coords, d_coords,
                                             boundary, is, js, ks, inters, d_inters,
                                             val_force_units)
    Enzyme.autodiff_deferred(
        specific_force_3_atoms_kernel!,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Const(ks),
        Duplicated(inters, d_inters),
        Const(val_force_units),
    )
    return nothing
end

function grad_specific_force_4_atoms_kernel!(fs_mat::CuDeviceMatrix, d_fs_mat, coords, d_coords,
                                             boundary, is, js, ks, ls, inters, d_inters,
                                             val_force_units)
    Enzyme.autodiff_deferred(
        specific_force_4_atoms_kernel!,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Const(ks),
        Const(ls),
        Duplicated(inters, d_inters),
        Const(val_force_units),
    )
    return nothing
end

function ChainRulesCore.rrule(::typeof(specific_force_gpu), inter_list,
                              coords::AbstractArray{SVector{D, C}}, boundary,
                              force_units, val_ft::Val{T}) where {D, C, T}
    if force_units != NoUnits
        error("Taking gradients through force calculation is not compatible with units, " *
              "system force units are $force_units")
    end
    Y = specific_force_gpu(inter_list, coords, boundary, force_units, val_ft)

    function specific_force_gpu_pullback(d_fs_mat)
        fs_mat = CUDA.zeros(T, D, length(coords))
        d_inter_list = zero(inter_list)
        d_coords = zero(coords)
        n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list.is))

        if inter_list isa InteractionList1Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_1_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is,
                    inter_list.inters, d_inter_list.inters, Val(force_units))
        elseif inter_list isa InteractionList2Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_2_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is, inter_list.js,
                    inter_list.inters, d_inter_list.inters, Val(force_units))
        elseif inter_list isa InteractionList3Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_3_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is, inter_list.js, inter_list.ks,
                    inter_list.inters, d_inter_list.inters, Val(force_units))
        elseif inter_list isa InteractionList4Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_4_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                    inter_list.inters, d_inter_list.inters, Val(force_units))
        end

        return NoTangent(), d_inter_list, d_coords, NoTangent(), NoTangent(), NoTangent()
    end

    return Y, specific_force_gpu_pullback
end

function grad_specific_pe_1_atoms_kernel!(energy::CuDeviceVector, d_energy, coords, d_coords,
                                          boundary, is, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        specific_pe_1_atoms_kernel!,
        Duplicated(energy, d_energy),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Duplicated(inters, d_inters),
        Const(val_energy_units),
    )
    return nothing
end

function grad_specific_pe_2_atoms_kernel!(energy::CuDeviceVector, d_energy, coords, d_coords,
                                          boundary, is, js, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        specific_pe_2_atoms_kernel!,
        Duplicated(energy, d_energy),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Duplicated(inters, d_inters),
        Const(val_energy_units),
    )
    return nothing
end

function grad_specific_pe_3_atoms_kernel!(energy::CuDeviceVector, d_energy, coords, d_coords,
                                          boundary, is, js, ks, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        specific_pe_3_atoms_kernel!,
        Duplicated(energy, d_energy),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Const(ks),
        Duplicated(inters, d_inters),
        Const(val_energy_units),
    )
    return nothing
end

function grad_specific_pe_4_atoms_kernel!(energy::CuDeviceVector, d_energy, coords, d_coords,
                                          boundary, is, js, ks, ls, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        specific_pe_4_atoms_kernel!,
        Duplicated(energy, d_energy),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Const(ks),
        Const(ls),
        Duplicated(inters, d_inters),
        Const(val_energy_units),
    )
    return nothing
end

function ChainRulesCore.rrule(::typeof(specific_pe_gpu), inter_list,
                              coords::AbstractArray{SVector{D, C}}, boundary,
                              energy_units, val_ft::Val{T}) where {D, C, T}
    if energy_units != NoUnits
        error("Taking gradients through potential energy calculation is not compatible with " *
              "units, system energy units are $energy_units")
    end
    Y = specific_pe_gpu(inter_list, coords, boundary, energy_units, val_ft)

    function specific_pe_gpu_pullback(d_pe_vec_arg)
        pe_vec = CUDA.zeros(T, 1)
        d_pe_vec = CuArray([d_pe_vec_arg[1]])
        d_inter_list = zero(inter_list)
        d_coords = zero(coords)
        n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list.is))

        if inter_list isa InteractionList1Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_pe_1_atoms_kernel!(
                    pe_vec, d_pe_vec, coords, d_coords, boundary,
                    inter_list.is,
                    inter_list.inters, d_inter_list.inters, Val(energy_units))
        elseif inter_list isa InteractionList2Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_pe_2_atoms_kernel!(
                    pe_vec, d_pe_vec, coords, d_coords, boundary,
                    inter_list.is, inter_list.js,
                    inter_list.inters, d_inter_list.inters, Val(energy_units))
        elseif inter_list isa InteractionList3Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_pe_3_atoms_kernel!(
                    pe_vec, d_pe_vec, coords, d_coords, boundary,
                    inter_list.is, inter_list.js, inter_list.ks,
                    inter_list.inters, d_inter_list.inters, Val(energy_units))
        elseif inter_list isa InteractionList4Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_pe_4_atoms_kernel!(
                    pe_vec, d_pe_vec, coords, d_coords, boundary,
                    inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                    inter_list.inters, d_inter_list.inters, Val(energy_units))
        end

        return NoTangent(), d_inter_list, d_coords, NoTangent(), NoTangent(), NoTangent()
    end

    return Y, specific_pe_gpu_pullback
end
