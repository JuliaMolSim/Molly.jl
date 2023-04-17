# Chain rules to allow differentiable simulations

@non_differentiable CUDA.zeros(args...)
@non_differentiable random_velocities(args...)
@non_differentiable random_velocities!(args...)
@non_differentiable cuda_threads_blocks_pairwise(args...)
@non_differentiable cuda_threads_blocks_specific(args...)
@non_differentiable check_force_units(args...)
@non_differentiable atoms_bonded_to_N(args...)
@non_differentiable lookup_table(args...)
@non_differentiable cuda_threads_blocks_gbsa(args...)
@non_differentiable find_neighbors(args...)
@non_differentiable DistanceNeighborFinder(args...)
@non_differentiable run_loggers!(args...)
@non_differentiable visualize(args...)
@non_differentiable place_atoms(args...)
@non_differentiable place_diatomics(args...)
@non_differentiable MolecularForceField(T::Type, ff_files::AbstractString...)
@non_differentiable MolecularForceField(ff_files::AbstractString...)
@non_differentiable System(coord_file::AbstractString, force_field::MolecularForceField)
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
        return NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList1Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:InteractionList2Atoms}, vs...)
    Y = T(vs...)
    function InteractionList2Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList2Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:InteractionList3Atoms}, vs...)
    Y = T(vs...)
    function InteractionList3Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters, NoTangent()
    end
    return Y, InteractionList3Atoms_pullback
end

function ChainRulesCore.rrule(T::Type{<:InteractionList4Atoms}, vs...)
    Y = T(vs...)
    function InteractionList4Atoms_pullback(Ȳ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), Ȳ.inters,
               NoTangent()
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
active_if_present(x) = length(x) > 0 ? Active(x) : Const(x)

nothing_to_notangent(x) = x
nothing_to_notangent(::Nothing) = NoTangent()

function ChainRulesCore.rrule(::typeof(forces_pair_spec), coords::AbstractArray{SVector{D, T}},
                              atoms, pairwise_inters_nonl, pairwise_inters_nl, sils_1_atoms,
                              sils_2_atoms, sils_3_atoms, sils_4_atoms, boundary, force_units,
                              neighbors, n_threads) where {D, T}
    if force_units != NoUnits
        error("taking gradients through force calculation is not compatible with units, " *
              "system force units are $force_units")
    end
    Y = forces_pair_spec(coords, atoms, pairwise_inters_nonl, pairwise_inters_nl, sils_1_atoms,
                         sils_2_atoms, sils_3_atoms, sils_4_atoms, boundary, force_units,
                         neighbors, n_threads)

    function forces_pair_spec_pullback(d_forces)
        fs = zero(coords)
        z = zero(T)
        d_coords = zero(coords)
        d_atoms = [Atom(charge=z, mass=z, σ=z, ϵ=z) for _ in 1:length(coords)]
        d_sils_1_atoms = zero.(sils_1_atoms)
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        grads = autodiff(
            Enzyme.Reverse,
            forces_pair_spec!,
            Const,
            Duplicated(fs, convert(typeof(fs), d_forces)),
            Duplicated(coords, d_coords),
            Duplicated(atoms, d_atoms),
            active_if_present(pairwise_inters_nonl),
            active_if_present(pairwise_inters_nl),
            duplicated_if_present(sils_1_atoms, d_sils_1_atoms),
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            Const(boundary),
            Const(force_units),
            Const(neighbors),
            Const(n_threads),
        )[1]
        d_pairwise_inters_nonl = nothing_to_notangent(grads[4])
        d_pairwise_inters_nl   = nothing_to_notangent(grads[5])
        return NoTangent(), d_coords, d_atoms, d_pairwise_inters_nonl, d_pairwise_inters_nl,
               d_sils_1_atoms, d_sils_2_atoms, d_sils_3_atoms, d_sils_4_atoms,
               NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return Y, forces_pair_spec_pullback
end

function ChainRulesCore.rrule(::typeof(potential_energy_pair_spec), coords, atoms,
                              pairwise_inters_nonl, pairwise_inters_nl, sils_1_atoms, sils_2_atoms,
                              sils_3_atoms, sils_4_atoms, boundary, energy_units, neighbors,
                              n_threads, val_ft::Val{T}) where T
    if energy_units != NoUnits
        error("taking gradients through potential energy calculation is not compatible with " *
              "units, system energy units are $energy_units")
    end
    Y = potential_energy_pair_spec(coords, atoms, pairwise_inters_nonl, pairwise_inters_nl,
                                   sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, boundary,
                                   energy_units, neighbors, n_threads, val_ft)

    function potential_energy_pair_spec_pullback(d_pe_vec)
        pe_vec = zeros(T, 1)
        z = zero(T)
        d_coords = zero(coords)
        d_atoms = [Atom(charge=z, mass=z, σ=z, ϵ=z) for _ in 1:length(coords)]
        d_sils_1_atoms = zero.(sils_1_atoms)
        d_sils_2_atoms = zero.(sils_2_atoms)
        d_sils_3_atoms = zero.(sils_3_atoms)
        d_sils_4_atoms = zero.(sils_4_atoms)
        grads = autodiff(
            Enzyme.Reverse,
            potential_energy_pair_spec!,
            Const,
            Duplicated(pe_vec, [d_pe_vec]),
            Duplicated(coords, d_coords),
            Duplicated(atoms, d_atoms),
            active_if_present(pairwise_inters_nonl),
            active_if_present(pairwise_inters_nl),
            duplicated_if_present(sils_1_atoms, d_sils_1_atoms),
            duplicated_if_present(sils_2_atoms, d_sils_2_atoms),
            duplicated_if_present(sils_3_atoms, d_sils_3_atoms),
            duplicated_if_present(sils_4_atoms, d_sils_4_atoms),
            Const(boundary),
            Const(energy_units),
            Const(neighbors),
            Const(n_threads),
            Const(val_ft),
        )[1]
        d_pairwise_inters_nonl = nothing_to_notangent(grads[4])
        d_pairwise_inters_nl   = nothing_to_notangent(grads[5])
        return NoTangent(), d_coords, d_atoms, d_pairwise_inters_nonl, d_pairwise_inters_nl,
               d_sils_1_atoms, d_sils_2_atoms, d_sils_3_atoms, d_sils_4_atoms,
               NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return Y, potential_energy_pair_spec_pullback
end

function grad_pairwise_force_kernel!(fs_mat, d_fs_mat, coords, d_coords, atoms, d_atoms,
                                     boundary, inters::I, grad_inters, neighbors, val_dims,
                                     val_force_units, ::Val{N}) where {I, N}
    shared_grad_inters = CuStaticSharedArray(I, N)
    sync_threads()

    grads = Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        pairwise_force_kernel!,
        Const,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Duplicated(atoms, d_atoms),
        Const(boundary),
        Active(inters),
        Const(neighbors),
        Const(val_dims),
        Const(val_force_units),
    )[1]

    tidx = threadIdx().x
    shared_grad_inters[tidx] = grads[5]
    sync_threads()

    if tidx == 1
        grad_inters_sum = shared_grad_inters[1]
        for ti in 2:N
            grad_inters_sum = map(+, grad_inters_sum, shared_grad_inters[ti])
        end
        grad_inters[blockIdx().x] = grad_inters_sum
    end
    return nothing
end

function ChainRulesCore.rrule(::typeof(pairwise_force_gpu), coords::AbstractArray{SVector{D, C}},
                              atoms, boundary, pairwise_inters, nbs, force_units,
                              val_ft::Val{T}) where {D, C, T}
    if force_units != NoUnits
        error("taking gradients through force calculation is not compatible with units, " *
              "system force units are $force_units")
    end
    Y = pairwise_force_gpu(coords, atoms, boundary, pairwise_inters, nbs, force_units, val_ft)

    function pairwise_force_gpu_pullback(d_fs_mat)
        n_atoms = length(atoms)
        z = zero(T)
        fs_mat = CUDA.zeros(T, D, n_atoms)
        d_coords = zero(coords)
        d_atoms = CuArray([Atom(charge=z, mass=z, σ=z, ϵ=z) for _ in 1:n_atoms])
        n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
        grad_pairwise_inters = CuArray(fill(pairwise_inters, n_blocks))

        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_pairwise_force_kernel!(fs_mat,
                d_fs_mat, coords, d_coords, atoms, d_atoms, boundary, pairwise_inters,
                grad_pairwise_inters, nbs, Val(D), Val(force_units), Val(n_threads_gpu))

        d_pairwise_inters = reduce((t1, t2) -> map(+, t1, t2), Array(grad_pairwise_inters))
        return NoTangent(), d_coords, d_atoms, NoTangent(), d_pairwise_inters, NoTangent(),
               NoTangent(), NoTangent()
    end

    return Y, pairwise_force_gpu_pullback
end

function grad_pairwise_pe_kernel!(pe_vec, d_pe_vec, coords, d_coords, atoms, d_atoms, boundary,
                                  inters::I, grad_inters, neighbors, val_energy_units,
                                  ::Val{N}) where {I, N}
    shared_grad_inters = CuStaticSharedArray(I, N)
    sync_threads()

    grads = Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        pairwise_pe_kernel!,
        Const,
        Duplicated(pe_vec, d_pe_vec),
        Duplicated(coords, d_coords),
        Duplicated(atoms, d_atoms),
        Const(boundary),
        Active(inters),
        Const(neighbors),
        Const(val_energy_units),
    )[1]

    tidx = threadIdx().x
    shared_grad_inters[tidx] = grads[5]
    sync_threads()

    if tidx == 1
        grad_inters_sum = shared_grad_inters[1]
        for ti in 2:N
            grad_inters_sum = map(+, grad_inters_sum, shared_grad_inters[ti])
        end
        grad_inters[blockIdx().x] = grad_inters_sum
    end
    return nothing
end

function ChainRulesCore.rrule(::typeof(pairwise_pe_gpu), coords::AbstractArray{SVector{D, C}},
                              atoms, boundary, pairwise_inters, nbs, energy_units,
                              val_ft::Val{T}) where {D, C, T}
    if energy_units != NoUnits
        error("taking gradients through potential energy calculation is not compatible with " *
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
        n_threads_gpu, n_blocks = cuda_threads_blocks_pairwise(length(nbs))
        grad_pairwise_inters = CuArray(fill(pairwise_inters, n_blocks))

        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_pairwise_pe_kernel!(pe_vec,
                d_pe_vec, coords, d_coords, atoms, d_atoms, boundary, pairwise_inters,
                grad_pairwise_inters, nbs, Val(energy_units), Val(n_threads_gpu))

        d_pairwise_inters = reduce((t1, t2) -> map(+, t1, t2), Array(grad_pairwise_inters))
        return NoTangent(), d_coords, d_atoms, NoTangent(), d_pairwise_inters, NoTangent(),
               NoTangent(), NoTangent()
    end

    return Y, pairwise_pe_gpu_pullback
end

function grad_specific_force_1_atoms_kernel!(fs_mat, d_fs_mat, coords, d_coords,
                                             boundary, is, inters, d_inters,
                                             val_dims, val_force_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_force_1_atoms_kernel!,
        Const,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Duplicated(inters, d_inters),
        Const(val_dims),
        Const(val_force_units),
    )
    return nothing
end

function grad_specific_force_2_atoms_kernel!(fs_mat, d_fs_mat, coords, d_coords,
                                             boundary, is, js, inters, d_inters,
                                             val_dims, val_force_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_force_2_atoms_kernel!,
        Const,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Duplicated(inters, d_inters),
        Const(val_dims),
        Const(val_force_units),
    )
    return nothing
end

function grad_specific_force_3_atoms_kernel!(fs_mat, d_fs_mat, coords, d_coords,
                                             boundary, is, js, ks, inters, d_inters,
                                             val_dims, val_force_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_force_3_atoms_kernel!,
        Const,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Const(ks),
        Duplicated(inters, d_inters),
        Const(val_dims),
        Const(val_force_units),
    )
    return nothing
end

function grad_specific_force_4_atoms_kernel!(fs_mat, d_fs_mat, coords, d_coords,
                                             boundary, is, js, ks, ls, inters, d_inters,
                                             val_dims, val_force_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_force_4_atoms_kernel!,
        Const,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Const(js),
        Const(ks),
        Const(ls),
        Duplicated(inters, d_inters),
        Const(val_dims),
        Const(val_force_units),
    )
    return nothing
end

function ChainRulesCore.rrule(::typeof(specific_force_gpu), inter_list,
                              coords::AbstractArray{SVector{D, C}}, boundary,
                              force_units, val_ft::Val{T}) where {D, C, T}
    if force_units != NoUnits
        error("taking gradients through force calculation is not compatible with units, " *
              "system force units are $force_units")
    end
    Y = specific_force_gpu(inter_list, coords, boundary, force_units, val_ft)

    function specific_force_gpu_pullback(d_fs_mat)
        fs_mat = CUDA.zeros(T, D, length(coords))
        d_inter_list = zero(inter_list)
        d_coords = zero(coords)
        n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))

        if inter_list isa InteractionList1Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_1_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is,
                    inter_list.inters, d_inter_list.inters, Val(D), Val(force_units))
        elseif inter_list isa InteractionList2Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_2_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is, inter_list.js,
                    inter_list.inters, d_inter_list.inters, Val(D), Val(force_units))
        elseif inter_list isa InteractionList3Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_3_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is, inter_list.js, inter_list.ks,
                    inter_list.inters, d_inter_list.inters, Val(D), Val(force_units))
        elseif inter_list isa InteractionList4Atoms
            CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_specific_force_4_atoms_kernel!(
                    fs_mat, d_fs_mat, coords, d_coords, boundary,
                    inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                    inter_list.inters, d_inter_list.inters, Val(D), Val(force_units))
        end

        return NoTangent(), d_inter_list, d_coords, NoTangent(), NoTangent(), NoTangent()
    end

    return Y, specific_force_gpu_pullback
end

function grad_specific_pe_1_atoms_kernel!(energy, d_energy, coords, d_coords,
                                          boundary, is, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_pe_1_atoms_kernel!,
        Const,
        Duplicated(energy, d_energy),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(is),
        Duplicated(inters, d_inters),
        Const(val_energy_units),
    )
    return nothing
end

function grad_specific_pe_2_atoms_kernel!(energy, d_energy, coords, d_coords,
                                          boundary, is, js, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_pe_2_atoms_kernel!,
        Const,
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

function grad_specific_pe_3_atoms_kernel!(energy, d_energy, coords, d_coords,
                                          boundary, is, js, ks, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_pe_3_atoms_kernel!,
        Const,
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

function grad_specific_pe_4_atoms_kernel!(energy, d_energy, coords, d_coords,
                                          boundary, is, js, ks, ls, inters, d_inters,
                                          val_energy_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        specific_pe_4_atoms_kernel!,
        Const,
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
        error("taking gradients through potential energy calculation is not compatible with " *
              "units, system energy units are $energy_units")
    end
    Y = specific_pe_gpu(inter_list, coords, boundary, energy_units, val_ft)

    function specific_pe_gpu_pullback(d_pe_vec_arg)
        pe_vec = CUDA.zeros(T, 1)
        d_pe_vec = CuArray([d_pe_vec_arg[1]])
        d_inter_list = zero(inter_list)
        d_coords = zero(coords)
        n_threads_gpu, n_blocks = cuda_threads_blocks_specific(length(inter_list))

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

function grad_gbsa_born_kernel!(Is, d_Is, I_grads, d_I_grads, coords, d_coords, offset_radii,
                                d_offset_radii, scaled_offset_radii, d_scaled_offset_radii,
                                dist_cutoff, offset, neck_scale::T, grad_neck_scale, neck_cut, d0s,
                                d_d0s, m0s, d_m0s, boundary, val_coord_units, ::Val{N}) where {T, N}
    shared_grad_neck_scale = CuStaticSharedArray(T, N)
    sync_threads()

    grads = Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        gbsa_born_kernel!,
        Const,
        Duplicated(Is, d_Is),
        Duplicated(I_grads, d_I_grads),
        Duplicated(coords, d_coords),
        Duplicated(offset_radii, d_offset_radii),
        Duplicated(scaled_offset_radii, d_scaled_offset_radii),
        Const(dist_cutoff),
        Const(offset),
        Active(neck_scale),
        Const(neck_cut),
        Duplicated(d0s, d_d0s),
        Duplicated(m0s, d_m0s),
        Const(boundary),
        Const(val_coord_units),
    )[1]

    tidx = threadIdx().x
    shared_grad_neck_scale[tidx] = grads[8]
    sync_threads()

    if tidx == 1
        grad_neck_scale_sum = shared_grad_neck_scale[1]
        for ti in 2:N
            grad_neck_scale_sum += shared_grad_neck_scale[ti]
        end
        if !iszero(grad_neck_scale_sum)
            Atomix.@atomic grad_neck_scale[1] += grad_neck_scale_sum
        end
    end
    return nothing
end

function ChainRulesCore.rrule(::typeof(gbsa_born_gpu), coords::AbstractArray{SVector{D, C}},
                              offset_radii, scaled_offset_radii, dist_cutoff, offset, neck_scale,
                              neck_cut, d0s, m0s, boundary, val_ft::Val{T}) where {D, C, T}
    if unit(C) != NoUnits
        error("taking gradients through force/energy calculation is not compatible with units, " *
              "coordinate units are $(unit(C))")
    end
    Y = gbsa_born_gpu(coords, offset_radii, scaled_offset_radii, dist_cutoff, offset, neck_scale,
                      neck_cut, d0s, m0s, boundary, val_ft)

    function gbsa_born_gpu_pullback(d_args)
        n_atoms = length(coords)
        d_Is      = d_args[1] == ZeroTangent() ? CUDA.zeros(T, n_atoms)          : d_args[1]
        d_I_grads = d_args[2] == ZeroTangent() ? CUDA.zeros(T, n_atoms, n_atoms) : d_args[2]
        Is = CUDA.zeros(T, n_atoms)
        I_grads = CUDA.zeros(T, n_atoms, n_atoms)
        d_coords = zero(coords)
        d_offset_radii = zero(offset_radii)
        d_scaled_offset_radii = zero(scaled_offset_radii)
        grad_neck_scale = CUDA.zeros(T, 1)
        d_d0s = zero(d0s)
        d_m0s = zero(m0s)
        n_inters = n_atoms ^ 2
        n_threads_gpu, n_blocks = cuda_threads_blocks_gbsa(n_inters)

        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_gbsa_born_kernel!(
                Is, d_Is, I_grads, d_I_grads, coords, d_coords, offset_radii,
                d_offset_radii, scaled_offset_radii, d_scaled_offset_radii,
                dist_cutoff, offset, neck_scale, grad_neck_scale, neck_cut, d0s,
                d_d0s, m0s, d_m0s, boundary, Val(C), Val(n_threads_gpu))

        d_neck_scale = Array(grad_neck_scale)[1]
        return NoTangent(), d_coords, d_offset_radii, d_scaled_offset_radii, NoTangent(),
               NoTangent(), d_neck_scale, NoTangent(), d_d0s, d_m0s, NoTangent(), NoTangent()
    end

    return Y, gbsa_born_gpu_pullback
end

function grad_gbsa_force_1_kernel!(fs_mat, d_fs_mat, born_forces_mod_ustrip,
                                   d_born_forces_mod_ustrip, coords, d_coords, boundary,
                                   dist_cutoff, factor_solute::T, grad_factor_solute,
                                   factor_solvent::T, grad_factor_solvent, kappa::T, grad_kappa,
                                   Bs, d_Bs, charges, d_charges, val_dims, val_force_units,
                                   ::Val{N}) where {T, N}
    shared_grad_factor_solute  = CuStaticSharedArray(T, N)
    shared_grad_factor_solvent = CuStaticSharedArray(T, N)
    shared_grad_kappa          = CuStaticSharedArray(T, N)
    sync_threads()

    grads = Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        gbsa_force_1_kernel!,
        Const,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(born_forces_mod_ustrip, d_born_forces_mod_ustrip),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(dist_cutoff),
        Active(factor_solute),
        Active(factor_solvent),
        Active(kappa),
        Duplicated(Bs, d_Bs),
        Duplicated(charges, d_charges),
        Const(val_dims),
        Const(val_force_units),
    )[1]

    tidx = threadIdx().x
    shared_grad_factor_solute[tidx]  = grads[6]
    shared_grad_factor_solvent[tidx] = grads[7]
    shared_grad_kappa[tidx]          = grads[8]
    sync_threads()

    if tidx == 1
        grad_factor_solute_sum = shared_grad_factor_solute[1]
        for ti in 2:N
            grad_factor_solute_sum += shared_grad_factor_solute[ti]
        end
        if !iszero(grad_factor_solute_sum)
            Atomix.@atomic grad_factor_solute[1] += grad_factor_solute_sum
        end
    elseif tidx == 2
        grad_factor_solvent_sum = shared_grad_factor_solvent[1]
        for ti in 2:N
            grad_factor_solvent_sum += shared_grad_factor_solvent[ti]
        end
        if !iszero(grad_factor_solvent_sum)
            Atomix.@atomic grad_factor_solvent[1] += grad_factor_solvent_sum
        end
    elseif tidx == 3
        grad_kappa_sum = shared_grad_kappa[1]
        for ti in 2:N
            grad_kappa_sum += shared_grad_kappa[ti]
        end
        if !iszero(grad_kappa_sum)
            Atomix.@atomic grad_kappa[1] += grad_kappa_sum
        end
    end
    return nothing
end

function ChainRulesCore.rrule(::typeof(gbsa_force_1_gpu), coords::AbstractArray{SVector{D, C}},
                              boundary, dist_cutoff, factor_solute, factor_solvent, kappa, Bs,
                              charges::AbstractArray{T}, force_units) where {D, C, T}
    if force_units != NoUnits
        error("taking gradients through force calculation is not compatible with units, " *
              "system force units are $force_units")
    end
    Y = gbsa_force_1_gpu(coords, boundary, dist_cutoff, factor_solute, factor_solvent, kappa,
                         Bs, charges, force_units)

    function gbsa_force_1_gpu_pullback(d_args)
        n_atoms = length(coords)
        d_fs_mat                 = d_args[1] == ZeroTangent() ? CUDA.zeros(T, D, n_atoms) : d_args[1]
        d_born_forces_mod_ustrip = d_args[2] == ZeroTangent() ? CUDA.zeros(T, n_atoms)    : d_args[2]
        fs_mat = CUDA.zeros(T, D, n_atoms)
        born_forces_mod_ustrip = CUDA.zeros(T, n_atoms)
        d_coords = zero(coords)
        grad_factor_solute  = CUDA.zeros(T, 1)
        grad_factor_solvent = CUDA.zeros(T, 1)
        grad_kappa          = CUDA.zeros(T, 1)
        d_Bs = zero(Bs)
        d_charges = zero(charges)
        n_inters = n_atoms_to_n_pairs(n_atoms) + n_atoms
        n_threads_gpu, n_blocks = cuda_threads_blocks_gbsa(n_inters)

        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_gbsa_force_1_kernel!(
            fs_mat, d_fs_mat, born_forces_mod_ustrip, d_born_forces_mod_ustrip, coords,
            d_coords, boundary, dist_cutoff, factor_solute, grad_factor_solute,
            factor_solvent, grad_factor_solvent, kappa, grad_kappa, Bs, d_Bs, charges,
            d_charges, Val(D), Val(force_units), Val(n_threads_gpu))

        d_factor_solute  = Array(grad_factor_solute )[1]
        d_factor_solvent = Array(grad_factor_solvent)[1]
        d_kappa          = Array(grad_kappa         )[1]
        return NoTangent(), d_coords, NoTangent(), NoTangent(), d_factor_solute,
               d_factor_solvent, d_kappa, d_Bs, d_charges, NoTangent()
    end

    return Y, gbsa_force_1_gpu_pullback
end

function grad_gbsa_force_2_kernel!(fs_mat, d_fs_mat, born_forces, d_born_forces, coords, d_coords,
                                   boundary, dist_cutoff, or, d_or, sor, d_sor, Bs, d_Bs, B_grads,
                                   d_B_grads, I_grads, d_I_grads, val_dims, val_force_units)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        gbsa_force_2_kernel!,
        Const,
        Duplicated(fs_mat, d_fs_mat),
        Duplicated(born_forces, d_born_forces),
        Duplicated(coords, d_coords),
        Const(boundary),
        Const(dist_cutoff),
        Duplicated(or, d_or),
        Duplicated(sor, d_sor),
        Duplicated(Bs, d_Bs),
        Duplicated(B_grads, d_B_grads),
        Duplicated(I_grads, d_I_grads),
        Const(val_dims),
        Const(val_force_units),
    )
    return nothing
end

function ChainRulesCore.rrule(::typeof(gbsa_force_2_gpu), coords::AbstractArray{SVector{D, C}},
                              boundary, dist_cutoff, Bs, B_grads, I_grads, born_forces, offset_radii,
                              scaled_offset_radii, force_units, val_ft::Val{T}) where {D, C, T}
    if force_units != NoUnits
        error("taking gradients through force calculation is not compatible with units, " *
              "system force units are $force_units")
    end
    Y = gbsa_force_2_gpu(coords, boundary, dist_cutoff, Bs, B_grads, I_grads, born_forces,
                         offset_radii, scaled_offset_radii, force_units, val_ft)

    function gbsa_force_2_gpu_pullback(d_fs_mat)
        n_atoms = length(coords)
        fs_mat = CUDA.zeros(T, D, n_atoms)
        d_coords = zero(coords)
        d_born_forces = zero(born_forces)
        d_offset_radii = zero(offset_radii)
        d_scaled_offset_radii = zero(scaled_offset_radii)
        d_Bs = zero(Bs)
        d_B_grads = zero(B_grads)
        d_I_grads = zero(I_grads)
        n_inters = n_atoms ^ 2
        n_threads_gpu, n_blocks = cuda_threads_blocks_gbsa(n_inters)

        CUDA.@sync @cuda threads=n_threads_gpu blocks=n_blocks grad_gbsa_force_2_kernel!(
            fs_mat, d_fs_mat, born_forces, d_born_forces, coords, d_coords, boundary,
            dist_cutoff, offset_radii, d_offset_radii, scaled_offset_radii,
            d_scaled_offset_radii, Bs, d_Bs, B_grads, d_B_grads, I_grads, d_I_grads,
            Val(D), Val(force_units))

        return NoTangent(), d_coords, NoTangent(), NoTangent(), d_Bs, d_B_grads, d_I_grads,
               d_born_forces, d_offset_radii, d_scaled_offset_radii, NoTangent(), NoTangent()
    end

    return Y, gbsa_force_2_gpu_pullback
end
