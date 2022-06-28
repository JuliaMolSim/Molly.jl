# Chain rules to allow differentiable simulations

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

function ChainRulesCore.rrule(::typeof(accumulateadd), x)
    Y = accumulateadd(x)
    function accumulateadd_pullback(Ȳ)
        return NoTangent(), reverse(accumulate(+, reverse(Ȳ)))
    end
    return Y, accumulateadd_pullback
end

function ChainRulesCore.rrule(::typeof(unsafe_getindex), arr, inds)
    Y = unsafe_getindex(arr, inds)
    function unsafe_getindex_pullback(Ȳ)
        dx = Zygote._zero(arr, eltype(Ȳ))
        dxv = @view dx[inds]
        dxv .= Zygote.accum.(dxv, Zygote._droplike(Ȳ, dxv))
        return NoTangent(), Zygote._project(arr, dx), nothing
    end
    return Y, unsafe_getindex_pullback
end

# Not faster on CPU
function ChainRulesCore.rrule(::typeof(getindices_i), arr::CuArray, neighbors)
    Y = getindices_i(arr, neighbors)
    @views @inbounds function getindices_i_pullback(Ȳ)
        return NoTangent(), accumulate_bounds(Ȳ, neighbors.atom_bounds_i), nothing
    end
    return Y, getindices_i_pullback
end

function ChainRulesCore.rrule(::typeof(getindices_j), arr::CuArray, neighbors)
    Y = getindices_j(arr, neighbors)
    @views @inbounds function getindices_j_pullback(Ȳ)
        return NoTangent(), accumulate_bounds(Ȳ[neighbors.sortperm_j], neighbors.atom_bounds_j), nothing
    end
    return Y, getindices_j_pullback
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
        return NoTangent(), NoTangent(), SVector{D, T}.(eachcol(reshape(Ȳ, D, length(Ȳ) ÷ D)))
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

function ChainRulesCore.rrule(::typeof(ustrip), x::Number)
    Y = ustrip(x)
    function ustrip_pullback(Ȳ)
        return NoTangent(), Ȳ * unit(x)
    end
    return Y, ustrip_pullback
end
