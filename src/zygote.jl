# Extend Zygote to work with static vectors on the GPU
# Here be dragons

using ForwardDiff: Chunk, Dual, partials, value

Zygote.unbroadcast(x::Tuple{Any}, x̄::Nothing) = nothing

Zygote.accum(x::AbstractArray{<:SizedVector}, ys::CuArray{<:SVector}...) = Zygote.accum.(convert(typeof(ys[1]), x), ys...)
Zygote.accum(x::CuArray{<:SVector}, ys::AbstractArray{<:SizedVector}...) = Zygote.accum.(x, convert.(typeof(x), ys)...)

function Zygote.accum(x::Atom{T, T, T, T}, y::Atom{T, T, T, T}) where T
    Atom{T, T, T, T}(0, x.charge + y.charge, x.mass + y.mass, x.σ + y.σ, x.ϵ + y.ϵ)
end

function Zygote.accum(x::NamedTuple{(:index, :charge, :mass, :σ, :ϵ), Tuple{Int64, T, T, T, T}}, y::Atom{T, T, T, T}) where T
    Atom{T, T, T, T}(0, x.charge + y.charge, x.mass + y.mass, x.σ + y.σ, x.ϵ + y.ϵ)
end

atomorempty(at::Atom, T) = at
atomorempty(at::Nothing, T) = Atom(0, zero(T), zero(T), zero(T), zero(T))

Zygote.z2d(dx::AbstractArray{Union{Nothing, Atom{T, T, T, T}}}, primal::AbstractArray{Atom{T, T, T, T}}) where {T} = atomorempty.(dx, T)

function Zygote.accum(x::LennardJones{S, C, W, F, E}, y::LennardJones{S, C, W, F, E}) where {S, C, W, F, E}
    LennardJones{S, C, W, F, E}(x.cutoff, x.nl_only, x.lorentz_mixing, x.weight_14 + y.weight_14, x.force_unit, x.energy_unit)
end

function Zygote.accum(x::NamedTuple{(:cutoff, :nl_only, :lorentz_mixing, :weight_14, :force_unit, :energy_unit), Tuple{C, Bool, Bool, W, F, E}}, y::LennardJones{S, C, W, F, E}) where {S, C, W, F, E}
    LennardJones{S, C, W, F, E}(x.cutoff, x.nl_only, x.lorentz_mixing, x.weight_14 + y.weight_14, x.force_unit, x.energy_unit)
end

function Zygote.accum_sum(xs::AbstractArray{LennardJones{S, C, W, F, E}}; dims=:) where {S, C, W, F, E}
    reduce(Zygote.accum, xs, dims=dims; init=LennardJones{S, C, W, F, E}(nothing, false, false, zero(W), NoUnits, NoUnits))
end

function Zygote.unbroadcast(x::AbstractArray{<:Real}, x̄::AbstractArray{<:StaticVector})
    N = ndims(x̄)
    if length(x) == length(x̄)
        Zygote._project(x, sum.(x̄))
    else
        dims = ntuple(d -> size(x, d) == 1 ? d : ndims(x̄)+1, ndims(x̄))
        Zygote._project(x, accum_sum(x̄; dims = dims))
    end
end

Zygote._zero(xs::AbstractArray{<:StaticVector}, T) = fill!(similar(xs, T), zero(T))

function Base.zero(::Type{Union{Nothing, SizedVector{D, T, Vector{T}}}}) where {D, T}
    zero(SizedVector{D, T, Vector{T}})
end

Base.:+(x::Real, y::SizedVector) = x .+ y
Base.:+(x::SizedVector, y::Real) = x .+ y

Base.:+(x::Real, y::Zygote.OneElement) = x .+ y
Base.:+(x::Zygote.OneElement, y::Real) = x .+ y

# See the dualize function in ForwardDiff
@generated function dualize_add1(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T,eltype(x),N+1})
    return quote
        chunk = Chunk{$N+1}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@generated function dualize_add3(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T,eltype(x),N+3})
    return quote
        chunk = Chunk{$N+3}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@generated function dualize_add3bef(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i+3}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T,eltype(x),N+3})
    return quote
        chunk = Chunk{$N+3}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@generated function dualize_fb1(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i+4}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T,eltype(x),N+18})
    return quote
        chunk = Chunk{$N+18}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@generated function dualize_fb2(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i+7}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T,eltype(x),N+18})
    return quote
        chunk = Chunk{$N+18}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@generated function dualize_fb3(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i+18}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T,eltype(x),N+18})
    return quote
        chunk = Chunk{$N+18}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

# Space for 4 duals given to interactions though only one used in this case
# No gradient for cutoff type
function dualize_fb(inter::LennardJones{S, C, W, F, E}) where {S, C, W, F, E}
    weight_14 = Dual(inter.weight_14, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false)
    return LennardJones{S, C, typeof(weight_14), F, E}(inter.cutoff, inter.nl_only, inter.lorentz_mixing,
                                                        weight_14, inter.force_unit, inter.energy_unit)
end

function dualize_fb(inter::CoulombReactionField{D, S, W, T, F, E, D2, K, R}) where {D, S, W, T, F, E, D2, K, R}
    cutoff_dist        = Dual(inter.cutoff_dist       , true , false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false)
    solvent_dielectric = Dual(inter.solvent_dielectric, false, true , false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false)
    weight_14          = Dual(inter.weight_14         , false, false, true , false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false)
    coulomb_const      = Dual(inter.coulomb_const     , false, false, false, true , false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false)
    sqdist_cutoff, krf, crf, krf_14, crf_14 = coulombproperties(cutoff_dist, solvent_dielectric)
    return CoulombReactionField{typeof(cutoff_dist), typeof(solvent_dielectric), typeof(weight_14), typeof(coulomb_const),
                                F, E, typeof(sqdist_cutoff), typeof(krf), typeof(crf)}(
                    cutoff_dist, solvent_dielectric, nl_only, weight_14, coulomb_const,
                    force_unit, energy_unit, sqdist_cutoff, krf, crf, krf_14, crf_14)
end

function dualize_atom_fb1(at::Atom)
    return Atom(at.index,
                Dual(at.charge, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false),
                Dual(at.mass  , false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false),
                Dual(at.σ     , false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false),
                Dual(at.ϵ     , false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false))
end

function dualize_atom_fb2(at::Atom)
    return Atom(at.index,
                Dual(at.charge, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false),
                Dual(at.mass  , false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false),
                Dual(at.σ     , false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false),
                Dual(at.ϵ     , false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false))
end

function dual_function_svec(f::F) where F
    function (arg1)
        ds1 = ForwardDiff.dualize(Nothing, arg1)
        return f(ds1)
    end
end

function dual_function_svec_real(f::F) where F
    function (arg1::SVector{D, T}, arg2) where {D, T}
        ds1 = dualize_add1(Nothing, arg1)
        # Leaving the integer type in here results in Float32 -> Float64 conversion
        ds2 = Zygote.dual(isa(arg2, Int) ? T(arg2) : arg2, (false, false, false, true))
        return f(ds1, ds2)
    end
end

function dual_function_svec_svec(f::F) where F
    function (arg1, arg2)
        ds1 = dualize_add3(Nothing, arg1)
        ds2 = dualize_add3bef(Nothing, arg2)
        return f(ds1, ds2)
    end
end

function dual_function_atom(f::F) where F
    function (arg1)
        ds1 = Atom(arg1.index,
                    Dual(arg1.charge, true, false, false, false),
                    Dual(arg1.mass  , false, true, false, false),
                    Dual(arg1.σ     , false, false, true, false),
                    Dual(arg1.ϵ     , false, false, false, true))
        return f(ds1)
    end
end

function dual_function_force_broadcast(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        ds1 = dualize_fb(arg1)
        ds2 = dualize_fb1(Nothing, arg2)
        ds3 = dualize_fb2(Nothing, arg3)
        ds4 = dualize_atom_fb1(arg4)
        ds5 = dualize_atom_fb2(arg5)
        ds6 = dualize_fb3(Nothing, arg6)
        ds7 = arg7
        ds8 = arg8
        return f(ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8)
    end
end

function matm(sv::Dual{Nothing, T}, y1::SVector{3, T}, i::Integer, j::Integer, k::Integer) where T
    partials(sv, i) * y1[1] + partials(sv, j) * y1[2] + partials(sv, k) * y1[3]
end

@inline function Zygote.broadcast_forward(f, arg1::AbstractArray{SVector{D, T}}) where {D, T}
    println("mo1")
    out = dual_function_svec(f).(arg1)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ)
        println("mo1backstart")
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials(o1))
            else
                SVector{D, T}(matm.(o1, (y1,), 1, 2, 3))
            end
        end
        darg1 = Zygote.unbroadcast(arg1, barg1)
        println("mo1backend")
        (nothing, nothing, darg1)
    end
    return y, bc_fwd_back
end

@inline function Zygote.broadcast_forward(f, arg1::AbstractArray{SVector{D, T}}, arg2) where {D, T}
    println("mo2")
    out = dual_function_svec_real(f).(arg1, arg2)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ)
        println("mo2backstart")
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials.((o1,), (1, 2, 3)))
            else
                SVector{D, T}(matm.(o1, (y1,), 1, 2, 3))
            end
        end
        darg1 = Zygote.unbroadcast(arg1, barg1)
        darg2 = Zygote.unbroadcast(arg2, broadcast((y1, o1) -> y1 .* partials.(o1, 4), ȳ, out))
        println("mo2backend")
        (nothing, nothing, darg1, darg2)
    end
    return y, bc_fwd_back
end

@inline function Zygote.broadcast_forward(f, arg1::AbstractArray{SVector{D, T}}, arg2::AbstractArray{SVector{D, T}}) where {D, T}
    println("mo3")
    out = dual_function_svec_svec(f).(arg1, arg2)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ)
        println("mo3backstart")
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials.((o1,), (1, 2, 3)))
            else
                SVector{D, T}(matm.(o1, (y1,), 1, 2, 3))
            end
        end
        darg1 = Zygote.unbroadcast(arg1, barg1)
        barg2 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials.((o1,), (4, 5, 6)))
            else
                SVector{D, T}(matm.(o1, (y1,), 4, 5, 6))
            end
        end
        darg2 = Zygote.unbroadcast(arg2, barg2)
        println("mo3backend")
        (nothing, nothing, darg1, darg2)
    end
    return y, bc_fwd_back
end

@inline function Zygote.broadcast_forward(f, arg1::AbstractArray{<:Atom})
    println("mo4")
    out = dual_function_atom(f).(arg1)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ)
        println("mo4backstart")
        barg1 = broadcast(ȳ, out) do y1, o1
            ps = partials(o1)
            Atom(0, y1 * ps[1], y1 * ps[2], y1 * ps[3], y1 * ps[4])
        end
        darg1 = Zygote.unbroadcast(arg1, barg1)
        println("mo4backend")
        (nothing, nothing, darg1)
    end
    return y, bc_fwd_back
end

function combine_dual_GeneralInteraction(y1::SVector{D, T}, o1::SVector{D, Dual{Nothing, T, P}}, i::Integer) where {D, T, P}
    ps1, ps2, ps3 = partials(o1[1]), partials(o1[2]), partials(o1[3])
    LennardJones{false, Nothing, T, typeof(NoUnits), typeof(NoUnits)}(
                    nothing, false, false, y1[1] * ps1[i] + y1[2] * ps2[i] + y1[3] * ps3[i],
                    NoUnits, NoUnits)
end

function combine_dual_Atom(y1::SVector{D, T}, o1::SVector{D, Dual{Nothing, T, P}}, i::Integer, j::Integer, k::Integer, l::Integer) where {D, T, P}
    ps1, ps2, ps3 = partials(o1[1]), partials(o1[2]), partials(o1[3])
    Atom(
        0,
        y1[1] * ps1[i] + y1[2] * ps2[i] + y1[3] * ps3[i],
        y1[1] * ps1[j] + y1[2] * ps2[j] + y1[3] * ps3[j],
        y1[1] * ps1[k] + y1[2] * ps2[k] + y1[3] * ps3[k],
        y1[1] * ps1[l] + y1[2] * ps2[l] + y1[3] * ps3[l],
    )
end

@inline function Zygote.broadcast_forward(f,
                                            arg1::Tuple{<:GeneralInteraction},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{SVector{D, T}},
                                            arg4::AbstractArray{<:Atom},
                                            arg5::AbstractArray{<:Atom},
                                            arg6::Tuple{SVector{D, T}},
                                            arg7::Base.RefValue{<:Unitful.FreeUnits},
                                            arg8) where {D, T}
    println("mo5")
    out = dual_function_force_broadcast(f).(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ)
        println("mo5backstart")
        darg1 = Zygote.unbroadcast(arg1, broadcast(combine_dual_GeneralInteraction, ȳ, out, 1))
        darg2 = Zygote.unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(matm.(o1, (y1,), 5, 6, 7 )), ȳ, out))
        darg3 = Zygote.unbroadcast(arg3, broadcast((y1, o1) -> SVector{D, T}(matm.(o1, (y1,), 8, 9, 10)), ȳ, out))
        darg4 = Zygote.unbroadcast(arg4, broadcast(combine_dual_Atom, ȳ, out, 11, 12, 13, 14))
        darg5 = Zygote.unbroadcast(arg5, broadcast(combine_dual_Atom, ȳ, out, 15, 16, 17, 18))
        darg6 = Zygote.unbroadcast(arg6, broadcast((y1, o1) -> SVector{D, T}(matm.(o1, (y1,), 19, 20, 21)), ȳ, out))
        darg7 = nothing
        darg8 = nothing
        println("mo5backend")
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6, darg7, darg8)
    end
    return y, bc_fwd_back
end
