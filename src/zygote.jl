# Extend Zygote to work with static vectors and custom types on the
#   fast broadcast/GPU path
# Here be dragons

using ForwardDiff: Chunk, Dual, partials, value
using ForwardDiff.ForwardDiffStaticArraysExt: dualize
using Zygote: unbroadcast

iszero_value(x::Dual) = iszero(value(x))
iszero_value(x) = iszero(x)

Zygote.accum(x::AbstractArray{<:SizedVector}, ys::AbstractArray{<:SVector}...) = Zygote.accum.(convert(typeof(ys[1]), x), ys...)
Zygote.accum(x::AbstractArray{<:SVector}, ys::AbstractArray{<:SizedVector}...) = Zygote.accum.(x, convert.(typeof(x), ys)...)

Zygote.accum(x::Vector{<:SVector} , y::CuArray{<:SVector}) = Zygote.accum(CuArray(x), y)
Zygote.accum(x::CuArray{<:SVector}, y::Vector{<:SVector} ) = Zygote.accum(x, CuArray(y))

Zygote.accum(x::SVector{D, T}, y::T) where {D, T} = x .+ y

Zygote.accum(u1::T, ::T) where {T <: Unitful.FreeUnits} = u1

Base.:+(x::Real, y::SizedVector) = x .+ y
Base.:+(x::SizedVector, y::Real) = x .+ y

Base.:+(x::Real, y::Zygote.OneElement) = x .+ y
Base.:+(x::Zygote.OneElement, y::Real) = x .+ y

function Zygote.accum(x::CuArray{Atom{T, T, T, T}},
                      y::Vector{NamedTuple{(:index, :charge, :mass, :σ, :ϵ, :solute)}}) where T
    CuArray(Zygote.accum(Array(x), y))
end

function Base.:+(x::Atom{T, T, T, T}, y::Atom{T, T, T, T}) where T
    Atom{T, T, T, T}(0, x.charge + y.charge, x.mass + y.mass, x.σ + y.σ, x.ϵ + y.ϵ, false)
end

function Base.:-(x::Atom{T, T, T, T}, y::Atom{T, T, T, T}) where T
    Atom{T, T, T, T}(0, x.charge - y.charge, x.mass - y.mass, x.σ - y.σ, x.ϵ - y.ϵ, false)
end

function Base.:+(x::Atom{T, T, T, T}, y::NamedTuple{(:index, :charge, :mass, :σ, :ϵ, :solute),
                    Tuple{Int, C, M, S, E, Bool}}) where {T, C, M, S, E}
    Atom{T, T, T, T}(
        0,
        Zygote.accum(x.charge, y.charge),
        Zygote.accum(x.mass, y.mass),
        Zygote.accum(x.σ, y.σ),
        Zygote.accum(x.ϵ, y.ϵ),
        false,
    )
end

function Base.:+(r::Base.RefValue{Any}, y::NamedTuple{(:atoms, :atoms_data, :masses,
                 :pairwise_inters, :specific_inter_lists, :general_inters, :constraints,
                 :coords, :velocities, :boundary, :neighbor_finder, :loggers, :force_units,
                 :energy_units, :k)})
    x = r.x
    (
        atoms=Zygote.accum(x.atoms, y.atoms),
        atoms_data=Zygote.accum(x.atoms_data, y.atoms_data),
        masses=Zygote.accum(x.masses, y.masses),
        pairwise_inters=Zygote.accum(x.pairwise_inters, y.pairwise_inters),
        specific_inter_lists=Zygote.accum(x.specific_inter_lists, y.specific_inter_lists),
        general_inters=Zygote.accum(x.general_inters, y.general_inters),
        constraints=Zygote.accum(x.constraints, y.constraints),
        coords=Zygote.accum(x.coords, y.coords),
        velocities=Zygote.accum(x.velocities, y.velocities),
        boundary=nothing,
        neighbor_finder=nothing,
        loggers=nothing,
        force_units=nothing,
        energy_units=nothing,
        k=Zygote.accum(x.k, y.k),
    )
end

function Base.:+(y::NamedTuple{(:atoms, :atoms_data, :masses,
                 :pairwise_inters, :specific_inter_lists, :general_inters, :constraints,
                 :coords, :velocities, :boundary, :neighbor_finder, :loggers, :force_units,
                 :energy_units, :k)}, r::Base.RefValue{Any})
    return r + y
end

function Zygote.accum(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{3, T, Vector{T}}}}, y::SVector{3, T}) where T
    CubicBoundary(SVector{3, T}(x.side_lengths .+ y))
end

function Zygote.accum(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{2, T, Vector{T}}}}, y::SVector{2, T}) where T
    RectangularBoundary(SVector{2, T}(x.side_lengths .+ y))
end

function Base.:+(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{3, T, Vector{T}}}}, y::CubicBoundary{T}) where T
    CubicBoundary(SVector{3, T}(x.side_lengths .+ y.side_lengths))
end

function Base.:+(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{2, T, Vector{T}}}}, y::RectangularBoundary{T}) where T
    RectangularBoundary(SVector{2, T}(x.side_lengths .+ y.side_lengths))
end

atom_or_empty(at::Atom, T) = at
atom_or_empty(at::Nothing, T) = zero(Atom{T, T, T, T})

Zygote.z2d(dx::AbstractArray{Union{Nothing, Atom{T, T, T, T}}}, primal::AbstractArray{Atom{T, T, T, T}}) where {T} = atom_or_empty.(dx, T)
Zygote.z2d(dx::SVector{3, T}, primal::T) where {T} = sum(dx)

function Zygote.unbroadcast(x::AbstractArray{<:Real}, x̄::AbstractArray{<:StaticVector})
    if length(x) == length(x̄)
        Zygote._project(x, sum.(x̄))
    else
        dims = ntuple(d -> size(x, d) == 1 ? d : ndims(x̄) + 1, ndims(x̄))
        Zygote._project(x, accum_sum(x̄; dims=dims))
    end
end

Zygote._zero(xs::AbstractArray{<:StaticVector}, T) = fill!(similar(xs, T), zero(T))

function Zygote._zero(xs::AbstractArray{Atom{T, T, T, T}}, ::Type{Atom{T, T, T, T}}) where T
    fill!(similar(xs), Atom{T, T, T, T}(0, zero(T), zero(T), zero(T), zero(T), false))
end

function Base.zero(::Type{Union{Nothing, SizedVector{D, T, Vector{T}}}}) where {D, T}
    zero(SizedVector{D, T, Vector{T}})
end

# Slower version than in Zygote but doesn't give wrong gradients on the GPU for repeated indices
# Here we just move it to the CPU then move it back
# See https://github.com/FluxML/Zygote.jl/pull/1131
Zygote.∇getindex(x::CuArray, inds::Tuple{AbstractArray{<:Integer}}) = dy -> begin
    inds1_cpu = Array(inds[1])
    dx = zeros(eltype(dy), length(x))
    dxv = view(dx, inds1_cpu)
    dxv .= Zygote.accum.(dxv, Zygote._droplike(Array(dy), dxv))
    return Zygote._project(x, CuArray(dx)), nothing
end

# Extend to add extra empty partials before (B) and after (A) the SVector partials
@generated function ForwardDiff.ForwardDiffStaticArraysExt.dualize(::Type{T}, x::StaticArray, ::Val{B}, ::Val{A}) where {T, B, A}
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i + $B}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T, eltype(x), N + B + A})
    return quote
        chunk = Chunk{$N + $B + $A}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@inline function sum_partials(sv::SVector{3, Dual{Nothing, T, P}}, y1, i::Integer) where {T, P}
    partials(sv[1], i) * y1[1] + partials(sv[2], i) * y1[2] + partials(sv[3], i) * y1[3]
end

sized_to_static(v::SizedVector{3, T, Vector{T}}) where {T} = SVector{3, T}(v[1], v[2], v[3])
sized_to_static(v::SizedVector{2, T, Vector{T}}) where {T} = SVector{2, T}(v[1], v[2])

function modify_grad(ȳ_in::AbstractArray{SizedVector{D, T, Vector{T}}}, arg::CuArray) where {D, T}
    CuArray(sized_to_static.(ȳ_in))
end

function modify_grad(ȳ_in::AbstractArray{SizedVector{D, T, Vector{T}}}, arg) where {D, T}
    sized_to_static.(ȳ_in)
end

modify_grad(ȳ_in, arg::CuArray) = CuArray(ȳ_in)
modify_grad(ȳ_in, arg) = ȳ_in

# Dualize a value with extra partials
macro dualize(x, n_partials::Integer, active_partial::Integer)
    ps = [i == active_partial for i in 1:n_partials]
    return :(ForwardDiff.Dual($(esc(x)), $(ps...)))
end

function dual_function_svec(f::F) where F
    function (arg1)
        ds1 = dualize(Nothing, arg1, Val(0), Val(0))
        return f(ds1)
    end
end

@inline function Zygote.broadcast_forward(f, arg1::AbstractArray{SVector{D, T}}) where {D, T}
    out = dual_function_svec(f).(arg1)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials(o1))
            else
                SVector{D, T}(sum_partials(o1, y1, 1), sum_partials(o1, y1, 2), sum_partials(o1, y1, 3))
            end
        end
        darg1 = unbroadcast(arg1, barg1)
        (nothing, nothing, darg1)
    end
    return y, bc_fwd_back
end

function dual_function_svec_real(f::F) where F
    function (arg1::SVector{D, T}, arg2) where {D, T}
        ds1 = dualize(Nothing, arg1, Val(0), Val(1))
        # Leaving the integer type in here results in Float32 -> Float64 conversion
        ds2 = Zygote.dual(arg2 isa Int ? T(arg2) : arg2, 4, Val(4))
        return f(ds1, ds2)
    end
end

@inline function Zygote.broadcast_forward(f, arg1::AbstractArray{SVector{D, T}}, arg2) where {D, T}
    out = dual_function_svec_real(f).(arg1, arg2)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials.((o1,), (1, 2, 3)))
            else
                SVector{D, T}(sum_partials(o1, y1, 1), sum_partials(o1, y1, 2), sum_partials(o1, y1, 3))
            end
        end
        darg1 = unbroadcast(arg1, barg1)
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> y1 .* partials.(o1, 4), ȳ, out))
        (nothing, nothing, darg1, darg2)
    end
    return y, bc_fwd_back
end

function dual_function_real_svec(f::F) where F
    function (arg1, arg2::SVector{D, T}) where {D, T}
        ds1 = Zygote.dual(arg1 isa Int ? T(arg1) : arg1, 1, Val(4))
        ds2 = dualize(Nothing, arg2, Val(1), Val(0))
        return f(ds1, ds2)
    end
end

@inline function Zygote.broadcast_forward(f,
                                            arg1::Union{AbstractArray{R}, Tuple{R}, R},
                                            arg2::AbstractArray{SVector{D, T}}) where {D, T, R <: Real}
    out = dual_function_real_svec(f).(arg1, arg2)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg2)
        darg1 = unbroadcast(arg1, broadcast((y1, o1) -> y1 .* partials.(o1, 1), ȳ, out))
        barg2 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials.((o1,), (2, 3, 4)))
            else
                SVector{D, T}(sum_partials(o1, y1, 2), sum_partials(o1, y1, 3), sum_partials(o1, y1, 4))
            end
        end
        darg2 = unbroadcast(arg2, barg2)
        (nothing, nothing, darg1, darg2)
    end
    return y, bc_fwd_back
end

function dual_function_svec_svec(f::F) where F
    function (arg1, arg2)
        ds1 = dualize(Nothing, arg1, Val(0), Val(3))
        ds2 = dualize(Nothing, arg2, Val(3), Val(0))
        return f(ds1, ds2)
    end
end

@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{SVector{D, T}},
                                            arg2::Union{AbstractArray{SVector{D, T}}, Tuple{SVector{D, T}}}) where {D, T}
    out = dual_function_svec_svec(f).(arg1, arg2)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials.((o1,), (1, 2, 3)))
            else
                SVector{D, T}(sum_partials(o1, y1, 1), sum_partials(o1, y1, 2), sum_partials(o1, y1, 3))
            end
        end
        darg1 = unbroadcast(arg1, barg1)
        barg2 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(partials.((o1,), (4, 5, 6)))
            else
                SVector{D, T}(sum_partials(o1, y1, 4), sum_partials(o1, y1, 5), sum_partials(o1, y1, 6))
            end
        end
        darg2 = unbroadcast(arg2, barg2)
        (nothing, nothing, darg1, darg2)
    end
    return y, bc_fwd_back
end

function dual_function_atom(f::F) where F
    function (arg1)
        c, m, σ, ϵ = arg1.charge, arg1.mass, arg1.σ, arg1.ϵ
        ds1 = Atom(
            arg1.index,
            @dualize(c, 4, 1),
            @dualize(m, 4, 2),
            @dualize(σ, 4, 3),
            @dualize(ϵ, 4, 4),
            arg1.solute,
        )
        return f(ds1)
    end
end

# For mass, charge etc.
@inline function Zygote.broadcast_forward(f, arg1::AbstractArray{<:Atom})
    out = dual_function_atom(f).(arg1)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        barg1 = broadcast(ȳ, out) do y1, o1
            ps = partials(o1)
            Atom(0, y1 * ps[1], y1 * ps[2], y1 * ps[3], y1 * ps[4], false)
        end
        darg1 = unbroadcast(arg1, barg1)
        (nothing, nothing, darg1)
    end
    return y, bc_fwd_back
end

function dual_function_born_radii_loop_OBC(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6)
        ds1 = dualize(Nothing, arg1, Val(0), Val(5))
        ds2 = dualize(Nothing, arg2, Val(3), Val(2))
        ds3 = Zygote.dual(arg3, 7, Val(8))
        ds4 = Zygote.dual(arg4, 8, Val(8))
        ds5 = arg5
        ds6 = arg6
        return f(ds1, ds2, ds3, ds4, ds5, ds6)
    end
end

# For born_radii_loop_OBC
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{SVector{D, T}},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{T},
                                            arg4::AbstractArray{T},
                                            arg5::T,
                                            arg6) where {D, T}
    out = dual_function_born_radii_loop_OBC(f).(arg1, arg2, arg3, arg4, arg5, arg6)
    y = value.(out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast((y1, o1) -> SVector{D, T}(partials(o1, 1) * y1,
                    partials(o1, 2) * y1, partials(o1, 3) * y1), ȳ, out))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(partials(o1, 4) * y1,
                    partials(o1, 5) * y1, partials(o1, 6) * y1), ȳ, out))
        darg3 = unbroadcast(arg3, broadcast((y1, o1) -> partials(o1, 7) * y1, ȳ, out))
        darg4 = unbroadcast(arg4, broadcast((y1, o1) -> partials(o1, 8) * y1, ȳ, out))
        darg5 = nothing
        darg6 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6)
    end
    return y, bc_fwd_back
end

# For born_radii_sum
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{T},
                                            arg2::T,
                                            arg3::AbstractArray{T},
                                            arg4,
                                            arg5,
                                            arg6) where T
    out = Zygote.dual_function(f).(arg1, arg2, arg3, arg4, arg5, arg6)
    y = broadcast(o1 -> (value(o1[1]), value(o1[2])), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast((y1, o1) -> partials(o1[1], 1) * y1[1] + partials(o1[2], 1) * y1[2], ȳ, out))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> partials(o1[1], 2) * y1[1] + partials(o1[2], 2) * y1[2], ȳ, out))
        darg3 = unbroadcast(arg3, broadcast((y1, o1) -> partials(o1[1], 3) * y1[1] + partials(o1[2], 3) * y1[2], ȳ, out))
        darg4 = unbroadcast(arg4, broadcast((y1, o1) -> partials(o1[1], 4) * y1[1] + partials(o1[2], 4) * y1[2], ȳ, out))
        darg5 = unbroadcast(arg5, broadcast((y1, o1) -> partials(o1[1], 5) * y1[1] + partials(o1[2], 5) * y1[2], ȳ, out))
        darg6 = unbroadcast(arg6, broadcast((y1, o1) -> partials(o1[1], 6) * y1[1] + partials(o1[2], 6) * y1[2], ȳ, out))
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6)
    end
    return y, bc_fwd_back
end

function dual_function_born_radii_loop_GBN2(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12)
        ds1  = dualize(Nothing, arg1, Val(0), Val(11))
        ds2  = dualize(Nothing, arg2, Val(3), Val(8))
        ds3  = Zygote.dual(arg3,  7 , Val(14))
        ds4  = Zygote.dual(arg4,  8 , Val(14))
        ds5  = Zygote.dual(arg5,  9 , Val(14))
        ds6  = arg6
        ds7  = Zygote.dual(arg7,  10, Val(14))
        ds8  = Zygote.dual(arg8,  11, Val(14))
        ds9  = Zygote.dual(arg9,  12, Val(14))
        ds10 = Zygote.dual(arg10, 13, Val(14))
        ds11 = Zygote.dual(arg11, 14, Val(14))
        ds12 = arg12
        return f(ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10, ds11, ds12)
    end
end

# For born_radii_loop_GBN2
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{SVector{D, T}},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{T},
                                            arg4::AbstractArray{T},
                                            arg5::AbstractArray{T},
                                            arg6::T,
                                            arg7::T,
                                            arg8::T,
                                            arg9::T,
                                            arg10::AbstractArray{T},
                                            arg11::AbstractArray{T},
                                            arg12) where {D, T}
    out = dual_function_born_radii_loop_GBN2(f).(arg1, arg2, arg3, arg4, arg5, arg6,
                                                    arg7, arg8, arg9, arg10, arg11, arg12)
    y = broadcast(o1 -> BornRadiiGBN2LoopResult{T, T}(value(o1.I), value(o1.I_grad)), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1  = unbroadcast(arg1, broadcast((y1, o1) -> SVector{D, T}(
                    partials(o1.I, 1) * y1.I + partials(o1.I_grad, 1) * y1.I_grad,
                    partials(o1.I, 2) * y1.I + partials(o1.I_grad, 2) * y1.I_grad,
                    partials(o1.I, 3) * y1.I + partials(o1.I_grad, 3) * y1.I_grad),
                    ȳ, out))
        darg2  = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(
                    partials(o1.I, 4) * y1.I + partials(o1.I_grad, 4) * y1.I_grad,
                    partials(o1.I, 5) * y1.I + partials(o1.I_grad, 5) * y1.I_grad,
                    partials(o1.I, 6) * y1.I + partials(o1.I_grad, 6) * y1.I_grad),
                    ȳ, out))
        darg3  = unbroadcast(arg3,  broadcast((y1, o1) -> partials(o1.I,  7) * y1.I + partials(o1.I_grad,  7) * y1.I_grad, ȳ, out))
        darg4  = unbroadcast(arg4,  broadcast((y1, o1) -> partials(o1.I,  8) * y1.I + partials(o1.I_grad,  8) * y1.I_grad, ȳ, out))
        darg5  = unbroadcast(arg5,  broadcast((y1, o1) -> partials(o1.I,  9) * y1.I + partials(o1.I_grad,  9) * y1.I_grad, ȳ, out))
        darg6  = nothing
        darg7  = unbroadcast(arg7,  broadcast((y1, o1) -> partials(o1.I, 10) * y1.I + partials(o1.I_grad, 10) * y1.I_grad, ȳ, out))
        darg8  = unbroadcast(arg8,  broadcast((y1, o1) -> partials(o1.I, 11) * y1.I + partials(o1.I_grad, 11) * y1.I_grad, ȳ, out))
        darg9  = unbroadcast(arg9,  broadcast((y1, o1) -> partials(o1.I, 12) * y1.I + partials(o1.I_grad, 12) * y1.I_grad, ȳ, out))
        darg10 = unbroadcast(arg10, broadcast((y1, o1) -> partials(o1.I, 13) * y1.I + partials(o1.I_grad, 13) * y1.I_grad, ȳ, out))
        darg11 = unbroadcast(arg11, broadcast((y1, o1) -> partials(o1.I, 14) * y1.I + partials(o1.I_grad, 14) * y1.I_grad, ȳ, out))
        darg12 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6,
                darg7, darg8, darg9, darg10, darg11, darg12)
    end
    return y, bc_fwd_back
end

function dual_function_gb_force_loop_1(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13)
        ds1  = dualize(Nothing, arg1, Val(0), Val(10))
        ds2  = dualize(Nothing, arg2, Val(3), Val(7))
        ds3  = arg3
        ds4  = arg4
        ds5  = Zygote.dual(arg5 , 7 , Val(13))
        ds6  = Zygote.dual(arg6 , 8 , Val(13))
        ds7  = Zygote.dual(arg7 , 9 , Val(13))
        ds8  = Zygote.dual(arg8 , 10, Val(13))
        ds9  = arg9
        ds10 = Zygote.dual(arg10, 11, Val(13))
        ds11 = Zygote.dual(arg11, 12, Val(13))
        ds12 = Zygote.dual(arg12, 13, Val(13))
        ds13 = arg13
        return f(ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10, ds11, ds12, ds13)
    end
end

# For gb_force_loop_1
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{SVector{D, T}},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{Int},
                                            arg4::AbstractArray{Int},
                                            arg5::AbstractArray{T},
                                            arg6::AbstractArray{T},
                                            arg7::AbstractArray{T},
                                            arg8::AbstractArray{T},
                                            arg9::T,
                                            arg10::T,
                                            arg11::T,
                                            arg12::T,
                                            arg13) where {D, T}
    out = dual_function_gb_force_loop_1(f).(arg1, arg2, arg3, arg4, arg5, arg6, arg7,
                                            arg8, arg9, arg10, arg11, arg12, arg13)
    y = broadcast(o1 -> ForceLoopResult1{T, SVector{D, T}}(value(o1.bi), value(o1.bj),
                                            value.(o1.fi), value.(o1.fj)), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.fi, y1.fi, 1) + sum_partials(o1.fj, y1.fj, 1) + partials(o1.bi, 1) * y1.bi + partials(o1.bj, 1) * y1.bj,
                    sum_partials(o1.fi, y1.fi, 2) + sum_partials(o1.fj, y1.fj, 2) + partials(o1.bi, 2) * y1.bi + partials(o1.bj, 2) * y1.bj,
                    sum_partials(o1.fi, y1.fi, 3) + sum_partials(o1.fj, y1.fj, 3) + partials(o1.bi, 3) * y1.bi + partials(o1.bj, 3) * y1.bj),
                    ȳ, out))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.fi, y1.fi, 4) + sum_partials(o1.fj, y1.fj, 4) + partials(o1.bi, 4) * y1.bi + partials(o1.bj, 4) * y1.bj,
                    sum_partials(o1.fi, y1.fi, 5) + sum_partials(o1.fj, y1.fj, 5) + partials(o1.bi, 5) * y1.bi + partials(o1.bj, 5) * y1.bj,
                    sum_partials(o1.fi, y1.fi, 6) + sum_partials(o1.fj, y1.fj, 6) + partials(o1.bi, 6) * y1.bi + partials(o1.bj, 6) * y1.bj),
                    ȳ, out))
        darg3 = nothing
        darg4 = nothing
        darg5 = unbroadcast(arg5, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi,  7) + sum_partials(o1.fj, y1.fj,  7) + partials(o1.bi,  7) * y1.bi + partials(o1.bj,  7) * y1.bj, ȳ, out))
        darg6 = unbroadcast(arg6, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi,  8) + sum_partials(o1.fj, y1.fj,  8) + partials(o1.bi,  8) * y1.bi + partials(o1.bj,  8) * y1.bj, ȳ, out))
        darg7 = unbroadcast(arg7, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi,  9) + sum_partials(o1.fj, y1.fj,  9) + partials(o1.bi,  9) * y1.bi + partials(o1.bj,  9) * y1.bj, ȳ, out))
        darg8 = unbroadcast(arg8, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi, 10) + sum_partials(o1.fj, y1.fj, 10) + partials(o1.bi, 10) * y1.bi + partials(o1.bj, 10) * y1.bj, ȳ, out))
        darg9 = nothing
        darg10 = unbroadcast(arg10, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi, 11) + sum_partials(o1.fj, y1.fj, 11) + partials(o1.bi, 11) * y1.bi + partials(o1.bj, 11) * y1.bj, ȳ, out))
        darg11 = unbroadcast(arg11, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi, 12) + sum_partials(o1.fj, y1.fj, 12) + partials(o1.bi, 12) * y1.bi + partials(o1.bj, 12) * y1.bj, ȳ, out))
        darg12 = unbroadcast(arg12, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi, 13) + sum_partials(o1.fj, y1.fj, 13) + partials(o1.bi, 13) * y1.bi + partials(o1.bj, 13) * y1.bj, ȳ, out))
        darg13 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6, darg7,
                darg8, darg9, darg10, darg11, darg12, darg13)
    end
    return y, bc_fwd_back
end

function dual_function_gb_force_loop_2(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        ds1 = dualize(Nothing, arg1, Val(0), Val(7))
        ds2 = dualize(Nothing, arg2, Val(3), Val(4))
        ds3 = Zygote.dual(arg3, 7 , Val(10))
        ds4 = Zygote.dual(arg4, 8 , Val(10))
        ds5 = Zygote.dual(arg5, 9 , Val(10))
        ds6 = Zygote.dual(arg6, 10, Val(10))
        ds7 = arg7
        ds8 = arg8
        return f(ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8)
    end
end

# For gb_force_loop_2
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{SVector{D, T}},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{T},
                                            arg4::AbstractArray{T},
                                            arg5::AbstractArray{T},
                                            arg6::AbstractArray{T},
                                            arg7::T,
                                            arg8) where {D, T}
    out = dual_function_gb_force_loop_2(f).(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    y = broadcast(o1 -> ForceLoopResult2{SVector{D, T}}(value.(o1.fi), value.(o1.fj)), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.fi, y1.fi, 1) + sum_partials(o1.fj, y1.fj, 1),
                    sum_partials(o1.fi, y1.fi, 2) + sum_partials(o1.fj, y1.fj, 2),
                    sum_partials(o1.fi, y1.fi, 3) + sum_partials(o1.fj, y1.fj, 3)),
                    ȳ, out))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.fi, y1.fi, 4) + sum_partials(o1.fj, y1.fj, 4),
                    sum_partials(o1.fi, y1.fi, 5) + sum_partials(o1.fj, y1.fj, 5),
                    sum_partials(o1.fi, y1.fi, 6) + sum_partials(o1.fj, y1.fj, 6)),
                    ȳ, out))
        darg3 = unbroadcast(arg3, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi,  7) + sum_partials(o1.fj, y1.fj,  7), ȳ, out))
        darg4 = unbroadcast(arg4, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi,  8) + sum_partials(o1.fj, y1.fj,  8), ȳ, out))
        darg5 = unbroadcast(arg5, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi,  9) + sum_partials(o1.fj, y1.fj,  9), ȳ, out))
        darg6 = unbroadcast(arg6, broadcast((y1, o1) -> sum_partials(o1.fi, y1.fi, 10) + sum_partials(o1.fj, y1.fj, 10), ȳ, out))
        darg7 = nothing
        darg8 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6, darg7, darg8)
    end
    return y, bc_fwd_back
end

function dual_function_gb_energy_loop(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
                arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18)
        ds1  = dualize(Nothing, arg1, Val(0), Val(14))
        ds2  = dualize(Nothing, arg2, Val(3), Val(11))
        ds3  = arg3
        ds4  = arg4
        # Using Zygote.dual errors on GPU so Dual is called explicitly
        ds5  = Dual(arg5 , (false, false, false, false, false, false, true , false, false, false, false, false, false, false, false, false, false))
        ds6  = Dual(arg6 , (false, false, false, false, false, false, false, true , false, false, false, false, false, false, false, false, false))
        ds7  = Dual(arg7 , (false, false, false, false, false, false, false, false, true , false, false, false, false, false, false, false, false))
        ds8  = Dual(arg8 , (false, false, false, false, false, false, false, false, false, true , false, false, false, false, false, false, false))
        ds9  = Dual(arg9 , (false, false, false, false, false, false, false, false, false, false, true , false, false, false, false, false, false))
        ds10 = arg10
        ds11 = Dual(arg11, (false, false, false, false, false, false, false, false, false, false, false, true , false, false, false, false, false))
        ds12 = Dual(arg12, (false, false, false, false, false, false, false, false, false, false, false, false, true , false, false, false, false))
        ds13 = Dual(arg13, (false, false, false, false, false, false, false, false, false, false, false, false, false, true , false, false, false))
        ds14 = Dual(arg14, (false, false, false, false, false, false, false, false, false, false, false, false, false, false, true , false, false))
        ds15 = Dual(arg15, (false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true , false))
        ds16 = Dual(arg16, (false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true ))
        ds17 = arg17
        ds18 = arg18
        return f(ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10,
                    ds11, ds12, ds13, ds14, ds15, ds16, ds17, ds18)
    end
end

# For gb_energy_loop
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{SVector{D, T}},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{Int},
                                            arg4::AbstractArray{Int},
                                            arg5::AbstractArray{T},
                                            arg6::AbstractArray{T},
                                            arg7::AbstractArray{T},
                                            arg8::AbstractArray{T},
                                            arg9::AbstractArray{T},
                                            arg10::T,
                                            arg11::T,
                                            arg12::T,
                                            arg13::T,
                                            arg14::T,
                                            arg15::T,
                                            arg16::T,
                                            arg17::Bool,
                                            arg18) where {D, T}
    out = dual_function_gb_energy_loop(f).(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
                                arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18)
    y = value.(out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast((y1, o1) -> SVector{D, T}(partials(o1, 1) * y1,
                    partials(o1, 2) * y1, partials(o1, 3) * y1), ȳ, out))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(partials(o1, 4) * y1,
                    partials(o1, 5) * y1, partials(o1, 6) * y1), ȳ, out))
        darg3 = nothing
        darg4 = nothing
        darg5 = unbroadcast(arg5, broadcast((y1, o1) -> partials(o1,  7) * y1, ȳ, out))
        darg6 = unbroadcast(arg6, broadcast((y1, o1) -> partials(o1,  8) * y1, ȳ, out))
        darg7 = unbroadcast(arg7, broadcast((y1, o1) -> partials(o1,  9) * y1, ȳ, out))
        darg8 = unbroadcast(arg8, broadcast((y1, o1) -> partials(o1, 10) * y1, ȳ, out))
        darg9 = unbroadcast(arg9, broadcast((y1, o1) -> partials(o1, 11) * y1, ȳ, out))
        darg10 = nothing
        darg11 = unbroadcast(arg11, broadcast((y1, o1) -> partials(o1, 12) * y1, ȳ, out))
        darg12 = unbroadcast(arg12, broadcast((y1, o1) -> partials(o1, 13) * y1, ȳ, out))
        darg13 = unbroadcast(arg13, broadcast((y1, o1) -> partials(o1, 14) * y1, ȳ, out))
        darg14 = unbroadcast(arg14, broadcast((y1, o1) -> partials(o1, 15) * y1, ȳ, out))
        darg15 = unbroadcast(arg15, broadcast((y1, o1) -> partials(o1, 16) * y1, ȳ, out))
        darg16 = unbroadcast(arg16, broadcast((y1, o1) -> partials(o1, 17) * y1, ȳ, out))
        darg17 = nothing
        darg18 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6, darg7, darg8, darg9,
                darg10, darg11, darg12, darg13, darg14, darg15, darg16, darg17, darg18)
    end
    return y, bc_fwd_back
end

@inline function Zygote.broadcast_forward(f::typeof(get_i1), arg1)
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (y1, zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_i2), arg1)
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (zero(y1), y1), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_I),
                                            arg1::AbstractArray{<:BornRadiiGBN2LoopResult})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (I=y1, I_grad=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_I_grad),
                                            arg1::AbstractArray{<:BornRadiiGBN2LoopResult})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (I=zero(y1), I_grad=y1), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_bi),
                                            arg1::AbstractArray{<:ForceLoopResult1})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1,
                            broadcast(y1 -> (bi=y1, bj=zero(y1), fi=zero(y1), fj=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_bj),
                                            arg1::AbstractArray{<:ForceLoopResult1})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1,
                            broadcast(y1 -> (bi=zero(y1), bj=y1, fi=zero(y1), fj=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_fi),
                                            arg1::AbstractArray{<:ForceLoopResult1})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1,
                            broadcast(y1 -> (bi=zero(y1), bj=zero(y1), fi=y1, fj=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_fj),
                                            arg1::AbstractArray{<:ForceLoopResult1})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1,
                            broadcast(y1 -> (bi=zero(y1), bj=zero(y1), fi=zero(y1), fj=y1), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_fi),
                                            arg1::AbstractArray{<:ForceLoopResult2})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (fi=y1, fj=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_fj),
                                            arg1::AbstractArray{<:ForceLoopResult2})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (fi=zero(y1), fj=y1), ȳ)))
end

# Use fast broadcast path on CPU
for op in (:+, :-, :*, :/, :mass, :charge, :remove_molar, :ustrip, :ustrip_vec, :wrap_coords,
            :born_radii_loop_OBC, :get_i1, :get_i2, :get_I, :get_I_grad, :born_radii_loop_GBN2,
            :get_bi, :get_bj, :get_fi, :get_fj, :gb_force_loop_1, :gb_force_loop_2, :gb_energy_loop)
    @eval Zygote.@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::typeof($op), args...) = Zygote.broadcast_forward(f, args...)
    # Avoid ambiguous dispatch
    @eval Zygote.@adjoint Broadcast.broadcasted(::CUDA.AbstractGPUArrayStyle  , f::typeof($op), args...) = Zygote.broadcast_forward(f, args...)
end
