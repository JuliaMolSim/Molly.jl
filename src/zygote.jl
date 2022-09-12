# Extend Zygote to work with static vectors and custom types on the
#   fast broadcast/GPU path
# Here be dragons

using ForwardDiff: Chunk, Dual, dualize, partials, value
using Zygote: unbroadcast

Zygote.accum(x::AbstractArray{<:SizedVector}, ys::AbstractArray{<:SVector}...) = Zygote.accum.(convert(typeof(ys[1]), x), ys...)
Zygote.accum(x::AbstractArray{<:SVector}, ys::AbstractArray{<:SizedVector}...) = Zygote.accum.(x, convert.(typeof(x), ys)...)

Zygote.accum(x::Vector{<:SVector} , y::CuArray{<:SVector}) = Zygote.accum(CuArray(x), y)
Zygote.accum(x::CuArray{<:SVector}, y::Vector{<:SVector} ) = Zygote.accum(x, CuArray(y))
Zygote.accum(x::Vector{<:SVector} , y::ROCArray{<:SVector}) = Zygote.accum(ROCArray(x), y)
Zygote.accum(x::ROCArray{<:SVector}, y::Vector{<:SVector} ) = Zygote.accum(x, ROCArray(y))

Zygote.accum(x::SVector{D, T}, y::T) where {D, T} = x .+ y

Base.:+(x::Real, y::SizedVector) = x .+ y
Base.:+(x::SizedVector, y::Real) = x .+ y

Base.:+(x::Real, y::Zygote.OneElement) = x .+ y
Base.:+(x::Zygote.OneElement, y::Real) = x .+ y

function Base.:+(x::Atom{T, T, T, T}, y::Atom{T, T, T, T}) where T
    Atom{T, T, T, T}(0, x.charge + y.charge, x.mass + y.mass, x.σ + y.σ, x.ϵ + y.ϵ, false)
end

function Base.:-(x::Atom{T, T, T, T}, y::Atom{T, T, T, T}) where T
    Atom{T, T, T, T}(0, x.charge - y.charge, x.mass - y.mass, x.σ - y.σ, x.ϵ - y.ϵ, false)
end

# For inject_gradients
function Base.:+(x::NamedTuple{(:atoms, :atoms_data, :pairwise_inters, :specific_inter_lists,
                    :general_inters, :constraints, :coords, :velocities, :boundary,
                    :neighbor_finder, :loggers, :force_units, :energy_units, :k),
                    Tuple{Nothing, Vector{Nothing}, Nothing, Nothing, Nothing, Nothing, Nothing,
                    Nothing, SVector{D, T}, Nothing, Nothing, Nothing, Nothing, Nothing}},
                    y::Base.RefValue{Any}) where {D, T}
    y
end

function Zygote.accum(x::LennardJones{S, C, W, WS, F, E}, y::LennardJones{S, C, W, WS, F, E}) where {S, C, W, WS, F, E}
    LennardJones{S, C, W, WS, F, E}(
        x.cutoff,
        x.nl_only,
        x.lorentz_mixing,
        x.weight_14 + y.weight_14,
        x.weight_solute_solvent + y.weight_solute_solvent,
        x.force_units,
        x.energy_units,
    )
end

function Zygote.accum(x::Coulomb{C, W, T, F, E}, y::Coulomb{C, W, T, F, E}) where {C, W, T, F, E}
    Coulomb{C, W, T, F, E}(
        x.cutoff,
        x.nl_only,
        x.weight_14 + y.weight_14,
        x.coulomb_const + y.coulomb_const,
        x.force_units,
        x.energy_units,
    )
end

function Zygote.accum(x::CoulombReactionField{D, S, W, T, F, E}, y::CoulombReactionField{D, S, W, T, F, E}) where {D, S, W, T, F, E}
    CoulombReactionField{D, S, W, T, F, E}(
        x.dist_cutoff + y.dist_cutoff,
        x.solvent_dielectric + y.solvent_dielectric,
        x.nl_only,
        x.weight_14 + y.weight_14,
        x.coulomb_const + y.coulomb_const,
        x.force_units,
        x.energy_units,
    )
end

function Zygote.accum_sum(xs::AbstractArray{Tuple{LennardJones{S, C, W, WS, F, E}, Coulomb{C, WC, T, F, E}}}; dims=:) where {S, C, W, WS, F, E, WC, T}
    reduce(Zygote.accum, xs, dims=dims; init=(
        LennardJones{S, C, W, WS, F, E}(nothing, false, false, zero(W), zero(WS), NoUnits, NoUnits),
        Coulomb{C, WC, T, F, E}(nothing, false, zero(WC), zero(T), NoUnits, NoUnits),
    ))
end

function Zygote.accum_sum(xs::AbstractArray{Tuple{LennardJones{S, C, W, WS, F, E}, CoulombReactionField{D, SO, WC, T, F, E}}}; dims=:) where {S, C, W, WS, F, E, D, SO, WC, T}
    reduce(Zygote.accum, xs, dims=dims; init=(
        LennardJones{S, C, W, WS, F, E}(nothing, false, false, zero(W), zero(WS), NoUnits, NoUnits),
        CoulombReactionField{D, SO, WC, T, F, E}(zero(D), zero(SO), false, zero(WC), zero(T), NoUnits, NoUnits),
    ))
end

function Zygote.accum(x::Tuple{NTuple{N, Int}, NTuple{N, T}, NTuple{N, E}, Bool},
                        y::Tuple{NTuple{N, Int}, NTuple{N, T}, NTuple{N, E}, Bool}) where {N, T, E}
    ntuple(n -> 0, N), x[2] .+ y[2], x[3] .+ y[3], false
end

function Zygote.accum(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{3, T, Vector{T}}}}, y::SVector{3, T}) where T
    CubicBoundary(x.side_lengths .+ y)
end

function Zygote.accum(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{2, T, Vector{T}}}}, y::SVector{2, T}) where T
    RectangularBoundary(x.side_lengths .+ y)
end

function Base.:+(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{3, T, Vector{T}}}}, y::CubicBoundary{T}) where T
    CubicBoundary(x.side_lengths .+ y.side_lengths)
end

function Base.:+(x::NamedTuple{(:side_lengths,), Tuple{SizedVector{2, T, Vector{T}}}}, y::RectangularBoundary{T}) where T
    RectangularBoundary(x.side_lengths .+ y.side_lengths)
end

Base.zero(::Type{Atom{T, T, T, T}}) where {T} = Atom(0, zero(T), zero(T), zero(T), zero(T), false)
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
Zygote.∇getindex(x::AT, inds::Tuple{AbstractArray{<:Integer}}) where AT <: Union{CuArray, ROCArray} = dy -> begin
    inds1_cpu = Array(inds[1])
    dx = zeros(eltype(dy), length(x))
    dxv = view(dx, inds1_cpu)
    dxv .= Zygote.accum.(dxv, Zygote._droplike(Array(dy), dxv))
    return Zygote._project(x, AT(dx)), nothing
end

# Extend to add extra empty partials before (B) and after (A) the SVector partials
@generated function ForwardDiff.dualize(::Type{T}, x::StaticArray, ::Val{B}, ::Val{A}) where {T, B, A}
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

function modify_grad(ȳ_in::AbstractArray{SizedVector{D, T, Vector{T}}}, arg::AT) where {D, T, AT <: Union{CuArray, ROCArray}}
    AT(sized_to_static.(ȳ_in))
end

function modify_grad(ȳ_in::AbstractArray{SizedVector{D, T, Vector{T}}}, arg) where {D, T}
    sized_to_static.(ȳ_in)
end

modify_grad(ȳ_in, arg::AT) where AT <: Union{CuArray, ROCArray} = AT(ȳ_in)
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
        ds2 = Zygote.dual(arg2 isa Int ? T(arg2) : arg2, (false, false, false, true))
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
        ds1 = Zygote.dual(arg1 isa Int ? T(arg1) : arg1, (true, false, false, false))
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
        ds1 = Atom(arg1.index, @dualize(c, 4, 1), @dualize(m, 4, 2), @dualize(σ, 4, 3),
                    @dualize(ϵ, 4, 4), arg1.solute)
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

# No gradient for cutoff type
function dualize_fb(inter::LennardJones{S, C, W, WS, F, E}) where {S, C, W, WS, F, E}
    w14, wss = inter.weight_14, inter.weight_solute_solvent
    dual_weight_14 = @dualize(w14, 20, 1)
    dual_weight_solute_solvent = @dualize(wss, 20, 2)
    return LennardJones{S, C, typeof(dual_weight_14), typeof(dual_weight_solute_solvent), F, E}(
                inter.cutoff, inter.nl_only, inter.lorentz_mixing, dual_weight_14,
                dual_weight_solute_solvent, inter.force_units, inter.energy_units)
end

function dualize_fb(inter::Coulomb{C, W, T, F, E}) where {C, W, T, F, E}
    w14, cc = inter.weight_14, inter.coulomb_const
    dual_weight_14     = @dualize(w14, 20, 3)
    dual_coulomb_const = @dualize(cc , 20, 4)
    return Coulomb{C, typeof(dual_weight_14), typeof(dual_coulomb_const), F, E}(
                inter.cutoff, inter.nl_only, dual_weight_14, dual_coulomb_const,
                inter.force_units, inter.energy_units)
end

function dualize_fb(inter::CoulombReactionField{D, S, W, T, F, E}) where {D, S, W, T, F, E}
    dc, sd, w14, cc = inter.dist_cutoff, inter.solvent_dielectric, inter.weight_14, inter.coulomb_const
    dual_dist_cutoff        = @dualize(dc , 20, 3)
    dual_solvent_dielectric = @dualize(sd , 20, 4)
    dual_weight_14          = @dualize(w14, 20, 5)
    dual_coulomb_const      = @dualize(cc , 20, 6)
    return CoulombReactionField{typeof(dual_dist_cutoff), typeof(dual_solvent_dielectric), typeof(dual_weight_14), typeof(dual_coulomb_const), F, E}(
                                dual_dist_cutoff, dual_solvent_dielectric, inter.nl_only, dual_weight_14,
                                dual_coulomb_const, inter.force_units, inter.energy_units)
end

function dualize_atom_fb1(at::Atom)
    c, m, σ, ϵ = at.charge, at.mass, at.σ, at.ϵ
    dual_charge = @dualize(c, 20, 13)
    dual_mass = @dualize(m, 20, 14)
    dual_σ = @dualize(σ, 20, 15)
    dual_ϵ = @dualize(ϵ, 20, 16)
    return Atom{typeof(dual_charge), typeof(dual_mass), typeof(dual_σ), typeof(dual_ϵ)}(
                at.index, dual_charge, dual_mass, dual_σ, dual_ϵ, at.solute)
end

function dualize_atom_fb2(at::Atom)
    c, m, σ, ϵ = at.charge, at.mass, at.σ, at.ϵ
    dual_charge = @dualize(c, 20, 17)
    dual_mass = @dualize(m, 20, 18)
    dual_σ = @dualize(σ, 20, 19)
    dual_ϵ = @dualize(ϵ, 20, 20)
    return Atom{typeof(dual_charge), typeof(dual_mass), typeof(dual_σ), typeof(dual_ϵ)}(
                at.index, dual_charge, dual_mass, dual_σ, dual_ϵ, at.solute)
end

function dual_function_force_broadcast(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        ds1 = map(dualize_fb, arg1)
        ds2 = dualize(Nothing, arg2, Val(6), Val(11))
        ds3 = dualize(Nothing, arg3, Val(9), Val(8))
        ds4 = dualize_atom_fb1(arg4)
        ds5 = dualize_atom_fb2(arg5)
        ds6 = arg6
        ds7 = arg7
        ds8 = arg8
        return f(ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8)
    end
end

function combine_dual_PairwiseInteraction_coul(y1::SVector{3, T}, o1::SVector{3, Dual{Nothing, T, P}}, i::Integer) where {T, P}
    (
        LennardJones{false, Nothing, T, T, typeof(NoUnits), typeof(NoUnits)}(
            nothing, false, false,
            y1[1] * partials(o1[1], i    ) + y1[2] * partials(o1[2], i    ) + y1[3] * partials(o1[3], i    ),
            y1[1] * partials(o1[1], i + 1) + y1[2] * partials(o1[2], i + 1) + y1[3] * partials(o1[3], i + 1),
            NoUnits, NoUnits,
        ),
        Coulomb{Nothing, T, T, typeof(NoUnits), typeof(NoUnits)}(
            nothing, false,
            y1[1] * partials(o1[1], i + 2) + y1[2] * partials(o1[2], i + 2) + y1[3] * partials(o1[3], i + 2),
            y1[1] * partials(o1[1], i + 3) + y1[2] * partials(o1[2], i + 3) + y1[3] * partials(o1[3], i + 3),
            NoUnits, NoUnits,
        ),
    )
end

function combine_dual_PairwiseInteraction_crf(y1::SVector{3, T}, o1::SVector{3, Dual{Nothing, T, P}}, i::Integer) where {T, P}
    (
        LennardJones{false, Nothing, T, T, typeof(NoUnits), typeof(NoUnits)}(
            nothing, false, false,
            y1[1] * partials(o1[1], i    ) + y1[2] * partials(o1[2], i    ) + y1[3] * partials(o1[3], i    ),
            y1[1] * partials(o1[1], i + 1) + y1[2] * partials(o1[2], i + 1) + y1[3] * partials(o1[3], i + 1),
            NoUnits, NoUnits,
        ),
        CoulombReactionField{T, T, T, T, typeof(NoUnits), typeof(NoUnits)}(
            y1[1] * partials(o1[1], i + 2) + y1[2] * partials(o1[2], i + 2) + y1[3] * partials(o1[3], i + 2),
            y1[1] * partials(o1[1], i + 3) + y1[2] * partials(o1[2], i + 3) + y1[3] * partials(o1[3], i + 3),
            false,
            y1[1] * partials(o1[1], i + 4) + y1[2] * partials(o1[2], i + 4) + y1[3] * partials(o1[3], i + 4),
            y1[1] * partials(o1[1], i + 5) + y1[2] * partials(o1[2], i + 5) + y1[3] * partials(o1[3], i + 5),
            NoUnits, NoUnits,
        ),
    )
end

function combine_dual_Atom(y1::SVector{3, T}, o1::SVector{3, Dual{Nothing, T, P}}, i::Integer, j::Integer, k::Integer, l::Integer) where {T, P}
    ps1, ps2, ps3 = partials(o1[1]), partials(o1[2]), partials(o1[3])
    Atom(
        0,
        y1[1] * ps1[i] + y1[2] * ps2[i] + y1[3] * ps3[i],
        y1[1] * ps1[j] + y1[2] * ps2[j] + y1[3] * ps3[j],
        y1[1] * ps1[k] + y1[2] * ps2[k] + y1[3] * ps3[k],
        y1[1] * ps1[l] + y1[2] * ps2[l] + y1[3] * ps3[l],
        false,
    )
end

# For force_nounit
@inline function Zygote.broadcast_forward(f,
                                            arg1::A,
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{SVector{D, T}},
                                            arg4::AbstractArray{<:Atom},
                                            arg5::AbstractArray{<:Atom},
                                            arg6,
                                            arg7::Base.RefValue{<:Unitful.FreeUnits},
                                            arg8) where {A, D, T}
    out = dual_function_force_broadcast(f).(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    y = map(x -> value.(x), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg2)
        if A <: Tuple{Tuple{X, Y}} where {X <: LennardJones, Y <: Coulomb}
            darg1 = unbroadcast(arg1, broadcast(combine_dual_PairwiseInteraction_coul, ȳ, out, 1))
        elseif A <: Tuple{Tuple{X, Y}} where {X <: LennardJones, Y <: CoulombReactionField}
            darg1 = unbroadcast(arg1, broadcast(combine_dual_PairwiseInteraction_crf , ȳ, out, 1))
        end
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(sum_partials(o1, y1,  7),
                                sum_partials(o1, y1,  8), sum_partials(o1, y1,  9)), ȳ, out))
        darg3 = unbroadcast(arg3, broadcast((y1, o1) -> SVector{D, T}(sum_partials(o1, y1, 10),
                                sum_partials(o1, y1, 11), sum_partials(o1, y1, 12)), ȳ, out))
        darg4 = unbroadcast(arg4, broadcast(combine_dual_Atom, ȳ, out, 13, 14, 15, 16))
        darg5 = unbroadcast(arg5, broadcast(combine_dual_Atom, ȳ, out, 17, 18, 19, 20))
        darg6 = nothing
        darg7 = nothing
        darg8 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6, darg7, darg8)
    end
    return y, bc_fwd_back
end

function dualize_fb(inter::HarmonicBond{K, D}) where {K, D}
    k, r0 = inter.k, inter.r0
    dual_k  = @dualize(k , 8, 1)
    dual_r0 = @dualize(r0, 8, 2)
    return HarmonicBond{typeof(dual_k), typeof(dual_r0)}(dual_k, dual_r0)
end

function dual_function_specific_2_atoms(f::F) where F
    function (arg1, arg2, arg3, arg4)
        ds1 = dualize_fb(arg1)
        ds2 = dualize(Nothing, arg2, Val(2), Val(3))
        ds3 = dualize(Nothing, arg3, Val(5), Val(0))
        ds4 = arg4
        return f(ds1, ds2, ds3, ds4)
    end
end

function combine_dual_SpecificInteraction(inter::HarmonicBond, y1, o1, i::Integer)
    (y1.f1[1] * partials(o1.f1[1], i    ) + y1.f1[2] * partials(o1.f1[2], i    ) + y1.f1[3] * partials(o1.f1[3], i    ) + y1.f2[1] * partials(o1.f2[1], i    ) + y1.f2[2] * partials(o1.f2[2], i    ) + y1.f2[3] * partials(o1.f2[3], i    ),
     y1.f1[1] * partials(o1.f1[1], i + 1) + y1.f1[2] * partials(o1.f1[2], i + 1) + y1.f1[3] * partials(o1.f1[3], i + 1) + y1.f2[1] * partials(o1.f2[1], i + 1) + y1.f2[2] * partials(o1.f2[2], i + 1) + y1.f2[3] * partials(o1.f2[3], i + 1))
end

# For force
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{<:SpecificInteraction},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{SVector{D, T}},
                                            arg4) where {D, T}
    out = dual_function_specific_2_atoms(f).(arg1, arg2, arg3, arg4)
    y = broadcast(o1 -> SpecificForce2Atoms{D, T}(value.(o1.f1), value.(o1.f2)), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast(combine_dual_SpecificInteraction, arg1, ȳ, out, 1))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 3) + sum_partials(o1.f2, y1.f2, 3),
                    sum_partials(o1.f1, y1.f1, 4) + sum_partials(o1.f2, y1.f2, 4),
                    sum_partials(o1.f1, y1.f1, 5) + sum_partials(o1.f2, y1.f2, 5)),
                    ȳ, out))
        darg3 = unbroadcast(arg3, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 6) + sum_partials(o1.f2, y1.f2, 6),
                    sum_partials(o1.f1, y1.f1, 7) + sum_partials(o1.f2, y1.f2, 7),
                    sum_partials(o1.f1, y1.f1, 8) + sum_partials(o1.f2, y1.f2, 8)),
                    ȳ, out))
        darg4 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4)
    end
    return y, bc_fwd_back
end

function dualize_fb(inter::HarmonicAngle{K, D}) where {K, D}
    k, θ0 = inter.k, inter.θ0
    dual_k  = @dualize(k , 11, 1)
    dual_θ0 = @dualize(θ0, 11, 2)
    return HarmonicAngle{typeof(dual_k), typeof(dual_θ0)}(dual_k, dual_θ0)
end

function dual_function_specific_3_atoms(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5)
        ds1 = dualize_fb(arg1)
        ds2 = dualize(Nothing, arg2, Val( 2), Val(6))
        ds3 = dualize(Nothing, arg3, Val( 5), Val(3))
        ds4 = dualize(Nothing, arg4, Val( 8), Val(0))
        ds5 = arg5
        return f(ds1, ds2, ds3, ds4, ds5)
    end
end

function combine_dual_SpecificInteraction(inter::HarmonicAngle, y1, o1, i::Integer)
    (y1.f1[1] * partials(o1.f1[1], i    ) + y1.f1[2] * partials(o1.f1[2], i    ) + y1.f1[3] * partials(o1.f1[3], i    ) + y1.f2[1] * partials(o1.f2[1], i    ) + y1.f2[2] * partials(o1.f2[2], i    ) + y1.f2[3] * partials(o1.f2[3], i    ) + y1.f3[1] * partials(o1.f3[1], i    ) + y1.f3[2] * partials(o1.f3[2], i    ) + y1.f3[3] * partials(o1.f3[3], i    ),
     y1.f1[1] * partials(o1.f1[1], i + 1) + y1.f1[2] * partials(o1.f1[2], i + 1) + y1.f1[3] * partials(o1.f1[3], i + 1) + y1.f2[1] * partials(o1.f2[1], i + 1) + y1.f2[2] * partials(o1.f2[2], i + 1) + y1.f2[3] * partials(o1.f2[3], i + 1) + y1.f3[1] * partials(o1.f3[1], i + 1) + y1.f3[2] * partials(o1.f3[2], i + 1) + y1.f3[3] * partials(o1.f3[3], i + 1))
end

# For force
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{<:SpecificInteraction},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{SVector{D, T}},
                                            arg4::AbstractArray{SVector{D, T}},
                                            arg5) where {D, T}
    out = dual_function_specific_3_atoms(f).(arg1, arg2, arg3, arg4, arg5)
    y = broadcast(o1 -> SpecificForce3Atoms{D, T}(value.(o1.f1), value.(o1.f2), value.(o1.f3)), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast(combine_dual_SpecificInteraction, arg1, ȳ, out, 1))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 3) + sum_partials(o1.f2, y1.f2, 3) + sum_partials(o1.f3, y1.f3, 3),
                    sum_partials(o1.f1, y1.f1, 4) + sum_partials(o1.f2, y1.f2, 4) + sum_partials(o1.f3, y1.f3, 4),
                    sum_partials(o1.f1, y1.f1, 5) + sum_partials(o1.f2, y1.f2, 5) + sum_partials(o1.f3, y1.f3, 5)),
                    ȳ, out))
        darg3 = unbroadcast(arg3, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 6) + sum_partials(o1.f2, y1.f2, 6) + sum_partials(o1.f3, y1.f3, 6),
                    sum_partials(o1.f1, y1.f1, 7) + sum_partials(o1.f2, y1.f2, 7) + sum_partials(o1.f3, y1.f3, 7),
                    sum_partials(o1.f1, y1.f1, 8) + sum_partials(o1.f2, y1.f2, 8) + sum_partials(o1.f3, y1.f3, 8)),
                    ȳ, out))
        darg4 = unbroadcast(arg4, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1,  9) + sum_partials(o1.f2, y1.f2,  9) + sum_partials(o1.f3, y1.f3,  9),
                    sum_partials(o1.f1, y1.f1, 10) + sum_partials(o1.f2, y1.f2, 10) + sum_partials(o1.f3, y1.f3, 10),
                    sum_partials(o1.f1, y1.f1, 11) + sum_partials(o1.f2, y1.f2, 11) + sum_partials(o1.f3, y1.f3, 11)),
                    ȳ, out))
        darg5 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5)
    end
    return y, bc_fwd_back
end

function dualize_fb(inter::PeriodicTorsion{6, T, E}) where {T, E}
    p1, p2, p3, p4, p5, p6 = inter.phases
    k1, k2, k3, k4, k5, k6 = inter.ks
    dual_phases = (
        @dualize(p1, 24,  1), @dualize(p2, 24,  2), @dualize(p3, 24,  3),
        @dualize(p4, 24,  4), @dualize(p5, 24,  5), @dualize(p6, 24,  6),
    )
    dual_ks = (
        @dualize(k1, 24,  7), @dualize(k2, 24,  8), @dualize(k3, 24,  9),
        @dualize(k4, 24, 10), @dualize(k5, 24, 11), @dualize(k6, 24, 12),
    )
    return PeriodicTorsion{6, eltype(dual_phases), eltype(dual_ks)}(inter.periodicities,
                            dual_phases, dual_ks, inter.proper)
end

function dual_function_specific_4_atoms(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6)
        ds1 = dualize_fb(arg1)
        ds2 = dualize(Nothing, arg2, Val(12), Val(9))
        ds3 = dualize(Nothing, arg3, Val(15), Val(6))
        ds4 = dualize(Nothing, arg4, Val(18), Val(3))
        ds5 = dualize(Nothing, arg5, Val(21), Val(0))
        ds6 = arg6
        return f(ds1, ds2, ds3, ds4, ds5, ds6)
    end
end

function combine_dual_SpecificInteraction(inter::PeriodicTorsion{6}, y1, o1, i::Integer)
    (
        (0, 0, 0, 0, 0, 0),
        (
            y1.f1[1] * partials(o1.f1[1], i     ) + y1.f1[2] * partials(o1.f1[2], i     ) + y1.f1[3] * partials(o1.f1[3], i     ) + y1.f2[1] * partials(o1.f2[1], i     ) + y1.f2[2] * partials(o1.f2[2], i     ) + y1.f2[3] * partials(o1.f2[3], i     ) + y1.f3[1] * partials(o1.f3[1], i     ) + y1.f3[2] * partials(o1.f3[2], i     ) + y1.f3[3] * partials(o1.f3[3], i     ) + y1.f4[1] * partials(o1.f4[1], i     ) + y1.f4[2] * partials(o1.f4[2], i     ) + y1.f4[3] * partials(o1.f4[3], i     ),
            y1.f1[1] * partials(o1.f1[1], i +  1) + y1.f1[2] * partials(o1.f1[2], i +  1) + y1.f1[3] * partials(o1.f1[3], i +  1) + y1.f2[1] * partials(o1.f2[1], i +  1) + y1.f2[2] * partials(o1.f2[2], i +  1) + y1.f2[3] * partials(o1.f2[3], i +  1) + y1.f3[1] * partials(o1.f3[1], i +  1) + y1.f3[2] * partials(o1.f3[2], i +  1) + y1.f3[3] * partials(o1.f3[3], i +  1) + y1.f4[1] * partials(o1.f4[1], i +  1) + y1.f4[2] * partials(o1.f4[2], i +  1) + y1.f4[3] * partials(o1.f4[3], i +  1),
            y1.f1[1] * partials(o1.f1[1], i +  2) + y1.f1[2] * partials(o1.f1[2], i +  2) + y1.f1[3] * partials(o1.f1[3], i +  2) + y1.f2[1] * partials(o1.f2[1], i +  2) + y1.f2[2] * partials(o1.f2[2], i +  2) + y1.f2[3] * partials(o1.f2[3], i +  2) + y1.f3[1] * partials(o1.f3[1], i +  2) + y1.f3[2] * partials(o1.f3[2], i +  2) + y1.f3[3] * partials(o1.f3[3], i +  2) + y1.f4[1] * partials(o1.f4[1], i +  2) + y1.f4[2] * partials(o1.f4[2], i +  2) + y1.f4[3] * partials(o1.f4[3], i +  2),
            y1.f1[1] * partials(o1.f1[1], i +  3) + y1.f1[2] * partials(o1.f1[2], i +  3) + y1.f1[3] * partials(o1.f1[3], i +  3) + y1.f2[1] * partials(o1.f2[1], i +  3) + y1.f2[2] * partials(o1.f2[2], i +  3) + y1.f2[3] * partials(o1.f2[3], i +  3) + y1.f3[1] * partials(o1.f3[1], i +  3) + y1.f3[2] * partials(o1.f3[2], i +  3) + y1.f3[3] * partials(o1.f3[3], i +  3) + y1.f4[1] * partials(o1.f4[1], i +  3) + y1.f4[2] * partials(o1.f4[2], i +  3) + y1.f4[3] * partials(o1.f4[3], i +  3),
            y1.f1[1] * partials(o1.f1[1], i +  4) + y1.f1[2] * partials(o1.f1[2], i +  4) + y1.f1[3] * partials(o1.f1[3], i +  4) + y1.f2[1] * partials(o1.f2[1], i +  4) + y1.f2[2] * partials(o1.f2[2], i +  4) + y1.f2[3] * partials(o1.f2[3], i +  4) + y1.f3[1] * partials(o1.f3[1], i +  4) + y1.f3[2] * partials(o1.f3[2], i +  4) + y1.f3[3] * partials(o1.f3[3], i +  4) + y1.f4[1] * partials(o1.f4[1], i +  4) + y1.f4[2] * partials(o1.f4[2], i +  4) + y1.f4[3] * partials(o1.f4[3], i +  4),
            y1.f1[1] * partials(o1.f1[1], i +  5) + y1.f1[2] * partials(o1.f1[2], i +  5) + y1.f1[3] * partials(o1.f1[3], i +  5) + y1.f2[1] * partials(o1.f2[1], i +  5) + y1.f2[2] * partials(o1.f2[2], i +  5) + y1.f2[3] * partials(o1.f2[3], i +  5) + y1.f3[1] * partials(o1.f3[1], i +  5) + y1.f3[2] * partials(o1.f3[2], i +  5) + y1.f3[3] * partials(o1.f3[3], i +  5) + y1.f4[1] * partials(o1.f4[1], i +  5) + y1.f4[2] * partials(o1.f4[2], i +  5) + y1.f4[3] * partials(o1.f4[3], i +  5),
        ),
        (
            y1.f1[1] * partials(o1.f1[1], i +  6) + y1.f1[2] * partials(o1.f1[2], i +  6) + y1.f1[3] * partials(o1.f1[3], i +  6) + y1.f2[1] * partials(o1.f2[1], i +  6) + y1.f2[2] * partials(o1.f2[2], i +  6) + y1.f2[3] * partials(o1.f2[3], i +  6) + y1.f3[1] * partials(o1.f3[1], i +  6) + y1.f3[2] * partials(o1.f3[2], i +  6) + y1.f3[3] * partials(o1.f3[3], i +  6) + y1.f4[1] * partials(o1.f4[1], i +  6) + y1.f4[2] * partials(o1.f4[2], i +  6) + y1.f4[3] * partials(o1.f4[3], i +  6),
            y1.f1[1] * partials(o1.f1[1], i +  7) + y1.f1[2] * partials(o1.f1[2], i +  7) + y1.f1[3] * partials(o1.f1[3], i +  7) + y1.f2[1] * partials(o1.f2[1], i +  7) + y1.f2[2] * partials(o1.f2[2], i +  7) + y1.f2[3] * partials(o1.f2[3], i +  7) + y1.f3[1] * partials(o1.f3[1], i +  7) + y1.f3[2] * partials(o1.f3[2], i +  7) + y1.f3[3] * partials(o1.f3[3], i +  7) + y1.f4[1] * partials(o1.f4[1], i +  7) + y1.f4[2] * partials(o1.f4[2], i +  7) + y1.f4[3] * partials(o1.f4[3], i +  7),
            y1.f1[1] * partials(o1.f1[1], i +  8) + y1.f1[2] * partials(o1.f1[2], i +  8) + y1.f1[3] * partials(o1.f1[3], i +  8) + y1.f2[1] * partials(o1.f2[1], i +  8) + y1.f2[2] * partials(o1.f2[2], i +  8) + y1.f2[3] * partials(o1.f2[3], i +  8) + y1.f3[1] * partials(o1.f3[1], i +  8) + y1.f3[2] * partials(o1.f3[2], i +  8) + y1.f3[3] * partials(o1.f3[3], i +  8) + y1.f4[1] * partials(o1.f4[1], i +  8) + y1.f4[2] * partials(o1.f4[2], i +  8) + y1.f4[3] * partials(o1.f4[3], i +  8),
            y1.f1[1] * partials(o1.f1[1], i +  9) + y1.f1[2] * partials(o1.f1[2], i +  9) + y1.f1[3] * partials(o1.f1[3], i +  9) + y1.f2[1] * partials(o1.f2[1], i +  9) + y1.f2[2] * partials(o1.f2[2], i +  9) + y1.f2[3] * partials(o1.f2[3], i +  9) + y1.f3[1] * partials(o1.f3[1], i +  9) + y1.f3[2] * partials(o1.f3[2], i +  9) + y1.f3[3] * partials(o1.f3[3], i +  9) + y1.f4[1] * partials(o1.f4[1], i +  9) + y1.f4[2] * partials(o1.f4[2], i +  9) + y1.f4[3] * partials(o1.f4[3], i +  9),
            y1.f1[1] * partials(o1.f1[1], i + 10) + y1.f1[2] * partials(o1.f1[2], i + 10) + y1.f1[3] * partials(o1.f1[3], i + 10) + y1.f2[1] * partials(o1.f2[1], i + 10) + y1.f2[2] * partials(o1.f2[2], i + 10) + y1.f2[3] * partials(o1.f2[3], i + 10) + y1.f3[1] * partials(o1.f3[1], i + 10) + y1.f3[2] * partials(o1.f3[2], i + 10) + y1.f3[3] * partials(o1.f3[3], i + 10) + y1.f4[1] * partials(o1.f4[1], i + 10) + y1.f4[2] * partials(o1.f4[2], i + 10) + y1.f4[3] * partials(o1.f4[3], i + 10),
            y1.f1[1] * partials(o1.f1[1], i + 11) + y1.f1[2] * partials(o1.f1[2], i + 11) + y1.f1[3] * partials(o1.f1[3], i + 11) + y1.f2[1] * partials(o1.f2[1], i + 11) + y1.f2[2] * partials(o1.f2[2], i + 11) + y1.f2[3] * partials(o1.f2[3], i + 11) + y1.f3[1] * partials(o1.f3[1], i + 11) + y1.f3[2] * partials(o1.f3[2], i + 11) + y1.f3[3] * partials(o1.f3[3], i + 11) + y1.f4[1] * partials(o1.f4[1], i + 11) + y1.f4[2] * partials(o1.f4[2], i + 11) + y1.f4[3] * partials(o1.f4[3], i + 11),
        ),
        false,
    )
end

# For force
@inline function Zygote.broadcast_forward(f,
                                            arg1::AbstractArray{<:SpecificInteraction},
                                            arg2::AbstractArray{SVector{D, T}},
                                            arg3::AbstractArray{SVector{D, T}},
                                            arg4::AbstractArray{SVector{D, T}},
                                            arg5::AbstractArray{SVector{D, T}},
                                            arg6) where {D, T}
    out = dual_function_specific_4_atoms(f).(arg1, arg2, arg3, arg4, arg5, arg6)
    y = broadcast(o1 -> SpecificForce4Atoms{D, T}(value.(o1.f1), value.(o1.f2), value.(o1.f3), value.(o1.f4)), out)
    function bc_fwd_back(ȳ_in)
        ȳ = modify_grad(ȳ_in, arg1)
        darg1 = unbroadcast(arg1, broadcast(combine_dual_SpecificInteraction, arg1, ȳ, out, 1))
        darg2 = unbroadcast(arg2, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 13) + sum_partials(o1.f2, y1.f2, 13) + sum_partials(o1.f3, y1.f3, 13) + sum_partials(o1.f4, y1.f4, 13),
                    sum_partials(o1.f1, y1.f1, 14) + sum_partials(o1.f2, y1.f2, 14) + sum_partials(o1.f3, y1.f3, 14) + sum_partials(o1.f4, y1.f4, 14),
                    sum_partials(o1.f1, y1.f1, 15) + sum_partials(o1.f2, y1.f2, 15) + sum_partials(o1.f3, y1.f3, 15) + sum_partials(o1.f4, y1.f4, 15)),
                    ȳ, out))
        darg3 = unbroadcast(arg3, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 16) + sum_partials(o1.f2, y1.f2, 16) + sum_partials(o1.f3, y1.f3, 16) + sum_partials(o1.f4, y1.f4, 16),
                    sum_partials(o1.f1, y1.f1, 17) + sum_partials(o1.f2, y1.f2, 17) + sum_partials(o1.f3, y1.f3, 17) + sum_partials(o1.f4, y1.f4, 17),
                    sum_partials(o1.f1, y1.f1, 18) + sum_partials(o1.f2, y1.f2, 18) + sum_partials(o1.f3, y1.f3, 18) + sum_partials(o1.f4, y1.f4, 18)),
                    ȳ, out))
        darg4 = unbroadcast(arg4, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 19) + sum_partials(o1.f2, y1.f2, 19) + sum_partials(o1.f3, y1.f3, 19) + sum_partials(o1.f4, y1.f4, 19),
                    sum_partials(o1.f1, y1.f1, 20) + sum_partials(o1.f2, y1.f2, 20) + sum_partials(o1.f3, y1.f3, 20) + sum_partials(o1.f4, y1.f4, 20),
                    sum_partials(o1.f1, y1.f1, 21) + sum_partials(o1.f2, y1.f2, 21) + sum_partials(o1.f3, y1.f3, 21) + sum_partials(o1.f4, y1.f4, 21)),
                    ȳ, out))
        darg5 = unbroadcast(arg5, broadcast((y1, o1) -> SVector{D, T}(
                    sum_partials(o1.f1, y1.f1, 22) + sum_partials(o1.f2, y1.f2, 22) + sum_partials(o1.f3, y1.f3, 22) + sum_partials(o1.f4, y1.f4, 22),
                    sum_partials(o1.f1, y1.f1, 23) + sum_partials(o1.f2, y1.f2, 23) + sum_partials(o1.f3, y1.f3, 23) + sum_partials(o1.f4, y1.f4, 23),
                    sum_partials(o1.f1, y1.f1, 24) + sum_partials(o1.f2, y1.f2, 24) + sum_partials(o1.f3, y1.f3, 24) + sum_partials(o1.f4, y1.f4, 24)),
                    ȳ, out))
        darg6 = nothing
        return (nothing, nothing, darg1, darg2, darg3, darg4, darg5, darg6)
    end
    return y, bc_fwd_back
end

function dual_function_born_radii_loop_OBC(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6)
        ds1 = dualize(Nothing, arg1, Val(0), Val(5))
        ds2 = dualize(Nothing, arg2, Val(3), Val(2))
        ds3 = Zygote.dual(arg3, (false, false, false, false, false, false, true , false))
        ds4 = Zygote.dual(arg4, (false, false, false, false, false, false, false, true ))
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

function dual_function_born_radii_loop_GBN2(f::F) where F
    function (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12)
        ds1  = dualize(Nothing, arg1, Val(0), Val(11))
        ds2  = dualize(Nothing, arg2, Val(3), Val(8))
        ds3  = Zygote.dual(arg3,  (false, false, false, false, false, false, true , false, false, false, false, false, false, false))
        ds4  = Zygote.dual(arg4,  (false, false, false, false, false, false, false, true , false, false, false, false, false, false))
        ds5  = Zygote.dual(arg5,  (false, false, false, false, false, false, false, false, true , false, false, false, false, false))
        ds6  = arg6
        ds7  = Zygote.dual(arg7,  (false, false, false, false, false, false, false, false, false, true , false, false, false, false))
        ds8  = Zygote.dual(arg8,  (false, false, false, false, false, false, false, false, false, false, true , false, false, false))
        ds9  = Zygote.dual(arg9,  (false, false, false, false, false, false, false, false, false, false, false, true , false, false))
        ds10 = Zygote.dual(arg10, (false, false, false, false, false, false, false, false, false, false, false, false, true , false))
        ds11 = Zygote.dual(arg11, (false, false, false, false, false, false, false, false, false, false, false, false, false, true ))
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
        ds5  = Zygote.dual(arg5 , (false, false, false, false, false, false, true , false, false, false, false, false, false))
        ds6  = Zygote.dual(arg6 , (false, false, false, false, false, false, false, true , false, false, false, false, false))
        ds7  = Zygote.dual(arg7 , (false, false, false, false, false, false, false, false, true , false, false, false, false))
        ds8  = Zygote.dual(arg8 , (false, false, false, false, false, false, false, false, false, true , false, false, false))
        ds9  = arg9
        ds10 = Zygote.dual(arg10, (false, false, false, false, false, false, false, false, false, false, true , false, false))
        ds11 = Zygote.dual(arg11, (false, false, false, false, false, false, false, false, false, false, false, true , false))
        ds12 = Zygote.dual(arg12, (false, false, false, false, false, false, false, false, false, false, true , false, true ))
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
        ds3 = Zygote.dual(arg3, (false, false, false, false, false, false, true , false, false, false))
        ds4 = Zygote.dual(arg4, (false, false, false, false, false, false, false, true , false, false))
        ds5 = Zygote.dual(arg5, (false, false, false, false, false, false, false, false, true , false))
        ds6 = Zygote.dual(arg6, (false, false, false, false, false, false, false, false, false, true ))
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
        ds5  = Zygote.dual(arg5 , (false, false, false, false, false, false, true , false, false, false, false, false, false, false, false, false, false))
        ds6  = Zygote.dual(arg6 , (false, false, false, false, false, false, false, true , false, false, false, false, false, false, false, false, false))
        ds7  = Zygote.dual(arg7 , (false, false, false, false, false, false, false, false, true , false, false, false, false, false, false, false, false))
        ds8  = Zygote.dual(arg8 , (false, false, false, false, false, false, false, false, false, true , false, false, false, false, false, false, false))
        ds9  = Zygote.dual(arg9 , (false, false, false, false, false, false, false, false, false, true , true , false, false, false, false, false, false))
        ds10 = arg10
        ds11 = Zygote.dual(arg11, (false, false, false, false, false, false, false, false, false, false, false, true , false, false, false, false, false))
        ds12 = Zygote.dual(arg12, (false, false, false, false, false, false, false, false, false, false, false, false, true , false, false, false, false))
        ds13 = Zygote.dual(arg13, (false, false, false, false, false, false, false, false, false, false, false, false, false, true , false, false, false))
        ds14 = Zygote.dual(arg14, (false, false, false, false, false, false, false, false, false, false, false, false, false, false, true , false, false))
        ds15 = Zygote.dual(arg15, (false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true , false))
        ds16 = Zygote.dual(arg16, (false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true ))
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

@inline function Zygote.broadcast_forward(f::typeof(get_f1),
                                            arg1::AbstractArray{<:SpecificForce2Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=y1, f2=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f1),
                                            arg1::AbstractArray{<:SpecificForce3Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=y1, f2=zero(y1), f3=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f1),
                                            arg1::AbstractArray{<:SpecificForce4Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=y1, f2=zero(y1), f3=zero(y1), f4=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f2),
                                            arg1::AbstractArray{<:SpecificForce2Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=zero(y1), f2=y1), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f2),
                                            arg1::AbstractArray{<:SpecificForce3Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=zero(y1), f2=y1, f3=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f2),
                                            arg1::AbstractArray{<:SpecificForce4Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=zero(y1), f2=y1, f3=zero(y1), f4=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f3),
                                            arg1::AbstractArray{<:SpecificForce3Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=zero(y1), f2=zero(y1), f3=y1), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f3),
                                            arg1::AbstractArray{<:SpecificForce4Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=zero(y1), f2=zero(y1), f3=y1, f4=zero(y1)), ȳ)))
end

@inline function Zygote.broadcast_forward(f::typeof(get_f4),
                                            arg1::AbstractArray{<:SpecificForce4Atoms})
    return f.(arg1), ȳ -> (nothing, nothing, unbroadcast(arg1, broadcast(y1 -> (f1=zero(y1), f2=zero(y1), f3=zero(y1), f4=y1), ȳ)))
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
            :get_f1, :get_f2, :get_f3, :get_f4, :born_radii_loop_OBC, :get_I, :get_I_grad,
            :born_radii_loop_GBN2, :get_bi, :get_bj, :get_fi, :get_fj, :gb_force_loop_1,
            :gb_force_loop_2, :gb_energy_loop)
    @eval Zygote.@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::typeof($op), args...) = Zygote.broadcast_forward(f, args...)
    # Avoid ambiguous dispatch
    @eval Zygote.@adjoint Broadcast.broadcasted(::CUDA.AbstractGPUArrayStyle  , f::typeof($op), args...) = Zygote.broadcast_forward(f, args...)
end

# Interactions not specified here run on the slow path on CPU
@eval Zygote.@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::typeof(force_nounit), inters::Tuple{Tuple{X, Y}}, args...) where {X <: LennardJones, Y <: CoulombReactionField} = Zygote.broadcast_forward(f, inters, args...)
@eval Zygote.@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::typeof(force_nounit), inters::Tuple{Tuple{X, Y}}, args...) where {X <: LennardJones, Y <: Coulomb             } = Zygote.broadcast_forward(f, inters, args...)

@eval Zygote.@adjoint Broadcast.broadcasted(::CUDA.AbstractGPUArrayStyle  , f::typeof(force_nounit), inters::Tuple{Tuple{X, Y}}, args...) where {X <: LennardJones, Y <: CoulombReactionField} = Zygote.broadcast_forward(f, inters, args...)
@eval Zygote.@adjoint Broadcast.broadcasted(::CUDA.AbstractGPUArrayStyle  , f::typeof(force_nounit), inters::Tuple{Tuple{X, Y}}, args...) where {X <: LennardJones, Y <: Coulomb             } = Zygote.broadcast_forward(f, inters, args...)

@eval Zygote.@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::typeof(force), inters::Vector{HarmonicBond{K, D}}      , args...) where {K, D}    = Zygote.broadcast_forward(f, inters, args...)
@eval Zygote.@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::typeof(force), inters::Vector{HarmonicAngle{K, D}}     , args...) where {K, D}    = Zygote.broadcast_forward(f, inters, args...)
@eval Zygote.@adjoint Broadcast.broadcasted(::Broadcast.AbstractArrayStyle, f::typeof(force), inters::Vector{PeriodicTorsion{N, T, E}}, args...) where {N, T, E} = Zygote.broadcast_forward(f, inters, args...)

@eval Zygote.@adjoint Broadcast.broadcasted(::CUDA.AbstractGPUArrayStyle  , f::typeof(force), inters::Vector{HarmonicBond{K, D}}      , args...) where {K, D}    = Zygote.broadcast_forward(f, inters, args...)
@eval Zygote.@adjoint Broadcast.broadcasted(::CUDA.AbstractGPUArrayStyle  , f::typeof(force), inters::Vector{HarmonicAngle{K, D}}     , args...) where {K, D}    = Zygote.broadcast_forward(f, inters, args...)
@eval Zygote.@adjoint Broadcast.broadcasted(::CUDA.AbstractGPUArrayStyle  , f::typeof(force), inters::Vector{PeriodicTorsion{N, T, E}}, args...) where {N, T, E} = Zygote.broadcast_forward(f, inters, args...)
