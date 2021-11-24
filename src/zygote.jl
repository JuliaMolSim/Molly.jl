# Extend Zygote to work with static vectors on the GPU
# Here be dragons

Zygote.accum_sum(xs::AbstractVector{<:StaticVector}; dims=:) = Zygote.accum_sum(sum.(xs); dims=:)

# See the dualize function in ForwardDiff
@generated function dualize_add1(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(ForwardDiff.Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, ForwardDiff.Dual{T,eltype(x),N+1})
    return quote
        chunk = ForwardDiff.Chunk{$N+1}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@generated function dualize_add3(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(ForwardDiff.Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, ForwardDiff.Dual{T,eltype(x),N+3})
    return quote
        chunk = ForwardDiff.Chunk{$N+3}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@generated function dualize_add3bef(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(ForwardDiff.Dual{T}(x[$i], chunk, Val{$i+3}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, ForwardDiff.Dual{T,eltype(x),N+3})
    return quote
        chunk = ForwardDiff.Chunk{$N+3}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

function dual_function_svec(f::F) where F
    function (arg1)
        ds1 = ForwardDiff.dualize(Nothing, arg1)
        return f(ds1)
    end
end

function dual_function_svec_real(f::F) where F
    function (arg1, arg2)
        ds1 = dualize_add1(Nothing, arg1)
        ds2 = Zygote.dual(arg2, (false, false, false, true))
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

function matm_123(sv::ForwardDiff.Dual{Nothing, T}, y1::SVector{3, T}) where T
    ForwardDiff.partials(sv, 1) * y1[1] + ForwardDiff.partials(sv, 2) * y1[2] + ForwardDiff.partials(sv, 3) * y1[3]
end
  
function matm_456(sv::ForwardDiff.Dual{Nothing, T}, y1::SVector{3, T}) where T
    ForwardDiff.partials(sv, 4) * y1[1] + ForwardDiff.partials(sv, 5) * y1[2] + ForwardDiff.partials(sv, 6) * y1[3]
end

@inline function Zygote.broadcast_forward(f, arg1::CuArray{SVector{D, T}}) where {D, T}
    out = dual_function_svec(f).(arg1)
    y = map(x -> ForwardDiff.value.(x), out)
    function bc_fwd_back(ȳ)
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(ForwardDiff.partials(o1))
            else
                SVector{D, T}(matm_123.(o1, (y1,)))
            end
        end
        darg1 = Zygote.unbroadcast(arg1, barg1)
        (nothing, nothing, darg1) # nothings for broadcasted & f
    end
    return y, bc_fwd_back
end

@inline function Zygote.broadcast_forward(f, arg1::CuArray{SVector{D, T}}, arg2::Real) where {D, T}
    out = dual_function_svec_real(f).(arg1, arg2)
    y = map(x -> ForwardDiff.value.(x), out)
    function bc_fwd_back(ȳ)
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(ForwardDiff.partials.((o1,), (1, 2, 3)))
            else
                SVector{D, T}(matm_123.(o1, (y1,)))
            end
        end
        darg1 = Zygote.unbroadcast(arg1, barg1)
        darg2 = Zygote.unbroadcast(arg2, broadcast((y1, o1) -> y1 .* ForwardDiff.partials.(o1, 4), ȳ, out))
        (nothing, nothing, darg1, darg2) # nothings for broadcasted & f
    end
    return y, bc_fwd_back
end

@inline function Zygote.broadcast_forward(f, arg1::CuArray{SVector{D, T}}, arg2::CuArray{SVector{D, T}}) where {D, T}
    out = dual_function_svec_svec(f).(arg1, arg2)
    y = map(x -> ForwardDiff.value.(x), out)
    function bc_fwd_back(ȳ)
        barg1 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(ForwardDiff.partials.((o1,), (1, 2, 3)))
            else
                SVector{D, T}(matm_123.(o1, (y1,)))
            end
        end
        darg1 = Zygote.unbroadcast(arg1, barg1)
        barg2 = broadcast(ȳ, out) do y1, o1
            if length(y1) == 1
                y1 .* SVector{D, T}(ForwardDiff.partials.((o1,), (4, 5, 6)))
            else
                SVector{D, T}(matm_456.(o1, (y1,)))
            end
        end
        darg2 = Zygote.unbroadcast(arg2, barg2)
        (nothing, nothing, darg1, darg2) # nothings for broadcasted & f
    end
    return y, bc_fwd_back
end
