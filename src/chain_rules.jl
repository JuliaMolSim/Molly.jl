# Chain rules to allow differentiable simulations

function ChainRulesCore.rrule(T::Type{<:SVector}, x::Number...)
    function SVector_pullback(Ȳ)
        return NoTangent(), Ȳ...
    end
    return T(x...), SVector_pullback
end

function ChainRulesCore.rrule(::typeof(accumulateadd), x)
    Y = accumulateadd(x)
    function accumulateadd_pullback(Ȳ)
        return NoTangent(), Ȳ .* collect(length(x):-1:1)
    end
    return Y, accumulateadd_pullback
end
