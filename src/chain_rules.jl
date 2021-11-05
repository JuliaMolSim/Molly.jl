# Chain rules to allow differentiable simulations

function ChainRulesCore.rrule(::typeof(accumulateadd), x)
    Y = accumulateadd(x)
    function accumulateadd_pullback(Ȳ)
        return NoTangent(), Ȳ .* collect(length(x):-1:1)
    end
    return Y, accumulateadd_pullback
end
