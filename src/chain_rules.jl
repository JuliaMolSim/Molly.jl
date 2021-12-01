# Chain rules to allow differentiable simulations

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
        return NoTangent(), Ȳ.index, Ȳ.charge, Ȳ.mass, Ȳ.σ, Ȳ.ϵ
    end
    return Y, Atom_pullback
end

function ChainRulesCore.rrule(::typeof(sparsevec), is, vs, l)
    Y = sparsevec(is, vs, l)
    function sparsevec_pullback(Ȳ)
        return NoTangent(), collect(1:length(Ȳ)), Ȳ, length(Ȳ)
    end
    return Y, sparsevec_pullback
end

function ChainRulesCore.rrule(::typeof(accumulateadd), x)
    Y = accumulateadd(x)
    function accumulateadd_pullback(Ȳ)
        return NoTangent(), reverse(accumulateadd(reverse(Ȳ)))
    end
    return Y, accumulateadd_pullback
end
