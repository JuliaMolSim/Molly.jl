"""
    virial_lambda_factor(inter, atoms::Tuple)

Factor applied to a specific interaction's contribution to the virial.

The default is `1`: the force already carries whatever alchemical scaling applies, either
because the interaction is not alchemical or because it scales itself. [`EwaldExclusion`](@ref)
(through its charge product) and `LennardJones14SoftCoreGapsys` are the two self-scaling
specific interactions, which is why neither carries a λ suffix — the suffix marks types that
consult a scheduler, and those two bake the coupling into their parameters instead.

Defaulting to `1` means an interaction nobody remembers to register keeps the physically
correct `Σ r ⊗ f` rather than being silently mis-scaled.
"""
@inline virial_lambda_factor(inter, atoms) = 1

# The bonded interactions an alchemical system is built from. `AbsoluteFESystem` and
# `RelativeFESystem` convert every specific interaction to its λ counterpart, so only the λ
# types can appear on an alchemical atom and only they need the coupling supplied.
const GeometryPreservingBonded = Union{
    HarmonicBondλ, MorseBondλ,
    HarmonicAngleλ, CosineAngleλ, UreyBradleyλ,
    PeriodicTorsionλ, RBTorsionλ, HarmonicTorsionλ, CMAPTorsionλ,
}

@inline function virial_lambda_factor(inter::GeometryPreservingBonded, atoms)
    λ_glob = λ_mixing(inter.λ_mixing, map(a -> a.λ, atoms))
    role = mix_roles(inter.scheduler, map(a -> a.alch_role, atoms))
    return scale_virial_dual(inter.scheduler, λ_glob, role)
end
