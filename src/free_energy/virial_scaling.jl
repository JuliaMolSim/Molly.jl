# Factor on a specific interaction's virial contribution. The default of 1 is correct whenever
# the force already carries the alchemical scaling.
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
