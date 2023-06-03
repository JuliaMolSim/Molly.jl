using AtomsBase
using AtomsBaseTesting


#This is copied from AtomsBaseTesting but converted to a System that is valid in Molly.
# The original system had a DirichletZero boundary condition and a Triclinic domain.
function make_test_system_molly(D=3; drop_atprop=Symbol[], drop_sysprop=Symbol[],
    extra_atprop=(; ), extra_sysprop=(; ), cellmatrix=:full)
    @assert D == 3
    n_atoms = 5

    # Generate some random data to store in Atoms
    atprop = Dict{Symbol,Any}(
    :position        => [randn(3) for _ = 1:n_atoms]u"Å",
    :velocity        => [randn(3) for _ = 1:n_atoms] * 10^6*u"m/s",
    #                   Note: reasonable velocity range in au
    :atomic_symbol   => [:H, :H, :C, :N, :He],
    :atomic_number   => [1, 1, 6, 7, 2],
    :charge          => [2, 1, 3.0, -1.0, 0.0]u"q",
    :atomic_mass     => 10rand(n_atoms)u"u",
    :vdw_radius      => randn(n_atoms)u"Å",
    :covalent_radius => randn(n_atoms)u"Å",
    :magnetic_moment => [0.0, 0.0, 1.0, -1.0, 0.0],
    )
    sysprop = Dict{Symbol,Any}(
    :extra_data   => 42,
    :charge       => -1u"q",
    :multiplicity => 2,
    )

    for prop in drop_atprop
    pop!(atprop, prop)
    end
    for prop in drop_sysprop
    pop!(sysprop, prop)
    end
    sysprop = merge(sysprop, pairs(extra_sysprop))
    atprop  = merge(atprop,  pairs(extra_atprop))

    atoms = map(1:n_atoms) do i
    atargs = Dict(k => v[i] for (k, v) in pairs(atprop)
    if !(k in (:position, :velocity)))
    if haskey(atprop, :velocity)
    Atom(atprop[:atomic_symbol][i], atprop[:position][i], atprop[:velocity][i];
    atargs...)
    else
    Atom(atprop[:atomic_symbol][i], atprop[:position][i]; atargs...)
    end
    end
    if cellmatrix == :lower_triangular
    box = [[1.54732, -0.807289, -0.500870],
    [    0.0, 0.4654985, 0.5615117],
    [    0.0,       0.0, 0.7928950]]u"Å"
    elseif cellmatrix == :upper_triangular
    box = [[1.54732, 0.0, 0.0],
    [-0.807289, 0.4654985, 0.0],
    [-0.500870, 0.5615117, 0.7928950]]u"Å"
    elseif cellmatrix == :diagonal
    box = [[1.54732, 0.0, 0.0],
    [0.0, 0.4654985, 0.0],
    [0.0, 0.0, 0.7928950]]u"Å"
    else
    box = [[-1.50304, 0.850344, 0.717239],
    [ 0.36113, 0.008144, 0.814712],
    [ 0.06828, 0.381122, 0.129081]]u"Å"
    end
    bcs = [Periodic(), Periodic(), DirichletZero()]
    system = atomic_system(atoms, box, bcs; sysprop...)

    (; system, atoms, atprop=NamedTuple(atprop), sysprop=NamedTuple(sysprop), box, bcs)
end

@testset "AbstractSystem -> Molly System" begin
    system = make_test_system_molly().system
    test_approx_eq(system, System(system)l cellmatrix = :diagonal)
end

