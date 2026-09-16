# Python ASE calculator test
# Test values from ASE v3.22.1

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using Molly
using PythonCall

using Test

@testset "Python ASE MACE" begin
    mc = pyimport("mace.calculators")
    ase_calc = mc.mace_off(model="medium", device="cuda")

    atoms = fill(Atom(mass=(14.0 / Unitful.Na)u"g/mol", charge=0.0), 2)
    coords = [SVector(2.0, 2.0, 1.0)u"Å", SVector(2.0, 2.0, 2.4)u"Å"]
    boundary = CubicBoundary(4.0u"Å")
    atoms_data = fill(AtomData(element="N"), 2)

    calc = ASECalculator(
        ase_calc=ase_calc,
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        atoms_data=atoms_data,
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        atoms_data=atoms_data,
        general_inters=(calc,),
        force_units=u"eV/Å",
        energy_units=u"eV",
    )

    @test potential_energy(sys) ≈ -2978.10774299578u"eV"
    @test forces(sys)[1][3] ≈ 12.0333717u"eV/Å"

    sim = SteepestDescentMinimizer(;
        step_size=0.1u"Å",
        max_steps=1_000,
        tol=5.0u"eV/Å",
    )

    simulate!(sys, sim)

    @test potential_energy(sys) < -2979.0u"eV"
end

@testset "Python ASE psi4" begin
    build = pyimport("ase.build")
    psi4 = pyimport("ase.calculators.psi4")

    py_atoms = build.molecule("H2O")
    ase_calc = psi4.Psi4(
        atoms=py_atoms,
        method="b3lyp",
        basis="6-311g_d_p_",
    )

    atoms = [Atom(mass=16.0u"u"), Atom(mass=1.0u"u"), Atom(mass=1.0u"u")]
    coords = SVector{3, Float64}.(eachrow(pyconvert(Matrix, py_atoms.get_positions()))) * u"Å"
    boundary = CubicBoundary(100.0u"Å")

    calc = ASECalculator(
        ase_calc=ase_calc,
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        elements=["O", "H", "H"],
    )

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        general_inters=(calc,),
        energy_units=u"eV",
        force_units=u"eV/Å",
    )

    @test potential_energy(sys) ≈ -2080.2391023909u"eV"
end
