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

@testset "Python ASE virial" begin
    # The ASE Lennard-Jones calculator should match the Molly one in a triclinic box
    lj = pyimport("ase.calculators.lj")
    n_atoms = 50
    boundary = TriclinicBoundary(SVector(18.0, 0.0, 0.0)u"Å", SVector(3.0, 17.0, 0.0)u"Å",
                                 SVector(2.0, -2.0, 18.0)u"Å")
    coords = place_atoms(n_atoms, boundary; min_dist=3.0u"Å")
    
    sigma, epsilon, rc = 3.4, 0.0104, 7.0
    atoms = fill(Atom(mass=39.948u"u", σ=sigma*u"Å", ϵ=epsilon*u"eV"), n_atoms)

    calc = ASECalculator(
        ase_calc=lj.LennardJones(; sigma, epsilon, rc),
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        elements=fill("Ar", n_atoms),
    )

    sys_ase = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        general_inters=(calc,),
        energy_units=u"eV",
        force_units=u"eV/Å",
    )

    sys_molly = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(LennardJones(cutoff=ShiftedPotentialCutoff(rc*u"Å")),),
        energy_units=u"eV",
        force_units=u"eV/Å",
    )

    @test potential_energy(sys_ase) ≈ potential_energy(sys_molly)
    @test ustrip_vec.(u"eV/Å", forces(sys_ase)) ≈ ustrip_vec.(u"eV/Å", forces(sys_molly))
    @test virial(sys_ase) ≈ virial(sys_molly)

    # Calculators that do not provide the stress, such as TIP3P, give no virial contribution
    tip3p = pyimport("ase.calculators.tip3p")
    atoms_w = repeat([Atom(mass=16.0u"u"), Atom(mass=1.0u"u"), Atom(mass=1.0u"u")], 2)
    coords_w = [SVector(0.0, 0.0, 0.0), SVector(0.96, 0.0, 0.0), SVector(-0.24, 0.93, 0.0),
                SVector(0.0, 0.0, 3.0), SVector(0.96, 0.0, 3.0), SVector(-0.24, 0.93, 3.0)]u"Å"
    boundary_w = CubicBoundary(Inf * u"Å")

    calc_w = ASECalculator(
        ase_calc=tip3p.TIP3P(),
        atoms=atoms_w,
        coords=coords_w,
        boundary=boundary_w,
        elements=["O", "H", "H", "O", "H", "H"],
    )

    sys_w = System(
        atoms=atoms_w,
        coords=coords_w,
        boundary=boundary_w,
        general_inters=(calc_w,),
        energy_units=u"eV",
        force_units=u"eV/Å",
    )

    @test iszero(virial(sys_w; strictness=:nowarn))
    @test_throws ErrorException virial(sys_w; strictness=:error)
end
