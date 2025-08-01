@testset "Interactions" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(1.4, 1.0, 1.0)u"nm"
    a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    a2 = Atom(charge=1.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1")
    boundary = CubicBoundary(2.0u"nm")
    dr12 = vector(c1, c2, boundary)
    dr13 = vector(c1, c3, boundary)

    @test Molly.lorentz_σ_mixing(a1, a2)        ≈ 0.25u"nm"
    @test Molly.lorentz_ϵ_mixing(a1, a2)        ≈ 0.15u"kJ * mol^-1"
    @test Molly.geometric_σ_mixing(a1, a2)      ≈ 0.2449489742783178u"nm"
    @test Molly.geometric_ϵ_mixing(a1, a2)      ≈ 0.14142135623730953u"kJ * mol^-1"
    @test Molly.waldman_hagler_σ_mixing(a1, a2) ≈ 0.271044458112581u"nm"
    @test Molly.waldman_hagler_ϵ_mixing(a1, a2) ≈ 0.07704164677744986u"kJ * mol^-1"
    @test Molly.fender_halsey_ϵ_mixing(a1, a2)  ≈ 0.13333333333333333u"kJ * mol^-1"

    @test !use_neighbors(LennardJones())
    @test  use_neighbors(LennardJones(use_neighbors=true))

    for inter in (LennardJones(), Mie(m=6, n=12), LennardJonesSoftCore(α=1, λ=0, p=2))
        @test isapprox(
            force(inter, dr12, a1, a1),
            SVector(16.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            force(inter, dr13, a1, a1),
            SVector(-1.375509739, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            potential_energy(inter, dr12, a1, a1),
            0.0u"kJ * mol^-1";
            atol=1e-9u"kJ * mol^-1",
        )
        @test isapprox(
            potential_energy(inter, dr13, a1, a1),
            -0.1170417309u"kJ * mol^-1";
            atol=1e-9u"kJ * mol^-1",
        )
    end

    inter = LennardJonesSoftCore(α=1, λ=0.5, p=2)
    @test isapprox(
        force(inter, dr12, a1, a1),
        SVector(6.144, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(-1.290499537, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr12, a1, a1),
        -0.128u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        -0.1130893709u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    AH_a1 = Molly.AshbaughHatchAtom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)
    inter = AshbaughHatch(weight_special=0.5)
    @test isapprox(
        force(inter, dr12, AH_a1, AH_a1, boundary),
        SVector(16.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, AH_a1, AH_a1, boundary),
        SVector(-1.375509739, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr12, AH_a1, AH_a1, boundary),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, AH_a1, AH_a1, boundary),
        -0.1170417309u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    AH_a1 = Molly.AshbaughHatchAtom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.5)
    @test isapprox(
        potential_energy(inter, dr13, AH_a1, AH_a1),
        -0.058520865u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        force(inter, dr13, AH_a1, AH_a1),
        SVector(-0.68775486946, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    c4 = SVector(1.28, 1.0, 1.0)u"nm"
    dr14 = vector(c1, c4, boundary)
    @test isapprox(
        potential_energy(inter, dr14, AH_a1, AH_a1),
        0.7205987916u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        force(inter, dr14, AH_a1, AH_a1),
        SVector(52.5306754422, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr14,  AH_a1, AH_a1, u"kJ * mol^-1 * nm^-1", true),
        0.5 * 0.7205987916u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        force(inter, dr14,  AH_a1, AH_a1, u"kJ * mol^-1 * nm^-1", true),
        SVector(0.5 * 52.5306754422, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    inter = SoftSphere()
    @test isapprox(
        force(inter, dr12, a1, a1),
        SVector(32.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(0.7602324486, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr12, a1, a1),
        0.8u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        0.0253410816u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    struct BuckinghamAtom{M, TA, TB, TC}
        mass::M
        A::TA
        B::TB
        C::TC
    end

    buck_c1 = SVector(10.0, 10.0, 10.0)u"Å"
    buck_c2 = SVector(13.0, 10.0, 10.0)u"Å"
    buck_c3 = SVector(14.0, 10.0, 10.0)u"Å"
    buck_a1 = BuckinghamAtom(1.0u"g/mol", 1400.0u"eV", 2.8u"Å^-1", 180.0u"eV * Å^6")
    buck_boundary = CubicBoundary(20.0u"Å")
    buck_dr12 = vector(buck_c1, buck_c2, buck_boundary)
    buck_dr13 = vector(buck_c1, buck_c3, buck_boundary)

    inter = Buckingham()
    @test isapprox(
        force(inter, buck_dr12, buck_a1, buck_a1, u"eV * Å^-1"),
        SVector(0.3876527503, 0.0, 0.0)u"eV * Å^-1";
        atol=1e-9u"eV * Å^-1",
    )
    @test isapprox(
        force(inter, buck_dr13, buck_a1, buck_a1, u"eV * Å^-1"),
        SVector(-0.0123151202, 0.0, 0.0)u"eV * Å^-1";
        atol=1e-9u"eV * Å^-1",
    )
    @test isapprox(
        potential_energy(inter, buck_dr12, buck_a1, buck_a1, u"eV"),
        0.0679006736u"eV";
        atol=1e-9u"eV",
    )
    @test isapprox(
        potential_energy(inter, buck_dr13, buck_a1, buck_a1, u"eV"),
        -0.0248014380u"eV";
        atol=1e-9u"eV",
    )

    for inter in (Coulomb(), CoulombSoftCore(α=1, λ=0, p=2))
        @test isapprox(
            force(inter, dr12, a1, a1),
            SVector(1543.727311, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
            atol=1e-5u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            force(inter, dr13, a1, a1),
            SVector(868.3466125, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
            atol=1e-5u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            potential_energy(inter, dr12, a1, a1),
            463.1181933u"kJ * mol^-1";
            atol=1e-5u"kJ * mol^-1",
        )
        @test isapprox(
            potential_energy(inter, dr13, a1, a1),
            347.338645u"kJ * mol^-1";
            atol=1e-5u"kJ * mol^-1",
        )
    end

    inter = CoulombSoftCore(α=1, λ=0.5, p=2)
    @test isapprox(
        force(inter, dr12, a1, a1),
        SVector(1189.895726, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(825.3456507, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr12, a1, a1),
        446.2108973u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        344.8276396u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )

    inter = Yukawa(; weight_special=0.5)
    @test isapprox(
        force(inter, dr12, a1, a1),
        SVector(1486.7077156786308, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(814.8981977722481, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, a1, a1, u"kJ * mol^-1 * nm^-1", true),
        SVector(0.5 * 814.8981977722481, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr12, a1, a1),
        343.08639592583785u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        232.8280565063u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1, u"kJ * mol^-1 * nm^-1", true),
        0.5 * 232.8280565063u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )
    c1_grav = SVector(1.0, 1.0, 1.0)u"m"
    c2_grav = SVector(6.0, 1.0, 1.0)u"m"
    a1_grav, a2_grav = Atom(mass=1e6u"kg"), Atom(mass=1e5u"kg")
    boundary_grav = CubicBoundary(20.0u"m")
    dr12_grav = vector(c1_grav, c2_grav, boundary_grav)
    inter = Gravity()
    @test isapprox(
        force(inter, dr12_grav, a1_grav, a2_grav),
        SVector(-0.266972, 0.0, 0.0)u"kg * m * s^-2";
        atol=1e-9u"kg * m * s^-2",
    )
    @test isapprox(
        potential_energy(inter, dr12_grav, a1_grav, a2_grav),
        -1.33486u"kg * m^2 * s^-2";
        atol=1e-9u"kg * m^2 * s^-2",
    )

    pr = HarmonicPositionRestraint(k=300_000.0u"kJ * mol^-1 * nm^-2", x0=c1)
    fs = force(pr, c2, boundary)
    @test isapprox(
        fs.f1,
        SVector(-90000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(pr, c1, boundary)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(pr, c2, boundary),
        13500.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(pr, c1, boundary),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    b1 = HarmonicBond(k=300_000.0u"kJ * mol^-1 * nm^-2", r0=0.2u"nm")
    b2 = HarmonicBond(k=100_000.0u"kJ * mol^-1 * nm^-2", r0=0.6u"nm")
    fs = force(b1, c1, c2, boundary)
    @test isapprox(
        fs.f1,
        SVector(30000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(-30000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(b2, c1, c3, boundary)
    @test isapprox(
        fs.f1,
        SVector(-20000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(20000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b1, c1, c2, boundary),
        1500.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(b2, c1, c3, boundary),
        2000.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    b1 = MorseBond(D=100.0u"kJ * mol^-1", a=10.0u"nm^-1", r0=0.2u"nm")
    b2 = MorseBond(D=200.0u"kJ * mol^-1", a=5.0u"nm^-1" , r0=0.6u"nm")
    fs = force(b1, c1, c2, boundary)
    @test isapprox(
        fs.f1,
        SVector(465.0883158697, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(-465.0883158697, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(b2, c1, c3, boundary)
    @test isapprox(
        fs.f1,
        SVector(-9341.5485409432, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(9341.5485409432, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b1, c1, c2, boundary),
        39.9576400894u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(b2, c1, c3, boundary),
        590.4984884025u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    boundary_fene = CubicBoundary(20.0u"nm")
    c1_fene = SVector(2.3, 0.0, 0.0)u"nm"
    c2_fene = SVector(1.0, 0.0, 0.0)u"nm"
    kbT = 2.479u"kJ * mol^-1"
    b1 = FENEBond(k=10.0u"nm^-2" * kbT, r0=1.6u"nm", σ=1.0u"nm", ϵ=kbT)
    b2 = FENEBond(k=0.0u"nm^-2"  * kbT, r0=1.6u"nm", σ=1.0u"nm", ϵ=kbT)
    fs = force(b1, c1_fene, c2_fene, boundary_fene)
    @test isapprox(
        fs.f1,
        SVector(-94.8288735632, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(94.8288735632, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b1, c1_fene, c2_fene, boundary_fene),
        34.2465108316u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    fs = force(b2, c1_fene, c2_fene, boundary_fene)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b2, c1_fene, c2_fene, boundary_fene),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    c3a = SVector(1.0, 1.2, 1.2)u"nm"
    a1 = HarmonicAngle(k=300.0u"kJ * mol^-1", θ0=0.8)
    fs = force(a1, c1, c2, c3a, boundary)
    @test isapprox(
        fs.f1,
        SVector(0.0, -31.1343284689, -31.1343284689)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(-21.9771730369, 14.6514486912, 14.6514486912)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f3,
        SVector(21.9771730369, 16.4828797777, 16.4828797777)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(a1, c1, c2, c3a, boundary),
        0.2908039228u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    boundary_cosine = CubicBoundary(10.0u"nm")
    c1_cosine = SVector(1.0, 0.0, 0.0)u"nm"
    c2_cosine = SVector(2.0, 0.0, 0.0)u"nm"
    c3_cosine = SVector(3.0, 0.0, 0.0)u"nm"
    c4_cosine = SVector(2.0, 1.0, 0.0)u"nm"
    ca1 = CosineAngle(k=10.0 * kbT, θ0=0.0)
    ca2 = CosineAngle(k=10.0 * kbT, θ0=π/2)
    fs = force(ca1, c1_cosine, c2_cosine, c3_cosine, boundary_cosine)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f3,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(ca2, c1_cosine, c2_cosine, c4_cosine, boundary_cosine)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f3,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(ca1, c1_cosine, c2_cosine, c3_cosine, boundary_cosine),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(ca2, c1_cosine, c2_cosine, c3_cosine, boundary_cosine),
        24.79u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(ca2, c1_cosine, c2_cosine, c4_cosine, boundary_cosine),
        49.58u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    a1 = UreyBradley(kangle=300.0u"kJ * mol^-1", θ0=0.8,
                     kbond=10_000.0u"kJ * mol^-1 * nm^-2", r0=0.3u"nm")
    fs = force(a1, c1, c2, c3a, boundary)
    @test isapprox(
        fs.f1,
        SVector(0.0, -152.4546720285, -152.4546720285)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(-21.9771730369, 14.6514486912, 14.6514486912)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f3,
        SVector(21.9771730369, 137.8032233372, 137.8032233372)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(a1, c1, c2, c3a, boundary),
        1.7626664989u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    inter = MullerBrown()
    local_min_1 = SVector(  0.6234994049304005, 0.028037758528718367)u"nm"
    local_min_2 = SVector(-0.05001082299878202,  0.46669410487256247)u"nm"

    @test isapprox(
        Molly.force_muller_brown(inter, local_min_1),
        SVector(0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    @test isapprox(
        Molly.force_muller_brown(inter, local_min_2),
        SVector(0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )

    @test isapprox(
        Molly.potential_muller_brown(inter, local_min_1),
        -108.166724117u"kJ * mol^-1";
        atol=1e-7u"kJ * mol^-1",
    )

    do_shortcut(atom_i, atom_j) = true

    for inter in (
            LennardJones(; shortcut=do_shortcut),
            LennardJonesSoftCore(α=1, λ=0, p=2; shortcut=do_shortcut),
            AshbaughHatch(; shortcut=do_shortcut),
            SoftSphere(; shortcut=do_shortcut),
            Mie(m=6, n=12; shortcut=do_shortcut),
            Buckingham(; shortcut=do_shortcut),
        )

        @test isapprox(
            force(inter, dr12, a1, a1),
            SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )

        @test isapprox(
            potential_energy(inter, dr12, a1, a1),
            0.0u"kJ * mol^-1";
            atol=1e-9u"kJ * mol^-1",
        )
    end
end

@testset "Ewald" begin
    dist_cutoff = 0.9u"nm"
    E_openmm = -5.465127432466375u"kJ/mol"
    Fs_openmm = [
        SVector(-72.48152122617766, 5.6452093242736225 , 101.4156707298087  ),
        SVector(17.520231752234416, 4.071455080698861  , -37.701631053185295),
        SVector(30.858153727989023, -12.062341554089436, -32.14366235405959 ),
        SVector(-7.936279084919704, -14.215671548792962, -8.295642564943837 ),
        SVector(2.4095151618606145, 7.275822557366837  , 4.433671630065675  ),
        SVector(7.141770437453555 , 8.540348761741292  , 5.30999589638612   ),
        SVector(-97.27674352036883, 14.881678867954054 , 63.35431221886955  ),
        SVector(48.485910228223275, 4.532352998517133  , -21.51089738652309 ),
        SVector(71.2789625237053  , -18.668854487669485, -74.8618171164182  ),
    ] * u"kJ * mol^-1 * nm^-1"

    for AT in array_list
        for n_threads in n_threads_list
            for T in (Float64, Float32)
                if n_threads > 1 && AT != Array
                    continue
                end
                ff = MolecularForceField(T, joinpath(ff_dir, "tip3p_standard.xml"))
                sys_init = System(
                    joinpath(data_dir, "water_3mol_cubic.pdb"),
                    ff;
                    array_type=AT,
                    dist_cutoff=T(dist_cutoff),
                    dist_neighbors=T(dist_cutoff),
                    nonbonded_method="ewald",
                    center_coords=false,
                )
                sys = System(
                    sys_init;
                    pairwise_inters=(sys_init.pairwise_inters[2],),
                    specific_inter_lists=(),
                )

                @test potential_energy(sys; n_threads=n_threads) ≈ E_openmm atol=1e-4u"kJ/mol"
                fs = from_device(forces(sys; n_threads=n_threads))
                @test maximum(norm.(fs .- Fs_openmm)) < 5e-4u"kJ * mol^-1 * nm^-1"
            end
        end
    end

    pme_data = (
        (
            "water_3mol_cubic.pdb",
            -5.460124320435284u"kJ/mol",
            [
                SVector(-72.57603365363543, 5.648072796188359  , 101.40821248959712 ),
                SVector(17.558243038254187, 4.075128117683555  , -37.70060863840432 ),
                SVector(30.881405092779705, -12.047169393065978, -32.137723916688024),
                SVector(-7.789998310481266, -14.185855369417702, -8.35080870148926  ),
                SVector(2.3519124244832277, 7.264285806008946  , 4.431212066763443  ),
                SVector(7.085282096874462 , 8.530075688459654  , 5.32165402278671   ),
                SVector(-97.20750157586099, 14.85484666061426  , 63.32187921636768  ),
                SVector(48.50069206640984 , 4.544995194749845  , -21.497171353580004),
                SVector(71.21703702929426 , -18.67010037709364 , -74.8362731945127  ),
            ] * u"kJ * mol^-1 * nm^-1",
        ),
        (
            "water_3mol_triclinic.pdb",
            -5.461196031062514u"kJ/mol",
            [
                SVector(-72.42120264368016 , 5.691981530694477  , 101.42104318240557 ),
                SVector(17.479150437776987 , 4.0540370559245105 , -37.70340648054405 ),
                SVector(30.81579291744146  , -12.071913504082112, -32.146120279797024),
                SVector(-7.9206682279130405, -14.187409961603702, -8.364883441632035 ),
                SVector(2.3887077140251414 , 7.267025286293812  , 4.440580554656442  ),
                SVector(7.142699528225474  , 8.538462949340726  , 5.330171779520562  ),
                SVector(-97.10424848645062 , 14.864897047240834 , 63.32009574641273  ),
                SVector(48.459298786113976 , 4.530578179190741  , -21.4941360532105  ),
                SVector(71.12951420225025  , -18.681760708802052, -74.84152091219767 ),
            ] * u"kJ * mol^-1 * nm^-1",
        ),
    )

    for (pdb_fp, E_openmm, Fs_openmm) in pme_data
        for AT in (Array,)
            for n_threads in n_threads_list
                for T in (Float64, Float32)
                    if n_threads > 1 && AT != Array
                        continue
                    end
                    ff = MolecularForceField(T, joinpath(ff_dir, "tip3p_standard.xml"))
                    sys_init = System(
                        joinpath(data_dir, pdb_fp),
                        ff;
                        array_type=AT,
                        dist_cutoff=T(dist_cutoff),
                        dist_neighbors=T(dist_cutoff),
                        nonbonded_method="pme",
                        center_coords=false,
                    )
                    sys = System(
                        sys_init;
                        pairwise_inters=(sys_init.pairwise_inters[2],),
                        specific_inter_lists=(),
                    )

                    @test potential_energy(sys; n_threads=n_threads) ≈ E_openmm atol=1e-4u"kJ/mol"
                    fs = from_device(forces(sys; n_threads=n_threads))
                    @test maximum(norm.(fs .- Fs_openmm)) < 5e-4u"kJ * mol^-1 * nm^-1"
                end
            end
        end
    end
end
