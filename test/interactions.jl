@testset "Interactions" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(1.4, 1.0, 1.0)u"nm"
    a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm")
    dr12 = vector(c1, c2, boundary)
    dr13 = vector(c1, c3, boundary)

    for inter in (LennardJones(), Mie(m=6, n=12))
        @test isapprox(
            force(inter, dr12, c1, c2, a1, a1, boundary),
            SVector(16.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            force(inter, dr13, c1, c3, a1, a1, boundary),
            SVector(-1.375509739, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            potential_energy(inter, dr12, c1, c2, a1, a1, boundary),
            0.0u"kJ * mol^-1",
            atol=1e-9u"kJ * mol^-1",
        )
        @test isapprox(
            potential_energy(inter, dr13, c1, c3, a1, a1, boundary),
            -0.1170417309u"kJ * mol^-1",
            atol=1e-9u"kJ * mol^-1",
        )
    end

    inter = SoftSphere()
    @test isapprox(
        force(inter, dr12, c1, c2, a1, a1, boundary),
        SVector(32.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, c1, c3, a1, a1, boundary),
        SVector(0.7602324486, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr12, c1, c2, a1, a1, boundary),
        0.8u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, c1, c3, a1, a1, boundary),
        0.0253410816u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )

    inter = Coulomb()
    @test isapprox(
        force(inter, dr12, c1, c2, a1, a1, boundary),
        SVector(1543.727311, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, c1, c3, a1, a1, boundary),
        SVector(868.3466125, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr12, c1, c2, a1, a1, boundary),
        463.1181933u"kJ * mol^-1",
        atol=1e-5u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, c1, c3, a1, a1, boundary),
        347.338645u"kJ * mol^-1",
        atol=1e-5u"kJ * mol^-1",
    )

    c1_grav = SVector(1.0, 1.0, 1.0)u"m"
    c2_grav = SVector(6.0, 1.0, 1.0)u"m"
    a1_grav, a2_grav = Atom(mass=1e6u"kg"), Atom(mass=1e5u"kg")
    boundary_grav = CubicBoundary(20.0u"m", 20.0u"m", 20.0u"m")
    dr12_grav = vector(c1_grav, c2_grav, boundary_grav)
    inter = Gravity()
    @test isapprox(
        force(inter, dr12_grav, c1_grav, c2_grav, a1_grav, a2_grav, boundary_grav),
        SVector(-0.266972, 0.0, 0.0)u"kg * m * s^-2",
        atol=1e-9u"kg * m * s^-2",
    )
    @test isapprox(
        potential_energy(inter, dr12_grav, c1_grav, c2_grav,
                         a1_grav, a2_grav, boundary_grav),
        -1.33486u"kg * m^2 * s^-2",
        atol=1e-9u"kg * m^2 * s^-2",
    )

    pr = HarmonicPositionRestraint(k=300_000.0u"kJ * mol^-1 * nm^-2", x0=c1)
    fs = force(pr, c2, boundary)
    @test isapprox(
        fs.f1,
        SVector(-90000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(pr, c1, boundary)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(pr, c2, boundary),
        13500.0u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(pr, c1, boundary),
        0.0u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )

    b1 = HarmonicBond(k=300_000.0u"kJ * mol^-1 * nm^-2", r0=0.2u"nm")
    b2 = HarmonicBond(k=100_000.0u"kJ * mol^-1 * nm^-2", r0=0.6u"nm")
    fs = force(b1, c1, c2, boundary)
    @test isapprox(
        fs.f1,
        SVector(30000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(-30000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(b2, c1, c3, boundary)
    @test isapprox(
        fs.f1,
        SVector(-20000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(20000.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b1, c1, c2, boundary),
        1500.0u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(b2, c1, c3, boundary),
        2000.0u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )

    b1 = MorseBond(D=100.0u"kJ * mol^-1", a=10.0u"nm^-1", r0=0.2u"nm")
    b2 = MorseBond(D=200.0u"kJ * mol^-1", a=5.0u"nm^-1" , r0=0.6u"nm")
    fs = force(b1, c1, c2, boundary)
    @test isapprox(
        fs.f1,
        SVector(465.0883158697, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(-465.0883158697, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(b2, c1, c3, boundary)
    @test isapprox(
        fs.f1,
        SVector(-9341.5485409432, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(9341.5485409432, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b1, c1, c2, boundary),
        39.9576400894u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(b2, c1, c3, boundary),
        590.4984884025u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )

    boundary_fene = CubicBoundary(20.0u"nm", 20.0u"nm", 20.0u"nm")
    c1_fene = SVector(2.3, 0.0, 0.0)u"nm"
    c2_fene = SVector(1.0, 0.0, 0.0)u"nm"
    kbT = 2.479u"kJ * mol^-1"
    b1 = FENEBond(k=10.0u"nm^-2" * kbT, r0=1.6u"nm", σ=1.0u"nm", ϵ=kbT)
    b2 = FENEBond(k=0.0u"nm^-2"  * kbT, r0=1.6u"nm", σ=1.0u"nm", ϵ=kbT)
    fs = force(b1, c1_fene, c2_fene, boundary_fene)
    @test isapprox(
        fs.f1,
        SVector(-94.8288735632, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(94.8288735632, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b1, c1_fene, c2_fene, boundary_fene),
        34.2465108316u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
    fs = force(b2, c1_fene, c2_fene, boundary_fene)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b2, c1_fene, c2_fene, boundary_fene),
        0.0u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )

    boundary_cosine = CubicBoundary(10.0u"nm", 10.0u"nm", 10.0u"nm")
    c1_cosine = SVector(1.0, 0.0, 0.0)u"nm"
    c2_cosine = SVector(2.0, 0.0, 0.0)u"nm"
    c3_cosine = SVector(3.0, 0.0, 0.0)u"nm"
    c4_cosine = SVector(2.0, 1.0, 0.0)u"nm"
    a1 = CosineAngle(10.0 * kbT, 0.0)
    a2 = CosineAngle(10.0 * kbT, π/2)
    fs = force(a1, c1_cosine, c2_cosine, c3_cosine, boundary_cosine)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f3,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    fs = force(a2, c1_cosine, c2_cosine, c4_cosine, boundary_cosine)
    @test isapprox(
        fs.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f2,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        fs.f3,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1",
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(a1, c1_cosine, c2_cosine, c3_cosine, boundary_cosine),
        0.0u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(a2, c1_cosine, c2_cosine, c3_cosine, boundary_cosine),
        24.79u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(a2, c1_cosine, c2_cosine, c4_cosine, boundary_cosine),
        49.58u"kJ * mol^-1",
        atol=1e-9u"kJ * mol^-1",
    )
end
