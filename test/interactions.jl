@testset "Interactions" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(1.4, 1.0, 1.0)u"nm"
    c4 = SVector(1.1, 1.0, 1.0)u"nm"
    a1 = Atom(atom_type=1, charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)
    a2 = Atom(atom_type=2, charge=1.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0)
    a3 = Atom(atom_type=3, charge=1.0, σ=0.4u"nm", ϵ=0.4u"kJ * mol^-1", λ=1.0)
    boundary = CubicBoundary(2.0u"nm")
    dr12 = vector(c1, c2, boundary)
    dr13 = vector(c1, c3, boundary)
    dr14 = vector(c1, c4, boundary)

    @test Molly.σ_mixing(Molly.LorentzMixing(), a1, a2)       ≈ 0.25u"nm"
    @test Molly.ϵ_mixing(Molly.LorentzMixing(), a1, a2)       ≈ 0.15u"kJ * mol^-1"
    @test Molly.σ_mixing(Molly.GeometricMixing(), a1, a2)     ≈ 0.2449489742783178u"nm"
    @test Molly.ϵ_mixing(Molly.GeometricMixing(), a1, a2)     ≈ 0.14142135623730953u"kJ * mol^-1"
    @test Molly.σ_mixing(Molly.WaldmanHaglerMixing(), a1, a2) ≈ 0.271044458112581u"nm"
    @test Molly.ϵ_mixing(Molly.WaldmanHaglerMixing(), a1, a2) ≈ 0.07704164677744986u"kJ * mol^-1"
    @test Molly.ϵ_mixing(Molly.FenderHalseyMixing(), a1, a2)  ≈ 0.13333333333333333u"kJ * mol^-1"

    σ_exceptions_dict = Dict((2, 1) => 0.5u"nm", (3, 3) => 0.6u"nm")
    σ_exceptions_static = Molly.ExceptionList(
        SVector{2}([(2, 1), (3, 3)]),
        SVector(0.5u"nm", 0.6u"nm"),
    )
    lorentz_mixing_dict = Molly.MixingException(Molly.LorentzMixing(), σ_exceptions_dict)
    lorentz_mixing_static = Molly.MixingException(Molly.LorentzMixing(), σ_exceptions_static)
    @test Molly.σ_mixing(lorentz_mixing_dict  , a1, a2) ≈ 0.5u"nm"
    @test Molly.σ_mixing(lorentz_mixing_dict  , a1, a3) ≈ 0.35u"nm"
    @test Molly.σ_mixing(lorentz_mixing_dict  , a3, a3) ≈ 0.6u"nm"
    @test Molly.σ_mixing(lorentz_mixing_static, a1, a2) ≈ 0.5u"nm"
    @test Molly.σ_mixing(lorentz_mixing_static, a1, a3) ≈ 0.35u"nm"
    @test Molly.σ_mixing(lorentz_mixing_static, a3, a3) ≈ 0.6u"nm"
    eld = Molly.ExceptionList(σ_exceptions_dict)
    @test (eld.keys == SVector{2}([(2, 1), (3, 3)]) && eld.values == SVector(0.5u"nm", 0.6u"nm")) ||
          (eld.keys == SVector{2}([(3, 3), (2, 1)]) && eld.values == SVector(0.6u"nm", 0.5u"nm"))

    ϵ_exceptions_dict = Dict((2, 1) => 0.5u"kJ * mol^-1", (3, 3) => 0.6u"kJ * mol^-1")
    ϵ_exceptions_static = Molly.ExceptionList(
        SVector{2}([(2, 1), (3, 3)]),
        SVector(0.5u"kJ * mol^-1", 0.6u"kJ * mol^-1"),
    )
    geometric_mixing_dict = Molly.MixingException(Molly.GeometricMixing(), ϵ_exceptions_dict)
    geometric_mixing_static = Molly.MixingException(Molly.GeometricMixing(), ϵ_exceptions_static)
    @test Molly.ϵ_mixing(geometric_mixing_dict  , a1, a2) ≈ 0.5u"kJ * mol^-1"
    @test Molly.ϵ_mixing(geometric_mixing_dict  , a1, a3) ≈ 0.28284271247461906u"kJ * mol^-1"
    @test Molly.ϵ_mixing(geometric_mixing_dict  , a3, a3) ≈ 0.6u"kJ * mol^-1"
    @test Molly.ϵ_mixing(geometric_mixing_static, a1, a2) ≈ 0.5u"kJ * mol^-1"
    @test Molly.ϵ_mixing(geometric_mixing_static, a1, a3) ≈ 0.28284271247461906u"kJ * mol^-1"
    @test Molly.ϵ_mixing(geometric_mixing_static, a3, a3) ≈ 0.6u"kJ * mol^-1"

    @test !use_neighbors(LennardJones())
    @test  use_neighbors(LennardJones(use_neighbors=true))
    @test !use_neighbors(DoubleExponential())
    @test  use_neighbors(DoubleExponential(use_neighbors=true))
    @test !use_neighbors(DoubleExponentialSoftCore())
    @test  use_neighbors(DoubleExponentialSoftCore(use_neighbors=true))

    for inter in (LennardJones(), Mie(m=6, n=12), LennardJonesSoftCoreBeutler(α=1), LennardJonesSoftCoreGapsys(α=1))
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

    for inter in (DoubleExponential(), DoubleExponentialSoftCore())
        @test isapprox(
            force(inter, dr12, a1, a1),
            SVector(8.635969842730873, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            force(inter, dr13, a1, a1),
            SVector(-1.3633017400293386, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            potential_energy(inter, dr12, a1, a1),
            -0.08148402650292114u"kJ * mol^-1";
            atol=1e-9u"kJ * mol^-1",
        )
        @test isapprox(
            potential_energy(inter, dr13, a1, a1),
            -0.12648034169670871u"kJ * mol^-1";
            atol=1e-9u"kJ * mol^-1",
        )
    end

    inter = DoubleExponential(weight_special=0.5)
    @test isapprox(
        force(inter, dr13, a1, a1, u"kJ * mol^-1 * nm^-1", true),
        SVector(-0.5 * 1.3633017400293386, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1, u"kJ * mol^-1", true),
        -0.5 * 0.12648034169670871u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    inter = Molly.LennardJones14(0.3u"nm", 0.2u"kJ * mol^-1", 1)
    @test isapprox(
        force(inter, c1, c2, boundary).f2,
        SVector(16.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, c1, c3, boundary).f2,
        SVector(-1.375509739, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, c1, c2, boundary),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, c1, c3, boundary),
        -0.1170417309u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ = 0.5)
    a2 = Atom(charge=1.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1", λ = 0.5)
    inter = LennardJonesSoftCoreBeutler(α=0.3)
    @test isapprox(
        force(inter, dr14, a1, a1),
        SVector(17.546838269368916, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(-0.6618297363923222, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr14, a1, a1),
        14.814529458827394u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        -0.05732007116542457u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    inter = LennardJonesSoftCoreGapsys(α=0.85)
    @test isapprox(
        force(inter, dr14, a1, a1),
        SVector(258.42288793054365, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(-0.68775486946106, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr14, a1, a1),
        25.90746475905439u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        -0.0585208654403687u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    @testset "Default decoupled LJ sterics" begin
        alch_i = Atom(
            charge=1.0,
            σ=0.3u"nm",
            ϵ=0.2u"kJ * mol^-1",
            λ=0.25,
            alch_role=Molly.InsertRole,
        )
        alch_j = Atom(
            charge=1.0,
            σ=0.3u"nm",
            ϵ=0.2u"kJ * mol^-1",
            λ=0.25,
            alch_role=Molly.InsertRole,
        )
        off_alch = Atom(
            charge=1.0,
            σ=0.3u"nm",
            ϵ=0.2u"kJ * mol^-1",
            λ=0.0,
            alch_role=Molly.InsertRole,
        )
        core = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)
        ref_inter = LennardJones()

        for inter in (LennardJonesSoftCoreBeutler(α=0.3), LennardJonesSoftCoreGapsys(α=0.85))
            @test isapprox(
                force(inter, dr13, alch_i, alch_j),
                force(ref_inter, dr13, alch_i, alch_j);
                atol=1e-9u"kJ * mol^-1 * nm^-1",
            )
            @test isapprox(
                potential_energy(inter, dr13, alch_i, alch_j),
                potential_energy(ref_inter, dr13, alch_i, alch_j);
                atol=1e-9u"kJ * mol^-1",
            )
            @test all(iszero, force(inter, dr13, off_alch, core))
            @test iszero(potential_energy(inter, dr13, off_alch, core))
        end
    end

    AH_a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ = 1.0)
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

    AH_a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ = 0.5)
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

    inter = DoubleExponentialSoftCore(weight_special=0.5)
    @test isapprox(
        force(inter, dr13, AH_a1, AH_a1),
        SVector(-0.8421368684324054, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, AH_a1, AH_a1),
        -0.08202077553076385u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        force(inter, dr13, AH_a1, AH_a1, u"kJ * mol^-1 * nm^-1", true),
        SVector(-0.5 * 0.8421368684324054, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, AH_a1, AH_a1, u"kJ * mol^-1", true),
        -0.5 * 0.08202077553076385u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    AH_off = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.0)
    @test all(iszero, force(inter, dr13, AH_off, AH_a1))
    @test iszero(potential_energy(inter, dr13, AH_off, AH_a1))

    inter = AshbaughHatch(weight_special=0.5)
    c5 = SVector(1.28, 1.0, 1.0)u"nm"
    dr15 = vector(c1, c5, boundary)
    @test isapprox(
        potential_energy(inter, dr15, AH_a1, AH_a1),
        0.7205987916u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        force(inter, dr15, AH_a1, AH_a1),
        SVector(52.5306754422, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr15,  AH_a1, AH_a1, u"kJ * mol^-1 * nm^-1", true),
        0.5 * 0.7205987916u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        force(inter, dr15,  AH_a1, AH_a1, u"kJ * mol^-1 * nm^-1", true),
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


    a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ = 1.0)
    a2 = Atom(charge=1.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1", λ = 1.0)
    for inter in (Coulomb(), CoulombSoftCoreBeutler(α=1), CoulombSoftCoreGapsys(α=1, σQ=1u"nm"))
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

    @testset "Scaled Coulomb matches pre-scaled charges" begin
        λ_state = 0.75
        scheduler = Molly.DefaultLambdaScheduler()
        λ_elec = Molly.scale_elec(scheduler, λ_state, Molly.InsertRole)
        raw_i = Atom(charge=1.0, λ=λ_state, alch_role=Molly.InsertRole)
        raw_j = Atom(charge=-0.8, λ=λ_state, alch_role=Molly.InsertRole)
        ref_i = Atom(charge=1.0 * λ_elec)
        ref_j = Atom(charge=-0.8 * λ_elec)
        rc_test = 1.0u"nm"

        for (scaled_inter, ref_inter) in (
            (CoulombScaled(scheduler=scheduler), Coulomb()),
            (CoulombReactionFieldScaled(dist_cutoff=rc_test, scheduler=scheduler),
             CoulombReactionField(dist_cutoff=rc_test)),
            (CoulombEwaldScaled(dist_cutoff=rc_test, scheduler=scheduler),
             CoulombEwald(dist_cutoff=rc_test)),
        )
            @test isapprox(force(scaled_inter, dr12, raw_i, raw_j),
                           force(ref_inter, dr12, ref_i, ref_j);
                           atol=1e-9u"kJ * mol^-1 * nm^-1")
            @test isapprox(potential_energy(scaled_inter, dr12, raw_i, raw_j),
                           potential_energy(ref_inter, dr12, ref_i, ref_j);
                           atol=1e-9u"kJ * mol^-1")
            @test isapprox(force(scaled_inter, dr12, raw_i, raw_j,
                                 u"kJ * mol^-1 * nm^-1", true),
                           force(ref_inter, dr12, ref_i, ref_j,
                                 u"kJ * mol^-1 * nm^-1", true);
                           atol=1e-9u"kJ * mol^-1 * nm^-1")
            @test isapprox(potential_energy(scaled_inter, dr12, raw_i, raw_j,
                                             u"kJ * mol^-1", true),
                           potential_energy(ref_inter, dr12, ref_i, ref_j,
                                            u"kJ * mol^-1", true);
                           atol=1e-9u"kJ * mol^-1")
        end

        off_i = Atom(charge=1.0, λ=0.25, alch_role=Molly.InsertRole)
        off_j = Atom(charge=-0.8, λ=0.25, alch_role=Molly.InsertRole)
        dr_zero = zero(dr12)
        for scaled_inter in (
            CoulombScaled(scheduler=scheduler),
            CoulombReactionFieldScaled(dist_cutoff=rc_test, scheduler=scheduler),
            CoulombEwaldScaled(dist_cutoff=rc_test, scheduler=scheduler),
        )
            @test all(iszero, force(scaled_inter, dr12, off_i, off_j))
            @test iszero(potential_energy(scaled_inter, dr12, off_i, off_j))
            @test all(iszero, force(scaled_inter, dr_zero, raw_i, raw_j))
            @test iszero(potential_energy(scaled_inter, dr_zero, raw_i, raw_j))
        end

        scheduler = Molly.EleScaledLambdaScheduler()
        λ_elec = Molly.scale_elec(scheduler, λ_state, Molly.InsertRole)
        ref_i = Atom(charge=1.0 * λ_elec)
        ref_j = Atom(charge=-0.8 * λ_elec)
        scaled_inter = CoulombScaled(scheduler=scheduler)

        @test isapprox(force(scaled_inter, dr12, raw_i, raw_j),
                       force(Coulomb(), dr12, ref_i, ref_j);
                       atol=1e-9u"kJ * mol^-1 * nm^-1")
        @test isapprox(potential_energy(scaled_inter, dr12, raw_i, raw_j),
                       potential_energy(Coulomb(), dr12, ref_i, ref_j);
                       atol=1e-9u"kJ * mol^-1")
    end

    a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ = 0.5)
    a2 = Atom(charge=1.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1", λ = 0.5)
    inter = CoulombSoftCoreBeutler(α=0.3)
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(421.030817792505, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr14, a1, a1),
        SVector(28.74409943236674, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        172.90839351598729u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr14, a1, a1),
        317.1911372361652u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )

    inter = CoulombSoftCoreGapsys(α=0.3, σQ=1u"nm")
    @test isapprox(
        force(inter, dr13, a1, a1),
        SVector(365.5054328521848, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        force(inter, dr14, a1, a1),
        SVector(638.4004446424633, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-5u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(inter, dr13, a1, a1),
        170.90026962853818u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(inter, dr14, a1, a1),
        321.4861512527354u"kJ * mol^-1";
        atol=1e-5u"kJ * mol^-1",
    )

    @testset "Soft-core Reaction Field" begin
        rc_test = 1.0u"nm"
        dr_rc = SVector(rc_test, 0.0u"nm", 0.0u"nm")
        dr_beyond = SVector(1.5u"nm", 0.0u"nm", 0.0u"nm")
        crf_ref = CoulombReactionField(dist_cutoff=rc_test, solvent_dielectric=78.3)
        cscrfb = CoulombSoftCoreBeutlerReactionField(
            dist_cutoff=rc_test,
            solvent_dielectric=78.3,
            α=1.0,
        )
        cscrfg = CoulombSoftCoreGapsysReactionField(
            dist_cutoff=rc_test,
            solvent_dielectric=78.3,
            α=0.3,
            σQ=1.0u"nm",
        )

        a1_rf = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)
        a1_l0 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.0)
        a1_l05 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.5)

        @testset "lambda one matches reaction field" begin
            for dr_test in (dr12, dr13, dr14)
                ref_f = force(crf_ref, dr_test, a1_rf, a1_rf)
                ref_pe = potential_energy(crf_ref, dr_test, a1_rf, a1_rf)
                @test isapprox(force(cscrfb, dr_test, a1_rf, a1_rf), ref_f;
                               atol=1e-9u"kJ * mol^-1 * nm^-1")
                @test isapprox(force(cscrfg, dr_test, a1_rf, a1_rf), ref_f;
                               atol=1e-9u"kJ * mol^-1 * nm^-1")
                @test isapprox(potential_energy(cscrfb, dr_test, a1_rf, a1_rf), ref_pe;
                               atol=1e-9u"kJ * mol^-1")
                @test isapprox(potential_energy(cscrfg, dr_test, a1_rf, a1_rf), ref_pe;
                               atol=1e-9u"kJ * mol^-1")
            end
        end

        @testset "lambda zero vanishes" begin
            for inter_rf in (cscrfb, cscrfg)
                @test all(iszero, force(inter_rf, dr12, a1_l0, a1_l0))
                @test iszero(potential_energy(inter_rf, dr12, a1_l0, a1_l0))
            end
        end

        @testset "lambda half stays finite" begin
            for inter_rf in (cscrfb, cscrfg)
                for dr_test in (dr12, dr13, dr14)
                    f_val = force(inter_rf, dr_test, a1_l05, a1_l05)
                    pe_val = potential_energy(inter_rf, dr_test, a1_l05, a1_l05)
                    @test all(isfinite, f_val)
                    @test isfinite(pe_val)
                    @test !all(iszero, f_val)
                    @test !iszero(pe_val)
                end
            end
        end

        @testset "potential is zero at cutoff" begin
            cscrfb_test = CoulombSoftCoreBeutlerReactionField(dist_cutoff=rc_test, α=0.3)
            cscrfg_test = CoulombSoftCoreGapsysReactionField(
                dist_cutoff=rc_test,
                α=0.3,
                σQ=1.0u"nm",
            )
            @test isapprox(potential_energy(cscrfb_test, dr_rc, a1_l05, a1_l05),
                           0.0u"kJ * mol^-1"; atol=1e-10u"kJ * mol^-1")
            @test isapprox(potential_energy(cscrfg_test, dr_rc, a1_l05, a1_l05),
                           0.0u"kJ * mol^-1"; atol=1e-10u"kJ * mol^-1")
        end

        @testset "values vanish beyond cutoff" begin
            for inter_rf in (cscrfb, cscrfg)
                @test all(iszero, force(inter_rf, dr_beyond, a1_l05, a1_l05))
                @test iszero(potential_energy(inter_rf, dr_beyond, a1_l05, a1_l05))
                @test all(iszero, force(inter_rf, dr_beyond, a1_rf, a1_rf))
                @test iszero(potential_energy(inter_rf, dr_beyond, a1_rf, a1_rf))
            end
        end

        @testset "special pairs use weighted plain coulomb at lambda one" begin
            ws = 0.5
            cscrfb_sp = CoulombSoftCoreBeutlerReactionField(dist_cutoff=rc_test, weight_special=ws)
            cscrfg_sp = CoulombSoftCoreGapsysReactionField(dist_cutoff=rc_test, weight_special=ws)
            coul_plain = Coulomb()

            for dr_test in (dr12, dr13)
                ref_f_sp = force(coul_plain, dr_test, a1_rf, a1_rf) * ws
                ref_pe_sp = potential_energy(coul_plain, dr_test, a1_rf, a1_rf) * ws
                @test isapprox(
                    force(cscrfb_sp, dr_test, a1_rf, a1_rf, u"kJ * mol^-1 * nm^-1", true),
                    ref_f_sp;
                    atol=1e-9u"kJ * mol^-1 * nm^-1",
                )
                @test isapprox(
                    force(cscrfg_sp, dr_test, a1_rf, a1_rf, u"kJ * mol^-1 * nm^-1", true),
                    ref_f_sp;
                    atol=1e-9u"kJ * mol^-1 * nm^-1",
                )
                @test isapprox(
                    potential_energy(cscrfb_sp, dr_test, a1_rf, a1_rf, u"kJ * mol^-1", true),
                    ref_pe_sp;
                    atol=1e-9u"kJ * mol^-1",
                )
                @test isapprox(
                    potential_energy(cscrfg_sp, dr_test, a1_rf, a1_rf, u"kJ * mol^-1", true),
                    ref_pe_sp;
                    atol=1e-9u"kJ * mol^-1",
                )
            end
        end

        @testset "conducting boundary conditions" begin
            crf_ref_inf = CoulombReactionField(dist_cutoff=rc_test, solvent_dielectric=Inf)
            cscrfb_inf = CoulombSoftCoreBeutlerReactionField(
                dist_cutoff=rc_test, solvent_dielectric=Inf, α=1.0,
            )
            cscrfg_inf = CoulombSoftCoreGapsysReactionField(
                dist_cutoff=rc_test, solvent_dielectric=Inf, α=0.3, σQ=1.0u"nm",
            )

            # λ = 1 reduces to base reaction field at Inf, with no NaN
            for dr_test in (dr12, dr13, dr14)
                ref_f = force(crf_ref_inf, dr_test, a1_rf, a1_rf)
                ref_pe = potential_energy(crf_ref_inf, dr_test, a1_rf, a1_rf)
                @test all(isfinite, ref_f)
                @test isfinite(ref_pe)
                @test isapprox(force(cscrfb_inf, dr_test, a1_rf, a1_rf), ref_f;
                               atol=1e-9u"kJ * mol^-1 * nm^-1")
                @test isapprox(force(cscrfg_inf, dr_test, a1_rf, a1_rf), ref_f;
                               atol=1e-9u"kJ * mol^-1 * nm^-1")
                @test isapprox(potential_energy(cscrfb_inf, dr_test, a1_rf, a1_rf), ref_pe;
                               atol=1e-9u"kJ * mol^-1")
                @test isapprox(potential_energy(cscrfg_inf, dr_test, a1_rf, a1_rf), ref_pe;
                               atol=1e-9u"kJ * mol^-1")
            end

            # λ = 0.5 stays finite and nonzero (previously NaN without isinf handling)
            for inter_rf in (cscrfb_inf, cscrfg_inf)
                for dr_test in (dr12, dr13, dr14)
                    f_val = force(inter_rf, dr_test, a1_l05, a1_l05)
                    pe_val = potential_energy(inter_rf, dr_test, a1_l05, a1_l05)
                    @test all(isfinite, f_val)
                    @test isfinite(pe_val)
                    @test !all(iszero, f_val)
                    @test !iszero(pe_val)
                end
            end

            # potential is zero at the cutoff under conducting boundary conditions
            @test isapprox(potential_energy(cscrfb_inf, dr_rc, a1_l05, a1_l05),
                           0.0u"kJ * mol^-1"; atol=1e-10u"kJ * mol^-1")
            @test isapprox(potential_energy(cscrfg_inf, dr_rc, a1_l05, a1_l05),
                           0.0u"kJ * mol^-1"; atol=1e-10u"kJ * mol^-1")

            # values vanish beyond cutoff under conducting boundary conditions
            for inter_rf in (cscrfb_inf, cscrfg_inf)
                @test all(iszero, force(inter_rf, dr_beyond, a1_l05, a1_l05))
                @test iszero(potential_energy(inter_rf, dr_beyond, a1_l05, a1_l05))
            end
        end
    end

    function softcore_short_range_elec_scale(scheduler, atom_i, atom_j)
        λ_i = Molly.scale_elec(scheduler, atom_i.λ, atom_i.alch_role)
        λ_j = Molly.scale_elec(scheduler, atom_j.λ, atom_j.alch_role)
        return λ_i * λ_j
    end

    function softcore_short_range_lambda(inter, atom_i, atom_j)
        λ_glob = Molly.λ_mixing(inter.λ_mixing, atom_i, atom_j)
        pair_role = Molly.mix_roles(inter.scheduler, atom_i.alch_role, atom_j.alch_role)
        return Molly.scale_elec(inter.scheduler, λ_glob, pair_role)
    end

    function expected_short_range(inter::CoulombSoftCoreBeutlerEwald, dr, atom_i, atom_j;
                                  special=false)
        λ_soft = softcore_short_range_lambda(inter, atom_i, atom_j)
        λ_elec = softcore_short_range_elec_scale(inter.scheduler, atom_i, atom_j)
        r = norm(dr)
        if λ_soft <= 0 || λ_elec <= 0 || iszero(r) || r > inter.dist_cutoff
            return (energy=zero(inter.coulomb_const) * u"nm^-1", force=zero(dr))
        end

        qij = atom_i.charge * atom_j.charge
        if λ_soft >= 1
            pe_soft = λ_elec * inter.coulomb_const * (qij / r)
            f_soft = λ_elec * inter.coulomb_const * (qij / r^2)
        else
            term = inter.α * (1 - λ_soft) * Molly.σ_mixing(inter.σ_mixing, atom_i, atom_j)^6 + r^6
            pe_soft = λ_elec * inter.coulomb_const * (qij / sqrt(cbrt(term)))
            f_soft = λ_elec * inter.coulomb_const * (qij / (term * sqrt(cbrt(term)))) * r^5
        end

        if special
            return (energy=pe_soft * inter.weight_special, force=(f_soft / r) * dr * inter.weight_special)
        end

        erfc_αr, force_screen = Molly.softcore_ewald_screen(inter, r)
        f = (f_soft * erfc_αr) + (pe_soft * force_screen)
        return (energy=pe_soft * erfc_αr, force=(f / r) * dr)
    end

    function expected_short_range(inter::CoulombSoftCoreGapsysEwald, dr, atom_i, atom_j;
                                  special=false)
        λ_soft = softcore_short_range_lambda(inter, atom_i, atom_j)
        λ_elec = softcore_short_range_elec_scale(inter.scheduler, atom_i, atom_j)
        r = norm(dr)
        if λ_soft <= 0 || λ_elec <= 0 || iszero(r) || r > inter.dist_cutoff
            return (energy=zero(inter.coulomb_const) * u"nm^-1", force=zero(dr))
        end

        qij = atom_i.charge * atom_j.charge
        if λ_soft >= 1
            pe_soft = λ_elec * inter.coulomb_const * (qij / r)
            f_soft = λ_elec * inter.coulomb_const * (qij / r^2)
        else
            R = inter.α * sqrt(cbrt(1 - λ_soft)) * (oneunit(r) + inter.σQ * abs(qij))
            if !(r < R)
                pe_soft = λ_elec * inter.coulomb_const * (qij / r)
                f_soft = λ_elec * inter.coulomb_const * (qij / r^2)
            else
                pe_soft = λ_elec * inter.coulomb_const * (((qij / R^3) * r^2) - (((3 * qij) / R^2) * r) +
                          ((3 * qij) / R))
                f_soft = λ_elec * inter.coulomb_const * (-(((2 * qij) / R^3) * r) + ((3 * qij) / R^2))
            end
        end

        if special
            return (energy=pe_soft * inter.weight_special, force=(f_soft / r) * dr * inter.weight_special)
        end

        erfc_αr, force_screen = Molly.softcore_ewald_screen(inter, r)
        f = (f_soft * erfc_αr) + (pe_soft * force_screen)
        return (energy=pe_soft * erfc_αr, force=(f / r) * dr)
    end

    @testset "Soft-core Ewald" begin
        c4_ewald = SVector(1.05, 1.0, 1.0)u"nm"
        dr14_ewald = vector(c1, c4_ewald, boundary)
        rc_test = 1.0u"nm"
        dr_beyond = SVector(1.5u"nm", 0.0u"nm", 0.0u"nm")

        ce_ref = CoulombEwald(dist_cutoff=rc_test)
        cscbe = CoulombSoftCoreBeutlerEwald(dist_cutoff=rc_test, α=1.0)
        cscge = CoulombSoftCoreGapsysEwald(
            dist_cutoff=rc_test,
            α=0.3,
            σQ=1.0u"nm",
        )

        a1_l1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)
        a1_l0 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.0)
        a1_l05 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.5)

        @testset "lambda one matches CoulombEwald" begin
            for dr_test in (dr12, dr13, dr14_ewald)
                ref_f = force(ce_ref, dr_test, a1_l1, a1_l1)
                ref_pe = potential_energy(ce_ref, dr_test, a1_l1, a1_l1)
                @test isapprox(force(cscbe, dr_test, a1_l1, a1_l1), ref_f;
                               atol=1e-9u"kJ * mol^-1 * nm^-1")
                @test isapprox(force(cscge, dr_test, a1_l1, a1_l1), ref_f;
                               atol=1e-9u"kJ * mol^-1 * nm^-1")
                @test isapprox(potential_energy(cscbe, dr_test, a1_l1, a1_l1), ref_pe;
                               atol=1e-9u"kJ * mol^-1")
                @test isapprox(potential_energy(cscge, dr_test, a1_l1, a1_l1), ref_pe;
                               atol=1e-9u"kJ * mol^-1")
            end
        end

        @testset "lambda zero vanishes" begin
            for inter_ewald in (cscbe, cscge)
                @test all(iszero, force(inter_ewald, dr12, a1_l0, a1_l0))
                @test iszero(potential_energy(inter_ewald, dr12, a1_l0, a1_l0))
            end
        end

        @testset "lambda half stays finite" begin
            for inter_ewald in (cscbe, cscge)
                for dr_test in (dr12, dr13, dr14_ewald)
                    f_val = force(inter_ewald, dr_test, a1_l05, a1_l05)
                    pe_val = potential_energy(inter_ewald, dr_test, a1_l05, a1_l05)
                    @test all(isfinite, f_val)
                    @test isfinite(pe_val)
                    @test !all(iszero, f_val)
                    @test !iszero(pe_val)
                end
            end
        end

        @testset "values vanish beyond cutoff" begin
            for inter_ewald in (cscbe, cscge)
                @test all(iszero, force(inter_ewald, dr_beyond, a1_l05, a1_l05))
                @test iszero(potential_energy(inter_ewald, dr_beyond, a1_l05, a1_l05))
                @test all(iszero, force(inter_ewald, dr_beyond, a1_l1, a1_l1))
                @test iszero(potential_energy(inter_ewald, dr_beyond, a1_l1, a1_l1))
            end
        end

        @testset "special pairs use weighted plain Coulomb at lambda one" begin
            ws = 0.5
            cscbe_sp = CoulombSoftCoreBeutlerEwald(dist_cutoff=rc_test, weight_special=ws)
            cscge_sp = CoulombSoftCoreGapsysEwald(dist_cutoff=rc_test, weight_special=ws)
            coul_plain = Coulomb()

            for dr_test in (dr12, dr13)
                ref_f_sp = force(coul_plain, dr_test, a1_l1, a1_l1) * ws
                ref_pe_sp = potential_energy(coul_plain, dr_test, a1_l1, a1_l1) * ws
                @test isapprox(
                    force(cscbe_sp, dr_test, a1_l1, a1_l1, u"kJ * mol^-1 * nm^-1", true),
                    ref_f_sp;
                    atol=1e-9u"kJ * mol^-1 * nm^-1",
                )
                @test isapprox(
                    force(cscge_sp, dr_test, a1_l1, a1_l1, u"kJ * mol^-1 * nm^-1", true),
                    ref_f_sp;
                    atol=1e-9u"kJ * mol^-1 * nm^-1",
                )
                @test isapprox(
                    potential_energy(cscbe_sp, dr_test, a1_l1, a1_l1, u"kJ * mol^-1", true),
                    ref_pe_sp;
                    atol=1e-9u"kJ * mol^-1",
                )
                @test isapprox(
                    potential_energy(cscge_sp, dr_test, a1_l1, a1_l1, u"kJ * mol^-1", true),
                    ref_pe_sp;
                    atol=1e-9u"kJ * mol^-1",
                )
            end
        end
    end

    @testset "Soft-core Exact Overlap Safeguards" begin
        dr_zero = zero(dr12)
        dr_eps = SVector(1e-12, 0.0, 0.0)u"nm"
        overlap_atom = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.5)
        overlap_inters = (
            LennardJonesSoftCoreBeutler(α=0.3),
            LennardJonesSoftCoreGapsys(α=0.85),
            CoulombSoftCoreBeutler(α=0.3),
            CoulombSoftCoreGapsys(α=0.3, σQ=1.0u"nm"),
            CoulombSoftCoreBeutlerReactionField(dist_cutoff=1.0u"nm", α=0.3),
            CoulombSoftCoreGapsysReactionField(dist_cutoff=1.0u"nm", α=0.3, σQ=1.0u"nm"),
            CoulombSoftCoreBeutlerEwald(dist_cutoff=1.0u"nm", α=0.3),
            CoulombSoftCoreGapsysEwald(dist_cutoff=1.0u"nm", α=0.3, σQ=1.0u"nm"),
        )

        λ_half = overlap_atom.λ
        qij = overlap_atom.charge * overlap_atom.charge
        σ6 = overlap_atom.σ^6
        C6 = 4 * overlap_atom.ϵ * σ6
        C12 = C6 * σ6
        rc = 1.0u"nm"
        ε_rf = 78.3
        krf = inv(rc^3) * ((ε_rf - 1) / (2 * ε_rf + 1))
        crf = inv(rc) * ((3 * ε_rf) / (2 * ε_rf + 1))
        σ6_shift = 0.3 * (1 - λ_half) * σ6
        R_beutler = sqrt(cbrt(σ6_shift))
        R_gapsys_coul = 0.3 * sqrt(cbrt(1 - λ_half)) * (1.0u"nm" + 1.0u"nm" * abs(qij))
        R_gapsys_lj = 0.85 * sqrt(cbrt((26 * σ6 * (1 - λ_half)) / 7))
        crf_λ = inv(sqrt(cbrt(σ6_shift + rc^6))) + krf * rc^2
        ewald_screen_beutler = Molly.softcore_ewald_screen(overlap_inters[7], zero(dr_zero[1]))[1]
        ewald_screen_gapsys = Molly.softcore_ewald_screen(overlap_inters[8], zero(dr_zero[1]))[1]
        ewald_charge_scale = Molly.scale_elec(Molly.DefaultLambdaScheduler(), λ_half, Molly.CoreRole)^2
        overlap_pe_beutler = λ_half * Molly.coulomb_const * (qij / R_beutler)
        overlap_pe_gapsys = λ_half * Molly.coulomb_const * ((3 * qij) / R_gapsys_coul)
        overlap_expectations = (
            λ_half * ((C12 / (σ6_shift * σ6_shift)) - (C6 / σ6_shift)),
            λ_half * ((91 * C12 / R_gapsys_lj^12) - (28 * C6 / R_gapsys_lj^6)),
            overlap_pe_beutler,
            overlap_pe_gapsys,
            λ_half * Molly.coulomb_const * qij * (inv(R_beutler) - crf_λ),
            λ_half * Molly.coulomb_const * qij * (3 * inv(R_gapsys_coul) - crf),
            ewald_charge_scale * Molly.coulomb_const * (qij / R_beutler) * ewald_screen_beutler,
            ewald_charge_scale * Molly.coulomb_const * ((3 * qij) / R_gapsys_coul) * ewald_screen_gapsys,
        )

        for (inter, expected_pe) in zip(overlap_inters, overlap_expectations)
            f_val = force(inter, dr_zero, overlap_atom, overlap_atom)
            pe_val = potential_energy(inter, dr_zero, overlap_atom, overlap_atom)
            pe_eps = potential_energy(inter, dr_eps, overlap_atom, overlap_atom)
            @test all(isfinite, f_val)
            @test all(iszero, f_val)
            @test isfinite(pe_val)
            @test isapprox(pe_val, expected_pe; rtol=1e-8, atol=1e-8u"kJ * mol^-1")
            @test isapprox(pe_val, pe_eps; rtol=1e-8, atol=1e-8u"kJ * mol^-1")
        end

        endpoint_atom = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)
        endpoint_inters = (
            LennardJonesSoftCoreBeutler(α=0.3),
            LennardJonesSoftCoreGapsys(α=0.85),
            CoulombSoftCoreBeutler(α=0.3),
            CoulombSoftCoreGapsys(α=0.3, σQ=1.0u"nm"),
            CoulombSoftCoreBeutlerReactionField(dist_cutoff=1.0u"nm", α=0.3),
            CoulombSoftCoreGapsysReactionField(dist_cutoff=1.0u"nm", α=0.3, σQ=1.0u"nm"),
            CoulombSoftCoreBeutlerEwald(dist_cutoff=1.0u"nm", α=0.3),
            CoulombSoftCoreGapsysEwald(dist_cutoff=1.0u"nm", α=0.3, σQ=1.0u"nm"),
        )

        for inter in endpoint_inters
            f_val = force(inter, dr_zero, endpoint_atom, endpoint_atom)
            pe_val = potential_energy(inter, dr_zero, endpoint_atom, endpoint_atom)
            @test all(isfinite, f_val)
            @test all(iszero, f_val)
            @test isfinite(pe_val)
            @test iszero(pe_val)
        end

        degenerate_inters = (
            LennardJonesSoftCoreBeutler(α=0.0),
            LennardJonesSoftCoreGapsys(α=0.0),
            CoulombSoftCoreBeutler(α=0.0),
            CoulombSoftCoreGapsys(α=0.0, σQ=1.0u"nm"),
            CoulombSoftCoreBeutlerReactionField(dist_cutoff=1.0u"nm", α=0.0),
            CoulombSoftCoreGapsysReactionField(dist_cutoff=1.0u"nm", α=0.0, σQ=1.0u"nm"),
            CoulombSoftCoreBeutlerEwald(dist_cutoff=1.0u"nm", α=0.0),
            CoulombSoftCoreGapsysEwald(dist_cutoff=1.0u"nm", α=0.0, σQ=1.0u"nm"),
        )

        for inter in degenerate_inters
            f_val = force(inter, dr_zero, overlap_atom, overlap_atom)
            pe_val = potential_energy(inter, dr_zero, overlap_atom, overlap_atom)
            @test all(isfinite, f_val)
            @test all(iszero, f_val)
            @test isfinite(pe_val)
            @test iszero(pe_val)
        end

        low_λ_atom = Atom(
            charge=1.0,
            σ=0.3u"nm",
            ϵ=0.2u"kJ * mol^-1",
            λ=0.25,
            alch_role=Molly.DeleteRole,
        )
        core_atom = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)

        for inter in (LennardJonesSoftCoreBeutler(α=0.3), LennardJonesSoftCoreGapsys(α=0.85))
            @test all(isfinite, force(inter, dr12, low_λ_atom, low_λ_atom))
            @test isfinite(potential_energy(inter, dr12, low_λ_atom, low_λ_atom))
            @test all(iszero, force(inter, dr12, low_λ_atom, core_atom))
            @test iszero(potential_energy(inter, dr12, low_λ_atom, core_atom))
        end
    end

    @testset "PME Scheduler Charge Scaling" begin
        boundary_pme = CubicBoundary(2.5u"nm")
        coords_pme = [
            SVector(0.2, 0.2, 0.2)u"nm",
            SVector(0.9, 0.7, 0.4)u"nm",
            SVector(1.6, 1.2, 1.1)u"nm",
        ]
        rc = 1.0u"nm"

        @testset "default scheduler matches pre-scaled charges" begin
            λ_state = 0.75
            scheduler = Molly.DefaultLambdaScheduler()
            λ_elec = Molly.scale_elec(scheduler, λ_state, Molly.InsertRole)

            atoms_raw = [
                Atom(charge=1.0, λ=λ_state, alch_role=Molly.InsertRole),
                Atom(charge=-0.8, λ=λ_state, alch_role=Molly.InsertRole),
                Atom(charge=0.3),
            ]
            atoms_ref = [
                Atom(charge=1.0 * λ_elec),
                Atom(charge=-0.8 * λ_elec),
                Atom(charge=0.3),
            ]

            pme_raw = PME(rc, atoms_raw, boundary_pme; scheduler=scheduler)
            pme_ref = PME(rc, atoms_ref, boundary_pme)
            sys_raw = System(atoms=atoms_raw, coords=coords_pme, boundary=boundary_pme,
                             pairwise_inters=(), general_inters=(pme_raw,))
            sys_ref = System(atoms=atoms_ref, coords=coords_pme, boundary=boundary_pme,
                             pairwise_inters=(), general_inters=(pme_ref,))

            @test isapprox(potential_energy(sys_raw), potential_energy(sys_ref);
                           atol=1e-9u"kJ * mol^-1")
            @test maximum(norm.(forces(sys_raw) .- forces(sys_ref))) <
                  1e-9u"kJ * mol^-1 * nm^-1"

            sys_raw_full = System(
                atoms=atoms_raw,
                coords=coords_pme,
                boundary=boundary_pme,
                pairwise_inters=(CoulombEwaldScaled(dist_cutoff=rc, scheduler=scheduler),),
                general_inters=(PME(rc, atoms_raw, boundary_pme; scheduler=scheduler),),
            )
            sys_ref_full = System(
                atoms=atoms_ref,
                coords=coords_pme,
                boundary=boundary_pme,
                pairwise_inters=(CoulombEwald(dist_cutoff=rc),),
                general_inters=(PME(rc, atoms_ref, boundary_pme),),
            )

            @test isapprox(potential_energy(sys_raw_full), potential_energy(sys_ref_full);
                           atol=1e-9u"kJ * mol^-1")
            @test maximum(norm.(forces(sys_raw_full) .- forces(sys_ref_full))) <
                  1e-9u"kJ * mol^-1 * nm^-1"
        end

        @testset "non-default scheduler matches pre-scaled charges" begin
            λ_state = 0.75
            scheduler = Molly.EleScaledLambdaScheduler()
            λ_elec = Molly.scale_elec(scheduler, λ_state, Molly.InsertRole)

            atoms_raw = [
                Atom(charge=1.2, λ=λ_state, alch_role=Molly.InsertRole),
                Atom(charge=-0.9, λ=λ_state, alch_role=Molly.InsertRole),
                Atom(charge=0.25),
            ]
            atoms_ref = [
                Atom(charge=1.2 * λ_elec),
                Atom(charge=-0.9 * λ_elec),
                Atom(charge=0.25),
            ]

            pme_raw = PME(rc, atoms_raw, boundary_pme; scheduler=scheduler)
            pme_ref = PME(rc, atoms_ref, boundary_pme)
            sys_raw = System(atoms=atoms_raw, coords=coords_pme, boundary=boundary_pme,
                             pairwise_inters=(), general_inters=(pme_raw,))
            sys_ref = System(atoms=atoms_ref, coords=coords_pme, boundary=boundary_pme,
                             pairwise_inters=(), general_inters=(pme_ref,))

            @test isapprox(potential_energy(sys_raw), potential_energy(sys_ref);
                           atol=1e-9u"kJ * mol^-1")
            @test maximum(norm.(forces(sys_raw) .- forces(sys_ref))) <
                  1e-9u"kJ * mol^-1 * nm^-1"
        end
    end

    @testset "Soft-core PME End-to-End" begin
        boundary_pme = CubicBoundary(2.2u"nm")
        coords_pme = [
            SVector(0.2, 0.2, 0.2)u"nm",
            SVector(0.26, 0.2, 0.2)u"nm",
        ]
        rc = 1.0u"nm"

        atoms_l1 = [
            Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0),
            Atom(charge=-1.0, σ=0.25u"nm", ϵ=0.15u"kJ * mol^-1", λ=1.0),
        ]
        atoms_l05 = [
            Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=0.5),
            Atom(charge=-1.0, σ=0.25u"nm", ϵ=0.15u"kJ * mol^-1", λ=0.5),
        ]

        sys_ref = System(
            atoms=atoms_l1,
            coords=coords_pme,
            boundary=boundary_pme,
            pairwise_inters=(CoulombEwald(dist_cutoff=rc),),
            general_inters=(PME(rc, atoms_l1, boundary_pme),),
        )
        sys_beutler_l1 = System(
            atoms=atoms_l1,
            coords=coords_pme,
            boundary=boundary_pme,
            pairwise_inters=(CoulombSoftCoreBeutlerEwald(dist_cutoff=rc, α=0.3),),
            general_inters=(PME(rc, atoms_l1, boundary_pme),),
        )
        sys_gapsys_l1 = System(
            atoms=atoms_l1,
            coords=coords_pme,
            boundary=boundary_pme,
            pairwise_inters=(CoulombSoftCoreGapsysEwald(dist_cutoff=rc, α=0.3, σQ=1.0u"nm"),),
            general_inters=(PME(rc, atoms_l1, boundary_pme),),
        )

        @test isapprox(potential_energy(sys_beutler_l1), potential_energy(sys_ref);
                       atol=1e-9u"kJ * mol^-1")
        @test isapprox(potential_energy(sys_gapsys_l1), potential_energy(sys_ref);
                       atol=1e-9u"kJ * mol^-1")
        @test maximum(norm.(forces(sys_beutler_l1) .- forces(sys_ref))) <
              1e-9u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces(sys_gapsys_l1) .- forces(sys_ref))) <
              1e-9u"kJ * mol^-1 * nm^-1"

        for pair_inter in (
            CoulombSoftCoreBeutlerEwald(dist_cutoff=rc, α=0.3),
            CoulombSoftCoreGapsysEwald(dist_cutoff=rc, α=0.3, σQ=1.0u"nm"),
        )
            sys_soft = System(
                atoms=atoms_l05,
                coords=coords_pme,
                boundary=boundary_pme,
                pairwise_inters=(pair_inter,),
                general_inters=(PME(rc, atoms_l05, boundary_pme),),
            )
            pe_soft = potential_energy(sys_soft)
            fs_soft = forces(sys_soft)
            @test isfinite(pe_soft)
            @test all(fi -> all(isfinite, fi), fs_soft)
        end

        @testset "soft-core short range stays aligned with Ewald and PME long range" begin
            λ_state = 0.75
            atoms_raw = [
                Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=λ_state, alch_role=Molly.InsertRole),
                Atom(charge=-1.0, σ=0.25u"nm", ϵ=0.15u"kJ * mol^-1", λ=λ_state, alch_role=Molly.InsertRole),
            ]

            for scheduler in (Molly.DefaultLambdaScheduler(), Molly.EleScaledLambdaScheduler())
                λ_elec = Molly.scale_elec(scheduler, λ_state, Molly.InsertRole)
                atoms_ref = [
                    Atom(charge=atoms_raw[1].charge * λ_elec),
                    Atom(charge=atoms_raw[2].charge * λ_elec),
                ]

                general_refs = (
                    (
                        Ewald(rc; scheduler=scheduler),
                        Ewald(rc),
                    ),
                    (
                        PME(rc, atoms_raw, boundary_pme; scheduler=scheduler),
                        PME(rc, atoms_ref, boundary_pme),
                    ),
                )

                for pair_inter in (
                    CoulombSoftCoreBeutlerEwald(dist_cutoff=rc, α=0.3, scheduler=scheduler),
                    CoulombSoftCoreGapsysEwald(dist_cutoff=rc, α=0.3, σQ=1.0u"nm", scheduler=scheduler),
                )
                    pair_expected = expected_short_range(
                        pair_inter,
                        vector(coords_pme[1], coords_pme[2], boundary_pme),
                        atoms_raw...,
                    )
                    for (general_raw, general_ref) in general_refs
                        sys_raw = System(
                            atoms=atoms_raw,
                            coords=coords_pme,
                            boundary=boundary_pme,
                            pairwise_inters=(pair_inter,),
                            general_inters=(general_raw,),
                        )
                        sys_ref = System(
                            atoms=atoms_ref,
                            coords=coords_pme,
                            boundary=boundary_pme,
                            pairwise_inters=(),
                            general_inters=(general_ref,),
                        )

                        force_expected = forces(sys_ref)
                        force_expected[1] -= pair_expected.force
                        force_expected[2] += pair_expected.force

                        @test isapprox(
                            potential_energy(sys_raw),
                            potential_energy(sys_ref) + pair_expected.energy;
                            atol=1e-8u"kJ * mol^-1",
                        )
                        @test maximum(norm.(forces(sys_raw) .- force_expected)) <
                              1e-8u"kJ * mol^-1 * nm^-1"
                    end
                end
            end
        end
    end

    @testset "AlchemicalPartition charge-dependent Ewald terms" begin
        boundary_pme = CubicBoundary(2.4u"nm")
        coords_pme = [
            SVector(0.2, 0.2, 0.2)u"nm",
            SVector(0.8, 0.4, 0.3)u"nm",
            SVector(1.4, 1.2, 0.9)u"nm",
        ]
        rc = 1.0u"nm"
        temp = 298.0u"K"
        dt = 0.001u"ps"
        λs = (1.0, 0.75, 0.6)
        eligible = trues(3, 3)

        atoms_λ(λ) = [
            Atom(charge=1.0,  σ=0.3u"nm",  ϵ=0.2u"kJ * mol^-1",
                 λ=λ, alch_role=Molly.InsertRole),
            Atom(charge=-1.0, σ=0.25u"nm", ϵ=0.15u"kJ * mol^-1",
                 λ=λ, alch_role=Molly.InsertRole),
            Atom(charge=0.3,  σ=0.28u"nm", ϵ=0.12u"kJ * mol^-1"),
        ]

        ewald_exclusions() = InteractionList2Atoms(
            Int32[1],
            Int32[2],
            [EwaldExclusion()],
            [""],
            Molly.EwaldExclusionData(rc),
        )

        function thermo_states(pairwise_inters, general_inter; specific_inter_lists=(),
                               special=falses(3, 3))
            return [
                begin
                    atoms = atoms_λ(λ)
                    sys = System(
                        atoms=atoms,
                        coords=coords_pme,
                        boundary=boundary_pme,
                        pairwise_inters=pairwise_inters,
                        specific_inter_lists=specific_inter_lists,
                        general_inters=(general_inter(atoms),),
                        neighbor_finder=DistanceNeighborFinder(
                            eligible=eligible,
                            special=special,
                            dist_cutoff=rc,
                        ),
                    )
                    ThermoState(sys, VelocityVerlet(dt=dt); temperature=temp)
                end for λ in λs
            ]
        end

        function test_partition_matches_direct(thermo_states, inter_type; has_exclusions=false)
            partition = Molly.AlchemicalPartition(thermo_states)
            partitioned = Molly.evaluate_energy_all!(partition, coords_pme, boundary_pme)
            direct = [potential_energy(ts.system) for ts in thermo_states]

            @test all(isapprox(pe_p, pe_d; rtol=1e-8, atol=1e-8u"kJ * mol^-1")
                      for (pe_p, pe_d) in zip(partitioned, direct))
            @test !any(inter -> inter isa inter_type, partition.master_sys.general_inters)
            @test all(any(inter -> inter isa inter_type, λ_sys.general_inters)
                      for λ_sys in partition.λ_systems)

            if has_exclusions
                is_ewald_exclusion_list(il) = il isa InteractionList2Atoms &&
                                              il.data isa Molly.EwaldExclusionData
                @test !any(is_ewald_exclusion_list, partition.master_sys.specific_inter_lists)
                @test all(any(is_ewald_exclusion_list, λ_sys.specific_inter_lists)
                          for λ_sys in partition.λ_systems)
            end
        end

        pme_states = thermo_states(
            (LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),
             CoulombSoftCoreBeutlerEwald(dist_cutoff=rc, α=0.3, use_neighbors=true)),
            atoms -> PME(rc, atoms, boundary_pme),
        )
        test_partition_matches_direct(pme_states, PME)

        ewald_states = thermo_states(
            (LennardJonesSoftCoreGapsys(α=0.85, use_neighbors=true),
             CoulombSoftCoreGapsysEwald(dist_cutoff=rc, α=0.3, σQ=1.0u"nm",
                                        use_neighbors=true)),
            atoms -> Ewald(rc),
        )
        test_partition_matches_direct(ewald_states, Ewald)

        special = falses(3, 3)
        special[1, 2] = special[2, 1] = true
        exclusion_states = thermo_states(
            (CoulombSoftCoreBeutlerEwald(dist_cutoff=rc, α=0.3, use_neighbors=true),),
            atoms -> PME(rc, atoms, boundary_pme);
            specific_inter_lists=(ewald_exclusions(),),
            special=special,
        )
        test_partition_matches_direct(exclusion_states, PME; has_exclusions=true)
    end

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

    # Test HarmonicBond at equilibrium distance
    b_eq = HarmonicBond(k=100_000.0u"kJ * mol^-1 * nm^-2", r0=0.3u"nm")
    fs_eq = force(b_eq, c1, c2, boundary)
    @test isapprox(
        fs_eq.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(b_eq, c1, c2, boundary),
        0.0u"kJ * mol^-1";
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

    # Test HarmonicAngle for collinear atoms (180 degrees)
    c3_colin = SVector(1.6, 1.0, 1.0)u"nm"
    a_colin = HarmonicAngle(k=100.0u"kJ * mol^-1", θ0=π)
    fs_colin = force(a_colin, c1, c2, c3_colin, boundary)
    @test isapprox(
        fs_colin.f1,
        SVector(0.0, 0.0, 0.0)u"kJ * mol^-1 * nm^-1";
        atol=1e-9u"kJ * mol^-1 * nm^-1",
    )
    @test isapprox(
        potential_energy(a_colin, c1, c2, c3_colin, boundary),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    # Test HarmonicAngle for 90 degree angle
    c3_90 = SVector(1.3, 1.3, 1.0)u"nm"
    a_90 = HarmonicAngle(k=200.0u"kJ * mol^-1", θ0=π/2)
    fs_90 = force(a_90, c1, c2, c3_90, boundary)
    pe_90 = potential_energy(a_90, c1, c2, c3_90, boundary)
    @test isapprox(
        pe_90,
        0.0u"kJ * mol^-1";
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

    # RBTorsion tests
    # Use proper non-collinear geometry for torsion - atoms arranged in a dihedral
    c1t = SVector(0.0, 0.0, 0.0)u"nm"
    c2t = SVector(0.1, 0.0, 0.0)u"nm"
    c3t = SVector(0.2, 0.1, 0.0)u"nm"
    c4t = SVector(0.3, 0.1, 0.1)u"nm"
    boundary_rb = CubicBoundary(5.0u"nm")
    
    rb1 = RBTorsion(f1=10.0u"kJ * mol^-1", f2=20.0u"kJ * mol^-1",
                    f3=30.0u"kJ * mol^-1", f4=5.0u"kJ * mol^-1")
    
    # Test that force calculation produces expected values (regression test)
    fs = force(rb1, c1t, c2t, c3t, c4t, boundary_rb)
    @test isapprox(norm(fs.f1), 497.6067743425172u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(norm(fs.f2), 673.7627575276049u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(norm(fs.f3), 351.86112450195805u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(norm(fs.f4), 287.2934051172337u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    
    # Test potential energy calculation
    pe = potential_energy(rb1, c1t, c2t, c3t, c4t, boundary_rb)
    @test isapprox(pe, 47.38033871712585u"kJ * mol^-1"; atol=1e-9u"kJ * mol^-1")

    # PeriodicTorsion tests
    pt1 = PeriodicTorsion(periodicities=(1, 2, 3), phases=(0.0, Float64(π/2), Float64(π)),
                          ks=(10.0u"kJ * mol^-1", 5.0u"kJ * mol^-1", 2.0u"kJ * mol^-1"),
                          proper=true)
    
    # Test with non-collinear atoms - forces should match expected values (regression test)
    fs = force(pt1, c1t, c2t, c3t, c4t, boundary_rb)
    @test isapprox(norm(fs.f1), 139.51649514944324u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(norm(fs.f2), 188.90622744571394u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(norm(fs.f3), 98.65305980755139u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(norm(fs.f4), 80.54988603092418u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    
    # Test potential energy calculation
    pe = potential_energy(pt1, c1t, c2t, c3t, c4t, boundary_rb)
    @test isapprox(pe, 4.587951202894674u"kJ * mol^-1"; atol=1e-9u"kJ * mol^-1")
    
    # Test zero PeriodicTorsion
    pt_zero = zero(pt1)
    @test pt_zero.ks == (0.0u"kJ * mol^-1", 0.0u"kJ * mol^-1", 0.0u"kJ * mol^-1")
    
    # Test addition of PeriodicTorsion (ensure types match)
    pt2 = PeriodicTorsion(periodicities=(1, 2, 3), phases=(0.0, π/4, π/3),
                          ks=(1.0u"kJ * mol^-1", 2.0u"kJ * mol^-1", 3.0u"kJ * mol^-1"),
                          proper=true)
    pt_sum = pt1 + pt2
    @test pt_sum.ks[1] ≈ 11.0u"kJ * mol^-1"
    @test pt_sum.ks[2] ≈ 7.0u"kJ * mol^-1"
    @test pt_sum.ks[3] ≈ 5.0u"kJ * mol^-1"

    # Test improper torsion
    pt_improper = PeriodicTorsion(periodicities=(2,), phases=(Float64(π),),
                                  ks=(15.0u"kJ * mol^-1",), proper=false)
    pe_improper = potential_energy(pt_improper, c1t, c2t, c3t, c4t, boundary_rb)
    @test isapprox(pe_improper, 20.0u"kJ * mol^-1"; atol=1e-9u"kJ * mol^-1")

    htor = HarmonicTorsion(1000.0u"kJ/mol", -1.8)
    c1ht = SVector(27.151, 33.362, 10.650)u"Å"
    c2ht = SVector(28.260, 33.943, 11.096)u"Å"
    c3ht = SVector(28.605, 33.965, 12.503)u"Å"
    c4ht = SVector(28.638, 35.461, 12.900)u"Å"
    boundary_htor = CubicBoundary(Inf * u"Å")
    pe_htor = potential_energy(htor, c1ht, c2ht, c3ht, c4ht, boundary_htor)
    @test pe_htor ≈ 67.60869243622506u"kJ/mol"
    fs_htor = force(htor, c1ht, c2ht, c3ht, c4ht, boundary_htor)
    @test fs_htor.f1 ≈ SVector(-228.63867893470425,  398.16345029859656,  49.837063486781354)u"kJ * mol^-1 * Å^-1"
    @test fs_htor.f2 ≈ SVector( 242.87672193557964, -596.3836043695466 , -50.228876881052855)u"kJ * mol^-1 * Å^-1"
    @test fs_htor.f3 ≈ SVector( 324.1211893467139 ,  212.83417983707614, -82.80324255936942 )u"kJ * mol^-1 * Å^-1"
    @test fs_htor.f4 ≈ SVector(-338.3592323475893 , -14.614025766126122,  83.19505595364092 )u"kJ * mol^-1 * Å^-1"

    ff_cmap = MolecularForceField(
        joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...;
        strictness=:nowarn,
    )
    sys_cmap = System(joinpath(data_dir, "6mrr_equil.pdb"), ff_cmap)
    cmap = CMAPTorsion(48384, 24)
    cmap_data = sys_cmap.specific_inter_lists[5].data
    pe_cmap = potential_energy(cmap, sys_cmap.coords[379], sys_cmap.coords[381],
                        sys_cmap.coords[383], sys_cmap.coords[393], sys_cmap.coords[395],
                        sys_cmap.boundary, nothing, nothing, nothing, nothing, nothing, nothing,
                        nothing, nothing, nothing, nothing, nothing, nothing, cmap_data)
    @test pe_cmap ≈ -10.833264876u"kJ * mol^-1"
    fs_cmap = force(cmap, sys_cmap.coords[379], sys_cmap.coords[381], sys_cmap.coords[383],
                    sys_cmap.coords[393], sys_cmap.coords[395], sys_cmap.boundary,
                    nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                    nothing, nothing, nothing, nothing, cmap_data)
    @test fs_cmap.f1 ≈ SVector(-1.033477158 , -1.186639956 ,  0.850729764 )u"kJ * mol^-1 * nm^-1"
    @test fs_cmap.f2 ≈ SVector(-12.152133492,  17.987633158, -38.576945178)u"kJ * mol^-1 * nm^-1"
    @test fs_cmap.f3 ≈ SVector( 25.299640846, -27.965252121,  73.477215636)u"kJ * mol^-1 * nm^-1"
    @test fs_cmap.f4 ≈ SVector(-27.701832125,  22.775084670, -83.963292267)u"kJ * mol^-1 * nm^-1"
    @test fs_cmap.f5 ≈ SVector( 15.587801928, -11.610825751,  48.212292045)u"kJ * mol^-1 * nm^-1"

    struct AlwaysShortcut end

    a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ=1.0)
    a2 = Atom(charge=1.0, σ=0.2u"nm", ϵ=0.1u"kJ * mol^-1", λ=1.0)
    for inter in (
            LennardJones(),
            Mie(m=6, n=12),
            LennardJonesSoftCoreBeutler(α=1),
            LennardJonesSoftCoreGapsys(α=1),
        )
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

    # Test Mie potential with different m and n values
    # Redefine atoms for pairwise interactions (a1 was overwritten by UreyBradley above)
    a1_mie = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", λ = 1.0)
    
    # Test Mie potential with m=4, n=6 (softer than LJ) - regression test
    mie_soft = Mie(m=4, n=6)
    f_soft = force(mie_soft, dr12, a1_mie, a1_mie)
    pe_soft = potential_energy(mie_soft, dr12, a1_mie, a1_mie)
    @test isapprox(norm(f_soft), 9.0u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(pe_soft, 0.0u"kJ * mol^-1"; atol=1e-9u"kJ * mol^-1")

    # Test Mie potential with m=12, n=24 (harder than LJ) - regression test
    mie_hard = Mie(m=12, n=24)
    f_hard = force(mie_hard, dr12, a1_mie, a1_mie)
    pe_hard = potential_energy(mie_hard, dr12, a1_mie, a1_mie)
    @test isapprox(norm(f_hard), 32.0u"kJ * mol^-1 * nm^-1"; atol=1e-9u"kJ * mol^-1 * nm^-1")
    @test isapprox(pe_hard, 0.0u"kJ * mol^-1"; atol=1e-9u"kJ * mol^-1")

    # Test soft-core with λ=0 (should give zero interaction)
    lj_sc_zero = LennardJonesSoftCoreBeutler(α=0.5)
    @test isapprox(
        potential_energy(lj_sc_zero, dr12, a1_mie, a1_mie),
        0.0u"kJ * mol^-1";
        atol=1e-9u"kJ * mol^-1",
    )

    # Test weight_special parameter
    coulomb_weighted = Coulomb(weight_special=0.5)
    f_full = force(Coulomb(), dr12, a1_mie, a1_mie)
    f_weighted = force(coulomb_weighted, dr12, a1_mie, a1_mie, u"kJ * mol^-1 * nm^-1", true)
    @test isapprox(f_weighted, f_full .* 0.5; atol=1e-9u"kJ * mol^-1 * nm^-1")

    pe_full = potential_energy(Coulomb(), dr12, a1_mie, a1_mie)
    pe_weighted = potential_energy(coulomb_weighted, dr12, a1_mie, a1_mie, u"kJ * mol^-1 * nm^-1", true)
    @test isapprox(pe_weighted, pe_full * 0.5; atol=1e-9u"kJ * mol^-1")

    # Test mixing rules with edge cases (zero values)
    a_zero = Atom(charge=0.0, σ=0.0u"nm", ϵ=0.0u"kJ * mol^-1")
    @test Molly.σ_mixing(Molly.LorentzMixing(), a1_mie, a_zero) ≈ 0.15u"nm"
    @test Molly.ϵ_mixing(Molly.LorentzMixing(), a1_mie, a_zero) ≈ 0.1u"kJ * mol^-1"
    @test Molly.σ_mixing(Molly.GeometricMixing(), a1_mie, a_zero) ≈ 0.0u"nm"
    @test Molly.ϵ_mixing(Molly.GeometricMixing(), a1_mie, a_zero) ≈ 0.0u"kJ * mol^-1"

    ljdc = LJDispersionCorrection([a1, a2], 1.0u"nm")
    @test ljdc.factor_6 + ljdc.factor_12 ≈ -0.00208532857855u"kJ * nm^3 * mol^-1"

    InteractionList2Atoms([1, 2], [3, 4], [0.0, 0.0])
    @test_throws ArgumentError InteractionList2Atoms([1, 2], [3, 4], [0.0])
end

@testset "Cutoffs" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.7, 1.0, 1.0)u"nm"
    c3 = SVector(2.0, 1.0, 1.0)u"nm"  # Distance of 1.0 nm > dist_cut (0.8 nm)
    c4 = SVector(1.95, 1.0, 1.0)u"nm"  # Distance of 0.95 nm > dist_cut (0.8 nm)
    a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
    boundary = CubicBoundary(2.0u"nm")
    dr12 = vector(c1, c2, boundary)
    dr13 = vector(c1, c3, boundary)
    dr14 = vector(c1, c4, boundary)
    dist_cut = 0.8u"nm"
    dist_act = 0.6u"nm"
    fu, eu = u"kJ * mol^-1 * nm^-1", u"kJ * mol^-1"

    cutoffs = [
        (NoCutoff()                           , -0.04196301990 * fu, -0.00492640193 * eu),
        (DistanceCutoff(dist_cut)             , -0.04196301990 * fu, -0.00492640193 * eu),
        (ShiftedPotentialCutoff(dist_cut)     , -0.04196301990 * fu, -0.00270785727 * eu),
        (ShiftedForceCutoff(dist_cut)         , -0.02537033587 * fu, -0.00104858887 * eu),
        (CubicSplineCutoff(dist_act, dist_cut), -0.06201171875 * fu, -0.00312500000 * eu),
        (PolynomialCutoff(dist_act, dist_cut) , -0.06716652806 * fu, -0.00246320097 * eu),
    ]

    for (cutoff, force_ref, pe_ref) in cutoffs
        inter = LennardJones(cutoff=cutoff)
        @test isapprox(
            force(inter, dr12, a1, a1)[1],
            force_ref;
            atol=1e-9u"kJ * mol^-1 * nm^-1",
        )
        @test isapprox(
            potential_energy(inter, dr12, a1, a1),
            pe_ref;
            atol=1e-9u"kJ * mol^-1",
        )

        # Test that force and potential energy are zero after the cutoff distance
        if !(cutoff isa NoCutoff)
            @test isapprox(
                force(inter, dr13, a1, a1)[1],
                0.0u"kJ * mol^-1 * nm^-1";
                atol=1e-12u"kJ * mol^-1 * nm^-1",
            )
            @test isapprox(
                potential_energy(inter, dr13, a1, a1),
                0.0u"kJ * mol^-1";
                atol=1e-12u"kJ * mol^-1",
            )
            @test isapprox(
                force(inter, dr14, a1, a1)[1],
                0.0u"kJ * mol^-1 * nm^-1";
                atol=1e-12u"kJ * mol^-1 * nm^-1",
            )
            @test isapprox(
                potential_energy(inter, dr14, a1, a1),
                0.0u"kJ * mol^-1";
                atol=1e-12u"kJ * mol^-1",
            )
        end
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
                if (n_threads > 1 && AT != Array) || (T == Float64 && AT == MtlArray)
                    continue
                end
                ff = MolecularForceField(T, joinpath(ff_dir, "tip3p_standard.xml"))
                sys_init = System(
                    joinpath(data_dir, "water_3mol_cubic.pdb"),
                    ff;
                    array_type=AT,
                    dist_cutoff=T(dist_cutoff),
                    dist_buffer=zero(T(dist_cutoff)),
                    nonbonded_method=:ewald,
                    dispersion_correction=false,
                    center_coords=false,
                    strictness=:nowarn,
                )
                sys = System(
                    sys_init;
                    pairwise_inters=(sys_init.pairwise_inters[2],),
                    specific_inter_lists=(sys_init.specific_inter_lists[end],),
                )

                @test potential_energy(sys; n_threads=n_threads) ≈ E_openmm atol=2e-4u"kJ/mol"
                fs = from_device(forces(sys; n_threads=n_threads))
                @test maximum(norm.(fs .- Fs_openmm)) < 5e-4u"kJ * mol^-1 * nm^-1"
                E_gi, fs_gi = AtomsCalculators.energy_forces(sys, sys.general_inters[1];
                                                             n_threads=n_threads)
                fs_gi2 = zero(fs_gi)
                E_gi2, fs_gi2 = AtomsCalculators.energy_forces!(fs_gi2, sys,
                                                sys.general_inters[1]; n_threads=n_threads)
                @test E_gi == E_gi2
                @test fs_gi == fs_gi2
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
        for AT in array_list
            for n_threads in n_threads_list
                for T in (Float64, Float32)
                    if (n_threads > 1 && AT != Array) || (T == Float64 && AT == MtlArray)
                        continue
                    end
                    ff = MolecularForceField(T, joinpath(ff_dir, "tip3p_standard.xml"))
                    sys_init = System(
                        joinpath(data_dir, pdb_fp),
                        ff;
                        array_type=AT,
                        dist_cutoff=T(dist_cutoff),
                        dist_buffer=zero(T(dist_cutoff)),
                        nonbonded_method=:pme,
                        dispersion_correction=false,
                        center_coords=false,
                        strictness=:nowarn,
                    )
                    sys = System(
                        sys_init;
                        pairwise_inters=(sys_init.pairwise_inters[2],),
                        specific_inter_lists=(sys_init.specific_inter_lists[end],),
                    )

                    @test potential_energy(sys; n_threads=n_threads) ≈ E_openmm atol=2e-4u"kJ/mol"
                    fs = from_device(forces(sys; n_threads=n_threads))
                    @test maximum(norm.(fs .- Fs_openmm)) < 5e-4u"kJ * mol^-1 * nm^-1"
                    E_gi, fs_gi = AtomsCalculators.energy_forces(sys, sys.general_inters[1];
                                                                 n_threads=n_threads)
                    fs_gi2 = zero(fs_gi)
                    E_gi2, fs_gi2 = AtomsCalculators.energy_forces!(fs_gi2, sys,
                                                    sys.general_inters[1]; n_threads=n_threads)
                    @test E_gi == E_gi2
                    @test fs_gi == fs_gi2
                end
            end
        end
    end
end

@testset "DPD interaction" begin
    r_c = 1.0
    a_param = 25.0
    γ_param = 4.5
    dt = 0.01
    σ_param = 3.0
    boundary = CubicBoundary(5.0)

    inter = DPDInteraction(a=a_param, γ=γ_param, σ=σ_param, r_c=r_c, dt=dt)

    @test !use_neighbors(inter)
    @test use_neighbors(DPDInteraction(use_neighbors=true))

    a1 = Atom(index=1, mass=1.0, charge=0.0, σ=0.0, ϵ=0.0)
    a2 = Atom(index=2, mass=1.0, charge=0.0, σ=0.0, ϵ=0.0)
    c1 = SVector(1.0, 1.0, 1.0)
    c2 = SVector(1.5, 1.0, 1.0)
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(0.0, 0.0, 0.0)
    dr = vector(c1, c2, boundary)
    r = norm(dr)

    # Conservative force only (zero velocities, deterministic random from hash)
    f = force(inter, dr, a1, a2, NoUnits, false, c1, c2, boundary, v1, v2, 0)
    w_R = 1 - r / r_c
    f_C_expected = a_param * w_R / r
    # The force should be in the +x direction (repulsive, pushing j away from i)
    @test f[1] > 0.0
    @test isapprox(f[2], 0.0; atol=1e-10)
    @test isapprox(f[3], 0.0; atol=1e-10)

    # Conservative potential energy
    pe = potential_energy(inter, dr, a1, a2, NoUnits)
    pe_expected = (a_param / 2) * r_c * w_R^2
    @test isapprox(pe, pe_expected; atol=1e-10)

    # Force is zero at and beyond cutoff
    c3 = SVector(2.0, 1.0, 1.0)
    dr_cutoff = vector(c1, c3, boundary)
    f_cutoff = force(inter, dr_cutoff, a1, a2, NoUnits, false, c1, c3, boundary, v1, v2, 0)
    @test all(isapprox.(f_cutoff, 0.0; atol=1e-10))
    @test isapprox(potential_energy(inter, dr_cutoff, a1, a2, NoUnits), 0.0; atol=1e-10)

    c4 = SVector(2.5, 1.0, 1.0)
    dr_beyond = vector(c1, c4, boundary)
    f_beyond = force(inter, dr_beyond, a1, a2, NoUnits, false, c1, c4, boundary, v1, v2, 0)
    @test all(isapprox.(f_beyond, 0.0; atol=1e-10))

    # Dissipative force: approaching particles should experience damping
    v1_approach = SVector(1.0, 0.0, 0.0)
    v2_still = SVector(0.0, 0.0, 0.0)
    inter_nodiss = DPDInteraction(a=0.0, γ=γ_param, σ=0.0, r_c=r_c, dt=dt)
    f_diss = force(inter_nodiss, dr, a1, a2, NoUnits, false, c1, c2, boundary,
                   v1_approach, v2_still, 0)
    # Dissipative force on j should push j away from i (same direction as approach)
    @test f_diss[1] > 0.0
    # Force should be purely along x axis
    @test isapprox(f_diss[2], 0.0; atol=1e-10)
    @test isapprox(f_diss[3], 0.0; atol=1e-10)

    # Receding particles: dissipative force should pull them back
    v1_recede = SVector(-1.0, 0.0, 0.0)
    f_diss_recede = force(inter_nodiss, dr, a1, a2, NoUnits, false, c1, c2, boundary,
                          v1_recede, v2_still, 0)
    @test f_diss_recede[1] < 0.0

    # Random noise symmetry: dpd_gaussian is symmetric in particle indices
    for step in 1:10
        @test Molly.dpd_gaussian(1, 2, step) == Molly.dpd_gaussian(2, 1, step)
        @test Molly.dpd_gaussian(5, 13, step) == Molly.dpd_gaussian(13, 5, step)
    end

    # Random noise varies with step number
    vals = [Molly.dpd_gaussian(1, 2, s) for s in 1:100]
    @test std(vals) > 0.5
end
