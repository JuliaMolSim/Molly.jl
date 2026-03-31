@testset "Soft-core Reaction Field" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(1.4, 1.0, 1.0)u"nm"
    c4 = SVector(1.1, 1.0, 1.0)u"nm"
    boundary = CubicBoundary(2.0u"nm")
    dr12 = vector(c1, c2, boundary)
    dr13 = vector(c1, c3, boundary)
    dr14 = vector(c1, c4, boundary)

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
            @test isapprox(
                force(cscrfb, dr_test, a1_rf, a1_rf),
                ref_f;
                atol=1e-9u"kJ * mol^-1 * nm^-1",
            )
            @test isapprox(
                force(cscrfg, dr_test, a1_rf, a1_rf),
                ref_f;
                atol=1e-9u"kJ * mol^-1 * nm^-1",
            )
            @test isapprox(
                potential_energy(cscrfb, dr_test, a1_rf, a1_rf),
                ref_pe;
                atol=1e-9u"kJ * mol^-1",
            )
            @test isapprox(
                potential_energy(cscrfg, dr_test, a1_rf, a1_rf),
                ref_pe;
                atol=1e-9u"kJ * mol^-1",
            )
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
        @test isapprox(
            potential_energy(cscrfb_test, dr_rc, a1_l05, a1_l05),
            0.0u"kJ * mol^-1";
            atol=1e-10u"kJ * mol^-1",
        )
        @test isapprox(
            potential_energy(cscrfg_test, dr_rc, a1_l05, a1_l05),
            0.0u"kJ * mol^-1";
            atol=1e-10u"kJ * mol^-1",
        )
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
end
