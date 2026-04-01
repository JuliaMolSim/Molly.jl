@testset "Soft-core Ewald" begin
    c1 = SVector(1.0, 1.0, 1.0)u"nm"
    c2 = SVector(1.3, 1.0, 1.0)u"nm"
    c3 = SVector(1.4, 1.0, 1.0)u"nm"
    c4 = SVector(1.05, 1.0, 1.0)u"nm"
    boundary = CubicBoundary(2.0u"nm")
    dr12 = vector(c1, c2, boundary)
    dr13 = vector(c1, c3, boundary)
    dr14 = vector(c1, c4, boundary)

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
        for dr_test in (dr12, dr13, dr14)
            ref_f = force(ce_ref, dr_test, a1_l1, a1_l1)
            ref_pe = potential_energy(ce_ref, dr_test, a1_l1, a1_l1)
            @test isapprox(
                force(cscbe, dr_test, a1_l1, a1_l1),
                ref_f;
                atol=1e-9u"kJ * mol^-1 * nm^-1",
            )
            @test isapprox(
                force(cscge, dr_test, a1_l1, a1_l1),
                ref_f;
                atol=1e-9u"kJ * mol^-1 * nm^-1",
            )
            @test isapprox(
                potential_energy(cscbe, dr_test, a1_l1, a1_l1),
                ref_pe;
                atol=1e-9u"kJ * mol^-1",
            )
            @test isapprox(
                potential_energy(cscge, dr_test, a1_l1, a1_l1),
                ref_pe;
                atol=1e-9u"kJ * mol^-1",
            )
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
            for dr_test in (dr12, dr13, dr14)
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

@testset "PME Scheduler Charge Scaling" begin
    boundary = CubicBoundary(2.5u"nm")
    coords = [
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

        pme_raw = PME(rc, atoms_raw, boundary; scheduler=scheduler)
        pme_ref = PME(rc, atoms_ref, boundary)
        sys_raw = System(atoms=atoms_raw, coords=coords, boundary=boundary,
                         pairwise_inters=(), general_inters=(pme_raw,))
        sys_ref = System(atoms=atoms_ref, coords=coords, boundary=boundary,
                         pairwise_inters=(), general_inters=(pme_ref,))

        @test isapprox(
            potential_energy(sys_raw),
            potential_energy(sys_ref);
            atol=1e-9u"kJ * mol^-1",
        )
        @test maximum(norm.(forces(sys_raw) .- forces(sys_ref))) <
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

        pme_raw = PME(rc, atoms_raw, boundary; scheduler=scheduler)
        pme_ref = PME(rc, atoms_ref, boundary)
        sys_raw = System(atoms=atoms_raw, coords=coords, boundary=boundary,
                         pairwise_inters=(), general_inters=(pme_raw,))
        sys_ref = System(atoms=atoms_ref, coords=coords, boundary=boundary,
                         pairwise_inters=(), general_inters=(pme_ref,))

        @test isapprox(
            potential_energy(sys_raw),
            potential_energy(sys_ref);
            atol=1e-9u"kJ * mol^-1",
        )
        @test maximum(norm.(forces(sys_raw) .- forces(sys_ref))) <
              1e-9u"kJ * mol^-1 * nm^-1"
    end
end

@testset "Soft-core PME End-to-End" begin
    boundary = CubicBoundary(2.2u"nm")
    coords = [
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
        coords=coords,
        boundary=boundary,
        pairwise_inters=(CoulombEwald(dist_cutoff=rc),),
        general_inters=(PME(rc, atoms_l1, boundary),),
    )
    sys_beutler_l1 = System(
        atoms=atoms_l1,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(CoulombSoftCoreBeutlerEwald(dist_cutoff=rc, α=0.3),),
        general_inters=(PME(rc, atoms_l1, boundary),),
    )
    sys_gapsys_l1 = System(
        atoms=atoms_l1,
        coords=coords,
        boundary=boundary,
        pairwise_inters=(CoulombSoftCoreGapsysEwald(dist_cutoff=rc, α=0.3, σQ=1.0u"nm"),),
        general_inters=(PME(rc, atoms_l1, boundary),),
    )

    @test isapprox(
        potential_energy(sys_beutler_l1),
        potential_energy(sys_ref);
        atol=1e-9u"kJ * mol^-1",
    )
    @test isapprox(
        potential_energy(sys_gapsys_l1),
        potential_energy(sys_ref);
        atol=1e-9u"kJ * mol^-1",
    )
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
            coords=coords,
            boundary=boundary,
            pairwise_inters=(pair_inter,),
            general_inters=(PME(rc, atoms_l05, boundary),),
        )
        pe_soft = potential_energy(sys_soft)
        fs_soft = forces(sys_soft)
        @test isfinite(pe_soft)
        @test all(fi -> all(isfinite, fi), fs_soft)
    end
end
