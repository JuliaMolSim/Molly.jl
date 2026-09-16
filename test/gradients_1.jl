@testset "Energy gradients" begin
    inter = LennardJones()
    boundary = CubicBoundary(5.0)
    a1, a2 = Atom(σ=0.3, ϵ=0.5), Atom(σ=0.3, ϵ=0.5)

    function force_direct(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, boundary)
        F = force(inter, vec, a1, a2, NoUnits)
        return F[1]
    end

    function pe(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, boundary)
        potential_energy(inter, vec, a1, a2, NoUnits)
    end

    function force_grad(dist)
        grads = autodiff(
            Reverse,
            pe,
            Active,
            Active(dist),
        )
        return -grads[1][1]
    end

    dists = collect(0.2:0.01:1.2)
    forces_direct = force_direct.(dists)
    forces_grad = force_grad.(dists)
    @test all(forces_direct .≈ forces_grad)
end

@testset "Interaction gradients" begin
    T = Float64
    cc = T(ustrip(Molly.coulomb_const))
    boundary = CubicBoundary(T(4.0))
    n_atoms = 12
    coords = place_atoms(n_atoms, boundary; min_dist=T(0.4), rng=Xoshiro(2024))
    atoms = [Atom(index=i, mass=T(10.0), charge=T(i % 2 == 0 ? 0.2 : -0.2), σ=T(0.3),
                  ϵ=T(0.4), λ=T(0.6)) for i in 1:n_atoms]
    nf = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=1,
                                dist_cutoff=T(1.5))
    nb_cutoff = T(1.2)

    coords_spec = [
        SVector(T(1.0), T(1.0), T(1.00)),
        SVector(T(1.6), T(1.1), T(1.05)),
        SVector(T(2.1), T(1.7), T(1.20)),
        SVector(T(2.4), T(2.4), T(1.90)),
        SVector(T(3.0), T(2.6), T(2.40)),
    ]
    atoms_spec = [Atom(index=i, mass=T(10.0), charge=T(0.1), σ=T(0.3), ϵ=T(0.4))
                  for i in eachindex(coords_spec)]
    is, js, ks, ls = Int32[1], Int32[2], Int32[3], Int32[4]

    function pe_coords(coords, atoms, boundary, pairwise_inters, specific_inter_lists,
                       general_inters, neighbor_finder, neighbors)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=specific_inter_lists,
            general_inters=general_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        return potential_energy(sys, neighbors; n_threads=1)
    end

    # The analytic force should be the negative gradient of the potential energy
    function test_force_is_energy_grad(name, coords, atoms, boundary; pairwise_inters=(),
                                       specific_inter_lists=(), general_inters=(),
                                       neighbor_finder=NoNeighborFinder(), tol=1e-10)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=specific_inter_lists,
            general_inters=general_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        neighbors = find_neighbors(sys; n_threads=1)
        fs = forces(sys, neighbors; n_threads=1)
        d_coords = zero(coords)
        autodiff(
            set_runtime_activity(Reverse),
            pe_coords,
            Active,
            Duplicated(copy(coords), d_coords),
            Const(atoms),
            Const(boundary),
            Const(pairwise_inters),
            Const(specific_inter_lists),
            Const(general_inters),
            Const(neighbor_finder),
            Const(neighbors),
        )
        max_force = maximum(maximum(abs, f) for f in fs)
        frac_diff = maximum(maximum(abs, d + f) for (d, f) in zip(d_coords, fs)) /
                        max(max_force, eps(T))
        @test max_force > eps(T) # Guard against a trivially zero force
        @test frac_diff < tol
    end

    pairwise_inters = [
        ("LennardJones NoCutoff"    , LennardJones(use_neighbors=true)),
        ("LennardJones Distance"    , LennardJones(cutoff=DistanceCutoff(nb_cutoff),
                                                    use_neighbors=true)),
        ("LennardJones ShiftedPot"  , LennardJones(cutoff=ShiftedPotentialCutoff(nb_cutoff),
                                                    use_neighbors=true)),
        ("LennardJones ShiftedForce", LennardJones(cutoff=ShiftedForceCutoff(nb_cutoff),
                                                    use_neighbors=true)),
        ("LennardJones CubicSpline" , LennardJones(cutoff=CubicSplineCutoff(T(0.8), nb_cutoff),
                                                    use_neighbors=true)),
        ("LennardJones Polynomial"  , LennardJones(cutoff=PolynomialCutoff(T(0.8), nb_cutoff),
                                                    use_neighbors=true)),
        ("SoftSphere"               , SoftSphere(cutoff=DistanceCutoff(nb_cutoff),
                                                    use_neighbors=true)),
        ("Mie"                      , Mie(m=T(6), n=T(12), cutoff=DistanceCutoff(nb_cutoff),
                                            use_neighbors=true)),
        ("DoubleExponential"        , DoubleExponential(α=T(16.766), β=T(4.427),
                                            cutoff=DistanceCutoff(nb_cutoff),
                                            use_neighbors=true)),
        ("AshbaughHatch"            , AshbaughHatch(cutoff=DistanceCutoff(nb_cutoff),
                                                    use_neighbors=true)),
        ("LennardJonesSCBeutler"    , LennardJonesSoftCoreBeutler(α=T(0.5),
                                            cutoff=DistanceCutoff(nb_cutoff),
                                            use_neighbors=true)),
        ("LennardJonesSCGapsys"     , LennardJonesSoftCoreGapsys(α=T(0.85),
                                            cutoff=DistanceCutoff(nb_cutoff),
                                            use_neighbors=true)),
        ("Coulomb"                  , Coulomb(cutoff=DistanceCutoff(nb_cutoff),
                                                use_neighbors=true, coulomb_const=cc)),
        ("CoulombReactionField"     , CoulombReactionField(dist_cutoff=nb_cutoff,
                                            use_neighbors=true, coulomb_const=cc)),
        ("CoulombSCBeutler"         , CoulombSoftCoreBeutler(cutoff=DistanceCutoff(nb_cutoff),
                                            use_neighbors=true, coulomb_const=cc)),
        ("CoulombSCGapsys"          , CoulombSoftCoreGapsys(cutoff=DistanceCutoff(nb_cutoff),
                                            σQ=T(1.0), use_neighbors=true,
                                            coulomb_const=cc)),
        ("Gravity"                  , Gravity(G=T(1.0), use_neighbors=true)),
    ]
    for (name, inter) in pairwise_inters
        test_force_is_energy_grad(name, coords, atoms, boundary; pairwise_inters=(inter,),
                                    neighbor_finder=nf)
    end

    # Interactions that do not use the neighbor list take a different code path
    lj_nonl = LennardJones(cutoff=DistanceCutoff(nb_cutoff))
    test_force_is_energy_grad("LennardJones no neighbors", coords, atoms, boundary;
                                pairwise_inters=(lj_nonl,))

    specific_inters = [
        ("HarmonicBond"    , InteractionList2Atoms(is, js,
                                [HarmonicBond(k=T(100.0), r0=T(0.5))])),
        ("MorseBond"       , InteractionList2Atoms(is, js,
                                [MorseBond(D=T(100.0), a=T(2.0), r0=T(0.5))])),
        ("FENEBond"        , InteractionList2Atoms(is, js,
                                [FENEBond(k=T(100.0), r0=T(1.4), σ=T(0.3), ϵ=T(0.4))])),
        ("HarmonicAngle"   , InteractionList3Atoms(is, js, ks,
                                [HarmonicAngle(k=T(10.0), θ0=T(2.0))])),
        ("CosineAngle"     , InteractionList3Atoms(is, js, ks,
                                [CosineAngle(k=T(10.0), θ0=T(2.0))])),
        ("UreyBradley"     , InteractionList3Atoms(is, js, ks,
                                [UreyBradley(kangle=T(10.0), θ0=T(2.0), kbond=T(10.0),
                                                r0=T(1.0))])),
        ("PeriodicTorsion" , InteractionList4Atoms(is, js, ks, ls,
                                [PeriodicTorsion(periodicities=[1, 2, 3],
                                                    phases=T[1.0, 0.0, -1.0],
                                                    ks=T[10.0, 5.0, 8.0], n_terms=6)])),
        ("RBTorsion"       , InteractionList4Atoms(is, js, ks, ls,
                                [RBTorsion(c0=T(1.0), c1=T(2.0), c2=T(3.0), c3=T(4.0),
                                            c4=T(0.5), c5=T(0.25))])),
        ("HarmonicTorsion" , InteractionList4Atoms(is, js, ks, ls,
                                [HarmonicTorsion(k=T(10.0), θ0=T(1.0))])),
        ("HarmonicPositionRestraint", InteractionList1Atoms(is,
                                [HarmonicPositionRestraint(k=T(100.0),
                                                x0=coords_spec[1] .+ T(0.1))])),
    ]
    for (name, inter_list) in specific_inters
        test_force_is_energy_grad(name, coords_spec, atoms_spec, boundary;
                                    specific_inter_lists=(inter_list,))
    end

    lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true)
    test_force_is_energy_grad("LJDispersionCorrection", coords, atoms, boundary;
                                pairwise_inters=(lj,), neighbor_finder=nf,
                                general_inters=(LJDispersionCorrection(atoms, nb_cutoff),))

    mb = MullerBrown(
        A=SVector(T(-200.0), T(-100.0), T(-170.0), T( 15.0)),
        a=SVector(T(  -1.0), T(  -1.0), T(  -6.5), T(  0.7)),
        b=SVector(T(   0.0), T(   0.0), T(  11.0), T(  0.6)),
        c=SVector(T( -10.0), T( -10.0), T(  -6.5), T(  0.7)),
        x0=SVector(T(  1.0), T(   0.0), T(  -0.5), T( -1.0)),
        y0=SVector(T(  0.0), T(   0.5), T(   1.5), T(  1.0)),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    test_force_is_energy_grad("MullerBrown", [SVector(T(-0.5), T(0.5)),
                                                SVector(T(0.2), T(0.9))],
                                [Atom(mass=T(1.0)) for i in 1:2],
                                RectangularBoundary(T(Inf)); general_inters=(mb,))

    n_bd = 10
    atoms_bd = atoms[1:n_bd]
    nf_bd = DistanceNeighborFinder(eligible=trues(n_bd, n_bd), n_steps=1,
                                    dist_cutoff=T(1.5))
    lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true)

    boundary_trc = TriclinicBoundary(
        SVector(T(4.0), T(0.0), T(0.0)),
        SVector(T(0.4), T(4.0), T(0.0)),
        SVector(T(0.2), T(0.3), T(4.0)),
    )
    boundaries = [
        ("TriclinicBoundary" , boundary_trc,
            place_atoms(n_bd, boundary_trc; min_dist=T(0.4), rng=Xoshiro(7))),
        ("RectangularBoundary", RectangularBoundary(T(4.0)),
            place_atoms(n_bd, RectangularBoundary(T(4.0)); min_dist=T(0.4), rng=Xoshiro(8))),
        ("Infinite boundary" , CubicBoundary(T(4.0), T(4.0), T(Inf)),
            place_atoms(n_bd, CubicBoundary(T(4.0)); min_dist=T(0.4), rng=Xoshiro(9))),
    ]
    for (name, boundary_test, coords_test) in boundaries
        test_force_is_energy_grad(name, coords_test, atoms_bd, boundary_test;
                                    pairwise_inters=(lj,), neighbor_finder=nf_bd)
    end

    crf = CoulombReactionField(dist_cutoff=nb_cutoff, use_neighbors=true, coulomb_const=cc)
    lj  = LennardJones(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true)

    function pe_atoms(atoms, coords, boundary, pairwise_inters, neighbor_finder, neighbors)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=pairwise_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        return potential_energy(sys, neighbors; n_threads=1)
    end

    for (name, inter, getter, setter) in (
            ("charge", crf, charge,
                (at, v) -> Atom(index=at.index, mass=at.mass, charge=v, σ=at.σ, ϵ=at.ϵ)),
            ("σ", lj, at -> at.σ,
                (at, v) -> Atom(index=at.index, mass=at.mass, charge=at.charge, σ=v, ϵ=at.ϵ)),
            ("ϵ", lj, at -> at.ϵ,
                (at, v) -> Atom(index=at.index, mass=at.mass, charge=at.charge, σ=at.σ, ϵ=v)),
        )
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=(inter,),
            neighbor_finder=nf,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        neighbors = find_neighbors(sys; n_threads=1)
        # Perturb the atom with the largest force, some atoms have no neighbors
        atom_i = argmax(map(f -> sum(abs2, f), forces(sys, neighbors; n_threads=1)))
        d_atoms = zero.(atoms)
        autodiff(
            set_runtime_activity(Reverse),
            pe_atoms,
            Active,
            Duplicated(copy(atoms), d_atoms),
            Const(coords),
            Const(boundary),
            Const((inter,)),
            Const(nf),
            Const(neighbors),
        )
        grad_fd = central_fdm(6, 1)(getter(atoms[atom_i])) do val
            atoms_mod = copy(atoms)
            atoms_mod[atom_i] = setter(atoms[atom_i], val)
            pe_atoms(atoms_mod, coords, boundary, (inter,), nf, neighbors)
        end
        grad_enzyme = getter(d_atoms[atom_i])
        frac_diff = abs(grad_enzyme - grad_fd) / abs(grad_fd)
        @test abs(grad_fd) > eps(T) # Guard against a trivially zero gradient
        @test frac_diff < 1e-8
    end
end

@testset "Differentiable PME" begin
    T = Float64
    ff = MolecularForceField(
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...,
        units=false,
    )

    for AT in array_list
        sys = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            units=false,
            array_type=AT,
            float_type=T,
            nonbonded_method=SetupPME(),
            grad_safe=true,
        )

        pme = sys.general_inters[1]
        Fs = zero(sys.coords)
        d_sys = zero(sys)
        d_pme = zero(pme)

        pe = Molly.ewald_pe_forces!(Fs, nothing, sys, pme, Val(false))
        Fs_ad = zero(sys.coords)

        pe_ad = autodiff(
            ReverseWithPrimal,
            Molly.ewald_pe_forces!,
            Active,
            Const(Fs_ad),
            Const(nothing),
            Duplicated(sys, d_sys),
            Duplicated(pme, d_pme),
            Const(Val(false)),
        )[2]

        @test pe_ad ≈ pe atol=1e-7
        @test from_device(Fs_ad) ≈ from_device(Fs) atol=1e-10
        @test from_device(-d_sys.coords) ≈ from_device(Fs) atol=1e-10

        coords_cpu, atoms_cpu = from_device(sys.coords), from_device(sys.atoms)

        function coord_fdm(c)
            coords_mod = copy(coords_cpu)
            coords_mod[1] = SVector(c, coords_mod[1][2], coords_mod[1][3])
            sys_mod = System(deepcopy(sys); coords=to_device(coords_mod, AT))
            return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
        end

        c = coords_cpu[1][1]
        coord_fdm(c)
        coord_grad = central_fdm(5, 1)(coord_fdm, c)
        @test from_device(d_sys.coords)[1][1] ≈ coord_grad atol=1e-6

        function charge_fdm(ch)
            atoms_mod = copy(atoms_cpu)
            at = atoms_cpu[1]
            atoms_mod[1] = Atom(mass=at.mass, charge=ch, σ=at.σ, ϵ=at.σ)
            sys_mod = System(deepcopy(sys); atoms=to_device(atoms_mod, AT))
            return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
        end

        at = atoms_cpu[1]
        charge_fdm(charge(at))
        charge_grad = central_fdm(5, 1)(charge_fdm, charge(at))
        @test charge(from_device(d_sys.atoms)[1]) ≈ charge_grad atol=1e-6

        # A loss that depends on the forces rather than the energy, which differentiates
        # the force interpolation and needs the second derivative of the B-splines
        function force_loss(Fs, pme, atoms, coords, boundary)
            Molly.ewald_pe_forces!(Fs, nothing, pme, atoms, coords, boundary, NoUnits,
                                   NoUnits, Val(false), true, Val(true), Val(T);
                                   n_threads=1)
            return sum(sum.(abs2, Fs))
        end

        d_coords_f = zero(sys.coords)
        autodiff(
            set_runtime_activity(Reverse),
            force_loss,
            Active,
            Duplicated(zero(sys.coords), zero(sys.coords)),
            Duplicated(pme, zero(pme)),
            Const(sys.atoms),
            Duplicated(copy(sys.coords), d_coords_f),
            Const(sys.boundary),
        )

        function force_loss_fdm(c)
            coords_mod = copy(coords_cpu)
            coords_mod[1] = SVector(c, coords_mod[1][2], coords_mod[1][3])
            return force_loss(zero(sys.coords), pme, sys.atoms,
                              to_device(coords_mod, AT), sys.boundary)
        end

        force_loss_fdm(c)
        force_grad = central_fdm(5, 1)(force_loss_fdm, c)
        @test from_device(d_coords_f)[1][1] ≈ force_grad rtol=1e-6
    end
end

@testset "Ewald gradients" begin
    T = Float64
    cc = T(ustrip(Molly.coulomb_const))
    n_atoms = 6
    boundary = CubicBoundary(T(3.0))
    coords = place_atoms(n_atoms, boundary; min_dist=T(0.4), rng=Xoshiro(11))
    charges = T[0.4, -0.4, 0.3, -0.3, 0.2, -0.2]
    atoms = [Atom(index=i, mass=T(10.0), charge=charges[i], σ=T(0.3), ϵ=T(0.4))
             for i in 1:n_atoms]
    neighbor_finder = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=1,
                                             dist_cutoff=T(1.4))
    dist_cutoff = T(1.0)

    function pe_ewald(coords, atoms, boundary, pairwise_inters, general_inters,
                      neighbor_finder, neighbors)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=pairwise_inters,
            general_inters=general_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        return potential_energy(sys, neighbors; n_threads=1)
    end

    # The fast erfc approximation is not the derivative of the approximation used in the
    #   force, so the energy gradient and the force only agree to the accuracy of the fit
    for (name, approximate_erfc, tol) in (("approximate erfc", true , 1e-5),
                                          ("exact erfc"      , false, 1e-10))
        coul_ewald = CoulombEwald(dist_cutoff=dist_cutoff, use_neighbors=true,
                                  coulomb_const=cc, approximate_erfc=approximate_erfc)
        for (inter_name, general_inter) in (
                ("Ewald", Ewald(dist_cutoff)),
                ("PME"  , PME(dist_cutoff, atoms, boundary; grad_safe=true, n_threads=1)),
            )
            sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=(coul_ewald,),
                general_inters=(general_inter,),
                neighbor_finder=neighbor_finder,
                force_units=NoUnits,
                energy_units=NoUnits,
            )
            neighbors = find_neighbors(sys; n_threads=1)
            fs = forces(sys, neighbors; n_threads=1)
            d_coords = zero(coords)
            autodiff(
                set_runtime_activity(Reverse),
                pe_ewald,
                Active,
                Duplicated(copy(coords), d_coords),
                Const(atoms),
                Const(boundary),
                Const((coul_ewald,)),
                # PME holds mesh buffers that carry gradient information, so it has to be
                #   Duplicated rather than Const
                Duplicated((general_inter,), (zero(general_inter),)),
                Const(neighbor_finder),
                Const(neighbors),
            )
            max_force = maximum(maximum(abs, f) for f in fs)
            frac_diff = maximum(maximum(abs, d + f) for (d, f) in zip(d_coords, fs)) / max_force
            @test frac_diff < tol
        end
    end
end
