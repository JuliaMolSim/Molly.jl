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
                                       neighbor_finder=nothing, tol=1e-10)
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
        ("LennardJones NoCutoff"    , LennardJones()),
        ("LennardJones Distance"    , LennardJones(cutoff=DistanceCutoff(nb_cutoff))),
        ("LennardJones ShiftedPot"  , LennardJones(cutoff=ShiftedPotentialCutoff(nb_cutoff))),
        ("LennardJones ShiftedForce", LennardJones(cutoff=ShiftedForceCutoff(nb_cutoff))),
        ("LennardJones CubicSpline" , LennardJones(cutoff=CubicSplineCutoff(T(0.8), nb_cutoff))),
        ("LennardJones Polynomial"  , LennardJones(cutoff=PolynomialCutoff(T(0.8), nb_cutoff))),
        ("SoftSphere"               , SoftSphere(cutoff=DistanceCutoff(nb_cutoff))),
        ("Mie"                      , Mie(m=T(6), n=T(12), cutoff=DistanceCutoff(nb_cutoff))),
        ("DoubleExponential"        , DoubleExponential(α=T(16.766), β=T(4.427),
                                            cutoff=DistanceCutoff(nb_cutoff))),
        ("AshbaughHatch"            , AshbaughHatch(cutoff=DistanceCutoff(nb_cutoff))),
        ("LennardJonesSCBeutler"    , LennardJonesSoftCoreBeutler(α=T(0.5),
                                            cutoff=DistanceCutoff(nb_cutoff))),
        ("LennardJonesSCGapsys"     , LennardJonesSoftCoreGapsys(α=T(0.85),
                                            cutoff=DistanceCutoff(nb_cutoff))),
        ("Coulomb"                  , Coulomb(cutoff=DistanceCutoff(nb_cutoff),
                                                coulomb_const=cc)),
        ("CoulombReactionField"     , CoulombReactionField(dist_cutoff=nb_cutoff,
                                            coulomb_const=cc)),
        ("CoulombSCBeutler"         , CoulombSoftCoreBeutler(cutoff=DistanceCutoff(nb_cutoff),
                                            coulomb_const=cc)),
        ("CoulombSCGapsys"          , CoulombSoftCoreGapsys(cutoff=DistanceCutoff(nb_cutoff),
                                            σQ=T(1.0), coulomb_const=cc)),
        ("Gravity"                  , Gravity(G=T(1.0))),
    ]
    for (name, inter) in pairwise_inters
        test_force_is_energy_grad(name, coords, atoms, boundary; pairwise_inters=(inter,),
                                    neighbor_finder=nf)
    end

    # Interactions that do not use the neighbor list take a different code path
    lj_nonl = LennardJones(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=false)
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

    lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff))
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
    lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff))

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

    crf = CoulombReactionField(dist_cutoff=nb_cutoff, coulomb_const=cc)
    lj  = LennardJones(cutoff=DistanceCutoff(nb_cutoff))

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

@testset "Reaction field exclusion gradients" begin
    T = Float64
    cc = T(ustrip(Molly.coulomb_const))
    boundary = CubicBoundary(T(3.0))
    coords = [SVector(T(1.0), T(1.0), T(1.0)), SVector(T(1.3), T(1.2), T(1.1)),
              SVector(T(1.6), T(1.0), T(0.8)), SVector(T(2.9), T(2.8), T(2.9)),
              SVector(T(0.1), T(0.2), T(0.15))]
    charges = T[0.8, -0.8, 0.5, -0.7, 0.4]
    n_atoms = length(coords)
    eligible = trues(n_atoms, n_atoms)
    for (i, j) in ((1, 2), (2, 3), (1, 3), (4, 5))
        eligible[i, j] = eligible[j, i] = false
    end
    nf = DistanceNeighborFinder(eligible=eligible, n_steps=1, dist_cutoff=T(1.5))
    crf = CoulombReactionField(dist_cutoff=T(1.1), solvent_dielectric=T(30.0),
                               solute_dielectric=T(2.5), use_neighbors=true, coulomb_const=cc)
    excl = Molly.reaction_field_exclusions(crf, eligible)

    function pe_rf(coords, atoms, boundary, pairwise_inters, specific_inter_lists,
                   neighbor_finder, neighbors)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=specific_inter_lists,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        return potential_energy(sys, neighbors; n_threads=1)
    end

    atoms = [Atom(index=i, mass=T(10.0), charge=charges[i], σ=T(0.0), ϵ=T(0.0))
             for i in 1:n_atoms]
    sys = System(atoms=atoms, coords=coords, boundary=boundary, pairwise_inters=(crf,),
                 specific_inter_lists=(excl,), neighbor_finder=nf, force_units=NoUnits,
                 energy_units=NoUnits)
    neighbors = find_neighbors(sys; n_threads=1)
    fs = forces(sys, neighbors; n_threads=1)

    d_coords = zero(coords)
    d_atoms = zero.(atoms)
    autodiff(
        set_runtime_activity(Reverse),
        Const(pe_rf),
        Active,
        Duplicated(copy(coords), d_coords),
        Duplicated(copy(atoms), d_atoms),
        Const(boundary),
        Const((crf,)),
        Const((excl,)),
        Const(nf),
        Const(neighbors),
    )
    @test maximum(maximum(abs, d + f) for (d, f) in zip(d_coords, fs)) /
            maximum(maximum(abs, f) for f in fs) < 1e-10

    for atom_i in 1:n_atoms
        grad_fd = central_fdm(6, 1)(charges[atom_i]) do q
            atoms_mod = copy(atoms)
            atoms_mod[atom_i] = Atom(index=atom_i, mass=T(10.0), charge=q, σ=T(0.0), ϵ=T(0.0))
            pe_rf(coords, atoms_mod, boundary, (crf,), (excl,), nf, neighbors)
        end
        @test abs(grad_fd) > eps(T) # Guard against a trivially zero gradient
        @test abs(charge(d_atoms[atom_i]) - grad_fd) / abs(grad_fd) < 1e-8
    end

    # dU/dλ with the bonded pair 1-2 inserted, as needed for thermodynamic integration
    crf_scaled = Molly.CoulombReactionFieldScaled(dist_cutoff=T(1.1), solvent_dielectric=T(30.0),
                                    solute_dielectric=T(2.5), use_neighbors=true, coulomb_const=cc,
                                    scheduler=Molly.LinearLambdaScheduler())
    excl_scaled = Molly.reaction_field_exclusions(crf_scaled, eligible)

    function pe_λ(λ, coords, boundary, pairwise_inters, specific_inter_lists, neighbor_finder,
                  neighbors, charges)
        atoms = [Atom(i, 1, T(10.0), charges[i], T(0.0), T(0.0), (i <= 2 ? λ : one(λ)),
                      (i <= 2 ? Molly.InsertRole : Molly.EnvRole)) for i in eachindex(charges)]
        return pe_rf(coords, atoms, boundary, pairwise_inters, specific_inter_lists,
                     neighbor_finder, neighbors)
    end

    # The inserted atoms only have excluded atoms within the cutoff, so dU/dλ comes from the
    #   exclusion and self terms
    λ0 = T(0.6)
    grad_enzyme = autodiff(
        set_runtime_activity(Reverse),
        Const(pe_λ),
        Active,
        Active(λ0),
        Const(coords), Const(boundary), Const((crf_scaled,)), Const((excl_scaled,)),
        Const(nf), Const(neighbors), Const(charges),
    )[1][1]
    grad_fd = central_fdm(6, 1)(λ0) do λ
        pe_λ(λ, coords, boundary, (crf_scaled,), (excl_scaled,), nf, neighbors, charges)
    end
    @test abs(grad_fd) > eps(T)
    @test abs(grad_enzyme - grad_fd) / abs(grad_fd) < 1e-8
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
        coul_ewald = CoulombEwald(dist_cutoff=dist_cutoff, coulomb_const=cc,
                                  approximate_erfc=approximate_erfc)
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

@testset "Enzyme $quantity gradient CPU $structure_file smoothness=$smoothness_correction" for smoothness_correction in (false, true),
        quantity in (:energy, :force_loss), structure_file in ("ethane.pdb", "water.pdb")
    reaxff = ReaxFF(
        force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
        float_type=Float64,
        array_type=Array,
        smoothness_correction=smoothness_correction,
    )

    sys = System(
        joinpath(data_dir, "ReaxFF", "data", structure_file),
        reaxff;
        boundary=CubicBoundary(2.5),
        array_type=Array,
        units=false,
        eem_tolerance=1e-12
    )

    neighbors = find_neighbors(sys; n_threads=1)

    function loss_function(reaxff, sys, neighbors)
        eem = sys.general_inters[1]
        fill!(eem.state.q, 0.0)

        forces, energy = Molly.enzyme_forces_and_energy(sys, reaxff, eem; neighbors)
        if quantity == :energy
            return energy
        else
            return sqrt(sum(abs2, forces))
        end
    end

    loss_function(reaxff, sys, neighbors)

    grad_sys = Enzyme.make_zero(sys)
    grad_reaxff = grad_sys.general_inters[2]

    Enzyme.autodiff(
        Reverse,
        loss_function,
        Active,
        Duplicated(reaxff, grad_reaxff),
        Duplicated(sys, grad_sys),
        Const(neighbors)
    )

    function fd_general_loss(parameters)
        sys_new = deepcopy(sys)
        reaxff_new = sys_new.general_inters[2]
        dp = reaxff_new.device_params

        reaxff_new.device_params = Molly.DeviceParams(
            parameters, dp.atom_parameters, dp.bond_parameters,
            dp.off_diagonal_parameters, dp.angle_entries,
            dp.angle_parameters, dp.torsion_parameters,
            dp.h_bond_parameters, dp.global_parameters,
            dp.taper_parameters, dp.h_bond_donors, dp.h_bond_acceptors
        )

        return loss_function(reaxff_new, sys_new, neighbors)
    end

    function fd_atom_loss(parameters)
        sys_new = deepcopy(sys)
        reaxff_new = sys_new.general_inters[2]
        reaxff_new.device_params.atom_parameters .= parameters
        return loss_function(reaxff_new, sys_new, neighbors)
    end

    function fd_bond_loss(parameters)
        sys_new = deepcopy(sys)
        reaxff_new = sys_new.general_inters[2]
        bonds = reaxff_new.device_params.bond_parameters

        for i in eachindex(bonds)
            bonds[i] = Molly.BondParameters(parameters[i]..., bonds[i].is_defined)
        end

        return loss_function(reaxff_new, sys_new, neighbors)
    end

    function fd_off_diagonal_loss(parameters)
        sys_new = deepcopy(sys)
        reaxff_new = sys_new.general_inters[2]
        pairs = reaxff_new.device_params.off_diagonal_parameters

        for i in eachindex(pairs)
            pairs[i] = Molly.OffDiagonalParameters(parameters[i]..., pairs[i].is_defined)
        end

        return loss_function(reaxff_new, sys_new, neighbors)
    end

    function fd_angle_loss(parameters)
        sys_new = deepcopy(sys)
        reaxff_new = sys_new.general_inters[2]
        angles = reaxff_new.device_params.angle_parameters

        for i in eachindex(angles)
            angles[i] = Molly.AngleParameters(parameters[i]..., angles[i].is_defined)
        end

        return loss_function(reaxff_new, sys_new, neighbors)
    end

    function fd_torsion_loss(parameters)
        sys_new = deepcopy(sys)
        reaxff_new = sys_new.general_inters[2]
        torsions = reaxff_new.device_params.torsion_parameters

        for i in eachindex(torsions)
            torsions[i] = Molly.TorsionParameters(parameters[i]..., torsions[i].is_defined)
        end

        return loss_function(reaxff_new, sys_new, neighbors)
    end

    function fd_h_bond_loss(parameters)
        sys_new = deepcopy(sys)
        reaxff_new = sys_new.general_inters[2]
        h_bonds = reaxff_new.device_params.h_bond_parameters

        for i in eachindex(h_bonds)
            h_bonds[i] = Molly.HBondParameters(parameters[i]..., h_bonds[i].is_defined)
        end

        return loss_function(reaxff_new, sys_new, neighbors)
    end

    parameters = reaxff.device_params.general_parameters
    fdm = central_fdm(5, 1)
    finite_diff_gradients = grad(fdm, fd_general_loss, parameters)[1]

    enzyme_gradients = grad_reaxff.device_params.general_parameters

    for name in fieldnames(typeof(parameters))
        # these parameters require updating cached values first
        if name in (:upper_taper_radius, :cutoff_bond_order)
            continue
        end

        @test isapprox(
            getfield(enzyme_gradients, name),
            getfield(finite_diff_gradients, name);
            rtol=1e-6,
            atol=1e-8
        )
    end

    atom_params = reaxff.device_params.atom_parameters
    finite_diff_atom_gradients = grad(fdm, fd_atom_loss, atom_params)[1]
    enzyme_atom_gradients = grad_reaxff.device_params.atom_parameters

    for i in eachindex(atom_params)
        for name in fieldnames(typeof(atom_params[i]))
            if name in (:D_ij, :r_vdW, :alpha, :ro_sigma, :ro_pi, :ro_pipi)
                continue
            end

            @test isapprox(
                getfield(enzyme_atom_gradients[i], name),
                getfield(finite_diff_atom_gradients[i], name);
                rtol=1e-6,
                atol=1e-8
            )
        end
    end

    bond_params = reaxff.device_params.bond_parameters
    bond_names = fieldnames(eltype(bond_params))[1:end-1]
    bond_values = [[getfield(bond, name) for name in bond_names] for bond in bond_params]

    finite_diff_bond_gradients = grad(fdm, fd_bond_loss, bond_values)[1]
    enzyme_bond_gradients = grad_reaxff.device_params.bond_parameters

    for i in eachindex(bond_params)
        for (j, name) in enumerate(bond_names)
            if name in (:ot_corr, :ovcoord)
                continue
            end

            @test isapprox(
                getfield(enzyme_bond_gradients[i], name),
                finite_diff_bond_gradients[i][j];
                rtol=1e-6,
                atol=1e-8
            )
        end
    end

    off_diagonal_params = reaxff.device_params.off_diagonal_parameters
    off_diagonal_names = fieldnames(eltype(off_diagonal_params))[1:end-1]
    off_diagonal_values = [[getfield(pair, name) for name in off_diagonal_names] for pair in off_diagonal_params]

    finite_diff_off_diagonal_gradients = grad(fdm, fd_off_diagonal_loss, off_diagonal_values)[1]
    enzyme_off_diagonal_gradients = grad_reaxff.device_params.off_diagonal_parameters

    for i in eachindex(off_diagonal_params)
        for (j, name) in enumerate(off_diagonal_names)
            if name in (:gamma_w_inv_pow, :gamma_ij_inv_pow)
                continue
            end

            @test isapprox(
                getfield(enzyme_off_diagonal_gradients[i], name),
                finite_diff_off_diagonal_gradients[i][j];
                rtol=1e-6,
                atol=1e-8
            )
        end
    end

    angle_params = reaxff.device_params.angle_parameters
    angle_names = fieldnames(eltype(angle_params))[1:end-1]
    angle_values = [[getfield(angle, name) for name in angle_names] for angle in angle_params]

    angle_fdm = central_fdm(5, 1; factor=100)
    finite_diff_angle_gradients = grad(angle_fdm, fd_angle_loss, angle_values)[1]
    enzyme_angle_gradients = grad_reaxff.device_params.angle_parameters

    for i in eachindex(angle_params)
        for (j, name) in enumerate(angle_names)
            @test isapprox(
                getfield(enzyme_angle_gradients[i], name),
                finite_diff_angle_gradients[i][j];
                rtol=1e-6,
                atol=1e-8
            )
        end
    end

    torsion_params = reaxff.device_params.torsion_parameters
    torsion_names = fieldnames(eltype(torsion_params))[1:end-1]
    torsion_values = [[getfield(torsion, name) for name in torsion_names] for torsion in torsion_params]

    finite_diff_torsion_gradients = grad(fdm, fd_torsion_loss, torsion_values)[1]
    enzyme_torsion_gradients = grad_reaxff.device_params.torsion_parameters

    for i in eachindex(torsion_params)
        for (j, name) in enumerate(torsion_names)
            @test isapprox(
                getfield(enzyme_torsion_gradients[i], name),
                finite_diff_torsion_gradients[i][j];
                rtol=1e-6,
                atol=1e-8
            )
        end
    end

    h_bond_params = reaxff.device_params.h_bond_parameters
    h_bond_names = fieldnames(eltype(h_bond_params))[1:end-1]
    h_bond_values = [[getfield(h_bond, name) for name in h_bond_names] for h_bond in h_bond_params]

    finite_diff_h_bond_gradients = grad(fdm, fd_h_bond_loss, h_bond_values)[1]
    enzyme_h_bond_gradients = grad_reaxff.device_params.h_bond_parameters

    for i in eachindex(h_bond_params)
        for (j, name) in enumerate(h_bond_names)
            @test isapprox(
                getfield(enzyme_h_bond_gradients[i], name),
                finite_diff_h_bond_gradients[i][j];
                rtol=1e-6,
                atol=1e-8
            )
        end
    end
end
