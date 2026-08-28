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
    AT = Array
    ff = MolecularForceField(
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...,
        units=false,
    )
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
    @test Fs_ad ≈ Fs atol=1e-10
    @test -d_sys.coords ≈ Fs atol=1e-10

    function coord_fdm(c)
        coords_mod = copy(sys.coords)
        coords_mod[1] = SVector(c, coords_mod[1][2], coords_mod[1][3])
        sys_mod = System(deepcopy(sys); coords=coords_mod)
        return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
    end

    c = sys.coords[1][1]
    coord_fdm(c)
    coord_grad = central_fdm(5, 1)(coord_fdm, c)
    @test d_sys.coords[1][1] ≈ coord_grad atol=1e-6

    function charge_fdm(ch)
        atoms_mod = copy(sys.atoms)
        at = sys.atoms[1]
        atoms_mod[1] = Atom(mass=at.mass, charge=ch, σ=at.σ, ϵ=at.σ)
        sys_mod = System(deepcopy(sys); atoms=atoms_mod)
        return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
    end

    at = sys.atoms[1]
    charge_fdm(charge(at))
    charge_grad = central_fdm(5, 1)(charge_fdm, charge(at))
    @test charge(d_sys.atoms[1]) ≈ charge_grad atol=1e-6
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

@testset "Virial correctness" begin
    FT = Float64
    AT = Array

    function potential_deformation(sys, neighbors, q)
        T = eltype(q)
        z, o = zero(T), one(T)
        F = @SMatrix [
            o + q[1]  q[4]       q[5];
            z         o + q[2]   q[6];
            z         z          o + q[3]
        ]

        sys_out = System(
            sys;
            coords=[F * coord for coord in sys.coords],
            boundary=TriclinicBoundary(F * Molly.boxmatrix(sys.boundary)),
        )
        return potential_energy(sys_out, neighbors; n_threads=1)
    end

    function virial_enzyme(sys, neighbors)
        T = eltype(eltype(sys.coords))
        q = zeros(T, 6)
        dq = zero(q)

        _, pe = autodiff(
            set_runtime_activity(ReverseWithPrimal),
            potential_deformation,
            Active,
            Const(sys),
            Const(neighbors),
            Duplicated(q, dq),
        )

        W = @SMatrix [
            -dq[1]  -dq[4]  -dq[5];
            -dq[4]  -dq[2]  -dq[6];
            -dq[5]  -dq[6]  -dq[3]
        ]

        return W, pe, dq
    end

    function potential_deformation_pme(pme, atoms, coords, boundary, force_units,
                                       energy_units, q)
        T = eltype(q)
        z, o = zero(T), one(T)
        F = @SMatrix [
            o + q[1]  q[4]       q[5];
            z         o + q[2]   q[6];
            z         z          o + q[3]
        ]
        boundary_new = TriclinicBoundary(F * Molly.boxmatrix(boundary))
        return Molly.ewald_pe_forces!(
            nothing,
            nothing,
            pme,
            atoms,
            [F * coord for coord in coords],
            boundary_new,
            force_units,
            energy_units,
            Val(false),
            false;
            n_threads=1,
        )
    end

    function virial_enzyme_pme(sys)
        pme = only(sys.general_inters)
        q = zeros(eltype(eltype(sys.coords)), 6)
        dq = zero(q)
        autodiff(
            set_runtime_activity(ReverseWithPrimal),
            potential_deformation_pme,
            Active,
            Duplicated(pme, zero(pme)),
            Const(sys.atoms),
            Const(sys.coords),
            Const(sys.boundary),
            Const(sys.force_units),
            Const(sys.energy_units),
            Duplicated(q, dq),
        )
        return @SMatrix [
            -dq[1]  -dq[4]  -dq[5];
            -dq[4]  -dq[2]  -dq[6];
            -dq[5]  -dq[6]  -dq[3]
        ]
    end

    function virial_pme(sys)
        W = zeros(FT, 3, 3)
        Molly.ewald_pe_forces!(
            zero(sys.coords),
            W,
            sys,
            only(sys.general_inters),
            Val(true);
            n_threads=1,
        )
        return W
    end

    function test_virial_match(W_reference, W_molly; relative_tol)
        @test maximum(abs, W_reference - W_molly) < 1e-6
        @test norm(W_reference - W_molly) / max(norm(W_molly), eps(FT)) < relative_tol
        @test abs(tr(W_reference) - tr(W_molly)) < 1e-6
    end

    function lj_dispersion_mechanical_adjustment(sys)
        V = volume(sys)
        correction = zero(eltype(eltype(sys.coords)))

        for inter in values(sys.general_inters)
            if inter isa LJDispersionCorrection
                U6  = inter.factor_6  / V
                U12 = inter.factor_12 / V

                # Enzyme differentiates U6 + U12; pressure uses 2U6 + 4U12.
                correction += (2 * U6 + 4 * U12) - (U6 + U12)
            end
        end

        return correction * I
    end

    ff = MolecularForceField(
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
        units=false,
        strictness=:nowarn,
    )

    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        units=false,
        array_type=AT,
        float_type=FT,
        nonbonded_method=SetupCoulombReactionField(),
    )

    sys_trc = System(sys; boundary=TriclinicBoundary(Molly.boxmatrix(sys.boundary)))
    neighbors_virial = Molly.find_neighbors(sys_trc; n_threads=1)

    W_enzyme, _, _ = virial_enzyme(sys_trc, neighbors_virial)
    W_enzyme_pressure = W_enzyme + lj_dispersion_mechanical_adjustment(sys_trc)
    W_molly = Molly.virial(sys_trc, neighbors_virial; n_threads=1)

    test_virial_match(W_enzyme_pressure, W_molly; relative_tol=1e-14)

    boundary_pme = TriclinicBoundary(@SMatrix [
        2.2  0.1  0.0;
        0.0  2.0  0.2;
        0.0  0.0  2.4
    ])
    atoms_pme = [
        Atom(mass=1.0, charge=1.0, σ=0.0, ϵ=0.0),
        Atom(mass=1.0, charge=-0.7, σ=0.0, ϵ=0.0),
        Atom(mass=1.0, charge=-0.3, σ=0.0, ϵ=0.0),
    ]
    coords_pme = [
        SVector(0.4, 0.6, 0.8),
        SVector(1.2, 0.7, 1.5),
        SVector(0.8, 1.4, 0.3),
    ]

    function pme_system(atoms, coords)
        pme = PME(
            0.9,
            atoms,
            boundary_pme;
            grad_safe=true,
            n_threads=1,
        )
        return System(
            atoms=atoms,
            coords=coords,
            boundary=boundary_pme,
            general_inters=(pme,),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    charged_atom = [Atom(mass=1.0, charge=1.0, σ=0.0, ϵ=0.0)]
    systems_pme = (
        ("reciprocal", pme_system(atoms_pme, coords_pme)),
        ("net charge", pme_system(charged_atom, [SVector(0.4, 0.6, 0.8)])),
    )

    for (name, sys_pme) in systems_pme
        test_virial_match(
            virial_enzyme_pme(sys_pme),
            virial_pme(sys_pme);
            relative_tol=1e-12,
        )
    end

    exclusion_list = InteractionList2Atoms(
        Int32[1],
        Int32[2],
        [EwaldExclusion()],
        [""],
        Molly.EwaldExclusionData(0.9),
    )
    sys_exclusion = System(
        atoms=atoms_pme[1:2],
        coords=coords_pme[1:2],
        boundary=boundary_pme,
        specific_inter_lists=(exclusion_list,),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    W_exclusion, _, _ = virial_enzyme(sys_exclusion, nothing)
    test_virial_match(W_exclusion, Molly.virial(sys_exclusion, nothing; n_threads=1);
                      relative_tol=1e-12)
end

@testset "CV gradients" begin
    function cv_gradient_enz(cv_type, coords, atoms=nothing, boundary=nothing, velocities=nothing)
        d_coords = zero(coords)
        unit_arr = Any[u"nm"]

        _, cv_val_ustrip = autodiff(
            set_runtime_activity(ReverseWithPrimal), # set_runtime_activity necessary for units
            Molly.calculate_cv_ustrip!,
            Active,
            Const(unit_arr),
            Const(cv_type),
            Duplicated(coords, d_coords),
            Const(atoms),
            Const(boundary),
            Const(velocities),
        )

        # Correct the units after the ustrip
        u = only(unit_arr)
        d_coords = d_coords .* u ./ unit(d_coords[1][1])^2

        return d_coords, cv_val_ustrip * u
    end

    function forces_test!(fs, sys, bias::BiasPotential; grad_cv=cv_gradient_enz, kwargs...)
        if bias.cv_type.correction == :pbc
            coords = Molly.unwrap_molecules(sys)
        else
            coords = sys.coords
        end

        # Gradient of CV with respect to coordinates
        d_coords, cv_sim = grad_cv(
            bias.cv_type,
            Molly.from_device(coords),
            Molly.from_device(sys.atoms),
            sys.boundary,
            Molly.from_device(sys.velocities),
        )

        # Gradient of bias function with respect to CV
        d_bias = bias_gradient(bias.bias_type, cv_sim)

        fs_svec = d_bias .* d_coords

        fs .-= Molly.to_device(fs_svec, typeof(fs))
        return fs
    end

    n_atoms = 100
    boundary = CubicBoundary(2.0u"nm")
    temp = 298.0u"K"
    atom_mass = 10.0u"g/mol"
    rng = Xoshiro(15)

    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    coords_ref = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng = rng)
    coords     = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng = rng)
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    cv_d_s   = CalcDist([1], [5], CalcSingleDist())
    cv_d_min = CalcDist([1, 2, 3, 4], [5, 6, 7, 8], CalcMinDist())
    cv_d_max = CalcDist([1, 2, 3, 4], [5, 6, 7, 8], CalcMaxDist())
    cv_d_cm  = CalcDist([1, 2, 3, 4], [5, 6, 7, 8], CalcCMDist())
    cv_rg    = CalcRg([1, 2, 3, 4])
    cv_rmsd  = CalcRMSD(coords_ref, [1,2,3,4],[1,2,3,4]) 
    cv_tor   = CalcTorsion([1,2,3,4])

    cvs = (cv_d_s, cv_d_min, cv_d_max, cv_d_cm, cv_rg, cv_rmsd, cv_tor)
    
    b1 = LinearBias(100.0u"kJ*mol^-1*nm^-1", 0.2u"nm")
    b2 = LinearBias(100.0u"kJ*mol^-1", 0.2)

    bias = (b1, b1, b1, b1, b1, b1, b2)

    for (c, b) in zip(cvs, bias)
        bias_pot = BiasPotential(c, b)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(),
            general_inters=(bias_pot,)
        )

        fs = forces(sys)
        fs_zero_enz = zero(fs)
        fs_zero_anl = zero(fs)

        forces_test!(fs_zero_enz, sys, bias_pot; grad_cv=cv_gradient_enz)
        forces_test!(fs_zero_anl, sys, bias_pot; grad_cv=cv_gradient)
        @test isapprox(ustrip_vec.(fs_zero_anl), ustrip_vec.(fs_zero_enz); atol=1e-6)
    end
end

@testset "Analysis gradients" begin
    T = Float64
    n_atoms = 12
    boundary = CubicBoundary(T(4.0))
    coords = place_atoms(n_atoms, boundary; min_dist=T(0.5), rng=Xoshiro(3))
    atoms = [Atom(index=i, mass=T(10.0) + i, charge=T(0.0), σ=T(0.3), ϵ=T(0.4))
             for i in 1:n_atoms]

    obs_rg(coords, atoms, boundary) = radius_gyration(coords, atoms)
    obs_hyd(coords, atoms, boundary) = hydrodynamic_radius(coords, boundary)
    obs_dist(coords, atoms, boundary) = sum(distances(coords, boundary))

    for (name, obs) in (("radius_gyration"    , obs_rg),
                        ("hydrodynamic_radius", obs_hyd),
                        ("distances"          , obs_dist))
        d_coords = zero(coords)
        autodiff(
            set_runtime_activity(Reverse),
            obs,
            Active,
            Duplicated(copy(coords), d_coords),
            Const(atoms),
            Const(boundary),
        )
        grad_fd = central_fdm(6, 1)(coords[1][1]) do x
            coords_mod = copy(coords)
            coords_mod[1] = SVector(x, coords[1][2], coords[1][3])
            return obs(coords_mod, atoms, boundary)
        end
        frac_diff = abs(d_coords[1][1] - grad_fd) / abs(grad_fd)
        @test frac_diff < 1e-8
    end

    # The RMSD is differentiable with respect to either set of coordinates, the rotation
    #   from the Kabsch algorithm is held constant which gives the exact gradient
    coords_ref = place_atoms(n_atoms, boundary; min_dist=T(0.5), rng=Xoshiro(4))
    rmsd_1(coords_1, coords_2) = rmsd(coords_1, coords_2)
    rmsd_2(coords_2, coords_1) = rmsd(coords_1, coords_2)

    for (name, obs, x0, other) in (("rmsd wrt coords_1", rmsd_1, coords, coords_ref),
                                   ("rmsd wrt coords_2", rmsd_2, coords_ref, coords))
        d_coords = zero(x0)
        autodiff(
            set_runtime_activity(Reverse),
            obs,
            Active,
            Duplicated(copy(x0), d_coords),
            Const(other),
        )
        grad_fd = central_fdm(6, 1)(x0[2][3]) do x
            coords_mod = copy(x0)
            coords_mod[2] = SVector(x0[2][1], x0[2][2], x)
            return obs(coords_mod, other)
        end
        frac_diff = abs(d_coords[2][3] - grad_fd) / abs(grad_fd)
        @test frac_diff < 1e-8
    end
end

@testset "Alchemical gradients" begin
    # dU/dλ is the quantity required for thermodynamic integration
    T = Float64
    cc = T(ustrip(Molly.coulomb_const))
    n_atoms = 20
    n_alchemical = 8
    boundary = CubicBoundary(T(3.0))
    coords = place_atoms(n_atoms, boundary; min_dist=T(0.5), rng=Xoshiro(1000))
    nb_cutoff = T(1.2)
    neighbor_finder = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=1,
                                             dist_cutoff=T(1.5))

    function pe_λ(λ, coords, boundary, pairwise_inters, neighbor_finder, neighbors, n_atoms,
                  n_alchemical, ::Val{T}) where T
        atoms = [Atom(i, 1, T(10.0), T(0.2) * (i % 2 == 0 ? -1 : 1), T(0.3), T(0.4),
                      (i <= n_alchemical ? λ : one(λ)),
                      (i <= n_alchemical ? Molly.InsertRole : Molly.CoreRole))
                 for i in 1:n_atoms]

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

    # With DefaultLambdaScheduler the steric term of an inserted atom is scaled over
    #   λ in [0, 0.5] and the electrostatic term over λ in [0.5, 1], so λ is chosen in the
    #   range where each interaction actually varies
    inters = [
        ("LennardJonesSCBeutler", LennardJonesSoftCoreBeutler(α=T(0.5),
                                        cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true),
                                  T(0.4)),
        ("LennardJonesSCGapsys" , LennardJonesSoftCoreGapsys(α=T(0.85),
                                        cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true),
                                  T(0.4)),
        ("CoulombSCBeutler"     , CoulombSoftCoreBeutler(cutoff=DistanceCutoff(nb_cutoff),
                                        use_neighbors=true, coulomb_const=cc),
                                  T(0.7)),
        ("CoulombSCGapsys"      , CoulombSoftCoreGapsys(cutoff=DistanceCutoff(nb_cutoff),
                                        σQ=T(1.0), use_neighbors=true, coulomb_const=cc),
                                  T(0.7)),
    ]
    for (name, inter, λ_start) in inters
        sys = System(
            atoms=[Atom(mass=T(10.0), charge=T(0.2), σ=T(0.3), ϵ=T(0.4)) for i in 1:n_atoms],
            coords=coords,
            boundary=boundary,
            pairwise_inters=(inter,),
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        neighbors = find_neighbors(sys; n_threads=1)
        grad_enzyme = autodiff(
            set_runtime_activity(Reverse),
            pe_λ,
            Active,
            Active(λ_start),
            Const(coords), Const(boundary), Const((inter,)), Const(neighbor_finder),
            Const(neighbors), Const(n_atoms), Const(n_alchemical), Const(Val(T)),
        )[1][1]
        grad_fd = central_fdm(6, 1)(λ_start) do λ
            pe_λ(λ, coords, boundary, (inter,), neighbor_finder, neighbors, n_atoms,
                 n_alchemical, Val(T))
        end
        frac_diff = abs(grad_enzyme - grad_fd) / abs(grad_fd)
        @test abs(grad_fd) > eps(T) # Guard against a trivially zero gradient
        @test frac_diff < 1e-8
    end
end

@testset "Second derivative gradients" begin
    # Forward over reverse gives Hessian information, e.g. for normal mode analysis
    T = Float64
    boundary = CubicBoundary(T(5.0))
    inter = LennardJones()
    atom = Atom(σ=T(0.3), ϵ=T(0.5))

    pe_dist(dist) = potential_energy(inter, SVector(dist, zero(T), zero(T)), atom, atom,
                                     NoUnits)
    dpe_dist(dist) = autodiff_deferred(Reverse, Const(pe_dist), Active, Active(dist))[1][1]
    d2pe_dist(dist) = autodiff(set_runtime_activity(Forward), dpe_dist, Duplicated,
                               Duplicated(dist, one(T)))[1]

    for dist in (T(0.35), T(0.5), T(0.8))
        grad_ad = d2pe_dist(dist)
        grad_fd = central_fdm(6, 2)(pe_dist, dist)
        frac_diff = abs(grad_ad - grad_fd) / abs(grad_fd)
        @test frac_diff < 1e-8
    end

    # A column of the Hessian with respect to the coordinates
    n_atoms = 4
    coords = [
        SVector(T(1.00), T(1.00), T(1.00)),
        SVector(T(1.35), T(1.05), T(1.00)),
        SVector(T(1.10), T(1.40), T(1.05)),
        SVector(T(1.40), T(1.40), T(1.35)),
    ]
    atoms = [Atom(mass=T(10.0), σ=T(0.3), ϵ=T(0.5)) for i in 1:n_atoms]

    function pe_pairs(coords, atoms, boundary)
        pe = zero(eltype(eltype(coords)))
        for i in eachindex(coords), j in (i + 1):length(coords)
            pe += potential_energy(LennardJones(), vector(coords[i], coords[j], boundary),
                                   atoms[i], atoms[j], NoUnits)
        end
        return pe
    end

    function grad_pe_pairs!(d_coords, coords, atoms, boundary)
        autodiff_deferred(Reverse, Const(pe_pairs), Active, Duplicated(coords, d_coords),
                          Const(atoms), Const(boundary))
        return nothing
    end

    # Seed a unit perturbation on the x coordinate of the first atom
    seed = [SVector(T(i == 1), zero(T), zero(T)) for i in 1:n_atoms]
    d_coords, dd_coords = zero(coords), zero(coords)
    autodiff(
        set_runtime_activity(Forward),
        grad_pe_pairs!,
        Const,
        Duplicated(d_coords, dd_coords),
        Duplicated(copy(coords), seed),
        Const(atoms),
        Const(boundary),
    )

    for atom_i in 1:n_atoms
        grad_fd = central_fdm(6, 1)(coords[1][1]) do x
            coords_mod = copy(coords)
            coords_mod[1] = SVector(x, coords[1][2], coords[1][3])
            d_coords_fd = zero(coords_mod)
            autodiff(Reverse, pe_pairs, Active, Duplicated(coords_mod, d_coords_fd),
                     Const(atoms), Const(boundary))
            return d_coords_fd[atom_i][1]
        end
        frac_diff = abs(dd_coords[atom_i][1] - grad_fd) / abs(grad_fd)
        @test frac_diff < 1e-8
    end
end

@testset "Differentiable simulation" begin
    runs = [ #               gpu    par    fwd    f32    obc2   gbn2   tol_σ tol_r0
        ("CPU"             , Array, false, false, false, false, false, 1e-4, 1e-4),
        ("CPU forward"     , Array, false, true , false, false, false, 0.5 , 0.1 ),
        ("CPU f32"         , Array, false, false, true , false, false, 0.01, 5e-4),
        ("CPU obc2"        , Array, false, false, false, true , false, 1e-4, 1e-4),
        ("CPU gbn2"        , Array, false, false, false, false, true , 1e-3, 1e-3),
        ("CPU gbn2 forward", Array, false, true , false, false, true , 0.5 , 0.1 ),
    ]
    if run_parallel_tests #                       gpu    par   fwd    f32    obc2   gbn2   tol_σ tol_r0
        push!(runs, ("CPU parallel"             , Array, true, false, false, false, false, 1e-4, 1e-4))
        push!(runs, ("CPU parallel forward"     , Array, true, true , false, false, false, 0.5 , 0.1 ))
        push!(runs, ("CPU parallel f32"         , Array, true, false, true , false, false, 0.01, 5e-4))
        push!(runs, ("CPU parallel obc2"        , Array, true, false, false, true , false, 1e-4, 1e-4))
        push!(runs, ("CPU parallel gbn2"        , Array, true, false, false, false, true , 1e-3, 1e-3))
        push!(runs, ("CPU parallel gbn2 forward", Array, true, true , false, false, true , 0.5 , 0.1 ))
    end
    for AT in array_list[2:end] # gpu  par    fwd    f32    obc2   gbn2   tol_σ tol_r0
        push!(runs, ("$AT"    ,   AT,  false, false, false, false, false, 0.25, 20.0))
        push!(runs, ("$AT f32",   AT,  false, false, true , false, false, 0.5 , 50.0))
    end

    function mean_min_separation(coords, boundary, ::Val{T}) where T
        min_seps = T[]
        for i in eachindex(coords)
            min_sq_sep = T(100.0)
            for j in eachindex(coords)
                if i != j
                    sq_dist = sum(abs2, vector(coords[i], coords[j], boundary))
                    min_sq_sep = min(sq_dist, min_sq_sep)
                end
            end
            push!(min_seps, sqrt(min_sq_sep))
        end
        return mean(min_seps)
    end

    function loss(σ, r0, coords, velocities, boundary, pairwise_inters, general_inters,
                  neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                  bond_is, bond_js, angles, torsions, rng, grad_safe, ::Val{T},
                  ::Val{AT}) where {T, AT}
        atoms = [Atom(i, 1, atom_mass, (i % 2 == 0 ? T(-0.02) : T(0.02)), σ, T(0.2), T(1.0), Molly.CoreRole)
                 for i in 1:n_atoms]
        bonds_inner = HarmonicBond{T, T}[]
        for i in 1:(n_atoms ÷ 2)
            push!(bonds_inner, HarmonicBond(T(100.0), bond_dists[i] * r0))
        end
        bonds = InteractionList2Atoms(
            bond_is,
            bond_js,
            to_device(bonds_inner, AT),
        )

        sys = System(
            atoms=to_device(atoms, AT),
            coords=to_device(coords, AT),
            boundary=boundary,
            velocities=to_device(velocities, AT),
            pairwise_inters=pairwise_inters,
            specific_inter_lists=(bonds, angles, torsions),
            general_inters=general_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
            grad_safe=grad_safe, # false to allow FD to test that the two paths are the same
        )

        simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=rng)

        return mean_min_separation(sys.coords, boundary, Val(T))
    end

    for (name, AT, parallel, forward, f32, obc2, gbn2, tol_σ, tol_r0) in runs
        T = (f32 ? Float32 : Float64)
        n_threads = (parallel ? Threads.nthreads() : 1)
        σ  = T(0.4)
        r0 = T(1.0)
        n_atoms = 50
        n_steps = 100
        atom_mass = T(10.0)
        boundary = CubicBoundary(T(3.0))
        temp = T(1.0)
        simulator = VelocityVerlet(
            dt=T(0.001),
            coupling=(ImmediateThermostat(temp),),
        )
        rng = Xoshiro(1000) # Same system every time, not required but increases stability
        coords = place_atoms(n_atoms, boundary; min_dist=T(0.6), max_attempts=500, rng=rng)
        velocities = [random_velocity(atom_mass, temp; rng=rng) for i in 1:n_atoms]
        nb_cutoff = T(1.2)
        lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true)
        crf = CoulombReactionField(
            dist_cutoff=nb_cutoff,
            solvent_dielectric=T(Molly.crf_solvent_dielectric),
            use_neighbors=true,
            coulomb_const=T(ustrip(Molly.coulomb_const)),
        )
        pairwise_inters = (lj, crf)
        bond_is = to_device(Int32.(collect(1:(n_atoms ÷ 2))), AT)
        bond_js = to_device(Int32.(collect((1 + n_atoms ÷ 2):n_atoms)), AT)
        bond_dists = [norm(vector(coords[i], coords[i + n_atoms ÷ 2], boundary))
                      for i in 1:(n_atoms ÷ 2)]
        angles_inner = [HarmonicAngle(k=T(10.0), θ0=T(2.0)) for i in 1:15]
        angles = InteractionList3Atoms(
            to_device(Int32.(collect( 1:15)), AT),
            to_device(Int32.(collect(16:30)), AT),
            to_device(Int32.(collect(31:45)), AT),
            to_device(angles_inner, AT),
        )
        torsions_inner = [PeriodicTorsion(
                periodicities=[1, 2, 3],
                phases=T[1.0, 0.0, -1.0],
                ks=T[10.0, 5.0, 8.0],
                n_terms=6,
            ) for i in 1:10]
        torsions = InteractionList4Atoms(
            to_device(Int32.(collect( 1:10)), AT),
            to_device(Int32.(collect(11:20)), AT),
            to_device(Int32.(collect(21:30)), AT),
            to_device(Int32.(collect(31:40)), AT),
            to_device(torsions_inner, AT),
        )
        atoms_setup = [Atom(charge=zero(T), σ=zero(T)) for i in 1:n_atoms]
        if obc2
            imp_obc2 = ImplicitSolventOBC(
                to_device(atoms_setup, AT),
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, fill(0, length(bond_is)));
                kappa=T(0.7),
                use_OBC2=true,
                n_threads=n_threads,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                to_device(atoms_setup, AT),
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, fill(0, length(bond_is)));
                kappa=T(0.7),
                n_threads=n_threads,
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceNeighborFinder(
            eligible=to_device(trues(n_atoms, n_atoms), AT),
            n_steps=10,
            dist_cutoff=T(1.5),
        )

        const_args = [
            Const(boundary), Const(pairwise_inters),
            Const(general_inters), Const(neighbor_finder), Const(simulator),
            Const(n_steps), Const(n_threads), Const(n_atoms), Const(atom_mass),
            Const(bond_dists), Const(bond_is), Const(bond_js), Const(angles),
            Const(torsions), Const(rng), Const(true), Const(Val(T)), Const(Val(AT)),
        ]
        if forward
            grad_enzyme = (
                autodiff(
                    set_runtime_activity(Forward),
                    loss,
                    Duplicated,
                    Duplicated(σ, one(T)),
                    Const(r0),
                    Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)),
                    const_args...,
                )[1],
                autodiff(
                    set_runtime_activity(Forward),
                    loss,
                    Duplicated,
                    Const(σ),
                    Duplicated(r0, one(T)),
                    Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)),
                    const_args...,
                )[1],
            )
        else
            grad_enzyme = autodiff(
                set_runtime_activity(Reverse),
                loss,
                Active,
                Active(σ),
                Active(r0),
                Duplicated(copy(coords), zero(coords)),
                Duplicated(copy(velocities), zero(velocities)),
                const_args...,
            )[1][1:2]
        end

        grad_fd = (
            central_fdm(6, 1)(
                σ -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, false, Val(T), Val(AT),
                ),
                σ,
            ),
            central_fdm(6, 1)(
                r0 -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, false, Val(T), Val(AT),
                ),
                r0,
            ),
        )
        for (prefix, genz, gfd, tol) in zip(("σ", "r0"), grad_enzyme, grad_fd, (tol_σ, tol_r0))
            if abs(gfd) < 1e-13
                ztol = (contains(name, "f32") ? 1e-8 : 1e-10)
                @test isnothing(genz) || abs(genz) < ztol
            elseif isnothing(genz)
                @test !isnothing(genz)
            else
                frac_diff = abs(genz - gfd) / abs(gfd)
                @test frac_diff < tol
            end
        end
    end
end

@testset "Simulator gradients" begin
    T = Float64
    cc = T(ustrip(Molly.coulomb_const))
    n_atoms = 50
    n_steps = 100
    atom_mass = T(10.0)
    boundary = CubicBoundary(T(3.0))
    temp = T(1.0)
    rng_setup = Xoshiro(1000)
    coords = place_atoms(n_atoms, boundary; min_dist=T(0.6), max_attempts=500, rng=rng_setup)
    velocities = [random_velocity(atom_mass, temp; rng=rng_setup) for i in 1:n_atoms]
    nb_cutoff = T(1.2)
    lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true)
    crf = CoulombReactionField(
        dist_cutoff=nb_cutoff,
        solvent_dielectric=T(Molly.crf_solvent_dielectric),
        use_neighbors=true,
        coulomb_const=cc,
    )
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=T(1.5),
    )
    σ_start = T(0.4)

    function mean_min_separation(coords, boundary)
        L = eltype(eltype(coords))
        min_seps = L[]
        for i in eachindex(coords)
            min_sq_sep = 100 * oneunit(L)^2
            for j in eachindex(coords)
                if i != j
                    sq_dist = sum(abs2, vector(coords[i], coords[j], boundary))
                    min_sq_sep = min(sq_dist, min_sq_sep)
                end
            end
            push!(min_seps, sqrt(min_sq_sep))
        end
        return mean(min_seps)
    end

    charge, ϵ, λ = T(0.02), T(0.2), T(1.0)

    function loss(σ, coords, velocities, boundary, pairwise_inters, neighbor_finder,
                  constraints, simulator, n_steps, n_atoms, atom_mass, rng)
        atoms = [Atom(i, 1, atom_mass, (i % 2 == 0 ? -charge : charge), σ, ϵ, λ,
                      Molly.CoreRole) for i in 1:n_atoms]

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            constraints=constraints,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulate!(sys, simulator, n_steps; n_threads=1, rng=rng)

        return mean_min_separation(sys.coords, boundary)
    end

    function test_sim_grad(name, simulator; nf=neighbor_finder, constraints=(), tol=1e-5)
        grad_enzyme = autodiff(
            set_runtime_activity(Reverse),
            loss,
            Active,
            Active(σ_start),
            Duplicated(copy(coords), zero(coords)),
            Duplicated(copy(velocities), zero(velocities)),
            Const(boundary), Const((lj, crf)), Const(nf), Const(constraints),
            Const(simulator), Const(n_steps), Const(n_atoms), Const(atom_mass),
            Const(Xoshiro(2000)),
        )[1][1]
        grad_fd = central_fdm(6, 1)(σ_start) do σ
            loss(σ, copy(coords), copy(velocities), boundary, (lj, crf), nf, constraints,
                 simulator, n_steps, n_atoms, atom_mass, Xoshiro(2000))
        end
        frac_diff = abs(grad_enzyme - grad_fd) / abs(grad_fd)
        @test frac_diff < tol
    end

    integrators = [
        ("VelocityVerlet"   , VelocityVerlet(dt=T(0.001),
                                    coupling=(ImmediateThermostat(temp),))),
        ("Verlet"           , Verlet(dt=T(0.001))),
        ("Langevin"         , Langevin(dt=T(0.001), temperature=temp, friction=T(1.0))),
        ("LangevinSplitting", LangevinSplitting(dt=T(0.001), temperature=temp,
                                    friction=T(1.0), splitting="BAOAB")),
        ("NoseHoover"       , NoseHoover(dt=T(0.001), temperature=temp)),
    ]
    for (name, simulator) in integrators
        test_sim_grad(name, simulator)
    end

    couplings = [
        ("BerendsenThermostat"     , BerendsenThermostat(temp, T(0.1))),
        ("AndersenThermostat"      , AndersenThermostat(temp, T(0.01))),
        ("VelocityRescaleThermostat", VelocityRescaleThermostat(temp, T(0.1))),
        ("CRescaleBarostat"        , CRescaleBarostat(T(1.0), temp;
                                                        compressibility=T(4.6e-5))),
    ]
    for (name, coupling) in couplings
        test_sim_grad(name, VelocityVerlet(dt=T(0.001), coupling=(coupling,)))
    end
    test_sim_grad("BerendsenBarostat", VelocityVerlet(dt=T(0.001), coupling=(
            BerendsenThermostat(temp, T(0.1)),
            BerendsenBarostat(T(1.0), T(1.0); compressibility=T(4.6e-5)),
        )))

    finders = [
        ("TreeNeighborFinder", TreeNeighborFinder(eligible=trues(n_atoms, n_atoms),
                                        n_steps=10, dist_cutoff=T(1.5))),
        ("CellListMapNeighborFinder", CellListMapNeighborFinder(
                                        eligible=trues(n_atoms, n_atoms), n_steps=10,
                                        dist_cutoff=T(1.4), boundary=boundary, x0=coords)),
    ]
    for (name, nf) in finders
        test_sim_grad(name, VelocityVerlet(dt=T(0.001),
                            coupling=(ImmediateThermostat(temp),)); nf=nf)
    end

    dist_constraints = [DistanceConstraint(Int32(2i - 1), Int32(2i), T(0.6)) for i in 1:10]
    shake = SHAKE_RATTLE(
        n_atoms=n_atoms,
        dist_tolerance=T(1e-10),
        vel_tolerance=T(1e-10),
        dist_constraints=dist_constraints,
    )
    # The gradient is only as accurate as the constraint solver tolerance
    test_sim_grad("SHAKE_RATTLE", VelocityVerlet(dt=T(0.001),
                    coupling=(ImmediateThermostat(temp),)); constraints=(shake,), tol=1e-4)

    # Loss functions can use quantities recorded during the simulation
    function loss_logged(σ, coords, velocities, boundary, pairwise_inters, neighbor_finder,
                            simulator, n_steps, n_atoms, atom_mass, rng)
        atoms = [Atom(i, 1, atom_mass, (i % 2 == 0 ? -charge : charge), σ, ϵ, λ,
                        Molly.CoreRole) for i in 1:n_atoms]

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            neighbor_finder=neighbor_finder,
            loggers=(coords=CoordinatesLogger(T, 25),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulate!(sys, simulator, n_steps; n_threads=1, rng=rng)

        logged_coords = values(sys.loggers.coords)
        sep_sum = zero(T)
        for cs in logged_coords
            sep_sum += mean_min_separation(cs, boundary)
        end
        return sep_sum / length(logged_coords)
    end

    simulator = VelocityVerlet(dt=T(0.001), coupling=(ImmediateThermostat(temp),))
    grad_enzyme = autodiff(
        set_runtime_activity(Reverse),
        loss_logged,
        Active,
        Active(σ_start),
        Duplicated(copy(coords), zero(coords)),
        Duplicated(copy(velocities), zero(velocities)),
        Const(boundary), Const((lj, crf)), Const(neighbor_finder), Const(simulator),
        Const(n_steps), Const(n_atoms), Const(atom_mass), Const(Xoshiro(2000)),
    )[1][1]
    grad_fd = central_fdm(6, 1)(σ_start) do σ
        loss_logged(σ, copy(coords), copy(velocities), boundary, (lj, crf),
                    neighbor_finder, simulator, n_steps, n_atoms, atom_mass, Xoshiro(2000))
    end
    frac_diff = abs(grad_enzyme - grad_fd) / abs(grad_fd)
    @test frac_diff < 1e-5

    boundary_units = CubicBoundary(3.0u"nm")
    lj_units = LennardJones(cutoff=DistanceCutoff(1.2u"nm"), use_neighbors=true)
    neighbor_finder_units = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    simulator_units = VelocityVerlet(dt=0.001u"ps",
                                        coupling=(ImmediateThermostat(1.0u"K"),))

    function loss_units(σ, coords, velocities, boundary, pairwise_inters, neighbor_finder,
                        simulator, n_steps, n_atoms, rng)
        atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=σ, ϵ=0.2u"kJ * mol^-1")
                    for i in 1:n_atoms]

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            neighbor_finder=neighbor_finder,
        )

        simulate!(sys, simulator, n_steps; n_threads=1, rng=rng)

        # Enzyme requires a unitless value to be returned
        return ustrip(u"nm", mean_min_separation(sys.coords, boundary))
    end

    coords_units = coords * u"nm"
    velocities_units = velocities * u"nm * ps^-1"
    grad_enzyme = autodiff(
        set_runtime_activity(Reverse),
        loss_units,
        Active,
        Active(σ_start * u"nm"),
        Duplicated(copy(coords_units), zero(coords_units)),
        Duplicated(copy(velocities_units), zero(velocities_units)),
        Const(boundary_units), Const((lj_units,)), Const(neighbor_finder_units),
        Const(simulator_units), Const(n_steps), Const(n_atoms), Const(Xoshiro(2000)),
    )[1][1]
    grad_fd = central_fdm(6, 1)(σ_start) do σ
        loss_units(σ * u"nm", copy(coords_units), copy(velocities_units), boundary_units,
                    (lj_units,), neighbor_finder_units, simulator_units, n_steps, n_atoms,
                    Xoshiro(2000))
    end
    frac_diff = abs(ustrip(grad_enzyme) - grad_fd) / abs(grad_fd)
    @test frac_diff < 1e-5
end

# A general interaction holding the parameters of a one hidden layer neural network,
#   standing in for the Flux models described in the differentiable simulation docs
struct NNBondsTest{W}
    weight_1::W
    bias_1::W
    weight_2::W
end

Base.zero(m::NNBondsTest) = NNBondsTest(zero(m.weight_1), zero(m.bias_1), zero(m.weight_2))

function Base.:+(m1::NNBondsTest, m2::NNBondsTest)
    return NNBondsTest(m1.weight_1 + m2.weight_1, m1.bias_1 + m2.bias_1,
                       m1.weight_2 + m2.weight_2)
end

function AtomsCalculators.forces!(fs, sys, inter::NNBondsTest; kwargs...)
    vec_ij = vector(sys.coords[1], sys.coords[3], sys.boundary)
    hidden = max.(inter.weight_1 .* norm(vec_ij) .+ inter.bias_1, zero(eltype(inter.bias_1)))
    f = sum(inter.weight_2 .* hidden) * normalize(vec_ij)
    fs[1] += f
    fs[3] -= f
    return fs
end

@testset "Neural network potential gradients" begin
    T = Float64
    boundary = CubicBoundary(T(5.0))
    coords = [
        SVector(T(2.3), T(2.07), T(0.0)),
        SVector(T(2.5), T(2.93), T(0.0)),
        SVector(T(2.7), T(2.07), T(0.0)),
    ]
    velocities = zero(coords)
    atoms = [Atom(index=i, mass=T(10.0), charge=T(0.0), σ=T(0.0), ϵ=T(0.0))
             for i in eachindex(coords)]
    simulator = VelocityVerlet(dt=T(0.02), coupling=(BerendsenThermostat(T(0.01), T(0.5)),))
    n_steps = 100
    dist_true = T(1.0)

    function loss_nn(model, coords, velocities, atoms, boundary, simulator, n_steps,
                     dist_true, rng)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            general_inters=(model,),
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulate!(sys, simulator, n_steps; n_threads=1, rng=rng)

        dist_end = (norm(vector(sys.coords[1], sys.coords[2], boundary)) +
                    norm(vector(sys.coords[2], sys.coords[3], boundary)) +
                    norm(vector(sys.coords[3], sys.coords[1], boundary))) / 3
        return abs(dist_end - dist_true)
    end

    model = NNBondsTest(
        T[0.5, -0.3, 0.8, 0.1, -0.6],
        T[0.1, 0.2, -0.1, 0.05, 0.3],
        T[0.4, -0.2, 0.7, -0.5, 0.3],
    )
    d_model = zero(model)
    autodiff(
        set_runtime_activity(Reverse),
        loss_nn,
        Active,
        Duplicated(model, d_model),
        Duplicated(copy(coords), zero(coords)),
        Duplicated(copy(velocities), zero(velocities)),
        Const(atoms), Const(boundary), Const(simulator), Const(n_steps), Const(dist_true),
        Const(Xoshiro(1)),
    )

    for param_i in (1, 3)
        grad_fd = central_fdm(6, 1)(model.weight_1[param_i]) do val
            weight_1 = copy(model.weight_1)
            weight_1[param_i] = val
            loss_nn(NNBondsTest(weight_1, model.bias_1, model.weight_2), copy(coords),
                    copy(velocities), atoms, boundary, simulator, n_steps, dist_true,
                    Xoshiro(1))
        end
        grad_enzyme = d_model.weight_1[param_i]
        frac_diff = abs(grad_enzyme - grad_fd) / abs(grad_fd)
        @test frac_diff < 1e-8
    end
end

@testset "Parameter injection gradients" begin
    T = Float64
    cc = T(ustrip(Molly.coulomb_const))
    n_atoms = 12
    boundary = CubicBoundary(T(4.0))
    coords = place_atoms(n_atoms, boundary; min_dist=T(0.4), rng=Xoshiro(2024))
    atoms = [Atom(index=i, mass=T(10.0), charge=T(i % 2 == 0 ? 0.2 : -0.2), σ=T(0.3),
                  ϵ=T(0.4), λ=T(0.6)) for i in 1:n_atoms]
    # All pairs are marked special so that the 1-4 weights affect the energy
    nf = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), special=trues(n_atoms, n_atoms),
                                n_steps=1, dist_cutoff=T(1.5))
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
    # inject_gradients uses atoms_data to look up the parameters of each atom type
    atoms_data      = [AtomData(atom_type="XX", element="C") for i in 1:n_atoms]
    atoms_data_spec = [AtomData(atom_type="XX", element="C") for i in eachindex(coords_spec)]

    function pe_injected(params_dic, sys_ref, coords, neighbors)
        atoms, pis, sis, gis = Molly.inject_gradients(sys_ref, params_dic)

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=sys_ref.boundary,
            pairwise_inters=pis,
            specific_inter_lists=sis,
            general_inters=gis,
            neighbor_finder=sys_ref.neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        return potential_energy(sys, neighbors; n_threads=1)
    end

    # Extract the parameters of an interaction, inject them back and check that the
    #   gradient with respect to each one matches finite differencing
    # Enzyme currently drops the gradient of the trailing fields of some interaction
    #   types when they are read back out of a System, those keys are listed in broken
    function test_params(name, sys_ref, params_dic, broken=())
        @test length(params_dic) > 0
        neighbors = find_neighbors(sys_ref; n_threads=1)
        grads_enzyme = Dict(k => zero(v) for (k, v) in params_dic)
        autodiff(
            set_runtime_activity(Reverse),
            Const(pe_injected),
            Active,
            Duplicated(params_dic, grads_enzyme),
            Const(sys_ref),
            Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
            Const(neighbors),
        )
        n_nonzero = 0
        for param in sort(collect(keys(params_dic)))
            grad_fd = central_fdm(6, 1)(params_dic[param]) do val
                dic = copy(params_dic)
                dic[param] = val
                pe_injected(dic, sys_ref, copy(sys_ref.coords), neighbors)
            end
            grad_enzyme = grads_enzyme[param]
            if abs(grad_fd) < 1e-10
                # The parameter does not affect the energy of this test system
                @test abs(grad_enzyme) < 1e-8
            else
                n_nonzero += 1
                frac_diff = abs(grad_enzyme - grad_fd) / abs(grad_fd)
                if any(endswith(param, b) for b in broken)
                    @test_broken frac_diff < 1e-6
                else
                    @test frac_diff < 1e-6
                end
            end
        end
        @test n_nonzero > 0 # Guard against a test where nothing affects the energy
    end

    function pairwise_sys(inter)
        return System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            atoms_data=atoms_data,
            pairwise_inters=(inter,),
            neighbor_finder=nf,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    function specific_sys(inter_list)
        return System(
            atoms=atoms_spec,
            coords=coords_spec,
            boundary=boundary,
            atoms_data=atoms_data_spec,
            specific_inter_lists=(inter_list,),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    w = T(0.5) # The 1-4 weight has to be a float to be injected back
    # See the note on test_params for the broken keys
    lost_w  = ("weight_14",)
    lost_wc = ("weight_14", "coulomb_const")
    pairwise_inters = [
        ("LennardJones"     , LennardJones(cutoff=DistanceCutoff(nb_cutoff),
                                    use_neighbors=true, weight_special=w), lost_w),
        ("Mie"              , Mie(m=T(6), n=T(12), cutoff=DistanceCutoff(nb_cutoff),
                                    use_neighbors=true, weight_special=w), lost_w),
        ("AshbaughHatch"    , AshbaughHatch(cutoff=DistanceCutoff(nb_cutoff),
                                    use_neighbors=true, weight_special=w), lost_w),
        ("DoubleExponential", DoubleExponential(α=T(16.766), β=T(4.427), weight_special=w,
                                    cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true), ()),
        ("DoubleExponentialSC", DoubleExponentialSoftCore(α=T(16.766), β=T(4.427),
                                    weight_special=w, cutoff=DistanceCutoff(nb_cutoff),
                                    use_neighbors=true), ()),
        ("Gravity"          , Gravity(G=T(1.0), use_neighbors=true), ()),
        ("LJSCBeutler"      , LennardJonesSoftCoreBeutler(α=T(0.5), weight_special=w,
                                    cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true), lost_w),
        ("LJSCGapsys"       , LennardJonesSoftCoreGapsys(α=T(0.85), weight_special=w,
                                    cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true), lost_w),
        ("Coulomb"          , Coulomb(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true,
                                    weight_special=w, coulomb_const=cc), ()),
        ("CoulombScaled"    , CoulombScaled(cutoff=DistanceCutoff(nb_cutoff),
                                    use_neighbors=true, weight_special=w, coulomb_const=cc), lost_wc),
        ("CoulombSCBeutler" , CoulombSoftCoreBeutler(cutoff=DistanceCutoff(nb_cutoff),
                                    use_neighbors=true, weight_special=w, coulomb_const=cc), lost_wc),
        ("CoulombSCGapsys"  , CoulombSoftCoreGapsys(cutoff=DistanceCutoff(nb_cutoff),
                                    σQ=T(1.0), use_neighbors=true, weight_special=w,
                                    coulomb_const=cc), lost_wc),
        ("CoulombRF"        , CoulombReactionField(dist_cutoff=nb_cutoff,
                                    use_neighbors=true, weight_special=w, coulomb_const=cc), ()),
        ("CRFScaled"        , CoulombReactionFieldScaled(dist_cutoff=nb_cutoff,
                                    use_neighbors=true, weight_special=w, coulomb_const=cc), lost_wc),
        ("CRFSCBeutler"     , CoulombSoftCoreBeutlerReactionField(dist_cutoff=nb_cutoff,
                                    use_neighbors=true, weight_special=w, coulomb_const=cc), lost_wc),
        ("CRFSCGapsys"      , CoulombSoftCoreGapsysReactionField(dist_cutoff=nb_cutoff,
                                    σQ=T(1.0), use_neighbors=true, weight_special=w,
                                    coulomb_const=cc), lost_wc),
        ("CoulombEwald"     , CoulombEwald(dist_cutoff=nb_cutoff, use_neighbors=true,
                                    weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), ()),
        ("CoulombEwaldScaled", CoulombEwaldScaled(dist_cutoff=nb_cutoff,
                                    use_neighbors=true, weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), lost_wc),
        ("CEwaldSCBeutler"  , CoulombSoftCoreBeutlerEwald(dist_cutoff=nb_cutoff,
                                    use_neighbors=true, weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), lost_wc),
        ("CEwaldSCGapsys"   , CoulombSoftCoreGapsysEwald(dist_cutoff=nb_cutoff, σQ=T(1.0),
                                    use_neighbors=true, weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), lost_wc),
        ("Yukawa"           , Yukawa(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true,
                                    weight_special=w, coulomb_const=cc, kappa=T(1.0)), ()),
    ]
    for (name, inter, broken) in pairwise_inters
        sys_ref = pairwise_sys(inter)
        params_dic = Dict{String, Float64}()
        Molly.extract_parameters!(params_dic, inter, nothing)
        test_params(name, sys_ref, params_dic, broken)
    end

    specific_inters = [
        ("HarmonicBond"  , InteractionList2Atoms(is, js,
                                [HarmonicBond(k=T(100.0), r0=T(0.5))], [""])),
        ("MorseBond"     , InteractionList2Atoms(is, js,
                                [MorseBond(D=T(100.0), a=T(2.0), r0=T(0.5))], [""])),
        ("FENEBond"      , InteractionList2Atoms(is, js,
                                [FENEBond(k=T(100.0), r0=T(1.4), σ=T(0.3), ϵ=T(0.4))],
                                [""])),
        ("LennardJones14", InteractionList2Atoms(is, ls,
                                [Molly.LennardJones14(T(0.3), T(0.4), T(0.5))], [""])),
        ("HarmonicAngle" , InteractionList3Atoms(is, js, ks,
                                [HarmonicAngle(k=T(10.0), θ0=T(2.0))], [""])),
        ("CosineAngle"   , InteractionList3Atoms(is, js, ks,
                                [CosineAngle(k=T(10.0), θ0=T(2.0))], [""])),
        ("UreyBradley"   , InteractionList3Atoms(is, js, ks,
                                [UreyBradley(kangle=T(10.0), θ0=T(2.0), kbond=T(10.0),
                                                r0=T(1.0))], [""])),
        ("PeriodicTorsion", InteractionList4Atoms(is, js, ks, ls,
                                [PeriodicTorsion(periodicities=[1, 2], phases=T[1.0, 0.0],
                                                    ks=T[10.0, 5.0], n_terms=2)], [""])),
        ("RBTorsion"     , InteractionList4Atoms(is, js, ks, ls,
                                [RBTorsion(c0=T(1.0), c1=T(2.0), c2=T(3.0), c3=T(4.0),
                                            c4=T(0.5), c5=T(0.25))], [""])),
        ("HarmonicTorsion", InteractionList4Atoms(is, js, ks, ls,
                                [HarmonicTorsion(k=T(10.0), θ0=T(1.0))], [""])),
        ("HarmonicPositionRestraint", InteractionList1Atoms(is,
                                [HarmonicPositionRestraint(k=T(100.0),
                                    x0=coords_spec[1] .+ T(0.1))], [""])),
    ]
    for (name, inter_list) in specific_inters
        sys_ref = specific_sys(inter_list)
        params_dic = Dict{String, Float64}()
        Molly.extract_parameters!(params_dic, inter_list, nothing)
        test_params(name, sys_ref, params_dic)
    end
end

@testset "Differentiable protein" begin
    function create_sys(AT, n_threads)
        ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml"])...; units=false)
        return System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff;
            units=false,
            array_type=AT,
            float_type=Float64,
            nonbonded_method=DistanceCutoff(1.0),
            dispersion_correction=false,
            grad_safe=true,
            strictness=:nowarn,
            n_threads=n_threads,
        )
    end

    function test_energy_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = Molly.inject_gradients(sys_ref, params_dic)

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=sys_ref.boundary,
            pairwise_inters=pis,
            specific_inter_lists=sis,
            general_inters=gis,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        return potential_energy(sys; n_threads=n_threads)
    end

    function test_forces_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = Molly.inject_gradients(sys_ref, params_dic)

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=sys_ref.boundary,
            pairwise_inters=pis,
            specific_inter_lists=sis,
            general_inters=gis,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        fs = forces(sys; n_threads=n_threads)
        return sum(sum.(abs2, fs))
    end

    function test_sim_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = Molly.inject_gradients(sys_ref, params_dic)

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=sys_ref.boundary,
            pairwise_inters=pis,
            specific_inter_lists=sis,
            general_inters=gis,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulator = Langevin(dt=0.001, temperature=300.0, friction=1.0)
        n_steps = 5
        rng = Xoshiro(1000)
        simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=rng)
        return sum(sum.(abs, sys.coords))
    end

    params_dic = Dict(
        "atom_C8_σ"                => 0.33996695084235345,
        "atom_C8_ϵ"                => 0.4577296,
        "atom_C9_σ"                => 0.33996695084235345,
        "atom_C9_ϵ"                => 0.4577296,
        "atom_CA_σ"                => 0.33996695084235345,
        "atom_CA_ϵ"                => 0.359824,
        "atom_CT_σ"                => 0.33996695084235345,
        "atom_CT_ϵ"                => 0.4577296,
        "atom_C_σ"                 => 0.33996695084235345,
        "atom_C_ϵ"                 => 0.359824,
        "atom_N3_σ"                => 0.32499985237759577,
        "atom_N3_ϵ"                => 0.71128,
        "atom_N_σ"                 => 0.32499985237759577,
        "atom_N_ϵ"                 => 0.71128,
        "atom_O2_σ"                => 0.2959921901149463,
        "atom_O2_ϵ"                => 0.87864,
        "atom_OH_σ"                => 0.30664733878390477,
        "atom_OH_ϵ"                => 0.8803136,
        "atom_O_σ"                 => 0.2959921901149463,
        "atom_O_ϵ"                 => 0.87864,
        "inter_CO_weight_14"       => 0.8333,
        "inter_LJ_weight_14"       => 0.5,
        "inter_PT_-/C/CT/-_k_1"    => 0.0,
        "inter_PT_-/C/N/-_k_1"     => -10.46,
        "inter_PT_-/CA/CA/-_k_1"   => -15.167,
        "inter_PT_-/CA/CT/-_k_1"   => 0.0,
        "inter_PT_-/CT/C8/-_k_1"   => 0.64852,
        "inter_PT_-/CT/C9/-_k_1"   => 0.64852,
        "inter_PT_-/CT/CT/-_k_1"   => 0.6508444444444447,
        "inter_PT_-/CT/N/-_k_1"    => 0.0,
        "inter_PT_-/CT/N3/-_k_1"   => 0.6508444444444447,
        "inter_PT_C/N/CT/C_k_1"    => -0.142256,
        "inter_PT_C/N/CT/C_k_2"    => 1.40164,
        "inter_PT_C/N/CT/C_k_3"    => 2.276096,
        "inter_PT_C/N/CT/C_k_4"    => 0.33472,
        "inter_PT_C/N/CT/C_k_5"    => 1.6736,
        "inter_PT_CT/CT/C/N_k_1"   => 0.8368,
        "inter_PT_CT/CT/C/N_k_2"   => 0.8368,
        "inter_PT_CT/CT/C/N_k_3"   => 1.6736,
        "inter_PT_CT/CT/N/C_k_1"   => 8.368,
        "inter_PT_CT/CT/N/C_k_2"   => 8.368,
        "inter_PT_CT/CT/N/C_k_3"   => 1.6736,
        "inter_PT_H/N/C/O_k_1"     => 8.368,
        "inter_PT_H/N/C/O_k_2"     => -10.46,
        "inter_PT_H1/CT/C/O_k_1"   => 3.3472,
        "inter_PT_H1/CT/C/O_k_2"   => -0.33472,
        "inter_PT_HC/CT/C4/CT_k_1" => 0.66944,
        "inter_PT_N/CT/C/N_k_1"    => 2.7196,
        "inter_PT_N/CT/C/N_k_10"   => 0.1046,
        "inter_PT_N/CT/C/N_k_11"   => -0.046024,
        "inter_PT_N/CT/C/N_k_2"    => -0.824248,
        "inter_PT_N/CT/C/N_k_3"    => 6.04588,
        "inter_PT_N/CT/C/N_k_4"    => 2.004136,
        "inter_PT_N/CT/C/N_k_5"    => -0.0799144,
        "inter_PT_N/CT/C/N_k_6"    => -0.016736,
        "inter_PT_N/CT/C/N_k_7"    => -1.06692,
        "inter_PT_N/CT/C/N_k_8"    => 0.3138,
        "inter_PT_N/CT/C/N_k_9"    => 0.238488,
    )

    platform_runs = [("CPU", Array, false)]
    if run_parallel_tests
        push!(platform_runs, ("CPU parallel", Array, true))
    end
    for AT in array_list[2:end]
        push!(platform_runs, ("$AT", AT, false))
    end
    test_runs = Any[
        ("Energy", test_energy_grad, 1e-8, 1e-10),
        ("Force" , test_forces_grad, 1e-8, 1e-10),
    ]
    if !running_CI
        push!(test_runs, ("Sim", test_sim_grad, 1e-2, nothing))
    end
    params_to_test = (
        "atom_N_σ",
        "atom_N_ϵ",
        "inter_PT_C/N/CT/C_k_1",
    )

    for (test_name, test_fn, tol_fd, tol_cross) in test_runs
        grads_ref = nothing # Single-threaded CPU gradients for every parameter
        for (platform, AT, parallel) in platform_runs
            if test_name == "Sim" && !startswith(platform, "CPU")
                continue
            end
            n_threads = (parallel ? Threads.nthreads() : 1)
            sys_ref = create_sys(AT, n_threads)
            grads_enzyme = Dict(k => 0.0 for k in keys(params_dic))
            autodiff(
                set_runtime_activity(Reverse),
                test_fn,
                Active,
                Duplicated(params_dic, grads_enzyme),
                Const(sys_ref),
                Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
                Duplicated(sys_ref.neighbor_finder, sys_ref.neighbor_finder),
                Const(n_threads),
            )
            for param in params_to_test
                genz = grads_enzyme[param]
                gfd = central_fdm(6, 1)(params_dic[param]) do val
                    dic = copy(params_dic)
                    dic[param] = val
                    test_fn(dic, sys_ref, copy(sys_ref.coords), sys_ref.neighbor_finder, n_threads)
                end
                frac_diff = abs(genz - gfd) / abs(gfd)
                @test frac_diff < tol_fd
            end
            if isnothing(tol_cross)
                continue # Random numbers on different backends may be different
            elseif isnothing(grads_ref)
                grads_ref = grads_enzyme
            else
                # Every force field parameter should give the same gradient on every
                # platform, measured relative to the largest gradient so that parameters
                # with a near-zero gradient do not dominate
                scale = maximum(abs, values(grads_ref))
                max_diff = maximum(abs(grads_enzyme[k] - grads_ref[k]) for k in keys(params_dic))
                @test max_diff / scale < tol_cross
            end
        end
    end
end
