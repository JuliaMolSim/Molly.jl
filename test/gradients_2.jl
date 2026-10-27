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
            boundary=TriclinicBoundary(F * Molly.box_matrix(sys.boundary)),
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
        boundary_new = TriclinicBoundary(F * Molly.box_matrix(boundary))
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

    sys_trc = System(sys; boundary=TriclinicBoundary(Molly.box_matrix(sys.boundary)))
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

@testset "Virial and pressure gradients" begin
    T = Float64
    n_atoms = 8
    boundary = CubicBoundary(T(3.0))
    coords = place_atoms(n_atoms, boundary; min_dist=T(0.5), rng=Xoshiro(3))
    velocities = [random_velocity(T(10.0), T(1.0); rng=Xoshiro(4 + i)) for i in 1:n_atoms]
    atoms = [Atom(index=i, mass=T(10.0), charge=(iseven(i) ? T(0.2) : T(-0.2)), σ=T(0.3),
                  ϵ=T(0.4)) for i in 1:n_atoms]
    neighbor_finder = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=1,
                                             dist_cutoff=T(1.5))
    pairwise_inters = (
        LennardJones(cutoff=DistanceCutoff(T(1.2))),
        CoulombReactionField(dist_cutoff=T(1.2),
                             solvent_dielectric=T(Molly.crf_solvent_dielectric),
                             coulomb_const=T(ustrip(Molly.coulomb_const))),
    )
    bonds = InteractionList2Atoms(Int32.(collect(1:4)), Int32.(collect(5:8)),
                                  [HarmonicBond(T(100.0), T(0.6)) for _ in 1:4])

    function build_sys(coords, velocities)
        return System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            specific_inter_lists=(bonds,),
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
            grad_safe=true,
        )
    end

    neighbors = find_neighbors(build_sys(coords, velocities); n_threads=1)
    loss_virial(  coords, velocities) = scalar_virial(  build_sys(coords, velocities),
                                                      neighbors; n_threads=1)
    loss_pressure(coords, velocities) = scalar_pressure(build_sys(coords, velocities),
                                                        neighbors; n_threads=1)

    # The virial does not depend on the velocities, the pressure does through the kinetic
    #   energy tensor
    for (f, uses_velocities) in ((loss_virial, false), (loss_pressure, true))
        d_coords, d_velocities = zero(coords), zero(velocities)
        autodiff(
            set_runtime_activity(Reverse),
            Const(f), # The loss closes over the interactions, which Enzyme cannot prove read only
            Active,
            Duplicated(copy(coords), d_coords),
            Duplicated(copy(velocities), d_velocities),
        )
        for i in (1, 2)
            grad_fd = central_fdm(5, 1)(coords[i][1]) do x
                coords_mod = copy(coords)
                coords_mod[i] = SVector(x, coords_mod[i][2], coords_mod[i][3])
                f(coords_mod, velocities)
            end
            @test d_coords[i][1] ≈ grad_fd rtol=1e-8
        end
        if uses_velocities
            grad_fd = central_fdm(5, 1)(velocities[1][1]) do x
                velocities_mod = copy(velocities)
                velocities_mod[1] = SVector(x, velocities_mod[1][2], velocities_mod[1][3])
                f(coords, velocities_mod)
            end
            @test d_velocities[1][1] ≈ grad_fd rtol=1e-8
        else
            @test iszero(d_velocities)
        end
    end
end

@testset "CV gradients" begin
    function forces_test!(fs, sys, bias; grad_cv=Molly.cv_gradient_enzyme, kwargs...)
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
        d_bias = Molly.bias_gradient(bias.bias_type, cv_sim)

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
    cv_ang   = CalcAngle([1,2,3])

    cvs = (cv_d_s, cv_d_min, cv_d_max, cv_d_cm, cv_rg, cv_rmsd, cv_tor, cv_ang)
    
    b1 = LinearBias(100.0u"kJ*mol^-1*nm^-1", 0.2u"nm")
    b2 = LinearBias(100.0u"kJ*mol^-1", 0.2)

    bias = (b1, b1, b1, b1, b1, b1, b2, b2)

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

        forces_test!(fs_zero_enz, sys, bias_pot; grad_cv=Molly.cv_gradient_enzyme)
        forces_test!(fs_zero_anl, sys, bias_pot; grad_cv=Molly.cv_gradient)
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
                      (i <= n_alchemical ? Molly.InsertRole : Molly.EnvRole))
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
                                        cutoff=DistanceCutoff(nb_cutoff)),
                                  T(0.4)),
        ("LennardJonesSCGapsys" , LennardJonesSoftCoreGapsys(α=T(0.85),
                                        cutoff=DistanceCutoff(nb_cutoff)),
                                  T(0.4)),
        ("CoulombSCBeutler"     , CoulombSoftCoreBeutler(cutoff=DistanceCutoff(nb_cutoff),
                                        coulomb_const=cc),
                                  T(0.7)),
        ("CoulombSCGapsys"      , CoulombSoftCoreGapsys(cutoff=DistanceCutoff(nb_cutoff),
                                        σQ=T(1.0), coulomb_const=cc),
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
    lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff))
    crf = CoulombReactionField(
        dist_cutoff=nb_cutoff,
        solvent_dielectric=T(Molly.crf_solvent_dielectric),
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
                      Molly.EnvRole) for i in 1:n_atoms]

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

    # The energy minimizers do not take a number of steps
    function loss_min(σ, coords, velocities, boundary, pairwise_inters, neighbor_finder,
                      minimizer, n_atoms, atom_mass)
        atoms = [Atom(i, 1, atom_mass, (i % 2 == 0 ? -charge : charge), σ, ϵ, λ,
                      Molly.EnvRole) for i in 1:n_atoms]

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=pairwise_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulate!(sys, minimizer; n_threads=1)

        return mean_min_separation(sys.coords, boundary)
    end

    function test_min_grad(name, minimizer; tol=1e-5)
        grad_enzyme = autodiff(
            set_runtime_activity(Reverse),
            loss_min,
            Active,
            Active(σ_start),
            Duplicated(copy(coords), zero(coords)),
            Duplicated(copy(velocities), zero(velocities)),
            Const(boundary), Const((lj, crf)), Const(neighbor_finder),
            Const(minimizer), Const(n_atoms), Const(atom_mass),
        )[1][1]
        grad_fd = central_fdm(6, 1)(σ_start) do σ
            loss_min(σ, copy(coords), copy(velocities), boundary, (lj, crf),
                     neighbor_finder, minimizer, n_atoms, atom_mass)
        end
        frac_diff = abs(grad_enzyme - grad_fd) / abs(grad_fd)
        @test frac_diff < tol
    end

    # A tolerance of zero means that the number of steps taken does not depend on σ, so
    #   the loss is smooth and can be compared to finite differences
    n_min_steps = 10
    minimizers = [
        ("SteepestDescentMinimizer", SteepestDescentMinimizer(step_size=T(0.01),
                                        tol=zero(T), max_steps=n_min_steps)),
        ("FIREMinimizer"           , FIREMinimizer(dt=T(0.001), dt_max=T(0.01),
                                        tol=zero(T), max_steps=n_min_steps)),
        ("LBFGSMinimizer"          , LBFGSMinimizer(step_size=T(0.01), tol=zero(T),
                                        max_steps=n_min_steps)),
    ]
    for (name, minimizer) in minimizers
        test_min_grad(name, minimizer)
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
                        Molly.EnvRole) for i in 1:n_atoms]

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
    lj_units = LennardJones(cutoff=DistanceCutoff(1.2u"nm"))
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

    function pe_injected(params_dic, sys_ref, plan, coords, neighbors)
        sys = inject_gradients(sys_ref, params_dic, plan, coords)
        return potential_energy(sys, neighbors; n_threads=1)
    end

    # Extract the parameters of an interaction, inject them back and check that the
    #   gradient with respect to each one matches finite differencing
    # Enzyme currently drops the gradient of the trailing fields of some interaction
    #   types when they are read back out of a System, those keys are listed in broken
    function test_params(name, sys_ref, params_dic, broken=())
        @test length(params_dic) > 0
        neighbors = find_neighbors(sys_ref; n_threads=1)
        plan = ParameterPlan(sys_ref, params_dic)
        grads_enzyme = Dict(k => zero(v) for (k, v) in params_dic)
        autodiff(
            set_runtime_activity(Reverse),
            Const(pe_injected),
            Active,
            Duplicated(params_dic, grads_enzyme),
            Const(sys_ref),
            Const(plan),
            Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
            Const(neighbors),
        )
        n_nonzero = 0
        for param in sort(collect(keys(params_dic)))
            grad_fd = central_fdm(6, 1)(params_dic[param]) do val
                dic = copy(params_dic)
                dic[param] = val
                pe_injected(dic, sys_ref, plan, copy(sys_ref.coords), neighbors)
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
                                    weight_special=w), lost_w),
        ("Mie"              , Mie(m=T(6), n=T(12), cutoff=DistanceCutoff(nb_cutoff),
                                    weight_special=w), lost_w),
        ("AshbaughHatch"    , AshbaughHatch(cutoff=DistanceCutoff(nb_cutoff),
                                    weight_special=w), lost_w),
        ("DoubleExponential", DoubleExponential(α=T(16.766), β=T(4.427), weight_special=w,
                                    cutoff=DistanceCutoff(nb_cutoff)), ()),
        ("DoubleExponentialSC", DoubleExponentialSoftCore(α=T(16.766), β=T(4.427),
                                    weight_special=w, cutoff=DistanceCutoff(nb_cutoff)), ()),
        ("Gravity"          , Gravity(G=T(1.0)), ()),
        ("LJSCBeutler"      , LennardJonesSoftCoreBeutler(α=T(0.5), weight_special=w,
                                    cutoff=DistanceCutoff(nb_cutoff)), ()),
        ("LJSCGapsys"       , LennardJonesSoftCoreGapsys(α=T(0.85), weight_special=w,
                                    cutoff=DistanceCutoff(nb_cutoff)), ()),
        ("Coulomb"          , Coulomb(cutoff=DistanceCutoff(nb_cutoff),
                                    weight_special=w, coulomb_const=cc), ()),
        ("CoulombScaled"    , Molly.CoulombScaled(cutoff=DistanceCutoff(nb_cutoff),
                                    weight_special=w, coulomb_const=cc), ()),
        ("CoulombSCBeutler" , CoulombSoftCoreBeutler(cutoff=DistanceCutoff(nb_cutoff),
                                    weight_special=w, coulomb_const=cc), ()),
        ("CoulombSCGapsys"  , CoulombSoftCoreGapsys(cutoff=DistanceCutoff(nb_cutoff),
                                    σQ=T(1.0), weight_special=w, coulomb_const=cc), ()),
        ("CoulombRF"        , CoulombReactionField(dist_cutoff=nb_cutoff,
                                    weight_special=w, coulomb_const=cc), ()),
        ("CRFScaled"        , Molly.CoulombReactionFieldScaled(dist_cutoff=nb_cutoff,
                                    weight_special=w, coulomb_const=cc), ()),
        ("CRFSCBeutler"     , CoulombSoftCoreBeutlerReactionField(dist_cutoff=nb_cutoff,
                                    weight_special=w, coulomb_const=cc), ()),
        ("CRFSCGapsys"      , CoulombSoftCoreGapsysReactionField(dist_cutoff=nb_cutoff,
                                    σQ=T(1.0), weight_special=w, coulomb_const=cc), ()),
        ("CoulombEwald"     , CoulombEwald(dist_cutoff=nb_cutoff,
                                    weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), ()),
        ("CoulombEwaldScaled", Molly.CoulombEwaldScaled(dist_cutoff=nb_cutoff,
                                    weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), ()),
        ("CEwaldSCBeutler"  , CoulombSoftCoreBeutlerEwald(dist_cutoff=nb_cutoff,
                                    weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), ()),
        ("CEwaldSCGapsys"   , CoulombSoftCoreGapsysEwald(dist_cutoff=nb_cutoff, σQ=T(1.0),
                                    weight_special=w, coulomb_const=cc,
                                    approximate_erfc=false), ()),
        ("Yukawa"           , Yukawa(cutoff=DistanceCutoff(nb_cutoff),
                                    weight_special=w, coulomb_const=cc, kappa=T(1.0)), ()),
    ]
    for (name, inter, broken) in pairwise_inters
        sys_ref = pairwise_sys(inter)
        params_dic = Dict{String, Float64}()
        Molly.extract_block!(params_dic, Molly.parameter_prefix(inter),
                             Molly.parameter_keys(inter), Molly.parameter_values(inter))
        test_params(name, sys_ref, params_dic, broken)
    end

    # A λ interaction followed by another one, as an alchemical system has them; the layout
    #   of the first must not drop the gradients of the second
    let inters = Dict(name => inter for (name, inter, _) in pairwise_inters),
            sys_ref = System(pairwise_sys(inters["DoubleExponentialSC"]);
                             pairwise_inters=(inters["DoubleExponentialSC"], inters["CoulombScaled"]))
        params_dic = Dict{String, Float64}()
        for inter in sys_ref.pairwise_inters
            Molly.extract_block!(params_dic, Molly.parameter_prefix(inter),
                                 Molly.parameter_keys(inter), Molly.parameter_values(inter))
        end
        test_params("DoubleExponentialSC + CoulombScaled", sys_ref, params_dic)
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
        for (inter, inter_type) in zip(inter_list.inters, inter_list.types)
            Molly.extract_block!(params_dic, Molly.parameter_prefix(inter, inter_type),
                                 Molly.parameter_keys(inter), Molly.parameter_values(inter))
        end
        test_params(name, sys_ref, params_dic)
    end

    # The λ versions that alchemical systems use have the keys of the interactions they
    #   replace. Dual topology keeps the bonded terms of deleted atoms at full strength, so
    #   the A copies of core atoms are used, whose bonded terms are scaled by 1 - λ
    atoms_λ = [Atom(index=i, mass=T(10.0), charge=T(0.1), σ=T(0.3), ϵ=T(0.4), λ=T(0.6),
                    alch_role=(i <= 2 ? Molly.CoreDRole : Molly.EnvRole))
               for i in eachindex(coords_spec)]
    scheduler = Molly.DefaultLambdaScheduler(dual=true)
    to_λ(inter) = Molly.to_lambda_function(inter; scheduler=scheduler)
    to_λ(inter::Molly.LennardJones14) = Molly.to_lambda_function(inter, Molly.GapsysSoftCore();
                                                                 scheduler=scheduler, float_type=T)
    to_λ(inter::Molly.LennardJones14, softcore) =
        Molly.to_lambda_function(inter, softcore; scheduler=scheduler, float_type=T)
    λ_inters = Any[(name, il, to_λ) for (name, il) in specific_inters if name != "FENEBond" &&
                                                     name != "HarmonicPositionRestraint"]
    for softcore in (Molly.BeutlerSoftCore(), Molly.ScaledSoftCore())
        lj14 = specific_inters[findfirst(x -> first(x) == "LennardJones14", specific_inters)][2]
        push!(λ_inters, ("LennardJones14 $(nameof(typeof(softcore)))", lj14,
                         inter -> to_λ(inter, softcore)))
    end
    for (name, inter_list, convert) in λ_inters
        il_λ = typeof(inter_list).name.wrapper(
            (getfield(inter_list, f) for f in fieldnames(typeof(inter_list))[1:(end - 3)])...,
            convert.(inter_list.inters), ["t"], inter_list.data)
        sys_ref = System(specific_sys(il_λ); atoms=atoms_λ)
        # The coupling must change the energy, or the λ part would go untested
        @test !(potential_energy(sys_ref) ≈ potential_energy(specific_sys(inter_list)))
        params_dic = extract_parameters(sys_ref)
        keys_base = [Molly.parameter_prefix(inter_list.inters[1], "t") * k
                     for k in Molly.parameter_keys(inter_list.inters[1])]
        @test all(k -> haskey(params_dic, k), keys_base)
        inter_params = Dict(k => params_dic[k] for k in keys_base)
        test_params("$(name)λ", sys_ref, inter_params)
    end

    # Single topology λ interactions store the parameters of both end states
    hb_single = Molly.to_lambda_function_single(HarmonicBond(k=T(100.0), r0=T(0.5)), nothing;
                    scheduler=Molly.DefaultLambdaScheduler(dual=false))
    sys_single = specific_sys(InteractionList2Atoms(is, js, [hb_single], ["t"]))
    @test_throws "single topology" extract_parameters(sys_single)
    @test_throws "single topology" ParameterPlan(sys_single, Dict("inter_HB_t_k" => T(100.0)))
end

@testset "Parameters with units" begin
    function garnet_sys(AT, units)
        ff = MolecularForceField(
            joinpath(ff_dir, "ethanol_garnet.xml");
            custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"),
            units=units,
        )
        return System(
            joinpath(data_dir, "ethanol_garnet.pdb"),
            ff;
            units=units,
            array_type=AT,
            float_type=Float64,
            nonbonded_method=SetupCoulombReactionField(),
            dist_cutoff=(units ? 1.0u"nm" : 1.0),
            strictness=:nowarn,
        )
    end

    # Single-threaded so that the energies and forces can be compared exactly
    pe(sys) = potential_energy(sys; n_threads=1)
    fs(sys) = from_device(forces(sys; n_threads=1))

    sys_units   = garnet_sys(Array, true)
    sys_nounits = garnet_sys(Array, false)
    params_dic = extract_parameters(sys_units)

    # Plain numbers in the units that each parameter is stored with in the system, which
    #   for this force field are the same numbers as for the unitless system
    @test params_dic isa Dict{String, Float64}
    @test params_dic == extract_parameters(sys_nounits)
    i_C1 = findfirst(ad -> ad.atom_type == "C1", sys_units.atoms_data)
    @test params_dic["atom_C1_σ"] == ustrip(u"nm", sys_units.atoms[i_C1].σ)
    @test params_dic["atom_C1_ϵ"] == ustrip(u"kJ * mol^-1", sys_units.atoms[i_C1].ϵ)
    @test params_dic["atom_C1_mass"] == ustrip(u"g/mol", sys_units.atoms[i_C1].mass)

    bonds_units = sys_units.specific_inter_lists[1]
    bond_key = "inter_HB_$(bonds_units.types[1])_k"
    @test params_dic[bond_key] == ustrip(u"kJ * mol^-1 * nm^-2", bonds_units.inters[1].k)

    # Per-interaction helpers take plain numbers in the units of the fields or values
    #   with units
    bond = bonds_units.inters[1]
    @test Molly.inject_parameters(bond, Molly.parameter_values(bond)...) == bond
    @test Molly.inject_parameters(bond, ustrip.(Molly.parameter_values(bond))...) == bond
    bond_Å = Molly.inject_parameters(bond, bond.k / 100 * u"nm^2/Å^2", 10 * bond.r0 * u"Å/nm")
    @test bond_Å.k ≈ bond.k
    @test bond_Å.r0 ≈ bond.r0
    torsion = sys_units.specific_inter_lists[3].inters[1]
    @test Molly.inject_parameters(torsion, ustrip.(Molly.parameter_values(torsion))...) == torsion

    # Injecting unchanged values leaves the system unchanged
    plan = ParameterPlan(sys_units, params_dic)
    sys_rt = inject_gradients(sys_units, params_dic, plan)
    @test sys_rt.atoms == sys_units.atoms
    @test sys_rt.masses == sys_units.masses
    @test all(il_rt.inters == il.inters for (il_rt, il) in zip(
                values(sys_rt.specific_inter_lists), values(sys_units.specific_inter_lists)))
    @test values(sys_rt.pairwise_inters) == values(sys_units.pairwise_inters)
    @test pe(sys_rt) == pe(sys_units)
    @test fs(sys_rt) == fs(sys_units)

    # Inject one modified value per parameter family, into the system with units and into
    #   the unitless system
    torsion_key = only(k for k in keys(params_dic)
                       if startswith(k, "inter_PT_H1/C1/C2/H4_") && endswith(k, "_k_3"))
    modified = Dict(
        "atom_C1_σ"     => params_dic["atom_C1_σ"] * 1.05,
        "atom_O1_ϵ"     => params_dic["atom_O1_ϵ"] * 0.9,
        "atom_H1_mass"  => params_dic["atom_H1_mass"] * 2,
        bond_key        => params_dic[bond_key] * 0.8,
        torsion_key     => params_dic[torsion_key] + 1.0,
        "inter_DEXP_alpha" => params_dic["inter_DEXP_alpha"] * 1.1,
        "atom_O1_charge" => params_dic["atom_O1_charge"] * 1.1,
    )
    params_mod = merge(params_dic, modified)
    sys_mod = inject_gradients(sys_units, params_mod, plan)
    sys_mod_nounits = inject_gradients(sys_nounits, params_mod)

    params_mod_extracted = extract_parameters(sys_mod)
    for (k, v) in modified
        @test params_mod_extracted[k] == v
    end
    @test params_mod_extracted == params_mod
    @test unit(sys_mod.atoms[i_C1].σ) == u"nm"
    @test eltype(sys_mod.masses) == eltype(sys_units.masses)
    @test sys_mod.total_mass ≈ sum(sys_mod.masses)
    @test sys_mod.total_mass > sys_units.total_mass

    E_mod = pe(sys_mod)
    @test !isapprox(E_mod, pe(sys_units); rtol=1e-6)
    @test ustrip(u"kJ * mol^-1", E_mod) ≈ pe(sys_mod_nounits) rtol=1e-12
    @test map(f -> ustrip.(u"kJ * mol^-1 * nm^-1", f), fs(sys_mod)) ≈ fs(sys_mod_nounits) rtol=1e-12

    # The dispersion correction takes up the injected α, σ and ϵ, as does its λ version, so
    #   both match those of a system set up with the modified force field
    xml = read(joinpath(ff_dir, "ethanol_garnet.xml"), String)
    xml_mod = replace(xml,
        "name=\"alpha\" defaultValue=\"12.159626\"" =>
            "name=\"alpha\" defaultValue=\"$(modified["inter_DEXP_alpha"])\"",
        "<Atom type=\"C1\" sigma=\"0.3358508050441742\"" =>
            "<Atom type=\"C1\" sigma=\"$(modified["atom_C1_σ"])\"",
        "epsilon=\"0.9317593574523926\"" => "epsilon=\"$(modified["atom_O1_ϵ"])\"",
    )
    @test count("$(modified["inter_DEXP_alpha"])", xml_mod) == 1
    garnet_mod = joinpath(mktempdir(), "ethanol_garnet.xml")
    write(garnet_mod, xml_mod)
    sys_rebuilt = System(
        joinpath(data_dir, "ethanol_garnet.pdb"),
        MolecularForceField(garnet_mod; units=false,
            custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"));
        units=false, float_type=Float64, nonbonded_method=SetupCoulombReactionField(),
        dist_cutoff=1.0, strictness=:nowarn,
    )
    disp(s) = potential_energy(s; pairwise_inters=(), specific_inter_lists=())
    @test disp(sys_mod_nounits) ≈ disp(sys_rebuilt) rtol=1e-12
    @test !isapprox(disp(sys_mod_nounits), disp(sys_nounits); rtol=1e-3)
    solute = collect(1:9)
    sys_λ_mod = inject_gradients(AbsoluteFESystem(sys_nounits, 0.8, solute), params_mod)
    @test disp(sys_λ_mod) ≈ disp(AbsoluteFESystem(sys_rebuilt, 0.8, solute)) rtol=1e-12
    @test disp(Molly.set_lambda!(sys_λ_mod, 0.6)) ≈
          disp(AbsoluteFESystem(sys_rebuilt, 0.6, solute)) rtol=1e-12

    for AT in array_list[2:end]
        sys_gpu = garnet_sys(AT, true)
        plan_gpu = ParameterPlan(sys_gpu, params_dic)
        @test extract_parameters(sys_gpu) == params_dic
        @test pe(inject_gradients(sys_gpu, params_dic, plan_gpu)) ≈ pe(sys_units) rtol=1e-10
        sys_mod_gpu = inject_gradients(sys_gpu, params_mod, plan_gpu)
        @test sys_mod_gpu.atoms isa AT
        @test extract_parameters(sys_mod_gpu) == params_mod
        @test pe(sys_mod_gpu) ≈ E_mod rtol=1e-10
    end
end

@testset "Charge parameters" begin
    # A copy of a force field file with the residue template charges of some atom types
    #   changed, to compare injected charges with building the system again
    function ff_file_with_charges(ff_file, charges)
        xml = read(ff_file, String)
        for (atom_type, q) in charges
            pattern = Regex("(type=\"$atom_type\" charge=\")[^\"]*")
            @test occursin(pattern, xml)
            xml = replace(xml, pattern => SubstitutionString("\\g<1>$q"))
        end
        path = joinpath(mktempdir(), basename(ff_file))
        write(path, xml)
        return path
    end

    pe(sys) = potential_energy(sys; n_threads=1)
    fs(sys) = from_device(forces(sys; n_threads=1))
    pme_of(sys) = only(inter for inter in values(sys.general_inters) if inter isa PME)

    garnet_file = joinpath(ff_dir, "ethanol_garnet.xml")
    # O2 is the water oxygen, O1 the ethanol oxygen
    charges_mod = ("O2" => -0.95, "O1" => -0.6)
    garnet_file_mod = ff_file_with_charges(garnet_file, charges_mod)
    function ethanol_solv_sys(ff_file, nonbonded_method; grad_safe=false)
        ff = MolecularForceField(
            ff_file;
            custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"),
            units=false,
        )
        return System(
            joinpath(data_dir, "ethanol_solv.pdb"),
            ff;
            units=false,
            float_type=Float64,
            nonbonded_method=nonbonded_method,
            dist_cutoff=1.0,
            grad_safe=grad_safe,
            strictness=:nowarn,
        )
    end

    nonbonded_methods = (("RF", SetupCoulombReactionField()), ("PME", SetupPME()))
    @testset "Injection $nb_name" for (nb_name, nonbonded_method) in nonbonded_methods
        sys = ethanol_solv_sys(garnet_file, nonbonded_method)
        params_dic = extract_parameters(sys)
        i_O2 = findfirst(ad -> ad.atom_type == "O2", sys.atoms_data)
        @test params_dic["atom_O2_charge"] == sys.atoms[i_O2].charge
        @test count(k -> endswith(k, "_charge"), keys(params_dic)) == 12

        # Injecting unchanged values leaves the system unchanged
        plan = ParameterPlan(sys, params_dic)
        sys_rt = inject_gradients(sys, params_dic, plan)
        @test sys_rt.atoms == sys.atoms
        @test pe(sys_rt) == pe(sys)

        # Changed charges give the system built with those charges, including the charge
        #   sums cached by a PME with fixed charges
        sys_mod = inject_gradients(sys, merge(params_dic,
                        Dict("atom_$(t)_charge" => q for (t, q) in charges_mod)), plan)
        sys_rebuilt = ethanol_solv_sys(garnet_file_mod, nonbonded_method)
        @test sys_mod.atoms == sys_rebuilt.atoms
        @test extract_parameters(sys_mod) == extract_parameters(sys_rebuilt)
        if nb_name == "PME"
            @test !isnothing(pme_of(sys_mod).pc_sum)
            @test pme_of(sys_mod).pc_sum != pme_of(sys).pc_sum
            @test pme_of(sys_mod).pc_sum == pme_of(sys_rebuilt).pc_sum
            @test pme_of(sys_mod).pc_abs2_sum == pme_of(sys_rebuilt).pc_abs2_sum
        end
        @test !isapprox(pe(sys_mod), pe(sys); rtol=1e-6)
        @test pe(sys_mod) ≈ pe(sys_rebuilt) rtol=1e-12
        @test fs(sys_mod) ≈ fs(sys_rebuilt) rtol=1e-12
    end

    # Enzyme against finite differencing, with every parameter of the system in the
    #   dictionary; σ and ϵ are checked too since they share the atom injection
    # A PME has to be passed as Duplicated, so the general interactions are passed
    #   separately; a grad-safe PME caches no charges, so injection leaves them unchanged
    function pe_injected(params_dic, sys_ref, plan, coords, neighbors, general_inters)
        sys = inject_gradients(sys_ref, params_dic, plan, coords)
        return potential_energy(sys, neighbors; n_threads=1, general_inters=general_inters)
    end

    @testset "Gradients $nb_name" for (nb_name, nonbonded_method) in nonbonded_methods
        sys = ethanol_solv_sys(garnet_file, nonbonded_method; grad_safe=true)
        params_dic = extract_parameters(sys)
        plan = ParameterPlan(sys, params_dic)
        neighbors = find_neighbors(sys; n_threads=1)
        # The dispersion correction alongside the PME checks that Enzyme handles general
        #   interactions of different types
        gis = sys.general_inters
        @test any(gi -> gi isa DEXPDispersionCorrection, gis)
        @test all(gi -> !(gi isa PME) || gi.grad_safe, gis)
        grads_enzyme = Dict(k => zero(v) for (k, v) in params_dic)
        autodiff(
            set_runtime_activity(Reverse),
            Const(pe_injected),
            Active,
            Duplicated(params_dic, grads_enzyme),
            Const(sys),
            Const(plan),
            Duplicated(copy(sys.coords), zero(sys.coords)),
            Const(neighbors),
            Duplicated(gis, Enzyme.make_zero(gis)),
        )
        for key in ("atom_O2_charge", "atom_H7_charge", "atom_O1_charge", "atom_C1_σ",
                    "atom_C1_ϵ")
            grad_fd = central_fdm(6, 1)(params_dic[key]) do val
                pe_injected(merge(params_dic, Dict(key => val)), sys, plan,
                            copy(sys.coords), neighbors, gis)
            end
            @test abs(grads_enzyme[key] - grad_fd) <= 1e-6 * abs(grad_fd)
        end
    end

    # The alchemical atoms keep their λ and role, so their charges are still scaled by λ
    @testset "Alchemical $nb_name" for (nb_name, nonbonded_method) in nonbonded_methods
        ff_files = joinpath.(ff_dir, ["tip3p_standard.xml", "benzene.xml"])
        ff_files_mod = [ff_file_with_charges(ff_files[1], ("tip3p-O" => -0.9,)),
                        ff_file_with_charges(ff_files[2], ("BZNC1" => -0.2,))]
        benzene_sys(files) = System(joinpath(data_dir, "benzene_solv.pdb"),
                                    MolecularForceField(files...; units=true);
                                    nonbonded_method=nonbonded_method, dist_cutoff=1.0u"nm",
                                    center_coords=false)
        sys, sys_rebuilt = benzene_sys(ff_files), benzene_sys(ff_files_mod)
        solute = collect(1:12)
        for S in (Molly.DefaultLambdaScheduler, Molly.GROMACSLambdaABFEScheduler)
            # Partly decoupled, so the charges of the solute are partly scaled
            sys_λ = AbsoluteFESystem(sys, 0.25, solute; scheduler=S(dual=true))
            sys_λ_rebuilt = AbsoluteFESystem(sys_rebuilt, 0.25, solute; scheduler=S(dual=true))
            params_dic = extract_parameters(sys_λ)
            params_mod = merge(params_dic, Dict("atom_tip3p-O_charge" => -0.9,
                                                "atom_BZNC1_charge" => -0.2))
            sys_λ_mod = inject_gradients(sys_λ, params_mod)
            @test sys_λ_mod.atoms == sys_λ_rebuilt.atoms
            @test any(a -> a.λ != 1, sys_λ_mod.atoms[solute])
            E_mod = ustrip(pe(sys_λ_mod))
            @test !isapprox(E_mod, ustrip(pe(sys_λ)); rtol=1e-6)
            @test E_mod ≈ ustrip(pe(sys_λ_rebuilt)) rtol=1e-12
        end
    end

    @testset "Charge types" begin
        # One atom type, XX, with two charges told apart by their charge types, and a mass
        #   that differs between its atoms
        atoms = [Atom(index=i, mass=m, charge=q, σ=0.3, ϵ=0.4)
                 for (i, (m, q)) in enumerate(((10.0, -0.2), (10.0, 0.2), (12.0, 0.2)))]
        function sys_with(atoms, charge_types)
            return System(
                atoms=atoms,
                coords=[SVector(x, 1.0, 1.0) for x in (0.5, 1.0, 1.5)],
                boundary=CubicBoundary(4.0),
                atoms_data=[AtomData(atom_type="XX", charge_type=t) for t in charge_types],
                force_units=NoUnits,
                energy_units=NoUnits,
            )
        end
        sys = sys_with(atoms, ("XX-", "XX+", "XX+"))
        params_dic = extract_parameters(sys)
        @test params_dic["atom_XX-_charge"] == -0.2
        @test params_dic["atom_XX+_charge"] == 0.2
        @test !haskey(params_dic, "atom_XX_charge")
        @test haskey(params_dic, "atom_XX_σ")
        @test !haskey(params_dic, "atom_XX_mass") # The masses differ
        @test inject_gradients(sys, params_dic).atoms == sys.atoms
        sys_mod = inject_gradients(sys, merge(params_dic, Dict("atom_XX+_charge" => 0.3)))
        @test charges(sys_mod) == [-0.2, 0.3, 0.3]
        @test_throws ArgumentError ParameterPlan(sys, merge(params_dic, Dict("atom_XX_mass" => 10.0)))

        # Masses that differ are left out silently, other parameters are reported
        @test_logs extract_parameters(sys)

        # Without charge types, the atom type groups charges that differ
        sys_type = sys_with(atoms, ("", "", ""))
        @test !haskey(extract_parameters(sys_type; strictness=:nowarn), "atom_XX_charge")
        @test_logs (:warn, r"atom_XX_charge") extract_parameters(sys_type; strictness=:warn)
        @test_throws ErrorException extract_parameters(sys_type; strictness=:error)
        @test_logs extract_parameters(sys_type; strictness=:nowarn)
        @test_throws ArgumentError ParameterPlan(sys_type, Dict("atom_XX_charge" => 0.0))

        # Atoms with two end states, as in single topology, are not supported
        atoms_two = [Atom(a.index, a.atom_type, a.mass, (a.charge, a.charge), a.σ, a.ϵ, a.λ,
                          a.alch_role) for a in atoms]
        sys_two = sys_with(atoms_two, ("", "", ""))
        @test_throws ArgumentError extract_parameters(sys_two)
        @test_throws ArgumentError ParameterPlan(sys_two, Dict{String, Float64}())
    end

    @testset "Protein charge types" begin
        # LEU CA shares its atom type with atoms of other charges, so it has its own
        #   charge type
        ff99_file = joinpath(ff_dir, "ff99SBildn.xml")
        xml = read(ff99_file, String)
        pattern = r"(<Residue name=\"LEU\">.*?<Atom name=\"CA\" type=\"CT\" charge=\")[^\"]*"s
        @test occursin(pattern, xml)
        ff99_file_mod = joinpath(mktempdir(), "ff99SBildn.xml")
        write(ff99_file_mod, replace(xml, pattern => s"\g<1>-0.1"))
        key = "atom_LEU/CT/CA_charge"
        protein_sys(ff_file, nonbonded_method; grad_safe=false) = System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            MolecularForceField(ff_file; units=false);
            units=false,
            nonbonded_method=nonbonded_method,
            dispersion_correction=false,
            grad_safe=grad_safe,
            strictness=:nowarn,
        )

        for (nb_name, nonbonded_method) in nonbonded_methods
            sys = protein_sys(ff99_file, nonbonded_method)
            params_dic = extract_parameters(sys)
            # Every charge type is a parameter
            @test count(k -> endswith(k, "_charge"), keys(params_dic)) ==
                    length(unique(ad.charge_type for ad in sys.atoms_data))
            @test params_dic[key] == -0.0518
            plan = ParameterPlan(sys, params_dic)
            @test inject_gradients(sys, params_dic, plan).atoms == sys.atoms

            sys_mod = inject_gradients(sys, merge(params_dic, Dict(key => -0.1)), plan)
            sys_rebuilt = protein_sys(ff99_file_mod, nonbonded_method)
            @test sys_mod.atoms == sys_rebuilt.atoms
            @test !isapprox(pe(sys_mod), pe(sys); rtol=1e-8)
            @test pe(sys_mod) ≈ pe(sys_rebuilt) rtol=1e-12
            @test fs(sys_mod) ≈ fs(sys_rebuilt) rtol=1e-12
        end

        # Enzyme against finite differencing with PME, passed as Duplicated
        sys = protein_sys(ff99_file, SetupPME(); grad_safe=true)
        params_dic = extract_parameters(sys)
        plan = ParameterPlan(sys, params_dic)
        neighbors = find_neighbors(sys; n_threads=1)
        gis = sys.general_inters
        grads_enzyme = Dict(k => zero(v) for (k, v) in params_dic)
        autodiff(
            set_runtime_activity(Reverse),
            Const(pe_injected),
            Active,
            Duplicated(params_dic, grads_enzyme),
            Const(sys),
            Const(plan),
            Duplicated(copy(sys.coords), zero(sys.coords)),
            Const(neighbors),
            Duplicated(gis, Enzyme.make_zero(gis)),
        )
        grad_fd = central_fdm(6, 1)(params_dic[key]) do val
            pe_injected(merge(params_dic, Dict(key => val)), sys, plan, copy(sys.coords),
                        neighbors, gis)
        end
        @test abs(grads_enzyme[key] - grad_fd) <= 1e-6 * abs(grad_fd)
    end
end

@testset "Dispersion correction gradients" begin
    # The corrections are recomputed from the injected parameters inside the differentiated
    #   function, so Enzyme gives their gradients, checked against finite differencing
    function disp_injected(params_dic, sys_ref, plan)
        sys = inject_gradients(sys_ref, params_dic, plan)
        return potential_energy(sys, nothing; n_threads=1, pairwise_inters=(),
                                specific_inter_lists=())
    end
    function build(ff_files, pdb; kwargs...)
        ff = MolecularForceField(joinpath.(ff_dir, ff_files)...; units=false, kwargs...)
        return System(joinpath(data_dir, pdb), ff; units=false, float_type=Float64,
                      nonbonded_method=SetupCoulombReactionField(), dist_cutoff=1.0,
                      strictness=:nowarn)
    end
    garnet_templates = joinpath(ff_dir, "ethanol_garnet_residues.xml")
    sys_dexp = build(["ethanol_garnet.xml"], "ethanol_solv.pdb";
                     custom_residue_templates=garnet_templates)
    sys_lj = build(["tip3p_standard.xml", "benzene.xml"], "benzene_solv.pdb")
    cases = (
        ("DEXP", sys_dexp, ("inter_DEXP_alpha", "inter_DEXP_beta", "atom_C1_σ", "atom_O1_ϵ"),
         DEXPDispersionCorrection),
        ("DEXP λ", AbsoluteFESystem(sys_dexp, 0.8, collect(1:9)),
         ("inter_DEXP_alpha", "inter_DEXP_beta", "atom_C1_σ", "atom_O1_ϵ"),
         Molly.DEXPDispersionCorrectionλ),
        ("LJ", sys_lj, ("atom_BZNC1_σ", "atom_BZNC1_ϵ", "atom_tip3p-O_σ"),
         LJDispersionCorrection),
        ("LJ λ", AbsoluteFESystem(sys_lj, 0.8, collect(1:12)),
         ("atom_BZNC1_σ", "atom_BZNC1_ϵ", "atom_tip3p-O_ϵ"), Molly.LJDispersionCorrectionλ),
    )
    @testset "$name" for (name, sys, ks, DC) in cases
        @test only(sys.general_inters) isa DC
        params_dic = extract_parameters(sys; strictness=:nowarn)
        plan = ParameterPlan(sys, params_dic)
        grads_enzyme = Dict(k => zero(v) for (k, v) in params_dic)
        autodiff(set_runtime_activity(Reverse), Const(disp_injected), Active,
                 Duplicated(params_dic, grads_enzyme), Const(sys), Const(plan))
        for key in ks
            grad_fd = central_fdm(6, 1)(params_dic[key]) do val
                disp_injected(merge(params_dic, Dict(key => val)), sys, plan)
            end
            @test !iszero(grad_fd)
            @test abs(grads_enzyme[key] - grad_fd) <= 1e-6 * abs(grad_fd)
        end
    end
end

@testset "Gradients of a converted system" begin
    ff_dexp = MolecularForceField(joinpath(ff_dir, "ethanol_garnet.xml"))
    ff_water = MolecularForceField(joinpath(ff_dir, "tip4pfb.xml"))

    sys_vac = System(joinpath(data_dir, "ethanol_vac.pdb"), ff_dexp; float_type=Float32,
                     nonbonded_method=SetupCoulombReactionField(), dist_cutoff=1.0u"nm",
                     strictness=:nowarn)
    sys_ref = System(sys_vac; float_type=Float64, units=false, grad_safe=true)
    params_dic = extract_parameters(sys_ref)
    plan = ParameterPlan(sys_ref, params_dic)
    neighbors = find_neighbors(sys_ref; n_threads=1)
    function pe_injected(params_dic, sys_ref, plan, coords, neighbors)
        sys = inject_gradients(sys_ref, params_dic, plan, coords)
        return potential_energy(sys, neighbors; n_threads=1)
    end
    grads = Dict(k => zero(v) for (k, v) in params_dic)
    autodiff(
        set_runtime_activity(Reverse),
        Const(pe_injected),
        Active,
        Duplicated(params_dic, grads),
        Const(sys_ref),
        Const(plan),
        Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
        Const(neighbors),
    )
    keys_checked = filter(k -> occursin("DEXP", k) || endswith(k, "_σ"),
                          collect(keys(params_dic)))
    @test !isempty(keys_checked)
    for key in keys_checked
        grad_fd = central_fdm(6, 1)(params_dic[key]) do val
            dic = copy(params_dic)
            dic[key] = val
            pe_injected(dic, sys_ref, plan, copy(sys_ref.coords), neighbors)
        end
        @test isapprox(grads[key], grad_fd; rtol=1e-6, atol=1e-8)
    end

    # The PME of a gradient safe converted system can be differentiated
    sys_water = System(joinpath(data_dir, "tip4pew.pdb"), ff_water; float_type=Float32,
                       nonbonded_method=SetupPME(), center_coords=false)
    sys = System(sys_water; float_type=Float64, units=false, grad_safe=true)
    pme = sys.general_inters[1]
    @test pme.grad_safe
    fs = zero(sys.coords)
    Molly.ewald_pe_forces!(fs, nothing, sys, pme, Val(false))
    d_sys = zero(sys)
    autodiff(
        Reverse,
        Molly.ewald_pe_forces!,
        Active,
        Const(zero(sys.coords)),
        Const(nothing),
        Duplicated(sys, d_sys),
        Duplicated(pme, zero(pme)),
        Const(Val(false)),
    )
    @test maximum(norm.(-d_sys.coords .- fs)) < 1e-8 * maximum(norm.(fs))
end
