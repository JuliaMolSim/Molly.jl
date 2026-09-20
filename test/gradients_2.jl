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
        LennardJones(cutoff=DistanceCutoff(T(1.2)), use_neighbors=true),
        CoulombReactionField(dist_cutoff=T(1.2),
                             solvent_dielectric=T(Molly.crf_solvent_dielectric),
                             use_neighbors=true,
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
        Molly.extract_block!(params_dic, parameter_prefix(inter), Molly.parameter_keys(inter),
                             Molly.parameter_values(inter))
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
        for (inter, inter_type) in zip(inter_list.inters, inter_list.types)
            Molly.extract_block!(params_dic, parameter_prefix(inter, inter_type),
                                 Molly.parameter_keys(inter), Molly.parameter_values(inter))
        end
        test_params(name, sys_ref, params_dic)
    end
end
