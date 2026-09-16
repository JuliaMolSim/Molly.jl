# Static analysis of the main code paths with JET

# Error analysis (test_call) is run on every entry point here. Optimization analysis
#   (test_opt) is only run where it can be clean, which is the analysis, spatial and
#   thermostat code: forces! and potential_energy deliberately use function barriers, since
#   Val(n_threads) and Val(needs_vir) are runtime values, so the call into the pairwise and
#   specific loops is a dynamic dispatch that specialises the loop body at run time.
#   Everything downstream of them, such as the barostats, inherits those reports.
# report_package is not used either, since most of its reports are either in dependencies
#   or on Any-argument paths that no concrete System reaches.

@testset "JET" begin
    jet_config = (target_modules=(Molly,),)

    function jet_lj_system(; units::Bool=true, T::Type=Float64, n_atoms::Integer=50)
        boundary = CubicBoundary(units ? T(3.0)u"nm" : T(3.0))
        coords = place_atoms(n_atoms, boundary; min_dist=(units ? T(0.3)u"nm" : T(0.3)))
        atoms = [Atom(mass=(units ? T(10.0)u"g/mol" : T(10.0)), charge=T(0.1),
                      σ=(units ? T(0.3)u"nm" : T(0.3)),
                      ϵ=(units ? T(0.2)u"kJ * mol^-1" : T(0.2))) for _ in 1:n_atoms]
        velocities = [units ? random_velocity(T(10.0)u"g/mol", T(100.0)u"K") :
                              random_velocity(T(10.0), T(100.0)) for _ in 1:n_atoms]
        cutoff = DistanceCutoff(units ? T(1.0)u"nm" : T(1.0))

        return System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(LennardJones(cutoff=cutoff, use_neighbors=true),
                             Coulomb(cutoff=cutoff, use_neighbors=true)),
            neighbor_finder=DistanceNeighborFinder(
                eligible=trues(n_atoms, n_atoms),
                n_steps=10,
                dist_cutoff=(units ? T(1.2)u"nm" : T(1.2)),
            ),
            loggers=(temp=TemperatureLogger(units ? Float64 : T, 10),),
            energy_units=(units ? u"kJ * mol^-1" : NoUnits),
            force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        )
    end

    ff_protein = MolecularForceField(
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...,
    )
    jet_protein(; kwargs...) = System(joinpath(data_dir, "6mrr_equil.pdb"), ff_protein;
                                      float_type=Float32, kwargs...)

    # Cover units/no units, cutoff/PME electrostatics, constraints and virtual sites
    jet_systems = [
        "LJ with units"    => jet_lj_system(),
        "LJ without units" => jet_lj_system(; units=false, T=Float32),
        "protein cutoff"   => jet_protein(),
        "protein PME"      => jet_protein(; nonbonded_method=SetupPME()),
        "protein hbonds"   => jet_protein(; constraints=:hbonds, rigid_water=true),
        "TIP4P"            => System(joinpath(data_dir, "tip4pew.pdb"),
                                     MolecularForceField(joinpath(ff_dir, "tip4pfb.xml"))),
    ]

    for (sys_name, sys) in jet_systems
        neighbors = find_neighbors(sys)
        JET.test_call(find_neighbors       , Base.typesof(sys); jet_config...)
        JET.test_call(forces               , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(forces_virial        , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(accelerations        , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(potential_energy     , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(kinetic_energy       , Base.typesof(sys); jet_config...)
        JET.test_call(kinetic_energy_tensor, Base.typesof(sys); jet_config...)
        JET.test_call(total_energy         , Base.typesof(sys); jet_config...)
        JET.test_call(temperature          , Base.typesof(sys); jet_config...)
        JET.test_call(virial               , Base.typesof(sys); jet_config...)
        JET.test_call(scalar_virial        , Base.typesof(sys); jet_config...)
        JET.test_call(pressure             , Base.typesof(sys); jet_config...)
        JET.test_call(scalar_pressure      , Base.typesof(sys); jet_config...)
        JET.test_call(masses               , Base.typesof(sys); jet_config...)
        JET.test_call(density              , Base.typesof(sys); jet_config...)
        JET.test_call(dipole_moment        , Base.typesof(sys); jet_config...)

        # The observables that do not need the forces or the potential energy are also
        #   gated on being free of runtime dispatch
        JET.test_opt(kinetic_energy       , Base.typesof(sys); jet_config...)
        JET.test_opt(kinetic_energy_tensor, Base.typesof(sys); jet_config...)
        JET.test_opt(temperature          , Base.typesof(sys); jet_config...)
        JET.test_opt(masses               , Base.typesof(sys); jet_config...)
        JET.test_opt(density              , Base.typesof(sys); jet_config...)
    end

    # These functions do not go through the force and energy function barriers, so they are
    #   also gated on being free of runtime dispatch with the optimization analysis
    for (sys_name, sys) in jet_systems
        boundary, coords = sys.boundary, sys.coords
        μ = SMatrix{3, 3}(1.01, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0, 0.0, 1.01)
        analysis_calls = [
            (displacements         , Base.typesof(coords, boundary)),
            (distances             , Base.typesof(coords, boundary)),
            (rmsd                  , Base.typesof(coords, coords)),
            (radius_gyration       , Base.typesof(coords, sys.atoms)),
            (hydrodynamic_radius   , Base.typesof(coords, boundary)),
            (Molly.molecule_centers, Base.typesof(coords, boundary, sys.topology)),
            (volume                , Base.typesof(boundary)),
            (box_center            , Base.typesof(boundary)),
            (scale_boundary        , Base.typesof(boundary, 1.01)),
            (random_coord          , Base.typesof(boundary)),
            (vector                , Base.typesof(coords[1], coords[2], boundary)),
            (wrap_coords           , Base.typesof(coords[1], boundary)),
            (bond_angle            , Base.typesof(coords[1], coords[2], coords[3], boundary)),
            (torsion_angle         , Base.typesof(coords[1], coords[2], coords[3], coords[4],
                                                  boundary)),
            (remove_CM_motion!     , Base.typesof(sys)),
            (random_velocities     , Base.typesof(sys, 100.0u"K")),
            (random_velocities!    , Base.typesof(sys, 100.0u"K")),
            (scale_coords!         , Base.typesof(sys, μ)),
            (rdf                   , Base.typesof(coords, boundary)),
        ]
        for (f, types) in analysis_calls
            JET.test_call(f, types; jet_config...)
            JET.test_opt( f, types; jet_config...)
        end
    end

    # Boundaries
    n_atoms = 50
    atoms = [Atom(mass=10.0u"g/mol", charge=0.1, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                for _ in 1:n_atoms]
    boundaries = [
        ("rectangular", RectangularBoundary(3.0u"nm"), 2),
        ("triclinic approx", TriclinicBoundary(SVector(3.0, 0.0, 0.0)u"nm",
                                                SVector(1.0, 3.0, 0.0)u"nm",
                                                SVector(1.0, 1.0, 3.0)u"nm"), 3),
        ("triclinic exact" , TriclinicBoundary(SVector(3.0, 0.0, 0.0)u"nm",
                                                SVector(1.0, 3.0, 0.0)u"nm",
                                                SVector(1.0, 1.0, 3.0)u"nm";
                                                approx_images=false), 3),
    ]
    for (b_name, boundary, dims) in boundaries
        coords = place_atoms(n_atoms, boundary; min_dist=0.2u"nm")
        velocities = [random_velocity(10.0u"g/mol", 100.0u"K"; dims=dims)
                        for _ in 1:n_atoms]
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
        )
        neighbors = find_neighbors(sys)
        JET.test_call(forces          , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(potential_energy, Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(virial          , Base.typesof(sys); jet_config...)
        JET.test_call(pressure        , Base.typesof(sys); jet_config...)
        JET.test_call(simulate!       , Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                        jet_config...)
        JET.test_call(displacements   , Base.typesof(coords, boundary); jet_config...)
        JET.test_call(volume          , Base.typesof(boundary); jet_config...)
        JET.test_call(box_center      , Base.typesof(boundary); jet_config...)
        JET.test_call(random_coord    , Base.typesof(boundary); jet_config...)
        JET.test_call(scale_boundary  , Base.typesof(boundary, 1.01); jet_config...)
    end

    # Neighbor finders
    n_atoms = 50
    boundary = CubicBoundary(3.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    atoms = [Atom(mass=10.0u"g/mol", charge=0.1, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                for _ in 1:n_atoms]
    velocities = [random_velocity(10.0u"g/mol", 100.0u"K") for _ in 1:n_atoms]
    neighbor_finders = [
        "NoNeighborFinder"          => NoNeighborFinder(),
        "DistanceNeighborFinder"    => DistanceNeighborFinder(
            eligible=trues(n_atoms, n_atoms), n_steps=10, dist_cutoff=1.2u"nm"),
        "TreeNeighborFinder"        => TreeNeighborFinder(
            eligible=trues(n_atoms, n_atoms), n_steps=10, dist_cutoff=1.2u"nm"),
        "CellListMapNeighborFinder" => CellListMapNeighborFinder(
            eligible=trues(n_atoms, n_atoms), n_steps=10, dist_cutoff=1.2u"nm",
            boundary=boundary),
    ]
    for (nf_name, neighbor_finder) in neighbor_finders
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm"),
                                use_neighbors=!(neighbor_finder isa NoNeighborFinder)),),
            neighbor_finder=neighbor_finder,
        )
        JET.test_call(find_neighbors, Base.typesof(sys); jet_config...)
        JET.test_call(forces        , Base.typesof(sys, find_neighbors(sys)); jet_config...)
        JET.test_call(simulate!     , Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                        jet_config...)
    end

    # Pairwise interactions
    n_atoms = 40
    boundary = CubicBoundary(3.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    velocities = [random_velocity(10.0u"g/mol", 100.0u"K") for _ in 1:n_atoms]
    cutoff = DistanceCutoff(1.0u"nm")
    neighbor_finder = DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=10,
                                                dist_cutoff=1.2u"nm")
    atoms = [Atom(mass=10.0u"g/mol", charge=(isodd(i) ? 0.2 : -0.2), σ=0.3u"nm",
                    ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    atoms_alch = [Atom(mass=10.0u"g/mol", charge=(isodd(i) ? 0.2 : -0.2), σ=0.3u"nm",
                        ϵ=0.2u"kJ * mol^-1", λ=0.5, alch_role=(isodd(i) ? 1 : 0))
                    for i in 1:n_atoms]

    # The interactions are grouped into one system each, rather than one system per
    #   interaction, since the analysis covers every element of the interaction tuple
    pairwise_inters_sets = [
        "pairwise" => (
            LennardJones(cutoff=cutoff, use_neighbors=true),
            AshbaughHatch(cutoff=cutoff, use_neighbors=true),
            SoftSphere(cutoff=cutoff, use_neighbors=true),
            Mie(m=6, n=12, cutoff=cutoff, use_neighbors=true),
            Buckingham(cutoff=cutoff, use_neighbors=true),
            DoubleExponential(α=16.766, β=4.427, cutoff=cutoff, use_neighbors=true),
            Coulomb(cutoff=cutoff, use_neighbors=true),
            CoulombReactionField(dist_cutoff=1.0u"nm", use_neighbors=true),
            Yukawa(cutoff=cutoff, use_neighbors=true, kappa=1.0u"nm^-1"),
            Gravity(use_neighbors=true),
        ),
        "cutoffs" => (
            LennardJones(cutoff=NoCutoff(), use_neighbors=true),
            LennardJones(cutoff=DistanceCutoff(1.0u"nm"), use_neighbors=true),
            LennardJones(cutoff=ShiftedPotentialCutoff(1.0u"nm"), use_neighbors=true),
            LennardJones(cutoff=ShiftedForceCutoff(1.0u"nm"), use_neighbors=true),
            LennardJones(cutoff=CubicSplineCutoff(0.6u"nm", 1.0u"nm"), use_neighbors=true),
            LennardJones(cutoff=PolynomialCutoff(0.6u"nm", 1.0u"nm"), use_neighbors=true),
        ),
    ]
    for (pis_name, pairwise_inters) in pairwise_inters_sets
        sys = System(atoms=atoms, coords=coords, boundary=boundary, velocities=velocities,
                        pairwise_inters=pairwise_inters, neighbor_finder=neighbor_finder)
        neighbors = find_neighbors(sys)
        JET.test_call(forces          , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(potential_energy, Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(virial          , Base.typesof(sys); jet_config...)
        JET.test_call(simulate!       , Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                        jet_config...)
    end

    # Soft core
    sys = System(
        atoms=atoms_alch,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(
            LennardJonesSoftCoreBeutler(cutoff=cutoff, use_neighbors=true, α=0.3),
            LennardJonesSoftCoreGapsys(cutoff=cutoff, use_neighbors=true, α=0.85),
            CoulombSoftCoreBeutler(cutoff=cutoff, use_neighbors=true, α=0.3),
            CoulombSoftCoreGapsys(cutoff=cutoff, use_neighbors=true, α=0.3,
                                    σQ=1.0u"nm"),
        ),
        neighbor_finder=neighbor_finder,
    )
    neighbors = find_neighbors(sys)
    JET.test_call(forces          , Base.typesof(sys, neighbors); jet_config...)
    JET.test_call(potential_energy, Base.typesof(sys, neighbors); jet_config...)
    JET.test_call(virial          , Base.typesof(sys); jet_config...)
    JET.test_call(simulate!       , Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                    jet_config...)

    # Specific interactions
    n_atoms_spec = 10
    coords_spec = place_atoms(n_atoms_spec, boundary; min_dist=0.4u"nm")
    atoms_spec = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                    for _ in 1:n_atoms_spec]
    velocities_spec = [random_velocity(10.0u"g/mol", 100.0u"K")
                        for _ in 1:n_atoms_spec]
    is, js, ks, ls = [1, 3], [2, 4], [3, 5], [4, 6]
    sys = System(
        atoms=atoms_spec,
        coords=coords_spec,
        boundary=boundary,
        velocities=velocities_spec,
        specific_inter_lists=(
            InteractionList1Atoms(is,
                [HarmonicPositionRestraint(k=100.0u"kJ * mol^-1 * nm^-2",
                                            x0=coords_spec[i]) for i in is]),
            InteractionList2Atoms(is, js,
                [HarmonicBond(k=100.0u"kJ * mol^-1 * nm^-2", r0=0.5u"nm") for _ in is]),
            InteractionList2Atoms(is, js,
                [MorseBond(D=100.0u"kJ * mol^-1", a=2.0u"nm^-1", r0=0.5u"nm")
                    for _ in is]),
            InteractionList2Atoms(is, js,
                [FENEBond(k=100.0u"kJ * mol^-1 * nm^-2", r0=1.4u"nm", σ=0.3u"nm",
                            ϵ=0.4u"kJ * mol^-1") for _ in is]),
            InteractionList3Atoms(is, js, ks,
                [HarmonicAngle(k=10.0u"kJ * mol^-1", θ0=2.0) for _ in is]),
            InteractionList3Atoms(is, js, ks,
                [CosineAngle(k=10.0u"kJ * mol^-1", θ0=2.0) for _ in is]),
            InteractionList3Atoms(is, js, ks,
                [UreyBradley(kangle=10.0u"kJ * mol^-1", θ0=2.0,
                                kbond=10.0u"kJ * mol^-1 * nm^-2", r0=1.0u"nm")
                    for _ in is]),
            InteractionList4Atoms(is, js, ks, ls,
                [PeriodicTorsion(periodicities=[1, 2, 3], phases=[1.0, 0.0, -1.0],
                                    ks=[10.0, 5.0, 8.0]u"kJ * mol^-1", n_terms=6)
                    for _ in is]),
            InteractionList4Atoms(is, js, ks, ls,
                [RBTorsion(c0=1.0u"kJ * mol^-1", c1=2.0u"kJ * mol^-1",
                            c2=3.0u"kJ * mol^-1", c3=4.0u"kJ * mol^-1",
                            c4=0.5u"kJ * mol^-1", c5=0.25u"kJ * mol^-1") for _ in is]),
            InteractionList4Atoms(is, js, ks, ls,
                [HarmonicTorsion(k=10.0u"kJ * mol^-1", θ0=1.0) for _ in is]),
        ),
    )
    neighbors = find_neighbors(sys)
    JET.test_call(forces          , Base.typesof(sys, neighbors); jet_config...)
    JET.test_call(potential_energy, Base.typesof(sys, neighbors); jet_config...)
    JET.test_call(virial          , Base.typesof(sys); jet_config...)
    JET.test_call(simulate!       , Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                    jet_config...)

    # Implicit solvent
    for solvent in (SetupImplicitSolventOBC(), SetupImplicitSolventGBN2())
        sys = jet_protein(; implicit_solvent=solvent, dist_cutoff=1.2f0u"nm")
        neighbors = find_neighbors(sys)
        JET.test_call(forces          , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(potential_energy, Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(simulate!       , Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                        jet_config...)
    end

    sys = System(
        atoms=[Atom(mass=1.0u"g/mol")],
        coords=[SVector(-0.5, 0.5)u"nm"],
        boundary=RectangularBoundary(Inf * u"nm"),
        velocities=[random_velocity(1.0u"g/mol", 100.0u"K"; dims=2)],
        general_inters=(MullerBrown(),),
    )
    neighbors = find_neighbors(sys)
    JET.test_call(forces          , Base.typesof(sys, neighbors); jet_config...)
    JET.test_call(potential_energy, Base.typesof(sys, neighbors); jet_config...)
    JET.test_call(simulate!       , Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                    jet_config...)

    # Simulators
    for (sys_name, sys) in jet_systems
        sys_name == "LJ without units" && continue
        dt, temp = 0.0005u"ps", 300.0u"K"
        friction, press = 1.0u"ps^-1", 1.0u"bar"
        simulators = [
            "VelocityVerlet"     => VelocityVerlet(dt=dt),
            "Verlet"             => Verlet(dt=dt),
            "StormerVerlet"      => StormerVerlet(dt=dt),
            "DPDVelocityVerlet"  => DPDVelocityVerlet(dt=dt),
            "Langevin"           => Langevin(dt=dt, temperature=temp, friction=friction),
            "LangevinSplitting"  => LangevinSplitting(dt=dt, temperature=temp,
                                        friction=10.0u"g * mol^-1 * ps^-1", splitting="BAOAB"),
            "OverdampedLangevin" => OverdampedLangevin(dt=dt, temperature=temp,
                                                       friction=friction),
            "NoseHoover"         => NoseHoover(dt=dt, temperature=temp),
            "MTSIntegrator"      => MTSIntegrator(dt=dt,
                                        pi_fractions=ntuple(i -> 1, length(sys.pairwise_inters)),
                                        si_fractions=ntuple(i -> 1,
                                                            length(sys.specific_inter_lists)),
                                        gi_fractions=ntuple(i -> 1, length(sys.general_inters)),
                                        remove_CM_motion=false),
            "MTSLangevinIntegrator" => MTSLangevinIntegrator(dt=dt, temperature=temp,
                                        friction=friction,
                                        pi_fractions=ntuple(i -> 1, length(sys.pairwise_inters)),
                                        si_fractions=ntuple(i -> 1,
                                                            length(sys.specific_inter_lists)),
                                        gi_fractions=ntuple(i -> 1, length(sys.general_inters)),
                                        remove_CM_motion=false),
            "MetropolisMonteCarlo" => MetropolisMonteCarlo(temperature=temp,
                                        trial_moves=random_uniform_translation!,
                                        trial_args=Dict(:shift_size => 0.1u"nm")),
            "AndersenThermostat" => Langevin(dt=dt, temperature=temp, friction=friction,
                                        coupling=AndersenThermostat(temp, 1.0u"ps")),
            "MonteCarloBarostat" => VelocityVerlet(dt=dt,
                                        coupling=MonteCarloBarostat(press, temp, sys.boundary)),
            "BerendsenBarostat"  => VelocityVerlet(dt=dt,
                                        coupling=BerendsenBarostat(press, 1.0u"ps")),
        ]
        for (sim_name, sim) in simulators
            JET.test_call(simulate!, Base.typesof(sys, sim, 1); jet_config...)
        end
        minimizer = SteepestDescentMinimizer(step_size=0.01u"nm", max_steps=2,
                                                tol=1000.0u"kJ * mol^-1 * nm^-1")
        JET.test_call(simulate!, Base.typesof(sys, minimizer); jet_config...)
    end

    # Couplers
    # The thermostats and barostats are analysed through apply_coupling! rather than through
    #   simulate!, so that the coupling types are covered without a simulator for each
    for (sys_name, sys) in jet_systems
        sys_name == "LJ without units" && continue
        temp, press = 300.0u"K", 1.0u"bar"
        press_2, press_6 = fill(press, 2), fill(press, 6)
        compress_2 = fill(4.6e-5u"bar^-1", 2)
        compress_6 = fill(4.6e-5u"bar^-1", 6)
        sim = VelocityVerlet(dt=0.0005u"ps")
        neighbors = find_neighbors(sys)
        buffers = Molly.init_buffers!(sys, 1)
        couplers = [
            "ImmediateThermostat"       => ImmediateThermostat(temp),
            "VelocityRescaleThermostat" => VelocityRescaleThermostat(temp, 0.1u"ps"),
            "AndersenThermostat"        => AndersenThermostat(temp, 0.1u"ps"),
            "BerendsenThermostat"       => BerendsenThermostat(temp, 0.1u"ps"),
            "BerendsenBarostat"         => BerendsenBarostat(press, 0.1u"fs";
                                             max_scale_frac=0.01),
            "BerendsenBarostat semiisotropic" => BerendsenBarostat(press_2, 0.1u"fs";
                                             compressibility=compress_2,
                                             coupling_type=:semiisotropic, max_scale_frac=0.01),
            "BerendsenBarostat anisotropic"   => BerendsenBarostat(press_6, 0.1u"fs";
                                             compressibility=compress_6,
                                             coupling_type=:anisotropic, max_scale_frac=0.01),
            "CRescaleBarostat"          => CRescaleBarostat(press, 0.1u"fs";
                                             max_scale_frac=0.01),
            "CRescaleBarostat semiisotropic"  => CRescaleBarostat(press_2, 0.1u"fs";
                                             compressibility=compress_2,
                                             coupling_type=:semiisotropic, max_scale_frac=0.01),
            "CRescaleBarostat anisotropic"    => CRescaleBarostat(press_6, 0.1u"fs";
                                             compressibility=compress_6,
                                             coupling_type=:anisotropic, max_scale_frac=0.01),
            "MonteCarloBarostat"        => MonteCarloBarostat(press, temp, sys.boundary),
            "MonteCarloBarostat semiisotropic" => MonteCarloBarostat(press_2, temp,
                                             sys.boundary; coupling_type=:semiisotropic),
            "MonteCarloBarostat anisotropic"   => MonteCarloBarostat(press_6, temp,
                                             sys.boundary; coupling_type=:anisotropic),
        ]
        for (coupler_name, coupler) in couplers
            types = Base.typesof(sys, buffers, coupler, sim, neighbors, 1)
            JET.test_call(apply_coupling!, types; jet_config...)
            # The barostats call pressure and potential_energy, which go through the
            #   function barriers, so only the thermostats can be gated on runtime dispatch
            if coupler isa Molly.AbstractThermostat
                JET.test_opt(apply_coupling!, types; jet_config...)
            end
        end
    end

    # Loggers
    n_atoms = 50
    boundary = CubicBoundary(3.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    atoms = [Atom(mass=10.0u"g/mol", charge=0.1, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
                for _ in 1:n_atoms]
    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=[random_velocity(10.0u"g/mol", 100.0u"K") for _ in 1:n_atoms],
        pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm"), use_neighbors=true),),
        neighbor_finder=DistanceNeighborFinder(eligible=trues(n_atoms, n_atoms), n_steps=10,
                                                dist_cutoff=1.2u"nm"),
        loggers=(
            temp=TemperatureLogger(10),
            coords=CoordinatesLogger(10),
            velocities=VelocitiesLogger(10),
            box=BoxLogger(10),
            total_energy=TotalEnergyLogger(10),
            kinetic_energy=KineticEnergyLogger(10),
            potential_energy=PotentialEnergyLogger(10),
            forces=ForcesLogger(10),
            volume=VolumeLogger(10),
            density=DensityLogger(10),
            virial=VirialLogger(10),
            scalar_virial=ScalarVirialLogger(10),
            pressure=PressureLogger(10),
            scalar_pressure=ScalarPressureLogger(10),
            displacements=DisplacementsLogger(10, coords),
            average=AverageObservableLogger(Molly.potential_energy_wrapper,
                                            typeof(atoms[1].ϵ), 10),
            autocorrelation=AutoCorrelationLogger(Molly.potential_energy_wrapper,
                                                    typeof(atoms[1].ϵ), 10, 5),
        ),
    )
    neighbors = find_neighbors(sys)
    buffers = Molly.init_buffers!(sys, 1)
    JET.test_call(apply_loggers!, Base.typesof(sys, neighbors, 1, buffers, true);
                    jet_config...)
    JET.test_call(simulate!, Base.typesof(sys, VelocityVerlet(dt=0.0002u"ps"), 1);
                    jet_config...)
    for (logger_name, logger) in pairs(sys.loggers)
        JET.test_call(log_property!, Base.typesof(logger, sys, neighbors, 1, buffers);
                        jet_config...)
    end

    # Constraints
    for (algo_name, algo) in ("SHAKE" => SetupSHAKE_RATTLE(), "LINCS" => SetupLINCS())
        sys = jet_protein(; constraints=:hbonds, rigid_water=true, constraint_algorithm=algo)
        coord_storage = copy(sys.coords)
        vel_storage = copy(sys.velocities)
        JET.test_call(apply_position_constraints!, Base.typesof(sys, coord_storage);
                      jet_config...)
        JET.test_call(apply_position_constraints!,
                      Base.typesof(sys, coord_storage, vel_storage, 0.0005f0u"ps");
                      jet_config...)
        JET.test_call(apply_velocity_constraints!, Base.typesof(sys); jet_config...)
        JET.test_call(check_position_constraints , Base.typesof(sys); jet_config...)
        JET.test_call(check_velocity_constraints , Base.typesof(sys); jet_config...)
        JET.test_call(check_constraints          , Base.typesof(sys); jet_config...)
        dt, temp = 0.0005u"ps", 300.0u"K"
        simulators = [
            "VelocityVerlet"    => VelocityVerlet(dt=dt),
            "Verlet"            => Verlet(dt=dt),
            "Langevin"          => Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1"),
            "LangevinSplitting" => LangevinSplitting(dt=dt, temperature=temp,
                                        friction=10.0u"g * mol^-1 * ps^-1", splitting="BAOAB"),
        ]
        for (sim_name, sim) in simulators
            JET.test_call(simulate!, Base.typesof(sys, sim, 1); jet_config...)
        end
    end
end
