# Static analysis of the main code paths with JET

# Optimization analysis (@test_opt) is not currently clean because forces! etc.
#   deliberately use function barriers: Val(n_threads) and Val(needs_vir) are runtime
#   values, so the call into the pairwise and specific loops is a dynamic dispatch.

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
        "protein PME"      => jet_protein(; nonbonded_method=:pme),
        "protein hbonds"   => jet_protein(; constraints=:hbonds, rigid_water=true),
        "TIP4P"            => System(joinpath(data_dir, "tip4pew.pdb"),
                                     MolecularForceField(joinpath(ff_dir, "tip4pfb.xml"))),
    ]

    for (sys_name, sys) in jet_systems
        neighbors = find_neighbors(sys)
        JET.test_call(find_neighbors  , Base.typesof(sys); jet_config...)
        JET.test_call(forces          , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(forces_virial   , Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(potential_energy, Base.typesof(sys, neighbors); jet_config...)
        JET.test_call(kinetic_energy  , Base.typesof(sys); jet_config...)
        JET.test_call(total_energy    , Base.typesof(sys); jet_config...)
        JET.test_call(temperature     , Base.typesof(sys); jet_config...)
        JET.test_call(virial          , Base.typesof(sys); jet_config...)
        JET.test_call(pressure        , Base.typesof(sys); jet_config...)
        JET.test_call(masses          , Base.typesof(sys); jet_config...)
    end

    for (sys_name, sys) in jet_systems
        sys_name == "LJ without units" && continue
        dt, temp = 0.0005u"ps", 300.0u"K"
        friction, press = 1.0u"ps^-1", 1.0u"bar"
        simulators = [
            "VelocityVerlet"     => VelocityVerlet(dt=dt),
            "Verlet"             => Verlet(dt=dt),
            "StormerVerlet"      => StormerVerlet(dt=dt),
            "Langevin"           => Langevin(dt=dt, temperature=temp, friction=friction),
            "LangevinSplitting"  => LangevinSplitting(dt=dt, temperature=temp,
                                        friction=10.0u"g * mol^-1 * ps^-1", splitting="BAOAB"),
            "OverdampedLangevin" => OverdampedLangevin(dt=dt, temperature=temp,
                                                       friction=friction),
            "NoseHoover"         => NoseHoover(dt=dt, temperature=temp),
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
end
