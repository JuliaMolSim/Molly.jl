@testset "Immediate thermostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                temperature=TemperatureLogger(10),
            ),
        )

        coupling = (ImmediateThermostat(temp),)
        simulator = VelocityVerlet(
            dt=0.001u"ps",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        temps_traj = values(sys.loggers.temperature)[2001:end]
        @test 9.5u"K" < mean(temps_traj) < 10.5u"K"
        @test std(temps_traj) < 1.0u"K"
    end
end

@testset "Velocity rescale thermostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                temperature=TemperatureLogger(10),
            ),
        )

        coupling = (VelocityRescaleThermostat(temp, 0.1u"ps"),)
        simulator = VelocityVerlet(
            dt=0.001u"ps",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        temps_traj = values(sys.loggers.temperature)[2001:end]
        @test 9.5u"K" < mean(temps_traj) < 10.5u"K"
        @test std(temps_traj) < 1.0u"K"
    end
end

@testset "Andersen thermostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                temperature=TemperatureLogger(10),
            ),
        )

        coupling = (AndersenThermostat(temp, 0.1u"ps"),)
        simulator = VelocityVerlet(
            dt=0.001u"ps",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        temps_traj = values(sys.loggers.temperature)[2001:end]
        @test 9.5u"K" < mean(temps_traj) < 10.5u"K"
        @test std(temps_traj) < 1.0u"K"
    end
end

@testset "Berendsen thermostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                temperature=TemperatureLogger(10),
            ),
        )

        coupling = (BerendsenThermostat(temp, 0.1u"ps"),)
        simulator = VelocityVerlet(
            dt=0.001u"ps",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        temps_traj = values(sys.loggers.temperature)[2001:end]
        @test 9.5u"K" < mean(temps_traj) < 10.5u"K"
        @test std(temps_traj) < 1.0u"K"
    end
end

@testset "Berendsen isotropic barostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    box_size_wrapper(sys, args...; kwargs...) = sys.boundary.side_lengths[1]
    BoxSizeLogger(n_steps) = GeneralObservableLogger(box_size_wrapper, typeof(1.0u"nm"), n_steps)

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                pressure=PressureLogger(10),
                scalar_pressure=ScalarPressureLogger(10),
                volume=VolumeLogger(10),
            ),
        )

        coupling = (BerendsenBarostat(1.0u"bar", 0.1u"fs"; max_scale_frac=0.01),)
        simulator = Langevin(
            dt=0.001u"ps",
            temperature=temp,
            friction=1.0u"ps^-1",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        P_iso = [tr(P) / 3 for P in values(sys.loggers.pressure)[2001:end]]

        @test 0.75u"bar" < mean(P_iso) < 1.25u"bar" # Corrected for tensorial pressure
        @test std(P_iso) < 0.5u"bar"

        # (5nm)^3 to (5.5nm)^3
        @test 120.0u"nm^3" < mean(values(sys.loggers.volume)[2001:end]) < 165.0u"nm^3"
        @test std(values(sys.loggers.volume)[2001:end]) < 25.0u"nm^3"
    end
end

@testset "Berendsen semiisotropic barostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                pressure=PressureLogger(10),
                scalar_pressure=ScalarPressureLogger(10),
                volume=VolumeLogger(10),
            ),
        )

        P = [1.0u"bar", 1.0u"bar"]
        K = [4.6e-5u"bar^-1", 4.6e-5u"bar^-1"]

        coupling = (BerendsenBarostat(P, 0.1u"fs"; compressibility=K, max_scale_frac=0.01,
                                      n_steps=1, coupling_type=:semiisotropic),)
        simulator = Langevin(
            dt=0.001u"ps",
            temperature=temp,
            friction=1.0u"ps^-1",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        P_xy = [(P[1,1] + P[2,2]) / 2 for P in values(sys.loggers.pressure)[2001:end]]
        P_z  = [P[3,3] for P in values(sys.loggers.pressure)[2001:end]]

        @test 0.75u"bar" < mean(P_xy) < 1.25u"bar" # Corrected for tensorial pressure
        @test 0.75u"bar" < mean(P_z)  < 1.25u"bar" # Corrected for tensorial pressure
        
        @test std(P_xy) < 0.5u"bar"
        @test std(P_z)  < 0.5u"bar"

        # (5nm)^3 to (5.5nm)^3
        @test 120.0u"nm^3" < mean(values(sys.loggers.volume)[2001:end]) < 165.0u"nm^3"
        @test std(values(sys.loggers.volume)[2001:end]) < 25.0u"nm^3"
    end
end

@testset "Berendsen anisotropic barostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    Ls   = [4.0u"nm", 4.0u"nm", 4.0u"nm"]
    As   = [pi/2u"rad", pi/2u"rad", pi/2u"rad"]
    boundary = TriclinicBoundary(Ls, As)

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                pressure=PressureLogger(10),
                scalar_pressure=ScalarPressureLogger(10),
                volume=VolumeLogger(10),
            ),
        )

        P = [1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar"]
        K = [4.6e-5u"bar^-1", 4.6e-5u"bar^-1", 4.6e-5u"bar^-1",
             4.6e-5u"bar^-1", 4.6e-5u"bar^-1", 4.6e-5u"bar^-1"]

        coupling = (BerendsenBarostat(P, 0.1u"fs"; compressibility=K, max_scale_frac=0.01,
                                      n_steps=1, coupling_type=:anisotropic),)
        simulator = Langevin(
            dt=0.001u"ps",
            temperature=temp,
            friction=1.0u"ps^-1",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        P_x = [P[1,1] for P in values(sys.loggers.pressure)[2001:end]]
        P_y = [P[2,2] for P in values(sys.loggers.pressure)[2001:end]]
        P_z = [P[3,3] for P in values(sys.loggers.pressure)[2001:end]]

        @test 0.75u"bar" < mean(P_x) < 1.25u"bar" # Corrected for tensorial pressure
        @test 0.75u"bar" < mean(P_y) < 1.25u"bar" # Corrected for tensorial pressure
        @test 0.75u"bar" < mean(P_z) < 1.25u"bar" # Corrected for tensorial pressure

        @test std(P_x) < 0.5u"bar"
        @test std(P_y) < 0.5u"bar"
        @test std(P_z) < 0.5u"bar"

        # (5nm)^3 to (5.5nm)^3
        @test 120.0u"nm^3" < mean(values(sys.loggers.volume)[2001:end]) < 165.0u"nm^3"
        @test std(values(sys.loggers.volume)[2001:end]) < 25.0u"nm^3"
    end
end

@testset "C-Rescale isotropic barostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    box_size_wrapper(sys, args...; kwargs...) = sys.boundary.side_lengths[1]
    BoxSizeLogger(n_steps) = GeneralObservableLogger(box_size_wrapper, typeof(1.0u"nm"), n_steps)

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                pressure=PressureLogger(10),
                scalar_pressure=ScalarPressureLogger(10),
                volume=VolumeLogger(10),
            ),
        )

        coupling = (CRescaleBarostat(1.0u"bar", 0.1u"fs"; max_scale_frac=0.01, n_steps=1),)
        simulator = Langevin(
            dt=0.001u"ps",
            temperature=temp,
            friction=1.0u"ps^-1",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        P_iso = [tr(P) / 3 for P in values(sys.loggers.pressure)[2001:end]]

        @test 0.75u"bar" < mean(P_iso) < 1.25u"bar" # Corrected for tensorial pressure
        @test std(P_iso) < 0.5u"bar"

        # (5nm)^3 to (5.5nm)^3
        @test 120.0u"nm^3" < mean(values(sys.loggers.volume)[2001:end]) < 165.0u"nm^3"
        @test std(values(sys.loggers.volume)[2001:end]) < 25.0u"nm^3"
    end
end

@testset "C-Rescale semiisotropic barostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    boundary = CubicBoundary(4.0u"nm")

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                pressure=PressureLogger(10),
                scalar_pressure=ScalarPressureLogger(10),
                volume=VolumeLogger(10),
            ),
        )

        P = [1.0u"bar", 1.0u"bar"]
        K = [4.6e-5u"bar^-1", 4.6e-5u"bar^-1"]

        coupling = (CRescaleBarostat(P, 0.1u"fs"; compressibility=K, max_scale_frac=0.01,
                                     n_steps=1, coupling_type=:semiisotropic),)
        simulator = Langevin(
            dt=0.001u"ps",
            temperature=temp,
            friction=1.0u"ps^-1",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        P_xy = [(P[1,1] + P[2,2]) / 2 for P in values(sys.loggers.pressure)[2001:end]]
        P_z  = [P[3,3] for P in values(sys.loggers.pressure)[2001:end]]

        @test 0.75u"bar" < mean(P_xy) < 1.25u"bar" # Corrected for tensorial pressure
        @test 0.75u"bar" < mean(P_z)  < 1.25u"bar" # Corrected for tensorial pressure

        @test std(P_xy) < 0.5u"bar"
        @test std(P_z) < 0.5u"bar"

        # (5nm)^3 to (5.5nm)^3
        @test 120.0u"nm^3" < mean(values(sys.loggers.volume)[2001:end]) < 165.0u"nm^3"
        @test std(values(sys.loggers.volume)[2001:end]) < 25.0u"nm^3"
    end
end

@testset "C-Rescale anisotropic barostat" begin
    n_atoms = 100
    n_steps = 40_000
    temp = 10.0u"K"
    Ls   = [4.0u"nm", 4.0u"nm", 4.0u"nm"]
    As   = [pi/2u"rad", pi/2u"rad", pi/2u"rad"]
    boundary = TriclinicBoundary(Ls, As)

    for AT in array_list
        sys = System(
            atoms=to_device([Atom(mass=10.0u"g/mol", σ=0.04u"nm", ϵ=0.1u"kJ * mol^-1")
                             for _ in 1:n_atoms], AT),
            coords=to_device(place_atoms(n_atoms, boundary), AT),
            boundary=boundary,
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(1.0u"nm")),),
            loggers=(
                pressure=PressureLogger(10),
                scalar_pressure=ScalarPressureLogger(10),
                volume=VolumeLogger(10),
            ),
        )

        P = [1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar"]
        K = [4.6e-5u"bar^-1", 4.6e-5u"bar^-1", 4.6e-5u"bar^-1",
             4.6e-5u"bar^-1", 4.6e-5u"bar^-1", 4.6e-5u"bar^-1"]

        coupling = (CRescaleBarostat(P, 0.1u"fs"; compressibility=K, max_scale_frac=0.01,
                                     n_steps=1, coupling_type=:anisotropic),)
        simulator = Langevin(
            dt=0.001u"ps",
            temperature=temp,
            friction=1.0u"ps^-1",
            coupling=coupling,
        )

        random_velocities!(sys, temp)
        simulate!(sys, simulator, n_steps)

        P_x = [P[1,1] for P in values(sys.loggers.pressure)[2001:end]]
        P_y = [P[2,2] for P in values(sys.loggers.pressure)[2001:end]]
        P_z = [P[3,3] for P in values(sys.loggers.pressure)[2001:end]]

        @test 0.75u"bar" < mean(P_x) < 1.25u"bar" # Corrected for tensorial pressure
        @test 0.75u"bar" < mean(P_y) < 1.25u"bar" # Corrected for tensorial pressure
        @test 0.75u"bar" < mean(P_z) < 1.25u"bar" # Corrected for tensorial pressure

        @test std(P_x) < 0.5u"bar"
        @test std(P_y) < 0.5u"bar"
        @test std(P_z) < 0.5u"bar"

        # (5nm)^3 to (5.5nm)^3
        @test 120.0u"nm^3" < mean(values(sys.loggers.volume)[2001:end]) < 165.0u"nm^3"
        @test std(values(sys.loggers.volume)[2001:end]) < 25.0u"nm^3"
    end
end

@testset "Monte Carlo isotropic barostat" begin
    # See http://www.sklogwiki.org/SklogWiki/index.php/Argon for parameters
    rng = Xoshiro(10)
    n_atoms = 25
    n_steps = 100_000
    atom_mass = 39.947u"g/mol"
    boundary = CubicBoundary(8.0u"nm")
    temp = 288.15u"K"
    press = 1.0u"bar"
    dt = 0.0005u"ps"
    friction = 1.0u"ps^-1"
    lang = Langevin(dt=dt, temperature=temp, friction=friction)
    atoms = fill(Atom(mass=atom_mass, σ=0.3345u"nm", ϵ=1.0451u"kJ * mol^-1"), n_atoms)
    coords = place_atoms(n_atoms, boundary; min_dist=1.0u"nm", rng=rng)
    n_log_steps = 500

    sys = System(
        atoms=atoms,
        coords=copy(coords),
        boundary=boundary,
        pairwise_inters=(LennardJones(),),
        loggers=(
            temperature=TemperatureLogger(n_log_steps),
            total_energy=TotalEnergyLogger(n_log_steps),
            kinetic_energy=KineticEnergyLogger(n_log_steps),
            potential_energy=PotentialEnergyLogger(n_log_steps),
            virial=VirialLogger(n_log_steps),
            scalar_virial=ScalarVirialLogger(n_log_steps),
            pressure=PressureLogger(n_log_steps),
            scalar_pressure=ScalarPressureLogger(n_log_steps),
            volume=VolumeLogger(n_log_steps),
        ),
    )

    simulate!(deepcopy(sys), lang, 1_000; n_threads=1, rng=rng)
    @time simulate!(sys, lang, n_steps; n_threads=1, rng=rng)

    P_iso = [tr(P) / 3 for P in values(sys.loggers.pressure)]
    Vir   = [tr(V) for V in values(sys.loggers.virial)]

    @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
    @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
    @test -5.0u"kJ * mol^-1" < mean(Vir) < 5.0u"kJ * mol^-1"
    @test 1.75u"bar" < mean(P_iso) < 2.25u"bar"
    @test 0.1u"bar" < std(P_iso) < 0.5u"bar"
    @test all(values(sys.loggers.volume) .== 512.0u"nm^3")
    @test sys.boundary == CubicBoundary(8.0u"nm")

    barostat = MonteCarloBarostat(press, temp, boundary; coupling_type = :isotropic)
    lang_baro = Langevin(dt=dt, temperature=temp, friction=friction, coupling=(barostat,))
    vvand_baro = VelocityVerlet(dt=dt, coupling=(AndersenThermostat(temp, 1.0u"ps"), barostat))

    for sim in (lang_baro, vvand_baro)
        for AT in array_list
            if AT <: AbstractGPUArray && sim == vvand_baro
                continue
            end

            sys = System(
                atoms=to_device(atoms, AT),
                coords=to_device(copy(coords), AT),
                boundary=boundary,
                pairwise_inters=(LennardJones(),),
                loggers=(
                    temperature=TemperatureLogger(n_log_steps),
                    total_energy=TotalEnergyLogger(n_log_steps),
                    kinetic_energy=KineticEnergyLogger(n_log_steps),
                    potential_energy=PotentialEnergyLogger(n_log_steps),
                    virial=VirialLogger(n_log_steps),
                    scalar_virial=ScalarVirialLogger(n_log_steps),
                    pressure=PressureLogger(n_log_steps),
                    scalar_pressure=ScalarPressureLogger(n_log_steps),
                    volume=VolumeLogger(n_log_steps),
                ),
            )

            simulate!(deepcopy(sys), sim, 1_000; n_threads=1, rng=rng)
            @time simulate!(sys, sim, n_steps; n_threads=1, rng=rng)

            P_iso = [tr(P) / 3 for P in values(sys.loggers.pressure)]
            Vir   = [tr(V) for V in values(sys.loggers.virial)]

            @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
            @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
            @test -5.0u"kJ * mol^-1" < mean(Vir) < 5.0u"kJ * mol^-1"
            @test 0.75u"bar" < mean(P_iso) < 1.25u"bar"
            @test 0.1u"bar" < std(P_iso) < 0.5u"bar"
            @test 857.0u"nm^3" < mean(values(sys.loggers.volume)) < 1157.0u"nm^3"
            @test std(values(sys.loggers.volume)) < 300u"nm^3"
            @test sys.boundary != CubicBoundary(8.0u"nm")
        end
    end
end

@testset "Monte Carlo semiisotropic barostat" begin
    # See http://www.sklogwiki.org/SklogWiki/index.php/Argon for parameters
    rng = Xoshiro(10)
    n_atoms = 25
    n_steps = 100_000
    atom_mass = 39.947u"g/mol"
    boundary = CubicBoundary(8.0u"nm")
    temp = 288.15u"K"
    press = 1.0u"bar"
    dt = 0.0005u"ps"
    friction = 1.0u"ps^-1"
    lang = Langevin(dt=dt, temperature=temp, friction=friction)
    atoms = fill(Atom(mass=atom_mass, σ=0.3345u"nm", ϵ=1.0451u"kJ * mol^-1"), n_atoms)
    coords = place_atoms(n_atoms, boundary; min_dist=1.0u"nm", rng=rng)
    n_log_steps = 500

    sys = System(
        atoms=atoms,
        coords=copy(coords),
        boundary=boundary,
        pairwise_inters=(LennardJones(),),
        loggers=(
            temperature=TemperatureLogger(n_log_steps),
            total_energy=TotalEnergyLogger(n_log_steps),
            kinetic_energy=KineticEnergyLogger(n_log_steps),
            potential_energy=PotentialEnergyLogger(n_log_steps),
            virial=VirialLogger(n_log_steps),
            scalar_virial=ScalarVirialLogger(n_log_steps),
            pressure=PressureLogger(n_log_steps),
            scalar_pressure=ScalarPressureLogger(n_log_steps),
            volume=VolumeLogger(n_log_steps),
        ),
    )

    simulate!(deepcopy(sys), lang, 1_000; n_threads=1, rng=rng)
    @time simulate!(sys, lang, n_steps; n_threads=1, rng=rng)

    P_iso = [tr(P) / 3 for P in values(sys.loggers.pressure)]
    Vir   = [tr(V) for V in values(sys.loggers.virial)]

    @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
    @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
    @test -5.0u"kJ * mol^-1" < mean(Vir) < 5.0u"kJ * mol^-1"
    @test 1.7u"bar" < mean(P_iso) < 2.2u"bar"
    @test 0.1u"bar" < std(P_iso) < 0.5u"bar"
    @test all(values(sys.loggers.volume) .== 512.0u"nm^3")
    @test sys.boundary == CubicBoundary(8.0u"nm")

    P = [1.0u"bar", 1.0u"bar"]

    barostat = MonteCarloBarostat(P, temp, boundary; coupling_type = :semiisotropic)
    lang_baro = Langevin(dt=dt, temperature=temp, friction=friction, coupling=(barostat,))
    vvand_baro = VelocityVerlet(dt=dt, coupling=(AndersenThermostat(temp, 1.0u"ps"), barostat))

    for sim in (lang_baro, vvand_baro)
        for AT in array_list
            if AT <: AbstractGPUArray && sim == vvand_baro
                continue
            end

            sys = System(
                atoms=to_device(atoms, AT),
                coords=to_device(copy(coords), AT),
                boundary=boundary,
                pairwise_inters=(LennardJones(),),
                loggers=(
                    temperature=TemperatureLogger(n_log_steps),
                    total_energy=TotalEnergyLogger(n_log_steps),
                    kinetic_energy=KineticEnergyLogger(n_log_steps),
                    potential_energy=PotentialEnergyLogger(n_log_steps),
                    virial=VirialLogger(n_log_steps),
                    scalar_virial=ScalarVirialLogger(n_log_steps),
                    pressure=PressureLogger(n_log_steps),
                    scalar_pressure=ScalarPressureLogger(n_log_steps),
                    volume=VolumeLogger(n_log_steps),
                ),
            )

            simulate!(deepcopy(sys), sim, 1_000; n_threads=1, rng=rng)
            @time simulate!(sys, sim, n_steps; n_threads=1, rng=rng)

            P_xy = [(P[1,1] + P[2,2]) / 2 for P in values(sys.loggers.pressure)]
            P_z  = [P[3,3] for P in values(sys.loggers.pressure)]

            Vir   = [tr(V) for V in values(sys.loggers.virial)]

            @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
            @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
            @test -5.0u"kJ * mol^-1" < mean(Vir) < 5.0u"kJ * mol^-1"
            @test 0.75u"bar" < mean(P_xy) < 1.25u"bar"
            @test 0.1u"bar" < std(P_xy) < 0.5u"bar"
            @test 0.75u"bar" < mean(P_z) < 1.25u"bar"
            @test 0.1u"bar" < std(P_z) < 0.5u"bar"
            @test 857.0u"nm^3" < mean(values(sys.loggers.volume)) < 1157.0u"nm^3"
            @test std(values(sys.loggers.volume)) < 300u"nm^3"
            @test sys.boundary != CubicBoundary(8.0u"nm")
        end
    end
end

@testset "Monte Carlo anisotropic barostat" begin
    # See http://www.sklogwiki.org/SklogWiki/index.php/Argon for parameters
    rng = Xoshiro(10)
    n_atoms = 25
    n_steps = 100_000
    atom_mass = 39.947u"g/mol"
    boundary = CubicBoundary(8.0u"nm")
    temp = 288.15u"K"
    press = 1.0u"bar"
    dt = 0.0005u"ps"
    friction = 1.0u"ps^-1"
    lang = Langevin(dt=dt, temperature=temp, friction=friction)
    atoms = fill(Atom(mass=atom_mass, σ=0.3345u"nm", ϵ=1.0451u"kJ * mol^-1"), n_atoms)
    coords = place_atoms(n_atoms, boundary; min_dist=1.0u"nm", rng=rng)
    n_log_steps = 500

    sys = System(
        atoms=atoms,
        coords=copy(coords),
        boundary=boundary,
        pairwise_inters=(LennardJones(),),
        loggers=(
            temperature=TemperatureLogger(n_log_steps),
            total_energy=TotalEnergyLogger(n_log_steps),
            kinetic_energy=KineticEnergyLogger(n_log_steps),
            potential_energy=PotentialEnergyLogger(n_log_steps),
            virial=VirialLogger(n_log_steps),
            scalar_virial=ScalarVirialLogger(n_log_steps),
            pressure=PressureLogger(n_log_steps),
            scalar_pressure=ScalarPressureLogger(n_log_steps),
            volume=VolumeLogger(n_log_steps),
        ),
    )

    simulate!(deepcopy(sys), lang, 1_000; n_threads=1, rng=rng)
    @time simulate!(sys, lang, n_steps; n_threads=1, rng=rng)

    P_iso = [tr(P) / 3 for P in values(sys.loggers.pressure)]
    Vir   = [tr(V) for V in values(sys.loggers.virial)]

    @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
    @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
    @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
    @test -5.0u"kJ * mol^-1" < mean(Vir) < 5.0u"kJ * mol^-1"
    @test 1.7u"bar" < mean(P_iso) < 2.2u"bar"
    @test 0.1u"bar" < std(P_iso) < 0.5u"bar"
    @test all(values(sys.loggers.volume) .== 512.0u"nm^3")
    @test sys.boundary == CubicBoundary(8.0u"nm")

    P = [1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar", 1.0u"bar"]

    barostat = MonteCarloBarostat(P, temp, boundary; coupling_type = :anisotropic)
    lang_baro = Langevin(dt=dt, temperature=temp, friction=friction, coupling=(barostat,))
    vvand_baro = VelocityVerlet(dt=dt, coupling=(AndersenThermostat(temp, 1.0u"ps"), barostat))

    for sim in (lang_baro, vvand_baro)
        for AT in array_list
            if AT <: AbstractGPUArray && sim == vvand_baro
                continue
            end

            sys = System(
                atoms=to_device(atoms, AT),
                coords=to_device(copy(coords), AT),
                boundary=boundary,
                pairwise_inters=(LennardJones(),),
                loggers=(
                    temperature=TemperatureLogger(n_log_steps),
                    total_energy=TotalEnergyLogger(n_log_steps),
                    kinetic_energy=KineticEnergyLogger(n_log_steps),
                    potential_energy=PotentialEnergyLogger(n_log_steps),
                    virial=VirialLogger(n_log_steps),
                    scalar_virial=ScalarVirialLogger(n_log_steps),
                    pressure=PressureLogger(n_log_steps),
                    scalar_pressure=ScalarPressureLogger(n_log_steps),
                    volume=VolumeLogger(n_log_steps),
                ),
            )

            simulate!(deepcopy(sys), sim, 1_000; n_threads=1, rng=rng)
            @time simulate!(sys, sim, n_steps; n_threads=1, rng=rng)

            P_x = [P[1,1] for P in values(sys.loggers.pressure)]
            P_y = [P[2,2] for P in values(sys.loggers.pressure)]
            P_z = [P[3,3] for P in values(sys.loggers.pressure)]

            Vir   = [tr(V) for V in values(sys.loggers.virial)]

            @test 260.0u"K" < mean(values(sys.loggers.temperature)) < 300.0u"K"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.total_energy  )) < 120.0u"kJ * mol^-1"
            @test 50.0u"kJ * mol^-1" < mean(values(sys.loggers.kinetic_energy)) < 120.0u"kJ * mol^-1"
            @test mean(values(sys.loggers.potential_energy)) < 0.0u"kJ * mol^-1"
            @test -5.0u"kJ * mol^-1" < mean(Vir) < 5.0u"kJ * mol^-1"
            @test 0.75u"bar" < mean(P_x) < 1.25u"bar"
            @test 0.1u"bar" < std(P_x) < 0.5u"bar"
            @test 0.75u"bar" < mean(P_y) < 1.25u"bar"
            @test 0.1u"bar" < std(P_y) < 0.5u"bar"
            @test 0.75u"bar" < mean(P_z) < 1.25u"bar"
            @test 0.1u"bar" < std(P_z) < 0.5u"bar"
            @test 857.0u"nm^3" < mean(values(sys.loggers.volume)) < 1157.0u"nm^3"
            @test std(values(sys.loggers.volume)) < 300u"nm^3"
            @test sys.boundary != CubicBoundary(8.0u"nm")
        end
    end
end
