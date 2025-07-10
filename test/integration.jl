@testset "ACEmd" begin
    # See https://acesuit.github.io/ACEmd.jl/stable/molly
    fname_ace = joinpath(pkgdir(ACEmd), "data", "TiAl.json")
    fname_xyz = joinpath(pkgdir(ACEmd), "data", "TiAl-big.xyz")

    data = ExtXYZ.Atoms(read_frame(fname_xyz))
    pot = ACEmd.load_ace_model(fname_ace)

    sys = System(data, pot)
    temp = 298.0u"K"
    vel = random_velocities(sys, temp)

    sys = Molly.System(
        sys;
        velocities=vel,
        loggers=(temp=TemperatureLogger(10),)
    )

    simulator = VelocityVerlet(
        dt=1.0u"fs",
        coupling=AndersenThermostat(temp, 1.0u"ps"),
    )

    simulate!(sys, simulator, 100)

    @test sys.energy_units == u"eV"
    @test sys.force_units == u"eV / Å"
    @test eltype(eltype(sys.coords)) == typeof(1.0u"Å")
    @test potential_energy(sys) < 0.0u"eV"
end
