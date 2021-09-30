# Benchmark suite for Molly
# Run with something like:
#   using Molly, PkgBenchmark
#   results = benchmarkpkg(Molly, BenchmarkConfig(env=Dict("JULIA_NUM_THREADS" => 16)))
#   export_markdown(out_file, results)

using Molly
using BenchmarkTools
using CUDA

using Base.Threads

if nthreads() > 1
    @info "The parallel benchmarks will be run as Julia is running on $(nthreads()) threads"
else
    @warn "The parallel benchmarks will not be run as Julia is running on 1 thread"
end

if CUDA.functional()
    @info "The GPU benchmarks will be run as a CUDA-enabled device is available"
else
    @warn "The GPU benchmarks will not be run as a CUDA-enabled device is not available"
end

CUDA.allowscalar(false) # Check that we never do scalar indexing on the GPU

const SUITE = BenchmarkGroup(
    [],
    "interactions" => BenchmarkGroup(),
    "spatial"      => BenchmarkGroup(),
    "simulation"   => BenchmarkGroup(),
)

c1 = SVector(1.0, 1.0, 1.0)u"nm"
c2 = SVector(1.4, 1.0, 1.0)u"nm"
a1 = Atom(charge=1.0u"q", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
box_size = 2.0u"nm"
coords = [c1, c2]
s = Simulation(atoms=[a1, a1], coords=coords, box_size=box_size)
b1 = HarmonicBond(i=1, j=2, b0=0.6u"nm", kb=100_000.0u"kJ * mol^-1 * nm^-2")

SUITE["interactions"]["LennardJones force" ] = @benchmarkable force($(LennardJones()), $(c1), $(c2), $(a1), $(a1), $(box_size))
SUITE["interactions"]["LennardJones energy"] = @benchmarkable Molly.potential_energy($(LennardJones()), $(s), 1, 2)
SUITE["interactions"]["Coulomb force"      ] = @benchmarkable force($(Coulomb()), $(c1), $(c2), $(a1), $(a1), $(box_size))
SUITE["interactions"]["Coulomb energy"     ] = @benchmarkable Molly.potential_energy($(Coulomb()), $(s), 1, 2)
SUITE["interactions"]["HarmonicBond force" ] = @benchmarkable force($(b1), $(coords), $(s))
SUITE["interactions"]["HarmonicBond energy"] = @benchmarkable Molly.potential_energy($(b1), $(s))

SUITE["spatial"]["vector1D"] = @benchmarkable vector1D($(4.0u"nm"), $(6.0u"nm"), $(10.0u"nm"))
SUITE["spatial"]["vector"  ] = @benchmarkable vector($(SVector(4.0, 1.0, 6.0)u"nm"), $(SVector(6.0, 9.0, 4.0)u"nm"), $(10.0u"nm"))

n_atoms = 400
mass = 10.0u"u"
box_size = 6.0u"nm"
temp = 1.0u"K"
starting_coords = placediatomics(n_atoms ÷ 2, box_size, 0.2u"nm", 0.2u"nm")
starting_velocities = [velocity(mass, temp) for i in 1:n_atoms]
starting_coords_f32 = [Float32.(c) for c in starting_coords]
starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

function runsim(nl::Bool, parallel::Bool, gpu_diff_safe::Bool, f32::Bool, gpu::Bool)
    n_atoms = 400
    n_steps = 200
    mass = f32 ? 10.0f0u"u" : 10.0u"u"
    box_size = f32 ? 6.0f0u"nm" : 6.0u"nm"
    timestep = f32 ? 0.02f0u"ps" : 0.02u"ps"
    temp = f32 ? 1.0f0u"K" : 1.0u"K"
    simulator = VelocityVerlet()
    thermostat = NoThermostat()
    b0 = f32 ? 0.2f0u"nm" : 0.2u"nm"
    kb = f32 ? 10_000.0f0u"kJ * mol^-1 * nm^-2" : 10_000.0u"kJ * mol^-1 * nm^-2"
    bonds = [HarmonicBond(i=((i * 2) - 1), j=(i * 2), b0=b0, kb=kb) for i in 1:(n_atoms ÷ 2)]
    specific_inter_lists = (bonds,)

    neighbor_finder = NoNeighborFinder()
    cutoff = ShiftedPotentialCutoff(1.2u"nm")
    general_inters = (LennardJones(nl_only=false, cutoff=cutoff),)
    if nl
        neighbor_finder = DistanceNeighborFinder(trues(n_atoms, n_atoms), 10, f32 ? 1.5f0u"nm" : 1.5u"nm")
        general_inters = (LennardJones(nl_only=true, cutoff=cutoff),)
    end

    if gpu
        coords = cu(deepcopy(f32 ? starting_coords_f32 : starting_coords))
        velocities = cu(deepcopy(f32 ? starting_velocities_f32 : starting_velocities))
        atoms = cu([Atom(charge=f32 ? 0.0f0u"q" : 0.0u"q", mass=mass, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                            ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms])
    else
        coords = deepcopy(f32 ? starting_coords_f32 : starting_coords)
        velocities = deepcopy(f32 ? starting_velocities_f32 : starting_velocities)
        atoms = [Atom(charge=f32 ? 0.0f0u"q" : 0.0u"q", mass=mass, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                        ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms]
    end

    s = Simulation(
        simulator=simulator,
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        temperature=temp,
        box_size=box_size,
        neighbor_finder=neighbor_finder,
        thermostat=thermostat,
        timestep=timestep,
        n_steps=n_steps,
        gpu_diff_safe=gpu_diff_safe,
    )

    c = simulate!(s; parallel=parallel)
    return c
end

runs = [
    ("in-place"        , [false, false, false, false, false]),
    ("in-place NL"     , [true , false, false, false, false]),
    ("in-place f32"    , [false, false, false, true , false]),
    ("out-of-place"    , [false, false, true , false, false]),
    ("out-of-place f32", [false, false, true , true , false]),
]
if nthreads() > 1
    push!(runs, ("in-place parallel"   , [false, true , false, false, false]))
    push!(runs, ("in-place NL parallel", [true , true , false, false, false]))
end
if CUDA.functional()
    push!(runs, ("out-of-place gpu"    , [false, false, true , false, true ]))
    push!(runs, ("out-of-place gpu f32", [false, false, true , true , true ]))
end

for (name, args) in runs
    runsim(args...) # Run once for setup
    SUITE["simulation"][name] = @benchmarkable runsim($(args[1]), $(args[2]), $(args[3]), $(args[4]), $(args[5]))
end
