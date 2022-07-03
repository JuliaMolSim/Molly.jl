# Benchmark suite for Molly
# Run with something like:
#   using Molly, PkgBenchmark
#   results = benchmarkpkg(Molly, BenchmarkConfig(env=Dict("JULIA_NUM_THREADS" => 16)))
#   export_markdown(out_file, results)

using Molly
using BenchmarkTools
using CUDA

using Base.Threads
using DelimitedFiles

# Allow CUDA device to be specified
const DEVICE = get(ENV, "DEVICE", "0")

run_parallel_tests = Threads.nthreads() > 1
if run_parallel_tests
    @info "The parallel benchmarks will be run as Julia is running on $(Threads.nthreads()) threads"
else
    @warn "The parallel benchmarks will not be run as Julia is running on 1 thread"
end

run_gpu_tests = CUDA.functional()
if run_gpu_tests
    device!(parse(Int, DEVICE))
    @info "The GPU tests will be run on device $DEVICE"
else
    @warn "The GPU benchmarks will not be run as a CUDA-enabled device is not available"
end

CUDA.allowscalar(false) # Check that we never do scalar indexing on the GPU

const SUITE = BenchmarkGroup(
    [],
    "interactions" => BenchmarkGroup(),
    "spatial"      => BenchmarkGroup(),
    "simulation"   => BenchmarkGroup(),
    "protein"      => BenchmarkGroup(),
)

c1 = SVector(1.0, 1.0, 1.0)u"nm"
c2 = SVector(1.4, 1.0, 1.0)u"nm"
a1 = Atom(charge=1.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1")
boundary = CubicBoundary(2.0u"nm", 2.0u"nm", 2.0u"nm")
coords = [c1, c2]
dr = vector(c1, c2, boundary)
b1 = HarmonicBond(b0=0.6u"nm", kb=100_000.0u"kJ * mol^-1 * nm^-2")

SUITE["interactions"]["LennardJones force" ] = @benchmarkable force($(LennardJones()), $(dr), $(c1), $(c2), $(a1), $(a1), $(boundary))
SUITE["interactions"]["LennardJones energy"] = @benchmarkable potential_energy($(LennardJones()), $(dr), $(c1), $(c2), $(a1), $(a1), $(boundary))
SUITE["interactions"]["Coulomb force"      ] = @benchmarkable force($(Coulomb()), $(dr), $(c1), $(c2), $(a1), $(a1), $(boundary))
SUITE["interactions"]["Coulomb energy"     ] = @benchmarkable potential_energy($(Coulomb()), $(dr), $(c1), $(c2), $(a1), $(a1), $(boundary))
SUITE["interactions"]["HarmonicBond force" ] = @benchmarkable force($(b1), $(c1), $(c2), $(boundary))
SUITE["interactions"]["HarmonicBond energy"] = @benchmarkable potential_energy($(b1), $(c1), $(c2), $(boundary))

SUITE["spatial"]["vector_1D"] = @benchmarkable vector_1D($(4.0u"nm"), $(6.0u"nm"), $(10.0u"nm"))
SUITE["spatial"]["vector"   ] = @benchmarkable vector($(SVector(4.0, 1.0, 1.0)u"nm"), $(SVector(6.0, 4.0, 3.0)u"nm"), $(CubicBoundary(SVector(10.0, 5.0, 3.5)u"nm")))

n_atoms = 400
atom_mass = 10.0u"u"
boundary = CubicBoundary(6.0u"nm", 6.0u"nm", 6.0u"nm")
starting_coords = place_diatomics(n_atoms ÷ 2, boundary, 0.2u"nm", 0.2u"nm")
starting_velocities = [velocity(atom_mass, 1.0u"K") for i in 1:n_atoms]
starting_coords_f32 = [Float32.(c) for c in starting_coords]
starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

function test_sim(nl::Bool, parallel::Bool, gpu_diff_safe::Bool, f32::Bool, gpu::Bool)
    n_atoms = 400
    n_steps = 200
    atom_mass = f32 ? 10.0f0u"u" : 10.0u"u"
    boundary = f32 ? CubicBoundary(6.0f0u"nm", 6.0f0u"nm", 6.0f0u"nm") : CubicBoundary(6.0u"nm", 6.0u"nm", 6.0u"nm")
    simulator = VelocityVerlet(dt=f32 ? 0.02f0u"ps" : 0.02u"ps")
    b0 = f32 ? 0.2f0u"nm" : 0.2u"nm"
    kb = f32 ? 10_000.0f0u"kJ * mol^-1 * nm^-2" : 10_000.0u"kJ * mol^-1 * nm^-2"
    bonds = [HarmonicBond(b0=b0, kb=kb) for i in 1:(n_atoms ÷ 2)]
    specific_inter_lists = (InteractionList2Atoms(collect(1:2:n_atoms), collect(2:2:n_atoms),
                            repeat([""], length(bonds)), gpu ? cu(bonds) : bonds),)

    neighbor_finder = NoNeighborFinder()
    cutoff = DistanceCutoff(f32 ? 1.0f0u"nm" : 1.0u"nm")
    pairwise_inters = (LennardJones(nl_only=false, cutoff=cutoff),)
    if nl
        if gpu_diff_safe
            neighbor_finder = DistanceVecNeighborFinder(nb_matrix=gpu ? cu(trues(n_atoms, n_atoms)) : trues(n_atoms, n_atoms),
                                                        n_steps=10, dist_cutoff=f32 ? 1.5f0u"nm" : 1.5u"nm")
        else
            neighbor_finder = DistanceNeighborFinder(nb_matrix=trues(n_atoms, n_atoms), n_steps=10,
                                                        dist_cutoff=f32 ? 1.5f0u"nm" : 1.5u"nm")
        end
        pairwise_inters = (LennardJones(nl_only=true, cutoff=cutoff),)
    end

    if gpu
        coords = cu(deepcopy(f32 ? starting_coords_f32 : starting_coords))
        velocities = cu(deepcopy(f32 ? starting_velocities_f32 : starting_velocities))
        atoms = cu([Atom(charge=f32 ? 0.0f0 : 0.0, mass=atom_mass, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                            ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms])
    else
        coords = deepcopy(f32 ? starting_coords_f32 : starting_coords)
        velocities = deepcopy(f32 ? starting_velocities_f32 : starting_velocities)
        atoms = [Atom(charge=f32 ? 0.0f0 : 0.0, mass=atom_mass, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                        ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms]
    end

    n_threads = parallel ? Threads.nthreads() : 1

    s = System(
        atoms=atoms,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        coords=coords,
        velocities=velocities,
        boundary=boundary,
        neighbor_finder=neighbor_finder,
        gpu_diff_safe=gpu_diff_safe,
    )

    simulate!(s, simulator, n_steps; n_threads=n_threads)
    return s.coords
end

runs = [
    ("in-place"        , [false, false, false, false, false]),
    ("in-place NL"     , [true , false, false, false, false]),
    ("in-place f32"    , [false, false, false, true , false]),
    ("out-of-place"    , [false, false, true , false, false]),
    ("out-of-place NL" , [true , false, true , false, false]),
    ("out-of-place f32", [false, false, true , true , false]),
]
if run_parallel_tests
    push!(runs, ("in-place parallel"   , [false, true , false, false, false]))
    push!(runs, ("in-place NL parallel", [true , true , false, false, false]))
end
if run_gpu_tests
    push!(runs, ("out-of-place gpu"       , [false, false, true , false, true ]))
    push!(runs, ("out-of-place gpu f32"   , [false, false, true , true , true ]))
    push!(runs, ("out-of-place gpu NL"    , [true , false, true , false, true ]))
    push!(runs, ("out-of-place gpu f32 NL", [true , false, true , true , true ]))
end

for (name, args) in runs
    test_sim(args...) # Run once for setup
    SUITE["simulation"][name] = @benchmarkable test_sim($(args[1]), $(args[2]), $(args[3]), $(args[4]), $(args[5]))
end

data_dir = normpath(@__DIR__, "..", "data")
ff_dir = joinpath(data_dir, "force_fields")
openmm_dir = joinpath(data_dir, "openmm_6mrr")

ff = OpenMMForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)
velocities = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))u"nm * ps^-1"
s = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; velocities=velocities)
simulator = VelocityVerlet(dt=0.0005u"ps")
n_steps = 25

simulate!(s, simulator, n_steps; n_threads=Threads.nthreads())
SUITE["protein"]["in-place NL parallel"] = @benchmarkable simulate!($(s), $(simulator), $(n_steps); n_threads=Threads.nthreads())
