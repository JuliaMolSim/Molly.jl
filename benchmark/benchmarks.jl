# Benchmark suite for Molly
# Run with something like:
#   using Molly, PkgBenchmark
#   results = benchmarkpkg(Molly, BenchmarkConfig(env=Dict("JULIA_NUM_THREADS" => 16)))
#   export_markdown(out_file, results)

using Molly
using BenchmarkTools
using CUDA

using DelimitedFiles

const run_parallel_tests = Threads.nthreads() > 1
if run_parallel_tests
    @info "The parallel benchmarks will be run as Julia is running on $(Threads.nthreads()) threads"
else
    @warn "The parallel benchmarks will not be run as Julia is running on 1 thread"
end

# Allow GPU device to be specified
const DEVICE = parse(Int, get(ENV, "DEVICE", "0"))

const run_cuda_tests = CUDA.functional()
if run_cuda_tests
    device!(DEVICE)
    @info "The CUDA benchmarks will be run on device $DEVICE"
else
    @warn "The CUDA benchmarks will not be run as a CUDA-enabled device is not available"
end

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
boundary = CubicBoundary(2.0u"nm")
coords = [c1, c2]
dr = vector(c1, c2, boundary)
b1 = HarmonicBond(k=100_000.0u"kJ * mol^-1 * nm^-2", r0=0.6u"nm")

SUITE["interactions"]["LennardJones force" ] = @benchmarkable force($(LennardJones()), $(dr), $(a1), $(a1))
SUITE["interactions"]["LennardJones energy"] = @benchmarkable potential_energy($(LennardJones()), $(dr), $(a1), $(a1))
SUITE["interactions"]["Coulomb force"      ] = @benchmarkable force($(Coulomb()), $(dr), $(a1), $(a1))
SUITE["interactions"]["Coulomb energy"     ] = @benchmarkable potential_energy($(Coulomb()), $(dr), $(a1), $(a1))
SUITE["interactions"]["HarmonicBond force" ] = @benchmarkable force($(b1), $(c1), $(c2), $(boundary))
SUITE["interactions"]["HarmonicBond energy"] = @benchmarkable potential_energy($(b1), $(c1), $(c2), $(boundary))

SUITE["spatial"]["vector_1D"] = @benchmarkable vector_1D($(4.0u"nm"), $(6.0u"nm"), $(10.0u"nm"))
SUITE["spatial"]["vector"   ] = @benchmarkable vector($(SVector(4.0, 1.0, 1.0)u"nm"), $(SVector(6.0, 4.0, 3.0)u"nm"), $(CubicBoundary(SVector(10.0, 5.0, 3.5)u"nm")))

n_atoms = 400
atom_mass = 10.0u"g/mol"
boundary = CubicBoundary(6.0u"nm")
const starting_coords = place_diatomics(n_atoms ÷ 2, boundary, 0.2u"nm"; min_dist=0.2u"nm")
const starting_velocities = [random_velocity(atom_mass, 1.0u"K") for i in 1:n_atoms]
const starting_coords_f32 = [Float32.(c) for c in starting_coords]
const starting_velocities_f32 = [Float32.(c) for c in starting_velocities]

function test_sim(nl::Bool, parallel::Bool, f32::Bool, ::Type{AT}) where AT
    n_atoms = 400
    n_steps = 200
    atom_mass = f32 ? 10.0f0u"g/mol" : 10.0u"g/mol"
    boundary = f32 ? CubicBoundary(6.0f0u"nm") : CubicBoundary(6.0u"nm")
    simulator = VelocityVerlet(dt=f32 ? 0.02f0u"ps" : 0.02u"ps")
    k = f32 ? 10_000.0f0u"kJ * mol^-1 * nm^-2" : 10_000.0u"kJ * mol^-1 * nm^-2"
    r0 = f32 ? 0.2f0u"nm" : 0.2u"nm"
    bonds = [HarmonicBond(k=k, r0=r0) for i in 1:(n_atoms ÷ 2)]
    specific_inter_lists = (InteractionList2Atoms(
        AT(Int32.(collect(1:2:n_atoms))),
        AT(Int32.(collect(2:2:n_atoms))),
        AT(bonds),
    ),)

    neighbor_finder = NoNeighborFinder()
    cutoff = DistanceCutoff(f32 ? 1.0f0u"nm" : 1.0u"nm")
    pairwise_inters = (LennardJones(use_neighbors=false, cutoff=cutoff),)
    if nl
        neighbor_finder = DistanceNeighborFinder(
            eligible=AT(trues(n_atoms, n_atoms)),
            n_steps=10,
            dist_cutoff=f32 ? 1.5f0u"nm" : 1.5u"nm",
        )
        pairwise_inters = (LennardJones(use_neighbors=true, cutoff=cutoff),)
    end

    coords = AT(copy(f32 ? starting_coords_f32 : starting_coords))
    velocities = AT(copy(f32 ? starting_velocities_f32 : starting_velocities))
    atoms = AT([Atom(charge=f32 ? 0.0f0 : 0.0, mass=atom_mass, σ=f32 ? 0.2f0u"nm" : 0.2u"nm",
                     ϵ=f32 ? 0.2f0u"kJ * mol^-1" : 0.2u"kJ * mol^-1") for i in 1:n_atoms])

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        neighbor_finder=neighbor_finder,
    )

    n_threads = parallel ? Threads.nthreads() : 1
    simulate!(sys, simulator, n_steps; n_threads=n_threads)
    return sys.coords
end

runs = [
    ("CPU"       , [false, false, false, Array]),
    ("CPU f32"   , [false, false, true , Array]),
    ("CPU NL"    , [true , false, false, Array]),
    ("CPU f32 NL", [true , false, true , Array]),
]
if run_parallel_tests
    push!(runs, ("CPU parallel"       , [false, true , false, Array]))
    push!(runs, ("CPU parallel f32"   , [false, true , true , Array]))
    push!(runs, ("CPU parallel NL"    , [true , true , false, Array]))
    push!(runs, ("CPU parallel f32 NL", [true , true , true , Array]))
end
if run_cuda_tests
    push!(runs, ("GPU"       , [false, false, false, CuArray]))
    push!(runs, ("GPU f32"   , [false, false, true , CuArray]))
    push!(runs, ("GPU NL"    , [true , false, false, CuArray]))
    push!(runs, ("GPU f32 NL", [true , false, true , CuArray]))
end

for (name, args) in runs
    test_sim(args...) # Run once for setup
    SUITE["simulation"][name] = @benchmarkable test_sim($(args[1]), $(args[2]), $(args[3]), $(args[4]))
end

data_dir = normpath(@__DIR__, "..", "data")
ff_dir = joinpath(data_dir, "force_fields")
openmm_dir = joinpath(data_dir, "openmm_6mrr")

ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...)
velocities = SVector{3}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))u"nm * ps^-1"
sys = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; velocities=velocities)
sim = VelocityVerlet(dt=0.0005u"ps")
n_steps = 25

simulate!(sys, sim, n_steps; n_threads=Threads.nthreads())
SUITE["protein"]["CPU parallel NL"] = @benchmarkable simulate!($(sys), $(sim), $(n_steps); n_threads=Threads.nthreads())
