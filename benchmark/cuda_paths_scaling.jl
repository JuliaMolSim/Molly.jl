using Molly
using CUDA
using BenchmarkTools
using StaticArrays
using Unitful
using Printf
using Random
using Statistics
using DelimitedFiles

# Explicitly set the device
device!(0)

# Ensure we are using the GPU
if !CUDA.functional()
    error("CUDA is not functional. This benchmark requires a GPU.")
end

const BENCHMARK_STEPS = 1_000
const BENCHMARK_DT = 0.002u"ps"

env_int(name::AbstractString, default::Int) = something(tryparse(Int, get(ENV, name, string(default))), default)
env_float(name::AbstractString, default::Float64) = something(tryparse(Float64, get(ENV, name, string(default))), default)

function generate_lattice_coords(n_atoms, box_size, T; seed=42, jitter_scale=T(0.05))
    cells = ceil(Int, cbrt(n_atoms))
    spacing = box_size / T(cells)
    Random.seed!(seed)

    coords = Vector{SVector{3, typeof(box_size)}}(undef, n_atoms)
    idx = 1
    for iz in 0:(cells - 1), iy in 0:(cells - 1), ix in 0:(cells - 1)
        idx > n_atoms && break
        base = SVector{3, T}(ix + 0.5f0, iy + 0.5f0, iz + 0.5f0) * spacing
        jitter = spacing * jitter_scale .* SVector{3, T}(
            rand(T) - 0.5f0,
            rand(T) - 0.5f0,
            rand(T) - 0.5f0,
        )
        coords[idx] = SVector{3, typeof(box_size)}(base .+ jitter)
        idx += 1
    end
    return coords
end


function setup_benchmark_system(n_atoms;
                                r_cut=1.2u"nm",
                                density=1000.0u"kg/m^3",
                                seed=42)
    T = Float32
    atom_mass = T(39.948)u"g/mol" # Argon
    σ = T(0.34)u"nm"
    ϵ = T(0.997)u"kJ * mol^-1"

    vol = (n_atoms * atom_mass) / (T(density) * T(6.02214076e23)u"mol^-1")
    box_size = uconvert(u"nm", (vol |> upreferred)^(1/3))
    box_size = max(box_size, T(2.5) * r_cut)
    boundary = CubicBoundary(box_size)

    coords = generate_lattice_coords(n_atoms, box_size, T; seed=seed)
    velocities = [zero(SVector{3, typeof(T(1.0)u"nm/ps")}) for _ in 1:n_atoms]
    atoms = [Atom(index=i, mass=atom_mass, charge=T(0.0), σ=σ, ϵ=ϵ) for i in 1:n_atoms]

    pairwise_inters = (LennardJones(
        cutoff=DistanceCutoff(r_cut),
        use_neighbors=true,
    ),)
    
    sys_gpu = System(
        atoms=CuArray(atoms),
        coords=CuArray(coords),
        velocities=CuArray(velocities),
        boundary=boundary,
        pairwise_inters=pairwise_inters,
        neighbor_finder=Molly.GPUNeighborFinder(
            n_atoms=n_atoms,
            excluded_pairs=(),
            special_pairs=(),
            dist_cutoff=r_cut,
            device_vector_type=CuArray{Int32, 1},
        ),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )
    
    return sys_gpu
end

function run_scaling_benchmark()
    atom_counts = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    samples = env_int("MOLLY_CUDA_BENCH_SAMPLES", 50)
    benchmark_seconds = env_float("MOLLY_CUDA_BENCH_SECONDS", 120.0)
    csv_file = "benchmark_results_sim.csv"
    sim = VelocityVerlet(dt=BENCHMARK_DT, remove_CM_motion=false)
    
    results = Any[["Atoms", "Median (ms/step)", "Mean (ms/step)", "StdDev (ms/step)"]]
    
    println("Molly CUDA 1000-Step Simulation Scaling Benchmark")
    println("CUDA device: ", CUDA.name(CUDA.device()))
    println("Samples target per size: ", samples)
    println("Time budget per size (s): ", benchmark_seconds)
    println("Steps per sample: ", BENCHMARK_STEPS)
    println()
    @printf("%10s | %10s | %10s | %10s\n", "Atoms", "Median", "Mean", "StdDev")
    println("-"^54)

    for n in atom_counts
        sys = setup_benchmark_system(n)
        # Warm up the simulation path, including steady-state neighbor refresh behavior.
        simulate!(sys, sim, BENCHMARK_STEPS; run_loggers=false)
        CUDA.synchronize()

        # Benchmark the average cost per simulation step over a longer trajectory.
        t = @benchmark begin
            simulate!($sys, $sim, $BENCHMARK_STEPS; run_loggers=false)
            CUDA.synchronize()
        end samples=samples seconds=benchmark_seconds evals=1

        m_median = median(t.times) / (1e6 * BENCHMARK_STEPS)
        m_mean = mean(t.times) / (1e6 * BENCHMARK_STEPS)
        m_std = std(t.times) / (1e6 * BENCHMARK_STEPS)

        @printf("%10d | %8.3f ms | %8.3f ms | %8.3f ms\n", n, m_median, m_mean, m_std)
        push!(results, Any[n, m_median, m_mean, m_std])
        println("-"^54)
    end
    
    writedlm(csv_file, results, ',')
    println("Results saved to $csv_file")
end

run_scaling_benchmark()
