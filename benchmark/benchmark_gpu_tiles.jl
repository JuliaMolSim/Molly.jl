using CUDA
using BenchmarkTools
using StaticArrays
using Unitful
using LinearAlgebra
using Random
using Printf

include("gpu_profile_utils.jl")

env_int(name::AbstractString, default::Int) = something(tryparse(Int, get(ENV, name, string(default))), default)

function setup_benchmark_system(n_atoms;
                                r_cut=1.2u"nm",
                                density=1400.0u"kg/m^3",
                                box_multiplier=1.0,
                                seed=42)
    T = Float32
    atom_mass = T(39.948)u"g/mol" # Argon
    σ = T(0.34)u"nm"
    ϵ = T(0.997)u"kJ * mol^-1"

    vol = (n_atoms * atom_mass) / (T(density) * T(6.02214076e23)u"mol^-1")
    box_size = uconvert(u"nm", (vol |> upreferred)^(1/3))
    box_size = max(box_size * T(box_multiplier), T(2.5) * r_cut)
    boundary = CubicBoundary(box_size)

    Random.seed!(seed)
    coords = [SVector{3, T}(rand(T), rand(T), rand(T)) * box_size for _ in 1:n_atoms]
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

function print_profile(name, force_profile, energy_profile)
    force_times = force_profile.times
    stats = force_profile.tile_stats
    @printf(
        "%-14s tiles=%7d clean=%7d masked=%7d frac=%7.4f overflow=%d\n",
        name,
        stats.num_tiles,
        stats.clean_tiles,
        stats.masked_tiles,
        stats.interacting_fraction,
        stats.overflow_count,
    )
    @printf(
        "  force stages : morton=%7.3f compress=%7.3f reorder=%7.3f bounds=%7.3f tile=%7.3f kernel=%7.3f reverse=%7.3f\n",
        force_times.morton_sort_ms,
        force_times.compress_ms,
        force_times.reorder_ms,
        force_times.bounds_ms,
        force_times.tile_find_ms,
        force_times.force_kernel_ms,
        force_times.reverse_reorder_ms,
    )
    energy_times = energy_profile.times
    @printf(
        "  energy stages: morton=%7.3f compress=%7.3f reorder=%7.3f bounds=%7.3f tile=%7.3f kernel=%7.3f\n",
        energy_times.morton_sort_ms,
        energy_times.compress_ms,
        energy_times.reorder_ms,
        energy_times.bounds_ms,
        energy_times.tile_find_ms,
        energy_times.energy_kernel_ms,
    )
end

function run_benchmark()
    device!(env_int("MOLLY_CUDA_DEVICE", 0))
    samples = env_int("MOLLY_CUDA_BENCH_SAMPLES", 10)
    n_atoms = env_int("MOLLY_CUDA_TILE_BENCH_ATOMS", 4096)

    cases = [
        (
            name = "dense_f32",
            description = "High-density cutoff regime",
            sys = setup_benchmark_system(n_atoms; density=1400.0u"kg/m^3", box_multiplier=1.0, seed=42),
        ),
        (
            name = "sparse_f32",
            description = "Sparse cutoff regime",
            sys = setup_benchmark_system(n_atoms; density=1400.0u"kg/m^3", box_multiplier=4.0, seed=43),
        ),
    ]


    println("Molly GPU Nonbonded Tile Benchmark")
    println("CUDA device: ", CUDA.name(CUDA.device()))
    println("Atoms per case: ", n_atoms)
    println("Benchmark samples per case: ", samples)
    println()

    println(rpad("Case", 14), " | ", rpad("Forces median", 14), " | ", rpad("PE median", 12), " | Interacting tiles")
    println("-"^80)

    for case in cases
        sys = case.sys
        n_threads = 1
        blocks_y = optimize_cuda_launch_config!(sys)
        println("USING $(blocks_y) blocks")
        buffers_f = Molly.init_buffers!(sys, n_threads)
        buffers_e = Molly.init_buffers!(sys, n_threads, true)
        
        fs = Molly.zero_forces(sys)
        Molly.forces!(fs, sys, nothing, buffers_f, Val(false), 0)
        potential_energy(sys, nothing, buffers_e, 0)
        CUDA.synchronize()

        profile_gpu_force_path!(sys; buffers=buffers_f)
        sys.neighbor_finder.initialized = false
        force_profile = profile_gpu_force_path!(sys; buffers=buffers_f)
        profile_gpu_energy_path!(sys; buffers=buffers_e)
        energy_profile = profile_gpu_energy_path!(sys; buffers=buffers_e)

        # Use a local counter to increment step_n and trigger reordering as in a real simulation
        step_counter = 0
        forces_ms = 1_000 * @belapsed begin
            Molly.forces!($fs, $sys, nothing, $buffers_f, Val(false), $step_counter)
            CUDA.synchronize()
            $step_counter += 1
        end samples=samples evals=1

        step_counter = 0
        pe_ms = 1_000 * @belapsed begin
            potential_energy($sys, nothing, $buffers_e, $step_counter)
            CUDA.synchronize()
            $step_counter += 1
        end samples=samples evals=1

        println(
            rpad(case.name, 14), " | ",
            @sprintf("%10.3f ms", forces_ms), " | ",
            @sprintf("%8.3f ms", pe_ms), " | ",
            string(energy_profile.tile_stats.num_tiles),
        )
        print_profile(case.name, force_profile, energy_profile)
        println()
    end
end

run_benchmark()
