using Molly
using CUDA
using BenchmarkTools
using Printf

include("gpu_profile_utils.jl")

const CUDA_TUNING_ENV_VARS = [
    "MOLLY_CUDA_FORCE_BLOCK_Y",
    "MOLLY_CUDA_FORCE_MAXREGS",
    "MOLLY_CUDA_TILE_THREADS_X",
    "MOLLY_CUDA_TILE_THREADS_Y",
    "MOLLY_CUDA_ENERGY_BLOCK_Y",
]

env_int(name::AbstractString, default::Int) = something(tryparse(Int, get(ENV, name, string(default))), default)
selected_force_path_label(sys) = String(gpu_cuda_ext().selected_force_path(sys))
selected_energy_path_label(sys) = String(gpu_cuda_ext().selected_energy_path(sys))

function apply_policy!(policy)
    for name in CUDA_TUNING_ENV_VARS
        delete!(ENV, name)
    end

    Molly.reset_cuda_launch_config!()
    if policy.config !== nothing
        Molly.set_cuda_launch_config!(policy.config)
    end
    return nothing
end

function bench_policy(sys, policy; samples)
    apply_policy!(policy)
    force_path = selected_force_path_label(sys)
    energy_path = selected_energy_path_label(sys)

    forces(sys)
    potential_energy(sys)
    CUDA.synchronize()

    force_profile = nothing
    if force_path != "dense"
        profile_gpu_force_path!(sys; buffers=Molly.init_buffers!(sys, 1))
        sys.neighbor_finder.initialized = false
        force_profile = profile_gpu_force_path!(sys; buffers=Molly.init_buffers!(sys, 1))
    end
    energy_profile = if energy_path != "dense"
        profile_gpu_energy_path!(sys; buffers=Molly.init_buffers!(sys, 1, true))
        profile_gpu_energy_path!(sys; buffers=Molly.init_buffers!(sys, 1, true))
    else
        nothing
    end

    t_forces = @belapsed begin
        forces($sys)
        CUDA.synchronize()
    end samples=samples evals=1

    t_pe = @belapsed begin
        potential_energy($sys)
        CUDA.synchronize()
    end samples=samples evals=1

    return (
        forces_ms = 1_000 * t_forces,
        pe_ms = 1_000 * t_pe,
        total_ms = 1_000 * (t_forces + t_pe),
        force_path = force_path,
        energy_path = energy_path,
        force_profile = force_profile,
        energy_profile = energy_profile,
    )
end

function print_result(name, result)
    @printf(
        "%-16s forces_ms=%8.3f pe_ms=%8.3f total_ms=%8.3f\n",
        name,
        result.forces_ms,
        result.pe_ms,
        result.total_ms,
    )
    println("  selected force path: ", result.force_path)
    println("  selected energy path: ", result.energy_path)
    if result.energy_profile === nothing
        println("  energy stages: skipped (dense pairwise-path override/selection active)")
    else
        stats = result.energy_profile.tile_stats
        @printf(
            "  tiles=%d clean=%d masked=%d frac=%7.4f overflow=%d\n",
            stats.num_tiles,
            stats.clean_tiles,
            stats.masked_tiles,
            stats.interacting_fraction,
            stats.overflow_count,
        )
    end
    if result.force_profile === nothing
        println("  force stages : skipped (dense pairwise-path override/selection active)")
    else
        force_times = result.force_profile.times
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
    end
    if result.energy_profile !== nothing
        energy_times = result.energy_profile.times
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
end

function prepare_system()
    device!(env_int("MOLLY_CUDA_DEVICE", 1))

    data_dir = normpath(dirname(pathof(Molly)), "..", "data")
    ff_dir = joinpath(data_dir, "force_fields")

    T = Float32
    ff = MolecularForceField(
        T,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
        units = true,
    )

    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        units = true,
        array_type = CuArray,
        nonbonded_method = :none,
    )

    minim_steps = env_int("MOLLY_CUDA_BENCH_MIN_STEPS", 5_000)
    if minim_steps > 0
        minim = SteepestDescentMinimizer(
            step_size = T(0.05)u"nm",
            max_steps = minim_steps,
            log_stream = devnull,
        )
        simulate!(sys, minim)
    end

    random_velocities!(sys, T(310)u"K")
    CUDA.synchronize()

    return sys
end

function main()
    samples = env_int("MOLLY_CUDA_BENCH_SAMPLES", 10)
    sys = prepare_system()

    println("CUDA device: ", CUDA.name(CUDA.device()))
    println("Neighbor finder: ", typeof(sys.neighbor_finder))
    println("Benchmark samples per policy: ", samples)

    policies = [
        (
            name = "legacy_explicit",
            description = "Previous hard-coded values",
            config = Molly.CUDALaunchConfig(
                force_block_y = 8,
                force_maxregs = 64,
                tile_threads = (16, 16),
                energy_block_y = 8,
            ),
        ),
        (
            name = "tuned_explicit",
            description = "Best explicit values from the prior sweep",
            config = Molly.CUDALaunchConfig(
                force_block_y = 12,
                force_maxregs = 56,
                tile_threads = (16, 8),
                energy_block_y = 4,
            ),
        ),
        (
            name = "auto",
            description = "No launch overrides; use compiler/runtime choices",
            config = nothing,
        ),
    ]

    results = NamedTuple[]
    println()
    for policy in policies
        println(policy.name, ": ", policy.description)
        result = bench_policy(sys, policy; samples)
        push!(results, (; policy, result))
        print_result(policy.name, result)
        println()
    end

    baseline = only(filter(entry -> entry.policy.name == "legacy_explicit", results))
    auto = only(filter(entry -> entry.policy.name == "auto", results))
    tuned = only(filter(entry -> entry.policy.name == "tuned_explicit", results))

    auto_vs_legacy = 100 * (auto.result.total_ms - baseline.result.total_ms) / baseline.result.total_ms
    auto_vs_tuned = 100 * (auto.result.total_ms - tuned.result.total_ms) / tuned.result.total_ms

    @printf("AUTO vs legacy total delta: %+6.2f%%\n", auto_vs_legacy)
    @printf("AUTO vs tuned total delta:  %+6.2f%%\n", auto_vs_tuned)
    Molly.reset_cuda_launch_config!()
end

main()
