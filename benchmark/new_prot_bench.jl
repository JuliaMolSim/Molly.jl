using Molly
using CUDA
using BenchmarkTools
using Printf

const CUDA_TUNING_ENV_VARS = [
    "MOLLY_CUDA_FORCE_BLOCK_Y",
    "MOLLY_CUDA_FORCE_MAXREGS",
    "MOLLY_CUDA_TILE_THREADS_X",
    "MOLLY_CUDA_TILE_THREADS_Y",
    "MOLLY_CUDA_ENERGY_BLOCK_Y",
]

env_int(name::AbstractString, default::Int) = something(tryparse(Int, get(ENV, name, string(default))), default)

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

    forces(sys)
    potential_energy(sys)
    CUDA.synchronize()

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
