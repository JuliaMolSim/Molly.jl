# GPU Protein Simulation Benchmark for Launch Configurations

using Molly
using CUDA
using DelimitedFiles
using Unitful
using Printf
using BenchmarkTools

include("gpu_profile_utils.jl")

const data_dir = normpath(dirname(pathof(Molly)), "..", "data")
const ff_dir = joinpath(data_dir, "force_fields")
const openmm_dir = joinpath(data_dir, "openmm_6mrr")

function setup_protein_system(; f32=true, units=false, r_cut=1.0)
    T = f32 ? Float32 : Float64
    
    # Setup force field
    ff = MolecularForceField(
        T,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
        units=units,
    )

    # Load initial velocities
    velocities_nounits = SVector{3, T}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))
    velocities = Molly.add_units(velocities_nounits, u"nm * ps^-1", units)
    
    # Define system
    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        velocities=CuArray(velocities),
        units=units,
        array_type=CuArray,
        dist_cutoff=Molly.add_units(r_cut, u"nm", units),
        nonbonded_method=:cutoff,
    )

    return sys
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

function run_benchmark(sys, name, block_y_override)
    Molly.reset_cuda_launch_config!()
    
    if block_y_override !== nothing
        Molly.set_cuda_launch_config!(Molly.CUDALaunchConfig(
            force_block_y = block_y_override,
            energy_block_y = block_y_override
        ))
    else
        # Let the heuristic decide
        Molly.optimize_cuda_launch_config!(sys)
    end
    
    config = Molly.cuda_launch_config()
    current_block_y = config.force_block_y

    buffers_f = Molly.init_buffers!(sys, 1)
    buffers_e = Molly.init_buffers!(sys, 1, true)
    fs = Molly.zero_forces(sys)
    
    # Warmup
    Molly.forces!(fs, sys, nothing, buffers_f, Val(false), 0)
    Molly.forces!(fs, sys, nothing, buffers_f, Val(false), 1)
    potential_energy(sys, nothing, buffers_e, 0)
    potential_energy(sys, nothing, buffers_e, 1)
    CUDA.synchronize()

    # Profile kernels for diagnostic info
    buffers_f_prof = Molly.init_buffers!(sys, 1)
    sys.neighbor_finder.initialized = false
    force_profile = profile_gpu_force_path!(sys; buffers=buffers_f_prof)
    
    buffers_e_prof = Molly.init_buffers!(sys, 1, true)
    sys.neighbor_finder.initialized = false
    energy_profile = profile_gpu_energy_path!(sys; buffers=buffers_e_prof)

    samples = 1000
    
    # Benchmark Forces (excluding reorder steps)
    step_counter = 2
    forces_ms = 1_000 * @belapsed begin
        Molly.forces!($fs, $sys, nothing, $buffers_f, Val(false), $step_counter)
        CUDA.synchronize()
        $step_counter += 1
    end samples=samples evals=1
    
    # Benchmark Energy (excluding reorder steps)
    step_counter = 2
    pe_ms = 1_000 * @belapsed begin
        potential_energy($sys, nothing, $buffers_e, $step_counter)
        CUDA.synchronize()
        $step_counter += 1
    end samples=samples evals=1

    println(rpad(name, 14), " | block_y=", rpad(string(current_block_y), 4), " | Forces: ", @sprintf("%7.3f", forces_ms), " ms | PE: ", @sprintf("%7.3f", pe_ms), " ms")
    print_profile(name, force_profile, energy_profile)
    println()
end

function main()
    
    println("==================================================")
    println("Molly GPU Protein Benchmark (6mrr_equil.pdb)")
    println("CUDA device: ", CUDA.name(CUDA.device()))
    
    sys = setup_protein_system(f32=true, units=false, r_cut=1.0)
    n_atoms = length(sys.atoms)
    println("Atoms      : ", n_atoms)
    println("==================================================\n")

    configs = [
        ("block_y = 1",   1),
        ("block_y = 2",   2),
        ("block_y = 4",   4),
        ("block_y = 8",   8),
        ("block_y = 16", 16),
        ("Heuristic", nothing)
    ]

    for (name, override) in configs
        run_benchmark(sys, name, override)
    end
end

main()
