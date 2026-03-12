using Molly
using CUDA
using BenchmarkTools
using StaticArrays
using Unitful
using LinearAlgebra
using Random

# Function to setup a system of a given size
function setup_benchmark_system(n_atoms, r_cut=1.2u"nm")
    atom_mass = 39.948u"g/mol" # Argon
    σ = 0.34u"nm"
    ϵ = 0.997u"kJ * mol^-1"
    
    # Calculate box size for reasonable density (~1.4 g/cm^3 for liquid Argon)
    density = 1400.0u"kg/m^3"
    # vol = (n_atoms * atom_mass) / (density * Na)
    vol = (n_atoms * atom_mass) / (density * 6.02214076e23u"mol^-1")
    box_size = uconvert(u"nm", (vol |> upreferred)^(1/3))
    
    # Ensure box is at least 2.5*r_cut to avoid small box issues
    box_size = max(box_size, 2.5 * r_cut)
    boundary = CubicBoundary(box_size)
    
    Random.seed!(42)
    coords = [SVector(rand(), rand(), rand()) * box_size for _ in 1:n_atoms]
    velocities = [zero(SVector{3, typeof(1.0u"nm/ps")}) for _ in 1:n_atoms]
    atoms = [Atom(index=i, mass=atom_mass, charge=0.0, σ=σ, ϵ=ϵ) for i in 1:n_atoms]
    
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
            eligible=CuArray(ones(Bool, n_atoms, n_atoms)),
            special=CuArray(zeros(Bool, n_atoms, n_atoms)),
            dist_cutoff=r_cut
        ),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )
    
    return sys_gpu
end

function run_benchmark()
    # Test across different system sizes
    sizes = [1024, 4096, 16384, 32768]
    
    println("Molly GPU Nonbonded Force Benchmark (Tile-based Algorithm)")
    println("-"^70)
    println(rpad("Atoms", 10), " | ", rpad("Median Time", 20), " | ", "Interacting Tiles")
    println("-"^70)

    for n in sizes
        sys = setup_benchmark_system(n)
        
        # Initialize buffers manually to access tile count
        buffers = Molly.init_buffers!(sys, 1)
        fs = Molly.zero_forces(sys)
        
        # Warmup and count interacting tiles
        Molly.forces!(fs, sys, nothing, buffers, Val(false), 0)
        num_tiles = Array(buffers.num_interacting_tiles)[1]
        
        # Benchmark
        # We benchmark the forces! call which is the core loop
        # Use parentheses around CUDA.@sync to avoid greedy macro argument consumption
        b = @benchmark (CUDA.@sync Molly.forces!($fs, $sys, nothing, $buffers, Val(false), 0)) samples=10 evals=1
        
        t_median = median(b).time / 1e6 # ms
        
        println(rpad(n, 10), " | ", rpad(string(round(t_median, digits=3), " ms"), 20), " | ", num_tiles)
    end
    println("-"^70)
end

# Run the benchmark
run_benchmark()
