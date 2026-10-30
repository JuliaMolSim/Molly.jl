# Energy conservation test

using Molly
using AMDGPU
using CUDA
using GPUArrays

using Random
using Test

@testset "Lennard-Jones energy conservation" begin
    function test_energy_conservation(nl::Bool, ::Type{AT}, n_threads::Integer,
                                      n_steps::Integer) where AT
        n_atoms = 2_000
        atom_mass = 40.0u"g/mol"
        temp = 1.0u"K"
        boundary = CubicBoundary(5.0u"nm")
        simulator = VelocityVerlet(dt=0.001u"ps", remove_CM_motion=false)

        atoms = [Atom(mass=atom_mass, charge=0.0, σ=0.05u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
        dist_cutoff = 3.0u"nm"
        cutoffs = (
            DistanceCutoff(dist_cutoff),
            ShiftedPotentialCutoff(dist_cutoff),
            ShiftedForceCutoff(dist_cutoff),
            CubicSplineCutoff(dist_cutoff, dist_cutoff + 0.5u"nm"),
        )

        for cutoff in cutoffs
            coords = place_atoms(n_atoms, boundary; min_dist=0.1u"nm")
            if nl
                if Molly.uses_gpu_neighbor_finder(AT)
                    neighbor_finder=GPUNeighborFinder(
                        n_atoms=n_atoms,
                        dist_cutoff=dist_cutoff,
                        array_type=AT,
                    )
                else
                    neighbor_finder=DistanceNeighborFinder(
                        eligible=to_device(trues(n_atoms, n_atoms), AT),
                        n_steps=10,
                        dist_cutoff=dist_cutoff,
                    )
                end
            else
                neighbor_finder = NoNeighborFinder()
            end

            sys = System(
                atoms=to_device(atoms, AT),
                coords=to_device(coords, AT),
                boundary=boundary,
                pairwise_inters=(LennardJones(cutoff=cutoff, use_neighbors=ifelse(nl, true, false)),),
                neighbor_finder=neighbor_finder,
                loggers=(
                    coords=CoordinatesLogger(100),
                    energy=TotalEnergyLogger(100),
                ),
            )
            random_velocities!(sys, temp)

            E0 = total_energy(sys; n_threads=n_threads)
            simulate!(deepcopy(sys), simulator, 20; n_threads=n_threads)
            @time simulate!(sys, simulator, n_steps; n_threads=n_threads)

            Es = values(sys.loggers.energy)
            @test isapprox(Es[1], E0; atol=1e-7u"kJ * mol^-1")

            max_ΔE = maximum(abs.(Es .- E0))
            platform_str = (AT <: AbstractGPUArray ? "$AT" : "CPU $n_threads thread(s)")
            cutoff_str = Base.typename(typeof(cutoff)).wrapper
            @info "$platform_str - $cutoff_str - max energy difference $max_ΔE"
            @test max_ΔE < 5e-4u"kJ * mol^-1"

            final_coords = last(values(sys.loggers.coords))
            @test all(all(c .> 0.0u"nm") for c in final_coords)
            @test all(all(c .< boundary) for c in final_coords)
        end
    end

    test_energy_conservation(true , Array, 1, 10_000)
    test_energy_conservation(false, Array, 1, 10_000)
    if Threads.nthreads() > 1
        test_energy_conservation(true , Array, Threads.nthreads(), 50_000)
        test_energy_conservation(false, Array, Threads.nthreads(), 50_000)
    end
    if CUDA.functional()
        test_energy_conservation(true , CuArray, 1, 100_000)
        test_energy_conservation(false, CuArray, 1, 100_000)
    end
    if AMDGPU.functional()
        test_energy_conservation(true , ROCArray, 1, 100_000)
        test_energy_conservation(false, ROCArray, 1, 100_000)
    end
end

@testset "ReaxFF energy conservation" begin
    data_dir = normpath(@__DIR__, "..", "..", "data")

    for array_type in (CUDA.functional() ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "$device" begin
            float_type = Float64
            qeq_float_type = Float64
            qeq_tolerance = 1e-12

            reaxff = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Monti_2013_CHONSMgPNaCuCl.ff"),
                float_type=float_type,
                array_type=array_type,
                smoothness_correction=true
            )

            sys = System(
                joinpath(data_dir, "ReaxFF", "data", "crambin.pdb"),
                reaxff,
                boundary=CubicBoundary(float_type(5.3004)u"nm"),
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                loggers=(energy=TotalEnergyLogger(100),)
            )

            temp = 300.0u"K"
            minimizer = SteepestDescentMinimizer()
            simulate!(sys, minimizer; run_loggers=false, show_progress=false)
            Random.seed!(2024)
            random_velocities!(sys, temp)
            remove_CM_motion!(sys)

            simulator = VelocityVerlet(
                dt=0.1u"fs",
                remove_CM_motion=false
            )

            # 10 ps equilibration
            simulate!(sys, simulator, 100_000; run_loggers=false, show_progress=false)
            E0 = total_energy(sys)

            n_steps = array_type === Array ? 700_000 : 20_000_000
            simulate!(sys, simulator, n_steps; show_progress=false)

            E = values(sys.loggers.energy)
            max_ΔE = maximum(abs.(E .- E0))
            @test max_ΔE < 10.0u"kJ * mol^-1"
        end
    end
end
