using Molly
using CUDA

using Test

@testset "Lennard-Jones energy conservation" begin
    function test_energy_conservation(gpu::Bool, n_threads::Integer, n_steps::Integer)
        n_atoms = 2_000
        atom_mass = 40.0u"u"
        temp = 1.0u"K"
        boundary = CubicBoundary(50.0u"nm")
        simulator = VelocityVerlet(dt=0.001u"ps", remove_CM_motion=false)

        atoms = [Atom(charge=0.0, mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
        dist_cutoff = 3.0u"nm"
        cutoffs = (
            DistanceCutoff(dist_cutoff),
            ShiftedPotentialCutoff(dist_cutoff),
            ShiftedForceCutoff(dist_cutoff),
            CubicSplineCutoff(dist_cutoff, dist_cutoff + 0.5u"nm"),
        )

        for cutoff in cutoffs
            coords = place_atoms(n_atoms, boundary; min_dist=0.6u"nm")

            sys = System(
                atoms=(gpu ? CuArray(atoms) : atoms),
                coords=(gpu ? CuArray(coords) : coords),
                boundary=boundary,
                pairwise_inters=(LennardJones(cutoff=cutoff, use_neighbors=false),),
                loggers=(
                    coords=CoordinateLogger(100),
                    energy=TotalEnergyLogger(100),
                ),
            )
            random_velocities!(sys, temp)

            E0 = total_energy(sys; n_threads=n_threads)
            simulate!(deepcopy(sys), simulator, 20; n_threads=n_threads)
            @time simulate!(sys, simulator, n_steps; n_threads=n_threads)

            Es = values(sys.loggers.energy)
            @test Es[1] == E0

            max_ΔE = maximum(abs.(Es .- E0))
            platform_str = gpu ? "GPU" : "CPU $n_threads thread(s)"
            cutoff_str = Base.typename(typeof(cutoff)).wrapper
            @info "$platform_str - $cutoff_str - max energy difference $max_ΔE"
            @test max_ΔE < 5e-4u"kJ * mol^-1"

            final_coords = last(values(sys.loggers.coords))
            @test all(all(c .> 0.0u"nm") for c in final_coords)
            @test all(all(c .< boundary) for c in final_coords)
        end
    end

    test_energy_conservation(false, 1, 10_000)
    if Threads.nthreads() > 1
        test_energy_conservation(false, Threads.nthreads(), 50_000)
    end
    if CUDA.functional()
        test_energy_conservation(true, 1, 100_000)
    end
end
