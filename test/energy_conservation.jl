using Molly

using Base.Threads
using Statistics
using Test

@testset "Lennard-Jones energy conservation" begin
    temp = 1.0u"K"
    n_steps = 10_000
    boundary = CubicBoundary(50.0u"nm", 50.0u"nm", 50.0u"nm")
    n_atoms = 2_000
    atom_mass = 40.0u"u"
    simulator = VelocityVerlet(dt=0.005u"ps")

    n_threads_list = Threads.nthreads() > 1 ? (1, Threads.nthreads()) : (1,)
    lj_potentials = (
        LennardJones(cutoff=DistanceCutoff(        3.0u"nm"), nl_only=false),
        LennardJones(cutoff=ShiftedPotentialCutoff(3.0u"nm"), nl_only=false),
        LennardJones(cutoff=ShiftedForceCutoff(    3.0u"nm"), nl_only=false),
    )

    for n_threads in n_threads_list
        @testset "$lj_potential" for lj_potential in lj_potentials
            s = System(
                atoms=[Atom(charge=0.0, mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms],
                pairwise_inters=(lj_potential,),
                coords=place_atoms(n_atoms, boundary, 0.6u"nm"),
                velocities=[velocity(atom_mass, temp) for i in 1:n_atoms],
                boundary=boundary,
                loggers=(
                    coords=CoordinateLogger(100),
                    energy=TotalEnergyLogger(100),
                ),
            )

            E0 = total_energy(s)
            @time simulate!(s, simulator, n_steps; n_threads=n_threads)

            ΔE = total_energy(s) - E0
            @test abs(ΔE) < 2e-2u"kJ * mol^-1"

            Es = values(s.loggers.energy)
            maxΔE = maximum(abs.(Es .- E0))
            @test maxΔE < 2e-2u"kJ * mol^-1"

            @test abs(Es[end] - Es[1]) < 2e-2u"kJ * mol^-1"

            final_coords = last(values(s.loggers.coords))
            @test all(all(c .> 0.0u"nm") for c in final_coords)
            @test all(all(c .< boundary) for c in final_coords)
        end
    end
end
