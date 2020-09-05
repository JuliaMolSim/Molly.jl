using Molly

using Base.Threads
using Statistics
using Test

@testset "Lennard-Jones gas energy conservation" begin
    temp = 1.
    timestep = 0.005
    n_steps = 10_000
    box_size = 50.0
    n_atoms = 2000
    m = 0.8 * box_size^3 / n_atoms

    parallel_list = nthreads() > 1 ? (false, true) : (false,)
    lj_potentials = (LennardJones{false}(ShiftedPotentialCutoff(3.), false),
                     LennardJones{true}(ShiftedPotentialCutoff(3.), false),
                     LennardJones{false}(ShiftedForceCutoff(3.), false),
                     LennardJones{true}(ShiftedForceCutoff(3.), false))

    for parallel in parallel_list
        @testset "$lj_potential" for lj_potential in lj_potentials
            s = Simulation(
                simulator=VelocityVerlet(),
                atoms=[Atom(attype="Ar", name="Ar", resnum=i, resname="Ar", charge=0.0,
                            mass=m, σ=0.3, ϵ=0.2) for i in 1:n_atoms],
                general_inters=(lj_potential,),
                coords=placeatoms(n_atoms, box_size, 0.6),
                velocities=[velocity(10.0, temp) .* 0.01 for i in 1:n_atoms],
                temperature=temp,
                box_size=box_size,
                loggers=Dict("coords" => CoordinateLogger(100),
                            "energy" => EnergyLogger(100)),
                timestep=timestep,
                n_steps=n_steps
            )

            E0 = energy(s)
            @time simulate!(s, parallel=parallel)

            ΔE = energy(s) - E0
            @test abs(ΔE) < 2e-2

            Es = s.loggers["energy"].energies
            maxΔE = maximum(abs.(Es .- E0))
            @test maxΔE < 2e-2

            @test abs(Es[end] - Es[1]) < 2e-2
            @test std(Es.-Es[1])/n_atoms < timestep^2

            final_coords = last(s.loggers["coords"].coords)
            @test minimum(minimum.(final_coords)) > 0.0
            @test maximum(maximum.(final_coords)) < box_size
        end
    end
end
