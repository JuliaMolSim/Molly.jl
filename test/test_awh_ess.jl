using Molly
using Test
using Statistics
using StaticArrays

@testset "AWH ESS and Autocorrelation Correction" begin
    n_atoms = 20
    n_steps = 1000
    atom_mass = 10.0u"g/mol"
    boundary = CubicBoundary(2.0u"nm")
    coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
    temp = 298.0u"K"

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    n_windows = 2
    λ_vals = [1.0, 0.5]
    
    thermo_states = ThermoState[]
    for i in 1:n_windows
        atoms_λ = [Atom(mass=atom_mass, charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1", 
                        λ = λ_vals[i]) for _ in 1:n_atoms]
        
        sys = System(
            atoms=atoms_λ,
            coords=coords,
            boundary=boundary,
            pairwise_inters=(LennardJonesSoftCoreBeutler(α=0.3, use_neighbors=true),),
            neighbor_finder=neighbor_finder
        )
        intg = Langevin(dt=0.005u"ps", temperature=temp, friction=0.1u"ps^-1")
        push!(thermo_states, ThermoState(sys, intg; temperature=temp))
    end

    awh_state = AWHState(thermo_states; first_state=1, n_bias=10)

    # Use a small log_freq to ensure we get some history
    awh_sim = AWHSimulation(
        awh_state;
        num_md_steps=10,
        update_freq=5,
        well_tempered_factor=10.0,
        coverage_threshold=1.0,
        log_freq=5
    )

    # Initial checks
    @test isempty(awh_sim.state.stats.ess_history)
    @test awh_sim.state.N_eff == 0.0

    # Run simulation
    simulate!(awh_sim, n_steps)

    # 1. Check if ess_history is populated
    stats = awh_sim.state.stats
    @test !isempty(stats.ess_history)
    @test length(stats.ess_history[1]) == n_windows
    
    # Kish ESS should be between 0 and update_freq (5 samples per update)
    # Actually, it's (sum w)^2 / sum w^2. For n samples, it's at most n.
    # Our update_freq is 5, so ESS per update block should be <= 5.
    @test all(0 .<= vcat(stats.ess_history...) .<= 5.1)

    # 2. Check N_eff growth
    # Without scaling, N_eff would be exactly n_iterations = 1000/10 = 100.
    # With scaling, N_eff = sum(1/g). Since g >= 1, N_eff should be <= 100.
    @test 0 < awh_sim.state.N_eff <= 100.1
    
    # 3. Check extract_awh_data
    data = extract_awh_data(awh_sim)
    @test haskey(data, :ess_history)
    @test data.ess_history == stats.ess_history
    
    println("AWH ESS history length: ", length(data.ess_history))
    println("Final N_eff: ", awh_sim.state.N_eff)
end
