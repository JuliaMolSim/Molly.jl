# Lambda Dynamics with Bias Updated Gibbs Sampling + MBAR, after sampleSystem.py of
#   https://github.com/Vilseck-Lab/LaDyBUGS (openmm/3.ladybugs/example/run-00), for the MUP1 test
#   case of the paper: six 2-alkyl-thiazolines in one hybrid system, AMBER ff14SB + TIP3P with
#   GAFF 2.11 / AM1-BCC ligands, run with AMBER's own nonbonded settings rather than the paper's.
#   Energies and biases are in kJ/mol; lambda states are numbered from 1 and bias loops from 0,
#   as in the paper.
#
#   Nothing local is needed to run it: these lines put Molly and CairoMakie in a throwaway
#   environment and leave the results in the directory they are run from. The arguments are the
#   number of replicates (1 by default) and the device, gpu or cpu (gpu by default):
#
#     julia -t 8 -e 'using Pkg
#         Pkg.activate(temp=true)
#         Pkg.add(url="https://github.com/JuliaMolSim/Molly.jl", rev="master")  # or a fork, branch
#         Pkg.add(["CairoMakie", "CUDA", "StatsBase"])
#         using Molly
#         include(joinpath(pkgdir(Molly), "scripts", "ladybugs_tutorial",
#                          "ladybugs_tutorial.jl"))' 1 gpu
#
#   With the repository cloned already, make this directory the environment instead, once:
#
#     julia --project=. -e 'using Pkg; Pkg.develop(path="../.."); Pkg.instantiate()'
#
#   and then run it from wherever the results should go:
#
#     julia -t 8 --project=<this directory> <this directory>/ladybugs_tutorial.jl 1 gpu
#
#   -t 8 gives the threads of a cpu run, and of the parts that stay on the cpu in a gpu one. The
#   input files come from the mup1_data artifact of Molly, downloaded on the first run.
#
# Each replicate runs both legs, in <working directory>/complex<n> and solvent<n>, and the
#   comparison with the paper is plotted at the end. Expect a few hours on one GPU per leg.

using Molly
using CUDA
using Random
include(joinpath(@__DIR__, "helper_function", "multi_system.jl"))
include(joinpath(@__DIR__, "helper_function", "ladybugs_functions.jl"))
include(joinpath(@__DIR__, "helper_function", "plot_results.jl"))

## command line arguments
numReplica = parse(Int, get(ARGS, 1, "1"))   ## replicates, each of them running both legs
device     = get(ARGS, 2, "gpu")
device in ("gpu", "cpu") || error("the second argument is the device, gpu or cpu, not $device")

## requisite files. `mup1_data` is a lazy artifact of Molly, so the inputs are fetched the first
##   time the tutorial runs and then kept in the Julia depot.
data_dir     = mup1_data()
lstate_file  = joinpath(data_dir, "LambdaStates.txt")   # end states first
amber_dir    = joinpath(dirname(pathof(Molly)), "..", "data", "force_fields", "amber14")
ff_files     = joinpath.(amber_dir, ["protein.ff14SB.xml", "tip3p.xml"])
substituents = ["me", "eth", "ipr", "npr", "ibu", "rbut"]
main_dir     = pwd()                 # the runs and the plot go here

## MD parameters
FT       = Float32
AT       = device == "gpu" ? CuArray : Array
kb       = 0.0083144626            # kJ/(mol*K)
temp     = 298.15u"K"
pressure = FT(1.01325)u"bar"
stepsize = FT(0.002)u"ps"
fricCoef = FT(10.0)u"ps^-1"
beta     = 1 / (kb * ustrip(temp)) # mol/kJ

## LaDyBUGS sampling parameters  (this example will run for 5 ns)
numStepPerCycle = 100                   ## MD steps per Gibbs sampler step
numCycle        = 1000                  ## Gibbs sampler steps between bias updates
numBiasLoops    = 25                    ## bias updates with MBAR (every numCycle steps)
                                        ## total sampling = numStepPerCycle * numCycle * numBiasLoops * stepsize

# exponential bias function variables;  exp_bias = exp_mult * exp_base ^ (counts[i] - lowest_count)
start_bias = 10 * 4.184   ## kJ/mol, the linear bias step until the first MBAR biases
exp_base   = 2.0
exp_mult   = 1 * 4.184    ## kJ/mol (1 kcal/mol in the paper)

## The alchemical set-up. The core is the same in every ligand; the ring carbon C2 that follows it
##   has another name in each, and is treated as a core atom here. Angles and torsions with at
##   least one alchemical and one core atom scale with λ (scale_redundant), and of the dihedrals
##   the paper scales only N002/S001 - C2 - Cα - Cβ, listed per substituent as (C2, Cα, [Cβ...]).
core_names = ["S001", "N002", "C003", "H004", "H005", "C006", "H008", "H007"]
scale_redundant = true
dihedrals_to_scale = Dict(2 => ("C014", "C015", ["C018"]), 3 => ("C022", "C023", ["C025", "C029"]),
                          4 => ("C033", "C034", ["C037"]), 5 => ("C044", "C045", ["C048"]),
                          6 => ("C058", "C059", ["C061", "C065"]))
lj_softcore = :gapsys
c_softcore  = :gapsys

#######################
###### LaDyBUGS #######
#######################

for rep in 1:numReplica
    for leg in ["complex", "solvent"]
        ##### System and directory setup
        ## Creating folders for results
        cd(mkpath(joinpath(main_dir, leg * string(rep))))
        foreach(mkpath, ("lambda_energies", "mbar_results", "frames"))

        ## Load in pre-built lambda states
        Lstates = readdlm(lstate_file, FT)
        n_states, n_subs = size(Lstates)
        endstates = 1:n_subs

        println("\nThere are ", n_states, " lambda states\n")
        println("These are the endstates: ", endstates, "\n")

        ## Pick a random lambda to start with
        lambda_idx = rand(1:n_states)
        println("Starting Lambda IDX = ", lambda_idx)

        ## A record of the settings, next to the results
        log_parameters("parameters.txt",
                       ["leg" => leg, "replicate" => rep, "device" => device, "lambda states" => n_states,
                        "start lambda state" => lambda_idx, "MD steps per cycle" => numStepPerCycle,
                        "cycles per bias loop" => numCycle, "bias loops" => numBiasLoops,
                        "timestep" => stepsize, "temperature" => temp, "pressure" => pressure,
                        "friction" => fricCoef, "LJ softcore" => lj_softcore,
                        "Coulomb softcore" => c_softcore, "scale redundant" => scale_redundant,
                        "start bias" => start_bias, "exp base" => exp_base, "exp mult" => exp_mult])

        ## One system per substituent.
        function mup1_system(sub)
            pdb = joinpath(data_dir, leg == "complex" ? "mup1_$sub.pdb" : "mup1_$(sub)_solvent.pdb")
            ff = MolecularForceField(ff_files..., joinpath(data_dir, "mup1_$sub.xml");
                                    units=true, float_type=FT, strictness=:nowarn)
            # AMBER's own nonbonded scheme: PME, and the defaults for the rest - lj_cutoff falls back to
            #   DistanceCutoff(dist_cutoff) and dispersion_correction to the force field's own setting,
            #   which is true for ff14SB. MultiFESystem replaces Molly's correction with the one that
            #   knows substituents never see each other, and set_lambdas! recomputes it at every λ.
            return System(pdb, ff; array_type=AT, float_type=FT, dist_cutoff=FT(1.2)u"nm",
                        nonbonded_method=SetupPME(), constraints=:hbonds, rigid_water=true,
                        center_coords=false)
        end

        ## The inputs of MultiFESystem: which hybrid atoms are the shared core, which are unique to one
        ##   substituent, how the core of each system maps onto that of the first, and the dihedrals to scale
        function mup1_hybrid_inputs(systems)
            ligands = [[i for (i, d) in enumerate(sys.atoms_data) if d.res_name == "LIG"] for sys in systems]
            for (sys, lig) in zip(systems, ligands)
                @assert [sys.atoms_data[i].atom_name for i in lig[eachindex(core_names)]] == core_names
            end
            n_core = length(core_names) + 1          # the core atoms and C2
            mapping = Dict{String, Vector{Int}}("core" => ligands[1][1:n_core])
            for (n, lig) in enumerate(ligands)
                mapping["unique_sys$n"] = lig[n_core+1:end]
            end
            core_map = Dict(n => Dict(zip(ligands[1][1:n_core], ligands[n][1:n_core])) for n in 2:length(systems))
            scaled_torsions = Dict(map(collect(dihedrals_to_scale)) do (n, (c2_name, ca, cbs))
                index = Dict(systems[n].atoms_data[i].atom_name => i for i in ligands[n])
                n => [(index[a], index[c2_name], index[ca], index[cb]) for a in ("N002", "S001") for cb in cbs]
            end)
            return mapping, core_map, scaled_torsions
        end

        ## Setup the individual systems and then create the hybrid system
        systems = mup1_system.(substituents)
        mapping, core_map, scaled_torsions = mup1_hybrid_inputs(systems)
        sys, groups = MultiFESystem(systems, Lstates[lambda_idx, :], mapping, core_map; temp=temp,
                                    scheduler=ProbabilityLambdaScheduler(dual=true, Tscaled=false),
                                    LJsoftcore=lj_softcore, Csoftcore=c_softcore,
                                    scale_redundant=scale_redundant, scaled_torsions=scaled_torsions)
        alch = alchemical_part(sys, groups)
        simulator = Langevin(dt=stepsize, temperature=temp, friction=fricCoef, remove_CM_motion=false,
                            coupling=MonteCarloBarostat(pressure, temp, sys.boundary; n_steps=25))

        ## Open output file for writing
        foutput = open("output", "w")
        write(foutput, "Step energy lambda_index\n")

        # -----------------------------------
        ##### Minimize and Equilibrate
        write_structure("beforemin.pdb", sys)
        println("Initial energy is ", potential_energy(sys))
        for i in 1:5
            simulate!(sys, SteepestDescentMinimizer(max_steps=200))
            println("Energy after Min Loop ", i, " is ", potential_energy(sys))
        end
        println()
        write_structure("aftermin.pdb", sys)

        # initial equilibrium: 10 ps of sampling with 2 fs timestep
        random_velocities!(sys, temp)
        simulate!(sys, simulator, 5_000)
        write_structure("frames/traj.EQ.pdb", sys)

        # calc Es at every lambda state and sample lambdas at EQ coords
        biases    = zeros(n_states)
        bias_fill = zeros(n_states)     # the MBAR part of the biases, the floor of the exponential ones
        counts    = zeros(Int, n_states)
        linear    = true                # linear (pre-exponential) biases, until the first MBAR biases
        energies  = state_energies!(alch, sys, groups, Lstates)
        @printf(foutput, "EQ %f %d\n", energies[lambda_idx], lambda_idx)
        res       = sampleLambda((energies .+ biases) .* beta)

        # -----------------------------------
        ##### Start Production Sampling
        time_step = 0  # Gibbs step counter
        for loop in 0:(numBiasLoops - 1)
            # open bias, count, lambda, and energy files at the start of the loop
            bfp, cfp, efp, lfp = (open("lambda_energies/$name.$loop.txt", "w")
                                for name in ("bias", "count", "energy", "lambda"))

            t_loop, t_md, t_energies = time(), 0.0, 0.0   # wall time per part (GPU synchronised)
            for i in 0:(numCycle - 1)
                # set up new lambda state and sample for "numStepPerCycle" MD steps
                set_lambdas!(sys, groups, Lstates[res.idx, :])
                t_md += @elapsed (simulate!(sys, simulator, numStepPerCycle); AT == CuArray && CUDA.synchronize())

                # save counts and biases per step
                write_row(cfp, Printf.Format("%d "), Int.(1:n_states .== res.idx))
                write_row(bfp, Printf.Format("%f "), biases)

                # add bias before sampling lambda
                counts[res.idx] += 1
                time_step += 1
                if linear
                    biases[res.idx] += start_bias
                else
                    update_exp_biases!(biases, bias_fill, counts, exp_base, exp_mult)
                end

                # calc Es at every lambda state
                t_energies += @elapsed (energies = state_energies!(alch, sys, groups, Lstates))
                @printf(foutput, "%d %f %d\n", i, energies[res.idx], res.idx)
                flush(foutput)
                write_row(efp, Printf.Format("%f "), energies)
                write_row(lfp, Printf.Format("%.2f "), Lstates[res.idx, :])

                # sample lambdas at XYZ coords, from the reduced energies with biases
                res = sampleLambda((energies .+ biases) .* beta)
                println("Finished step ", i, " of loop ", loop, ": volume ", volume(sys.boundary),
                        ", temperature ", temperature(sys))
            end
            close.((bfp, cfp, efp, lfp))

            # print biases and counts at the end of production
            open("lambda_energies/total_BCs.$loop.txt", "w") do tfp
                for BC in 1:n_states
                    @printf(tfp, "%d %f %d\n", BC, biases[BC], counts[BC])
                end
            end

            # calculate MBAR free energies, load those in as new biases, and resample lambdas again
            t_cycles = time() - t_loop
            t_mbar = @elapsed (newbiases = calcFastMBAR(n_states, beta, numCycle, loop + 1))
            @printf(foutput, "timing loop %d (s per cycle): MD %.3f, state energies %.3f, rest %.3f; MBAR %.1f s\n",
                    loop, t_md / numCycle, t_energies / numCycle, (t_cycles - t_md - t_energies) / numCycle, t_mbar)
            linear = false
            bias_fill = copy(newbiases)
            update_exp_biases!(biases, bias_fill, counts, exp_base, exp_mult)
            res = sampleLambda((energies .+ biases) .* beta)   # energies of the last step, NEW biases

            write(foutput, "###### BIAS LOOP $loop FINISHED ######\n")
        end
        close(foutput)
    end
end

## Make a plot of the results in comparison with the paper: the ΔG of binding of every replicate
##   whose two legs have finished, averaged, so a replicate on its own gives no error bars.
cd(main_dir)
plot_mup1(main_dir)
