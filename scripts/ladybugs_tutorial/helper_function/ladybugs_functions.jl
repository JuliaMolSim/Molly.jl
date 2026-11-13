# The functions of LaDyBUGS (functions.py of the paper's repository), with Molly's MBAR in place of
#   FastMBAR. Energies and biases are in kJ/mol, `beta` in mol/kJ, states are numbered from 1.

using Molly: iterate_mbar, mbar_iteration, mbar_logD, logsumexp
using StatsBase: sample, Weights
using DelimitedFiles
using Printf

# ----------------------------------------------------------------------------
# The part of the hybrid that depends on λ, as Molly's AlchemicalPartition splits it off: only the
#   pairs with an alchemical atom (and all bonded terms). Its energy differs from the full one by
#   the environment-environment pairs, which are the same in every state.
function alchemical_part(sys, groups)
    eligible, special = Molly.neighbor_finder_masks(sys.neighbor_finder, length(sys))
    env = setdiff(1:length(sys), alchemical_atoms(groups))
    eligible[env, env] .= false
    AT = Molly.array_type(sys)
    nf = Molly.build_neighbor_finder(sys.neighbor_finder, Molly.to_device(eligible, AT),
                                     Molly.to_device(special, AT); boundary=sys.boundary)
    return System(sys; neighbor_finder=nf, loggers=(),
                  general_inters=filter(inter -> !(inter isa BiasPotential), sys.general_inters))
end
# ----------------------------------------------------------------------------
# The energy of every lambda state at the current coordinates of `sys`, kJ/mol: that of the λ part
#   `alch`, so without the pairs within the environment, as the alchemical force group of the paper
function state_energies!(alch, sys, groups, Lstates)
    alch.coords, alch.boundary = sys.coords, sys.boundary
    neighbors = find_neighbors(alch)
    return map(axes(Lstates, 1)) do lam
        set_lambdas!(alch, groups, Lstates[lam, :])
        Float64(ustrip(u"kJ * mol^-1", potential_energy(alch, neighbors)))
    end
end
# ----------------------------------------------------------------------------
function write_row(fp, fmt, values)
    for v in values
        Printf.format(fp, fmt, v)
    end
    write(fp, "\n")
    flush(fp)
end
# ----------------------------------------------------------------------------
function sampleLambda(reducedEnergy)
    prob = exp.(-(reducedEnergy .- minimum(reducedEnergy)))
    prob ./= sum(prob)
    return (idx=sample(1:length(prob), Weights(prob)), prob=prob)
end
# ----------------------------------------------------------------------------
# Counts per target weight w (mean 1): states with a larger weight can be visited more before their bias rises
function update_exp_biases!(biases, bias_fill, counts, exp_base, exp_mult, w=ones(length(counts)))
    lowest = minimum(counts ./ w)
    for i in eachindex(biases)
        biases[i] = bias_fill[i] + exp_mult * exp_base^(counts[i] / w[i] - lowest)
    end
end
# ----------------------------------------------------------------------------
# Free energies of states without samples, on the same scale as `F` of the sampled states
#   (Shirts and Chodera 2008, Eq. 15). `u` and `u_unsampled` are frames × states.
function perturbed_free_energies(u, u_unsampled, F, logN)
    logD = mbar_logD(u, F, logN)
    return [-logsumexp(-u_unsampled[:, k] .- logD) for k in axes(u_unsampled, 2)]
end
# ----------------------------------------------------------------------------
function calcFastMBAR(lambstates, beta, numCycle, numBiasLoops)
    println("Running MBAR after ", numCycle * numBiasLoops, " steps")

    # load in the counts and energies: frames × states, which is the layout Molly wants
    files(name) = ["lambda_energies/$name.$loop.txt" for loop in 0:(numBiasLoops - 1)]
    counts = reduce(vcat, readdlm.(files("count"), Int))
    u = reduce(vcat, readdlm.(files("energy"), Float64)) .* beta
    rF = mbar_free_energies(u, counts, beta)

    open("mbar_results/mbar.$(numBiasLoops - 1).txt", "w") do fp
        for l in 1:lambstates
            @printf(fp, "%d %.6f\n", l, rF[l])
        end
    end

    # the new biases
    return -rF
end
# ----------------------------------------------------------------------------
# Free energies (kJ/mol, state 1 = 0) from reduced energies u (frames × states) and one-hot counts (frames × states)
function mbar_free_energies(u, counts, beta)
    lambstates = size(u, 2)
    N_k = vec(sum(counts, dims=1))

    # MBAR over the sampled states, the unsampled ones are perturbed states
    sampled, unsampled = findall(>(0), N_k), findall(==(0), N_k)
    new_index = zeros(Int, lambstates)
    new_index[sampled] = 1:length(sampled)
    win_of = [new_index[argmax(view(counts, n, :))] for n in axes(counts, 1)]

    u_sampled = u[:, sampled]
    F, logN = iterate_mbar(u_sampled, win_of, N_k[sampled])
    residual = maximum(abs, mbar_iteration(u_sampled, F, logN) .- F)
    residual > 1e-3 && @warn "MBAR is not converged, the last iteration changes F by $residual kT"

    Fcomplete = zeros(lambstates)
    Fcomplete[sampled] = F
    Fcomplete[unsampled] = perturbed_free_energies(u_sampled, u[:, unsampled], F, logN)

    # remove "reduction" and make the first value the relative 0.0
    rF = Fcomplete ./ beta
    return rF .- rF[1]
end
# ----------------------------------------------------------------------------
# A record of the settings a run was made with, written next to its results
function log_parameters(file, parameters)
    open(file, "w") do fp
        for (name, value) in parameters
            println(fp, rpad(name, 22), value)
        end
    end
end
