using CairoMakie
using DelimitedFiles
using LazyArtifacts
using Printf
using Statistics

# The tutorial's input files: a lazy artifact of Molly, bound in the Artifacts.toml at the root of
#   the repository, three directories up from this one
mup1_data() = LazyArtifacts.ensure_artifact_installed("mup1_data",
                            normpath(@__DIR__, "..", "..", "..", "Artifacts.toml"))

kcal = 4.184
hue  = (run="#2a78d6", paper15="#15996a", paper5="#c07c00")
ink, muted, grid, surface = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
band_fill = "#f1f0ed"

# The end states of the last MBAR result of one run, kcal/mol relative to the first end state
function end_state_dG(dir, n_sub)
    results = joinpath(dir, "mbar_results")
    files = isdir(results) ? filter(startswith("mbar."), readdir(results)) : String[]
    isempty(files) && return nothing
    last_loop = argmax(f -> parse(Int, split(f, ".")[2]), files)
    dG = readdlm(joinpath(results, last_loop))[1:n_sub, 2] ./ kcal
    return dG .- dG[1]
end

# The runs of one leg that have a result, as replicate number => dG
function leg_results(root, leg, n_sub)
    out = Dict{Int, Vector{Float64}}()
    for dir in readdir(root)
        rep = startswith(dir, leg) ? tryparse(Int, dir[nextind(dir, length(leg)):end]) : nothing
        isnothing(rep) && continue
        dG = end_state_dG(joinpath(root, dir), n_sub)
        isnothing(dG) || (out[rep] = dG)
    end
    return out
end

rmse(x, expt) = sqrt(mean((x .- expt) .^ 2))
kendall(x, expt) = mean(sign((x[i] - x[j]) * (expt[i] - expt[j]))
                        for i in eachindex(x) for j in 1:(i - 1))

function plot_mup1(root="."; out="mup1_results.pdf",
                   ref_file=joinpath(mup1_data(), "mup1_paper_results.txt"))
    ref = readdlm(ref_file, comments=true)
    subs, ligands = String.(ref[:, 1]), String.(ref[:, 2])
    expt, paper15, paper5 = (Float64.(ref[:, i]) for i in 3:5)

    cplx = leg_results(root, "complex", length(subs))
    solv = leg_results(root, "solvent", length(subs))
    reps = sort(collect(intersect(keys(cplx), keys(solv))))
    if isempty(reps)
        @warn "no replicate in $root has a result for both legs, so there is nothing to plot"
        return nothing
    end

    # dG of binding per replicate, each centred on the experimental mean as in Eq. 6 of the paper,
    #   then the mean over replicates and the standard error of that mean
    centre(d) = d .- mean(d) .+ mean(expt)
    per_rep = reduce(hcat, centre(cplx[r] .- solv[r]) for r in reps)
    ours = vec(mean(per_rep, dims=2))
    sem = length(reps) > 1 ? vec(std(per_rep, dims=2)) ./ sqrt(length(reps)) : zeros(length(subs))

    # Predicted against experimental, with the diagonal and a band of half a kcal/mol around it
    all_values = [expt; ours .- sem; ours .+ sem; paper15; paper5]
    lo, hi = minimum(all_values) - 0.4, maximum(all_values) + 0.4
    fig = Figure(size=(620, 620), backgroundcolor=surface)
    ax = Axis(fig[1, 1], backgroundcolor=surface, aspect=1,
              xlabel="experimental ΔG of binding  (kcal/mol)",
              ylabel="predicted ΔG of binding  (kcal/mol)",
              xlabelcolor=ink, ylabelcolor=ink, xgridcolor=grid, ygridcolor=grid,
              topspinevisible=false, rightspinevisible=false, leftspinecolor=grid,
              bottomspinecolor=grid, xticklabelcolor=muted, yticklabelcolor=muted,
              title="MUP1",
              titlecolor=ink, titlealign=:left, titlesize=25,)
    limits!(ax, lo, hi, lo, hi)
    band!(ax, [lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5], color=band_fill)
    lines!(ax, [lo, hi], [lo, hi], color=muted, linewidth=2)

    ours_label = "Molly implementation (mean ± sem)"
    for (name, y, colour, marker, filled, size) in
            ((ours_label, ours, hue.run, :circle, true, 12),
             ("paper, 15 ns", paper15, hue.paper15, :utriangle, true, 14),
             ("paper, 5 ns", paper5, hue.paper5, :utriangle, false, 13))
        label = @sprintf("%s  (RMSE %.2f, τ %.2f)", name, rmse(y, expt), kendall(y, expt))
        y === ours && length(reps) > 1 &&
            errorbars!(ax, expt, y, sem, color=colour, linewidth=2, whiskerwidth=7)
        scatter!(ax, expt, y, marker=marker, markersize=size, label=label,
                 color=(filled ? colour : surface), strokewidth=2,
                 strokecolor=(filled ? surface : colour))
    end
    # only this run is labelled per ligand, so the three markers of a ligand stay readable, and
    #   the label goes to the side of the point the diagonal leaves free
    for i in eachindex(subs)
        above = ours[i] > expt[i]
        text!(ax, expt[i], ours[i], text=subs[i], color=muted, fontsize=11,
              align=(above ? :right : :left, above ? :bottom : :top),
              offset=(above ? -9 : 9, above ? 6 : -6))
    end
    axislegend(ax, position=:lt, framevisible=false, labelcolor=muted, labelsize=12, rowgap=2)
    save(out, fig)
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_mup1(get(ARGS, 1, "."))
end
