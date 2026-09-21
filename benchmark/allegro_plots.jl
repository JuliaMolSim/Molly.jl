# Figures for the Allegro benchmarks. Reads the JSON written by benchmark/allegro.jl
# (benchmark/results/allegro_*.json) and writes ~150 dpi PNGs to benchmark/images/.
# Self-contained: skips any series whose input JSON is missing.
#
#   julia --project=<env> benchmark/allegro_plots.jl
# Needs CairoMakie + JSON3 in the environment.

using CairoMakie, JSON3
CairoMakie.activate!(type = "png")

const RES = joinpath(@__DIR__, "results")
const IMG = joinpath(@__DIR__, "images")
mkpath(IMG)

load_json(p) = isfile(p) ? JSON3.read(read(p, String)) : nothing
getk(d, k)   = isnothing(d) ? nothing : get(d, k, nothing)

# (sizes, mins) sorted by size from a {"<n>": {"min":..}} sub-dict.
function series(d)
    isnothing(d) && return (Int[], Float64[])
    ks = sort(parse.(Int, collect(string.(keys(d)))))
    (ks, [Float64(d[string(k)]["min"]) for k in ks])
end

energy   = load_json(joinpath(RES, "allegro_energy.json"))
forces   = load_json(joinpath(RES, "allegro_forces.json"))
fdforces = load_json(joinpath(RES, "allegro_fdforces.json"))

# --- energy / forces vs N ----------------------------------------------------------
function vs_N_plot(title, out, specs)
    fig = Figure(size = (820, 560))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "time (ms)", title = title)
    plotted = false
    for (lbl, col, ls, (xs, ys)) in specs
        isempty(xs) && continue
        scatterlines!(ax, xs, ys, label = lbl, markersize = 9, color = col, linestyle = ls)
        plotted = true
    end
    plotted || return
    axislegend(ax, position = :lt, labelsize = 11)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end

vs_N_plot("Allegro energy vs system size (native Julia CPU)", "allegro_energy_vs_N.png", [
    ("energy", :navy, :solid, series(getk(energy, "cpu"))),
])

vs_N_plot("Allegro forces vs system size: analytic vs finite differences", "allegro_forces_vs_N.png", [
    ("analytic (energy + forces)", :seagreen, :solid, series(getk(forces,   "cpu"))),
    ("finite differences (6N)",    :crimson,  :dash,  series(getk(fdforces, "cpu"))),
])

# --- analytic-vs-finite-diff speedup vs N ------------------------------------------
function speedup_plot(out)
    xf, yf = series(getk(forces,   "cpu"))
    xd, yd = series(getk(fdforces, "cpu"))
    (isempty(xf) || isempty(xd)) && return
    common = sort(collect(intersect(xf, xd))); isempty(common) && return
    ana = Dict(xf .=> yf); fd = Dict(xd .=> yd)
    sp = [fd[x] / ana[x] for x in common]
    fig = Figure(size = (760, 520))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "speedup (×)",
               title = "Allegro forces: analytic speedup over finite differences")
    scatterlines!(ax, common, sp, markersize = 11, color = :seagreen)
    hlines!(ax, [1.0], color = :gray, linestyle = :dash)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
speedup_plot("allegro_forces_speedup.png")

# --- CUDA vs CPU energy (from allegro_cuda_compare.jl, both series on the same box) -------------
cuda = load_json(joinpath(RES, "allegro_cuda_energy.json"))
if !isnothing(cuda)
    vs_N_plot("Allegro energy: CPU vs NVIDIA CUDA (RTX 5080)", "allegro_cuda_energy_vs_N.png", [
        ("CPU (1 thread)", :navy,     :solid, series(getk(cuda, "cpu"))),
        ("CUDA (RTX 5080)", :seagreen, :solid, series(getk(cuda, "cuda"))),
    ])
    xc, yc = series(getk(cuda, "cpu")); xg, yg = series(getk(cuda, "cuda"))
    common = sort(collect(intersect(xc, xg)))
    if !isempty(common)
        cpu = Dict(xc .=> yc); gpu = Dict(xg .=> yg)
        sp = [cpu[x] / gpu[x] for x in common]
        fig = Figure(size = (760, 520))
        ax  = Axis(fig[1, 1], xscale = log10, xlabel = "number of atoms",
                   ylabel = "CUDA speedup over CPU (×)",
                   title = "Allegro energy: CUDA speedup over CPU (RTX 5080)")
        scatterlines!(ax, common, sp, markersize = 11, color = :seagreen)
        hlines!(ax, [1.0], color = :gray, linestyle = :dash)
        save(joinpath(IMG, "allegro_cuda_speedup.png"), fig, px_per_unit = 2)
        println("wrote images/allegro_cuda_speedup.png")
    end
end

println("done — images in ", IMG)
