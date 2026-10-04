# Figures for the Allegro benchmarks. Reads the JSON written by benchmark/allegro.jl and
# benchmark/allegro_package_bench.py (benchmark/results/allegro_bench*.json) and writes ~150 dpi PNGs
# to benchmark/images/. Self-contained: any series whose input is missing is simply skipped.
#
#   julia --project=<env-with-CairoMakie+JSON3> benchmark/allegro_plots.jl
#
# Apple-machine runs (CPU t1/t8 + Metal) are read from results/allegro_bench_apple.json; the RTX 5080
# box runs (CUDA + CPU + the nequip-allegro package) from results/allegro_bench_cyclops.json. If only
# the single-machine results/allegro_bench.json exists it is used for both. Metal is Float32; the rest
# are Float64. CPU+Metal are Apple Silicon while CUDA/package are the RTX 5080 host, so the backend
# plot is cross-machine — read the scaling shape and the within-machine GPU-over-CPU speedup, not the
# absolute cross-device level.

using CairoMakie, JSON3
CairoMakie.activate!(type = "png")

const RES = joinpath(@__DIR__, "results")
const IMG = joinpath(@__DIR__, "images")
mkpath(IMG)

load_json(p) = isfile(p) ? JSON3.read(read(p, String)) : nothing
getk(d, k)   = isnothing(d) ? nothing : get(d, k, nothing)

const APPLE   = something(load_json(joinpath(RES, "allegro_bench_apple.json")),
                          load_json(joinpath(RES, "allegro_bench.json")))
const CYCLOPS = something(load_json(joinpath(RES, "allegro_bench_cyclops.json")),
                          load_json(joinpath(RES, "allegro_bench.json")), Dict())

# (atoms, ms) sorted by size from a {"n<N>": {"ms_energy_forces":..}} sub-dict.
function series(d)
    isnothing(d) && return (Int[], Float64[])
    ns = sort([parse(Int, replace(string(k), "n" => "")) for k in keys(d)])
    (ns, [Float64(d["n$n"]["ms_energy_forces"]) for n in ns])
end

# Molly (solid) lines are drawn thicker/larger so they stand out against the reference package.
function vs_N_plot(title, out, specs; legendpos = :lt)
    fig = Figure(size = (880, 580))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "energy + forces time (ms)", title = title)
    plotted = false
    for (lbl, col, ls, (xs, ys)) in specs
        isempty(xs) && continue
        lw = ls == :solid ? 3.4 : 1.8
        ms = ls == :solid ? 12 : 8
        scatterlines!(ax, xs, ys, label = lbl, markersize = ms, linewidth = lw, color = col, linestyle = ls)
        plotted = true
    end
    plotted || return
    axislegend(ax, position = legendpos, labelsize = 11)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end

# --- all backends: energy + forces vs N (CPU t1/t8 + Metal on Apple, CUDA on RTX 5080) ----------
vs_N_plot("Allegro energy + forces vs system size (Metal = Apple, CUDA = RTX 5080)",
          "allegro_backends_vs_N.png", [
    ("CPU t1 (Apple)",    :royalblue,  :solid, series(getk(APPLE,   "native_cpu_t1"))),
    ("CPU t8 (Apple)",    :navy,       :solid, series(getk(APPLE,   "native_cpu_t8"))),
    ("Metal (Apple)",     :darkorange, :solid, series(getk(APPLE,   "native_metal"))),
    ("CPU t8 (RTX host)", :slategray,  :solid, series(getk(CYCLOPS, "native_cpu_t8"))),
    ("CUDA (RTX 5080)",   :seagreen,   :solid, series(getk(CYCLOPS, "native_cuda"))),
])

# --- GPU speedup over host CPU-t8 (each backend over its OWN machine's CPU-t8) -------------------
function speedup_plot(out, pairs)
    fig = Figure(size = (820, 560))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "GPU speedup over host CPU-t8 (×)",
               title = "Allegro energy + forces: GPU speedup over host CPU (t8)")
    plotted = false
    for (lbl, col, (xc, yc), (xg, yg)) in pairs
        (isempty(xc) || isempty(xg)) && continue
        common = sort(collect(intersect(xc, xg))); isempty(common) && continue
        cpu = Dict(xc .=> yc); gpu = Dict(xg .=> yg)
        scatterlines!(ax, common, [cpu[x] / gpu[x] for x in common], label = lbl,
                      markersize = 11, linewidth = 2.6, color = col)
        plotted = true
    end
    plotted || return
    hlines!(ax, [1.0], color = :gray, linestyle = :dash)
    axislegend(ax, position = :lt, labelsize = 11)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
speedup_plot("allegro_gpu_speedup.png", [
    ("CUDA / CPU-t8 (RTX 5080)", :seagreen,   series(getk(CYCLOPS, "native_cpu_t8")), series(getk(CYCLOPS, "native_cuda"))),
    ("Metal / CPU-t8 (Apple)",   :darkorange, series(getk(APPLE,   "native_cpu_t8")), series(getk(APPLE,   "native_metal"))),
])

# --- native Molly vs the real nequip-allegro package, same hardware (RTX 5080 / its CPU) ---------
# Identical architecture + weights; Molly solid, package dashed. CUDA and CPU-t8 on the same box.
vs_N_plot("Allegro: native Molly vs nequip-allegro package (same hardware)",
          "allegro_vs_package.png", [
    ("Molly CUDA",      :seagreen,   :solid, series(getk(CYCLOPS, "native_cuda"))),
    ("package CUDA",    :darkorange, :dash,  series(getk(CYCLOPS, "package_cuda"))),
    ("Molly CPU t8",    :navy,       :solid, series(getk(CYCLOPS, "native_cpu_t8"))),
    ("package CPU t8",  :crimson,    :dash,  series(getk(CYCLOPS, "package_cpu_t8"))),
])

println("done — images in ", IMG)
