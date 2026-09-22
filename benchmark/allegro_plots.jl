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

# --- all backends: energy vs N (CPU t1/t8 + Metal on Apple M3, CUDA on RTX 5080) ----------------
# CPU + Metal are Apple M3; CUDA is the RTX 5080 host — cross-machine, so read the scaling shape,
# not the absolute cross-device level. Metal is Float32, CPU/CUDA Float64.
metal = load_json(joinpath(RES, "allegro_metal_energy.json"))   # cpu_t1, cpu_t8, metal (Apple M3)
cuda  = load_json(joinpath(RES, "allegro_cuda_energy.json"))    # cpu_t1, cpu_t8, cuda (RTX 5080)
vs_N_plot("Allegro energy: CPU (t1/t8) vs Metal vs CUDA", "allegro_backends_energy_vs_N.png", [
    ("CPU t1 (M3)",     :royalblue,  :solid, series(getk(metal, "cpu_t1"))),
    ("CPU t8 (M3)",     :navy,       :solid, series(getk(metal, "cpu_t8"))),
    ("Metal (M3)",      :darkorange, :solid, series(getk(metal, "metal"))),
    ("CUDA (RTX 5080)", :seagreen,   :solid, series(getk(cuda,  "cuda"))),
])

# --- GPU speedup over host CPU-t8 (each backend over its OWN machine's CPU-t8) ------------------
function speedup_over_cpu8(pairs, out)
    fig = Figure(size = (780, 540))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "GPU speedup over host CPU-t8 (×)",
               title = "Allegro energy: GPU speedup over host CPU (t8)")
    plotted = false
    for (lbl, col, (xc, yc), (xg, yg)) in pairs
        (isempty(xc) || isempty(xg)) && continue
        common = sort(collect(intersect(xc, xg))); isempty(common) && continue
        cpu = Dict(xc .=> yc); gpu = Dict(xg .=> yg)
        scatterlines!(ax, common, [cpu[x]/gpu[x] for x in common], label = lbl,
                      markersize = 10, color = col)
        plotted = true
    end
    plotted || return
    hlines!(ax, [1.0], color = :gray, linestyle = :dash)
    axislegend(ax, position = :lt)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
speedup_over_cpu8([
    ("Metal / CPU-t8 (M3)",       :darkorange, series(getk(metal, "cpu_t8")), series(getk(metal, "metal"))),
    ("CUDA / CPU-t8 (RTX 5080)",  :seagreen,   series(getk(cuda,  "cpu_t8")), series(getk(cuda,  "cuda"))),
], "allegro_gpu_speedup.png")

# --- head-to-head: Molly native vs the real nequip-allegro (PyTorch), same RTX 5080 host ----------
# Comparable-size Allegro (l_max=2, 2 layers, ~4 tensor / 16 scalar channels), NOT identical ops.
# Molly series read "min"; nequip series read "energy_ms". Molly solid, nequip dashed.
function series_key(d, k)
    isnothing(d) && return (Int[], Float64[])
    ks = sort(parse.(Int, collect(string.(keys(d)))))
    (ks, [Float64(d[string(k2)][k]) for k2 in ks])
end
nq_cuda = load_json(joinpath(RES, "allegro_torch_cuda.json"))
nq_cpu  = load_json(joinpath(RES, "allegro_torch_cpu.json"))
if !isnothing(nq_cuda) && !isnothing(cuda)
    fig = Figure(size = (860, 580))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "energy time (ms)",
               title = "Allegro energy: Molly vs nequip-allegro (comparable model; Metal on M3, rest on RTX 5080)")
    # nequip-allegro has NO Apple GPU path (it requires float64; MPS is float32-only), so Metal is a
    # Molly-only series. CPU/CUDA are the RTX 5080 host; Molly Metal is the Apple M3 (cross-machine).
    specs = [
        ("Molly CUDA (RTX 5080)",       :seagreen,   :solid,   series(getk(cuda, "cuda"))),
        ("Molly Metal (M3)",            :purple,     :solid,   series(getk(metal, "metal"))),
        ("Molly CPU t8",                :navy,       :solid,   series(getk(cuda, "cpu_t8"))),
        ("nequip-allegro CUDA (eager)", :darkorange, :dash,    series_key(getk(nq_cuda, "cuda"), "energy_ms")),
        ("nequip-allegro CUDA (compiled)", :goldenrod, :dashdot, series_key(getk(nq_cuda, "cuda_c"), "energy_ms")),
        ("nequip-allegro CPU t8",       :crimson,    :dash,    series_key(getk(nq_cpu, "cpu_t8"), "energy_ms")),
    ]
    for (lbl, col, ls, (xs, ys)) in specs
        isempty(xs) && continue
        scatterlines!(ax, xs, ys, label = lbl, markersize = 9, color = col, linestyle = ls)
    end
    axislegend(ax, position = :lt, labelsize = 11)
    save(joinpath(IMG, "allegro_vs_nequip_energy.png"), fig, px_per_unit = 2)
    println("wrote images/allegro_vs_nequip_energy.png")
end

println("done — images in ", IMG)
