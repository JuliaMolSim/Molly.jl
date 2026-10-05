# Figures for the Allegro benchmarks: the native-Molly vs nequip-allegro vs allegro-jax head-to-head.
# Reads the JSON written by benchmark/allegro.jl (Molly), benchmark/allegro_package_bench.py
# (nequip-allegro) and benchmark/allegro_jax_bench.py (allegro-jax) from benchmark/results/ and writes
# ~150 dpi PNGs to benchmark/images/. Self-contained: any missing series is skipped.
#
#   julia --project=<env-with-CairoMakie+JSON3> benchmark/allegro_plots.jl
#
# Molly runs: results/allegro_bench_apple.json (CPU/Metal, Apple) + results/allegro_bench_cyclops.json
# (CUDA/CPU, RTX 5080 box); each key -> {"n<N>": {ms_energy, ms_energy_forces}}. nequip-allegro:
# results/allegro_torch_{cuda,cpu}.json; allegro-jax: results/allegro_jax_{cuda,cpu}.json; each key
# (cuda / cpu_t8) -> {"<N>": {energy_ms, forces_ms}}. Metal = Apple, the rest = RTX 5080 host, so these
# are cross-machine: read the scaling shape and within-machine GPU-over-CPU speedup, not absolute level.
# Colour encodes backend, linestyle encodes implementation (Molly solid, nequip dashed, jax dotted).

using CairoMakie, JSON3
CairoMakie.activate!(type = "png")

const RES = joinpath(@__DIR__, "results")
const IMG = joinpath(@__DIR__, "images")
mkpath(IMG)

load_json(p) = isfile(p) ? JSON3.read(read(p, String)) : nothing
getk(d, k)   = isnothing(d) ? nothing : get(d, k, nothing)

const APPLE   = load_json(joinpath(RES, "allegro_bench_apple.json"))
const CYCLOPS = load_json(joinpath(RES, "allegro_bench_cyclops.json"))
const NQ_CUDA = getk(load_json(joinpath(RES, "allegro_torch_cuda.json")), "cuda")
const NQ_CPU  = getk(load_json(joinpath(RES, "allegro_torch_cpu.json")),  "cpu_t8")
const JX_CUDA = getk(load_json(joinpath(RES, "allegro_jax_cuda.json")), "cuda")
const JX_CPU  = getk(load_json(joinpath(RES, "allegro_jax_cpu.json")),  "cpu_t8")

# Molly sub-dict: {"n<N>": {ms_energy|ms_energy_forces}}. ref (nequip/jax): {"<N>": {energy_ms|forces_ms}}.
function mseries(d, field)
    isnothing(d) && return (Int[], Float64[])
    ns = sort([parse(Int, replace(string(k), "n" => "")) for k in keys(d)])
    (ns, [Float64(d["n$n"][field]) for n in ns])
end
function rseries(d, field)
    isnothing(d) && return (Int[], Float64[])
    ns = sort([parse(Int, string(k)) for k in keys(d)])
    (ns, [Float64(d[string(n)][field]) for n in ns])
end

function overlay_plot(title, out, specs)
    fig = Figure(size = (960, 640))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "time (ms)", title = title)
    plotted = false
    for (lbl, col, ls, (xs, ys)) in specs
        isempty(xs) && continue
        lw = ls == :solid ? 3.4 : 1.8          # Molly (solid) stands out against the reference impls
        ms = ls == :solid ? 12 : 8
        scatterlines!(ax, xs, ys, label = lbl, markersize = ms, linewidth = lw, color = col, linestyle = ls)
        plotted = true
    end
    plotted || return
    axislegend(ax, position = :lt, labelsize = 11, nbanks = 1)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end

# --- all implementations: energy, then energy + forces -----------------------------------------
for (field_m, field_r, titl,            out) in [
    ("ms_energy",        "energy_ms", "Allegro energy: all implementations (Metal = Apple, rest = RTX 5080)", "allegro_benchmark_energy.png"),
    ("ms_energy_forces", "forces_ms", "Allegro energy + forces: all implementations (Metal = Apple, rest = RTX 5080)", "allegro_benchmark_force.png"),
]
    overlay_plot(titl, out, [
        ("Molly CUDA (RTX 5080)", :seagreen,   :solid, mseries(getk(CYCLOPS, "native_cuda"),   field_m)),
        ("Molly Metal (Apple)",   :purple,     :solid, mseries(getk(APPLE,   "native_metal"),  field_m)),
        ("Molly CPU t8",          :navy,       :solid, mseries(getk(CYCLOPS, "native_cpu_t8"), field_m)),
        ("nequip-allegro CUDA",   :darkorange, :dash,  rseries(NQ_CUDA, field_r)),
        ("nequip-allegro CPU t8", :crimson,    :dash,  rseries(NQ_CPU,  field_r)),
        ("allegro-jax CUDA",      :teal,       :dot,   rseries(JX_CUDA, field_r)),
        ("allegro-jax CPU t8",    :goldenrod,  :dot,   rseries(JX_CPU,  field_r)),
    ])
end

# --- GPU speedup over host CPU-t8 (each GPU over its OWN host's CPU-t8), energy and forces -------
function gpu_speedup_plot(title, out, pairs)
    fig = Figure(size = (900, 600))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "GPU speedup over host CPU-t8 (×)", title = title)
    plotted = false
    for (lbl, col, ls, (xc, yc), (xg, yg)) in pairs
        (isempty(xc) || isempty(xg)) && continue
        common = sort(collect(intersect(xc, xg))); isempty(common) && continue
        cpu = Dict(xc .=> yc); gpu = Dict(xg .=> yg)
        scatterlines!(ax, common, [cpu[x] / gpu[x] for x in common], label = lbl,
                      markersize = 11, linewidth = 2.6, color = col, linestyle = ls)
        plotted = true
    end
    plotted || return
    hlines!(ax, [1.0], color = :gray, linestyle = :dash)
    axislegend(ax, position = :lt, labelsize = 11)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
for (fm, fr, titl, out) in [
    ("ms_energy",        "energy_ms", "Allegro energy: GPU speedup over host CPU (t8)",         "allegro_energy_gpu_speedup.png"),
    ("ms_energy_forces", "forces_ms", "Allegro energy + forces: GPU speedup over host CPU (t8)", "allegro_forces_gpu_speedup.png"),
]
    gpu_speedup_plot(titl, out, [
        ("Molly CUDA / CPU-t8 (RTX 5080)", :seagreen,   :solid, mseries(getk(CYCLOPS, "native_cpu_t8"), fm), mseries(getk(CYCLOPS, "native_cuda"), fm)),
        ("Molly Metal / CPU-t8 (Apple)",   :purple,     :solid, mseries(getk(APPLE,   "native_cpu_t8"), fm), mseries(getk(APPLE,   "native_metal"), fm)),
        ("nequip CUDA / CPU-t8",           :darkorange, :dash,  rseries(NQ_CPU, fr), rseries(NQ_CUDA, fr)),
        ("allegro-jax CUDA / CPU-t8",      :teal,       :dot,   rseries(JX_CPU, fr), rseries(JX_CUDA, fr)),
    ])
end

println("done — images in ", IMG)
