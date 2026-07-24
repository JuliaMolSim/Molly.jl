# Publication figures for the ANI benchmarks. Reads the JSON written by the timing scripts
# (benchmark/results/*.json) and the TorchANI / bio-mlff reference JSON, and writes ~150 dpi PNGs
# to benchmark/images/. Self-contained: skips any series whose input JSON is missing.
#
#   julia --project=<env> benchmark/ani_plots.jl
# Needs CairoMakie + JSON3 in the environment.

using CairoMakie, JSON3
CairoMakie.activate!(type = "png")

const RES = joinpath(@__DIR__, "results")
const IMG = joinpath(@__DIR__, "images")
const REF = joinpath(@__DIR__, "..", "data", "ani_reference")
mkpath(IMG)

load_json(p) = isfile(p) ? JSON3.read(read(p, String)) : nothing
getk(d, k) = isnothing(d) ? nothing : get(d, k, nothing)

# Pull (sizes, mins) sorted by size from a {"<n>": {"min":..}} sub-dict.
function series(d)
    isnothing(d) && return (Int[], Float64[])
    ks = sort(parse.(Int, collect(string.(keys(d)))))
    (ks, [Float64(d[string(k)]["min"]) for k in ks])
end

# Pull (sizes, times) from a TorchANI/bio-mlff JSON: {"sizes": {"<n>": {"energy_ms"/"forces_ms"}}}.
function series_ref(path, key)
    tj = load_json(path); (isnothing(tj) || !haskey(tj, "sizes")) && return (Int[], Float64[])
    ks = sort(parse.(Int, collect(string.(keys(tj["sizes"])))))
    (ks, [Float64(tj["sizes"][string(k)][key]) for k in ks])
end

energy      = load_json(joinpath(RES, "ani_energy.json"))       # Molly t8 CPU + Metal
forces      = load_json(joinpath(RES, "ani_forces.json"))
energy_t1   = load_json(joinpath(RES, "ani_energy_t1.json"))    # Molly t1 CPU
forces_t1   = load_json(joinpath(RES, "ani_forces_t1.json"))
cuda_energy = load_json(joinpath(RES, "ani_cuda_energy.json"))  # Molly CUDA (RTX 5080)
cuda_forces = load_json(joinpath(RES, "ani_cuda_forces.json"))

# --- Figures: energy / forces vs N — all implementations on one chart --------------
# Molly / TorchANI / bio-mlff, each CPU (t1 + t8) and GPU. Framework encoded by linestyle
# (Molly solid, TorchANI dashed, bio-mlff dotted). NB the CPU + Apple-Metal series are Apple
# Silicon (M3) while the CUDA series are the RTX 5080 host — cross-machine, so read the scaling
# shape rather than the absolute cross-device level.
const STYLE = Dict(
    "Molly CPU (t1)"    => (:royalblue,   :solid),
    "Molly CPU (t8)"    => (:navy,        :solid),
    "Molly Metal"       => (:darkorange,  :solid),
    "Molly CUDA"        => (:seagreen,    :solid),
    "TorchANI CPU (t1)" => (:orchid,      :dash),
    "TorchANI CPU (t8)" => (:crimson,     :dash),
    "TorchANI CUDA"     => (:deepskyblue, :dash),
    "bio-mlff CPU (t1)" => (:mediumpurple,:dot),
    "bio-mlff CPU (t8)" => (:purple,      :dot),
    "bio-mlff MPS"      => (:sienna,      :dot),
    "bio-mlff CUDA"     => (:teal,        :dot),
)

function vs_N_plot(quantity, tkey, out)
    mj    = quantity == "energy" ? energy      : forces
    mj_t1 = quantity == "energy" ? energy_t1   : forces_t1
    cj    = quantity == "energy" ? cuda_energy : cuda_forces
    specs = [
        ("Molly CPU (t1)",    series(getk(mj_t1, "cpu"))),
        ("Molly CPU (t8)",    series(getk(mj, "cpu"))),
        ("Molly Metal",       series(getk(mj, "metal"))),
        ("Molly CUDA",        series(getk(cj, "cuda"))),
        ("TorchANI CPU (t1)", series_ref(joinpath(REF, "6mrr_timing_torchani_cpu_t1.json"), tkey)),
        ("TorchANI CPU (t8)", series_ref(joinpath(REF, "6mrr_timing_torchani_cpu_t8.json"), tkey)),
        ("TorchANI CUDA",     series_ref(joinpath(REF, "6mrr_timing_torchani_cuda.json"), tkey)),
        ("bio-mlff CPU (t1)", series_ref(joinpath(REF, "biomlff_cpu_t1.json"), tkey)),
        ("bio-mlff CPU (t8)", series_ref(joinpath(REF, "biomlff_cpu_t8.json"), tkey)),
        ("bio-mlff MPS",      series_ref(joinpath(REF, "biomlff_mps.json"), tkey)),
        ("bio-mlff CUDA",     series_ref(joinpath(REF, "biomlff_cuda.json"), tkey)),
    ]
    fig = Figure(size = (900, 640))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "$quantity time (ms)",
               title = "ANI-2x $quantity: Molly vs TorchANI vs bio-mlff (CPU t1/t8 + GPU)")
    plotted = false
    for (lbl, (xs, ys)) in specs
        isempty(xs) && continue
        col, ls = STYLE[lbl]
        scatterlines!(ax, xs, ys, label = lbl, markersize = 8, color = col, linestyle = ls)
        plotted = true
    end
    plotted || return
    axislegend(ax, position = :lt, labelsize = 10, nbanks = 2)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
vs_N_plot("forces", "forces_ms", "forces_vs_N.png")
vs_N_plot("energy", "energy_ms", "energy_vs_N.png")

# --- Figure: GPU speedup over host CPU (t8) vs N — Metal + CUDA in one plot ---------
# Each device's speedup over its own host CPU-t8. NB the Metal series' CPU baseline is Apple
# Silicon and the CUDA series' is the cyclops host; compare the scaling shape, not the absolute CPU.
function speedup_multi(specs, title, out)
    fig = Figure(size = (760, 520))
    ax  = Axis(fig[1, 1], xscale = log10, xlabel = "number of atoms",
               ylabel = "GPU speedup over host CPU-t8 (×)", title = title)
    plotted = false
    for (data, cpukey, gpukey, lbl) in specs
        isnothing(data) && continue
        xc, yc = series(getk(data, cpukey))
        xg, yg = series(getk(data, gpukey))
        common = intersect(xc, xg)
        isempty(common) && continue
        cpu = Dict(xc .=> yc); gpu = Dict(xg .=> yg)
        xs = sort(collect(common)); sp = [cpu[x] / gpu[x] for x in xs]
        scatterlines!(ax, xs, sp, label = lbl, markersize = 10)
        plotted = true
    end
    plotted || return
    hlines!(ax, [1.0], color = :gray, linestyle = :dash)
    axislegend(ax, position = :lt)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
speedup_multi([(forces, "cpu", "metal", "Metal / CPU"), (cuda_forces, "cpu", "cuda", "CUDA / CPU")],
              "ANI-2x forces: GPU speedup over host CPU (t8)", "forces_speedup.png")
speedup_multi([(energy, "cpu", "metal", "Metal / CPU"), (cuda_energy, "cpu", "cuda", "CUDA / CPU")],
              "ANI-2x energy: GPU speedup over host CPU (t8)", "energy_speedup.png")

println("done — images in ", IMG)
