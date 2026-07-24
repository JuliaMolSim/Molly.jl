# Publication figures for the ANI benchmarks. Reads the JSON written by the timing scripts
# (benchmark/results/*.json) and the TorchANI reference JSON, and writes ~150 dpi PNGs to
# benchmark/images/. Self-contained: skips any figure whose input JSON is missing.
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

# Pull (sizes, mins) sorted by size from a {"<n>": {"min":..}} sub-dict.
function series(d)
    isnothing(d) && return (Int[], Float64[])
    ks = sort(parse.(Int, collect(string.(keys(d)))))
    (ks, [Float64(d[string(k)]["min"]) for k in ks])
end

# Pull (sizes, times) from a TorchANI reference JSON: {"sizes": {"<n>": {"energy_ms"/"forces_ms"}}}.
function series_torch(path, key)
    tj = load_json(path); (isnothing(tj) || !haskey(tj, "sizes")) && return (Int[], Float64[])
    ks = sort(parse.(Int, collect(string.(keys(tj["sizes"])))))
    (ks, [Float64(tj["sizes"][string(k)][key]) for k in ks])
end

forces      = load_json(joinpath(RES, "ani_forces.json"))
energy      = load_json(joinpath(RES, "ani_energy.json"))
cuda_energy = load_json(joinpath(RES, "ani_cuda_energy.json"))
cuda_forces = load_json(joinpath(RES, "ani_cuda_forces.json"))

# --- Figures: energy / forces vs N — every backend on one chart --------------------
# Molly CPU/Metal/CUDA + TorchANI CPU/CUDA. NB Molly CPU/Metal + TorchANI CPU are Apple Silicon,
# while Molly CUDA + TorchANI CUDA are the RTX 5080 host — cross-machine, so read the scaling shape
# rather than the absolute cross-device level. TorchANI series are dashed.
function vs_N_plot(mj, cj, tkey, quantity, out)
    isnothing(mj) && isnothing(cj) && return
    fig = Figure(size = (820, 560))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "number of atoms",
               ylabel = "$quantity time (ms)",
               title = "ANI-2x $quantity: Molly CPU/Metal/CUDA vs TorchANI CPU/CUDA")
    for (lbl, xy) in (("Molly CPU (t8)",  series(get(mj, "cpu", nothing))),
                      ("Molly Metal",     series(get(mj, "metal", nothing))),
                      ("Molly CUDA",      series(get(cj, "cuda", nothing))),
                      ("TorchANI CPU",    series_torch(joinpath(REF, "6mrr_timing_torchani_cpu.json"), tkey)),
                      ("TorchANI CUDA",   series_torch(joinpath(REF, "6mrr_timing_torchani_cuda.json"), tkey)))
        xs, ys = xy; isempty(xs) && continue
        scatterlines!(ax, xs, ys, label = lbl, markersize = 10,
                      linestyle = startswith(lbl, "TorchANI") ? :dash : :solid)
    end
    axislegend(ax, position = :lt)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
vs_N_plot(forces, cuda_forces, "forces_ms", "forces", "forces_vs_N.png")
vs_N_plot(energy, cuda_energy, "energy_ms", "energy", "energy_vs_N.png")

# --- Figure: GPU speedup over host CPU (t8) vs N — every backend in one plot --------
# One figure per quantity, overlaying each device's speedup over its own host CPU-t8. NB the
# Metal series' CPU baseline is Apple Silicon and the CUDA series' is the cyclops host, so each
# line is GPU-vs-its-own-host; compare the scaling shape, not the absolute CPU.
function speedup_multi(specs, title, out)
    fig = Figure(size = (760, 520))
    ax  = Axis(fig[1, 1], xscale = log10, xlabel = "number of atoms",
               ylabel = "GPU speedup over host CPU-t8 (×)", title = title)
    plotted = false
    for (data, cpukey, gpukey, lbl) in specs
        isnothing(data) && continue
        xc, yc = series(get(data, cpukey, nothing))
        xg, yg = series(get(data, gpukey, nothing))
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
