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

# --- Figure: forces vs N, Molly CPU (Enzyme) vs Metal (on-device) ------------------
forces = load_json(joinpath(RES, "ani_forces.json"))
if !isnothing(forces)
    fig = Figure(size = (760, 520))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10,
               xlabel = "number of atoms", ylabel = "forces time (ms)",
               title = "ANI-2x forces: Molly analytic (CPU + Metal) vs TorchANI CPU")
    xc, yc = series(get(forces, "cpu", nothing))
    xm, ym = series(get(forces, "metal", nothing))
    !isempty(xc) && scatterlines!(ax, xc, yc, label = "Molly CPU (analytic, t8)", markersize = 10)
    !isempty(xm) && scatterlines!(ax, xm, ym, label = "Molly Metal (analytic)", markersize = 10)
    # TorchANI forces timing, if the reference script was run. Format: {"sizes": {"<n>": {"forces_ms"}}}.
    let tj = load_json(joinpath(REF, "6mrr_timing_torchani_cpu.json"))   # TorchANI CPU (MPS unusable)
        if !isnothing(tj) && haskey(tj, "sizes")
            ks = sort(parse.(Int, collect(string.(keys(tj["sizes"])))))
            ys = [Float64(tj["sizes"][string(k)]["forces_ms"]) for k in ks]
            scatterlines!(ax, ks, ys, label = "TorchANI CPU", linestyle = :dash, markersize = 8)
        end
    end
    axislegend(ax, position = :lt)
    save(joinpath(IMG, "forces_vs_N.png"), fig, px_per_unit = 2)
    println("wrote images/forces_vs_N.png")
end

# --- Figure: energy vs N, CPU vs Metal ---------------------------------------------
energy = load_json(joinpath(RES, "ani_energy.json"))
if !isnothing(energy)
    fig = Figure(size = (760, 520))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10,
               xlabel = "number of atoms", ylabel = "energy time (ms)",
               title = "ANI-2x energy: Molly CPU vs Apple Metal")
    for (key, lbl) in (("cpu", "Molly CPU (t8)"), ("metal", "Molly Metal"))
        xs, ys = series(get(energy, key, nothing))
        !isempty(xs) && scatterlines!(ax, xs, ys, label = lbl, markersize = 10)
    end
    let tj = load_json(joinpath(REF, "6mrr_timing_torchani_cpu.json"))   # TorchANI CPU (MPS unusable)
        if !isnothing(tj) && haskey(tj, "sizes")
            ks = sort(parse.(Int, collect(string.(keys(tj["sizes"])))))
            ys = [Float64(tj["sizes"][string(k)]["energy_ms"]) for k in ks]
            scatterlines!(ax, ks, ys, label = "TorchANI CPU", linestyle = :dash, markersize = 8)
        end
    end
    axislegend(ax, position = :lt)
    save(joinpath(IMG, "energy_vs_N.png"), fig, px_per_unit = 2)
    println("wrote images/energy_vs_N.png")
end

# --- Figure: GPU speedup over host CPU (t8) vs N — every backend in one plot --------
# One figure per quantity, overlaying each device's speedup over its own host CPU-t8. NB the
# Metal series' CPU baseline is Apple Silicon and the CUDA series' is the cyclops host, so each
# line is GPU-vs-its-own-host; compare the scaling shape, not the absolute CPU.
cuda_energy = load_json(joinpath(RES, "ani_cuda_energy.json"))
cuda_forces = load_json(joinpath(RES, "ani_cuda_forces.json"))
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

# --- Figures: NVIDIA CUDA (RTX 5080) — Molly CPU vs CUDA vs TorchANI CUDA -----------
# From benchmark/ani_cuda_compare.jl ({"cpu","cuda"}) + the TorchANI CUDA reference JSON. The CPU
# column here is the cyclops host (broadwell, t8), not the Apple numbers used elsewhere in the doc.
function cuda_plot(res, torch_key, quantity, out)
    isnothing(res) && return
    fig = Figure(size = (760, 520))
    ax  = Axis(fig[1, 1], xscale = log10, yscale = log10,
               xlabel = "number of atoms", ylabel = "$quantity time (ms)",
               title = "ANI-2x $quantity: Molly CPU vs CUDA vs TorchANI (RTX 5080)")
    xc, yc = series(get(res, "cpu", nothing))
    xg, yg = series(get(res, "cuda", nothing))
    !isempty(xc) && scatterlines!(ax, xc, yc, label = "Molly CPU (t8)", markersize = 10)
    !isempty(xg) && scatterlines!(ax, xg, yg, label = "Molly CUDA", markersize = 10)
    let tj = load_json(joinpath(REF, "6mrr_timing_torchani_cuda.json"))
        if !isnothing(tj) && haskey(tj, "sizes")
            ks = sort(parse.(Int, collect(string.(keys(tj["sizes"])))))
            ys = [Float64(tj["sizes"][string(k)][torch_key]) for k in ks]
            scatterlines!(ax, ks, ys, label = "TorchANI CUDA", linestyle = :dash, markersize = 8)
        end
    end
    axislegend(ax, position = :lt)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
cuda_plot(cuda_energy, "energy_ms", "energy", "cuda_energy_vs_N.png")
cuda_plot(cuda_forces, "forces_ms", "forces", "cuda_forces_vs_N.png")

println("done — images in ", IMG)
