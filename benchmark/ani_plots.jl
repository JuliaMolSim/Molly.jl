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
               title = "ANI-2x forces: Molly CPU (Enzyme) vs Apple Metal (on-device)")
    xc, yc = series(get(forces, "cpu_enzyme", nothing))
    xm, ym = series(get(forces, "metal", nothing))
    !isempty(xc) && scatterlines!(ax, xc, yc, label = "Molly CPU (Enzyme, t8)", markersize = 10)
    !isempty(xm) && scatterlines!(ax, xm, ym, label = "Molly Metal (analytic)", markersize = 10)
    # TorchANI forces timing, if the user ran the reference script.
    for (dev, lbl) in (("cpu", "TorchANI CPU"), ("mps", "TorchANI MPS"))
        tj = load_json(joinpath(REF, "6mrr_timing_torchani_$(dev).json"))
        isnothing(tj) && continue
        xs = Int[]; ys = Float64[]
        for e in tj
            (haskey(e, "n_atoms") && haskey(e, "forces_ms")) || continue
            push!(xs, Int(e["n_atoms"])); push!(ys, Float64(e["forces_ms"]))
        end
        !isempty(xs) && scatterlines!(ax, xs, ys, label = lbl, linestyle = :dash, markersize = 8)
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
    axislegend(ax, position = :lt)
    save(joinpath(IMG, "energy_vs_N.png"), fig, px_per_unit = 2)
    println("wrote images/energy_vs_N.png")
end

# --- Figure: CPU→Metal speedup vs N (forces + energy) ------------------------------
function speedup_plot(data, cpukey, gpukey, title, out)
    isnothing(data) && return
    xc, yc = series(get(data, cpukey, nothing))
    xm, ym = series(get(data, gpukey, nothing))
    common = intersect(xc, xm)
    isempty(common) && return
    cpu = Dict(xc .=> yc); gpu = Dict(xm .=> ym)
    xs = sort(collect(common)); sp = [cpu[x] / gpu[x] for x in xs]
    fig = Figure(size = (720, 480))
    ax  = Axis(fig[1, 1], xscale = log10, xlabel = "number of atoms",
               ylabel = "CPU / Metal speedup (×)", title = title)
    scatterlines!(ax, xs, sp, markersize = 10)
    hlines!(ax, [1.0], color = :gray, linestyle = :dash)
    save(joinpath(IMG, out), fig, px_per_unit = 2)
    println("wrote images/", out)
end
speedup_plot(forces, "cpu_enzyme", "metal", "ANI-2x forces: CPU→Metal speedup", "forces_speedup.png")
speedup_plot(energy, "cpu", "metal", "ANI-2x energy: CPU→Metal speedup", "energy_speedup.png")

println("done — images in ", IMG)
