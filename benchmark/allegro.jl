# Allegro potential benchmarks (native Julia, CPU reference forward).
#
# Times three quantities across system sizes with the committed reference model
# (data/allegro_reference/allegro_model.h5, C=4 H=16 nb=8 layers=2 l_max=2 r_c=4 Å):
#   * energy             — allegro_total_energy
#   * energy + forces    — allegro_energy_and_forces (analytic: one taped forward + one backward)
#   * finite-diff forces — 6N energy evaluations (the approach the analytic backward replaced)
#
# Prints a Markdown table and writes JSON to benchmark/results/ for the plotting script:
#   results/allegro_energy.json     {"cpu": {"<n>": {min,median,iqr,bytes}}}
#   results/allegro_forces.json     {"cpu": {...}}   # analytic energy+forces
#   results/allegro_fdforces.json   {"cpu": {...}}   # finite differences (small N only)
#
# Standalone:
#   julia --project=<env-with-Molly+HDF5+JSON3> benchmark/allegro.jl
# Config via environment variables:
#   ALLEGRO_SIZES     comma list of atom counts        (default "16,32,64,128,256")
#   ALLEGRO_FD_MAX    largest N to also finite-diff     (default "64")
#   ALLEGRO_SPACING   lattice spacing in Å              (default "2.5")

using Molly, HDF5, Printf, Random
using Molly: SVector
try; using JSON3; catch; end

include(joinpath(@__DIR__, "allegro_bench_common.jl"))

const H5   = joinpath(@__DIR__, "..", "data", "allegro_reference", "allegro_model.h5")
const RES  = joinpath(@__DIR__, "results")

_sizes()   = parse.(Int, split(get(ENV, "ALLEGRO_SIZES", "16,32,64,128,256"), ","))
_fd_max()  = parse(Int, get(ENV, "ALLEGRO_FD_MAX", "64"))
_spacing() = parse(Float64, get(ENV, "ALLEGRO_SPACING", "2.5"))

# Jittered cubic lattice of n atoms (open boundary), random species in 1:S — realistic neighbour
# counts and no singular pair.
function make_system(n; a=_spacing(), jitter=0.2, S=2, seed=1)
    rng = MersenneTwister(seed)
    side = ceil(Int, cbrt(n))
    coords = SVector{3,Float64}[]
    for x in 0:side-1, y in 0:side-1, z in 0:side-1
        length(coords) == n && break
        push!(coords, SVector{3,Float64}(a*x, a*y, a*z) .+
              jitter .* (2 .* SVector{3,Float64}(rand(rng), rand(rng), rand(rng)) .- 1))
    end
    return coords, rand(rng, 1:S, n)
end

function fd_forces(m, coords, species, rc; h=1e-5)
    n = length(coords); cc = collect(SVector{3,Float64}, coords)
    F = Vector{SVector{3,Float64}}(undef, n)
    for i in 1:n
        g = zeros(3)
        for b in 1:3
            o = cc[i]
            cc[i] = SVector{3,Float64}(ntuple(k -> k==b ? o[k]+h : o[k], 3))
            Ep = Molly.allegro_total_energy(m, cc, species, nothing, rc)
            cc[i] = SVector{3,Float64}(ntuple(k -> k==b ? o[k]-h : o[k], 3))
            Em = Molly.allegro_total_energy(m, cc, species, nothing, rc)
            cc[i] = o; g[b] = -(Ep - Em) / (2h)
        end
        F[i] = SVector{3,Float64}(g...)
    end
    return F
end

function run_bench()
    if !isfile(H5)
        @warn "allegro_model.h5 not found — run test/allegro_reference.py first"; return
    end
    pot = AllegroPotential(H5; T=Float64)
    m = pot.model; rc = m.r_c
    sizes = _sizes(); fd_max = _fd_max()

    println("="^72)
    println("Allegro benchmark | ", run_header())
    println("model: C=$(m.C) H=$(m.H) nb=$(m.nb) layers=$(m.L) l_max=2 r_c=$(rc) Å")
    println("sizes=$(sizes) fd_max=$(fd_max)")
    println("="^72, "\n")

    E_res  = Dict{String,Any}("cpu" => Dict{String,Any}())
    F_res  = Dict{String,Any}("cpu" => Dict{String,Any}())
    FD_res = Dict{String,Any}("cpu" => Dict{String,Any}())

    @printf("| %6s | %8s | %13s | %15s | %17s | %9s |\n",
            "atoms", "edges", "energy (ms)", "E+forces (ms)", "fd forces (ms)", "speedup")
    @printf("|%s|%s|%s|%s|%s|%s|\n", "-"^8, "-"^10, "-"^15, "-"^17, "-"^19, "-"^11)
    for n in sizes
        coords, species = make_system(n)
        n_edges = sum(length, Molly.neighbour_lists(coords, nothing, rc))
        e  = bench(() -> Molly.allegro_total_energy(m, coords, species, nothing, rc))
        f  = bench(() -> Molly.allegro_energy_and_forces(m, coords, species, nothing, rc))
        E_res["cpu"][string(n)] = Dict("min"=>e.min, "median"=>e.median, "iqr"=>e.iqr, "bytes"=>e.bytes, "edges"=>n_edges)
        F_res["cpu"][string(n)] = Dict("min"=>f.min, "median"=>f.median, "iqr"=>f.iqr, "bytes"=>f.bytes, "edges"=>n_edges)
        fd_str = "—"; sp = "—"
        if n <= fd_max
            fd = bench(() -> fd_forces(m, coords, species, rc); repeats=3, samples=3)
            FD_res["cpu"][string(n)] = Dict("min"=>fd.min, "median"=>fd.median, "iqr"=>fd.iqr, "edges"=>n_edges)
            fd_str = @sprintf("%.2f", fd.min); sp = @sprintf("%.0f×", fd.min / f.min)
        end
        @printf("| %6d | %8d | %13.3f | %15.3f | %17s | %9s |\n", n, n_edges, e.min, f.min, fd_str, sp)
    end

    if isdefined(Main, :JSON3)
        write_json(joinpath(RES, "allegro_energy.json"),   merge(Dict("header"=>run_env()), E_res))
        write_json(joinpath(RES, "allegro_forces.json"),   merge(Dict("header"=>run_env()), F_res))
        write_json(joinpath(RES, "allegro_fdforces.json"), merge(Dict("header"=>run_env()), FD_res))
    end
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_bench()
end
