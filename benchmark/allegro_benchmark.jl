# Benchmark for the native-Julia many-body Allegro potential (CPU reference forward).
#
# Times three quantities across a range of system sizes, using the committed reference model in
# data/allegro_reference/allegro_model.h5 (C=4, H=16, nb=8, L=2, l_max=2, r_c=4 Å):
#   * energy            — allegro_total_energy
#   * energy + forces   — allegro_energy_and_forces (analytic, one taped forward + one backward)
#   * finite-diff forces — 6N energy evaluations (the approach the analytic backward replaced)
#
# Systems are jittered cubic lattices (spacing 2.5 Å, open boundary) so the neighbour count per
# atom is realistic and no pair is singular. Run with a Molly-dev'd environment that also has HDF5:
#   julia --project=<env> benchmark/allegro_benchmark.jl
#
# Results are printed as a Markdown table (see benchmark/allegro_benchmark.md for a recorded run).

using Molly, HDF5, Printf
using Molly: SVector
using Random

const H5 = joinpath(@__DIR__, "..", "data", "allegro_reference", "allegro_model.h5")

# Jittered cubic lattice of n atoms, spacing a, random species in 1:S.
function make_system(n; a=2.5, jitter=0.2, S=2, seed=1)
    rng = MersenneTwister(seed)
    side = ceil(Int, cbrt(n))
    coords = SVector{3,Float64}[]
    for x in 0:side-1, y in 0:side-1, z in 0:side-1
        length(coords) == n && break
        push!(coords, SVector{3,Float64}(a*x, a*y, a*z) .+ jitter .* (2 .* SVector{3,Float64}(rand(rng), rand(rng), rand(rng)) .- 1))
    end
    species = rand(rng, 1:S, n)
    return coords, species
end

# Median wall time (seconds) of f() over `samples` runs after one warm-up.
function timeit(f; samples=5)
    f()  # warm up / compile
    ts = Float64[]
    for _ in 1:samples
        push!(ts, @elapsed f())
    end
    sort!(ts)
    return ts[cld(length(ts), 2)]
end

function fd_forces(m, coords, species, rc; h=1e-5)
    n = length(coords)
    cc = collect(SVector{3,Float64}, coords)
    F = Vector{SVector{3,Float64}}(undef, n)
    for i in 1:n
        g = zeros(3)
        for b in 1:3
            o = cc[i]
            cc[i] = SVector{3,Float64}(ntuple(k -> k==b ? o[k]+h : o[k], 3))
            Ep = Molly.allegro_total_energy(m, cc, species, nothing, rc)
            cc[i] = SVector{3,Float64}(ntuple(k -> k==b ? o[k]-h : o[k], 3))
            Em = Molly.allegro_total_energy(m, cc, species, nothing, rc)
            cc[i] = o
            g[b] = -(Ep - Em) / (2h)
        end
        F[i] = SVector{3,Float64}(g...)
    end
    return F
end

function main()
    pot = AllegroPotential(H5; T=Float64)
    m = pot.model
    rc = m.r_c
    sizes = [16, 32, 64, 128, 256]
    fd_max_n = 64   # finite-diff is 6N× the energy cost; only run it for smaller systems

    println("Native-Julia Allegro benchmark (CPU, Float64)")
    println("model: C=$(m.C) H=$(m.H) nb=$(m.nb) layers=$(m.L) l_max=2 r_c=$(rc) Å\n")
    @printf("| %6s | %8s | %12s | %14s | %16s | %10s |\n",
            "atoms", "edges", "energy (ms)", "E+forces (ms)", "fd forces (ms)", "speedup")
    @printf("| %6s | %8s | %12s | %14s | %16s | %10s |\n",
            ":----:", ":----:", ":----:", ":----:", ":----:", ":----:")
    for n in sizes
        coords, species = make_system(n)
        nbr = Molly.neighbour_lists(coords, nothing, rc)
        n_edges = sum(length, nbr)
        t_E  = timeit(() -> Molly.allegro_total_energy(m, coords, species, nothing, rc)) * 1e3
        t_EF = timeit(() -> Molly.allegro_energy_and_forces(m, coords, species, nothing, rc)) * 1e3
        if n <= fd_max_n
            t_fd = timeit(() -> fd_forces(m, coords, species, rc); samples=3) * 1e3
            sp = @sprintf("%.0f×", t_fd / t_EF)
            fd_str = @sprintf("%.2f", t_fd)
        else
            fd_str = "—"
            sp = "—"
        end
        @printf("| %6d | %8d | %12.3f | %14.3f | %16s | %10s |\n",
                n, n_edges, t_E, t_EF, fd_str, sp)
    end
end

main()
