# Driver: run the Allegro timing sweep (CPU), write JSON to benchmark/results/, then render the
# CairoMakie figures to benchmark/images/. Each phase is isolated so one failure (e.g. CairoMakie
# absent) does not abort the rest — the JSON already written stays usable.
#
#   julia --project=<env-with-Molly+HDF5+JSON3+CairoMakie> benchmark/run_allegro_benchmarks.jl
# Env: ALLEGRO_SIZES, ALLEGRO_FD_MAX, ALLEGRO_SPACING (see allegro.jl),
#      ALLEGRO_SKIP_PLOTS (set to skip CairoMakie).

println("="^72)
println("Allegro benchmark driver — results → benchmark/results/, figures → benchmark/images/")
println("="^72)

function phase(name, path)
    println("\n---- ", name, " ----")
    try
        include(path)
    catch err
        @warn "phase '$name' failed; continuing" exception = (err, catch_backtrace())
    end
end

phase("timing (CPU)", joinpath(@__DIR__, "allegro.jl"))

if !haskey(ENV, "ALLEGRO_SKIP_PLOTS")
    phase("figures (CairoMakie)", joinpath(@__DIR__, "allegro_plots.jl"))
else
    println("\n(ALLEGRO_SKIP_PLOTS set — skipping figures)")
end

println("\ndone.")
