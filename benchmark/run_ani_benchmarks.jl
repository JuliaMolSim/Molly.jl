# Driver: run the ANI energy + forces timing sweeps (CPU + Apple Metal), write JSON to
# benchmark/results/, then render the CairoMakie figures to benchmark/images/. After running
# the TorchANI reference (test/torchani_reference.py --benchmark), the figures/tables also
# pick up the TorchANI series for a head-to-head.
#
#   julia --project=<env> -t8 benchmark/run_ani_benchmarks.jl
# Env: ANI_SIZES (energy), ANI_FSIZES (Metal forces), ANI_ENZYME_SIZES (CPU forces),
#      ANI_ENSEMBLE (0|full), ANI_SKIP_PLOTS (set to skip CairoMakie).
#
# Each phase is isolated so one failure (e.g. no GPU, or CairoMakie absent) does not abort
# the rest — the JSON already written stays usable.

println("="^72)
println("ANI benchmark driver — results → benchmark/results/, figures → benchmark/images/")
println("="^72)

function phase(name, path)
    println("\n---- ", name, " ----")
    try
        include(path)
    catch err
        @warn "phase '$name' failed; continuing" exception = (err, catch_backtrace())
    end
end

phase("energy (CPU + Metal)", joinpath(@__DIR__, "ani_gpu_compare.jl"))
phase("forces (CPU Enzyme + Metal)", joinpath(@__DIR__, "ani_forces_gpu.jl"))

if !haskey(ENV, "ANI_SKIP_PLOTS")
    phase("figures (CairoMakie)", joinpath(@__DIR__, "ani_plots.jl"))
else
    println("\n(ANI_SKIP_PLOTS set — skipping figures)")
end

println("\ndone.")
