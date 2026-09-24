# Allegro 6mrr trajectory benchmark: a native-Molly Allegro NVE run on the full 6mrr system
# (15,954 atoms, H/C/N/O, water-dominated, periodic) driven by the GPU-portable analytic forces
# (compute_allegro_forces_ka). Measures MD throughput (ns/day) and — the correctness check — the
# total-energy drift of NVE integration. Because the forces are the exact analytic gradient of the
# energy, a correct implementation conserves total energy (KE+PE) to a bounded drift regardless of
# whether the weights are physically trained; this validates the force/energy pair *dynamically* at
# biomolecular scale. Physical realism (a trajectory that stays folded/structured) needs a trained
# H/C/N/O model and is the separate follow-up.
#
#   ALLEGRO_TRAJ_BACKEND=metal julia --project=<env> benchmark/allegro_trajectory.jl   # Apple M3
#   ALLEGRO_TRAJ_BACKEND=cuda  julia --project=<env> benchmark/allegro_trajectory.jl   # RTX 5080
# Env: ALLEGRO_TRAJ_STEPS (default 200), ALLEGRO_TRAJ_DT (fs, default 0.25), ALLEGRO_TRAJ_T0 (K,
# initial Maxwell-Boltzmann temperature, default 50).

using Molly, HDF5, JSON3, Random, Printf, LinearAlgebra
using Molly: SVector, to_device

const BK = lowercase(get(ENV, "ALLEGRO_TRAJ_BACKEND", "metal"))
if BK == "cuda"
    using CUDA
    const AT = CuArray; const TT = Float64
    dev(x) = to_device(x, CuArray); devsync() = CUDA.synchronize()
    backend() = CUDABackend()
elseif BK == "metal"
    using Metal
    const AT = MtlArray; const TT = Float32
    dev(x) = to_device(x, MtlArray); devsync() = Metal.synchronize()
    backend() = Metal.MetalBackend()
else
    const AT = Array; const TT = Float64
    dev(x) = x; devsync() = nothing
    backend() = Molly.KernelAbstractions.CPU()
end

const H5  = joinpath(@__DIR__, "..", "data", "allegro_reference", "allegro_model.h5")
const PDB = joinpath(@__DIR__, "..", "data", "6mrr_equil.pdb")
const RES = joinpath(@__DIR__, "results")
const EV_PER_AMU_A2_FS2 = 103.6426965                # 1 amu·Å²/fs² in eV (MD unit bridge)
_steps() = parse(Int, get(ENV, "ALLEGRO_TRAJ_STEPS", "200"))
_dt()    = parse(Float64, get(ENV, "ALLEGRO_TRAJ_DT", "0.25"))
_T0()    = parse(Float64, get(ENV, "ALLEGRO_TRAJ_T0", "50"))

const ELEM_MASS = Dict("H"=>1.008, "C"=>12.011, "N"=>14.007, "O"=>15.999, "S"=>32.06)
# The native reference model is 2-species, so 6mrr's four elements are mapped H→1, {C,N,O}→2. This
# is irrelevant to throughput and to energy conservation (both depend on the compute, not on the
# chemical meaning of the species index); a physical run needs a 4-species trained model.
const ELEM_SPECIES = Dict("H"=>1, "C"=>2, "N"=>2, "O"=>2, "S"=>2)

function parse_pdb(path)
    coords = SVector{3,Float64}[]; elems = String[]; box = SVector{3,Float64}(0,0,0)
    for ln in eachline(path)
        if startswith(ln, "CRYST1")
            box = SVector{3,Float64}(parse(Float64, ln[7:15]), parse(Float64, ln[16:24]), parse(Float64, ln[25:33]))
        elseif startswith(ln, "ATOM") || startswith(ln, "HETATM")
            x = parse(Float64, ln[31:38]); y = parse(Float64, ln[39:46]); z = parse(Float64, ln[47:54])
            el = strip(ln[77:78])
            push!(coords, SVector{3,Float64}(x, y, z)); push!(elems, el)
        end
    end
    return coords, elems, box
end

function main()
    coords_h, elems, box = parse_pdb(PDB)
    n = length(coords_h)
    species = [ELEM_SPECIES[e] for e in elems]
    masses  = Float64[ELEM_MASS[e] for e in elems]
    pot = AllegroPotential(H5; T=Float64); m = pot.model; rc = m.r_c
    boundary = CubicBoundary(box[1], box[2], box[3])
    steps = _steps(); dt = _dt(); T0 = _T0()
    println("6mrr trajectory | backend=$BK T=$TT | atoms=$n  box=$(round.(box; digits=2))  rc=$rc")
    println("model: C=$(m.C) H=$(m.H) L=$(m.L) S=$(m.S) | steps=$steps dt=$dt fs  T0=$T0 K")

    gpu = Molly.build_allegro_gpu(m, backend(), TT)
    cdev = dev([SVector{3,TT}(TT.(c)...) for c in coords_h])
    forces_ka() = (E, F) = Molly.compute_allegro_forces_ka(m, cdev, species, boundary;
                             backend=backend(), T=TT, gpu=gpu)

    # one call to check it runs at scale + get the edge count and a timing sample
    E0, Fdev = Molly.compute_allegro_forces_ka(m, cdev, species, boundary; backend=backend(), T=TT, gpu=gpu)
    devsync()
    F0 = Array(Fdev)
    sumF = sqrt(sum(abs2, sum(F0[:, i] for i in 1:n)))
    maxF = maximum(abs, F0)
    @printf("initial: E=%.4f eV  ΣF=%.2e  max|F|=%.3f eV/Å\n", Float64(E0), sumF, maxF)

    # NVE velocity-Verlet in eV / Å / amu / fs, forces from the GPU kernel each step.
    rng = MersenneTwister(1)
    kB = 8.617333262e-5   # eV/K
    v = [SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) .* sqrt(kB * T0 / masses[i] / EV_PER_AMU_A2_FS2) for i in 1:n]
    x = copy(coords_h)
    ke(v) = 0.5 * EV_PER_AMU_A2_FS2 * sum(masses[i] * sum(abs2, v[i]) for i in 1:n)
    acc(Fh) = [SVector{3,Float64}(Fh[1,i], Fh[2,i], Fh[3,i]) ./ (masses[i] * EV_PER_AMU_A2_FS2) for i in 1:n]

    cdev .= dev([SVector{3,TT}(TT.(c)...) for c in x])
    E, Fdev = Molly.compute_allegro_forces_ka(m, cdev, species, boundary; backend=backend(), T=TT, gpu=gpu)
    a = acc(Array(Fdev))
    E_tot0 = Float64(E) + ke(v)
    times = Float64[]; drifts = Float64[]
    for s in 1:steps
        t0 = time()
        @inbounds for i in 1:n
            v[i] += 0.5 * dt * a[i]
            x[i] += dt * v[i]
        end
        cdev .= dev([SVector{3,TT}(TT.(x[i])...) for i in 1:n])
        E, Fdev = Molly.compute_allegro_forces_ka(m, cdev, species, boundary; backend=backend(), T=TT, gpu=gpu)
        devsync()
        Fh = Array(Fdev); a = acc(Fh)
        @inbounds for i in 1:n
            v[i] += 0.5 * dt * a[i]
        end
        push!(times, (time() - t0) * 1e3)
        E_tot = Float64(E) + ke(v)
        push!(drifts, (E_tot - E_tot0) / n * 1e3)   # meV/atom
        (s <= 3 || s % 50 == 0) && @printf("  step %4d  %8.1f ms  E_tot=%.4f eV  drift=%.4f meV/atom\n",
                                            s, times[end], E_tot, drifts[end])
    end
    ms = minimum(times); med = sort(times)[cld(length(times), 2)]
    nsday = (dt * 1e-6) / (med * 1e-3) * 86400   # ns simulated per wall-day at median step time
    maxdrift = maximum(abs, drifts)
    @printf("\nstep time: min %.1f ms, median %.1f ms  →  %.3f ns/day\n", ms, med, nsday)
    @printf("NVE energy drift over %d steps (%.1f fs): max |ΔE| = %.4f meV/atom\n",
            steps, steps*dt, maxdrift)

    outdir = RES; mkpath(outdir)
    key = BK == "cuda" ? "cuda" : (BK == "metal" ? "metal" : "cpu")
    path = joinpath(outdir, "allegro_trajectory.json")
    # Parse an existing file into plain nested Dicts so re-serialising the merged result works (a raw
    # JSON3 object read back in is immutable and JSON3.pretty chokes on the mix).
    prev = isfile(path) ? JSON3.read(read(path, String), Dict{String,Dict{String,Any}}) :
                          Dict{String,Dict{String,Any}}()
    prev[key] = Dict{String,Any}("atoms"=>n, "steps"=>steps, "dt_fs"=>dt, "step_ms_min"=>ms,
                     "step_ms_med"=>med, "ns_per_day"=>nsday, "max_drift_meV_atom"=>maxdrift,
                     "E0_eV"=>Float64(E0), "maxF"=>maxF)
    open(path, "w") do io; JSON3.pretty(io, prev); end
    println("wrote ", path)
end

main()
