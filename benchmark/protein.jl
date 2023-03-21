# Protein simulation benchmark

using Molly
using CUDA

using DelimitedFiles

const n_steps = 500
const data_dir = normpath(dirname(pathof(Molly)), "..", "data")
const ff_dir = joinpath(data_dir, "force_fields")
const openmm_dir = joinpath(data_dir, "openmm_6mrr")

function setup_system(gpu::Bool, f32::Bool, units::Bool)
    T = f32 ? Float32 : Float64
    ff = OpenMMForceField(
        T,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...;
        units=units,
    )

    velocities_nounits = SVector{3, T}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))
    velocities = units ? velocities_nounits * u"nm * ps^-1" : velocities_nounits
    dist_cutoff = T(1.0)
    dist_neighbors = T(1.2)

    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        velocities=gpu ? CuArray(velocities) : velocities,
        units=units,
        gpu=gpu,
        dist_cutoff=(units ? dist_cutoff * u"nm" : dist_cutoff),
        dist_neighbors=(units ? dist_neighbors * u"nm" : dist_neighbors),
    )

    dt = T(0.0005)
    sim = VelocityVerlet(dt=(units ? dt * u"ps" : dt), remove_CM_motion=false)

    return sys, sim
end

runs = [
    # run_name                   gpu    parr   f32    units
    ("CPU"                     , false, false, false, true ),
    ("CPU f32"                 , false, false, true , true ),
    ("CPU f32 nounits"         , false, false, true , false),
    ("CPU parallel"            , false, true , false, true ),
    ("CPU parallel f32"        , false, true , true , true ),
    ("CPU parallel f32 nounits", false, true , true , false),
    ("GPU"                     , true , false, false, true ),
    ("GPU f32"                 , true , false, true , true ),
    ("GPU f32 nounits"         , true , false, true , false),
]

for (run_name, gpu, parallel, f32, units) in runs
    n_threads = parallel ? Threads.nthreads() : 1
    sys, sim = setup_system(gpu, f32, units)
    simulate!(sys, sim, 5; n_threads=n_threads)
    println(run_name)
    @time simulate!(sys, sim, n_steps; n_threads=n_threads)
end
