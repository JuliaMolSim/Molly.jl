# Protein simulation benchmark

using Molly
using CUDA

using DelimitedFiles

const n_steps = 1_000
const n_threads = Threads.nthreads()
const data_dir = normpath(dirname(pathof(Molly)), "..", "data")
const ff_dir = joinpath(data_dir, "force_fields")
const openmm_dir = joinpath(data_dir, "openmm_6mrr")

function setup_system(gpu::Bool, f32::Bool, units::Bool)
    T = f32 ? Float32 : Float64
    ff = MolecularForceField(
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
    # run_name                             gpu    parr   f32    units
    ("CPU 1 thread"                      , false, false, false, true ),
    ("CPU 1 thread f32"                  , false, false, true , true ),
    ("CPU 1 thread f32 nounits"          , false, false, true , false),
    ("CPU $n_threads threads"            , false, true , false, true ),
    ("CPU $n_threads threads f32"        , false, true , true , true ),
    ("CPU $n_threads threads f32 nounits", false, true , true , false),
    ("GPU"                               , true , false, false, true ),
    ("GPU f32"                           , true , false, true , true ),
    ("GPU f32 nounits"                   , true , false, true , false),
]

for (run_name, gpu, parallel, f32, units) in runs
    n_threads_used = parallel ? n_threads : 1
    sys, sim = setup_system(gpu, f32, units)
    simulate!(deepcopy(sys), sim, 20; n_threads=n_threads_used)
    println(run_name)
    @time simulate!(sys, sim, n_steps; n_threads=n_threads_used)
end
