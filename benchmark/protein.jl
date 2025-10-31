# Protein simulation benchmark

using Molly
using CUDA

using DelimitedFiles

const n_steps = 1_000
const n_threads = Threads.nthreads()
const data_dir = normpath(dirname(pathof(Molly)), "..", "data")
const ff_dir = joinpath(data_dir, "force_fields")
const openmm_dir = joinpath(data_dir, "openmm_6mrr")

function setup_system(::Type{AT}, f32::Bool, units::Bool) where AT
    T = f32 ? Float32 : Float64
    ff = MolecularForceField(
        T,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml", "his.xml"])...;
        units=units,
    )

    velocities_nounits = SVector{3, T}.(eachrow(readdlm(joinpath(openmm_dir, "velocities_300K.txt"))))
    velocities = units ? velocities_nounits * u"nm * ps^-1" : velocities_nounits
    dist_cutoff = T(1.0)

    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        velocities=AT(velocities),
        units=units,
        array_type=AT,
        dist_cutoff=(units ? dist_cutoff * u"nm" : dist_cutoff),
        nonbonded_method=:cutoff,
    )

    dt = T(0.0005)
    sim = VelocityVerlet(dt=(units ? dt * u"ps" : dt), remove_CM_motion=false)

    return sys, sim
end

runs = [
    # run_name                             gpu      parr   f32    units
    ("CPU 1 thread"                      , Array  , false, false, true ),
    ("CPU 1 thread f32"                  , Array  , false, true , true ),
    ("CPU 1 thread f32 nounits"          , Array  , false, true , false),
    ("CPU $n_threads threads"            , Array  , true , false, true ),
    ("CPU $n_threads threads f32"        , Array  , true , true , true ),
    ("CPU $n_threads threads f32 nounits", Array  , true , true , false),
    ("CUDA"                              , CuArray, false, false, true ),
    ("CUDA f32"                          , CuArray, false, true , true ),
    ("CUDA f32 nounits"                  , CuArray, false, true , false),
]

for (run_name, AT, parallel, f32, units) in runs
    n_threads_used = parallel ? n_threads : 1
    sys, sim = setup_system(AT, f32, units)
    simulate!(deepcopy(sys), sim, 20; n_threads=n_threads_used)
    println(run_name)
    @time simulate!(sys, sim, n_steps; n_threads=n_threads_used)
end
