struct PotentialFunction{F}
    f::F
    # fwdiff gradient caches
end

function Molly.potential_energy(V::PotentialFunction,
    sys,
    neighbors=nothing;
    n_threads=Threads.nthreads())

    return V.f(sys.coords)
end

function Molly.forces(V::PotentialFunction,
    sys,
    neighbors=nothing;
    n_threads=Threads.nthreads())

    return ForwardDiff.gradient(V.f, sys.coords)
end


