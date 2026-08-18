# Timing benchmark for the native Allegro potential (energy + analytic forces) as a function of
# system size. Loads a model exported by test/allegro_reference.py.
#
#   julia --project -e 'using Molly, Lux, HDF5; include("benchmark/allegro_benchmark.jl")'
#
# The current CPU forward/backward is an O(N²) all-pairs reference within the cutoff; a
# neighbour-list path and native GPU kernels are planned, so absolute numbers here are a baseline.

using Molly
using Molly: SVector
using Lux, HDF5
import AtomsCalculators
using Printf
using Random

const H5 = get(ENV, "ALLEGRO_H5",
    joinpath(@__DIR__, "..", "data", "allegro_reference", "allegro_model.h5"))

if !isfile(H5)
    @warn "Allegro model not found at $H5 — run test/allegro_reference.py first."
else
    pot = AllegroPotential(H5; T=Float32)
    rc_nm = Float64(pot.model.r_c) / 10  # cutoff in nm

    function random_system(n; density=0.02, seed=1)
        rng = MersenneTwister(seed)
        L = cbrt(n / density)                          # nm box
        coords = [SVector{3,Float64}(rand(rng) * L, rand(rng) * L, rand(rng) * L) for _ in 1:n]
        atoms = [Atom(mass=1.0) for _ in 1:n]
        elems = rand(rng, ["H", "C"], n)
        atoms_data = [AtomData(element=e) for e in elems]
        System(atoms=atoms, coords=coords, boundary=CubicBoundary(L),
               atoms_data=atoms_data, general_inters=(allegro=pot,),
               energy_units=NoUnits, force_units=NoUnits)
    end

    @printf("%-8s %-14s %-14s\n", "n_atoms", "energy (ms)", "forces (ms)")
    for n in (20, 50, 100, 200)
        sys = random_system(n)
        AtomsCalculators.potential_energy(sys, pot)     # warmup
        fs = [zero(SVector{3,Float64}) for _ in 1:n]
        AtomsCalculators.forces!(fs, sys, pot)
        te = @elapsed AtomsCalculators.potential_energy(sys, pot)
        tf = @elapsed (fill!(fs, zero(SVector{3,Float64})); AtomsCalculators.forces!(fs, sys, pot))
        @printf("%-8d %-14.3f %-14.3f\n", n, te * 1e3, tf * 1e3)
    end
end
