# Tests for the native Allegro potential (energy; forces added in a later milestone). Guarded on
# the reference files produced by test/allegro_reference.py (which needs e3nn + torch offline). The
# equivariant primitives themselves are tested separately in test/equivariant.jl.

using Molly
using Molly: SVector
using HDF5
using JSON3
import AtomsCalculators
using Test

# loading Lux + HDF5 activates the MollyAllegroExt extension
using Lux

const ALLEGRO_DIR = joinpath(@__DIR__, "..", "data", "allegro_reference")
const ALLEGRO_H5 = joinpath(ALLEGRO_DIR, "allegro_model.h5")
const ALLEGRO_JSON = joinpath(ALLEGRO_DIR, "allegro_model.json")

if isfile(ALLEGRO_H5) && isfile(ALLEGRO_JSON)
    @testset "Allegro potential (energy)" begin
        pot = AllegroPotential(ALLEGRO_H5; T=Float64)
        ref = JSON3.read(read(ALLEGRO_JSON, String))
        species_syms = String.(ref.species)

        # Build a unitless System (coords in nm = Å/10) and compare potential_energy to reference.
        function build_sys(sysj)
            coords = [SVector{3,Float64}(c[1] / 10, c[2] / 10, c[3] / 10) for c in sysj.coords_A]
            n = length(coords)
            atoms = [Atom(mass=1.0) for _ in 1:n]
            elems = [species_syms[Int(s) + 1] for s in sysj.species]  # json species are 0-based
            atoms_data = [AtomData(element=e) for e in elems]
            System(atoms=atoms, coords=coords, boundary=CubicBoundary(100.0),
                   atoms_data=atoms_data, general_inters=(allegro=pot,),
                   energy_units=NoUnits, force_units=NoUnits)
        end

        for sysj in ref.systems
            sys = build_sys(sysj)
            E = AtomsCalculators.potential_energy(sys, pot)
            @test isapprox(E, Float64(sysj.energy); rtol=1e-4)

            # rotation invariance through the calculator
            th = 0.7
            R = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1]
            rot_coords = [SVector{3,Float64}((R * [c...])...) for c in sys.coords]
            sys_rot = System(atoms=sys.atoms, coords=rot_coords, boundary=sys.boundary,
                             atoms_data=sys.atoms_data, general_inters=(allegro=pot,),
                             energy_units=NoUnits, force_units=NoUnits)
            @test isapprox(E, AtomsCalculators.potential_energy(sys_rot, pot); atol=1e-8)
        end
    end
else
    @warn "Skipping Allegro potential tests — reference files not found. " *
          "Run test/allegro_reference.py (needs e3nn) to generate them."
end
