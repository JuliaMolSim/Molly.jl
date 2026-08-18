# Tests for the native Allegro potential (energy + analytic forces). Guarded on the reference files
# produced by test/allegro_reference.py (which needs e3nn + torch offline). The equivariant
# primitives themselves are tested separately in test/equivariant.jl.

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
    @testset "Allegro potential" begin
        pot = AllegroPotential(ALLEGRO_H5; T=Float64)
        ref = JSON3.read(read(ALLEGRO_JSON, String))
        species_syms = String.(ref.species)
        rc = pot.model.r_c

        # Build a unitless System (coords in nm = Å/10) from Å coordinates and 0-based species.
        function mk_sys(coords_A, species0)
            coords = [SVector{3,Float64}(c[1] / 10, c[2] / 10, c[3] / 10) for c in coords_A]
            n = length(coords)
            atoms = [Atom(mass=1.0) for _ in 1:n]
            elems = [species_syms[Int(s) + 1] for s in species0]
            atoms_data = [AtomData(element=e) for e in elems]
            System(atoms=atoms, coords=coords, boundary=CubicBoundary(100.0),
                   atoms_data=atoms_data, general_inters=(allegro=pot,),
                   energy_units=NoUnits, force_units=NoUnits)
        end

        @testset "energy vs reference + rotation invariance" begin
            for sysj in ref.systems
                sys = mk_sys(sysj.coords_A, sysj.species)
                E = AtomsCalculators.potential_energy(sys, pot)
                @test isapprox(E, Float64(sysj.energy); rtol=1e-4)
                th = 0.7
                R = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1]
                rc_coords = [SVector{3,Float64}((R * [c...])...) for c in sys.coords]
                sys_rot = System(atoms=sys.atoms, coords=rc_coords, boundary=sys.boundary,
                                 atoms_data=sys.atoms_data, general_inters=(allegro=pot,),
                                 energy_units=NoUnits, force_units=NoUnits)
                @test isapprox(E, AtomsCalculators.potential_energy(sys_rot, pot); atol=1e-8)
            end
        end

        @testset "analytic forces vs reference, ΣF≈0, finite differences" begin
            for sysj in ref.systems
                coords_A = [SVector{3,Float64}(c...) for c in sysj.coords_A]
                species = [Int(s) + 1 for s in sysj.species]  # json 0-based → 1-based
                # core analytic forces (eV/Å) vs the reference finite-diff forces
                F = Molly.allegro_forces(pot.model, coords_A, species, nothing, rc)
                Fref = [SVector{3,Float64}(fr...) for fr in sysj.forces]
                for i in eachindex(F)
                    @test isapprox(F[i], Fref[i]; atol=1e-5)
                end
                @test isapprox(sum(F), zero(SVector{3,Float64}); atol=1e-8)  # Newton's 3rd law

                # extension forces! : ΣF≈0 and matches −finite-diff of potential_energy (nm coords)
                sys = mk_sys(sysj.coords_A, sysj.species)
                fs = [zero(SVector{3,Float64}) for _ in 1:length(sys.coords)]
                AtomsCalculators.forces!(fs, sys, pot)
                @test isapprox(sum(fs), zero(SVector{3,Float64}); atol=1e-8)
                h = 1e-6
                shift(c, i, dx) = [j == i ? SVector{3,Float64}(c[j][1] + dx, c[j][2], c[j][3]) : c[j]
                                   for j in eachindex(c)]
                remake(cc) = System(atoms=sys.atoms, coords=cc, boundary=sys.boundary,
                                    atoms_data=sys.atoms_data, general_inters=(allegro=pot,),
                                    energy_units=NoUnits, force_units=NoUnits)
                Ep = AtomsCalculators.potential_energy(remake(shift(sys.coords, 1, h)), pot)
                Em = AtomsCalculators.potential_energy(remake(shift(sys.coords, 1, -h)), pot)
                @test isapprox(fs[1][1], -(Ep - Em) / (2h); rtol=1e-4)
            end
        end
    end
else
    @warn "Skipping Allegro potential tests — reference files not found. " *
          "Run test/allegro_reference.py (needs e3nn) to generate them."
end
