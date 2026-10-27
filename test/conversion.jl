# Whether `b`, part of a converted system, holds the values of `a` converted to the float type
#   `T`, with the units stripped if `strip_units`
converted_equal(a::AbstractFloat, b, T, strip_units) = (b isa T && b == T(a))
function converted_equal(a::Unitful.Quantity, b, T, strip_units)
    if strip_units
        return converted_equal(ustrip(a), b, T, strip_units)
    end
    return b isa Unitful.Quantity && unit(b) == unit(a) &&
           converted_equal(ustrip(a), ustrip(b), T, strip_units)
end
converted_equal(a::Unitful.Units, b, T, strip_units) = (b == (strip_units ? NoUnits : a))
function converted_equal(a::Union{Tuple, NamedTuple}, b, T, strip_units)
    return length(a) == length(b) &&
           all(map((x, y) -> converted_equal(x, y, T, strip_units), Tuple(a), Tuple(b)))
end
function converted_equal(a::AbstractArray, b, T, strip_units)
    return size(a) == size(b) &&
           all(map((x, y) -> converted_equal(x, y, T, strip_units), from_device(a),
                   from_device(b)))
end
function converted_equal(a::Union{Integer, AbstractString, Symbol, Nothing, Function}, b, T,
                         strip_units)
    return isequal(a, b)
end
# The other fields of a TriclinicBoundary are calculated from the basis vectors
function converted_equal(a::TriclinicBoundary, b, T, strip_units)
    return b isa TriclinicBoundary &&
           converted_equal(a.basis_vectors, b.basis_vectors, T, strip_units)
end
function converted_equal(a::A, b::B, T, strip_units) where {A, B}
    return A.name.wrapper === B.name.wrapper &&
           all(i -> converted_equal(getfield(a, i), getfield(b, i), T, strip_units),
               1:fieldcount(A))
end

widened_equal(a, b) = converted_equal(a, b, Float64, true)

max_force_diff(fs, fs_ref) = maximum(norm.(ustrip_vec.(from_device(fs)) .-
                                           ustrip_vec.(from_device(fs_ref))))
max_force(fs) = maximum(norm.(ustrip_vec.(from_device(fs))))
nf_type(sys) = typeof(sys.neighbor_finder).name.wrapper
nf_masks(sys) = Molly.neighbor_finder_masks(sys.neighbor_finder, length(sys))

# An interaction whose constructor does not take its fields
struct UnconvertibleInteraction{T} <: PairwiseInteraction
    a::T
    b::T

    UnconvertibleInteraction(a) = new{typeof(a)}(a, 2a)
end

# A custom atom type, used with Buckingham
struct ConversionBuckinghamAtom{M, TA, TB, TC}
    mass::M
    A::TA
    B::TB
    C::TC
end

@testset "System conversion" begin
    ff_water = MolecularForceField(joinpath(ff_dir, "tip4pfb.xml"))
    ff_tip3p = MolecularForceField(joinpath(ff_dir, "tip3p_standard.xml"))
    ff_dexp = MolecularForceField(joinpath(ff_dir, "ethanol_garnet.xml"))
    ff_protein = MolecularForceField(joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...;
                                     strictness=:nowarn)
    ff_implicit = MolecularForceField(joinpath(ff_dir, "ff99SBildn.xml"))
    ff_benzene = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "benzene.xml"])...)

    implicit_builder(solvent) = (FT, AT) -> System(
        joinpath(data_dir, "6mrr_nowater.pdb"),
        ff_implicit;
        boundary=CubicBoundary(FT(100.0)u"nm"),
        array_type=AT,
        float_type=FT,
        dist_cutoff=5.0u"nm",
        nonbonded_method=DistanceCutoff(5.0u"nm"),
        dispersion_correction=false,
        implicit_solvent=solvent,
        strictness=:nowarn,
        n_threads=1,
    )

    # Systems covering the setup options: PME, Ewald and reaction field electrostatics,
    #   virtual sites, LINCS and SHAKE constraints, CMAP, implicit solvent,
    #   custom non-bonded interactions and a triclinic box
    builders = (
        ("water PME", (FT, AT) -> System(
            joinpath(data_dir, "tip4pew.pdb"),
            ff_water;
            array_type=AT,
            float_type=FT,
            nonbonded_method=SetupPME(),
            center_coords=false,
        )),
        ("DEXP reaction field", (FT, AT) -> System(
            joinpath(data_dir, "ethanol_solv.pdb"),
            ff_dexp;
            array_type=AT,
            float_type=FT,
            nonbonded_method=SetupCoulombReactionField(),
            dist_cutoff=1.0u"nm",
        )),
        ("protein PME with LINCS", (FT, AT) -> System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff_protein;
            array_type=AT,
            float_type=FT,
            nonbonded_method=SetupPME(),
            center_coords=false,
            constraints=:hbonds,
            rigid_water=true,
        )),
        ("benzene Ewald with SHAKE", (FT, AT) -> System(
            joinpath(data_dir, "benzene_solv.pdb"),
            ff_benzene;
            array_type=AT,
            float_type=FT,
            nonbonded_method=SetupEwald(),
            dist_cutoff=1.0u"nm",
            center_coords=false,
            constraints=:hbonds,
            constraint_algorithm=SetupSHAKE_RATTLE(),
            rigid_water=true,
        )),
        ("protein GBN2", implicit_builder(SetupImplicitSolventGBN2(kappa=1.0u"nm^-1"))),
        ("triclinic water", (FT, AT) -> System(
            joinpath(data_dir, "water_3mol_triclinic.pdb"),
            ff_tip3p;
            array_type=AT,
            float_type=FT,
            nonbonded_method=SetupCoulombReactionField(),
            dist_cutoff=0.5u"nm",
            strictness=:nowarn,
        )),
    )

    # Float32 energies are compared with this tolerance, set by the Float32 error of each system
    rtols_float32 = Dict(
        "DEXP reaction field"      => 1e-5,
        "benzene Ewald with SHAKE" => 5e-4,
        "triclinic water"          => 2e-5,
        "protein GBN2"             => 1e-5,
    )
    rtol_float32(name) = get(rtols_float32, name, 1e-6)

    @testset "Precision and units" begin
        for (name, build) in builders
            sys_64, sys_32 = build(Float64, Array), build(Float32, Array)

            # Float64 with units to Float64 without units gives the same system, with the
            #   neighbor finder keeping its type and the storage of its matrices
            sys_conv = System(sys_64; units=false, grad_safe=true)
            @test sys_conv.energy_units == NoUnits
            @test sys_conv.df == sys_64.df
            @test nf_type(sys_conv) === nf_type(sys_64)
            @test typeof(sys_conv.neighbor_finder.eligible).name.wrapper ===
                    typeof(sys_64.neighbor_finder.eligible).name.wrapper
            E_64 = potential_energy(sys_64; n_threads=1)
            fs_64 = forces(sys_64; n_threads=1)
            @test isapprox(potential_energy(sys_conv; n_threads=1), ustrip(E_64); rtol=1e-12)
            @test max_force_diff(forces(sys_conv; n_threads=1), fs_64) < 1e-10 * max_force(fs_64)

            # Float32 to Float64 keeps the Float32 values exactly. The energy is compared to
            #   the Float32 one, as the conversion can not recover the precision lost in the
            #   original values.
            sys_conv = System(sys_32; float_type=Float64, units=false)
            E_32 = potential_energy(sys_32; n_threads=1)
            fs_32 = forces(sys_32; n_threads=1)
            @test isapprox(potential_energy(sys_conv; n_threads=1), ustrip(E_32);
                           rtol=rtol_float32(name))
            @test widened_equal(sys_32.atoms, sys_conv.atoms)
            @test widened_equal(sys_32.coords, sys_conv.coords)
            @test widened_equal(sys_32.boundary, sys_conv.boundary)
            @test widened_equal(sys_32.k, sys_conv.k)
            @test widened_equal(sys_32.pairwise_inters, sys_conv.pairwise_inters)
            @test widened_equal(sys_32.specific_inter_lists, sys_conv.specific_inter_lists)
            # Converting to Float64 and back gives the original energy and forces. Units are
            #   kept as the unitless implicit solvent calculation takes a different Float32
            #   path.
            sys_back = System(System(sys_32; float_type=Float64); float_type=Float32)
            @test isapprox(potential_energy(sys_back; n_threads=1), E_32; rtol=1e-8)
            @test max_force_diff(forces(sys_back; n_threads=1), fs_32) < 1e-6 * max_force(fs_32)

            # Float64 to Float32 rounds the values, giving the atoms of a system set up in
            #   Float32. Parameters that setup derives in the float type, such as mixed 1-4
            #   parameters, can differ from those in the last digit.
            sys_down = System(sys_64; float_type=Float32)
            @test sys_down.energy_units == sys_64.energy_units
            @test converted_equal(sys_64.coords, sys_down.coords, Float32, false)
            @test converted_equal(sys_64.pairwise_inters, sys_down.pairwise_inters, Float32,
                                  false)
            @test converted_equal(sys_64.specific_inter_lists, sys_down.specific_inter_lists,
                                  Float32, false)
            @test converted_equal(sys_32.atoms, sys_down.atoms, Float32, false)
            @test isapprox(potential_energy(sys_down; n_threads=1), E_32;
                           rtol=rtol_float32(name))
        end
    end

    @testset "Devices" begin
        for AT in array_list
            AT == Array && continue
            for (name, build) in builders
                sys_gpu, sys_cpu = build(Float32, AT), build(Float32, Array)
                E_gpu = potential_energy(sys_gpu)

                # GPU to CPU, keeping the Float32 values
                sys_conv = System(sys_gpu; float_type=Float64, units=false, array_type=Array,
                                  grad_safe=true)
                @test array_type(sys_conv) == Array
                # The neighbor finder is the one that setup chooses on the new device
                @test nf_type(sys_conv) === nf_type(sys_cpu)
                @test nf_masks(sys_conv) == nf_masks(sys_gpu)
                @test sys_conv.df == sys_gpu.df
                @test widened_equal(sys_gpu.atoms, sys_conv.atoms)
                @test widened_equal(sys_gpu.coords, sys_conv.coords)
                @test widened_equal(sys_gpu.specific_inter_lists, sys_conv.specific_inter_lists)
                @test isapprox(potential_energy(sys_conv), ustrip(E_gpu); rtol=rtol_float32(name))

                # CPU to GPU gives the system that setup gives on the GPU
                sys_moved = System(sys_cpu; array_type=AT)
                @test array_type(sys_moved) == AT
                @test Molly.float_type_high(sys_moved) == Molly.float_type_high(sys_gpu)
                @test nf_type(sys_moved) === nf_type(sys_gpu)
                @test nf_masks(sys_moved) == nf_masks(sys_gpu)
                @test sys_moved.df == sys_gpu.df
                @test isapprox(potential_energy(sys_moved), E_gpu; rtol=1e-6)

                # Back to the GPU from the unitless Float64 CPU system
                sys_back = System(sys_conv; float_type=Float32, array_type=AT, grad_safe=false)
                @test nf_type(sys_back) === nf_type(sys_gpu)
                @test isapprox(potential_energy(sys_back), ustrip(E_gpu); rtol=1e-5)
            end

            # Collective variables keep their atom indices on the CPU
            sys = builders[2][2](Float64, Array)
            bias = BiasPotential(CalcDist([1], [5]),
                                 SquareBias(100.0u"kJ * mol^-1 * nm^-2", 0.5u"nm"))
            sys = System(sys; general_inters=(bias,))
            sys_gpu = System(sys; array_type=AT, float_type=Float32)
            @test sys_gpu.general_inters[1].cv_type.atom_inds_1 isa Vector{Int}
            @test isapprox(potential_energy(sys_gpu), potential_energy(sys);
                           rtol=rtols_float32["DEXP reaction field"])
        end
    end

    # Backends without Float64 support, such as Metal, give Float32 by default
    for AT in array_list_metal
        AT in array_list && continue
        sys_moved = System(builders[2][2](Float64, Array); array_type=AT)
        @test float_type(sys_moved) == Float32
        @test Molly.float_type_high(sys_moved) == Float32
    end

    @testset "Neighbor finders" begin
        build_dexp = builders[2][2]
        # Each CPU neighbor finder is kept on the CPU
        for nf_type_cpu in (DistanceNeighborFinder, TreeNeighborFinder, CellListMapNeighborFinder)
            sys = System(joinpath(data_dir, "ethanol_solv.pdb"), ff_dexp; float_type=Float32,
                         nonbonded_method=SetupCoulombReactionField(), dist_cutoff=1.0u"nm",
                         neighbor_finder_type=nf_type_cpu)
            sys_conv = System(sys; float_type=Float64, units=false, grad_safe=true)
            @test sys_conv.neighbor_finder isa nf_type_cpu
            @test nf_masks(sys_conv) == nf_masks(sys)
            @test isapprox(potential_energy(sys_conv), ustrip(potential_energy(sys));
                           rtol=rtols_float32["DEXP reaction field"])
        end

        for AT in array_list
            AT == Array && continue
            # A GPUNeighborFinder can not be used with grad_safe, so the one setup chooses for
            #   a gradient safe system is used
            sys = build_dexp(Float32, AT)
            sys_gs = System(sys; grad_safe=true)
            @test !(sys_gs.neighbor_finder isa GPUNeighborFinder)
            @test nf_masks(sys_gs) == nf_masks(sys)
            @test isapprox(potential_energy(sys_gs), potential_energy(sys);
                           rtol=rtols_float32["DEXP reaction field"])

            # GPUCellListNeighborFinder is kept on the same GPU and replaced on the CPU. The
            #   water box is too small for a cell list.
            sys = System(joinpath(data_dir, "ethanol_solv.pdb"), ff_dexp; array_type=AT,
                         float_type=Float32, nonbonded_method=SetupCoulombReactionField(),
                         dist_cutoff=1.0u"nm", neighbor_finder_type=GPUCellListNeighborFinder)
            E = potential_energy(sys)
            sys_64 = System(sys; float_type=Float64)
            @test sys_64.neighbor_finder isa GPUCellListNeighborFinder
            @test isapprox(potential_energy(sys_64), E; rtol=rtols_float32["DEXP reaction field"])
            sys_cpu = System(sys; float_type=Float64, units=false, array_type=Array)
            @test sys_cpu.neighbor_finder isa CellListMapNeighborFinder
            @test nf_masks(sys_cpu) == nf_masks(sys)
            @test isapprox(potential_energy(sys_cpu), ustrip(E);
                           rtol=rtols_float32["DEXP reaction field"])
        end
    end

    @testset "Simulation" begin
        # A converted system follows the same trajectory, including virtual sites,
        #   constraints and neighbor list updates
        for (name, build) in builders[[1, 3]]
            sys = build(Float64, Array)
            sys = System(sys; velocities=random_velocities(sys, 300.0u"K"; rng=Xoshiro(10)))
            sys_conv = System(sys; units=false)
            simulate!(sys, VelocityVerlet(dt=0.001u"ps"), 20; n_threads=1)
            simulate!(sys_conv, VelocityVerlet(dt=0.001), 20; n_threads=1)
            @test maximum(norm.(ustrip_vec.(sys.coords) .- sys_conv.coords)) < 1e-10
            @test maximum(norm.(ustrip_vec.(sys.velocities) .- sys_conv.velocities)) < 1e-8
        end

        # A system moved to the GPU follows the trajectory of one set up there, with
        #   constraints and virtual sites
        for AT in array_list
            AT == Array && continue
            for (name, build) in builders[[1, 4]]
                sys_moved, sys_gpu = System(build(Float32, Array); array_type=AT), build(Float32, AT)
                simulate!(sys_moved, VelocityVerlet(dt=0.001u"ps"), 20)
                simulate!(sys_gpu, VelocityVerlet(dt=0.001u"ps"), 20)
                @test maximum(norm.(from_device(sys_moved.coords) .-
                                    from_device(sys_gpu.coords))) < 1e-4u"nm"
            end
        end
    end

    @testset "Systems set up by hand" begin
        # Custom atom type, units with Å and eV, Float32 to Float64
        n_atoms = 20
        boundary = CubicBoundary(3.0f0u"nm")
        coords = place_atoms(n_atoms, boundary; min_dist=0.3f0u"nm", rng=Xoshiro(1))
        atoms = [ConversionBuckinghamAtom(10.0f0u"g/mol", 1400.0f0u"kJ * mol^-1",
                                          28.0f0u"nm^-1", 0.0018f0u"kJ * mol^-1 * nm^6")
                 for _ in 1:n_atoms]
        bonds = InteractionList2Atoms(collect(Int32, 1:2:n_atoms), collect(Int32, 2:2:n_atoms),
                    [HarmonicBond(k=1000.0f0u"kJ * mol^-1 * nm^-2", r0=0.4f0u"nm")
                     for _ in 1:2:n_atoms])
        for AT in array_list
            sys = System(
                atoms=to_device(atoms, AT),
                coords=to_device(coords, AT),
                boundary=boundary,
                pairwise_inters=(Buckingham(cutoff=DistanceCutoff(1.0f0u"nm")),),
                specific_inter_lists=(to_device(bonds, AT),),
                neighbor_finder=DistanceNeighborFinder(n_atoms=n_atoms,
                                    dist_cutoff=1.2f0u"nm", array_type=AT),
            )
            sys_conv = System(sys; float_type=Float64, units=false, array_type=Array)
            @test eltype(sys_conv.atoms) <: ConversionBuckinghamAtom{Float64, Float64}
            @test widened_equal(sys.atoms, sys_conv.atoms)
            @test isapprox(potential_energy(sys_conv), ustrip(potential_energy(sys)); rtol=1e-6)
        end

        # 2D system with an infinite boundary and no neighbor finder
        sys = System(
            atoms=[Atom(mass=1.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:2],
            coords=[SVector(-0.5, 0.25)u"nm", SVector(0.6, 0.4)u"nm"],
            boundary=RectangularBoundary(Inf*u"nm"),
            general_inters=(MullerBrown(),),
        )
        sys_conv = System(sys; units=false)
        @test sys_conv.general_inters[1].energy_units == NoUnits
        @test isapprox(potential_energy(sys_conv), ustrip(potential_energy(sys)); rtol=1e-12)
        @test max_force_diff(forces(sys_conv), forces(sys)) < 1e-10 * max_force(forces(sys))
    end

    @testset "Alchemical systems" begin
        # The GROMACS scheduler gives PME_λ, along with soft core and λ bonded interactions
        sys_solv = System(joinpath(data_dir, "benzene_solv.pdb"), ff_benzene;
                          nonbonded_method=SetupPME(), dist_cutoff=1.0u"nm",
                          center_coords=false)
        sys = AbsoluteFESystem(sys_solv, 0.5, collect(1:12);
                               scheduler=Molly.GROMACSLambdaABFEScheduler(dual=true))
        sys_conv = System(sys; units=false)
        @test isapprox(potential_energy(sys_conv; n_threads=1),
                       ustrip(potential_energy(sys; n_threads=1)); rtol=1e-12)
    end

    @testset "Interaction types" begin
        # Each interaction type converts, see the systems above for energies
        T = Float32
        c = Molly.SystemConversion{Float64, Array, Float64}(true, false)
        atoms = [Atom(index=i, mass=T(10.0), charge=T(0.1), σ=T(0.3), ϵ=T(0.4)) for i in 1:4]
        cutoff = DistanceCutoff(T(1.0))
        cc = T(ustrip(Molly.coulomb_const))
        λ_kwargs = (λ_mixing=Molly.MinimumMixing(), scheduler=Molly.DefaultLambdaScheduler())
        inters = (
            LennardJones(cutoff=cutoff, weight_special=T(0.5)),
            Molly.LennardJonesScaled(cutoff=cutoff),
            LennardJonesSoftCoreBeutler(α=T(0.5), cutoff=cutoff),
            LennardJonesSoftCoreGapsys(α=T(0.85), cutoff=cutoff),
            Molly.LennardJones14(T(0.3), T(0.4), T(0.5)),
            Molly.LennardJones14Scaled(; σ14_mixed=T(0.3), ϵ14_mixed=T(0.4), weight_14=T(0.5),
                                       λ_kwargs...),
            Molly.LennardJones14SoftCoreBeutler(; σ14_mixed=T(0.3), ϵ14_mixed=T(0.4),
                                                weight_14=T(0.5), α=T(0.5), λ_kwargs...),
            Molly.LennardJones14SoftCoreGapsys(; σ14_mixed=T(0.3), ϵ14_mixed=T(0.4),
                                               weight_14=T(0.5), α=T(0.85), λ_kwargs...),
            AshbaughHatch(cutoff=cutoff),
            SoftSphere(cutoff=cutoff),
            Mie(m=T(6), n=T(12), cutoff=cutoff),
            Buckingham(cutoff=cutoff),
            DoubleExponential(α=T(16.766), β=T(4.427), cutoff=cutoff),
            DoubleExponentialSoftCore(α=T(16.766), β=T(4.427), cutoff=cutoff),
            Coulomb(cutoff=cutoff, coulomb_const=cc),
            Molly.CoulombScaled(cutoff=cutoff, coulomb_const=cc),
            CoulombSoftCoreBeutler(cutoff=cutoff, coulomb_const=cc),
            CoulombSoftCoreGapsys(cutoff=cutoff, σQ=T(1.0), coulomb_const=cc),
            CoulombReactionField(dist_cutoff=T(1.0), coulomb_const=cc),
            Molly.CoulombReactionFieldScaled(dist_cutoff=T(1.0), coulomb_const=cc),
            CoulombSoftCoreBeutlerReactionField(dist_cutoff=T(1.0), coulomb_const=cc),
            CoulombSoftCoreGapsysReactionField(dist_cutoff=T(1.0), σQ=T(1.0), coulomb_const=cc),
            CoulombEwald(dist_cutoff=T(1.0), coulomb_const=cc),
            Molly.CoulombEwaldScaled(dist_cutoff=T(1.0), coulomb_const=cc),
            CoulombSoftCoreBeutlerEwald(dist_cutoff=T(1.0), coulomb_const=cc),
            CoulombSoftCoreGapsysEwald(dist_cutoff=T(1.0), σQ=T(1.0), coulomb_const=cc),
            Yukawa(cutoff=cutoff, coulomb_const=cc, kappa=T(1.0)),
            Gravity(G=T(1.0)),
            DPDInteraction(a=T(25.0), γ=T(4.5), σ=T(3.0), r_c=T(1.0), dt=T(0.01),
                           key=UInt64(1)),
            HarmonicPositionRestraint(k=T(100.0), x0=SVector{3, T}(1.0, 1.0, 1.0)),
            HarmonicBond(k=T(100.0), r0=T(0.5)),
            Molly.HarmonicBondλ(; k=T(100.0), r0=T(0.5), λ_kwargs...),
            MorseBond(D=T(100.0), a=T(2.0), r0=T(0.5)),
            Molly.MorseBondλ(; D=T(100.0), a=T(2.0), r0=T(0.5), λ_kwargs...),
            FENEBond(k=T(100.0), r0=T(1.4), σ=T(0.3), ϵ=T(0.4)),
            HarmonicAngle(k=T(10.0), θ0=T(2.0)),
            Molly.HarmonicAngleλ(; k=T(10.0), θ0=T(2.0), λ_kwargs...),
            CosineAngle(k=T(10.0), θ0=T(2.0)),
            Molly.CosineAngleλ(; k=T(10.0), θ0=T(2.0), λ_kwargs...),
            UreyBradley(kangle=T(10.0), θ0=T(2.0), kbond=T(10.0), r0=T(1.0)),
            Molly.UreyBradleyλ(; kangle=T(10.0), θ0=T(2.0), kbond=T(10.0), r0=T(1.0),
                               λ_kwargs...),
            PeriodicTorsion(periodicities=[1, 2], phases=T[1.0, 0.0], ks=T[10.0, 5.0],
                            n_terms=2),
            Molly.PeriodicTorsionλ(; periodicities=(1, 2), phases=(T(1.0), T(0.0)),
                                   ks=(T(10.0), T(5.0)), proper=true, λ_kwargs...),
            RBTorsion(c0=T(1.0), c1=T(2.0), c2=T(3.0), c3=T(4.0), c4=T(0.5), c5=T(0.25)),
            Molly.RBTorsionλ(; c0=T(1.0), c1=T(2.0), c2=T(3.0), c3=T(4.0), c4=T(0.5),
                             c5=T(0.25), λ_kwargs...),
            HarmonicTorsion(k=T(10.0), θ0=T(2.0)),
            Molly.HarmonicTorsionλ(; k=T(10.0), θ0=T(2.0), λ_kwargs...),
            CMAPTorsion(1, 24),
            Molly.CMAPTorsionλ(; index=1, size=24, λ_kwargs...),
            EwaldExclusion(),
            Molly.EwaldExclusionData(T(1.0)),
            Ewald(T(1.0)),
            LJDispersionCorrection(atoms, T(1.0)),
            DEXPDispersionCorrection(atoms, T(1.0), T(16.766), T(4.427)),
            MullerBrown(),
        )
        for inter in inters
            @test widened_equal(inter, Molly.convert_component(c, inter))
        end

        @test Molly.convert_component(c, Dict("C" => T(0.17)u"nm")) == Dict("C" => Float64(T(0.17)))

        # A struct that can not be rebuilt from its fields gives an error naming it
        @test_throws "UnconvertibleInteraction" Molly.convert_component(c,
                                                    UnconvertibleInteraction(T(1.0)))
        ani = ANIPotential(nothing, [], [], Dict{String, Int}(), (;), T[], T(5.2),
                           Ref{Any}(nothing))
        @test_throws "ANIPotential" Molly.convert_component(c, ani)
    end

    @testset "Keyword arguments" begin
        sys = builders[2][2](Float32, Array)
        sys_log = System(sys; loggers=(temp=TemperatureLogger(10),))
        sys_nolog = @test_logs (:warn, r"loggers are dropped") match_mode=:any System(
                        sys_log; float_type=Float64, strictness=:warn)
        @test sys_nolog.loggers == ()
        loggers = (temp=TemperatureLogger(Float64, 10),)
        @test System(sys_log; float_type=Float64, loggers=loggers).loggers === loggers
        # float_type_high keeps the higher precision on the same device
        @test Molly.float_type_high(System(System(sys; float_type=Float64); float_type=Float32)) ==
                Float64
        # Keyword arguments given with a conversion are converted too
        sys_kw = System(sys; float_type=Float64, units=false, coords=sys.coords .* 1.01f0)
        @test widened_equal(sys.coords .* 1.01f0, sys_kw.coords)
        # Without a conversion the copy constructor does not change the values
        @test System(sys; coords=sys.coords).atoms === sys.atoms
        # grad_safe alone sets up PME again for gradients and keeps the loggers
        sys_pme = System(builders[1][2](Float64, Array); loggers=(temp=TemperatureLogger(10),))
        sys_gs = @test_nowarn System(sys_pme; grad_safe=true, strictness=:warn)
        @test sys_gs.general_inters[1].grad_safe
        @test sys_gs.loggers === sys_pme.loggers
        @test isapprox(potential_energy(sys_gs; n_threads=1),
                       potential_energy(sys_pme; n_threads=1); rtol=1e-12)
        # A SHAKE tolerance too small for Float32 gives the warning of the SHAKE constructor
        sys_shake = System(joinpath(data_dir, "water_3mol_triclinic.pdb"), ff_tip3p;
                           nonbonded_method=SetupCoulombReactionField(), dist_cutoff=0.5u"nm",
                           constraints=:hbonds, rigid_water=true,
                           constraint_algorithm=SetupSHAKE_RATTLE(dist_tolerance=1e-8u"nm",
                                                vel_tolerance=1e-8u"nm^2 * ps^-1"),
                           strictness=:nowarn)
        @test_logs (:warn, r"SHAKE dist_tolerance") match_mode=:any System(sys_shake;
                                                    float_type=Float32, strictness=:warn)
        sys_nounits = System(sys; units=false)
        @test_throws ArgumentError System(sys_nounits; units=true)
    end
end
