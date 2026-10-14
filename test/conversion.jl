# Whether `b`, part of a converted system, holds the values of `a` widened to Float64 with the
#   units stripped
widened_equal(a::AbstractFloat, b) = (b isa Float64 && b == Float64(a))
widened_equal(a::Unitful.Quantity, b) = widened_equal(ustrip(a), b)
widened_equal(a::Unitful.Units, b) = (b == NoUnits)
widened_equal(a::Union{Tuple, NamedTuple}, b) = (length(a) == length(b) &&
                                                 all(map(widened_equal, Tuple(a), Tuple(b))))
function widened_equal(a::AbstractArray, b)
    return size(a) == size(b) && all(map(widened_equal, from_device(a), from_device(b)))
end
widened_equal(a::Union{Integer, AbstractString, Symbol, Nothing, Function}, b) = isequal(a, b)

function widened_equal(a::A, b::B) where {A, B}
    if A.name.wrapper !== B.name.wrapper
        return false
    end
    return all(i -> widened_equal(getfield(a, i), getfield(b, i)), 1:fieldcount(A))
end

max_force_diff(fs, fs_ref) = maximum(norm.(ustrip_vec.(from_device(fs)) .-
                                           ustrip_vec.(from_device(fs_ref))))
max_force(fs) = maximum(norm.(ustrip_vec.(from_device(fs))))

# An interaction whose constructor does not take its fields
struct UnconvertibleInteraction{T} <: PairwiseInteraction
    a::T
    b::T

    UnconvertibleInteraction(a) = new{typeof(a)}(a, 2a)
end

@testset "System conversion" begin
    ff_water = MolecularForceField(joinpath(ff_dir, "tip4pfb.xml"))
    ff_dexp = MolecularForceField(joinpath(ff_dir, "ethanol_garnet.xml"))
    ff_protein = MolecularForceField(joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...;
                                     strictness=:nowarn)
    ff_implicit = MolecularForceField(joinpath(ff_dir, "ff99SBildn.xml"))

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
        ("protein PME with constraints", (FT, AT) -> System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff_protein;
            array_type=AT,
            float_type=FT,
            nonbonded_method=SetupPME(),
            center_coords=false,
            constraints=:hbonds,
            rigid_water=true,
        )),
        ("protein implicit solvent", (FT, AT) -> System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff_implicit;
            boundary=CubicBoundary(FT(100.0)u"nm"),
            array_type=AT,
            float_type=FT,
            dist_cutoff=5.0u"nm",
            nonbonded_method=DistanceCutoff(5.0u"nm"),
            dispersion_correction=false,
            implicit_solvent=SetupImplicitSolventGBN2(kappa=1.0u"nm^-1"),
            strictness=:nowarn,
            n_threads=1,
        )),
    )

    # Tolerance for comparing the energy of a system converted from Float32 to the Float32
    #   energy, which is limited by the error of the Float32 calculation. The steep repulsion
    #   of the double exponential potential gives a Float32 error of about 8e-6 for the DEXP
    #   term of the ethanol system.
    rtols_float32 = Dict("DEXP reaction field" => 1e-5)

    for (name, build) in builders, FT in (Float64, Float32)
        sys = build(FT, Array)
        sys_conv = System(sys; float_type=Float64, units=false, array_type=Array, grad_safe=true)
        @test float_type(sys_conv) == Float64
        @test Molly.float_type_high(sys_conv) == Float64
        @test array_type(sys_conv) == Array
        @test sys_conv.energy_units == NoUnits
        @test sys_conv.force_units == NoUnits
        @test eltype(eltype(sys_conv.coords)) == Float64
        @test sys_conv.grad_safe
        @test sys_conv.df == sys.df
        @test length(sys_conv.constraints) == length(sys.constraints)
        # The neighbor finder keeps the storage of its matrices, sparse or dense
        @test typeof(sys_conv.neighbor_finder).name.wrapper ===
                typeof(sys.neighbor_finder).name.wrapper
        @test typeof(sys_conv.neighbor_finder.eligible).name.wrapper ===
                typeof(sys.neighbor_finder.eligible).name.wrapper
        @test Molly.neighbor_finder_masks(sys_conv.neighbor_finder, length(sys)) ==
                Molly.neighbor_finder_masks(sys.neighbor_finder, length(sys))

        E = potential_energy(sys; n_threads=1)
        E_conv = potential_energy(sys_conv; n_threads=1)
        fs, fs_conv = forces(sys; n_threads=1), forces(sys_conv; n_threads=1)
        if FT == Float64
            @test isapprox(E_conv, ustrip(E); rtol=1e-12)
            @test max_force_diff(fs_conv, fs) < 1e-10 * max_force(fs)
        else
            # The conversion can not recover the precision lost in the original values, so
            #   the energy is compared to the Float32 one, and the values exactly
            @test isapprox(E_conv, ustrip(E); rtol=get(rtols_float32, name, 1e-6))
            # Converting to Float64 and back gives the original energy and forces. Units are
            #   kept as the unitless implicit solvent calculation takes a different Float32
            #   path.
            sys_back = System(System(sys; float_type=Float64); float_type=Float32)
            @test float_type(sys_back) == Float32
            @test isapprox(potential_energy(sys_back; n_threads=1), E; rtol=1e-8)
            @test max_force_diff(forces(sys_back; n_threads=1), fs) < 1e-6 * max_force(fs)
            @test widened_equal(sys.atoms, sys_conv.atoms)
            @test widened_equal(sys.coords, sys_conv.coords)
            @test widened_equal(sys.velocities, sys_conv.velocities)
            @test widened_equal(sys.boundary, sys_conv.boundary)
            @test widened_equal(sys.k, sys_conv.k)
            @test widened_equal(sys.pairwise_inters, sys_conv.pairwise_inters)
            @test widened_equal(sys.specific_inter_lists, sys_conv.specific_inter_lists)
        end
    end

    # GPU to CPU and back
    for AT in array_list
        AT == Array && continue
        for (name, build) in builders[1:2]
            sys = build(Float32, AT)
            sys_cpu = System(sys; float_type=Float64, units=false, array_type=Array,
                             grad_safe=true)
            @test array_type(sys_cpu) == Array
            @test sys_cpu.coords isa Array
            # A change of device gives the neighbor finder that setup uses on the new device
            @test sys_cpu.neighbor_finder isa CellListMapNeighborFinder
            @test Molly.neighbor_finder_masks(sys_cpu.neighbor_finder, length(sys)) ==
                    Molly.neighbor_finder_masks(sys.neighbor_finder, length(sys))
            E = potential_energy(sys)
            @test isapprox(potential_energy(sys_cpu), ustrip(E);
                           rtol=get(rtols_float32, name, 1e-6))
            @test widened_equal(sys.atoms, sys_cpu.atoms)
            @test widened_equal(sys.coords, sys_cpu.coords)
            @test widened_equal(sys.pairwise_inters, sys_cpu.pairwise_inters)
            @test widened_equal(sys.specific_inter_lists, sys_cpu.specific_inter_lists)

            sys_gpu = System(sys_cpu; float_type=Float32, array_type=AT, grad_safe=false)
            @test array_type(sys_gpu) == AT
            @test float_type(sys_gpu) == Float32
            @test typeof(sys_gpu.neighbor_finder).name.wrapper ===
                    typeof(sys.neighbor_finder).name.wrapper
            @test isapprox(potential_energy(sys_gpu), ustrip(E); rtol=1e-5)
        end

        # A GPUNeighborFinder can not be used with grad_safe, so the one setup chooses for a
        #   gradient safe system is used
        sys = builders[2][2](Float32, AT)
        sys_gs = System(sys; float_type=Float64, grad_safe=true)
        @test array_type(sys_gs) == AT
        @test !(sys_gs.neighbor_finder isa GPUNeighborFinder)
        @test Molly.neighbor_finder_masks(sys_gs.neighbor_finder, length(sys)) ==
                Molly.neighbor_finder_masks(sys.neighbor_finder, length(sys))
        @test isapprox(potential_energy(sys_gs), potential_energy(sys);
                       rtol=rtols_float32["DEXP reaction field"])

        # GPUCellListNeighborFinder is kept on the same GPU and replaced on the CPU. The water
        #   box is too small for a cell list.
        sys = System(joinpath(data_dir, "ethanol_solv.pdb"), ff_dexp; array_type=AT,
                     float_type=Float32, nonbonded_method=SetupCoulombReactionField(),
                     dist_cutoff=1.0u"nm", neighbor_finder_type=GPUCellListNeighborFinder)
        E = potential_energy(sys)
        sys_64 = System(sys; float_type=Float64)
        @test sys_64.neighbor_finder isa GPUCellListNeighborFinder
        @test array_type(sys_64) == AT
        @test isapprox(potential_energy(sys_64), E; rtol=rtols_float32["DEXP reaction field"])
        sys_cpu = System(sys; float_type=Float64, units=false, array_type=Array)
        @test sys_cpu.neighbor_finder isa CellListMapNeighborFinder
        @test Molly.neighbor_finder_masks(sys_cpu.neighbor_finder, length(sys)) ==
                Molly.neighbor_finder_masks(sys.neighbor_finder, length(sys))
        @test isapprox(potential_energy(sys_cpu), ustrip(E);
                       rtol=rtols_float32["DEXP reaction field"])
    end

    # Alchemical interactions
    ff_solv = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "benzene.xml"])...)
    sys_solv = System(joinpath(data_dir, "benzene_solv.pdb"), ff_solv; nonbonded_method=SetupPME(),
                      dist_cutoff=1.0u"nm", center_coords=false)
    for scheduler in (DefaultLambdaScheduler(dual=true), GROMACSLambdaABFEScheduler(dual=true)),
            softcore in (:beutler, :gapsys)
        sys = AbsoluteFESystem(sys_solv, 0.5, collect(1:12); scheduler=scheduler,
                               LJsoftcore=softcore, Csoftcore=softcore)
        sys_conv = System(sys; units=false)
        @test isapprox(potential_energy(sys_conv; n_threads=1),
                       ustrip(potential_energy(sys; n_threads=1)); rtol=1e-12)
        @test widened_equal(sys.pairwise_inters, sys_conv.pairwise_inters)
        @test widened_equal(sys.specific_inter_lists, sys_conv.specific_inter_lists)
    end

    # Each interaction type converts
    T = Float32
    c = Molly.SystemConversion{Float64, Array, Float64}(true, false)
    atoms = [Atom(index=i, mass=T(10.0), charge=T(0.1), σ=T(0.3), ϵ=T(0.4)) for i in 1:4]
    cutoff = DistanceCutoff(T(1.0))
    cc = T(ustrip(Molly.coulomb_const))
    λ_kwargs = (λ_mixing=Molly.MinimumMixing(), scheduler=DefaultLambdaScheduler())
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
        DPDInteraction(a=T(25.0), γ=T(4.5), σ=T(3.0), r_c=T(1.0), dt=T(0.01), key=UInt64(1)),
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
        Molly.UreyBradleyλ(; kangle=T(10.0), θ0=T(2.0), kbond=T(10.0), r0=T(1.0), λ_kwargs...),
        PeriodicTorsion(periodicities=[1, 2], phases=T[1.0, 0.0], ks=T[10.0, 5.0], n_terms=2),
        Molly.PeriodicTorsionλ(; periodicities=(1, 2), phases=(T(1.0), T(0.0)), ks=(T(10.0), T(5.0)),
                         proper=true, λ_kwargs...),
        RBTorsion(c0=T(1.0), c1=T(2.0), c2=T(3.0), c3=T(4.0), c4=T(0.5), c5=T(0.25)),
        Molly.RBTorsionλ(; c0=T(1.0), c1=T(2.0), c2=T(3.0), c3=T(4.0), c4=T(0.5), c5=T(0.25),
                   λ_kwargs...),
        HarmonicTorsion(k=T(10.0), θ0=T(2.0)),
        Molly.HarmonicTorsionλ(; k=T(10.0), θ0=T(2.0), λ_kwargs...),
        CMAPTorsion(1, 24),
        Molly.CMAPTorsionλ(; index=1, size=24, λ_kwargs...),
        EwaldExclusion(),
        Molly.EwaldExclusionData(T(1.0)),
        Ewald(T(1.0)),
        LJDispersionCorrection(atoms, T(1.0)),
        MullerBrown(),
    )
    for inter in inters
        inter_conv = Molly.convert_component(c, inter)
        @test widened_equal(inter, inter_conv)
    end
    @test Molly.convert_component(c, MullerBrown()).energy_units == NoUnits

    # A struct that can not be rebuilt from its fields gives an error naming it
    @test_throws "UnconvertibleInteraction" Molly.convert_component(c,
                                                UnconvertibleInteraction(T(1.0)))

    # Settings of the converted system
    sys = builders[2][2](Float32, Array)
    sys_log = System(sys; loggers=(temp=TemperatureLogger(10),))
    @test System(sys_log; float_type=Float64).loggers == ()
    loggers = (temp=TemperatureLogger(Float64, 10),)
    @test System(sys_log; float_type=Float64, loggers=loggers).loggers === loggers
    sys_units = System(sys; float_type=Float64)
    @test sys_units.energy_units == sys.energy_units
    @test eltype(eltype(sys_units.coords)) <: Unitful.Quantity{Float64}
    @test Molly.float_type_high(System(sys_units; float_type=Float32)) == Float64
    # Without a conversion the copy constructor does not change the values
    @test System(sys; grad_safe=true).atoms === sys.atoms
    sys_nounits = System(sys; units=false)
    @test float_type(sys_nounits) == Float32
    @test_throws ArgumentError System(sys_nounits; units=true)

    # Gradients of the energy of a converted system
    ff_vac = MolecularForceField(joinpath(ff_dir, "ethanol_garnet.xml"))
    sys_vac = System(joinpath(data_dir, "ethanol_vac.pdb"), ff_vac; float_type=Float32,
                     nonbonded_method=SetupCoulombReactionField(), dist_cutoff=1.0u"nm",
                     strictness=:nowarn)
    sys_ref = System(sys_vac; float_type=Float64, units=false, grad_safe=true)
    params_dic = extract_parameters(sys_ref)
    plan = ParameterPlan(sys_ref, params_dic)
    neighbors = find_neighbors(sys_ref; n_threads=1)
    function pe_injected(params_dic, sys_ref, plan, coords, neighbors)
        sys = inject_gradients(sys_ref, params_dic, plan, coords)
        return potential_energy(sys, neighbors; n_threads=1)
    end
    grads = Dict(k => zero(v) for (k, v) in params_dic)
    autodiff(
        set_runtime_activity(Reverse),
        Const(pe_injected),
        Active,
        Duplicated(params_dic, grads),
        Const(sys_ref),
        Const(plan),
        Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
        Const(neighbors),
    )
    keys_checked = filter(k -> occursin("DEXP", k) || endswith(k, "_σ"), collect(keys(params_dic)))
    @test !isempty(keys_checked)
    for key in keys_checked
        grad_fd = central_fdm(6, 1)(params_dic[key]) do val
            dic = copy(params_dic)
            dic[key] = val
            pe_injected(dic, sys_ref, plan, copy(sys_ref.coords), neighbors)
        end
        @test isapprox(grads[key], grad_fd; rtol=1e-6, atol=1e-8)
    end
end
