# Loading the TYK2 test data from OpenFE from artifact
tyk2_dir = LazyArtifacts.ensure_artifact_installed("tyk2_data",
                normpath(@__DIR__, "..", "Artifacts.toml"))

function end_state_map(mapping, core_mapAB, n_end, dual, state)
    h2e, real = Int[], Bool[]
    for i in mapping["core"]
        j = (state == :A ? i : core_mapAB[i])
        append!(h2e, dual ? [j, j] : [j])
        append!(real, dual ? [state == :A, state == :B] : [true])
    end
    for (s, unique) in ((:A, mapping["unique_A"]), (:B, mapping["unique_B"]))
        append!(h2e, s == state ? unique : zeros(Int, length(unique)))
        append!(real, fill(s == state, length(unique)))
    end
    ligand = Set(state == :A ? [mapping["core"]; mapping["unique_A"]] :
                               [[core_mapAB[i] for i in mapping["core"]]; mapping["unique_B"]])
    env = [i for i in 1:n_end if !(i in ligand)]
    return [h2e; env], [real; trues(length(env))]
end

function real_terms(il, real)
    eltype(il.inters) <: Union{EwaldExclusion, CoulombReactionFieldExclusion,
                               Molly.LennardJones14λ} && return il
    atom_fields = fieldnames(typeof(il))[1:(end - 3)]
    keep = [all(real[getfield(il, f)[n]] for f in atom_fields) for n in eachindex(il.inters)]
    return typeof(il).name.wrapper((getfield(il, f)[keep] for f in atom_fields)...,
                                   il.inters[keep], il.types[keep], il.data)
end

function end_state_diff(sys, ref, h2e, real)
    coords = [h2e[i] > 0 ? ref.coords[h2e[i]] : c for (i, c) in enumerate(sys.coords)]
    sys = System(sys; coords=coords,
                 specific_inter_lists=Tuple(real_terms(il, real) for il in sys.specific_inter_lists))
    place_virtual_sites!(sys)
    fs, fs_ref = forces(sys), forces(ref)
    fs_end = zero(fs_ref)
    for i in eachindex(fs)
        h2e[i] > 0 && (fs_end[h2e[i]] += fs[i])
    end
    return (abs(potential_energy(sys) - potential_energy(ref)), maximum(norm.(fs_end .- fs_ref)),
            maximum(norm, fs[h2e .== 0]; init=norm(zero(fs[1]))))
end

function ligand_pdb(file)
    lines = readlines(file)
    serials = Set(l[7:11] for l in lines if startswith(l, "HETATM") && l[18:20] == "UNK")
    keep(l) = (startswith(l, "HETATM") || startswith(l, "CONECT")) && l[7:11] in serials
    out = tempname() * ".pdb"
    write(out, join(filter(keep, lines), "\n") * "\nEND\n")
    return out
end

function tyk2_ligands(; kwargs...)
    build(lig) = System(ligand_pdb(joinpath(data_dir, "tyk2_$lig.pdb")),
                        MolecularForceField(joinpath(data_dir, "$lig.xml"); units=true);
                        boundary=CubicBoundary(Inf*u"nm"), nonbonded_method=DistanceCutoff(1.0u"nm"),
                        dist_cutoff=1.0u"nm", kwargs...)
    sysA, sysB = build("ejm31"), build("ejm50")
    mapping = Dict("unique_A" => [31], "unique_B" => [31, 33], "core" => [i for i in 1:32 if i != 31])
    core_mapAB = Dict(i => findfirst(d -> d.atom_name == sysA.atoms_data[i].atom_name, sysB.atoms_data)
                      for i in mapping["core"])
    core_mapAB[32] = 32
    return sysA, sysB, mapping, core_mapAB
end

const lambda_schedulers = (
    Molly.DefaultLambdaScheduler,
    Molly.LinearLambdaScheduler,
    Molly.GROMACSLambdaABFEScheduler,
    Molly.GROMACSLambdaRBFEScheduler,
    Molly.OpenFEScheduler,
    Molly.NAMDLambdaScheduler,
    Molly.QuartersLambdaScheduler,
    Molly.EleScaledLambdaScheduler,
)
const softcores = (:beutler, :gapsys, :scaled)

@testset "OpenFE TYK2 comparison" begin
    # --- Variables ---
    FT = Float64
    AT = Array

    # --- OpenFE reference results (bonded terms do not depend on λ, so they are stored once) ---
    energy_ref = BSON.load(joinpath(tyk2_dir, "openfe", "energy_openfe.bson"))
    forces_ref = BSON.load(joinpath(tyk2_dir, "openfe", "forces_openfe.bson"))
    bonded = ("bond_only", "angle_only", "torsion_only")
    openfe_ref(ref, inter, λtag) = ref[Symbol(inter in bonded ? inter : "$(inter)_$(λtag)")]

    # --- Force Field Setup ---
    ff_A = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "amber14/protein.ff14SB.xml"])...,
                                            joinpath(data_dir, "ejm31.xml"),
                                            ; units=true)

    ff_B = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "amber14/protein.ff14SB.xml"])...,
                                            joinpath(data_dir, "ejm50.xml")
                                            ; units=true)

    # --- Load Systems ---
    sysA = System(
            joinpath(data_dir, "tyk2_ejm31.pdb"),
            ff_A;
            nonbonded_method=SetupPME(approximate_erfc=false),
            center_coords=false,
            dist_cutoff=FT(0.9)u"nm",
        )

    sysB = System(
            joinpath(data_dir, "tyk2_ejm50.pdb"),
            ff_B;
            nonbonded_method=SetupPME(approximate_erfc=false),
            center_coords=false,
            dist_cutoff=FT(0.9)u"nm",
        )

    mapping = Dict("unique_A"=>[4701], "unique_B" => [4701,4703])
    core = []
    for i in 4671:4702
        if !(i in mapping["unique_A"])
            push!(core, i)
        end
    end
    mapping["core"] = core
    core_mapAB = Dict()
    for (i,a) in enumerate(sysA.atoms_data)
        if i in mapping["core"]
            for (j,b) in enumerate(sysB.atoms_data)
                if a.atom_name == b.atom_name && a.atom_name == b.atom_name
                    core_mapAB[i] = j
                end
            end
        end
    end
    core_mapAB[4702] = 4702

    # --- OpenMM positions, read once ---
    openmm_coords_fp = joinpath(tyk2_dir, "openfe", "positions_openmm.txt")
    coords_openmm = Dict()
    atom_order = Dict()
    chains = "ABCDE"
    res_id = 0
    resi_n = 0
    chain = ""
    for (ind, row) in enumerate(eachrow(readdlm(openmm_coords_fp)))
        ele = split(row[1], ",")
        atom_name = ele[1]
        chain_id = string(chains[parse(Int, ele[2])])
        res_name = ele[3]
        resi = parse(Int, ele[4])
        if chain != chain_id
            chain = chain_id
            res_id = 0
            resi_n = 0
        end
        if resi_n != resi
            res_id += 1
            resi_n = resi
        end
        coords_openmm[(atom_name, res_id, res_name, chain_id)] = (
            SVector{3, FT}(parse(FT, ele[5]), parse(FT, ele[6]), parse(FT, ele[7]))*u"nm")
        atom_order[(atom_name, res_id, res_name, chain_id)] = ind
    end

    let disp(s) = potential_energy(s; pairwise_inters=(), specific_inter_lists=(),
                      general_inters=Tuple(g for g in s.general_inters
                                           if g isa LJDispersionCorrection ||
                                              g isa Molly.LJDispersionCorrectionλ))
        for (λ, ref) in ((0.0, sysA), (1.0, sysB))
            sys_dual = RelativeFESystem(sysA, sysB, FT(λ), mapping, core_mapAB;
                                        scheduler=Molly.DefaultLambdaScheduler(dual=true))
            @test isapprox(disp(sys_dual), disp(ref); rtol=1e-7)
        end
    end

    inters = ("bond_only", "angle_only", "torsion_only", "nonbonded", "PME", "all")

    for (λ, λtag) in ((0.0, "l0"), (0.25, "l25"), (0.5, "l5"), (0.75, "l75"), (1.0, "l1"))
        # --- Hybrid system at the OpenMM positions ---
        sys = RelativeFESystem(sysA, sysB, FT(λ), mapping, core_mapAB;
                               scheduler=Molly.OpenFEScheduler(dual=false),
                               LJsoftcore=:gapsys,
                               Csoftcore=:scaled)

        new_coords = typeof(sys.coords[1])[]
        OP_map_idx = []
        MO_map_idx = []
        for (a, d) in zip(sys.atoms, sys.atoms_data)
            key = (d.atom_name, d.res_number, d.res_name, d.chain_id)
            if haskey(coords_openmm, key)
                push!(new_coords, coords_openmm[key])
                push!(OP_map_idx, atom_order[key])
                push!(MO_map_idx, a.index)
            else
                push!(new_coords, zero(sys.coords[1]))
            end
        end
        new_coords = wrap_coords.(new_coords, (sys.boundary,))
        test_sys = System(sys, coords=new_coords)
        place_virtual_sites!(test_sys)
        neighbors = find_neighbors(test_sys)

        # --- Energy and forces per interaction ---
        for inter in inters
            if inter == "all"
                pin = test_sys.pairwise_inters
            elseif inter == "nonbonded"
                pin = test_sys.pairwise_inters
            else
                pin = ()
            end

            if inter == "all"
                sils = test_sys.specific_inter_lists
            elseif inter == "bond_only"
                sils = test_sys.specific_inter_lists[1:1]
            elseif inter == "angle_only"
                sils = test_sys.specific_inter_lists[2:2]
            elseif inter == "torsion_only"
                sils = test_sys.specific_inter_lists[3:3]
            elseif inter == "nonbonded"
                sils = test_sys.specific_inter_lists[4:4]
            else
                sils = ()
            end

            if inter == "all"
                gis = test_sys.general_inters
            elseif inter == "PME"
                gis = test_sys.general_inters[1:1]
            elseif inter == "nonbonded"
                gis = test_sys.general_inters[2:2]
            else
                gis = ()
            end

            sys_part = System(test_sys,
                pairwise_inters=pin,
                specific_inter_lists=sils,
                general_inters=gis,
            )

            strict = λ > 0
            loose = inter in ("PME", "all") || (strict && inter == "nonbonded")

            forces_molly = forces(sys_part, neighbors; n_threads=1)
            forces_openmm = SVector{3}.(eachrow(openfe_ref(forces_ref, inter, λtag)))u"kJ * mol^-1 * nm^-1"
            ftol = (loose ? (strict ? 1e-5 : 1e-3) : (strict ? 1e-7 : 1e-6))u"kJ * mol^-1 * nm^-1"
            @test maximum(norm.(forces_molly[MO_map_idx] .- forces_openmm[OP_map_idx])) < ftol

            E_molly = potential_energy(sys_part, neighbors)
            E_openmm = openfe_ref(energy_ref, inter, λtag) * u"kJ * mol^-1"
            etol = (loose ? 1e-3 : (strict ? 1e-6 : 1e-4))u"kJ * mol^-1"
            @test abs(E_molly - E_openmm) < etol
        end
    end

end

@testset "CHARMM TYK2 end states" begin
    # CHARMM36 protein with the CGenFF TYK2 ligands: separate 1-4 Lennard-Jones and CMAP terms,
    #   and a lone pair on each aromatic chlorine as a LocalCoordinatesSite
    FT = Float64
    build(lig) = System(
        joinpath(tyk2_dir, "tyk2_charmm_$(lig)_conect.pdb"),
        MolecularForceField(
            joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...,
            joinpath(tyk2_dir, "$(lig)_charmm.xml");
            units=true,
            strictness=:nowarn,
        );
        nonbonded_method=SetupPME(approximate_erfc=false), center_coords=false,
        dist_cutoff=FT(0.9)u"nm",
    )
    sysA, sysB = build("ejm31"), build("ejm50")
    @test length(sysA.virtual_sites) == 2 && length(sysB.virtual_sites) == 2
    @test all(vs -> vs.type == 5, sysA.virtual_sites) # LocalCoordinatesSite

    ligand(sys) = [i for i in eachindex(sys) if sys.atoms_data[i].res_name in ("LIG", "UNK")]
    ligA, ligB = ligand(sysA), ligand(sysB)
    nameA = Dict(sysA.atoms_data[i].atom_name => i for i in ligA)
    nameB = Dict(sysB.atoms_data[i].atom_name => i for i in ligB)
    uniq_A, uniq_B = ["H11"], ["H11", "O3"]
    core = [i for i in ligA if haskey(nameB, sysA.atoms_data[i].atom_name) &&
                               !(sysA.atoms_data[i].atom_name in uniq_A)]
    mapping = Dict("core" => core, "unique_A" => sort([nameA[n] for n in uniq_A]),
                   "unique_B" => sort([nameB[n] for n in uniq_B]))
    core_mapAB = Dict(i => nameB[sysA.atoms_data[i].atom_name] for i in core)

    # A CHARMM protein carries backbone CMAP terms, whose grid single topology can not interpolate
    @test_throws ArgumentError RelativeFESystem(sysA, sysB, FT(0), mapping, core_mapAB;
                        scheduler=Molly.OpenFEScheduler(dual=false), LJsoftcore=:gapsys,
                        Csoftcore=:gapsys)

    for scheduler in (Molly.DefaultLambdaScheduler(dual=true),)
        for (λ, ref, state) in ((0.0, sysA, :A), (1.0, sysB, :B))
            sys = RelativeFESystem(sysA, sysB, FT(λ), mapping, core_mapAB; scheduler=scheduler,
                                   LJsoftcore=:gapsys, Csoftcore=:gapsys)
            inter_types = [eltype(il.inters) for il in sys.specific_inter_lists]
            @test any(T -> T <: Molly.LennardJones14SoftCoreGapsys, inter_types)
            @test any(T -> T <: Molly.CMAPTorsionλ, inter_types)
            # Both lone pairs are carried over, in dual topology once per copy of the core
            n_lp = (scheduler.dual ? 4 : 2)
            @test count(vs -> vs.type == 5, sys.virtual_sites) == n_lp

            dE, dF, F_dummy = end_state_diff(sys, ref, end_state_map(mapping, core_mapAB, length(ref),
                                                                     scheduler.dual, state)...)
            @test dE < 1e-6u"kJ * mol^-1"
            @test dF < 1e-6u"kJ * mol^-1 * nm^-1"
            @test iszero(F_dummy)
        end
    end
end

@testset "End states" begin
    etol, ftol = 1e-6u"kJ * mol^-1", 1e-6u"kJ * mol^-1 * nm^-1"
    solute = collect(1:12)
    ff_solv = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "benzene.xml"])...;
                                  units=true)
    is_disp(g) = g isa LJDispersionCorrection || g isa Molly.LJDispersionCorrectionλ
    without_disp(s) = Tuple(g for g in s.general_inters if !is_disp(g))

    for nonbonded in (SetupPME(), SetupCoulombReactionField(),
                      SetupCoulombReactionField(exclusion_correction=true))
        sys = System(joinpath(data_dir, "benzene_solv.pdb"), ff_solv; nonbonded_method=nonbonded,
                     dist_cutoff=1.0u"nm", center_coords=false)
        E_ref = potential_energy(sys; general_inters=without_disp(sys))
        F_ref = forces(sys; general_inters=without_disp(sys))
        disp(s) = potential_energy(s; pairwise_inters=(), specific_inter_lists=(),
                                   general_inters=Tuple(g for g in s.general_inters if is_disp(g)))
        solvent = setdiff(eachindex(sys.atoms), solute)
        zeroed = [i in solute ? Atom(a.index, a.atom_type, a.mass, zero(a.charge), a.σ, zero(a.ϵ),
                                     a.λ, a.alch_role) : a for (i, a) in enumerate(sys.atoms)]
        F_decoupled = forces(System(sys; atoms=zeroed))

        # Every scheduler and every soft core combination, at both end states. 
        for S in lambda_schedulers, LJsoftcore in softcores, Csoftcore in softcores
            sys_0 = AbsoluteFESystem(sys, 0.0, solute; scheduler=S(dual=true),
                                     LJsoftcore=LJsoftcore, Csoftcore=Csoftcore)
            @test abs(potential_energy(sys_0; general_inters=without_disp(sys_0)) - E_ref) < etol
            @test maximum(norm.(forces(sys_0; general_inters=without_disp(sys_0)) .- F_ref)) < ftol
            sys_1 = AbsoluteFESystem(sys, 1.0, solute; scheduler=S(dual=true),
                                     LJsoftcore=LJsoftcore, Csoftcore=Csoftcore)
            @test maximum(norm.(forces(sys_1)[solvent] .- F_decoupled[solvent])) < ftol
            S(dual=true) isa Molly.OpenFEScheduler || @test abs(disp(sys_0) - disp(sys)) < etol
        end
    end

    sys_c = System(joinpath(data_dir, "benzene_solv.pdb"), ff_solv; nonbonded_method=SetupPME(),
                   dist_cutoff=1.0u"nm", center_coords=false, constraints=:hbonds,
                   constraint_algorithm=SetupLINCS(), rigid_water=true)
    n_cons(s) = sum(length(c.dist_constraints) for c in s.constraints)
    for S in (Molly.DefaultLambdaScheduler, Molly.OpenFEScheduler)
        sys_0 = AbsoluteFESystem(sys_c, 0.0, solute; scheduler=S(dual=true))
        @test n_cons(sys_0) == n_cons(sys_c)
        @test abs(potential_energy(sys_0; general_inters=without_disp(sys_0)) -
                  potential_energy(sys_c; general_inters=without_disp(sys_c))) < etol
    end
end

@testset "OpenFE sigma interpolation" begin
    core = Atom(1, 1, 12.0u"g/mol", (0.0, 0.0), (0.30u"nm", 0.35u"nm"),
                (0.5u"kJ*mol^-1", 0.5u"kJ*mol^-1"), 0.0, Molly.CoreRole)
    dummy(role, λ) = Atom(2, 1, 12.0u"g/mol", (0.0, 0.0), (0.25u"nm", 0.25u"nm"),
                          (0.5u"kJ*mol^-1", 0.5u"kJ*mol^-1"), λ, role)
    mixed(S, role, λ, special) = Molly.λ_params_function(S(dual=false), Molly.MinimumMixing(),
                            Molly.LorentzMixing(), Molly.GeometricMixing(),
                            Atom(core.index, core.atom_type, core.mass,
                            core.charge, core.σ, core.ϵ, λ, core.alch_role), dummy(role, λ),
                            special, core.σ)[4]
    for (role, λ, pinned) in ((Molly.InsertRole, 0.25, 0.3u"nm"), (Molly.DeleteRole, 0.75, 0.275u"nm"))
        @test mixed(Molly.OpenFEScheduler, role, λ, false) ≈ 0.2875u"nm"
        @test mixed(Molly.OpenFEScheduler, role, λ, true) ≈ pinned
        @test mixed(Molly.DefaultLambdaScheduler, role, λ, false) ≈ pinned
    end
end

@testset "Unitless alchemical systems" begin
    # The soft core parameters with units, such as the σQ of the Gapsys Coulomb soft cores,
    #   follow the units of the system
    function benzene_λ(units, nonbonded, Csoftcore)
        ff = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "benzene.xml"])...;
                                 units=units)
        sys = System(joinpath(data_dir, "benzene_solv.pdb"), ff; units=units,
                     nonbonded_method=nonbonded, dist_cutoff=(units ? 1.0u"nm" : 1.0),
                     center_coords=false)
        return AbsoluteFESystem(sys, 0.5, collect(1:12); Csoftcore=Csoftcore)
    end
    for nonbonded in (SetupPME(), SetupCoulombReactionField(),
                      SetupCoulombReactionField(exclusion_correction=true)), Csoftcore in softcores
        E_units = potential_energy(benzene_λ(true, nonbonded, Csoftcore); n_threads=1)
        E_nounits = potential_energy(benzene_λ(false, nonbonded, Csoftcore); n_threads=1)
        @test E_nounits ≈ ustrip(u"kJ * mol^-1", E_units) rtol=1e-12
    end
end

@testset "Double exponential end states" begin
    etol, ftol = 1e-6u"kJ * mol^-1", 1e-6u"kJ * mol^-1 * nm^-1"
    solute = collect(1:9)
    ff = MolecularForceField(joinpath(ff_dir, "ethanol_garnet.xml");
                             custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"))
    for nonbonded in (SetupPME(), SetupCoulombReactionField())
        sys = System(joinpath(data_dir, "ethanol_solv.pdb"), ff; nonbonded_method=nonbonded,
                     dist_cutoff=1.0u"nm", dispersion_correction=true, strictness=:nowarn)
        @test any(inter isa DoubleExponential for inter in values(sys.pairwise_inters))
        E_ref, F_ref = potential_energy(sys), forces(sys)
        solvent = setdiff(eachindex(sys.atoms), solute)
        zeroed = [i in solute ? Atom(a.index, a.atom_type, a.mass, zero(a.charge), a.σ, zero(a.ϵ),
                                     a.λ, a.alch_role) : a for (i, a) in enumerate(sys.atoms)]
        F_decoupled = forces(System(sys; atoms=zeroed))

        # The double exponential has its own soft core, whichever soft core is chosen for
        #   Lennard-Jones
        for S in lambda_schedulers, LJsoftcore in softcores
            sys_0 = AbsoluteFESystem(sys, 0.0, solute; scheduler=S(dual=true), LJsoftcore=LJsoftcore)
            @test any(inter isa DoubleExponentialSoftCore for inter in values(sys_0.pairwise_inters))
            # As in OpenFE, the OpenFE scheduler leaves the alchemical atoms out of the
            #   dispersion correction, so it is compared without it
            if S == Molly.OpenFEScheduler
                no_disp(s) = Tuple(g for g in s.general_inters if !(g isa DExpDispersionCorrection ||
                                                                g isa Molly.DExpDispersionCorrectionλ))
                @test abs(potential_energy(sys_0; general_inters=no_disp(sys_0)) -
                          potential_energy(sys; general_inters=no_disp(sys))) < etol
            else
                @test abs(potential_energy(sys_0) - E_ref) < etol
            end
            @test maximum(norm.(forces(sys_0) .- F_ref)) < ftol
            sys_1 = AbsoluteFESystem(sys, 1.0, solute; scheduler=S(dual=true), LJsoftcore=LJsoftcore)
            @test maximum(norm.(forces(sys_1)[solvent] .- F_decoupled[solvent])) < ftol
        end
    end
end

@testset "Double exponential dispersion correction" begin
    # The soft core changes the shape of the double exponential at every distance, so the
    #   correction is checked against the integrals of the soft-core pair potentials themselves
    α, β, rs, rc = 16.766, 4.427, 0.6, 1.0
    types = [(0.30, 0.5), (0.30, 0.5), (0.25, 0.1), (0.35, 0.8), (0.32, 0.6), (0.28, 0.3)]
    atoms = [Atom(index=i, mass=10.0u"g/mol", σ=(σ * u"nm"), ϵ=(ϵ * u"kJ * mol^-1"))
             for (i, (σ, ϵ)) in enumerate(types)]
    coords = [SVector(0.5 + 1.2 * (i - 1), 1.0, 1.0)u"nm" for i in eachindex(atoms)]
    disp(s) = potential_energy(s; pairwise_inters=(), specific_inter_lists=())
    simpson(f, a, b, n) = (b - a) / 3n * sum((i in (0, n) ? 1 : (isodd(i) ? 4 : 2)) *
                                             f(a + i * (b - a) / n) for i in 0:n)

    for cutoff in (DistanceCutoff(rc * u"nm"), ShiftedPotentialCutoff(rc * u"nm"),
                   PolynomialCutoff(rs * u"nm", rc * u"nm"))
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=CubicBoundary(8.0u"nm"),
            atoms_data=[AtomData(element="C") for _ in atoms],
            pairwise_inters=(DoubleExponential(α=α, β=β, cutoff=cutoff, use_neighbors=false),),
            general_inters=(DExpDispersionCorrection(atoms, cutoff, α, β),),
        )
        for S in (Molly.DefaultLambdaScheduler, Molly.LinearLambdaScheduler,
                  Molly.GROMACSLambdaABFEScheduler)
            # λ = 0 is the coupled end state
            @test disp(AbsoluteFESystem(sys, 0.0, [1, 2]; scheduler=S(dual=true))) ≈
                  disp(sys) rtol=1e-12
            for λ in (0.5, 0.8, 0.95, 1.0)
                sys_λ = AbsoluteFESystem(sys, λ, [1, 2]; scheduler=S(dual=true))
                sc = only(sys_λ.pairwise_inters)
                sc_nocut = DoubleExponentialSoftCore(α=α, β=β, λ_mixing=sc.λ_mixing,
                                                     scheduler=sc.scheduler)
                atoms_λ = Array(sys_λ.atoms)
                # Each atom counts by its steric coupling, so the decoupled solute counts as absent
                n_eff = sum(first(Molly.scale_sterics(sc.scheduler, a.λ, a.alch_role, Val(true)))
                            for a in atoms_λ)
                # Over all pairs including the pairs of each atom with itself, the integral of
                #   r^2 times the potential that the cutoff leaves out, and the potential at the
                #   cutoff that the shifted potential leaves out
                I_sum, V_rc_sum = 0.0, 0.0
                for i in eachindex(atoms_λ), j in i:length(atoms_λ)
                    V(r) = ustrip(potential_energy(sc_nocut, SVector(r, 0.0, 0.0)u"nm",
                                                   atoms_λ[i], atoms_λ[j]))
                    V_cut(r) = ustrip(potential_energy(sc, SVector(r, 0.0, 0.0)u"nm",
                                                       atoms_λ[i], atoms_λ[j]))
                    I_sum += simpson(r -> r^2 * V(r), rc, 13.0, 100_000)
                    if cutoff isa ShiftedPotentialCutoff
                        I_sum += simpson(r -> r^2 * (V(r) - V_cut(r)), 0.0, rc, 10_000)
                        V_rc_sum += V(rc)
                    elseif cutoff isa PolynomialCutoff
                        I_sum += simpson(r -> r^2 * (V(r) - V_cut(r)), rs, rc, 10_000)
                    end
                end
                n_pairs = n_eff * (n_eff + 1) / 2
                # As in GROMACS each atom has one fewer partner within the shifted cutoff
                E_ref = 2π * n_eff^2 / ustrip(volume(sys_λ)) * I_sum / n_pairs -
                        n_eff / 2 * V_rc_sum / n_pairs
                @test ustrip(disp(sys_λ)) ≈ E_ref rtol=1e-9
                @test disp(Molly.set_lambda!(AbsoluteFESystem(sys, 0.0, [1, 2];
                                                              scheduler=S(dual=true)), λ)) ≈
                      disp(sys_λ) rtol=1e-12
            end
        end
    end
end

@testset "Double exponential relative end states" begin
    # State B is the same ethanol with a different charge and double exponential parameters on
    #   the oxygen; the hydroxyl hydrogen is unique to each state
    garnet = joinpath(ff_dir, "ethanol_garnet.xml")
    xml = read(garnet, String)
    pattern_q = r"(<Atom name=\"O1\" type=\"O1\" charge=\")[^\"]*"
    pattern_dexp = r"(<Atom type=\"O1\" sigma=\")0\.31561335921287537(\" epsilon=\")[^\"]*"
    @test occursin(pattern_q, xml) && occursin(pattern_dexp, xml)
    garnet_B = joinpath(mktempdir(), "ethanol_garnet.xml")
    write(garnet_B, replace(xml, pattern_q => s"\g<1>-0.7", pattern_dexp => s"\g<1>0.33\g<2>0.8"))
    mapping = Dict("core" => collect(1:8), "unique_A" => [9], "unique_B" => [9])
    core_mapAB = Dict(i => i for i in 1:8)

    for nonbonded in (SetupPME(), SetupCoulombReactionField(),
                      SetupCoulombReactionField(exclusion_correction=true))
        build(file) = System(
            joinpath(data_dir, "ethanol_solv.pdb"),
            MolecularForceField(file;
                custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"));
            nonbonded_method=nonbonded, dist_cutoff=1.0u"nm", dispersion_correction=true,
            strictness=:nowarn,
        )
        sysA, sysB = build(garnet), build(garnet_B)
        @test abs(potential_energy(sysA) - potential_energy(sysB)) > 1.0u"kJ * mol^-1"
        is_disp(g) = g isa DExpDispersionCorrection || g isa Molly.DExpDispersionCorrectionλ
        no_disp(s) = System(s; general_inters=Tuple(g for g in s.general_inters if !is_disp(g)))
        disp(s) = potential_energy(s; pairwise_inters=(), specific_inter_lists=(),
                                   general_inters=Tuple(g for g in s.general_inters if is_disp(g)))
        for dual in (true, false), (λ, ref, state) in ((0.0, sysA, :A), (1.0, sysB, :B))
            sys = RelativeFESystem(sysA, sysB, λ, deepcopy(mapping), core_mapAB;
                                   scheduler=Molly.DefaultLambdaScheduler(dual=dual))
            @test any(inter isa DoubleExponentialSoftCore for inter in values(sys.pairwise_inters))
            # Single topology counts the dummy atom in the effective number of atoms of the
            #   dispersion correction, so the correction is compared on its own
            @test disp(sys) ≈ disp(ref) rtol=1e-7
            dE, dF, F_dummy = end_state_diff(no_disp(sys), no_disp(ref),
                                    end_state_map(mapping, core_mapAB, length(ref), dual, state)...)
            @test dE < 1e-6u"kJ * mol^-1"
            @test dF < 1e-6u"kJ * mol^-1 * nm^-1"
            @test iszero(F_dummy)
        end
        if nonbonded isa SetupCoulombReactionField && nonbonded.exclusion_correction
            # The charged hydroxyl hydrogens of the two states get no correction between them
            for dual in (true, false)
                sys = RelativeFESystem(sysA, sysB, 0.5, deepcopy(mapping), core_mapAB;
                                       scheduler=Molly.LinearLambdaScheduler(dual=dual))
                il = only(il for il in sys.specific_inter_lists
                          if il.data isa CoulombReactionFieldExclusionData)
                roles = [atom.alch_role for atom in sys.atoms]
                @test !any(minmax(roles[i], roles[j]) == minmax(Molly.InsertRole, Molly.DeleteRole)
                           for (i, j) in zip(il.is, il.js))
            end

            sysB_off = System(
                joinpath(data_dir, "ethanol_solv.pdb"),
                MolecularForceField(garnet_B;
                    custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"));
                nonbonded_method=SetupCoulombReactionField(), dist_cutoff=1.0u"nm",
                strictness=:nowarn,
            )
            @test_throws ArgumentError RelativeFESystem(sysA, sysB_off, 0.0, deepcopy(mapping),
                                                        core_mapAB)
            @test_throws ArgumentError RelativeFESystem(sysB_off, sysA, 0.0, deepcopy(mapping),
                                                        core_mapAB)
        end
    end
end

@testset "Dual topology parameters" begin
    # Both end states are ethanol, so a parameter is shared by the A and B copies of an atom
    garnet = joinpath(ff_dir, "ethanol_garnet.xml")
    xml = read(garnet, String)
    pattern = r"(<Atom name=\"C1\" type=\"C1\" charge=\")[^\"]*"
    @test occursin(pattern, xml)
    garnet_mod = joinpath(mktempdir(), "ethanol_garnet.xml")
    write(garnet_mod, replace(xml, pattern => s"\g<1>-0.6"))
    build(file) = System(
        joinpath(data_dir, "ethanol_solv.pdb"),
        MolecularForceField(file;
            custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"));
        nonbonded_method=SetupPME(), dist_cutoff=1.0u"nm", strictness=:nowarn,
    )
    sys_ref, sys_ref_mod = build(garnet), build(garnet_mod)
    mapping = Dict("core" => collect(1:8), "unique_A" => [9], "unique_B" => [9])
    core_mapAB = Dict(i => i for i in 1:8)
    relative(sys, λ; dual=true) = RelativeFESystem(sys, sys, λ, deepcopy(mapping), core_mapAB;
                                       scheduler=Molly.DefaultLambdaScheduler(dual=dual))

    for λ in (0.0, 0.5, 1.0)
        sys, sys_rebuilt = relative(sys_ref, λ), relative(sys_ref_mod, λ)
        params_dic = extract_parameters(sys)
        # The B copies of the core atoms have zero mass, so their types have no mass key
        @test !haskey(params_dic, "atom_C1_mass")
        @test haskey(params_dic, "atom_H7_mass")
        @test inject_gradients(sys, params_dic).atoms == sys.atoms
        sys_mod = inject_gradients(sys, merge(params_dic, Dict("atom_C1_charge" => -0.6)))
        @test sys_mod.atoms == sys_rebuilt.atoms
        @test count(a -> a.charge == -0.6, sys_mod.atoms) == 2
        @test ustrip(potential_energy(sys_mod)) ≈ ustrip(potential_energy(sys_rebuilt)) rtol=1e-12
    end

    sys_single = relative(sys_ref, 0.5; dual=false)
    @test_throws ArgumentError extract_parameters(sys_single)
    @test_throws ArgumentError ParameterPlan(sys_single, Dict{String, Float64}())
end

@testset "Alchemical systems without a neighbor finder" begin
    ff = MolecularForceField(joinpath(ff_dir, "ethanol_garnet.xml");
                             custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"))
    sys = System(joinpath(data_dir, "ethanol_solv.pdb"), ff; dist_cutoff=1.0u"nm",
                 neighbor_finder_type=Nothing, strictness=:nowarn)

    # The hybrid system excludes the pairs between the A and B sides with the neighbor finder
    mapping = Dict("core" => collect(1:8), "unique_A" => [9], "unique_B" => [9])
    core_mapAB = Dict(i => i for i in 1:8)
    for dual in (true, false)
        @test_throws ArgumentError RelativeFESystem(sys, sys, 0.5, deepcopy(mapping), core_mapAB;
                                        scheduler=Molly.DefaultLambdaScheduler(dual=dual))
    end

    # An absolute system keeps the pairs of the original system, so it does not need one
    is_disp(g) = g isa LJDispersionCorrection || g isa Molly.LJDispersionCorrectionλ
    pe_no_disp(s) = potential_energy(s; general_inters=Tuple(g for g in s.general_inters
                                                              if !is_disp(g)))
    sys_0 = AbsoluteFESystem(sys, 0.0, collect(1:9))
    @test isnothing(sys_0.neighbor_finder)
    @test isapprox(pe_no_disp(sys_0), pe_no_disp(sys); rtol=1e-10)
end

@testset "Alchemical parameter keys" begin
    # A key of the original system whose interactions include one between alchemical atoms,
    #   with a nonzero value
    function alchemical_key(sys, prefix, alchemical)
        for il in sys.specific_inter_lists
            idx = [getfield(il, f) for f in fieldnames(typeof(il))[1:(end - 3)]]
            inters = Array(il.inters)
            for n in eachindex(inters)
                all(is -> is[n] in alchemical, idx) || continue
                p = Molly.parameter_prefix(inters[n], il.types[n])
                (isnothing(p) || !startswith(p, prefix)) && continue
                for (suffix, v) in zip(Molly.parameter_keys(inters[n]),
                                       Molly.parameter_values(inters[n]))
                    iszero(ustrip(v)) || return p * suffix
                end
            end
        end
        error("no $prefix key on the alchemical atoms")
    end

    pe(s) = potential_energy(s; n_threads=1)
    fs(s) = forces(s; n_threads=1)

    # `build(sys)` makes the alchemical system from an original system; `ref` is the original
    #   system whose keys must all be kept
    function check_keys(ref, build, alchemical, prefixes)
        sys = build(ref)
        params_ref = extract_parameters(ref; strictness=:nowarn)
        params = extract_parameters(sys; strictness=:nowarn)
        # The B copies of the core atoms of a dual topology system have zero mass
        @test all(k -> haskey(params, k) || endswith(k, "_mass"), keys(params_ref))
        @test all(k -> params[k] == params_ref[k], intersect(keys(params), keys(params_ref)))

        sys_rt = inject_gradients(sys, params)
        @test sys_rt.atoms == sys.atoms
        @test pe(sys_rt) == pe(sys)
        @test fs(sys_rt) == fs(sys)

        # Changing a bonded, torsion or 1-4 key equals building the alchemical system from the
        #   changed original system
        changed = Dict(k => 1.1 * params_ref[k]
                       for k in (alchemical_key(ref, p, alchemical) for p in prefixes))
        sys_mod = inject_gradients(sys, merge(params, changed))
        sys_rebuilt = build(inject_gradients(ref, merge(params_ref, changed)))
        @test ustrip(pe(sys_mod)) ≈ ustrip(pe(sys_rebuilt)) rtol=1e-12
        @test ustrip.(fs(sys_mod)) ≈ ustrip.(fs(sys_rebuilt)) rtol=1e-12
        @test pe(sys_mod) != pe(sys)
    end

    ff = MolecularForceField(joinpath(ff_dir, "ethanol_garnet.xml");
                             custom_residue_templates=joinpath(ff_dir, "ethanol_garnet_residues.xml"))
    ethanol = System(joinpath(data_dir, "ethanol_solv.pdb"), ff; nonbonded_method=SetupPME(),
                     dist_cutoff=1.0u"nm", strictness=:nowarn)
    solute = collect(1:9)
    garnet_prefixes = ("inter_HB_", "inter_HA_", "inter_PT_")
    for λ in (0.0, 0.5, 1.0)
        check_keys(ethanol, s -> AbsoluteFESystem(s, λ, solute), solute, garnet_prefixes)
    end
    mapping = Dict("core" => collect(1:8), "unique_A" => [9], "unique_B" => [9])
    core_mapAB = Dict(i => i for i in 1:8)
    check_keys(ethanol, s -> RelativeFESystem(s, s, 0.5, deepcopy(mapping), core_mapAB),
               solute, garnet_prefixes)

    # End states that parametrise a bond type differently: its keys are left out, and refused
    #   by the plan, so that injecting a dictionary does not overwrite the B bonds with A values
    garnet = joinpath(ff_dir, "ethanol_garnet.xml")
    xml = read(garnet, String)
    pattern = r"(<Bond type1=\"C1\" type2=\"C2\" length=\"[^\"]*\" k=\")[^\"]*"
    @test occursin(pattern, xml)
    garnet_B = joinpath(mktempdir(), "ethanol_garnet.xml")
    write(garnet_B, replace(xml, pattern => s"\g<1>150000.0"))
    ethanol_B = System(joinpath(data_dir, "ethanol_solv.pdb"),
                       MolecularForceField(garnet_B; custom_residue_templates=joinpath(ff_dir,
                                                         "ethanol_garnet_residues.xml"));
                       nonbonded_method=SetupPME(), dist_cutoff=1.0u"nm", strictness=:nowarn)
    sys_AB = RelativeFESystem(ethanol, ethanol_B, 0.5, deepcopy(mapping), core_mapAB)
    k_key = only(k for k in keys(extract_parameters(ethanol)) if
                     occursin(r"^inter_HB_(C1/C2|C2/C1)_k$", k))
    params_AB = @test_logs (:warn, Regex(k_key)) extract_parameters(sys_AB)
    @test !haskey(params_AB, k_key)
    @test haskey(params_AB, replace(k_key, r"_k$" => "_r0"))
    @test_throws ErrorException extract_parameters(sys_AB; strictness=:error)
    @test_throws ArgumentError ParameterPlan(sys_AB, Dict(k_key => 1.0))
    @test pe(inject_gradients(sys_AB, params_AB)) == pe(sys_AB)

    # CHARMM36 protein with a CGenFF ligand: separate 1-4 Lennard-Jones, harmonic (improper)
    #   torsions and CMAP terms, with every 1-4 soft core
    build_tyk2(lig) = System(
        joinpath(tyk2_dir, "tyk2_charmm_$(lig)_conect.pdb"),
        MolecularForceField(
            joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...,
            joinpath(tyk2_dir, "$(lig)_charmm.xml");
            units=true,
            strictness=:nowarn,
        );
        nonbonded_method=SetupPME(approximate_erfc=false), center_coords=false,
        dist_cutoff=0.9u"nm",
    )
    sysA = build_tyk2("ejm31")
    ligand = [i for i in eachindex(sysA) if sysA.atoms_data[i].res_name in ("LIG", "UNK")]
    charmm_prefixes = ("inter_HB_", "inter_HA_", "inter_PT_", "inter_LJ14_")
    for (λ, LJsoftcore) in ((0.0, :gapsys), (0.5, :beutler), (1.0, :scaled))
        check_keys(sysA, s -> AbsoluteFESystem(s, λ, ligand; LJsoftcore=LJsoftcore), ligand,
                   charmm_prefixes)
    end
    check_keys(sysA, s -> RelativeFESystem(s, s, 0.5,
                   Dict("core" => ligand, "unique_A" => Int[], "unique_B" => Int[]),
                   Dict(i => i for i in ligand)), ligand, charmm_prefixes)
end
