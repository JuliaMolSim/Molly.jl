# --- End state helpers ---
# Hybrid atom -> end state atom (0 for a dummy), in the atom order of RelativeFESystem: core atoms
#   (an A and a B copy each in dual topology), atoms unique to A, to B, environment. `real` marks
#   the atoms of the end state. `n_end` is the number of atoms of the end state system.
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

# The bonded terms of the dummy atoms stay on at the end states and are left out; the nonbonded
#   terms of a dummy are off at the end states
function real_terms(il, real)
    eltype(il.inters) <: Union{EwaldExclusion, Molly.LennardJones14λ} && return il
    atom_fields = fieldnames(typeof(il))[1:(end - 3)]
    keep = [all(real[getfield(il, f)[n]] for f in atom_fields) for n in eachindex(il.inters)]
    return typeof(il).name.wrapper((getfield(il, f)[keep] for f in atom_fields)...,
                                   il.inters[keep], il.types[keep], il.data)
end

# Energy and force differences of a hybrid system at an end state to the end state system `ref`,
#   at the positions of `ref`, and the largest force on a dummy atom. The forces on the two copies
#   of a core atom in dual topology are summed.
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

# The TYK2 ligand of a complex PDB file alone, for small systems in vacuum
function ligand_pdb(file)
    lines = readlines(file)
    serials = Set(l[7:11] for l in lines if startswith(l, "HETATM") && l[18:20] == "UNK")
    keep(l) = (startswith(l, "HETATM") || startswith(l, "CONECT")) && l[7:11] in serials
    out = tempname() * ".pdb"
    write(out, join(filter(keep, lines), "\n") * "\nEND\n")
    return out
end

# ejm31 (A) and ejm50 (B) in vacuum, with the mapping of the OpenFE TYK2 test
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

const lambda_schedulers = (DefaultLambdaScheduler, LinearLambdaScheduler, GROMACSLambdaABFEScheduler,
                           GROMACSLambdaRBFEScheduler, OpenFEScheduler, NAMDLambdaScheduler,
                           QuartersLambdaScheduler, EleScaledLambdaScheduler)
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
                                        scheduler=DefaultLambdaScheduler(dual=true))
            @test isapprox(disp(sys_dual), disp(ref); rtol=1e-7)
        end
    end

    inters = ("bond_only", "angle_only", "torsion_only", "nonbonded", "PME", "all")

    for (λ, λtag) in ((0.0, "l0"), (0.25, "l25"), (0.5, "l5"), (0.75, "l75"), (1.0, "l1"))
        # --- Hybrid system at the OpenMM positions ---
        sys = RelativeFESystem(sysA, sysB, FT(λ), mapping, core_mapAB;
                               scheduler=OpenFEScheduler(dual=false),
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
    build(lig) = System(joinpath(tyk2_dir, "tyk2_charmm_$(lig)_conect.pdb"),
                        MolecularForceField(joinpath.(ff_dir, ["charmm36.xml", "charmm36_water.xml"])...,
                                            joinpath(tyk2_dir, "$(lig)_charmm.xml"); units=true);
                        nonbonded_method=SetupPME(approximate_erfc=false), center_coords=false,
                        dist_cutoff=FT(0.9)u"nm")
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
                        scheduler=OpenFEScheduler(dual=false), LJsoftcore=:gapsys,
                        Csoftcore=:gapsys)

    for scheduler in (DefaultLambdaScheduler(dual=true),)
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

    for nonbonded in (SetupPME(), SetupCoulombReactionField())
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
            S(dual=true) isa OpenFEScheduler || @test abs(disp(sys_0) - disp(sys)) < etol
        end
    end

    sys_c = System(joinpath(data_dir, "benzene_solv.pdb"), ff_solv; nonbonded_method=SetupPME(),
                   dist_cutoff=1.0u"nm", center_coords=false, constraints=:hbonds,
                   constraint_algorithm=SetupLINCS(), rigid_water=true)
    n_cons(s) = sum(length(c.dist_constraints) for c in s.constraints)
    for S in (DefaultLambdaScheduler, OpenFEScheduler)
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
        @test mixed(OpenFEScheduler, role, λ, false) ≈ 0.2875u"nm"
        @test mixed(OpenFEScheduler, role, λ, true) ≈ pinned
        @test mixed(DefaultLambdaScheduler, role, λ, false) ≈ pinned
    end
end
