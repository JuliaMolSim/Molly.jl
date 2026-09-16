@testset "OpenFE TYK2 comparison" begin
    # --- Variables ---
    FT = Float64
    AT = Array

    # --- Force Field Setup ---
    ff_A = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "amber14/protein.ff14SB.xml"])...,
                                            joinpath(data_dir, "ejm31.xml"),
                                            ; units=true)

    ff_B = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "amber14/protein.ff14SB.xml"])...,
                                            joinpath(data_dir, "ejm50.xml")
                                            ; units=true)

    # --- Load Systems ---
    sysA = System(
            joinpath(data_dir,"tyk2_ejm31.pdb"),
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

    # --- Hybrid System Setup ---
    sys = RelativeFESystem(sysA, sysB, FT(0.0), mapping, core_mapAB;
                            scheduler=Molly.OpenFEScheduler(dual=false),
                            array_type=AT, 
                            float_type=FT, 
                            LJsoftcore="gapsys",
                            Csoftcore="scaled"
                            )

    #--- Load Positions ---
    openmm_coords_fp = joinpath(tyk2_dir, "openfe", "positions_openmm.txt")
    coords_openmm = Dict()
    atom_order = Dict()
    chains = "ABCDE"
    res_id = 0
    resi_n = 0
    chain = ""
    counter = 0
    ind = 1
    for row in eachrow(readdlm(openmm_coords_fp))
        ele = split(row[1], ",")
        atom_name = ele[1]
        chain_id = string(chains[parse(Int, ele[2])])
        res_name = ele[3]
        resi = parse(Int, ele[4])
        if chain!=chain_id
            chain = chain_id
            res_id = 0
            resi_n = 0
        end
        if resi_n!=resi
            res_id +=1
            resi_n = resi
        end
        coords_openmm[(atom_name, res_id, res_name, chain_id)] = (
            SVector{3, FT}(parse(FT,ele[5]),parse(FT,ele[6]),parse(FT,ele[7]))*u"nm")
        atom_order[(atom_name, res_id, res_name, chain_id)] = ind
        ind += 1
    end

    new_coords = typeof(sys.coords[1])[]
    OP_map_idx = []
    MO_map_idx = []
    for (a,d,c) in zip(sys.atoms, sys.atoms_data, sys.coords)
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

    test_sys = System(sys,
                    coords=new_coords,)
    place_virtual_sites!(test_sys)
    neighbors = find_neighbors(test_sys)

    # --- Basic Benchmark ---
    bench_result = @benchmark potential_energy($test_sys, $neighbors; n_threads=1)
    # @test bench_result.allocs <= 11
    # @test bench_result.memory <= 496
    forces_t = Molly.zero_forces(test_sys)
    buffers = Molly.init_buffers!(test_sys, 1)
    bench_result = @benchmark Molly.forces!($forces_t, $test_sys, $neighbors, 0, $buffers, Val(false);
                                            n_threads=1)
    # @test bench_result.allocs <= 17
    # @test bench_result.memory <= 864

    inters = (
        "bond_only", "angle_only", "torsion_only", "nonbonded", "PME", "all"
    )

    # --- Energy and Forces for λ=0 ---
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

        forces_molly = forces(sys_part, neighbors; n_threads=1)
        openmm_forces_fp = joinpath(tyk2_dir, "openfe", "forces_openfe_$(inter)_l0.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        # All forces must match at some threshold
        ftol = (inter == "PME" || inter == "all" ? 1e-3 : 1e-6)u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces_molly[MO_map_idx] .- forces_openmm[OP_map_idx])) < ftol

        E_molly = potential_energy(sys_part, neighbors)
        openmm_E_fp = joinpath(tyk2_dir, "openfe", "energy_openfe_$(inter)_l0.txt")
        E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
        # Energy must match at some threshold
        etol = (inter == "PME" || inter == "all" ? 1e-3 : 1e-4)u"kJ * mol^-1"
        @test abs(E_molly - E_openmm) < etol
    end

    # --- Hybrid System Setup λ=1 ---
    sys = RelativeFESystem(sysA, sysB, FT(1.0), mapping, core_mapAB;
                            scheduler=Molly.OpenFEScheduler(dual=false),
                            array_type=AT, 
                            float_type=FT, 
                            LJsoftcore="gapsys",
                            Csoftcore="scaled"
                            )

    #--- Load Positions ---
    openmm_coords_fp = joinpath(tyk2_dir, "openfe", "positions_openmm.txt")
    coords_openmm = Dict()
    atom_order = Dict()
    chains = "ABCDE"
    res_id = 0
    resi_n = 0
    chain = ""
    counter = 0
    ind = 1
    for row in eachrow(readdlm(openmm_coords_fp))
        ele = split(row[1], ",")
        atom_name = ele[1]
        chain_id = string(chains[parse(Int, ele[2])])
        res_name = ele[3]
        resi = parse(Int, ele[4])
        if chain!=chain_id
            chain = chain_id
            res_id = 0
            resi_n = 0
        end
        if resi_n!=resi
            res_id +=1
            resi_n = resi
        end
        coords_openmm[(atom_name, res_id, res_name, chain_id)] = (
            SVector{3, FT}(parse(FT,ele[5]),parse(FT,ele[6]),parse(FT,ele[7]))*u"nm")
        atom_order[(atom_name, res_id, res_name, chain_id)] = ind
        ind += 1
    end

    new_coords = typeof(sys.coords[1])[]
    OP_map_idx = []
    MO_map_idx = []
    for (a,d,c) in zip(sys.atoms, sys.atoms_data, sys.coords)
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

    test_sys = System(sys,
                    coords=new_coords,)
    place_virtual_sites!(test_sys)
    neighbors = find_neighbors(test_sys)

    # --- Energy and Forces for λ=1 ---
    neighbors = find_neighbors(test_sys)
    forces_t = Molly.zero_forces(test_sys)
    buffers = Molly.init_buffers!(test_sys, 1)

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

        forces_molly = forces(sys_part, neighbors; n_threads=1)
        openmm_forces_fp = joinpath(tyk2_dir, "openfe", "forces_openfe_$(inter)_l1.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        # All forces must match at some threshold
        ftol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-5 : 1e-7)u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces_molly[MO_map_idx] .- forces_openmm[OP_map_idx])) < ftol

        E_molly = potential_energy(sys_part, neighbors)
        openmm_E_fp = joinpath(tyk2_dir, "openfe", "energy_openfe_$(inter)_l1.txt")
        E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
        # Energy must match at some threshold
        etol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-3 : 1e-6)u"kJ * mol^-1"
        @test abs(E_molly - E_openmm) < etol
    end

    # --- Hybrid System Setup λ=0.25 ---
    sys = RelativeFESystem(sysA, sysB, FT(0.25), mapping, core_mapAB;
                            scheduler=Molly.OpenFEScheduler(dual=false),
                            array_type=AT, 
                            float_type=FT, 
                            LJsoftcore="gapsys",
                            Csoftcore="scaled"
                            )

    #--- Load Positions ---
    openmm_coords_fp = joinpath(tyk2_dir, "openfe", "positions_openmm.txt")
    coords_openmm = Dict()
    atom_order = Dict()
    chains = "ABCDE"
    res_id = 0
    resi_n = 0
    chain = ""
    counter = 0
    ind = 1
    for row in eachrow(readdlm(openmm_coords_fp))
        ele = split(row[1], ",")
        atom_name = ele[1]
        chain_id = string(chains[parse(Int, ele[2])])
        res_name = ele[3]
        resi = parse(Int, ele[4])
        if chain!=chain_id
            chain = chain_id
            res_id = 0
            resi_n = 0
        end
        if resi_n!=resi
            res_id +=1
            resi_n = resi
        end
        coords_openmm[(atom_name, res_id, res_name, chain_id)] = (
            SVector{3, FT}(parse(FT,ele[5]),parse(FT,ele[6]),parse(FT,ele[7]))*u"nm")
        atom_order[(atom_name, res_id, res_name, chain_id)] = ind
        ind += 1
    end

    new_coords = typeof(sys.coords[1])[]
    OP_map_idx = []
    MO_map_idx = []
    for (a,d,c) in zip(sys.atoms, sys.atoms_data, sys.coords)
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

    test_sys = System(sys,
                    coords=new_coords,)
    place_virtual_sites!(test_sys)
    neighbors = find_neighbors(test_sys)

    # --- Energy and Forces for λ=0.25 ---
    neighbors = find_neighbors(test_sys)
    forces_t = Molly.zero_forces(test_sys)
    buffers = Molly.init_buffers!(test_sys, 1)

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

        forces_molly = forces(sys_part, neighbors; n_threads=1)
        openmm_forces_fp = joinpath(tyk2_dir, "openfe", "forces_openfe_$(inter)_l25.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        # All forces must match at some threshold
        ftol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-5 : 1e-7)u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces_molly[MO_map_idx] .- forces_openmm[OP_map_idx])) < ftol

        E_molly = potential_energy(sys_part, neighbors)
        openmm_E_fp = joinpath(tyk2_dir, "openfe", "energy_openfe_$(inter)_l25.txt")
        E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
        # Energy must match at some threshold
        etol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-3 : 1e-6)u"kJ * mol^-1"
        @test abs(E_molly - E_openmm) < etol
    end

    # --- Hybrid System Setup λ=0.5 ---
    sys = RelativeFESystem(sysA, sysB, FT(0.5), mapping, core_mapAB;
                            scheduler=Molly.OpenFEScheduler(dual=false),
                            array_type=AT, 
                            float_type=FT, 
                            LJsoftcore="gapsys",
                            Csoftcore="scaled"
                            )

    #--- Load Positions ---
    openmm_coords_fp = joinpath(tyk2_dir, "openfe", "positions_openmm.txt")
    coords_openmm = Dict()
    atom_order = Dict()
    chains = "ABCDE"
    res_id = 0
    resi_n = 0
    chain = ""
    counter = 0
    ind = 1
    for row in eachrow(readdlm(openmm_coords_fp))
        ele = split(row[1], ",")
        atom_name = ele[1]
        chain_id = string(chains[parse(Int, ele[2])])
        res_name = ele[3]
        resi = parse(Int, ele[4])
        if chain!=chain_id
            chain = chain_id
            res_id = 0
            resi_n = 0
        end
        if resi_n!=resi
            res_id +=1
            resi_n = resi
        end
        coords_openmm[(atom_name, res_id, res_name, chain_id)] = (
            SVector{3, FT}(parse(FT,ele[5]),parse(FT,ele[6]),parse(FT,ele[7]))*u"nm")
        atom_order[(atom_name, res_id, res_name, chain_id)] = ind
        ind += 1
    end

    new_coords = typeof(sys.coords[1])[]
    OP_map_idx = []
    MO_map_idx = []
    for (a,d,c) in zip(sys.atoms, sys.atoms_data, sys.coords)
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

    test_sys = System(sys,
                    coords=new_coords,)
    place_virtual_sites!(test_sys)
    neighbors = find_neighbors(test_sys)

    # --- Energy and Forces for λ=0.5 ---
    neighbors = find_neighbors(test_sys)
    forces_t = Molly.zero_forces(test_sys)
    buffers = Molly.init_buffers!(test_sys, 1)

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

        forces_molly = forces(sys_part, neighbors; n_threads=1)
        openmm_forces_fp = joinpath(tyk2_dir, "openfe", "forces_openfe_$(inter)_l5.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        # All forces must match at some threshold
        ftol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-5 : 1e-7)u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces_molly[MO_map_idx] .- forces_openmm[OP_map_idx])) < ftol

        E_molly = potential_energy(sys_part, neighbors)
        openmm_E_fp = joinpath(tyk2_dir, "openfe", "energy_openfe_$(inter)_l5.txt")
        E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
        # Energy must match at some threshold
        etol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-3 : 1e-6)u"kJ * mol^-1"
        @test abs(E_molly - E_openmm) < etol
    end

  # --- Hybrid System Setup λ=0.75 ---
    sys = RelativeFESystem(sysA, sysB, FT(0.75), mapping, core_mapAB;
                            scheduler=Molly.OpenFEScheduler(dual=false),
                            array_type=AT, 
                            float_type=FT, 
                            LJsoftcore="gapsys",
                            Csoftcore="scaled"
                            )

    #--- Load Positions ---
    openmm_coords_fp = joinpath(tyk2_dir, "openfe", "positions_openmm.txt")
    coords_openmm = Dict()
    atom_order = Dict()
    chains = "ABCDE"
    res_id = 0
    resi_n = 0
    chain = ""
    counter = 0
    ind = 1
    for row in eachrow(readdlm(openmm_coords_fp))
        ele = split(row[1], ",")
        atom_name = ele[1]
        chain_id = string(chains[parse(Int, ele[2])])
        res_name = ele[3]
        resi = parse(Int, ele[4])
        if chain!=chain_id
            chain = chain_id
            res_id = 0
            resi_n = 0
        end
        if resi_n!=resi
            res_id +=1
            resi_n = resi
        end
        coords_openmm[(atom_name, res_id, res_name, chain_id)] = (
            SVector{3, FT}(parse(FT,ele[5]),parse(FT,ele[6]),parse(FT,ele[7]))*u"nm")
        atom_order[(atom_name, res_id, res_name, chain_id)] = ind
        ind += 1
    end

    new_coords = typeof(sys.coords[1])[]
    OP_map_idx = []
    MO_map_idx = []
    for (a,d,c) in zip(sys.atoms, sys.atoms_data, sys.coords)
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

    test_sys = System(sys,
                    coords=new_coords,)
    place_virtual_sites!(test_sys)
    neighbors = find_neighbors(test_sys)

    # --- Energy and Forces for λ=0.75 ---
    neighbors = find_neighbors(test_sys)
    forces_t = Molly.zero_forces(test_sys)
    buffers = Molly.init_buffers!(test_sys, 1)

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

        forces_molly = forces(sys_part, neighbors; n_threads=1)
        openmm_forces_fp = joinpath(tyk2_dir, "openfe", "forces_openfe_$(inter)_l75.txt")
        forces_openmm = SVector{3}.(eachrow(readdlm(openmm_forces_fp)))u"kJ * mol^-1 * nm^-1"
        # All forces must match at some threshold
        ftol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-5 : 1e-7)u"kJ * mol^-1 * nm^-1"
        @test maximum(norm.(forces_molly[MO_map_idx] .- forces_openmm[OP_map_idx])) < ftol

        E_molly = potential_energy(sys_part, neighbors)
        openmm_E_fp = joinpath(tyk2_dir, "openfe", "energy_openfe_$(inter)_l75.txt")
        E_openmm = readdlm(openmm_E_fp)[1] * u"kJ * mol^-1"
        # Energy must match at some threshold
        etol = (inter == "PME" || inter == "all" || inter == "nonbonded" ? 1e-3 : 1e-6)u"kJ * mol^-1"
        @test abs(E_molly - E_openmm) < etol
    end

end