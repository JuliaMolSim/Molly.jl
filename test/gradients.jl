@testset "Gradients" begin
    inter = LennardJones()
    boundary = CubicBoundary(5.0)
    a1, a2 = Atom(σ=0.3, ϵ=0.5), Atom(σ=0.3, ϵ=0.5)

    function force_direct(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, boundary)
        F = force(inter, vec, a1, a2, NoUnits)
        return F[1]
    end

    function pe(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, boundary)
        potential_energy(inter, vec, a1, a2, NoUnits)
    end

    function force_grad(dist)
        grads = autodiff(
            Reverse,
            pe,
            Active,
            Active(dist),
        )
        return -grads[1][1]
    end

    dists = collect(0.2:0.01:1.2)
    forces_direct = force_direct.(dists)
    forces_grad = force_grad.(dists)
    @test all(forces_direct .≈ forces_grad)
end

@testset "Differentiable simulation" begin
    runs = [ #               gpu    par    fwd    f32    obc2   gbn2   tol_σ tol_r0
        ("CPU"             , Array, false, false, false, false, false, 1e-4, 1e-4),
        ("CPU forward"     , Array, false, true , false, false, false, 0.5 , 0.1 ),
        ("CPU f32"         , Array, false, false, true , false, false, 0.01, 5e-4),
        ("CPU obc2"        , Array, false, false, false, true , false, 1e-4, 1e-4),
        ("CPU gbn2"        , Array, false, false, false, false, true , 1e-4, 1e-4),
        ("CPU gbn2 forward", Array, false, true , false, false, true , 0.5 , 0.1 ),
    ]
    if run_parallel_tests #                  gpu      par    fwd    f32    obc2   gbn2   tol_σ tol_r0
        push!(runs, ("CPU parallel"        , Array, true , false, false, false, false, 1e-4, 1e-4))
        push!(runs, ("CPU parallel forward", Array, true , true , false, false, false, 0.5 , 0.1 ))
        push!(runs, ("CPU parallel f32"    , Array, true , false, true , false, false, 0.01, 5e-4))
    end
    for AT in array_list[2:end] #            gpu    par    fwd    f32    obc2   gbn2   tol_σ tol_r0
        push!(runs, ("$AT"                 , AT   , false, false, false, false, false, 0.25, 20.0))
        push!(runs, ("$AT forward"         , AT   , false, true , false, false, false, 0.25, 20.0))
        push!(runs, ("$AT f32"             , AT   , false, false, true , false, false, 0.5 , 50.0))
        push!(runs, ("$AT obc2"            , AT   , false, false, false, true , false, 0.25, 20.0))
        push!(runs, ("$AT gbn2"            , AT   , false, false, false, false, true , 0.25, 20.0))
    end

    function mean_min_separation(coords, boundary, ::Val{T}) where T
        min_seps = T[]
        for i in eachindex(coords)
            min_sq_sep = T(100.0)
            for j in eachindex(coords)
                if i != j
                    sq_dist = sum(abs2, vector(coords[i], coords[j], boundary))
                    min_sq_sep = min(sq_dist, min_sq_sep)
                end
            end
            push!(min_seps, sqrt(min_sq_sep))
        end
        return mean(min_seps)
    end

    function loss(σ, r0, coords, velocities, boundary, pairwise_inters, general_inters,
                  neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                  bond_is, bond_js, angles, torsions, rng, ::Val{T}, ::Val{AT}) where {T, AT}
        atoms = [Atom(i, 1, atom_mass, (i % 2 == 0 ? T(-0.02) : T(0.02)), σ, T(0.2)) for i in 1:n_atoms]
        bonds_inner = HarmonicBond{T, T}[]
        for i in 1:(n_atoms ÷ 2)
            push!(bonds_inner, HarmonicBond(T(100.0), bond_dists[i] * r0))
        end
        bonds = InteractionList2Atoms(
            bond_is,
            bond_js,
            AT(bonds_inner),
        )

        sys = System(
            atoms=(AT == Array ? atoms : AT(atoms)),
            coords=AT(coords),
            boundary=boundary,
            velocities=AT(velocities),
            pairwise_inters=pairwise_inters,
            specific_inter_lists=(bonds, angles, torsions),
            general_inters=general_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=rng)

        return mean_min_separation(sys.coords, boundary, Val(T))
    end

    for (name, AT, parallel, forward, f32, obc2, gbn2, tol_σ, tol_r0) in runs
        T = f32 ? Float32 : Float64
        σ  = T(0.4)
        r0 = T(1.0)
        n_atoms = 50
        n_steps = 100
        atom_mass = T(10.0)
        boundary = CubicBoundary(T(3.0))
        temp = T(1.0)
        simulator = VelocityVerlet(
            dt=T(0.001),
            coupling=RescaleThermostat(temp),
        )
        rng = Xoshiro(1000) # Same system every time, not required but increases stability
        coords = place_atoms(n_atoms, boundary; min_dist=T(0.6), max_attempts=500, rng=rng)
        velocities = [random_velocity(atom_mass, temp; rng=rng) for i in 1:n_atoms]
        nb_cutoff = T(1.2)
        lj = LennardJones(cutoff=DistanceCutoff(nb_cutoff), use_neighbors=true)
        crf = CoulombReactionField(
            dist_cutoff=nb_cutoff,
            solvent_dielectric=T(Molly.crf_solvent_dielectric),
            use_neighbors=true,
            coulomb_const=T(ustrip(Molly.coulomb_const)),
        )
        pairwise_inters = (lj, crf)
        bond_is = AT(Int32.(collect(1:(n_atoms ÷ 2))))
        bond_js = AT(Int32.(collect((1 + n_atoms ÷ 2):n_atoms)))
        bond_dists = [norm(vector(Array(coords)[i], Array(coords)[i + n_atoms ÷ 2], boundary))
                      for i in 1:(n_atoms ÷ 2)]
        angles_inner = [HarmonicAngle(k=T(10.0), θ0=T(2.0)) for i in 1:15]
        angles = InteractionList3Atoms(
            AT(Int32.(collect( 1:15))),
            AT(Int32.(collect(16:30))),
            AT(Int32.(collect(31:45))),
            AT(angles_inner),
        )
        torsions_inner = [PeriodicTorsion(
                periodicities=[1, 2, 3],
                phases=T[1.0, 0.0, -1.0],
                ks=T[10.0, 5.0, 8.0],
                n_terms=6,
            ) for i in 1:10]
        torsions = InteractionList4Atoms(
            AT(Int32.(collect( 1:10))),
            AT(Int32.(collect(11:20))),
            AT(Int32.(collect(21:30))),
            AT(Int32.(collect(31:40))),
            AT(torsions_inner),
        )
        atoms_setup = [Atom(charge=zero(T), σ=zero(T)) for i in 1:n_atoms]
        if obc2
            imp_obc2 = ImplicitSolventOBC(
                AT(atoms_setup),
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, nothing);
                kappa=T(0.7),
                use_OBC2=true,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                AT(atoms_setup),
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, nothing);
                kappa=T(0.7),
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceNeighborFinder(
            eligible=AT(trues(n_atoms, n_atoms)),
            n_steps=10,
            dist_cutoff=T(1.5),
        )
        n_threads = parallel ? Threads.nthreads() : 1

        const_args = [
            Const(boundary), Const(pairwise_inters),
            Const(general_inters), Const(neighbor_finder), Const(simulator),
            Const(n_steps), Const(n_threads), Const(n_atoms), Const(atom_mass),
            Const(bond_dists), Const(bond_is), Const(bond_js), Const(angles),
            Const(torsions), Const(rng), Const(Val(T)), Const(Val(AT)),
        ]
        if forward
            grad_enzyme = (
                autodiff(
                    set_runtime_activity(Forward), loss, Duplicated,
                    Duplicated(σ, one(T)), Const(r0), Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)), const_args...,
                )[1],
                autodiff(
                    set_runtime_activity(Forward), loss, Duplicated,
                    Const(σ), Duplicated(r0, one(T)), Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)), const_args...,
                )[1],
            )
        else
            grad_enzyme = autodiff(
                set_runtime_activity(Reverse), loss, Active,
                Active(σ), Active(r0), Duplicated(copy(coords), zero(coords)),
                Duplicated(copy(velocities), zero(velocities)), const_args...,
            )[1][1:2]
        end

        grad_fd = (
            central_fdm(6, 1)(
                σ -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, Val(T), Val(AT),
                ),
                σ,
            ),
            central_fdm(6, 1)(
                r0 -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, Val(T), Val(AT),
                ),
                r0,
            ),
        )
        for (prefix, genz, gfd, tol) in zip(("σ", "r0"), grad_enzyme, grad_fd, (tol_σ, tol_r0))
            if abs(gfd) < 1e-13
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Enzyme $genz"
                ztol = contains(name, "f32") ? 1e-8 : 1e-10
                @test isnothing(genz) || abs(genz) < ztol
            elseif isnothing(genz)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Enzyme $genz"
                @test !isnothing(genz)
            else
                frac_diff = abs(genz - gfd) / abs(gfd)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Enzyme $genz, fractional difference $frac_diff"
                @test frac_diff < tol
            end
        end
    end
end

@testset "Differentiable protein" begin
    function create_sys(AT)
        ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "his.xml"])...; units=false)
        return System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff;
            units=false,
            array_type=AT,
            implicit_solvent="gbn2",
            kappa=0.7,
        )
    end

    EnzymeRules.inactive(::typeof(create_sys), args...) = nothing

    function test_energy_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = inject_gradients(sys_ref, params_dic)
    
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=sys_ref.boundary,
            pairwise_inters=pis,
            specific_inter_lists=sis,
            general_inters=gis,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    
        return potential_energy(sys; n_threads=n_threads)
    end
    
    function test_forces_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = inject_gradients(sys_ref, params_dic)

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=sys_ref.boundary,
            pairwise_inters=pis,
            specific_inter_lists=sis,
            general_inters=gis,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        fs = forces(sys; n_threads=n_threads)
        return sum(sum.(abs, fs))
    end

    function test_sim_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = inject_gradients(sys_ref, params_dic)
    
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=sys_ref.boundary,
            pairwise_inters=pis,
            specific_inter_lists=sis,
            general_inters=gis,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulator = Langevin(dt=0.001, temperature=300.0, friction=1.0)
        n_steps = 5
        rng = Xoshiro(1000)
        simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=rng)
        return sum(sum.(abs, sys.coords))
    end

    params_dic = Dict(
        "atom_C8_σ"                => 0.33996695084235345,
        "atom_C8_ϵ"                => 0.4577296,
        "atom_C9_σ"                => 0.33996695084235345,
        "atom_C9_ϵ"                => 0.4577296,
        "atom_CA_σ"                => 0.33996695084235345,
        "atom_CA_ϵ"                => 0.359824,
        "atom_CT_σ"                => 0.33996695084235345,
        "atom_CT_ϵ"                => 0.4577296,
        "atom_C_σ"                 => 0.33996695084235345,
        "atom_C_ϵ"                 => 0.359824,
        "atom_N3_σ"                => 0.32499985237759577,
        "atom_N3_ϵ"                => 0.71128,
        "atom_N_σ"                 => 0.32499985237759577,
        "atom_N_ϵ"                 => 0.71128,
        "atom_O2_σ"                => 0.2959921901149463,
        "atom_O2_ϵ"                => 0.87864,
        "atom_OH_σ"                => 0.30664733878390477,
        "atom_OH_ϵ"                => 0.8803136,
        "atom_O_σ"                 => 0.2959921901149463,
        "atom_O_ϵ"                 => 0.87864,
        "inter_CO_weight_14"       => 0.8333,
        "inter_GB_neck_cut"        => 0.68,
        "inter_GB_neck_scale"      => 0.826836,
        "inter_GB_offset"          => 0.0195141,
        "inter_GB_params_C_α"      => 0.733756,
        "inter_GB_params_C_β"      => 0.506378,
        "inter_GB_params_C_γ"      => 0.205844,
        "inter_GB_params_N_α"      => 0.503364,
        "inter_GB_params_N_β"      => 0.316828,
        "inter_GB_params_N_γ"      => 0.192915,
        "inter_GB_params_O_α"      => 0.867814,
        "inter_GB_params_O_β"      => 0.876635,
        "inter_GB_params_O_γ"      => 0.387882,
        "inter_GB_probe_radius"    => 0.14,
        "inter_GB_radius_C"        => 0.17,
        "inter_GB_radius_N"        => 0.155,
        "inter_GB_radius_O"        => 0.15,
        "inter_GB_radius_O_CAR"    => 0.14,
        "inter_GB_sa_factor"       => 28.3919551,
        "inter_GB_screen_C"        => 1.058554,
        "inter_GB_screen_N"        => 0.733599,
        "inter_GB_screen_O"        => 1.061039,
        "inter_LJ_weight_14"       => 0.5,
        "inter_PT_-/C/CT/-_k_1"    => 0.0,
        "inter_PT_-/C/N/-_k_1"     => -10.46,
        "inter_PT_-/CA/CA/-_k_1"   => -15.167,
        "inter_PT_-/CA/CT/-_k_1"   => 0.0,
        "inter_PT_-/CT/C8/-_k_1"   => 0.64852,
        "inter_PT_-/CT/C9/-_k_1"   => 0.64852,
        "inter_PT_-/CT/CT/-_k_1"   => 0.6508444444444447,
        "inter_PT_-/CT/N/-_k_1"    => 0.0,
        "inter_PT_-/CT/N3/-_k_1"   => 0.6508444444444447,
        "inter_PT_C/N/CT/C_k_1"    => -0.142256,
        "inter_PT_C/N/CT/C_k_2"    => 1.40164,
        "inter_PT_C/N/CT/C_k_3"    => 2.276096,
        "inter_PT_C/N/CT/C_k_4"    => 0.33472,
        "inter_PT_C/N/CT/C_k_5"    => 1.6736,
        "inter_PT_CT/CT/C/N_k_1"   => 0.8368,
        "inter_PT_CT/CT/C/N_k_2"   => 0.8368,
        "inter_PT_CT/CT/C/N_k_3"   => 1.6736,
        "inter_PT_CT/CT/N/C_k_1"   => 8.368,
        "inter_PT_CT/CT/N/C_k_2"   => 8.368,
        "inter_PT_CT/CT/N/C_k_3"   => 1.6736,
        "inter_PT_H/N/C/O_k_1"     => 8.368,
        "inter_PT_H/N/C/O_k_2"     => -10.46,
        "inter_PT_H1/CT/C/O_k_1"   => 3.3472,
        "inter_PT_H1/CT/C/O_k_2"   => -0.33472,
        "inter_PT_HC/CT/C4/CT_k_1" => 0.66944,
        "inter_PT_N/CT/C/N_k_1"    => 2.7196,
        "inter_PT_N/CT/C/N_k_10"   => 0.1046,
        "inter_PT_N/CT/C/N_k_11"   => -0.046024,
        "inter_PT_N/CT/C/N_k_2"    => -0.824248,
        "inter_PT_N/CT/C/N_k_3"    => 6.04588,
        "inter_PT_N/CT/C/N_k_4"    => 2.004136,
        "inter_PT_N/CT/C/N_k_5"    => -0.0799144,
        "inter_PT_N/CT/C/N_k_6"    => -0.016736,
        "inter_PT_N/CT/C/N_k_7"    => -1.06692,
        "inter_PT_N/CT/C/N_k_8"    => 0.3138,
        "inter_PT_N/CT/C/N_k_9"    => 0.238488,
    )

    platform_runs = [("CPU", Array, false)]
    if run_parallel_tests
        push!(platform_runs, ("CPU parallel", Array, true))
    end
    for AT in array_list[2:end]
        push!(platform_runs, ("$AT", AT, false))
    end
    test_runs = [
        ("Energy", test_energy_grad, 1e-8),
        ("Force" , test_forces_grad, 1e-8),
    ]
    if !running_CI
        push!(test_runs, ("Sim", test_sim_grad, 1e-2))
    end
    params_to_test = (
        #"inter_LJ_weight_14",
        "atom_N_ϵ",
        "inter_PT_C/N/CT/C_k_1",
        "inter_GB_screen_O",
        #"inter_GB_neck_scale",
    )

    for (test_name, test_fn, test_tol) in test_runs
        for (platform, AT, parallel) in platform_runs
            sys_ref = create_sys(AT)
            n_threads = parallel ? Threads.nthreads() : 1
            grads_enzyme = Dict(k => 0.0 for k in keys(params_dic))
            autodiff(
                set_runtime_activity(Reverse), test_fn, Active,
                Duplicated(params_dic, grads_enzyme), Const(sys_ref),
                Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
                Duplicated(sys_ref.neighbor_finder, sys_ref.neighbor_finder),
                Const(n_threads),
            )
            #@test count(!iszero, values(grads_enzyme)) == 67
            for param in params_to_test
                genz = grads_enzyme[param]
                gfd = central_fdm(6, 1)(params_dic[param]) do val
                    dic = copy(params_dic)
                    dic[param] = val
                    test_fn(dic, sys_ref, copy(sys_ref.coords), sys_ref.neighbor_finder, n_threads)
                end
                frac_diff = abs(genz - gfd) / abs(gfd)
                @info "$(rpad(test_name, 6)) - $(rpad(platform, 12)) - $(rpad(param, 21)) - " *
                      "FD $gfd, Enzyme $genz, fractional difference $frac_diff"
                @test frac_diff < test_tol
            end
        end
    end
end
