@testset "Gradients" begin
    inter = LennardJones(force_units=NoUnits, energy_units=NoUnits)
    boundary = CubicBoundary(5.0)
    a1, a2 = Atom(σ=0.3, ϵ=0.5), Atom(σ=0.3, ϵ=0.5)

    function force_direct(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, boundary)
        F = force(inter, vec, c1, c2, a1, a2, boundary)
        return F[1]
    end

    function force_grad(dist)
        grad = gradient(dist) do dist
            c1 = SVector(1.0, 1.0, 1.0)
            c2 = SVector(dist + 1.0, 1.0, 1.0)
            vec = vector(c1, c2, boundary)
            potential_energy(inter, vec, c1, c2, a1, a2, boundary)
        end
        return -grad[1]
    end

    dists = collect(0.2:0.01:1.2)
    forces_direct = force_direct.(dists)
    forces_grad = force_grad.(dists)
    @test all(isapprox.(forces_direct, forces_grad))
end

@testset "Differentiable simulation" begin
    abs2_vec(x) = abs2.(x)

    # Function is strange in order to work with gradients on the GPU
    function mean_min_separation(coords, boundary)
        diffs = displacements(coords, boundary)
        disps = Array(sum.(abs2_vec.(diffs)))
        disps_diag = disps .+ Diagonal(100 * ones(typeof(boundary[1]), length(coords)))
        return mean(sqrt.(minimum(disps_diag; dims=1)))
    end

    function test_simulation_grad(gpu::Bool, parallel::Bool, forward::Bool, f32::Bool, pis::Bool,
                                  sis::Bool, obc2::Bool, gbn2::Bool)
        n_atoms = 50
        n_steps = 100
        atom_mass = f32 ? 10.0f0 : 10.0
        boundary = f32 ? CubicBoundary(3.0f0) : CubicBoundary(3.0)
        temp = f32 ? 1.0f0 : 1.0
        simulator = VelocityVerlet(
            dt=f32 ? 0.001f0 : 0.001,
            coupling=RescaleThermostat(temp),
        )
        coords = place_atoms(n_atoms, boundary; min_dist=f32 ? 0.6f0 : 0.6, max_attempts=500)
        velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]
        coords_dual = [ForwardDiff.Dual.(x, f32 ? 0.0f0 : 0.0) for x in coords]
        velocities_dual = [ForwardDiff.Dual.(x, f32 ? 0.0f0 : 0.0) for x in velocities]
        nb_cutoff = f32 ? 1.2f0 : 1.2
        lj = LennardJones(
            cutoff=DistanceCutoff(nb_cutoff),
            use_neighbors=true,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        crf = CoulombReactionField(
            dist_cutoff=nb_cutoff,
            solvent_dielectric=f32 ? Float32(Molly.crf_solvent_dielectric) : Molly.crf_solvent_dielectric,
            use_neighbors=true,
            coulomb_const=f32 ? Float32(ustrip(Molly.coulombconst)) : ustrip(Molly.coulombconst),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        pairwise_inters = pis ? (lj, crf) : ()
        bond_is = gpu ? CuArray(Int32.(collect(1:(n_atoms ÷ 2)))) : Int32.(collect(1:(n_atoms ÷ 2)))
        bond_js = gpu ? CuArray(Int32.(collect((1 + n_atoms ÷ 2):n_atoms))) : Int32.(collect((1 + n_atoms ÷ 2):n_atoms))
        bond_dists = [norm(vector(Array(coords)[i], Array(coords)[i + n_atoms ÷ 2], boundary)) for i in 1:(n_atoms ÷ 2)]
        angles_inner = [HarmonicAngle(k=f32 ? 10.0f0 : 10.0, θ0=f32 ? 2.0f0 : 2.0) for i in 1:15]
        angles = InteractionList3Atoms(
            gpu ? CuArray(Int32.(collect( 1:15))) : Int32.(collect( 1:15)),
            gpu ? CuArray(Int32.(collect(16:30))) : Int32.(collect(16:30)),
            gpu ? CuArray(Int32.(collect(31:45))) : Int32.(collect(31:45)),
            gpu ? CuArray(angles_inner) : angles_inner,
        )
        torsions_inner = [PeriodicTorsion(
                periodicities=[1, 2, 3],
                phases=f32 ? [1.0f0, 0.0f0, -1.0f0] : [1.0, 0.0, -1.0],
                ks=f32 ? [10.0f0, 5.0f0, 8.0f0] : [10.0, 5.0, 8.0],
                n_terms=6,
            ) for i in 1:10]
        torsions = InteractionList4Atoms(
            gpu ? CuArray(Int32.(collect( 1:10))) : Int32.(collect( 1:10)),
            gpu ? CuArray(Int32.(collect(11:20))) : Int32.(collect(11:20)),
            gpu ? CuArray(Int32.(collect(21:30))) : Int32.(collect(21:30)),
            gpu ? CuArray(Int32.(collect(31:40))) : Int32.(collect(31:40)),
            gpu ? CuArray(torsions_inner) : torsions_inner,
        )
        atoms_setup = [Atom(charge=f32 ? 0.0f0 : 0.0, σ=f32 ? 0.0f0 : 0.0) for i in 1:n_atoms]
        if obc2
            imp_obc2 = ImplicitSolventOBC(
                gpu ? CuArray(atoms_setup) : atoms_setup,
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, nothing);
                kappa=(f32 ? 0.7f0 : 0.7),
                use_OBC2=true,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                gpu ? CuArray(atoms_setup) : atoms_setup,
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, nothing);
                kappa=(f32 ? 0.7f0 : 0.7),
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceNeighborFinder(
            eligible=gpu ? CuArray(trues(n_atoms, n_atoms)) : trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=f32 ? 1.5f0 : 1.5,
        )

        function loss(σ, r0)
            if f32
                atoms = [Atom(i, i % 2 == 0 ? -0.02f0 : 0.02f0, atom_mass, σ, 0.2f0, false) for i in 1:n_atoms]
            else
                atoms = [Atom(i, i % 2 == 0 ? -0.02 : 0.02, atom_mass, σ, 0.2, false) for i in 1:n_atoms]
            end

            bonds_inner = [HarmonicBond(f32 ? 100.0f0 : 100.0, bond_dists[i] * r0) for i in 1:(n_atoms ÷ 2)]
            bonds = InteractionList2Atoms(
                bond_is,
                bond_js,
                gpu ? CuArray(bonds_inner) : bonds_inner,
            )
            cs = deepcopy(forward ? coords_dual : coords)
            vs = deepcopy(forward ? velocities_dual : velocities)

            s = System(
                atoms=gpu ? CuArray(atoms) : atoms,
                pairwise_inters=pairwise_inters,
                specific_inter_lists=sis ? (bonds, angles, torsions) : (),
                general_inters=general_inters,
                coords=gpu ? CuArray(cs) : cs,
                velocities=gpu ? CuArray(vs) : vs,
                boundary=boundary,
                neighbor_finder=neighbor_finder,
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            simulate!(s, simulator, n_steps; n_threads=(parallel ? Threads.nthreads() : 1))

            return mean_min_separation(s.coords, boundary)
        end

        return loss
    end

    runs = [ #                gpu    par    fwd    f32    pis    sis    obc2   gbn2    tol_σ tol_r0
        ("CPU"             , [false, false, false, false, true , true , false, false], 0.1 , 1.0 ),
        ("CPU forward"     , [false, false, true , false, true , true , false, false], 0.01, 0.05),
        ("CPU f32"         , [false, false, false, true , true , true , false, false], 0.2 , 10.0),
        ("CPU nospecific"  , [false, false, false, false, true , false, false, false], 0.1 , 0.0 ),
        ("CPU nopairwise"  , [false, false, false, false, false, true , false, false], 0.0 , 1.0 ),
        ("CPU obc2"        , [false, false, false, false, true , true , true , false], 0.1 , 20.0),
        ("CPU gbn2"        , [false, false, false, false, true , true , false, true ], 0.1 , 20.0),
        ("CPU gbn2 forward", [false, false, true , false, true , true , false, true ], 0.02, 0.05),
    ]
    if run_parallel_tests #                   gpu    par    fwd    f32    pis    sis    obc2   gbn2    tol_σ tol_r0
        push!(runs, ("CPU parallel"        , [false, true , false, false, true , true , false, false], 0.1 , 1.0 ))
        push!(runs, ("CPU parallel forward", [false, true , true , false, true , true , false, false], 0.01, 0.05))
        push!(runs, ("CPU parallel f32"    , [false, true , false, true , true , true , false, false], 0.2 , 10.0))
    end
    if run_gpu_tests #                        gpu    par    fwd    f32    pis    sis    obc2   gbn2    tol_σ tol_r0
        push!(runs, ("GPU"                 , [true , false, false, false, true , true , false, false], 0.25, 20.0))
        push!(runs, ("GPU f32"             , [true , false, false, true , true , true , false, false], 0.5 , 50.0))
        push!(runs, ("GPU nospecific"      , [true , false, false, false, true , false, false, false], 0.25, 0.0 ))
        push!(runs, ("GPU nopairwise"      , [true , false, false, false, false, true , false, false], 0.0 , 10.0))
        push!(runs, ("GPU obc2"            , [true , false, false, false, true , true , true , false], 0.25, 20.0))
        push!(runs, ("GPU gbn2"            , [true , false, false, false, true , true , false, true ], 0.25, 20.0))
    end

    for (name, args, tol_σ, tol_r0) in runs
        forward, f32 = args[3], args[4]
        σ  = f32 ? 0.4f0 : 0.4
        r0 = f32 ? 1.0f0 : 1.0
        f = test_simulation_grad(args...)
        if forward
            # Run once to setup
            grad_zygote = (
                gradient((σ, r0) -> Zygote.forwarddiff(σ  -> f(σ, r0), σ ), σ, r0)[1],
                gradient((σ, r0) -> Zygote.forwarddiff(r0 -> f(σ, r0), r0), σ, r0)[2],
            )
            grad_zygote = (
                gradient((σ, r0) -> Zygote.forwarddiff(σ  -> f(σ, r0), σ ), σ, r0)[1],
                gradient((σ, r0) -> Zygote.forwarddiff(r0 -> f(σ, r0), r0), σ, r0)[2],
            )
        else
            # Run once to setup
            grad_zygote = gradient(f, σ, r0)
            grad_zygote = gradient(f, σ, r0)
        end
        grad_fd = (
            central_fdm(6, 1)(σ  -> ForwardDiff.value(f(σ, r0)), σ ),
            central_fdm(6, 1)(r0 -> ForwardDiff.value(f(σ, r0)), r0),
        )
        for (prefix, gzy, gfd, tol) in zip(("σ", "r0"), grad_zygote, grad_fd, (tol_σ, tol_r0))
            if abs(gfd) < 1e-13
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Zygote $gzy"
                ztol = contains(name, "f32") ? 1e-8 : 1e-10
                @test isnothing(gzy) || abs(gzy) < ztol
            elseif isnothing(gzy)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Zygote $gzy"
                @test !isnothing(gzy)
            else
                frac_diff = abs(gzy - gfd) / abs(gfd)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Zygote $gzy, fractional difference $frac_diff"
                @test frac_diff < tol
            end
        end
    end
end

@testset "Differentiable protein" begin
    function create_sys(gpu)
        ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "his.xml"])...; units=false)
        return System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff;
            units=false,
            gpu=gpu,
            implicit_solvent="gbn2",
            kappa=0.7,
        )
    end

    function test_energy_grad(gpu::Bool, parallel::Bool)
        sys_ref = create_sys(gpu)
    
        function loss(params_dic)
            n_threads = parallel ? Threads.nthreads() : 1
            atoms, pairwise_inters, specific_inter_lists, general_inters = inject_gradients(
                                                                                sys_ref, params_dic)

            sys = System(
                atoms=atoms,
                pairwise_inters=pairwise_inters,
                specific_inter_lists=specific_inter_lists,
                general_inters=general_inters,
                coords=sys_ref.coords,
                boundary=sys_ref.boundary,
                neighbor_finder=sys_ref.neighbor_finder,
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            neighbors = find_neighbors(sys; n_threads=n_threads)
            return potential_energy(sys, neighbors; n_threads=n_threads)
        end

        return loss
    end

    sum_abs(x) = sum(abs, x)

    function test_force_grad(gpu::Bool, parallel::Bool)
        sys_ref = create_sys(gpu)

        function loss(params_dic)
            n_threads = parallel ? Threads.nthreads() : 1
            atoms, pairwise_inters, specific_inter_lists, general_inters = inject_gradients(
                                                                                sys_ref, params_dic)

            sys = System(
                atoms=atoms,
                pairwise_inters=pairwise_inters,
                specific_inter_lists=specific_inter_lists,
                general_inters=general_inters,
                coords=sys_ref.coords,
                boundary=sys_ref.boundary,
                neighbor_finder=sys_ref.neighbor_finder,
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            neighbors = find_neighbors(sys; n_threads=n_threads)
            fs = forces(sys, neighbors; n_threads=n_threads)
            return sum(sum_abs.(fs))
        end

        return loss
    end

    function test_sim_grad(gpu::Bool, parallel::Bool)
        sys_ref = create_sys(gpu)

        function loss(params_dic)
            n_threads = parallel ? Threads.nthreads() : 1
            atoms, pairwise_inters, specific_inter_lists, general_inters = inject_gradients(
                                                                                sys_ref, params_dic)

            sys = System(
                atoms=atoms,
                pairwise_inters=pairwise_inters,
                specific_inter_lists=specific_inter_lists,
                general_inters=general_inters,
                coords=sys_ref.coords,
                boundary=sys_ref.boundary,
                neighbor_finder=sys_ref.neighbor_finder,
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            simulator = Langevin(dt=0.001, temperature=300.0, friction=1.0)
            n_steps = 3
            rand_seed = 1000
            simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=Xoshiro(rand_seed))
            return sum(sum_abs.(sys.coords))
        end

        return loss
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

    platform_runs = [("CPU", [false, false])]
    if run_parallel_tests
        push!(platform_runs, ("CPU parallel", [false, true]))
    end
    if run_gpu_tests
        push!(platform_runs, ("GPU", [true, false]))
    end
    test_runs = [
        ("Energy", test_energy_grad, 1e-8),
        ("Force" , test_force_grad , 1e-8),
        ("Sim"   , test_sim_grad   , 0.01),
    ]
    params_to_test = (
        "inter_LJ_weight_14",
        "atom_N_ϵ",
        "inter_PT_C/N/CT/C_k_1",
        "inter_GB_screen_O",
        "inter_GB_neck_scale",
    )

    for (test_name, test_fn, test_tol) in test_runs
        for (platform, args) in platform_runs
            f = test_fn(args...)
            grads_zygote = CUDA.allowscalar() do
                gradient(f, params_dic)[1]
            end
            @test count(!iszero, values(grads_zygote)) == 67
            for param in params_to_test
                gzy = grads_zygote[param]
                gfd = central_fdm(6, 1)(params_dic[param]) do val
                    dic = deepcopy(params_dic)
                    dic[param] = val
                    f(dic)
                end
                frac_diff = abs(gzy - gfd) / abs(gfd)
                @info "$(rpad(test_name, 6)) - $(rpad(platform, 12)) - $(rpad(param, 21)) - FD $gfd, Zygote $gzy, fractional difference $frac_diff"
                @test frac_diff < test_tol
            end
        end
    end
end
