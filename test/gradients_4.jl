@testset "Differentiable protein" begin
    function create_sys(AT, n_threads, nonbonded_method)
        ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml"])...; units=false)
        return System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff;
            units=false,
            array_type=AT,
            float_type=Float64,
            nonbonded_method=nonbonded_method,
            dispersion_correction=false,
            grad_safe=true,
            strictness=:nowarn,
            n_threads=n_threads,
        )
    end

    function test_energy_grad(params_dic, sys_ref, plan, coords, n_threads)
        sys = inject_gradients(sys_ref, params_dic, plan, coords)
        return potential_energy(sys; n_threads=n_threads)
    end

    function test_forces_grad(params_dic, sys_ref, plan, coords, n_threads)
        sys = inject_gradients(sys_ref, params_dic, plan, coords)
        fs = forces(sys; n_threads=n_threads)
        return sum(sum.(abs2, fs))
    end

    function test_sim_grad(params_dic, sys_ref, plan, coords, n_threads)
        sys = inject_gradients(sys_ref, params_dic, plan, coords)
        simulator = Langevin(dt=0.001, temperature=300.0, friction=1.0)
        n_steps = 5
        rng = Xoshiro(1000)
        simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=rng)
        return sum(sum.(abs, sys.coords))
    end

    platform_runs = [("CPU", Array, false)]
    if Threads.nthreads() > 1
        push!(platform_runs, ("CPU parallel", Array, true))
    end
    for AT in array_list[2:end]
        push!(platform_runs, ("$AT", AT, false))
    end
    nonbonded_methods = (("cutoff", DistanceCutoff(1.0)), ("PME", SetupPME()))

    params_dics = map(nonbonded_methods) do (_, nonbonded_method)
        extract_parameters(create_sys(Array, 1, nonbonded_method))
    end
    test_runs = Any[
        ("Energy", test_energy_grad, (1e-7, 1e-7), 1e-14  , central_fdm(6, 1)),
        ("Force" , test_forces_grad, (1e-9, 1e-9), 1e-14  , central_fdm(6, 1)),
        ("Sim"   , test_sim_grad   , (1e-2, 1e-2), nothing, central_fdm(6, 1; max_range=1e-4)),
    ]
    params_to_test = (
        "atom_N_σ",
        "atom_N_ϵ",
        "inter_PT_C/N/CT/C_k_1",
    )

    for (test_name, test_fn, tol_fds, tol_cross, fdm) in test_runs
        for (nb_i, (nb_name, nonbonded_method)) in enumerate(nonbonded_methods)
            if test_name == "Sim" && nb_name == "PME"
                continue
            end
            tol_fd, params_dic = tol_fds[nb_i], params_dics[nb_i]
            grads_ref = nothing # Single-threaded CPU gradients for every parameter
            for (platform, AT, parallel) in platform_runs
                if test_name == "Sim" && !startswith(platform, "CPU")
                    continue
                end
                n_threads = (parallel ? Threads.nthreads() : 1)
                sys_ref = create_sys(AT, n_threads, nonbonded_method)
                plan = ParameterPlan(sys_ref, params_dic)
                grads_enzyme = Dict(k => 0.0 for k in keys(params_dic))
                autodiff(
                    set_runtime_activity(Reverse),
                    test_fn,
                    Active,
                    Duplicated(params_dic, grads_enzyme),
                    Const(sys_ref),
                    Const(plan),
                    Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
                    Const(n_threads),
                )
                for param in params_to_test
                    genz = grads_enzyme[param]
                    gfd = fdm(params_dic[param]) do val
                        dic = copy(params_dic)
                        dic[param] = val
                        test_fn(dic, sys_ref, plan, copy(sys_ref.coords), n_threads)
                    end
                    frac_diff = abs(genz - gfd) / abs(gfd)
                    @test frac_diff < tol_fd
                end
                if isnothing(tol_cross)
                    continue # Random numbers on different backends may be different
                elseif isnothing(grads_ref)
                    grads_ref = grads_enzyme
                else
                    # Every force field parameter should give the same gradient on every
                    # platform, measured relative to the largest gradient so that parameters
                    # with a near-zero gradient do not dominate
                    scale = maximum(abs, values(grads_ref))
                    max_diff = maximum(abs(grads_enzyme[k] - grads_ref[k])
                                       for k in keys(params_dic))
                    @test max_diff / scale < tol_cross
                end
            end
        end
    end
end
