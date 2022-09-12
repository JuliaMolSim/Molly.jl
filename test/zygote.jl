@testset "Gradients" begin
    inter = LennardJones(force_units=NoUnits, energy_units=NoUnits)
    boundary = CubicBoundary(5.0, 5.0, 5.0)
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

    abs2_vec(x) = abs2.(x)

    # Function is strange in order to work with gradients on the GPU
    function mean_min_separation(coords, boundary)
        diffs = displacements(coords, boundary)
        disps = Array(sum.(abs2_vec.(diffs)))
        disps_diag = disps .+ Diagonal(100 * ones(typeof(boundary[1]), length(coords)))
        return mean(sqrt.(minimum(disps_diag; dims=1)))
    end

    function test_grad(gpu::Bool, forward::Bool, f32::Bool, pis::Bool,
                        sis::Bool, obc2::Bool, gbn2::Bool; AT = Array)
        n_atoms = 50
        n_steps = 100
        atom_mass = f32 ? 10.0f0 : 10.0
        boundary = f32 ? CubicBoundary(3.0f0, 3.0f0, 3.0f0) : CubicBoundary(3.0, 3.0, 3.0)
        temp = f32 ? 1.0f0 : 1.0
        simulator = VelocityVerlet(
            dt=f32 ? 0.001f0 : 0.001,
            coupling=RescaleThermostat(temp),
        )
        coords = place_atoms(n_atoms, boundary; min_dist=f32 ? 0.6f0 : 0.6, max_attempts=500)
        velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]
        coords_dual = [ForwardDiff.Dual.(x, f32 ? 0.0f0 : 0.0) for x in coords]
        velocities_dual = [ForwardDiff.Dual.(x, f32 ? 0.0f0 : 0.0) for x in velocities]
        nb_cutoff = f32 ? 1.2f0 : 1.2
        lj = LennardJones(
            cutoff=DistanceCutoff(nb_cutoff),
            nl_only=true,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        crf = CoulombReactionField(
            dist_cutoff=nb_cutoff,
            solvent_dielectric=f32 ? Float32(Molly.crf_solvent_dielectric) : Molly.crf_solvent_dielectric,
            nl_only=true,
            coulomb_const=f32 ? Float32(ustrip(Molly.coulombconst)) : ustrip(Molly.coulombconst),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        pairwise_inters = pis ? (lj, crf) : ()
        bond_is, bond_js = collect(1:(n_atoms ÷ 2)), collect((1 + n_atoms ÷ 2):n_atoms)
        bond_dists = [norm(vector(Array(coords)[i], Array(coords)[i + n_atoms ÷ 2], boundary)) for i in 1:(n_atoms ÷ 2)]
        angles_inner = [HarmonicAngle(k=f32 ? 10.0f0 : 10.0, θ0=f32 ? 2.0f0 : 2.0) for i in 1:15]
        angles = InteractionList3Atoms(
            collect(1:15),
            collect(16:30),
            collect(31:45),
            fill("", 15),
            gpu ? AT(angles_inner) : angles_inner,
        )
        torsions_inner = [PeriodicTorsion(
                periodicities=[1, 2, 3],
                phases=f32 ? [1.0f0, 0.0f0, -1.0f0] : [1.0, 0.0, -1.0],
                ks=f32 ? [10.0f0, 5.0f0, 8.0f0] : [10.0, 5.0, 8.0],
                n_terms=6,
            ) for i in 1:10]
        torsions = InteractionList4Atoms(
            collect(1:10),
            collect(11:20),
            collect(21:30),
            collect(31:40),
            fill("", 10),
            gpu ? AT(torsions_inner) : torsions_inner,
        )
        atoms_setup = [Atom(charge=f32 ? 0.0f0 : 0.0, σ=f32 ? 0.0f0 : 0.0) for i in 1:n_atoms]
        if obc2
            imp_obc2 = ImplicitSolventOBC(
                gpu ? AT(atoms_setup) : atoms_setup,
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, [""], nothing);
                use_OBC2=true,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                gpu ? AT(atoms_setup) : atoms_setup,
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, [""], nothing),
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceVecNeighborFinder(
            nb_matrix=gpu ? AT(trues(n_atoms, n_atoms)) : trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=f32 ? 1.5f0 : 1.5,
        )

        function loss(σ, k)
            if f32
                atoms = [Atom(i, i % 2 == 0 ? -0.02f0 : 0.02f0, atom_mass, σ, 0.2f0, false) for i in 1:n_atoms]
            else
                atoms = [Atom(i, i % 2 == 0 ? -0.02 : 0.02, atom_mass, σ, 0.2, false) for i in 1:n_atoms]
            end

            bonds_inner = [HarmonicBond(k, bond_dists[i]) for i in 1:(n_atoms ÷ 2)]
            bonds = InteractionList2Atoms(
                bond_is,
                bond_js,
                fill("", length(bonds_inner)),
                gpu ? AT(bonds_inner) : bonds_inner,
            )
            cs = deepcopy(forward ? coords_dual : coords)
            vs = deepcopy(forward ? velocities_dual : velocities)

            s = System(
                atoms=gpu ? AT(atoms) : atoms,
                pairwise_inters=pairwise_inters,
                specific_inter_lists=sis ? (bonds, angles, torsions) : (),
                general_inters=general_inters,
                coords=gpu ? AT(cs) : cs,
                velocities=gpu ? AT(vs) : vs,
                boundary=boundary,
                neighbor_finder=neighbor_finder,
                gpu_diff_safe=true,
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            simulate!(s, simulator, n_steps)

            return mean_min_separation(s.coords, boundary)
        end

        return loss
    end

    runs = [ #                gpu    fwd    f32    pis    sis    obc2   gbn2
        ("cpu"             , [false, false, false, true , true , false, false], 0.1 , 0.25),
        ("cpu forward"     , [false, true , false, true , true , false, false], 0.01, 0.01),
        ("cpu f32"         , [false, false, true , true , true , false, false], 0.2 , 10.0),
        ("cpu nospecific"  , [false, false, false, true , false, false, false], 0.1 , 0.0 ),
        ("cpu nopairwise"  , [false, false, false, false, true , false, false], 0.0 , 0.25),
        ("cpu obc2"        , [false, false, false, true , true , true , false], 0.1 , 0.25),
        ("cpu gbn2"        , [false, false, false, true , true , false, true ], 0.1 , 0.25),
        ("cpu gbn2 forward", [false, true , false, true , true , false, true ], 0.02, 0.02),
    ]
    if run_gpu_tests #                         gpu    fwd    f32    pis    sis    obc2   gbn2
        if run_cuda_tests
            push!(runs, ("cuda"             , [true , false, false, true , true , false, false, AT = CuArray], 0.25, 20.0))
            push!(runs, ("cuda forward"     , [true , true , false, true , true , false, false, AT = CuArray], 0.01, 0.01))
            push!(runs, ("cuda f32"         , [true , false, true , true , true , false, false, AT = CuArray], 0.5 , 50.0))
            push!(runs, ("cuda nospecific"  , [true , false, false, true , false, false, false, AT = CuArray], 0.25, 0.0 ))
            push!(runs, ("cuda nopairwise"  , [true , false, false, false, true , false, false, AT = CuArray], 0.0 , 10.0))
            push!(runs, ("cuda obc2"        , [true , false, false, true , true , true , false, AT = CuArray], 0.25, 20.0))
            push!(runs, ("cuda gbn2"        , [true , false, false, true , true , false, true , AT = CuArray], 0.25, 20.0))
            push!(runs, ("cuda gbn2 forward", [true , true , false, true , true , false, true , AT = CuArray], 0.02, 0.02))
        end
        if run_rocm_tests
            push!(runs, ("rocm"             , [true , false, false, true , true , false, false, AT = ROCArray], 0.25, 20.0))
            push!(runs, ("rocm forward"     , [true , true , false, true , true , false, false, AT = ROCArray], 0.01, 0.01))
            push!(runs, ("rocm f32"         , [true , false, true , true , true , false, false, AT = ROCArray], 0.5 , 50.0))
            push!(runs, ("rocm nospecific"  , [true , false, false, true , false, false, false, AT = ROCArray], 0.25, 0.0 ))
            push!(runs, ("rocm nopairwise"  , [true , false, false, false, true , false, false, AT = ROCArray], 0.0 , 10.0))
            push!(runs, ("rocm obc2"        , [true , false, false, true , true , true , false, AT = ROCArray], 0.25, 20.0))
            push!(runs, ("rocm gbn2"        , [true , false, false, true , true , false, true , AT = ROCArray], 0.25, 20.0))
            push!(runs, ("rocm gbn2 forward", [true , true , false, true , true , false, true , AT = ROCArray], 0.02, 0.02))
        end
    end

    for (name, args, tol_σ, tol_k) in runs
        forward, f32 = args[2], args[3]
        σ = f32 ? 0.4f0 : 0.4
        k = f32 ? 100.0f0 : 100.0
        f = test_grad(args...)
        if forward
            # Run once to setup
            grad_zygote = (
                gradient((σ, k) -> Zygote.forwarddiff(σ -> f(σ, k), σ), σ, k)[1],
                gradient((σ, k) -> Zygote.forwarddiff(k -> f(σ, k), k), σ, k)[2],
            )
            grad_zygote = (
                gradient((σ, k) -> Zygote.forwarddiff(σ -> f(σ, k), σ), σ, k)[1],
                gradient((σ, k) -> Zygote.forwarddiff(k -> f(σ, k), k), σ, k)[2],
            )
        else
            # Run once to setup
            grad_zygote = gradient(f, σ, k)
            grad_zygote = gradient(f, σ, k)
        end
        grad_fd = (
            central_fdm(6, 1)(σ -> ForwardDiff.value(f(σ, k)), σ),
            central_fdm(6, 1)(k -> ForwardDiff.value(f(σ, k)), k),
        )
        for (prefix, gzy, gfd, tol) in zip(("σ", "k"), grad_zygote, grad_fd, (tol_σ, tol_k))
            if abs(gfd) < 1e-13
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Zygote $gzy"
                ztol = contains(name, "f32") ? 1e-8 : 1e-10
                @test isnothing(gzy) || abs(gzy) < ztol
            else
                frac_diff = abs(gzy - gfd) / abs(gfd)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Zygote $gzy, fractional difference $frac_diff"
                @test frac_diff < tol
            end
        end
    end
end
