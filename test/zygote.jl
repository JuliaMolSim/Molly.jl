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

    function test_grad(gpu::Bool, parallel::Bool, forward::Bool, f32::Bool, pis::Bool,
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
                use_OBC2=true,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                gpu ? CuArray(atoms_setup) : atoms_setup,
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, nothing),
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceVecNeighborFinder(
            nb_matrix=gpu ? CuArray(trues(n_atoms, n_atoms)) : trues(n_atoms, n_atoms),
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

    runs = [ #                gpu    par    fwd    f32    pis    sis    obc2   gbn2
        ("CPU"             , [false, false, false, false, true , true , false, false], 0.1 , 0.25),
        ("CPU forward"     , [false, false, true , false, true , true , false, false], 0.01, 0.01),
        ("CPU f32"         , [false, false, false, true , true , true , false, false], 0.2 , 10.0),
        ("CPU nospecific"  , [false, false, false, false, true , false, false, false], 0.1 , 0.0 ),
        ("CPU nopairwise"  , [false, false, false, false, false, true , false, false], 0.0 , 0.25),
        ("CPU obc2"        , [false, false, false, false, true , true , true , false], 0.1 , 0.25),
        ("CPU gbn2"        , [false, false, false, false, true , true , false, true ], 0.1 , 0.25),
        ("CPU gbn2 forward", [false, false, true , false, true , true , false, true ], 0.02, 0.02),
    ]
    if run_parallel_tests #                   gpu    par    fwd    f32    pis    sis    obc2   gbn2
        push!(runs, ("CPU parallel"        , [false, true , false, false, true , true , false, false], 0.1 , 0.25))
        push!(runs, ("CPU parallel forward", [false, true , true , false, true , true , false, false], 0.01, 0.01))
        push!(runs, ("CPU parallel f32"    , [false, true , false, true , true , true , false, false], 0.2 , 10.0))
    end
    if run_gpu_tests #                        gpu    par    fwd    f32    pis    sis    obc2   gbn2
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
        f = test_grad(args...)
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
