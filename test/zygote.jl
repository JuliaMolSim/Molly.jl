@testset "Gradients" begin
    inter = LennardJones(force_units=NoUnits, energy_units=NoUnits)
    box_size = SVector(5.0, 5.0, 5.0)
    a1, a2 = Atom(σ=0.3, ϵ=0.5), Atom(σ=0.3, ϵ=0.5)

    function force_direct(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, box_size)
        F = force(inter, vec, c1, c2, a1, a2, box_size)
        return F[1]
    end

    function force_grad(dist)
        grad = gradient(dist) do dist
            c1 = SVector(1.0, 1.0, 1.0)
            c2 = SVector(dist + 1.0, 1.0, 1.0)
            vec = vector(c1, c2, box_size)
            potential_energy(inter, vec, c1, c2, a1, a2, box_size)
        end
        return -grad[1]
    end

    dists = collect(0.2:0.01:1.2)
    forces_direct = force_direct.(dists)
    forces_grad = force_grad.(dists)
    @test all(isapprox.(forces_direct, forces_grad))

    abs2_vec(x) = abs2.(x)

    # Function is strange in order to work with gradients on the GPU
    function mean_min_separation(coords, box_size)
        n_atoms = length(coords)
        coords_rep = repeat(reshape(coords, n_atoms, 1), 1, n_atoms)
        vec2arg(c1, c2) = vector(c1, c2, box_size)
        diffs = vec2arg.(coords_rep, permutedims(coords_rep, (2, 1)))
        disps = Array(sum.(abs2_vec.(diffs)))
        disps_diag = disps .+ Diagonal(100 * ones(typeof(box_size[1]), n_atoms))
        return mean(sqrt.(minimum(disps_diag; dims=1)))
    end

    function test_grad(gpu::Bool, forward::Bool, f32::Bool, pis::Bool,
                        sis::Bool, obc2::Bool, gbn2::Bool)
        n_atoms = 50
        n_steps = 100
        atom_mass = f32 ? 10.0f0 : 10.0
        box_size = f32 ? SVector(3.0f0, 3.0f0, 3.0f0) : SVector(3.0, 3.0, 3.0)
        temp = f32 ? 1.0f0 : 1.0
        simulator = VelocityVerlet(
            dt=f32 ? 0.001f0 : 0.001,
            coupling=RescaleThermostat(temp),
        )
        coords = place_atoms(n_atoms, box_size, f32 ? 0.6f0 : 0.6)
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
        bond_dists = [norm(vector(Array(coords)[i], Array(coords)[i + n_atoms ÷ 2], box_size)) for i in 1:(n_atoms ÷ 2)]
        angles_inner = [HarmonicAngle(th0=f32 ? 2.0f0 : 2.0, cth=f32 ? 10.0f0 : 10.0) for i in 1:15]
        angles = InteractionList3Atoms(
            collect(1:15),
            collect(16:30),
            collect(31:45),
            repeat([""], 15),
            gpu ? cu(angles_inner) : angles_inner,
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
            repeat([""], 10),
            gpu ? cu(torsions_inner) : torsions_inner,
        )
        atoms_setup = [Atom(charge=f32 ? 0.0f0 : 0.0, σ=f32 ? 0.0f0 : 0.0) for i in 1:n_atoms]
        if obc2
            imp_obc2 = ImplicitSolventOBC(
                gpu ? cu(atoms_setup) : atoms_setup,
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, [""], nothing);
                use_OBC2=true,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                gpu ? cu(atoms_setup) : atoms_setup,
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, [""], nothing),
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceVecNeighborFinder(
            nb_matrix=gpu ? cu(trues(n_atoms, n_atoms)) : trues(n_atoms, n_atoms),
            n_steps=10,
            dist_cutoff=f32 ? 1.5f0 : 1.5,
        )

        function loss(σ, kb)
            if f32
                atoms = [Atom(i, i % 2 == 0 ? -0.02f0 : 0.02f0, atom_mass, σ, 0.2f0, false) for i in 1:n_atoms]
            else
                atoms = [Atom(i, i % 2 == 0 ? -0.02 : 0.02, atom_mass, σ, 0.2, false) for i in 1:n_atoms]
            end

            bonds_inner = [HarmonicBond(bond_dists[i], kb) for i in 1:(n_atoms ÷ 2)]
            bonds = InteractionList2Atoms(
                bond_is,
                bond_js,
                repeat([""], length(bonds_inner)),
                gpu ? cu(bonds_inner) : bonds_inner,
            )
            cs = deepcopy(forward ? coords_dual : coords)
            vs = deepcopy(forward ? velocities_dual : velocities)

            s = System(
                atoms=gpu ? cu(atoms) : atoms,
                pairwise_inters=pairwise_inters,
                specific_inter_lists=sis ? (bonds, angles, torsions) : (),
                general_inters=general_inters,
                coords=gpu ? cu(cs) : cs,
                velocities=gpu ? cu(vs) : vs,
                box_size=box_size,
                neighbor_finder=neighbor_finder,
                gpu_diff_safe=true,
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            simulate!(s, simulator, n_steps)

            return mean_min_separation(s.coords, box_size)
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
        ("cpu gbn2 forward", [false, true , false, true , true , false, true ], 0.01, 0.01),
    ]
    if run_gpu_tests #                    gpu    fwd    f32    pis    sis    obc2   gbn2
        push!(runs, ("gpu"             , [true , false, false, true , true , false, false], 0.25, 20.0))
        push!(runs, ("gpu forward"     , [true , true , false, true , true , false, false], 0.01, 0.01))
        push!(runs, ("gpu f32"         , [true , false, true , true , true , false, false], 0.5 , 50.0))
        push!(runs, ("gpu nospecific"  , [true , false, false, true , false, false, false], 0.25, 0.0 ))
        push!(runs, ("gpu nopairwise"  , [true , false, false, false, true , false, false], 0.0 , 10.0))
        push!(runs, ("gpu obc2"        , [true , false, false, true , true , true , false], 0.25, 20.0))
        push!(runs, ("gpu gbn2"        , [true , false, false, true , true , false, true ], 0.25, 20.0))
        push!(runs, ("gpu gbn2 forward", [true , true , false, true , true , false, true ], 0.01, 0.01))
    end

    for (name, args, tol_σ, tol_kb) in runs
        forward, f32 = args[2], args[3]
        σ = f32 ? 0.4f0 : 0.4
        kb = f32 ? 100.0f0 : 100.0
        f = test_grad(args...)
        if forward
            # Run once to setup
            grad_zygote = (
                gradient((σ, kb) -> Zygote.forwarddiff(σ  -> f(σ, kb), σ ), σ, kb)[1],
                gradient((σ, kb) -> Zygote.forwarddiff(kb -> f(σ, kb), kb), σ, kb)[2],
            )
            grad_zygote = (
                gradient((σ, kb) -> Zygote.forwarddiff(σ  -> f(σ, kb), σ ), σ, kb)[1],
                gradient((σ, kb) -> Zygote.forwarddiff(kb -> f(σ, kb), kb), σ, kb)[2],
            )
        else
            # Run once to setup
            grad_zygote = gradient(f, σ, kb)
            grad_zygote = gradient(f, σ, kb)
        end
        grad_fd = (
            central_fdm(6, 1)(σ  -> ForwardDiff.value(f(σ, kb)), σ ),
            central_fdm(6, 1)(kb -> ForwardDiff.value(f(σ, kb)), kb),
        )
        for (prefix, gzy, gfd, tol) in zip(("σ", "kb"), grad_zygote, grad_fd, (tol_σ, tol_kb))
            if abs(gfd) < 1e-13
                @test isnothing(gzy) || abs(gzy) < 1e-12
            else
                frac_diff = abs(gzy - gfd) / abs(gfd)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Zygote $gzy, fractional difference $frac_diff"
                @test frac_diff < tol
            end
        end
    end
end
