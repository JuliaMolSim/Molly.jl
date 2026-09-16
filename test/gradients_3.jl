@testset "Differentiable simulation" begin
    runs = [ #               gpu    par    fwd    f32    obc2   gbn2   tol_σ tol_r0
        ("CPU"             , Array, false, false, false, false, false, 1e-8, 1e-5),
        ("CPU forward"     , Array, false, true , false, false, false, 0.5 , 0.1 ),
        ("CPU f32"         , Array, false, false, true , false, false, 0.01, 5e-4),
        ("CPU obc2"        , Array, false, false, false, true , false, 1e-3, 1e-3),
        ("CPU gbn2"        , Array, false, false, false, false, true , 1e-3, 1e-3),
        ("CPU gbn2 forward", Array, false, true , false, false, true , 0.5 , 0.1 ),
    ]
    if Threads.nthreads() > 1 #                   gpu    par   fwd    f32    obc2   gbn2   tol_σ tol_r0
        push!(runs, ("CPU parallel"             , Array, true, false, false, false, false, 1e-8, 1e-5))
        push!(runs, ("CPU parallel forward"     , Array, true, true , false, false, false, 0.5 , 0.1 ))
        push!(runs, ("CPU parallel f32"         , Array, true, false, true , false, false, 0.01, 5e-4))
        push!(runs, ("CPU parallel obc2"        , Array, true, false, false, true , false, 1e-3, 1e-3))
        push!(runs, ("CPU parallel gbn2"        , Array, true, false, false, false, true , 1e-3, 1e-3))
        push!(runs, ("CPU parallel gbn2 forward", Array, true, true , false, false, true , 0.5 , 0.1 ))
    end
    #=for AT in array_list[2:end] # gpu  par    fwd    f32    obc2   gbn2   tol_σ tol_r0
        push!(runs, ("$AT"    ,   AT,  false, false, false, false, false, 0.25, 20.0))
        push!(runs, ("$AT f32",   AT,  false, false, true , false, false, 0.5 , 50.0))
    end=#

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
                  bond_is, bond_js, angles, torsions, rng, grad_safe, ::Val{T},
                  ::Val{AT}) where {T, AT}
        atoms = [Atom(i, 1, atom_mass, (i % 2 == 0 ? T(-0.02) : T(0.02)), σ, T(0.2), T(1.0), Molly.CoreRole)
                 for i in 1:n_atoms]
        bonds_inner = HarmonicBond{T, T}[]
        for i in 1:(n_atoms ÷ 2)
            push!(bonds_inner, HarmonicBond(T(100.0), bond_dists[i] * r0))
        end
        bonds = InteractionList2Atoms(
            bond_is,
            bond_js,
            to_device(bonds_inner, AT),
        )

        sys = System(
            atoms=to_device(atoms, AT),
            coords=to_device(coords, AT),
            boundary=boundary,
            velocities=to_device(velocities, AT),
            pairwise_inters=pairwise_inters,
            specific_inter_lists=(bonds, angles, torsions),
            general_inters=general_inters,
            neighbor_finder=neighbor_finder,
            force_units=NoUnits,
            energy_units=NoUnits,
            grad_safe=grad_safe, # false for FD to test that the two paths are the same
        )

        simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=rng)

        return mean_min_separation(sys.coords, boundary, Val(T))
    end

    for (name, AT, parallel, forward, f32, obc2, gbn2, tol_σ, tol_r0) in runs
        T = (f32 ? Float32 : Float64)
        n_threads = (parallel ? Threads.nthreads() : 1)
        σ  = T(0.4)
        r0 = T(1.0)
        n_atoms = 50
        n_steps = 100
        atom_mass = T(10.0)
        boundary = CubicBoundary(T(3.0))
        temp = T(1.0)
        simulator = VelocityVerlet(
            dt=T(0.001),
            coupling=(ImmediateThermostat(temp),),
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
        bond_is = to_device(Int32.(collect(1:(n_atoms ÷ 2))), AT)
        bond_js = to_device(Int32.(collect((1 + n_atoms ÷ 2):n_atoms)), AT)
        bond_dists = [norm(vector(coords[i], coords[i + n_atoms ÷ 2], boundary))
                      for i in 1:(n_atoms ÷ 2)]
        angles_inner = [HarmonicAngle(k=T(10.0), θ0=T(2.0)) for i in 1:15]
        angles = InteractionList3Atoms(
            to_device(Int32.(collect( 1:15)), AT),
            to_device(Int32.(collect(16:30)), AT),
            to_device(Int32.(collect(31:45)), AT),
            to_device(angles_inner, AT),
        )
        torsions_inner = [PeriodicTorsion(
                periodicities=[1, 2, 3],
                phases=T[1.0, 0.0, -1.0],
                ks=T[10.0, 5.0, 8.0],
                n_terms=6,
            ) for i in 1:10]
        torsions = InteractionList4Atoms(
            to_device(Int32.(collect( 1:10)), AT),
            to_device(Int32.(collect(11:20)), AT),
            to_device(Int32.(collect(21:30)), AT),
            to_device(Int32.(collect(31:40)), AT),
            to_device(torsions_inner, AT),
        )
        atoms_setup = [Atom(charge=zero(T), σ=zero(T)) for i in 1:n_atoms]
        if obc2
            imp_obc2 = ImplicitSolventOBC(
                to_device(atoms_setup, AT),
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, fill(0, length(bond_is)));
                kappa=T(0.7),
                use_OBC2=true,
                n_threads=n_threads,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                to_device(atoms_setup, AT),
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, fill(0, length(bond_is)));
                kappa=T(0.7),
                n_threads=n_threads,
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceNeighborFinder(
            eligible=to_device(trues(n_atoms, n_atoms), AT),
            n_steps=10,
            dist_cutoff=T(1.5),
        )

        const_args = [
            Const(boundary), Const(pairwise_inters),
            Const(general_inters), Const(neighbor_finder), Const(simulator),
            Const(n_steps), Const(n_threads), Const(n_atoms), Const(atom_mass),
            Const(bond_dists), Const(bond_is), Const(bond_js), Const(angles),
            Const(torsions), Const(rng), Const(true), Const(Val(T)), Const(Val(AT)),
        ]
        if forward
            grad_enzyme = (
                autodiff(
                    set_runtime_activity(Forward),
                    loss,
                    Duplicated,
                    Duplicated(σ, one(T)),
                    Const(r0),
                    Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)),
                    const_args...,
                )[1],
                autodiff(
                    set_runtime_activity(Forward),
                    loss,
                    Duplicated,
                    Const(σ),
                    Duplicated(r0, one(T)),
                    Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)),
                    const_args...,
                )[1],
            )
        else
            grad_enzyme = autodiff(
                set_runtime_activity(Reverse),
                loss,
                Active,
                Active(σ),
                Active(r0),
                Duplicated(copy(coords), zero(coords)),
                Duplicated(copy(velocities), zero(velocities)),
                const_args...,
            )[1][1:2]
        end

        grad_fd = (
            central_fdm(6, 1)(
                σ -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, false, Val(T), Val(AT),
                ),
                σ,
            ),
            central_fdm(6, 1)(
                r0 -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, false, Val(T), Val(AT),
                ),
                r0,
            ),
        )
        for (prefix, genz, gfd, tol) in zip(("σ", "r0"), grad_enzyme, grad_fd, (tol_σ, tol_r0))
            if abs(gfd) < 1e-13
                ztol = (contains(name, "f32") ? 1e-8 : 1e-10)
                @test isnothing(genz) || abs(genz) < ztol
            elseif isnothing(genz)
                @test !isnothing(genz)
            else
                frac_diff = abs(genz - gfd) / abs(gfd)
                @test frac_diff < tol
            end
        end
    end
end
