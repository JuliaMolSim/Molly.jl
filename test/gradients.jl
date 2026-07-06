@testset "Energy gradients" begin
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

@testset "Differentiable PME" begin
    T = Float64
    AT = Array
    ff = MolecularForceField(
        T,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...,
        units=false,
    )
    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        units=false,
        array_type=AT,
        nonbonded_method=:pme,
        grad_safe=true,
    )

    pme = sys.general_inters[1]
    Fs = zero(sys.coords)
    d_sys = zero(sys)
    d_pme = zero(pme)

    pe = Molly.ewald_pe_forces!(Fs, nothing, sys, pme, Val(false))
    Fs_ad = zero(sys.coords)

    pe_ad = autodiff(
        ReverseWithPrimal,
        Molly.ewald_pe_forces!,
        Active,
        Const(Fs_ad),
        Const(nothing),
        Duplicated(sys, d_sys),
        Duplicated(pme, d_pme),
        Const(Val(false)),
    )[2]

    @test pe_ad ≈ pe atol=1e-7
    @test Fs_ad ≈ Fs atol=1e-10
    @test -d_sys.coords ≈ Fs atol=1e-10

    function coord_fdm(c)
        coords_mod = copy(sys.coords)
        coords_mod[1] = SVector(c, coords_mod[1][2], coords_mod[1][3])
        sys_mod = System(deepcopy(sys); coords=coords_mod)
        return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
    end

    c = sys.coords[1][1]
    coord_fdm(c)
    coord_grad = central_fdm(5, 1)(coord_fdm, c)
    @test d_sys.coords[1][1] ≈ coord_grad atol=1e-6

    function charge_fdm(ch)
        atoms_mod = copy(sys.atoms)
        at = sys.atoms[1]
        atoms_mod[1] = Atom(mass=at.mass, charge=ch, σ=at.σ, ϵ=at.σ)
        sys_mod = System(deepcopy(sys); atoms=atoms_mod)
        return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
    end

    at = sys.atoms[1]
    charge_fdm(charge(at))
    charge_grad = central_fdm(5, 1)(charge_fdm, charge(at))
    @test charge(d_sys.atoms[1]) ≈ charge_grad atol=1e-6
end

function injected_charge_energy(params, sys_ref, idx_bundle, coords, neighbor_finder, n_threads)
    atom_idxs, pairwise_idxs, specific_idxs, general_idxs = idx_bundle
    atoms, pis, sis, gis = Molly.inject_gradients(
        sys_ref, params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs
    )
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

@testset "Charge-aware parameter injection" begin
    boundary = CubicBoundary(2.5)
    coords_rf = [
        SVector(0.2, 0.2, 0.2),
        SVector(0.8, 0.5, 0.4),
    ]
    coords_pme = [
        SVector(0.2, 0.2, 0.2),
        SVector(0.9, 0.7, 0.4),
        SVector(1.6, 1.2, 1.1),
    ]
    rc = 1.0

    function charge_updated_atom(atom, new_charge)
        return Atom(atom.index, atom.atom_type, atom.mass, new_charge, atom.σ, atom.ϵ, atom.λ, atom.alch_role)
    end

    function reaction_field_system(AT)
        atoms = [
            Atom(index=1, atom_type=1, mass=10.0, charge=0.4, σ=0.31, ϵ=0.2),
            Atom(index=2, atom_type=2, mass=12.0, charge=-0.3, σ=0.28, ϵ=0.15),
        ]
        atoms_dev = to_device(atoms, AT)
        return System(
            atoms=atoms_dev,
            coords=to_device(coords_rf, AT),
            boundary=boundary,
            pairwise_inters=(CoulombReactionField(
                dist_cutoff=rc,
                solvent_dielectric=78.5,
                coulomb_const=ustrip(Molly.coulomb_const),
            ),),
            general_inters=(),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    function pme_system(AT; scheduler=Molly.DefaultLambdaScheduler(), pairwise=true)
        λ_state = 0.75
        atoms = [
            Atom(index=1, atom_type=1, mass=10.0, charge=1.0, σ=0.3, ϵ=0.2,
                 λ=λ_state, alch_role=Molly.InsertRole),
            Atom(index=2, atom_type=2, mass=12.0, charge=-0.8, σ=0.28, ϵ=0.15,
                 λ=λ_state, alch_role=Molly.InsertRole),
            Atom(index=3, atom_type=3, mass=14.0, charge=0.25, σ=0.26, ϵ=0.12),
        ]
        atoms_dev = to_device(atoms, AT)
        pairwise_inters = pairwise ? (CoulombEwald(
            dist_cutoff=rc,
            coulomb_const=ustrip(Molly.coulomb_const),
        ),) : ()
        return System(
            atoms=atoms_dev,
            coords=to_device(coords_pme, AT),
            boundary=boundary,
            pairwise_inters=pairwise_inters,
            general_inters=(PME(rc, atoms_dev, boundary; scheduler=scheduler),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    @testset "atom parameter helpers" begin
        sys = reaction_field_system(Array)
        atom_params, atom_idxs, atom_names = Molly.extract_atom_parameters(sys)
        idx_mass, idx_charge, idx_σ, idx_ϵ = atom_idxs

        @test length(atom_idxs) == 4
        @test all(>(0), idx_charge)
        @test atom_params[idx_charge[1]] == charge(from_device(sys.atoms)[1])
        @test occursin("_charge_", atom_names[idx_charge[1]])

        atom = from_device(sys.atoms)[1]
        atom_injected = Molly.inject_atom(atom, [11.0, 0.6, 0.4, 0.3], 1, 2, 3, 4)
        @test mass(atom_injected) == 11.0
        @test charge(atom_injected) == 0.6
        @test atom_injected.σ == 0.4
        @test atom_injected.ϵ == 0.3
    end

    @testset "reaction field charge replay and legacy tuples" begin
        for AT in array_list
            sys = reaction_field_system(AT)
            params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs, _ =
                Molly.extract_parameters(sys)
            idx_charge = atom_idxs[2]
            new_charge = charge(from_device(sys.atoms)[1]) + 0.35
            params_mod = copy(params)
            params_mod[idx_charge[1]] = new_charge

            atoms_mod, pis_mod, sis_mod, gis_mod = Molly.inject_gradients(
                sys, params_mod, atom_idxs, pairwise_idxs, specific_idxs, general_idxs
            )
            sys_mod = System(
                sys;
                atoms=atoms_mod,
                pairwise_inters=pis_mod,
                specific_inter_lists=sis_mod,
                general_inters=gis_mod,
            )

            atoms_direct = copy(from_device(sys.atoms))
            atoms_direct[1] = charge_updated_atom(atoms_direct[1], new_charge)
            sys_direct = System(sys; atoms=to_device(atoms_direct, AT))

            atom_pair = from_device(atoms_mod)
            expected_energy = potential_energy(
                only(sys_mod.pairwise_inters),
                vector(coords_rf[1], coords_rf[2], boundary),
                atom_pair[1],
                atom_pair[2],
                NoUnits,
            )

            energy_tol = (AT <: GPUArrays.AbstractGPUArray ? 1e-5 : 1e-10)
            @test potential_energy(sys_mod) ≈ expected_energy atol=energy_tol
            @test potential_energy(sys_mod) ≈ potential_energy(sys_direct) atol=energy_tol

            legacy_atom_idxs = (atom_idxs[1], atom_idxs[3], atom_idxs[4])
            atoms_legacy, _, _, _ = Molly.inject_gradients(
                sys, params_mod, legacy_atom_idxs, pairwise_idxs, specific_idxs, general_idxs
            )
            @test charge(from_device(atoms_legacy)[1]) == charge(from_device(sys.atoms)[1])
        end
    end

    @testset "PME charge replay refreshes cached sums" begin
        for AT in array_list
            sys = pme_system(AT)
            params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs, _ =
                Molly.extract_parameters(sys)
            idx_charge = atom_idxs[2]
            new_charge = charge(from_device(sys.atoms)[1]) + 0.4
            params_mod = copy(params)
            params_mod[idx_charge[1]] = new_charge

            atoms_mod, pis_mod, sis_mod, gis_mod = Molly.inject_gradients(
                sys, params_mod, atom_idxs, pairwise_idxs, specific_idxs, general_idxs
            )
            sys_mod = System(
                sys;
                atoms=atoms_mod,
                pairwise_inters=pis_mod,
                specific_inter_lists=sis_mod,
                general_inters=gis_mod,
            )

            atoms_direct = copy(from_device(sys.atoms))
            atoms_direct[1] = charge_updated_atom(atoms_direct[1], new_charge)
            atoms_direct_dev = to_device(atoms_direct, AT)
            sys_direct = System(
                sys;
                atoms=atoms_direct_dev,
                general_inters=(PME(rc, atoms_direct_dev, boundary),),
            )

            pme_mod = only(sys_mod.general_inters)
            expected_partial_charges = [
                Molly.effective_charge(pme_mod.scheduler, atom, Val(Float64))
                for atom in from_device(sys_mod.atoms)
            ]
            @test pme_mod.pc_sum ≈ sum(expected_partial_charges) atol=1e-10
            @test pme_mod.pc_abs2_sum ≈ sum(abs2, expected_partial_charges) atol=1e-10
            @test pme_mod.pc_sum != only(sys.general_inters).pc_sum

            energy_tol = (AT <: GPUArrays.AbstractGPUArray ? 2e-4 : 1e-10)
            force_tol = (AT <: GPUArrays.AbstractGPUArray ? 5e-4 : 1e-10)
            @test potential_energy(sys_mod) ≈ potential_energy(sys_direct) atol=energy_tol
            forces_diff = from_device(forces(sys_mod)) .- from_device(forces(sys_direct))
            @test maximum(norm.(forces_diff)) < force_tol
        end
    end

    @testset "PME cached sums respect effective charges" begin
        sys = pme_system(Array; scheduler=Molly.EleScaledLambdaScheduler(), pairwise=false)
        params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs, _ =
            Molly.extract_parameters(sys)
        idx_charge = atom_idxs[2]
        params_mod = copy(params)
        params_mod[idx_charge[1]] += 0.2

        atoms_mod, pis_mod, sis_mod, gis_mod = Molly.inject_gradients(
            sys, params_mod, atom_idxs, pairwise_idxs, specific_idxs, general_idxs
        )
        sys_mod = System(
            sys;
            atoms=atoms_mod,
            pairwise_inters=pis_mod,
            specific_inter_lists=sis_mod,
            general_inters=gis_mod,
        )

        pme_mod = only(sys_mod.general_inters)
        expected_partial_charges = [
            Molly.effective_charge(pme_mod.scheduler, atom, Val(Float64))
            for atom in from_device(sys_mod.atoms)
        ]
        @test pme_mod.pc_sum ≈ sum(expected_partial_charges) atol=1e-10
        @test pme_mod.pc_abs2_sum ≈ sum(abs2, expected_partial_charges) atol=1e-10
    end
end

@testset "Differentiable charge parameters" begin
    boundary = CubicBoundary(2.5)
    coords = [
        SVector(0.2, 0.2, 0.2),
        SVector(0.75, 0.45, 0.4),
    ]
    sys_ref = System(
        atoms=[
            Atom(index=1, atom_type=1, mass=10.0, charge=0.35, σ=0.31, ϵ=0.2),
            Atom(index=2, atom_type=2, mass=12.0, charge=-0.28, σ=0.29, ϵ=0.16),
        ],
        coords=coords,
        boundary=boundary,
        pairwise_inters=(
            LennardJones(cutoff=DistanceCutoff(1.0)),
            CoulombReactionField(
                dist_cutoff=1.0,
                solvent_dielectric=78.5,
                coulomb_const=ustrip(Molly.coulomb_const),
            ),
        ),
        general_inters=(),
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs, _ =
        Molly.extract_parameters(sys_ref)
    idx_bundle = (atom_idxs, pairwise_idxs, specific_idxs, general_idxs)
    charge_idx = atom_idxs[2][1]
    sigma_idx = atom_idxs[3][1]
    coords_ref = copy(sys_ref.coords)
    neighbor_finder_ref = sys_ref.neighbor_finder

    base_energy = injected_charge_energy(params, sys_ref, idx_bundle, coords_ref, neighbor_finder_ref, 1)
    @test isfinite(base_energy)

    charge_grad_fdm = central_fdm(5, 1)(params[charge_idx]) do value
        params_mod = copy(params)
        params_mod[charge_idx] = value
        injected_charge_energy(params_mod, sys_ref, idx_bundle, coords_ref, neighbor_finder_ref, 1)
    end
    sigma_grad_fdm = central_fdm(5, 1)(params[sigma_idx]) do value
        params_mod = copy(params)
        params_mod[sigma_idx] = value
        injected_charge_energy(params_mod, sys_ref, idx_bundle, coords_ref, neighbor_finder_ref, 1)
    end

    @test isfinite(charge_grad_fdm)
    @test isfinite(sigma_grad_fdm)
end

if get(ENV, "RUN_DIFFERENTIABLE_SIM_TESTS", "0") == "1"
@testset "Virial Correctness" begin
    FT = Float64
    AT = Array

    function potential_deformation(sys, neighbors, q)
        T = eltype(q)
        z, o = zero(T), one(T)
        F = @SMatrix [
            o + q[1]  q[4]       q[5];
            z         o + q[2]   q[6];
            z         z          o + q[3]
        ]

        sys_out = System(
            sys;
            coords=[F * coord for coord in sys.coords],
            boundary=TriclinicBoundary(F * Molly.boxmatrix(sys.boundary)),
        )
        return potential_energy(sys_out, neighbors; n_threads=1)
    end

    function virial_enzyme(sys, neighbors)
        T = eltype(eltype(sys.coords))
        q = zeros(T, 6)
        dq = zero(q)

        _, pe = autodiff(
            set_runtime_activity(ReverseWithPrimal),
            potential_deformation,
            Active,
            Const(sys),
            Const(neighbors),
            Duplicated(q, dq),
        )

        W = @SMatrix [
            -dq[1]  -dq[4]  -dq[5];
            -dq[4]  -dq[2]  -dq[6];
            -dq[5]  -dq[6]  -dq[3]
        ]

        return W, pe, dq
    end

    function potential_deformation_pme(pme, atoms, coords, boundary, force_units,
                                       energy_units, q)
        T = eltype(q)
        z, o = zero(T), one(T)
        F = @SMatrix [
            o + q[1]  q[4]       q[5];
            z         o + q[2]   q[6];
            z         z          o + q[3]
        ]
        boundary_new = TriclinicBoundary(F * Molly.boxmatrix(boundary))
        return Molly.ewald_pe_forces!(
            nothing,
            nothing,
            pme,
            atoms,
            [F * coord for coord in coords],
            boundary_new,
            force_units,
            energy_units,
            Val(false),
            false;
            n_threads=1,
        )
    end

    function virial_enzyme_pme(sys)
        pme = only(sys.general_inters)
        q = zeros(eltype(eltype(sys.coords)), 6)
        dq = zero(q)
        autodiff(
            set_runtime_activity(ReverseWithPrimal),
            potential_deformation_pme,
            Active,
            Duplicated(pme, zero(pme)),
            Const(sys.atoms),
            Const(sys.coords),
            Const(sys.boundary),
            Const(sys.force_units),
            Const(sys.energy_units),
            Duplicated(q, dq),
        )
        return @SMatrix [
            -dq[1]  -dq[4]  -dq[5];
            -dq[4]  -dq[2]  -dq[6];
            -dq[5]  -dq[6]  -dq[3]
        ]
    end

    function virial_pme(sys)
        W = zeros(FT, 3, 3)
        Molly.ewald_pe_forces!(
            zero(sys.coords),
            W,
            sys,
            only(sys.general_inters),
            Val(true);
            n_threads=1,
        )
        return W
    end

    function test_virial_match(W_reference, W_molly; relative_tol)
        @test maximum(abs, W_reference - W_molly) < 1e-6
        @test norm(W_reference - W_molly) / max(norm(W_molly), eps(FT)) < relative_tol
        @test abs(tr(W_reference) - tr(W_molly)) < 1e-6
    end

    function lj_dispersion_mechanical_adjustment(sys)
        V = volume(sys)
        correction = zero(eltype(eltype(sys.coords)))

        for inter in values(sys.general_inters)
            if inter isa LJDispersionCorrection
                U6  = inter.factor_6  / V
                U12 = inter.factor_12 / V

                # Enzyme differentiates U6 + U12; pressure uses 2U6 + 4U12.
                correction += (2 * U6 + 4 * U12) - (U6 + U12)
            end
        end

        return correction * I
    end

    ff = MolecularForceField(
        FT,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
        units=false,
        strictness=:nowarn,
    )

    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        units=false,
        array_type=AT,
        nonbonded_method=:cutoff,
    )

    sys_trc = System(sys; boundary=TriclinicBoundary(Molly.boxmatrix(sys.boundary)))
    neighbors_virial = Molly.find_neighbors(sys_trc; n_threads=1)

    W_enzyme, _, _ = virial_enzyme(sys_trc, neighbors_virial)
    W_enzyme_pressure = W_enzyme + lj_dispersion_mechanical_adjustment(sys_trc)
    W_molly = Molly.virial(sys_trc, neighbors_virial; n_threads=1)

    test_virial_match(W_enzyme_pressure, W_molly; relative_tol=1e-14)

    boundary_pme = TriclinicBoundary(@SMatrix [
        2.2  0.1  0.0;
        0.0  2.0  0.2;
        0.0  0.0  2.4
    ])
    atoms_pme = [
        Atom(mass=1.0, charge=1.0, σ=0.0, ϵ=0.0),
        Atom(mass=1.0, charge=-0.7, σ=0.0, ϵ=0.0),
        Atom(mass=1.0, charge=-0.3, σ=0.0, ϵ=0.0),
    ]
    coords_pme = [
        SVector(0.4, 0.6, 0.8),
        SVector(1.2, 0.7, 1.5),
        SVector(0.8, 1.4, 0.3),
    ]

    function pme_system(atoms, coords)
        pme = PME(
            0.9,
            atoms,
            boundary_pme;
            grad_safe=true,
            n_threads=1,
        )
        return System(
            atoms=atoms,
            coords=coords,
            boundary=boundary_pme,
            general_inters=(pme,),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    charged_atom = [Atom(mass=1.0, charge=1.0, σ=0.0, ϵ=0.0)]
    systems_pme = (
        ("reciprocal", pme_system(atoms_pme, coords_pme)),
        ("net charge", pme_system(charged_atom, [SVector(0.4, 0.6, 0.8)])),
    )

    for (name, sys_pme) in systems_pme
        @testset "$name PME virial" begin
            test_virial_match(
                virial_enzyme_pme(sys_pme),
                virial_pme(sys_pme);
                relative_tol=1e-12,
            )
        end
    end

    exclusion_list = InteractionList2Atoms(
        Int32[1],
        Int32[2],
        [EwaldExclusion()],
        [""],
        Molly.EwaldExclusionData(0.9),
    )
    sys_exclusion = System(
        atoms=atoms_pme[1:2],
        coords=coords_pme[1:2],
        boundary=boundary_pme,
        specific_inter_lists=(exclusion_list,),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    W_exclusion, _, _ = virial_enzyme(sys_exclusion, nothing)
    test_virial_match(W_exclusion, Molly.virial(sys_exclusion, nothing; n_threads=1);
                      relative_tol=1e-12)
end

@testset "CV gradients" begin
    function cv_gradient_enz(cv_type, coords, atoms=nothing, boundary=nothing, velocities=nothing)
        d_coords = zero(coords)
        unit_arr = Any[u"nm"]

        _, cv_val_ustrip = autodiff(
            set_runtime_activity(ReverseWithPrimal), # set_runtime_activity necessary for units
            Molly.calculate_cv_ustrip!,
            Active,
            Const(unit_arr),
            Const(cv_type),
            Duplicated(coords, d_coords),
            Const(atoms),
            Const(boundary),
            Const(velocities),
        )

        # Correct the units after the ustrip
        u = only(unit_arr)
        d_coords = d_coords .* u ./ unit(d_coords[1][1])^2

        return d_coords, cv_val_ustrip * u
    end

    function forces_test!(fs, sys, bias::BiasPotential; grad_cv=cv_gradient_enz, kwargs...)
        if bias.cv_type.correction == :pbc
            coords = Molly.unwrap_molecules(sys)
        else
            coords = sys.coords
        end

        # Gradient of CV with respect to coordinates
        d_coords, cv_sim = grad_cv(
            bias.cv_type,
            Molly.from_device(coords),
            Molly.from_device(sys.atoms),
            sys.boundary,
            Molly.from_device(sys.velocities),
        )

        # Gradient of bias function with respect to CV
        d_bias = bias_gradient(bias.bias_type, cv_sim)

        fs_svec = d_bias .* d_coords

        fs .-= Molly.to_device(fs_svec, typeof(fs))
        return fs
    end

    n_atoms = 100
    boundary = CubicBoundary(2.0u"nm")
    temp = 298.0u"K"
    atom_mass = 10.0u"g/mol"
    rng = Xoshiro(15)

    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    coords_ref = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng = rng)
    coords     = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng = rng)
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    cv_d_s   = CalcDist([1], [5], CalcSingleDist())
    cv_d_min = CalcDist([1, 2, 3, 4], [5, 6, 7, 8], CalcMinDist())
    cv_d_max = CalcDist([1, 2, 3, 4], [5, 6, 7, 8], CalcMaxDist())
    cv_d_cm  = CalcDist([1, 2, 3, 4], [5, 6, 7, 8], CalcCMDist())
    cv_rg    = CalcRg([1, 2, 3, 4])
    cv_rmsd  = CalcRMSD(coords_ref, [1,2,3,4],[1,2,3,4]) 
    cv_tor   = CalcTorsion([1,2,3,4])

    cvs = (cv_d_s, cv_d_min, cv_d_max, cv_d_cm, cv_rg, cv_rmsd, cv_tor)
    
    b1 = LinearBias(100.0u"kJ*mol^-1*nm^-1", 0.2u"nm")
    b2 = LinearBias(100.0u"kJ*mol^-1", 0.2)

    bias = (b1, b1, b1, b1, b1, b1, b2)

    for (c, b) in zip(cvs, bias)
        bias_pot = BiasPotential(c, b)
        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(),
            general_inters=(bias_pot,)
        )

        fs = forces(sys)
        fs_zero_enz = zero(fs)
        fs_zero_anl = zero(fs)

        forces_test!(fs_zero_enz, sys, bias_pot; grad_cv=cv_gradient_enz)
        forces_test!(fs_zero_anl, sys, bias_pot; grad_cv=cv_gradient)
        @test isapprox(ustrip_vec.(fs_zero_anl), ustrip_vec.(fs_zero_enz); atol=1e-6)
    end
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
    if run_parallel_tests #                  gpu    par    fwd    f32    obc2   gbn2   tol_σ tol_r0
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
        )

        simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=rng)

        return mean_min_separation(sys.coords, boundary, Val(T))
    end

    for (name, AT, parallel, forward, f32, obc2, gbn2, tol_σ, tol_r0) in runs
        T = (f32 ? Float32 : Float64)
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
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                to_device(atoms_setup, AT),
                [AtomData(element="O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, fill(0, length(bond_is)));
                kappa=T(0.7),
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
        n_threads = (parallel ? Threads.nthreads() : 1)

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
                ztol = (contains(name, "f32") ? 1e-8 : 1e-10)
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
else
    @info "Skipping Differentiable simulation testset (known broken). Set RUN_DIFFERENTIABLE_SIM_TESTS=1 to run it."
end

if get(ENV, "RUN_DIFFERENTIABLE_PROTEIN_TESTS", "0") == "1"
@testset "Differentiable protein" begin
    function create_sys(AT)
        ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml"])...; units=false)
        return System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff;
            units=false,
            array_type=AT,
            nonbonded_method=:cutoff,
            dispersion_correction=false,
            implicit_solvent=:gbn2,
            kappa=0.7,
            grad_safe=true,
            strictness=:nowarn,
        )
    end

    function test_energy_grad(params, sys_ref, idx_bundle, coords, neighbor_finder, n_threads)
        atom_idxs, pairwise_idxs, specific_idxs, general_idxs = idx_bundle
        atoms, pis, sis, gis = Molly.inject_gradients(
            sys_ref, params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs
        )

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

    function test_forces_grad(params, sys_ref, idx_bundle, coords, neighbor_finder, n_threads)
        atom_idxs, pairwise_idxs, specific_idxs, general_idxs = idx_bundle
        atoms, pis, sis, gis = Molly.inject_gradients(
            sys_ref, params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs
        )

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

    function test_sim_grad(params, sys_ref, idx_bundle, coords, neighbor_finder, n_threads)
        atom_idxs, pairwise_idxs, specific_idxs, general_idxs = idx_bundle
        atoms, pis, sis, gis = Molly.inject_gradients(
            sys_ref, params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs
        )

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
        "atom_N_ϵ",
        "inter_PT_C/N/CT/C_k_1",
        "inter_GB_screen_O",
    )

    for (test_name, test_fn, test_tol) in test_runs
        for (platform, AT, parallel) in platform_runs
            sys_ref = create_sys(AT)
            params, atom_idxs, pairwise_idxs, specific_idxs, general_idxs, param_names =
                Molly.extract_parameters(sys_ref)
            idx_bundle = (atom_idxs, pairwise_idxs, specific_idxs, general_idxs)
            n_threads = (parallel ? Threads.nthreads() : 1)
            grads_enzyme = zero(params)
            autodiff(
                set_runtime_activity(Reverse),
                test_fn,
                Active,
                Duplicated(params, grads_enzyme),
                Const(sys_ref),
                Const(idx_bundle),
                Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
                Duplicated(sys_ref.neighbor_finder, sys_ref.neighbor_finder),
                Const(n_threads),
            )
            for param in params_to_test
                param_idx = findfirst(==(param), param_names)
                @test !isnothing(param_idx)
                idx = Int(param_idx)
                genz = grads_enzyme[idx]
                gfd = central_fdm(6, 1)(params[idx]) do val
                    params_perturbed = copy(params)
                    params_perturbed[idx] = val
                    test_fn(
                        params_perturbed,
                        sys_ref,
                        idx_bundle,
                        copy(sys_ref.coords),
                        sys_ref.neighbor_finder,
                        n_threads,
                    )
                end
                frac_diff = abs(genz - gfd) / abs(gfd)
                @info "$(rpad(test_name, 6)) - $(rpad(platform, 12)) - $(rpad(param, 21)) - " *
                      "FD $gfd, Enzyme $genz, fractional difference $frac_diff"
                tol = (test_name == "Force" && param == "atom_N_ϵ" ? 2e-3 : test_tol)
                @test frac_diff < tol
            end
        end
    end
end
else
    @info "Skipping Differentiable protein testset (currently crashes with Enzyme on Julia +1.11). Set RUN_DIFFERENTIABLE_PROTEIN_TESTS=1 to run it."
end
