@testset "ReaxFF Tests" begin
    data_dir = normpath(@__DIR__, "..", "data")

    reference_charge_data = readdlm(joinpath(data_dir, "ReaxFF", "data", "crambin_qeq_charges_1e-12.txt"), Float64)
    reference_charges = vec(reference_charge_data)

    for array_type in (run_cuda_tests ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "Crambin energy match reference $device" begin
            float_type = Float64
            units = true
            qeq_float_type = Float64
            qeq_tolerance = 1e-12

            reaxff = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Monti_2013_CHONSMgPNaCuCl.ff"),
                float_type=float_type,
                array_type=array_type
            )

            if units
                boundary = CubicBoundary(float_type(5.3004)u"nm")
            else
                boundary = CubicBoundary(float_type(5.3004))
            end

            sys = System(
                joinpath(data_dir, "ReaxFF", "data", "crambin.pdb"),
                reaxff,
                boundary=boundary,
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=units
            )

            E_system = potential_energy(sys)
            charges = Molly.from_device(charge.(sys.atoms))

            @test length(charges) == length(reference_charges) == length(sys)
            @test all(isapprox.(charges, reference_charges, rtol=0, atol=1e-9))
            @test isapprox(sum(charges), 0.0, rtol=0, atol=1e-10)

            (E_bond, E_atom, E_lone_pair, E_valence_angle, E_pen, E_coa,
            E_hbond, E_torsion, E_conjugation, E_vdW, E_coulomb,
            E_reaxff ) = Molly.compute_energy_components(sys, reaxff)
            E_qeq = ustrip(u"kcal/mol", units ? E_system : E_system * u"kJ/mol") - E_reaxff

            @test isapprox(E_bond, -1209510.63727572584, rtol=1e-12)
            @test isapprox(E_atom, -33989.3515242986, rtol=1e-12)
            @test isapprox(E_lone_pair, 50.8217514257422, rtol=1e-12)
            @test isapprox(E_valence_angle, 18271.4006496192, rtol=1e-12)
            @test isapprox(E_pen, 3.45965246824416, rtol=1e-12)
            @test isapprox(E_coa, -28.0561319269628, rtol=1e-12)
            @test isapprox(E_hbond, -15100.8081104149, rtol=1e-12)
            @test isapprox(E_torsion, 1708.78260186225, rtol=1e-12)
            @test isapprox(E_conjugation, -573.007274217245, rtol=1e-12)
            @test isapprox(E_vdW, 269108.308246941, rtol=1e-12)
            @test isapprox(E_coulomb, -521656.2647313786, rtol=1e-12)

            # rescale self energy from its legacy factor 23.02 to the Coulomb/QEq ratio
            @test isapprox(E_qeq, 349642.65907966491 * (332.06371 / 14.4) / 23.02, rtol=1e-12)
        end
    end

    for array_type in (run_cuda_tests ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "Ethane force match energy finite differences $device smoothness=$smoothness_correction" for smoothness_correction in (false, true)
            float_type = Float64
            units = true
            box_length = array_type === Array ? 2.5 : 3.5
            qeq_float_type = Float64
            qeq_tolerance = 1e-12

            reaxff = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=float_type,
                array_type=array_type,
                smoothness_correction=smoothness_correction,
            )

            if units
                boundary = CubicBoundary(float_type(box_length)u"nm")
            else
                boundary = CubicBoundary(float_type(box_length))
            end

            sys = System(
                joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"),
                reaxff,
                boundary=boundary,
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=units
            )

            coords = deepcopy(Molly.from_device(sys.coords))

            function fdenergy(R)
                reaxff_new = ReaxFF(
                    force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                    float_type=float_type,
                    array_type=array_type,
                    smoothness_correction=smoothness_correction,
                )

                sys_new = System(
                    joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"),
                    reaxff_new,
                    boundary=boundary,
                    array_type=array_type,
                    eem_float_type=qeq_float_type,
                    eem_tolerance=qeq_tolerance,
                    velocities=nothing,
                    units=units
                )

                sys_new.coords = Molly.to_device(R, array_type)

                E = potential_energy(sys_new)
                return E
            end

            fdm = central_fdm(5, 1)
            F2 = grad(fdm, fdenergy, coords)

            finite_diff_forces = -[Vector(ustrip.(coord)) for coord in F2[1]]

            reaxff_new = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=float_type,
                array_type=array_type,
                smoothness_correction=smoothness_correction,
            )

            sys_new = System(
                joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"),
                reaxff_new,
                boundary=boundary,
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=units
            )

            analytical_forces = [Vector(ustrip.(f)) for f in Molly.from_device(forces(sys_new))]

            @test length(finite_diff_forces) == length(analytical_forces) == length(sys_new)
            @test all(all(isapprox.(x, y, rtol=1e-5, atol=1e-8)) for (x, y) in zip(finite_diff_forces, analytical_forces))
        end
    end


    for array_type in (run_cuda_tests ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "Water force match energy finite differences $device smoothness=$smoothness_correction" for smoothness_correction in (false, true)
            float_type = Float64
            units = true
            box_length = array_type === Array ? 2.5 : 3.5
            qeq_float_type = Float64
            qeq_tolerance = 1e-12

            reaxff = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Zhang_2017_HO.ff"),
                float_type=float_type,
                array_type=array_type,
                smoothness_correction=smoothness_correction,
            )

            if units
                boundary = CubicBoundary(float_type(box_length)u"nm")
            else
                boundary = CubicBoundary(float_type(box_length))
            end

            sys = System(
                joinpath(data_dir, "ReaxFF", "data", "water.pdb"),
                reaxff,
                boundary=boundary,
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=units
            )

            coords = deepcopy(Molly.from_device(sys.coords))

            function fdenergy(R)
                reaxff_new = ReaxFF(
                    force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Zhang_2017_HO.ff"),
                    float_type=float_type,
                    array_type=array_type,
                    smoothness_correction=smoothness_correction,
                )

                sys_new = System(
                    joinpath(data_dir, "ReaxFF", "data", "water.pdb"),
                    reaxff_new,
                    boundary=boundary,
                    array_type=array_type,
                    eem_float_type=qeq_float_type,
                    eem_tolerance=qeq_tolerance,
                    velocities=nothing,
                    units=units
                )
                sys_new.coords = Molly.to_device(R, array_type)

                E = potential_energy(sys_new)
                return E
            end

            fdm = central_fdm(5, 1)
            F2 = grad(fdm, fdenergy, coords)

            finite_diff_forces = -[Vector(ustrip.(coord)) for coord in F2[1]]

            reaxff_new = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Zhang_2017_HO.ff"),
                float_type=float_type,
                array_type=array_type,
                smoothness_correction=smoothness_correction,
            )

            sys_new = System(
                joinpath(data_dir, "ReaxFF", "data", "water.pdb"),
                reaxff_new,
                boundary=boundary,
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=units
            )

            analytical_forces = [Vector(ustrip.(f)) for f in Molly.from_device(forces(sys_new))]

            @test length(finite_diff_forces) == length(analytical_forces) == length(sys_new)
            @test all(all(isapprox.(x, y, rtol=1e-5, atol=1e-8)) for (x, y) in zip(finite_diff_forces, analytical_forces))
        end
    end

    for array_type in (run_cuda_tests ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "Neighbor and CSR rebuilds $device" begin
            reaxff = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=Float64, 
                array_type=array_type
            )

            sys = System(
                joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"), 
                reaxff;
                boundary=CubicBoundary(3.5), 
                array_type=array_type,
                eem_tolerance=1e-12, 
                units=false
            )

            coords = copy(Molly.from_device(sys.coords))
            coords[3] += SVector(1.5, 0.0, 0.0)                 # move a hydrogen outside the cutoff
            sys.coords .= Molly.to_device(coords, array_type)
            neighbors = find_neighbors(sys)
            n_pairs = length(neighbors)
            E_before = potential_energy(sys, neighbors)

            coords[3], coords[4] = coords[4], coords[3]         # swap identical hydrogens
            sys.coords .= Molly.to_device(coords, array_type)
            neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, 1)
            E_after = potential_energy(sys, neighbors, 1)

            @test length(neighbors) == n_pairs == 21
            @test isapprox(E_before, E_after, rtol=1e-12)
        end
    end

    @testset "Enzyme $quantity gradient CPU $structure_file smoothness=$smoothness_correction" for smoothness_correction in (false, true),
            quantity in (:energy, :force_loss), structure_file in ("ethane.pdb", "water.pdb")
        reaxff = ReaxFF(
            force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
            float_type=Float64,
            array_type=Array,
            smoothness_correction=smoothness_correction,
        )

        sys = System(
            joinpath(data_dir, "ReaxFF", "data", structure_file),
            reaxff;
            boundary=CubicBoundary(2.5),
            array_type=Array,
            units=false,
            eem_tolerance=1e-12
        )

        neighbors = find_neighbors(sys; n_threads=1)

        function loss_function(reaxff, sys, neighbors)
            eem = sys.general_inters[1]
            fill!(eem.state.q, 0.0)

            forces, energy = Molly.enzyme_forces_and_energy(sys, reaxff, eem; neighbors)
            if quantity == :energy
                return energy
            else
                return sqrt(sum(abs2, forces))
            end
        end

        loss_function(reaxff, sys, neighbors)

        grad_sys = Enzyme.make_zero(sys)
        grad_reaxff = grad_sys.general_inters[2]

        Enzyme.autodiff(
            Reverse, 
            loss_function, 
            Active,
            Duplicated(reaxff, grad_reaxff),
            Duplicated(sys, grad_sys),
            Const(neighbors)
        )

        function fd_general_loss(parameters)
            sys_new = deepcopy(sys)
            reaxff_new = sys_new.general_inters[2]
            dp = reaxff_new.device_params

            reaxff_new.device_params = Molly.DeviceParams(
                parameters, dp.atom_parameters, dp.bond_parameters,
                dp.off_diagonal_parameters, dp.angle_entries,
                dp.angle_parameters, dp.torsion_parameters,
                dp.h_bond_parameters, dp.global_parameters,
                dp.taper_parameters, dp.h_bond_donors, dp.h_bond_acceptors
            )

            return loss_function(reaxff_new, sys_new, neighbors)
        end

        function fd_atom_loss(parameters)
            sys_new = deepcopy(sys)
            reaxff_new = sys_new.general_inters[2]
            reaxff_new.device_params.atom_parameters .= parameters
            return loss_function(reaxff_new, sys_new, neighbors)
        end

        function fd_bond_loss(parameters)
            sys_new = deepcopy(sys)
            reaxff_new = sys_new.general_inters[2]
            bonds = reaxff_new.device_params.bond_parameters

            for i in eachindex(bonds)
                bonds[i] = Molly.BondParameters(parameters[i]..., bonds[i].is_defined)
            end

            return loss_function(reaxff_new, sys_new, neighbors)
        end

        function fd_off_diagonal_loss(parameters)
            sys_new = deepcopy(sys)
            reaxff_new = sys_new.general_inters[2]
            pairs = reaxff_new.device_params.off_diagonal_parameters

            for i in eachindex(pairs)
                pairs[i] = Molly.OffDiagonalParameters(parameters[i]..., pairs[i].is_defined)
            end

            return loss_function(reaxff_new, sys_new, neighbors)
        end

        function fd_angle_loss(parameters)
            sys_new = deepcopy(sys)
            reaxff_new = sys_new.general_inters[2]
            angles = reaxff_new.device_params.angle_parameters

            for i in eachindex(angles)
                angles[i] = Molly.AngleParameters(parameters[i]..., angles[i].is_defined)
            end

            return loss_function(reaxff_new, sys_new, neighbors)
        end

        function fd_torsion_loss(parameters)
            sys_new = deepcopy(sys)
            reaxff_new = sys_new.general_inters[2]
            torsions = reaxff_new.device_params.torsion_parameters

            for i in eachindex(torsions)
                torsions[i] = Molly.TorsionParameters(parameters[i]..., torsions[i].is_defined)
            end

            return loss_function(reaxff_new, sys_new, neighbors)
        end

        function fd_h_bond_loss(parameters)
            sys_new = deepcopy(sys)
            reaxff_new = sys_new.general_inters[2]
            h_bonds = reaxff_new.device_params.h_bond_parameters

            for i in eachindex(h_bonds)
                h_bonds[i] = Molly.HBondParameters(parameters[i]..., h_bonds[i].is_defined)
            end

            return loss_function(reaxff_new, sys_new, neighbors)
        end

        parameters = reaxff.device_params.general_parameters
        fdm = central_fdm(5, 1)
        finite_diff_gradients = grad(fdm, fd_general_loss, parameters)[1]

        enzyme_gradients = grad_reaxff.device_params.general_parameters

        for name in fieldnames(typeof(parameters))
            # these parameters require updating cached values first
            if name in (:upper_taper_radius, :cutoff_bond_order)
                continue
            end

            @test isapprox(
                getfield(enzyme_gradients, name),
                getfield(finite_diff_gradients, name);
                rtol=1e-6,
                atol=1e-8
            )
        end

        atom_params = reaxff.device_params.atom_parameters
        finite_diff_atom_gradients = grad(fdm, fd_atom_loss, atom_params)[1]
        enzyme_atom_gradients = grad_reaxff.device_params.atom_parameters

        for i in eachindex(atom_params)
            for name in fieldnames(typeof(atom_params[i]))
                if name in (:D_ij, :r_vdW, :alpha, :ro_sigma, :ro_pi, :ro_pipi)
                    continue
                end

                @test isapprox(
                    getfield(enzyme_atom_gradients[i], name),
                    getfield(finite_diff_atom_gradients[i], name);
                    rtol=1e-6,
                    atol=1e-8
                )
            end
        end

        bond_params = reaxff.device_params.bond_parameters
        bond_names = fieldnames(eltype(bond_params))[1:end-1]
        bond_values = [[getfield(bond, name) for name in bond_names] for bond in bond_params]

        finite_diff_bond_gradients = grad(fdm, fd_bond_loss, bond_values)[1]
        enzyme_bond_gradients = grad_reaxff.device_params.bond_parameters

        for i in eachindex(bond_params)
            for (j, name) in enumerate(bond_names)
                if name in (:ot_corr, :ovcoord)
                    continue
                end

                @test isapprox(
                    getfield(enzyme_bond_gradients[i], name),
                    finite_diff_bond_gradients[i][j];
                    rtol=1e-6,
                    atol=1e-8
                )
            end
        end

        off_diagonal_params = reaxff.device_params.off_diagonal_parameters
        off_diagonal_names = fieldnames(eltype(off_diagonal_params))[1:end-1]
        off_diagonal_values = [[getfield(pair, name) for name in off_diagonal_names] for pair in off_diagonal_params]

        finite_diff_off_diagonal_gradients = grad(fdm, fd_off_diagonal_loss, off_diagonal_values)[1]
        enzyme_off_diagonal_gradients = grad_reaxff.device_params.off_diagonal_parameters

        for i in eachindex(off_diagonal_params)
            for (j, name) in enumerate(off_diagonal_names)
                if name in (:gamma_w_inv_pow, :gamma_ij_inv_pow)
                    continue
                end

                @test isapprox(
                    getfield(enzyme_off_diagonal_gradients[i], name),
                    finite_diff_off_diagonal_gradients[i][j];
                    rtol=1e-6,
                    atol=1e-8
                )
            end
        end

        angle_params = reaxff.device_params.angle_parameters
        angle_names = fieldnames(eltype(angle_params))[1:end-1]
        angle_values = [[getfield(angle, name) for name in angle_names] for angle in angle_params]

        finite_diff_angle_gradients = grad(fdm, fd_angle_loss, angle_values)[1]
        enzyme_angle_gradients = grad_reaxff.device_params.angle_parameters

        for i in eachindex(angle_params)
            for (j, name) in enumerate(angle_names)
                @test isapprox(
                    getfield(enzyme_angle_gradients[i], name),
                    finite_diff_angle_gradients[i][j];
                    rtol=1e-6,
                    atol=1e-8
                )
            end
        end

        torsion_params = reaxff.device_params.torsion_parameters
        torsion_names = fieldnames(eltype(torsion_params))[1:end-1]
        torsion_values = [[getfield(torsion, name) for name in torsion_names] for torsion in torsion_params]

        finite_diff_torsion_gradients = grad(fdm, fd_torsion_loss, torsion_values)[1]
        enzyme_torsion_gradients = grad_reaxff.device_params.torsion_parameters

        for i in eachindex(torsion_params)
            for (j, name) in enumerate(torsion_names)
                @test isapprox(
                    getfield(enzyme_torsion_gradients[i], name),
                    finite_diff_torsion_gradients[i][j];
                    rtol=1e-6,
                    atol=1e-8
                )
            end
        end

        h_bond_params = reaxff.device_params.h_bond_parameters
        h_bond_names = fieldnames(eltype(h_bond_params))[1:end-1]
        h_bond_values = [[getfield(h_bond, name) for name in h_bond_names] for h_bond in h_bond_params]

        finite_diff_h_bond_gradients = grad(fdm, fd_h_bond_loss, h_bond_values)[1]
        enzyme_h_bond_gradients = grad_reaxff.device_params.h_bond_parameters

        for i in eachindex(h_bond_params)
            for (j, name) in enumerate(h_bond_names)
                @test isapprox(
                    getfield(enzyme_h_bond_gradients[i], name),
                    finite_diff_h_bond_gradients[i][j];
                    rtol=1e-6,
                    atol=1e-8
                )
            end
        end
    end


    for array_type in (run_cuda_tests ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "Units vs Unitless $device" begin
            float_type = Float64
            box_length = array_type === Array ? 2.5 : 3.5
            qeq_float_type = Float64
            qeq_tolerance = 1e-12

            reaxff_units = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=float_type,
                array_type=array_type
            )

            sys_units = System(
                joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"),
                reaxff_units,
                boundary=CubicBoundary(float_type(box_length)u"nm"),
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=true
            )

            reaxff_unitless = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=float_type,
                array_type=array_type
            )

            sys_unitless = System(
                joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"),
                reaxff_unitless,
                boundary=CubicBoundary(float_type(box_length)),
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=false
            )

            E_units = ustrip(u"kJ/mol", potential_energy(sys_units))
            E_unitless = potential_energy(sys_unitless)

            @test isapprox(E_units, E_unitless, rtol=1e-12)

            F_units = [ustrip.(u"kJ/mol/nm", f) for f in Molly.from_device(forces(sys_units))]
            F_unitless = Molly.from_device(forces(sys_unitless))

            @test length(F_units) == length(F_unitless) == length(sys_units) == length(sys_unitless)
            @test all(all(isapprox.(x, y, rtol=1e-12, atol=1e-10)) for (x, y) in zip(F_units, F_unitless))
        end
    end

    for array_type in (run_cuda_tests ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "Double vs Single Precision $device" begin
            box_length = array_type === Array ? 2.5 : 3.5
            qeq_float_type = Float64
            qeq_tolerance = 1e-12

            reaxff_double = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=Float64,
                array_type=array_type
            )

            sys_double = System(
                joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"),
                reaxff_double,
                boundary=CubicBoundary(Float64(box_length)u"nm"),
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=true
            )

            reaxff_single = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=Float32,
                array_type=array_type
            )

            sys_single = System(
                joinpath(data_dir, "ReaxFF", "data", "ethane.pdb"),
                reaxff_single,
                boundary=CubicBoundary(Float32(box_length)u"nm"),
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                velocities=nothing,
                units=true
            )

            E_double = ustrip(u"kJ/mol", potential_energy(sys_double))
            E_single = ustrip(u"kJ/mol", potential_energy(sys_single))

            @test isapprox(E_double, E_single, rtol=1e-5)

            F_double = [ustrip.(u"kJ/mol/nm", f) for f in Molly.from_device(forces(sys_double))]
            F_single = [ustrip.(u"kJ/mol/nm", f) for f in Molly.from_device(forces(sys_single))]

            @test length(F_double) == length(F_single) == length(sys_double) == length(sys_single)
            @test all(all(isapprox.(x, y, rtol=1e-5, atol=2e-2)) for (x, y) in zip(F_double, F_single))
        end
    end
end
