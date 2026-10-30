@testset "ReaxFF Tests" begin
    data_dir = normpath(@__DIR__, "..", "data")

    for array_type in (run_cuda_tests ? (Array, CuArray) : (Array,))
        device = array_type === Array ? "CPU" : "GPU"

        @testset "Crambin reference match $device qeq_tolerance=$qeq_tolerance" for (qeq_tolerance, reference_file, charge_atol) in (
                (1e-12, "crambin_qeq_charges_1e-12.txt", 1e-9),
                (1e-6, "crambin_qeq_charges_1e-6.txt", 5e-5))
                
            reference_charge_data = readdlm(joinpath(data_dir, "ReaxFF", "data", reference_file), Float64)
            reference_charges = vec(reference_charge_data)

            float_type = Float64
            units = true
            qeq_float_type = Float64

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
            @test all(isapprox.(charges, reference_charges, rtol=0, atol=charge_atol))
            @test isapprox(sum(charges), 0.0, rtol=0, atol=1e-10)

            if qeq_tolerance == 1e-12
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

        @testset "Langevin with Monte Carlo barostat temperature bounds $device" begin
            float_type = Float64
            qeq_float_type = Float64
            qeq_tolerance = 1e-6
            boundary = CubicBoundary(float_type(3.5)u"nm")
            temp = 300.0u"K"

            reaxff = ReaxFF(
                force_field_file=joinpath(data_dir, "ReaxFF", "force_fields", "Chenoweth_2008_CHO.ff"),
                float_type=float_type,
                array_type=array_type
            )

            sys = System(
                joinpath(data_dir, "ReaxFF", "data", "methane_oxidation.pdb"),
                reaxff,
                boundary=boundary,
                array_type=array_type,
                eem_float_type=qeq_float_type,
                eem_tolerance=qeq_tolerance,
                loggers=(temperature=TemperatureLogger(40),)
            )

            minimizer = SteepestDescentMinimizer()
            simulate!(sys, minimizer; run_loggers=false, show_progress=false)
            Random.seed!(2024)
            random_velocities!(sys, temp)
            remove_CM_motion!(sys)

            barostat = MonteCarloBarostat(
                1.0u"bar", 
                temp, 
                sys.boundary;
                n_steps=30, 
                trial_find_neighbors=true
            )

            simulator = Langevin(
                dt=0.25u"fs",
                temperature=temp,
                friction=1.0u"ps^-1",
                coupling=barostat,
            )

            # 5ps equilibration
            simulate!(sys, simulator, 20_000; run_loggers=false, show_progress=false)

            # 10ps production
            simulate!(sys, simulator, 40_000; init_step=20_000, run_loggers=:skipstart, show_progress=false)

            temperatures = values(sys.loggers.temperature)
            @test length(temperatures) == 1_000
            @test sys.boundary != boundary
            @test 280.0u"K" < mean(temperatures) < 320.0u"K"
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
