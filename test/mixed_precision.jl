@testset "Mixed-Precision Nonbonded Energy" begin
    local_array_list = (@isdefined(array_list) ? array_list : (Array,))
    local_run_cuda_tests = (@isdefined(run_cuda_tests) ? run_cuda_tests : CUDA.functional())

    function precision_inputs(::Type{T}; n_atoms=96) where T
        coords = [
            SVector{3, T}(
                T(0.25 + 0.375 * ((i - 1) % 7) + 0.03125 * ((i - 1) % 3)),
                T(0.5 + 0.3125 * ((3 * i - 2) % 8) + 0.015625 * ((i - 1) % 5)),
                T(0.75 + 0.28125 * ((5 * i - 1) % 9) + 0.03125 * ((2 * i - 1) % 4)),
            ) for i in 1:n_atoms
        ]
        atoms = [
            Atom(
                index=i,
                mass=T(1.0),
                charge=T((isodd(i) ? -1 : 1) * (0.125 + 0.015625 * (i % 4))),
                σ=T(0.25 + 0.03125 * (i % 5)),
                ϵ=T(0.375 + 0.0625 * (i % 3)),
                λ=T(0.25 + 0.125 * (i % 4)),
            ) for i in 1:n_atoms
        ]
        velocities = fill(zero(SVector{3, T}), n_atoms)
        boundary = CubicBoundary(T(8.0))
        return atoms, coords, velocities, boundary
    end

    function precision_inputs_unitful(::Type{T}; n_atoms=96) where T
        coords = [
            SVector{3, T}(
                T(0.25 + 0.375 * ((i - 1) % 7) + 0.03125 * ((i - 1) % 3)),
                T(0.5 + 0.3125 * ((3 * i - 2) % 8) + 0.015625 * ((i - 1) % 5)),
                T(0.75 + 0.28125 * ((5 * i - 1) % 9) + 0.03125 * ((2 * i - 1) % 4)),
            )u"nm" for i in 1:n_atoms
        ]
        atoms = [
            Atom(
                index=i,
                mass=T(1.0)u"g * mol^-1",
                charge=T((isodd(i) ? -1 : 1) * (0.125 + 0.015625 * (i % 4))),
                σ=T(0.25 + 0.03125 * (i % 5))u"nm",
                ϵ=T(0.375 + 0.0625 * (i % 3))u"kJ * mol^-1",
                λ=T(0.25 + 0.125 * (i % 4)),
                alch_role=(iszero(i % 3) ? Molly.CoreRole : Molly.InsertRole),
            ) for i in 1:n_atoms
        ]
        velocities = fill(SVector{3, T}(zero(T), zero(T), zero(T))u"nm * ps^-1", n_atoms)
        boundary = CubicBoundary(T(8.0)u"nm")
        return atoms, coords, velocities, boundary
    end

    function build_pairwise_system(atoms, coords, velocities, boundary, inter;
                                   array_type=Array,
                                   nonbonded_energy_type=nothing,
                                   dist_cutoff=nothing,
                                   tiled=false,
                                   force_units=NoUnits,
                                   energy_units=NoUnits)
        n_atoms = length(atoms)
        pairwise_inters = (inter isa Tuple ? inter : (inter,))
        atoms_use = (array_type === Array ? atoms : array_type(atoms))
        coords_use = (array_type === Array ? coords : array_type(coords))
        velocities_use = (array_type === Array ? velocities : array_type(velocities))

        neighbor_finder = if tiled
            GPUNeighborFinder(
                n_atoms=n_atoms,
                dist_cutoff=dist_cutoff,
                device_vector_type=CuArray{Int32, 1},
            )
        elseif any(use_neighbors, pairwise_inters)
            DistanceNeighborFinder(
                eligible=trues(n_atoms, n_atoms),
                dist_cutoff=dist_cutoff,
            )
        else
            NoNeighborFinder()
        end

        return System(
            atoms=atoms_use,
            coords=coords_use,
            velocities=velocities_use,
            atoms_data=AtomData[],
            boundary=boundary,
            pairwise_inters=pairwise_inters,
            neighbor_finder=neighbor_finder,
            virtual_sites=Molly.VirtualSite{eltype(eltype(coords)), Int}[],
            force_units=force_units,
            energy_units=energy_units,
            nonbonded_energy_type=nonbonded_energy_type,
        )
    end

    function cpu_precision_triplet(inter32, inter64)
        atoms32, coords32, velocities32, boundary32 = precision_inputs(Float32)
        atoms64 = Molly._float_precision_convert(atoms32, Float64)
        coords64 = Molly._float_precision_convert(coords32, Float64)
        velocities64 = Molly._float_precision_convert(velocities32, Float64)
        boundary64 = Molly._float_precision_convert(boundary32, Float64)

        sys32 = build_pairwise_system(atoms32, coords32, velocities32, boundary32, inter32)
        sys32_mixed = build_pairwise_system(
            atoms32,
            coords32,
            velocities32,
            boundary32,
            inter32;
            nonbonded_energy_type=Float64,
        )
        sys64 = build_pairwise_system(atoms64, coords64, velocities64, boundary64, inter64)
        return sys32, sys32_mixed, sys64
    end

    function precision_inputs_alchemical(::Type{T}; n_atoms=48) where T
        coords = [
            SVector{3, T}(
                T(0.25 + 0.375 * ((i - 1) % 7) + 0.03125 * ((i - 1) % 3)),
                T(0.5 + 0.3125 * ((3 * i - 2) % 8) + 0.015625 * ((i - 1) % 5)),
                T(0.75 + 0.28125 * ((5 * i - 1) % 9) + 0.03125 * ((2 * i - 1) % 4)),
            ) for i in 1:n_atoms
        ]
        atoms = [
            Atom(
                index=i,
                mass=T(1.0 + 0.05 * (i % 3)),
                charge=T((isodd(i) ? -1 : 1) * (0.125 + 0.015625 * (i % 4))),
                σ=T(0.25 + 0.03125 * (i % 5)),
                ϵ=T(0.375 + 0.0625 * (i % 3)),
                λ=T(0.25 + 0.125 * (i % 4)),
                alch_role=(iszero(i % 3) ? Molly.CoreRole : Molly.InsertRole),
            ) for i in 1:n_atoms
        ]
        velocities = fill(zero(SVector{3, T}), n_atoms)
        boundary = CubicBoundary(T(8.0))
        return atoms, coords, velocities, boundary
    end

    struct BuckinghamAtom{I, M, Q, S, E, L, R, A, B, C}
        index::I
        mass::M
        charge::Q
        σ::S
        ϵ::E
        λ::L
        alch_role::R
        A::A
        B::B
        C::C
    end

    function precision_inputs_buckingham(::Type{T}; n_atoms=48) where T
        coords = [
            SVector{3, T}(
                T(0.25 + 0.375 * ((i - 1) % 7) + 0.03125 * ((i - 1) % 3)),
                T(0.5 + 0.3125 * ((3 * i - 2) % 8) + 0.015625 * ((i - 1) % 5)),
                T(0.75 + 0.28125 * ((5 * i - 1) % 9) + 0.03125 * ((2 * i - 1) % 4)),
            ) for i in 1:n_atoms
        ]
        atoms = [
            BuckinghamAtom(
                i,
                T(1.0 + 0.05 * (i % 3)),
                T((isodd(i) ? -1 : 1) * (0.125 + 0.015625 * (i % 4))),
                T(0.25 + 0.03125 * (i % 5)),
                T(0.375 + 0.0625 * (i % 3)),
                T(0.25 + 0.125 * (i % 4)),
                (iszero(i % 3) ? Molly.CoreRole : Molly.InsertRole),
                T(1200.0 + 15.0 * (i % 5)),
                T(2.25 + 0.1 * (i % 4)),
                T(120.0 + 7.0 * (i % 6)),
            ) for i in 1:n_atoms
        ]
        velocities = fill(zero(SVector{3, T}), n_atoms)
        boundary = CubicBoundary(T(8.0))
        return atoms, coords, velocities, boundary
    end

    function pairwise_case_energy(sys; n_threads=1)
        pairwise_inters = values(sys.pairwise_inters)
        neighbors = any(use_neighbors, pairwise_inters) ?
            find_neighbors(sys; n_threads=n_threads) : nothing
        return potential_energy(sys, neighbors; n_threads=n_threads)
    end

    function exhaustive_pairwise_cases()
        return (
            (
                name = "Lennard-Jones",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> LennardJones(cutoff=DistanceCutoff(T(6.0)), use_neighbors=true),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Mie",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> Mie(m=T(8.0), n=T(16.0), cutoff=DistanceCutoff(T(6.0)), use_neighbors=true),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "SoftSphere",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> SoftSphere(cutoff=DistanceCutoff(T(6.0)), use_neighbors=true),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Buckingham",
                inputs = precision_inputs_buckingham,
                n_atoms = 48,
                inter = T -> Buckingham(cutoff=DistanceCutoff(T(6.0)), use_neighbors=true),
                dist_cutoff = T -> T(6.0),
                atol = 2e-3,
                gpu_atol = 2e-2,
            ),
            (
                name = "Gravity",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> Gravity(cutoff=DistanceCutoff(T(6.0)), G=T(1.0), use_neighbors=true),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Ashbaugh-Hatch",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> AshbaughHatch(cutoff=DistanceCutoff(T(6.0)), use_neighbors=true, weight_special=T(0.5)),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Coulomb",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> Coulomb(cutoff=DistanceCutoff(T(6.0)), use_neighbors=true, coulomb_const=T(1.0)),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Coulomb Soft-Core Beutler",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> CoulombSoftCoreBeutler(
                    cutoff=DistanceCutoff(T(6.0)),
                    α=T(0.5),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Coulomb Soft-Core Gapsys",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> CoulombSoftCoreGapsys(
                    cutoff=DistanceCutoff(T(6.0)),
                    α=T(0.3),
                    σQ=T(1.0),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Coulomb Reaction Field",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> CoulombReactionField(
                    dist_cutoff=T(6.0),
                    solvent_dielectric=T(64.0),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Coulomb Soft-Core Beutler Reaction Field",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> CoulombSoftCoreBeutlerReactionField(
                    dist_cutoff=T(6.0),
                    solvent_dielectric=T(64.0),
                    α=T(0.5),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Coulomb Soft-Core Gapsys Reaction Field",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> CoulombSoftCoreGapsysReactionField(
                    dist_cutoff=T(6.0),
                    solvent_dielectric=T(64.0),
                    α=T(0.3),
                    σQ=T(1.0),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Coulomb Ewald",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> CoulombEwald(
                    dist_cutoff=T(6.0),
                    error_tol=T(inv(1024)),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-8,
                gpu_atol = 1e-7,
            ),
            (
                name = "Coulomb Soft-Core Beutler Ewald",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> CoulombSoftCoreBeutlerEwald(
                    dist_cutoff=T(6.0),
                    error_tol=T(inv(1024)),
                    α=T(0.5),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-8,
                gpu_atol = 1e-7,
            ),
            (
                name = "Coulomb Soft-Core Gapsys Ewald",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> CoulombSoftCoreGapsysEwald(
                    dist_cutoff=T(6.0),
                    error_tol=T(inv(1024)),
                    α=T(0.3),
                    σQ=T(1.0),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-8,
                gpu_atol = 1e-7,
            ),
            (
                name = "Yukawa",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> Yukawa(
                    cutoff=DistanceCutoff(T(6.0)),
                    use_neighbors=true,
                    coulomb_const=T(1.0),
                    kappa=T(0.2),
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Lennard-Jones Soft-Core Beutler",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> LennardJonesSoftCoreBeutler(
                    cutoff=DistanceCutoff(T(6.0)),
                    α=T(0.5),
                    use_neighbors=true,
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "Lennard-Jones Soft-Core Gapsys",
                inputs = precision_inputs_alchemical,
                n_atoms = 48,
                inter = T -> LennardJonesSoftCoreGapsys(
                    cutoff=DistanceCutoff(T(6.0)),
                    α=T(0.85),
                    use_neighbors=true,
                ),
                dist_cutoff = T -> T(6.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
            (
                name = "DPD",
                inputs = precision_inputs,
                n_atoms = 48,
                inter = T -> DPDInteraction(
                    a=T(25.0),
                    γ=T(4.5),
                    σ=T(3.0),
                    r_c=T(4.0),
                    dt=T(0.01),
                    use_neighbors=true,
                ),
                dist_cutoff = T -> T(4.0),
                atol = 1e-12,
                gpu_atol = 1e-8,
            ),
        )
    end

    function build_pairwise_case_system(case, ::Type{T}; nonbonded_energy_type=nothing) where T
        atoms, coords, velocities, boundary = case.inputs(T; n_atoms=case.n_atoms)
        return build_pairwise_system(
            atoms,
            coords,
            velocities,
            boundary,
            case.inter(T);
            nonbonded_energy_type=nonbonded_energy_type,
            dist_cutoff=case.dist_cutoff(T),
        )
    end

    function assert_cpu_pairwise_case(case)
        sys32 = build_pairwise_case_system(case, Float32)
        sys32_mixed = build_pairwise_case_system(case, Float32; nonbonded_energy_type=Float64)
        sys64 = build_pairwise_case_system(case, Float64)

        pe32 = pairwise_case_energy(sys32; n_threads=1)
        pe32_mixed = pairwise_case_energy(sys32_mixed; n_threads=1)
        pe64 = pairwise_case_energy(sys64; n_threads=1)

        @test typeof(pe32_mixed) === Float64
        @test pe32_mixed ≈ pe64 atol=case.atol rtol=1e-8
        @test abs(pe32_mixed - pe64) < abs(pe32 - pe64)
    end

    function assert_cuda_pairwise_case(case)
        sys32 = build_pairwise_case_system(case, Float32)
        sys32_mixed = build_pairwise_case_system(case, Float32; nonbonded_energy_type=Float64)
        sys64 = build_pairwise_case_system(case, Float64)

        pe64 = pairwise_case_energy(sys64; n_threads=1)
        pe_cpu_mixed = pairwise_case_energy(sys32_mixed; n_threads=1)

        sys_gpu_default = Molly.to_device(sys32, CuArray)
        sys_gpu_mixed = Molly.to_device(sys32_mixed, CuArray)
        pe_gpu_default = pairwise_case_energy(sys_gpu_default; n_threads=1)
        pe_gpu_mixed = pairwise_case_energy(sys_gpu_mixed; n_threads=1)

        @test typeof(pe_gpu_mixed) === Float64
        @test abs(pe_gpu_mixed - pe64) < abs(pe_gpu_default - pe64)
        @test pe_gpu_mixed ≈ pe_cpu_mixed atol=case.gpu_atol rtol=1e-6
    end

    @testset "Validation" begin
        atoms32, coords32, velocities32, boundary32 = precision_inputs(Float32; n_atoms=8)
        inter32 = LennardJones(cutoff=DistanceCutoff(Float32(6.0)), use_neighbors=false)

        @test_throws ArgumentError build_pairwise_system(
            atoms32,
            coords32,
            velocities32,
            boundary32,
            inter32;
            nonbonded_energy_type=Float64(1),
        )
        @test_throws ArgumentError build_pairwise_system(
            atoms32,
            coords32,
            velocities32,
            boundary32,
            inter32;
            nonbonded_energy_type=Int,
        )
        @test_throws ArgumentError build_pairwise_system(
            atoms32,
            coords32,
            velocities32,
            boundary32,
            inter32;
            nonbonded_energy_type=AbstractFloat,
        )

        sys_mixed = build_pairwise_system(
            atoms32,
            coords32,
            velocities32,
            boundary32,
            inter32;
            nonbonded_energy_type=Float64,
        )
        @test System(sys_mixed; nonbonded_energy_type=nothing).nonbonded_energy_type === nothing
        @test Molly.nonbonded_energy_type(System(sys_mixed; nonbonded_energy_type=nothing)) === Float32
    end

    function assert_cpu_mixed_precision(inter32, inter64; atol)
        sys32, sys32_mixed, sys64 = cpu_precision_triplet(inter32, inter64)

        pe32 = potential_energy(sys32; n_threads=1)
        pe32_mixed = potential_energy(sys32_mixed; n_threads=1)
        pe64 = potential_energy(sys64; n_threads=1)

        @test sys32.nonbonded_energy_type === nothing
        @test Molly.nonbonded_energy_type(sys32) === Float32
        @test sys32_mixed.nonbonded_energy_type === Float64
        @test Molly.nonbonded_energy_type(sys32_mixed) === Float64
        @test pe32_mixed ≈ pe64 atol=atol rtol=0
        @test abs(pe32_mixed - pe64) < abs(pe32 - pe64)
        @test forces(sys32) == forces(sys32_mixed)
    end

    @testset "Constructor and Device Transfer" begin
        inter32 = LennardJones(cutoff=DistanceCutoff(Float32(6.0)), use_neighbors=false)
        atoms32, coords32, velocities32, boundary32 = precision_inputs(Float32; n_atoms=12)
        sys = build_pairwise_system(
            atoms32,
            coords32,
            velocities32,
            boundary32,
            inter32;
            nonbonded_energy_type=Float64,
        )

        @test System(sys).nonbonded_energy_type === Float64

        for AT in local_array_list
            sys_dev = Molly.to_device(sys, AT)
            sys_host = Molly.from_device(sys_dev)

            @test sys_dev.nonbonded_energy_type === Float64
            @test sys_host.nonbonded_energy_type === Float64
        end

        if local_run_cuda_tests
            sys_cuda = Molly.to_device(sys, CuArray)
            buffers = Molly.init_buffers!(sys_cuda, 1, true)
            @test eltype(buffers.pe_vec_nounits) === Float64
        end
    end

    @testset "CPU Pairwise Precision" begin
        assert_cpu_mixed_precision(
            LennardJones(cutoff=DistanceCutoff(Float32(6.0)), use_neighbors=false),
            LennardJones(cutoff=DistanceCutoff(Float64(6.0)), use_neighbors=false);
            atol=1e-12,
        )
        assert_cpu_mixed_precision(
            CoulombReactionField(
                dist_cutoff=Float32(6.0),
                solvent_dielectric=Float32(64.0),
                use_neighbors=false,
                coulomb_const=Float32(1.0),
            ),
            CoulombReactionField(
                dist_cutoff=Float64(6.0),
                solvent_dielectric=Float64(64.0),
                use_neighbors=false,
                coulomb_const=Float64(1.0),
            );
            atol=1e-12,
        )
        assert_cpu_mixed_precision(
            CoulombEwald(
                dist_cutoff=Float32(6.0),
                error_tol=Float32(inv(1024)),
                use_neighbors=false,
                coulomb_const=Float32(1.0),
            ),
            CoulombEwald(
                dist_cutoff=Float64(6.0),
                error_tol=Float64(inv(1024)),
                use_neighbors=false,
                coulomb_const=Float64(1.0),
            );
            atol=1e-8,
        )
        assert_cpu_mixed_precision(
            LennardJonesSoftCoreBeutler(
                cutoff=DistanceCutoff(Float32(6.0)),
                α=Float32(0.5),
                use_neighbors=false,
            ),
            LennardJonesSoftCoreBeutler(
                cutoff=DistanceCutoff(Float64(6.0)),
                α=Float64(0.5),
                use_neighbors=false,
            );
            atol=1e-12,
        )
    end

    @testset "All Pairwise Interactions" begin
        for case in exhaustive_pairwise_cases()
            @testset "$(case.name)" begin
                assert_cpu_pairwise_case(case)

                if local_run_cuda_tests
                    assert_cuda_pairwise_case(case)
                end
            end
        end
    end

    if local_run_cuda_tests
        @testset "CUDA Generic and Tiled Pairwise Precision" begin
            atoms32, coords32, velocities32, boundary32 = precision_inputs(Float32)
            atoms64 = Molly._float_precision_convert(atoms32, Float64)
            coords64 = Molly._float_precision_convert(coords32, Float64)
            velocities64 = Molly._float_precision_convert(velocities32, Float64)
            boundary64 = Molly._float_precision_convert(boundary32, Float64)

            inter32_generic = LennardJones(cutoff=DistanceCutoff(Float32(6.0)), use_neighbors=false)
            inter64_generic = LennardJones(cutoff=DistanceCutoff(Float64(6.0)), use_neighbors=false)

            sys_cpu_mixed_generic = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32_generic;
                nonbonded_energy_type=Float64,
            )
            sys_ref_generic = build_pairwise_system(
                atoms64,
                coords64,
                velocities64,
                boundary64,
                inter64_generic,
            )

            pe_ref_generic = potential_energy(sys_ref_generic; n_threads=1)
            pe_cpu_mixed_generic = potential_energy(sys_cpu_mixed_generic; n_threads=1)

            sys_gpu = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32_generic;
                array_type=CuArray,
            )
            sys_gpu_mixed = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32_generic;
                array_type=CuArray,
                nonbonded_energy_type=Float64,
            )

            pe_gpu = potential_energy(sys_gpu, Molly.NoNeighborList(length(sys_gpu)))
            pe_gpu_mixed = potential_energy(sys_gpu_mixed, Molly.NoNeighborList(length(sys_gpu_mixed)))

            @test abs(pe_gpu_mixed - pe_ref_generic) < abs(pe_gpu - pe_ref_generic)
            @test pe_gpu_mixed ≈ pe_cpu_mixed_generic atol=1e-8 rtol=1e-8
            @test eltype(Molly.init_buffers!(sys_gpu_mixed, 1, true).pe_vec_nounits) === Float64

            inter32_tiled = LennardJones(cutoff=DistanceCutoff(Float32(6.0)), use_neighbors=true)
            inter64_tiled = LennardJones(cutoff=DistanceCutoff(Float64(6.0)), use_neighbors=true)

            sys_cpu_mixed_tiled = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32_tiled;
                nonbonded_energy_type=Float64,
                dist_cutoff=Float32(6.0),
            )
            sys_ref_tiled = build_pairwise_system(
                atoms64,
                coords64,
                velocities64,
                boundary64,
                inter64_tiled;
                dist_cutoff=Float64(6.0),
            )

            pe_ref_tiled = potential_energy(sys_ref_tiled, find_neighbors(sys_ref_tiled); n_threads=1)
            pe_cpu_mixed_tiled = potential_energy(sys_cpu_mixed_tiled, find_neighbors(sys_cpu_mixed_tiled); n_threads=1)

            sys_gpu_tiled = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32_tiled;
                array_type=CuArray,
                nonbonded_energy_type=Float64,
                dist_cutoff=Float32(6.0),
                tiled=true,
            )
            sys_gpu_tiled_default = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32_tiled;
                array_type=CuArray,
                dist_cutoff=Float32(6.0),
                tiled=true,
            )

            pe_gpu_tiled = potential_energy(sys_gpu_tiled, nothing)
            pe_gpu_tiled_default = potential_energy(sys_gpu_tiled_default, nothing)

            @test abs(pe_gpu_tiled - pe_ref_tiled) < abs(pe_gpu_tiled_default - pe_ref_tiled)
            @test pe_gpu_tiled ≈ pe_cpu_mixed_tiled atol=1e-8 rtol=1e-8
            @test eltype(Molly.init_buffers!(sys_gpu_tiled, 1, true).pe_vec_nounits) === Float64
        end

        @testset "CUDA Explicit Neighbor List with Unitful Soft-Core Pairwise Terms" begin
            atoms32, coords32, velocities32, boundary32 = precision_inputs_unitful(Float32; n_atoms=48)
            atoms64 = Molly._float_precision_convert(atoms32, Float64)
            coords64 = Molly._float_precision_convert(coords32, Float64)
            velocities64 = Molly._float_precision_convert(velocities32, Float64)
            boundary64 = Molly._float_precision_convert(boundary32, Float64)

            scheduler = Molly.EleScaledLambdaScheduler()
            inter32 = (
                CoulombSoftCoreGapsysReactionField(
                    dist_cutoff=Float32(6.0)u"nm",
                    solvent_dielectric=Float32(64.0),
                    α=Float32(0.3),
                    σQ=Float32(1.0)u"nm",
                    use_neighbors=true,
                    scheduler=scheduler,
                ),
                LennardJonesSoftCoreGapsys(
                    cutoff=DistanceCutoff(Float32(6.0)u"nm"),
                    α=Float32(0.85),
                    use_neighbors=true,
                    scheduler=scheduler,
                ),
            )
            inter64 = Molly._float_precision_convert(inter32, Float64)

            sys_cpu_mixed = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32;
                nonbonded_energy_type=Float64,
                dist_cutoff=Float32(6.0)u"nm",
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            sys_ref = build_pairwise_system(
                atoms64,
                coords64,
                velocities64,
                boundary64,
                inter64;
                dist_cutoff=Float64(6.0)u"nm",
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            sys_gpu_default = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32;
                array_type=CuArray,
                dist_cutoff=Float32(6.0)u"nm",
                tiled=true,
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            sys_gpu_mixed = build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32;
                array_type=CuArray,
                nonbonded_energy_type=Float64,
                dist_cutoff=Float32(6.0)u"nm",
                tiled=true,
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )

            pe_ref = potential_energy(sys_ref, find_neighbors(sys_ref); n_threads=1)
            pe_cpu_mixed = potential_energy(sys_cpu_mixed, find_neighbors(sys_cpu_mixed); n_threads=1)
            pe_gpu_default = potential_energy(sys_gpu_default, find_neighbors(sys_gpu_default))
            pe_gpu_mixed = potential_energy(sys_gpu_mixed, find_neighbors(sys_gpu_mixed))

            @test abs(pe_gpu_mixed - pe_ref) < abs(pe_gpu_default - pe_ref)
            @test pe_gpu_mixed ≈ pe_cpu_mixed atol=1e-6u"kJ * mol^-1" rtol=1e-6
        end
    end

    @testset "Specific and General Interactions in Mixed Precision" begin
        atoms32, coords32, velocities32, boundary32 = precision_inputs(Float32; n_atoms=4)
        atoms64 = Molly._float_precision_convert(atoms32, Float64)
        coords64 = Molly._float_precision_convert(coords32, Float64)
        velocities64 = Molly._float_precision_convert(velocities32, Float64)
        boundary64 = Molly._float_precision_convert(boundary32, Float64)

        bond32 = HarmonicBond(k=Float32(10000.0), r0=Float32(0.5))
        bond64 = HarmonicBond(k=Float64(10000.0), r0=Float64(0.5))
        bonds32 = InteractionList2Atoms([1, 3], [2, 4], [bond32, bond32])
        bonds64 = InteractionList2Atoms([1, 3], [2, 4], [bond64, bond64])

        gen32 = LJDispersionCorrection(atoms32, Float32(6.0))
        gen64 = LJDispersionCorrection(atoms64, Float64(6.0))

        sys32 = System(
            atoms=atoms32,
            coords=coords32,
            velocities=velocities32,
            boundary=boundary32,
            atoms_data=AtomData[],
            specific_inter_lists=(bonds32,),
            general_inters=(gen32,),
            virtual_sites=Molly.VirtualSite{Float32, Int}[],
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        sys32_mixed = System(sys32, nonbonded_energy_type=Float64)
        sys64 = System(
            atoms=atoms64,
            coords=coords64,
            velocities=velocities64,
            boundary=boundary64,
            specific_inter_lists=(bonds64,),
            general_inters=(gen64,),
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        pe32 = potential_energy(sys32, nothing)
        pe32_mixed = potential_energy(sys32_mixed, nothing)
        pe64 = potential_energy(sys64, nothing)

        @test typeof(pe32_mixed) === Float64
        @test abs(pe32_mixed - pe64) < abs(pe32 - pe64)
        @test pe32_mixed ≈ pe64 atol=1e-10

        if local_run_cuda_tests
            sys_gpu_default = Molly.to_device(sys32, CuArray)
            sys_gpu_mixed = Molly.to_device(sys32_mixed, CuArray)

            pe_gpu_default = potential_energy(sys_gpu_default, nothing)
            pe_gpu_mixed = potential_energy(sys_gpu_mixed, nothing)

            @test typeof(pe_gpu_mixed) === Float64
            @test abs(pe_gpu_mixed - pe64) < abs(pe_gpu_default - pe64)
            @test pe_gpu_mixed ≈ pe32_mixed atol=1e-10
        end
    end

    @testset "PME General Interaction in Mixed Precision" begin
        atoms32, coords32, velocities32, boundary32 = precision_inputs(Float32; n_atoms=48)
        atoms64 = Molly._float_precision_convert(atoms32, Float64)
        coords64 = Molly._float_precision_convert(coords32, Float64)
        velocities64 = Molly._float_precision_convert(velocities32, Float64)
        boundary64 = Molly._float_precision_convert(boundary32, Float64)

        rc32 = Float32(3.0)
        rc64 = Float64(3.0)
        error_tol32 = Float32(inv(1024))
        error_tol64 = Float64(inv(1024))

        inter32 = CoulombEwald(
            dist_cutoff=rc32,
            error_tol=error_tol32,
            use_neighbors=true,
            coulomb_const=Float32(1.0),
        )
        inter64 = CoulombEwald(
            dist_cutoff=rc64,
            error_tol=error_tol64,
            use_neighbors=true,
            coulomb_const=Float64(1.0),
        )
        pme32 = PME(rc32, atoms32, boundary32; error_tol=error_tol32)
        pme64 = PME(rc64, atoms64, boundary64; error_tol=error_tol64)

        sys32 = System(
            build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32;
                dist_cutoff=rc32,
            );
            general_inters=(pme32,),
        )
        sys32_mixed = System(
            build_pairwise_system(
                atoms32,
                coords32,
                velocities32,
                boundary32,
                inter32;
                nonbonded_energy_type=Float64,
                dist_cutoff=rc32,
            );
            general_inters=(pme32,),
        )
        sys64 = System(
            build_pairwise_system(
                atoms64,
                coords64,
                velocities64,
                boundary64,
                inter64;
                dist_cutoff=rc64,
            );
            general_inters=(pme64,),
        )

        neighbors32 = find_neighbors(sys32; n_threads=1)
        neighbors32_mixed = find_neighbors(sys32_mixed; n_threads=1)
        neighbors64 = find_neighbors(sys64; n_threads=1)

        pe32 = potential_energy(sys32, neighbors32; n_threads=1)
        pe32_mixed = potential_energy(sys32_mixed, neighbors32_mixed; n_threads=1)
        pe64 = potential_energy(sys64, neighbors64; n_threads=1)

        @test typeof(pe32_mixed) === Float64
        @test pe32_mixed ≈ pe64 atol=5e-7 rtol=0
        @test abs(pe32_mixed - pe64) < abs(pe32 - pe64)
    end

    @testset "Realistic Constructor Mixed Precision" begin
        ff32 = MolecularForceField(Float32, joinpath(ff_dir, "ff99SBildn.xml"); units=false)
        ff64 = MolecularForceField(Float64, joinpath(ff_dir, "ff99SBildn.xml"); units=false)

        for solvent_model in (:obc2, :gbn2)
            @testset "$(solvent_model)" begin
                sys32 = System(
                    joinpath(data_dir, "6mrr_nowater.pdb"),
                    ff32;
                    units=false,
                    array_type=Array,
                    boundary=CubicBoundary(Float32(100.0)),
                    dist_cutoff=Float32(5.0),
                    nonbonded_method=:cutoff,
                    implicit_solvent=solvent_model,
                    kappa=Float32(0.7),
                    dispersion_correction=false,
                    strictness=:nowarn,
                )
                sys32_mixed = System(sys32; nonbonded_energy_type=Float64)
                sys64 = System(
                    joinpath(data_dir, "6mrr_nowater.pdb"),
                    ff64;
                    units=false,
                    array_type=Array,
                    boundary=CubicBoundary(Float64(100.0)),
                    dist_cutoff=Float64(5.0),
                    nonbonded_method=:cutoff,
                    implicit_solvent=solvent_model,
                    kappa=Float64(0.7),
                    dispersion_correction=false,
                    strictness=:nowarn,
                )

                pe32 = potential_energy(sys32; n_threads=1)
                pe32_mixed = potential_energy(sys32_mixed; n_threads=1)
                pe64 = potential_energy(sys64; n_threads=1)

                @test !isempty(sys32.specific_inter_lists)
                @test !isempty(sys32.general_inters)
                @test typeof(pe32_mixed) === Float64
                @test abs(pe32_mixed - pe64) < abs(pe32 - pe64)
                @test pe32_mixed ≈ pe64 atol=5e-2 rtol=1e-5

                if local_run_cuda_tests
                    sys_gpu_default = Molly.to_device(sys32, CuArray)
                    sys_gpu_mixed = Molly.to_device(sys32_mixed, CuArray)
                    pe_gpu_default = potential_energy(sys_gpu_default; n_threads=1)
                    pe_gpu_mixed = potential_energy(sys_gpu_mixed; n_threads=1)

                    @test typeof(pe_gpu_mixed) === Float64
                    @test pe_gpu_default ≈ pe64 atol=5e-2 rtol=1e-5
                    @test pe_gpu_mixed ≈ pe64 atol=5e-2 rtol=1e-5
                    @test pe_gpu_mixed ≈ pe32_mixed atol=5e-2 rtol=1e-5
                end
            end
        end
    end
end

@testset "Mixed-Precision AWH" begin
    n_atoms = 10
    boundary = CubicBoundary(2.0u"nm")
    coords = [SVector{3, Float32}(0.1, 0.1, 0.1) * i for i in 1:n_atoms]u"nm"
    temp = 298.0u"K"

    @testset "Float64 nonbonded energy type" begin
        thermo_states = ThermoState[]
        for _ in 1:2
            atoms = [Atom(mass=10.0u"g/mol", charge=0.0, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:n_atoms]
            sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=false),),
                nonbonded_energy_type=Float64,
            )
            intg = VelocityVerlet(dt=0.002u"ps")
            push!(thermo_states, ThermoState(sys, intg; temperature=temp))
        end

        awh_state = AWHState(thermo_states; first_state=1)

        @test eltype(awh_state.f) == Float64
        @test eltype(awh_state.rho) == Float64
        @test eltype(awh_state.w_seg) == Float64
        @test eltype(awh_state.scratch_potentials) == Float64
        @test eltype(awh_state.scratch_z) == Float64
        @test eltype(awh_state.λ_β) == Float64
        @test eltype(awh_state.λ_p) == Float64
        @test eltype(eltype(ustrip.(awh_state.active_sys.coords))) == Float32

        pe = potential_energy(awh_state.active_sys)
        @test typeof(ustrip(pe)) == Float64

        awh_sim = AWHSimulation(awh_state; update_freq=1)
        simulate!(awh_sim, 10)
        @test all(isfinite, awh_sim.state.f)
    end

    @testset "Default precision follows system storage" begin
        thermo_states = ThermoState[]
        for _ in 1:2
            atoms = [Atom(mass=10.0f0u"g/mol", charge=0.0f0, σ=0.3f0u"nm", ϵ=0.2f0u"kJ * mol^-1") for _ in 1:n_atoms]
            sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                pairwise_inters=(LennardJones(use_neighbors=false),),
            )
            intg = VelocityVerlet(dt=0.002u"ps")
            push!(thermo_states, ThermoState(sys, intg; temperature=temp))
        end

        awh_state = AWHState(thermo_states; first_state=1)

        @test eltype(awh_state.f) == Float32
        @test eltype(awh_state.rho) == Float32
        @test eltype(awh_state.λ_β) == Float32
        @test eltype(awh_state.λ_p) == Float32
    end
end
