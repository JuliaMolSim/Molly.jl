@testset "GPU Optimizations" begin
    # Every GPU backend that supports the tiled kernels of GPUNeighborFinder, in Float32 on
    #   backends without Float64
    for AT in array_list_metal[2:end]
        Molly.uses_gpu_neighbor_finder(AT) || continue
        n_atoms = 100
        D = 3
        T = Molly.default_float_type(AT, Float64)
        coords = [SVector{D, T}(0.1 * i, 0.1 * i, 0.1 * i) for i in 1:n_atoms]
        boundary = CubicBoundary(T(20.0), T(20.0), T(20.0))
        atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0), λ=T(1.0)) for i in 1:n_atoms]
        
        sys = System(
            atoms=AT(atoms),
            coords=AT(coords),
            boundary=boundary,
            pairwise_inters=(LennardJones(),),
            neighbor_finder=GPUNeighborFinder(
                n_atoms=n_atoms,
                dist_cutoff=T(5.0),
                array_type=AT,
            ),
            force_units=NoUnits,
            energy_units=NoUnits
        )
        
        n_threads = 256
        buffers = Molly.init_buffers!(sys, n_threads)

        @testset "Tile launch config API" begin
            Molly.reset_tile_launch_config!(sys)
            cfg_auto = Molly.tile_launch_config(sys)
            @test cfg_auto.force_block_y == Molly.TILE_LAUNCH_UNSET
            @test cfg_auto.force_maxregs == Molly.TILE_LAUNCH_UNSET
            @test cfg_auto.tile_threads == Molly.TILE_THREADS_UNSET
            @test cfg_auto.energy_block_y == Molly.TILE_LAUNCH_UNSET
            @test Molly.config_force_block_y(cfg_auto) === nothing
            @test Molly.config_force_maxregs(cfg_auto) === nothing
            @test Molly.config_tile_threads(cfg_auto) === nothing
            @test Molly.config_energy_block_y(cfg_auto) === nothing

            cfg_explicit = Molly.set_tile_launch_config!(sys;
                force_block_y=T == Float64 ? 8 : 12,
                force_maxregs=56,
                tile_threads=(16, 8),
                energy_block_y=4,
            )
            cfg_current = Molly.tile_launch_config(sys)
            @test Molly.config_force_block_y(cfg_current) == Molly.config_force_block_y(cfg_explicit)
            @test Molly.config_force_maxregs(cfg_current) == Molly.config_force_maxregs(cfg_explicit)
            @test Molly.config_tile_threads(cfg_current) == Molly.config_tile_threads(cfg_explicit)
            @test Molly.config_energy_block_y(cfg_current) == Molly.config_energy_block_y(cfg_explicit)

            fs_api = forces(sys, find_neighbors(sys))
            @test length(fs_api) == n_atoms

            Molly.reset_tile_launch_config!(sys)
            cfg_reset = Molly.tile_launch_config(sys)
            @test Molly.config_force_block_y(cfg_reset) === nothing
            @test Molly.config_force_maxregs(cfg_reset) === nothing
            @test Molly.config_tile_threads(cfg_reset) === nothing
            @test Molly.config_energy_block_y(cfg_reset) === nothing
        end

        @testset "Tile launch autotuner" begin

            Molly.reset_tile_launch_config!(sys)
            Molly.reset_tile_launch_autotune_cache!()
            @test isempty(Molly.TILE_LAUNCH_AUTOTUNE_CACHE)

            chosen_block_y = optimize_tile_launch_config!(sys)
            tuned_cfg = Molly.tile_launch_config(sys)
            @test chosen_block_y == Molly.config_force_block_y(tuned_cfg)
            @test Molly.config_force_block_y(tuned_cfg) in Molly.AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES
            @test Molly.config_energy_block_y(tuned_cfg) in Molly.AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES
            @test Molly.config_tile_threads(tuned_cfg) in Molly.AUTOTUNE_TILE_THREAD_CANDIDATES
            @test length(Molly.TILE_LAUNCH_AUTOTUNE_CACHE) == 1

            # Test environment variable overrides
            ENV["MOLLY_TILE_FORCE_BLOCK_Y"] = "1"
            ENV["MOLLY_TILE_ENERGY_BLOCK_Y"] = "1"
            ENV["MOLLY_TILE_THREADS_X"] = "8"
            ENV["MOLLY_TILE_THREADS_Y"] = "8"
            
            Molly.reset_tile_launch_config!(sys)
            # optimize_tile_launch_config! returns the effective force_block_y
            eff_block_y = optimize_tile_launch_config!(sys)
            @test eff_block_y == 1
            
            # The global config should be updated with what was tuned (if not overridden in config),
            # but env vars are not stored in the config struct.
            # We can verify that the config has the tuned values for others if we didn't override them in config.
            
            delete!(ENV, "MOLLY_TILE_FORCE_BLOCK_Y")
            delete!(ENV, "MOLLY_TILE_ENERGY_BLOCK_Y")
            delete!(ENV, "MOLLY_TILE_THREADS_X")
            delete!(ENV, "MOLLY_TILE_THREADS_Y")

            Molly.reset_tile_launch_config!(sys)
            cached_block_y = optimize_tile_launch_config!(sys)
            cached_cfg = Molly.tile_launch_config(sys)
            @test cached_block_y == Molly.config_force_block_y(tuned_cfg)
            @test cached_cfg == tuned_cfg
            @test length(Molly.TILE_LAUNCH_AUTOTUNE_CACHE) == 1

            Molly.reset_tile_launch_config!(sys)
            Molly.set_tile_launch_config!(sys; force_block_y=8)
            optimize_tile_launch_config!(sys)
            merged_cfg = Molly.tile_launch_config(sys)
            @test Molly.config_force_block_y(merged_cfg) == 8
            @test Molly.config_energy_block_y(merged_cfg) !== nothing
            @test Molly.config_tile_threads(merged_cfg) !== nothing
            @test length(Molly.TILE_LAUNCH_AUTOTUNE_CACHE) == 1

            cfg_before_reset = Molly.tile_launch_config(sys)
            Molly.reset_tile_launch_autotune_cache!()
            @test isempty(Molly.TILE_LAUNCH_AUTOTUNE_CACHE)
            @test Molly.tile_launch_config(sys) == cfg_before_reset

            Molly.reset_tile_launch_config!(sys)
        end

        @testset "Setup-time tile launch autotune" begin

            ff = MolecularForceField(joinpath(ff_dir, "tip3p_standard.xml"); units=true)
            simulator = VelocityVerlet(dt=T(0.001))
            setup_sys(; autotune_launch=true, launch_config=Molly.TileLaunchConfig()) = System(
                joinpath(data_dir, "water_3mol_cubic.pdb"),
                ff;
                array_type=AT,
                float_type=T,
                nonbonded_method=SetupCoulombReactionField(),
                dist_cutoff=0.6u"nm",
                dist_buffer=0.1u"nm",
                launch_config=launch_config,
                autotune_launch=autotune_launch,
            )

            Molly.reset_tile_launch_autotune_cache!()
            auto_sys = setup_sys()
            auto_cfg = Molly.tile_launch_config(auto_sys)
            @test auto_sys.neighbor_finder isa GPUNeighborFinder
            @test auto_sys.neighbor_finder.dist_cutoff ≈ T(0.7)u"nm"
            @test Molly.config_force_block_y(auto_cfg) in Molly.AUTOTUNE_FORCE_BLOCK_Y_CANDIDATES
            @test Molly.config_energy_block_y(auto_cfg) in Molly.AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES
            @test Molly.config_tile_threads(auto_cfg) in Molly.AUTOTUNE_TILE_THREAD_CANDIDATES
            @test length(Molly.TILE_LAUNCH_AUTOTUNE_CACHE) == 1

            Molly.reset_tile_launch_autotune_cache!()
            opt_out_sys = setup_sys(autotune_launch=false)
            opt_out_cfg = Molly.tile_launch_config(opt_out_sys)
            @test Molly.config_force_block_y(opt_out_cfg) === nothing
            @test Molly.config_force_maxregs(opt_out_cfg) === nothing
            @test Molly.config_tile_threads(opt_out_cfg) === nothing
            @test Molly.config_energy_block_y(opt_out_cfg) === nothing
            @test isempty(Molly.TILE_LAUNCH_AUTOTUNE_CACHE)

            Molly.reset_tile_launch_autotune_cache!()
            merged_sys = setup_sys(launch_config=Molly.TileLaunchConfig(force_block_y=8))
            merged_cfg = Molly.tile_launch_config(merged_sys)
            @test Molly.config_force_block_y(merged_cfg) == 8
            @test Molly.config_energy_block_y(merged_cfg) in Molly.AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES
            @test Molly.config_tile_threads(merged_cfg) in Molly.AUTOTUNE_TILE_THREAD_CANDIDATES
            @test length(Molly.TILE_LAUNCH_AUTOTUNE_CACHE) == 1

            simulate_sys = System(sys; coords=copy(sys.coords), velocities=copy(sys.velocities))
            optimize_tile_launch_config!(simulate_sys)
            cfg_before = Molly.tile_launch_config(simulate_sys)
            simulate!(simulate_sys, simulator, 1)
            @test Molly.tile_launch_config(simulate_sys) == cfg_before
        end

        @testset "Tile launch autotune triclinic unitless" begin
            n_atoms_tri = 48
            coords_tri = [SVector{D, T}(0.12 * i, 0.07 * i, 0.05 * i) for i in 1:n_atoms_tri]
            boundary_tri = TriclinicBoundary(
                SVector(T(12.0), T(0.0), T(0.0)),
                SVector(T(1.5), T(11.0), T(0.0)),
                SVector(T(0.5), T(1.0), T(10.0)),
            )
            atoms_tri = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0), λ=T(1.0)) for i in 1:n_atoms_tri]

            sys_tri = System(
                atoms=AT(atoms_tri),
                coords=AT(coords_tri),
                boundary=boundary_tri,
                pairwise_inters=(LennardJones(cutoff=DistanceCutoff(T(4.0))),),
                neighbor_finder=GPUNeighborFinder(
                    n_atoms=n_atoms_tri,
                    dist_cutoff=T(4.0),
                    array_type=AT,
                ),
                force_units=NoUnits,
                energy_units=NoUnits,
            )

            Molly.reset_tile_launch_config!(sys_tri)
            Molly.reset_tile_launch_autotune_cache!()
            chosen_block_y = optimize_tile_launch_config!(sys_tri)
            tuned_cfg = Molly.tile_launch_config(sys_tri)

            @test chosen_block_y == Molly.config_force_block_y(tuned_cfg)
            @test Molly.config_energy_block_y(tuned_cfg) in Molly.AUTOTUNE_ENERGY_BLOCK_Y_CANDIDATES
            @test Molly.config_tile_threads(tuned_cfg) in Molly.AUTOTUNE_TILE_THREAD_CANDIDATES
        end

        @testset "Morton Code Granularity (Phase 1)" begin
            Molly.sort_reorder_seq!(buffers, sys.coords, sys.boundary)
            
            reorder_codes = Array(buffers.reorder_codes)
            @test length(unique(reorder_codes)) == n_atoms

            # Each axis has a power of two cells, chosen so that the cells are close to cubic
            w, n_cells_m1 = Molly.morton_grid(sys.boundary, 10)
            @test w == Molly.box_sides(sys.boundary) ./ 1024
            @test n_cells_m1 == SVector{3, UInt32}(1023, 1023, 1023)
            w, n_cells_m1 = Molly.morton_grid(CubicBoundary(T(16.0), T(16.0), T(4.0)), 10)
            @test n_cells_m1 == SVector{3, UInt32}(1023, 1023, 255)
            @test w[1] == w[3]
            w, n_cells_m1 = Molly.morton_grid(CubicBoundary(T(10.0), T(Inf), T(10.0)), 10)
            @test n_cells_m1 == SVector{3, UInt32}(1023, 0, 1023)
            w, n_cells_m1 = Molly.morton_grid(RectangularBoundary(T(4.0), T(9.0)), 10)
            @test n_cells_m1 == SVector{2, UInt32}(511, 1023)
        end
        
        @testset "Physical Data Reordering (Phase 2)" begin
            Molly.sort_reorder_seq!(buffers, sys.coords, sys.boundary)
            
            backend = Molly.get_backend(sys.coords)
            Molly.reorder_kernel!(backend, n_threads)(buffers.coords_reordered, sys.coords,
                                                      buffers.reorder_seq; ndrange=n_atoms)

            reordered_coords = Array(buffers.coords_reordered)
            orig_coords = Array(sys.coords)
            reorder_seq = Array(buffers.reorder_seq)
            
            for i in 1:n_atoms
                @test reordered_coords[i] ≈ orig_coords[reorder_seq[i]]
            end
            
            fill!(buffers.fs_mat, 0.0)
            fill!(buffers.fs_mat_reordered, 1.0)
            Molly.reverse_reorder_forces_kernel!(backend, n_threads)(buffers.fs_mat,
                        buffers.fs_mat_reordered, buffers.reorder_seq, Val(3);
                        ndrange=n_atoms)

            fs_mat = Array(buffers.fs_mat)
            for i in 1:n_atoms
                orig_idx = reorder_seq[i]
                @test fs_mat[1, orig_idx] ≈ 1.0
                @test fs_mat[2, orig_idx] ≈ 1.0
                @test fs_mat[3, orig_idx] ≈ 1.0
            end
        end
    end
end

# General interactions to check the order of the forces on the GPU, see
#   `concurrent_with_pairwise`: one that adds a constant force to every atom and one that
#   records the forces it is given
struct ConstantForceInter{F}
    f::F
    concurrent::Bool
end

Molly.concurrent_with_pairwise(inter::ConstantForceInter) = inter.concurrent

function AtomsCalculators.forces!(fs, sys, inter::ConstantForceInter; kwargs...)
    fs .+= (inter.f,)
    return fs
end

mutable struct RecordForcesInter
    seen::Any
end

function AtomsCalculators.forces!(fs, sys, inter::RecordForcesInter; kwargs...)
    inter.seen = from_device(fs)
    return fs
end

@testset "Order of general interactions on the GPU" begin
    rec = RecordForcesInter(nothing)
    c1 = ConstantForceInter(SVector(1.0, 2.0, 3.0), true)
    c2 = ConstantForceInter(SVector(10.0, 20.0, 30.0), true)
    @test Molly.n_general_alongside((c1, rec, c2)) == 1
    @test Molly.n_general_alongside((c1, c2)) == 2
    @test Molly.n_general_alongside((rec, c1)) == 0
    @test Molly.n_general_alongside(()) == 0

    for AT in array_list_metal[2:end]
        Molly.uses_gpu_neighbor_finder(AT) || continue
        n_atoms = 100
        T = Molly.default_float_type(AT, Float64)
        coords = place_atoms(n_atoms, CubicBoundary(T(3.0)); min_dist=T(0.25),
                             rng=Xoshiro(10))
        atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.3), ϵ=T(1.0), λ=T(1.0))
                 for i in 1:n_atoms]
        c1 = ConstantForceInter(SVector(T(1.0), T(2.0), T(3.0)), true)
        c2 = ConstantForceInter(SVector(T(10.0), T(20.0), T(30.0)), true)
        for grad_safe in (false, true)
            rec = RecordForcesInter(nothing)
            sys = System(
                atoms=AT(atoms),
                coords=AT(coords),
                boundary=CubicBoundary(T(3.0)),
                pairwise_inters=(LennardJones(cutoff=DistanceCutoff(T(1.0))),),
                general_inters=(c1, rec, c2),
                neighbor_finder=GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=T(1.2),
                                                  array_type=AT),
                force_units=NoUnits,
                energy_units=NoUnits,
                grad_safe=grad_safe,
            )
            fs = from_device(forces(sys))
            fs_pairwise = from_device(forces(System(sys; general_inters=())))
            @test maximum(norm.(fs_pairwise)) > 1
            # The recording interaction sees the forces of the interactions before it, and
            #   the one after it is added afterwards
            @test rec.seen ≈ fs_pairwise .+ (c1.f,)
            @test fs ≈ fs_pairwise .+ (c1.f + c2.f,)
        end
    end
end
