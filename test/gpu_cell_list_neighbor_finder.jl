@testset "GPU cell-list neighbor finder" begin
    function gpu_cell_list_test_system(
        coords_cpu;
        output=:ragged,
        cutoff=1.0f0,
        max_neighbors=64,
        boundary=CubicBoundary(10.0f0),
    )
        n_atoms = length(coords_cpu)

        atoms = CuArray([
            Molly.Atom(index=i, mass=1.0f0)
            for i in 1:n_atoms
        ])

        finder = GPUCellListNeighborFinder(
            dist_cutoff=cutoff,
            n_steps=10,
            max_neighbors=max_neighbors,
            output=output,
        )

        sys = System(
            atoms=atoms,
            coords=CuArray(coords_cpu),
            boundary=boundary,
            neighbor_finder=finder,
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        return sys, finder
    end

    @testset "Ragged output" begin
        coords = [
            SVector{3,Float32}(1.0, 2.0, 3.0),
            SVector{3,Float32}(1.5, 2.0, 3.0),
            SVector{3,Float32}(4.0, 2.0, 3.0),
        ]

        sys, _ = gpu_cell_list_test_system(coords)
        result = find_neighbors(sys)
        CUDA.synchronize()

        counts = Array(result.counts)
        matrix = Array(result.neighbors)

        @test counts == Int32[1, 1, 0]
        @test matrix[1:counts[1], 1] == Int32[2]
        @test matrix[1:counts[2], 2] == Int32[1]
        @test isempty(matrix[1:counts[3], 3])
        @test result.list === nothing
    end

    @testset "Geometric pair output" begin
        coords = [
            SVector{3,Float32}(1.0, 2.0, 3.0),
            SVector{3,Float32}(1.5, 2.0, 3.0),
            SVector{3,Float32}(4.0, 2.0, 3.0),
        ]

        sys, _ = gpu_cell_list_test_system(
            coords;
            output=:geometric_pairs,
        )

        result = find_neighbors(sys)
        CUDA.synchronize()

        pairs = Array(result.list[1:result.n])

        @test result.n == 1
        @test pairs == [(Int32(2), Int32(1), false)]
    end

    @testset "Periodic boundary" begin
        coords = [
            SVector{3,Float32}(0.1, 2.0, 3.0),
            SVector{3,Float32}(9.7, 2.0, 3.0),
            SVector{3,Float32}(5.0, 2.0, 3.0),
        ]

        sys, _ = gpu_cell_list_test_system(
            coords;
            output=:geometric_pairs,
            cutoff=0.5f0,
        )

        result = find_neighbors(sys)
        CUDA.synchronize()

        @test Array(result.counts) == Int32[1, 1, 0]
        @test result.n == 1
        @test Array(result.list[1:1]) ==
              [(Int32(2), Int32(1), false)]
    end

    @testset "Cell occupancy above warp size" begin
        n_atoms = 40

        coords = [
            SVector{3,Float32}(
                1.0f0 + Float32(i) * 0.001f0,
                2.0f0,
                3.0f0,
            )
            for i in 1:n_atoms
        ]

        sys, _ = gpu_cell_list_test_system(
            coords;
            output=:geometric_pairs,
            cutoff=1.0f0,
            max_neighbors=64,
        )

        result = find_neighbors(sys)
        CUDA.synchronize()

        @test Array(result.counts) == fill(Int32(39), n_atoms)
        @test result.n == n_atoms * (n_atoms - 1) ÷ 2
    end

    @testset "State reuse" begin
        coords = [
            SVector{3,Float32}(1.0, 2.0, 3.0),
            SVector{3,Float32}(1.5, 2.0, 3.0),
            SVector{3,Float32}(4.0, 2.0, 3.0),
        ]

        sys, finder = gpu_cell_list_test_system(
            coords;
            output=:geometric_pairs,
        )

        first_result = find_neighbors(sys)
        CUDA.synchronize()

        sys.coords .= CuArray([
            SVector{3,Float32}(1.0, 2.0, 3.0),
            SVector{3,Float32}(6.0, 2.0, 3.0),
            SVector{3,Float32}(4.0, 2.0, 3.0),
        ])

        second_result = find_neighbors(
            sys,
            finder,
            first_result,
            1,
            true,
        )
        CUDA.synchronize()

        @test Array(second_result.counts) == Int32[0, 0, 0]
        @test second_result.n == 0
        @test second_result.state.x === first_result.state.x
        @test second_result.neighbors === first_result.neighbors
        @test second_result.counts === first_result.counts
        @test second_result.list === first_result.list

        cached_result = find_neighbors(
            sys,
            finder,
            second_result,
            2,
            false,
        )

        @test cached_result === second_result
    end

    @testset "Neighbor capacity overflow" begin
        coords = [
            SVector{3,Float32}(1.0, 2.0, 3.0),
            SVector{3,Float32}(1.1, 2.0, 3.0),
            SVector{3,Float32}(1.2, 2.0, 3.0),
        ]

        for output in (:ragged, :geometric_pairs)
            sys, _ = gpu_cell_list_test_system(
                coords;
                output=output,
                cutoff=1.0f0,
                max_neighbors=1,
            )

            @test_throws ErrorException find_neighbors(sys)
        end
    end

    @testset "Boundary validation" begin
        coords = [
            SVector{3,Float32}(0.5, 0.5, 0.5),
            SVector{3,Float32}(1.0, 0.5, 0.5),
        ]

        small_sys, _ = gpu_cell_list_test_system(
            coords;
            cutoff=1.0f0,
            boundary=CubicBoundary(2.5f0),
        )

        @test_throws ArgumentError find_neighbors(small_sys)

        infinite_sys, _ = gpu_cell_list_test_system(
            coords;
            cutoff=1.0f0,
            boundary=CubicBoundary(Inf32),
        )

        @test_throws ArgumentError find_neighbors(infinite_sys)

        triclinic_boundary = TriclinicBoundary(
            SVector(
                SVector{3,Float32}(10.0, 0.0, 0.0),
                SVector{3,Float32}(0.5, 10.0, 0.0),
                SVector{3,Float32}(0.0, 0.0, 10.0),
            ),
        )

        triclinic_sys, _ = gpu_cell_list_test_system(
            coords;
            cutoff=1.0f0,
            boundary=triclinic_boundary,
        )

        @test_throws ArgumentError find_neighbors(triclinic_sys)
    end

    @testset "Finder validation" begin
        @test_throws ArgumentError GPUCellListNeighborFinder(
            dist_cutoff=0.0f0,
        )

        @test_throws ArgumentError GPUCellListNeighborFinder(
            dist_cutoff=Inf32,
        )

        @test_throws ArgumentError GPUCellListNeighborFinder(
            dist_cutoff=1.0f0,
            n_steps=0,
        )

        @test_throws ArgumentError GPUCellListNeighborFinder(
            dist_cutoff=1.0f0,
            max_neighbors=0,
        )

        @test_throws ArgumentError GPUCellListNeighborFinder(
            dist_cutoff=1.0f0,
            output=:invalid,
        )
    end
end