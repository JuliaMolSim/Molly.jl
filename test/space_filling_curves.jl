@testset "Space-filling curves" begin
    # The order of the atoms of GPUNeighborFinder along the Morton or the Hilbert curve, see
    #   `sort_reorder_seq!`. The kernels using the order are tested with both curves in
    #   tiled_kernels.jl and gpu_consistency.jl.

    # The Hilbert curve through the 2^bits cells along each axis visits every cell once,
    #   starts at the origin, goes from a cell to one that shares a face with it, and each
    #   aligned run of 2^(l * D) cells along it is an aligned cube of 2^l cells along each axis
    for (D, bits) in ((2, 1), (2, 2), (2, 5), (3, 1), (3, 2), (3, 4))
        n = 1 << bits
        cells = vec([ntuple(k -> UInt32(c[k] - 1), D)
                     for c in CartesianIndices(ntuple(_ -> n, D))])
        codes = [Molly.hilbert_code(c, bits, Val(D)) for c in cells]
        @test codes isa Vector{Int32}
        @test sort(codes) == Int32.(0:(n^D - 1))
        order = cells[sortperm(codes)]
        @test order[1] == ntuple(_ -> UInt32(0), D)
        @test all(k -> sum(abs.(Int.(order[k]) .- Int.(order[k - 1]))) == 1, 2:length(order))
        @test all(1:(bits - 1)) do l
            run_length = 1 << (D * l)
            all(1:(length(order) ÷ run_length)) do r
                run = order[((r - 1) * run_length + 1):(r * run_length)]
                allequal(map(c -> c .>> l, run))
            end
        end
    end

    # With the 10 bits used in 3D the codes fit in 30 bits, cells give different codes, and
    #   the cells of an octant of the grid share the top 3 bits of their codes
    rng = Xoshiro(10)
    cells = unique([Tuple(rand(rng, UInt32(0):UInt32(1023), 3)) for _ in 1:20_000])
    codes = [Molly.hilbert_code(c, 10, Val(3)) for c in cells]
    @test all(c -> 0 <= c < 2^30, codes)
    @test allunique(codes)
    @test all(k -> (cells[k] .>> 9 == cells[k + 1] .>> 9) == (codes[k] >> 27 == codes[k + 1] >> 27),
              1:(length(cells) - 1))

    # The cells are cubic, with 2^bits of them along the longest finite side
    T = Float64
    w, finite_sides = Molly.hilbert_grid(CubicBoundary(T(8.0)), 10)
    @test w == T(8.0) / 1024
    @test finite_sides == SVector(true, true, true)
    w, finite_sides = Molly.hilbert_grid(CubicBoundary(T(16.0), T(16.0), T(4.0)), 10)
    @test w == T(16.0) / 1024
    w, finite_sides = Molly.hilbert_grid(CubicBoundary(T(10.0), T(Inf), T(10.0)), 10)
    @test w == T(10.0) / 1024
    @test finite_sides == SVector(true, false, true)
    w, finite_sides = Molly.hilbert_grid(CubicBoundary(T(Inf)), 10)
    @test finite_sides == SVector(false, false, false)
    w, finite_sides = Molly.hilbert_grid(RectangularBoundary(T(4.0), T(9.0)), 10)
    @test w == T(9.0) / 1024
    @test finite_sides == SVector(true, true)
    w, finite_sides = Molly.hilbert_grid(TriclinicBoundary(SVector(T(4.0), T(0.0), T(0.0)),
                            SVector(T(1.0), T(5.0), T(0.0)), SVector(T(-1.0), T(1.0), T(3.0))), 10)
    @test w == T(5.0) / 1024
    w, finite_sides = Molly.hilbert_grid(CubicBoundary(T(8.0)u"nm"), 10)
    @test w == T(8.0)u"nm" / 1024

    # The cell of a coordinate is that of its image in the box, at the edges of the box too
    side, w = T(8.0), T(8.0) / 1024
    cell_index(x) = Molly.hilbert_cell_index(x, side, w, true, 10)
    @test cell_index(T(0.0)) == 0
    @test cell_index(T(4.0)) == 512
    @test cell_index(prevfloat(side)) == 1023
    @test cell_index(side) == 0
    @test cell_index(-eps(side)) == 1023
    @test cell_index(T(4.0) + 3 * side) == 512
    @test cell_index(T(4.0) - 2 * side) == 512
    @test Molly.hilbert_cell_index(T(123.0), T(Inf), w, false, 10) == 0
    @test Molly.hilbert_cell_index(T(4.0)u"nm", side * u"nm", w * u"nm", true, 10) == 512

    # The kernel gives the codes of the cells of the coordinates, in the box or outside it,
    #   and sorts the atoms by them
    tiled_array_types = filter(AT -> Molly.supports_tiled_kernels(Molly.array_type_backend(AT)),
                               array_list)
    boundaries = (
        CubicBoundary(T(6.0), T(5.0), T(3.0)),
        RectangularBoundary(T(6.0), T(2.5)),
        TriclinicBoundary(SVector(T(4.0), T(0.0), T(0.0)), SVector(T(1.0), T(4.0), T(0.0)),
                          SVector(T(-1.5), T(1.0), T(4.0))),
    )
    for AT in tiled_array_types, boundary in boundaries
        D = AtomsBase.n_dimensions(boundary)
        n_atoms = 500
        coords = place_atoms(n_atoms, boundary; min_dist=T(0.05), rng=Xoshiro(D))
        atoms = [Atom(index=i, mass=T(1.0), charge=T(0.0), σ=T(0.1), ϵ=T(0.1))
                 for i in 1:n_atoms]
        sys = System(atoms=to_device(atoms, AT), coords=to_device(coords, AT),
                     boundary=boundary,
                     pairwise_inters=(LennardJones(cutoff=DistanceCutoff(T(1.0))),),
                     neighbor_finder=GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=T(1.0),
                                                       curve=:hilbert, array_type=AT),
                     force_units=NoUnits, energy_units=NoUnits)
        @test sys.neighbor_finder.curve == :hilbert
        buffers = Molly.init_buffers_gpu(sys, 1)

        Molly.sort_reorder_seq!(buffers, sys.coords, sys.boundary; curve=:hilbert)
        codes = from_device(buffers.reorder_codes)[1:n_atoms]
        w, _ = Molly.hilbert_grid(boundary, 10)
        sides = Molly.box_sides(boundary)
        expected = map(coords) do c
            c_box = Molly.aligned_coord(c, boundary)
            cell = ntuple(k -> UInt32(clamp(floor(Int, mod(c_box[k], sides[k]) / w), 0, 1023)),
                          D)
            Molly.hilbert_code(cell, 10, Val(D))
        end
        @test codes == expected
        seq = from_device(buffers.reorder_seq)[1:n_atoms]
        @test sort(seq) == 1:n_atoms
        @test issorted(codes[seq])

        # Images of the atoms in other cells of the lattice are in the same cells
        # `box_matrix` is 3x3 for a 2D box as well
        B = Molly.box_matrix(boundary)[1:D, 1:D]
        shift_rng = Xoshiro(D + 10)
        coords_shifted = [c + SVector{D, T}(B * rand(shift_rng, -2:2, D)) for c in coords]
        Molly.sort_reorder_seq!(buffers, to_device(coords_shifted, AT), sys.boundary;
                                 curve=:hilbert)
        @test from_device(buffers.reorder_codes)[1:n_atoms] == codes

        # The Morton curve is still the default, and the curve is checked
        Molly.sort_reorder_seq!(buffers, sys.coords, sys.boundary)
        morton_codes = from_device(buffers.reorder_codes)[1:n_atoms]
        @test morton_codes != codes
        @test_throws ArgumentError Molly.sort_reorder_seq!(buffers, sys.coords, sys.boundary;
                                                            curve=:peano)
        if D == 3
            # The codes are Int32, which holds 3 * 10 bits but not 3 * 11
            @test_throws ArgumentError Molly.sort_reorder_seq!(buffers, sys.coords,
                                                    sys.boundary, 11; curve=:hilbert)
        end

        if AT <: AbstractGPUArray
            # Converting the system keeps the curve
            @test System(sys; coords=sys.coords).neighbor_finder.curve == :hilbert
        end

        if AT <: AbstractGPUArray
            # A new curve on the neighbor finder orders the atoms at the next force
            #   calculation, although they have not moved
            force_buffers = Molly.init_buffers!(sys, 1)
            fs = Molly.zero_forces(sys)
            Molly.forces!(fs, sys, nothing, 0, force_buffers, Val(false))
            @test from_device(force_buffers.reorder_codes)[1:n_atoms] == codes
            sys.neighbor_finder.curve = :morton
            Molly.forces!(fs, sys, nothing, 1, force_buffers, Val(false))
            @test from_device(force_buffers.reorder_codes)[1:n_atoms] == morton_codes
            @test from_device(fs) ≈ from_device(forces(sys))
            sys.neighbor_finder.curve = :hilbert
        end
    end

    @test GPUNeighborFinder(n_atoms=10, dist_cutoff=1.0, array_type=Array).curve == :morton
    @test GPUNeighborFinder(n_atoms=10, dist_cutoff=1.0, curve=:hilbert,
                            array_type=Array).curve == :hilbert
    @test_throws ArgumentError GPUNeighborFinder(n_atoms=10, dist_cutoff=1.0, curve=:peano,
                                                 array_type=Array)

    # The curve of an existing neighbor finder is checked as well, and a change of curve
    #   makes the atoms be ordered again
    nf = GPUNeighborFinder(n_atoms=10, dist_cutoff=1.0, array_type=Array)
    generation = nf.reorder_generation
    @test_throws ArgumentError (nf.curve = :peano)
    @test nf.curve == :morton
    nf.curve = :hilbert
    @test nf.curve == :hilbert
    @test nf.reorder_generation == generation + 1
    nf.curve = :hilbert
    @test nf.reorder_generation == generation + 1
    @test occursin("curve = hilbert", sprint(show, nf))

    # A coordinate that is not finite throws an error, as with the Morton curve
    @test Molly.hilbert_cell_index(1.2f0, 2.0f0, 0.5f0, true, 2) == UInt32(2)
    @test Molly.hilbert_cell_index(-0.1f0, 2.0f0, 0.5f0, true, 2) == UInt32(3)
    @test_throws InexactError Molly.hilbert_cell_index(NaN32, 2.0f0, 0.5f0, true, 2)
end
