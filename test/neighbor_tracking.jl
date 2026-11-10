# Neighbor finders that rebuild the neighbor list when the atoms have moved far enough to
#   invalidate it, auto_update=true

# The pairs within `cutoff` of each other, as (i, j) with i < j
function brute_force_pairs(coords, boundary, cutoff)
    n_atoms = length(coords)
    pairs = Set{Tuple{Int, Int}}()
    for i in 1:n_atoms, j in (i + 1):n_atoms
        if norm(vector(coords[i], coords[j], boundary)) <= cutoff
            push!(pairs, (i, j))
        end
    end
    return pairs
end

neighbor_pair_set(nbs) = Set((min(Int(i), Int(j)), max(Int(i), Int(j)))
                             for (i, j, _) in from_device(neighbor_pairs(nbs)))

# A general interaction with no forces or energy that records whether the neighbor list it
#   is given has every pair within `cutoff`, it does not count as using the neighbor list
#   so it does not change when the list is rebuilt
mutable struct ListCompletenessCheck{C}
    cutoff::C
    n_checks::Int
    n_incomplete::Int
end

ListCompletenessCheck(cutoff) = ListCompletenessCheck(cutoff, 0, 0)

function check_list_complete!(c::ListCompletenessCheck, sys, neighbors)
    c.n_checks += 1
    needed = brute_force_pairs(from_device(sys.coords), sys.boundary, c.cutoff)
    if !issubset(needed, neighbor_pair_set(neighbors))
        c.n_incomplete += 1
    end
    return c
end

function AtomsCalculators.forces!(fs, sys, inter::ListCompletenessCheck; neighbors=nothing,
                                  kwargs...)
    check_list_complete!(inter, sys, neighbors)
    return fs
end

function AtomsCalculators.potential_energy(sys, inter::ListCompletenessCheck;
                                           neighbors=nothing, kwargs...)
    check_list_complete!(inter, sys, neighbors)
    return zero(float_type(sys)) * sys.energy_units
end

@testset "Neighbor list displacement criterion" begin
    r_l, r_c = 1.0, 0.8
    tri_basis = (SVector(3.0, 0.0, 0.0), SVector(0.8, 2.9, 0.0), SVector(0.6, 0.5, 3.1))
    boundaries = (
        CubicBoundary(3.0),
        CubicBoundary(Inf),
        RectangularBoundary(4.0),
        TriclinicBoundary(tri_basis...),
        TriclinicBoundary(tri_basis...; approx_images=false),
    )
    box_matrix(b::CubicBoundary) = SMatrix{3, 3}(Diagonal(b.side_lengths))
    box_matrix(b::RectangularBoundary) = SMatrix{2, 2}(Diagonal(b.side_lengths))
    box_matrix(b::TriclinicBoundary) = SMatrix{3, 3}(hcat(b.basis_vectors...))

    # A random deformation of the box that keeps the form Molly requires
    function deform_boundary(b::CubicBoundary, rng, size)
        isinf(b.side_lengths[1]) && return b
        return CubicBoundary(b.side_lengths .* (1 .+ size .* randn(rng, SVector{3})))
    end
    function deform_boundary(b::RectangularBoundary, rng, size)
        return RectangularBoundary(b.side_lengths .* (1 .+ size .* randn(rng, SVector{2})))
    end
    function deform_boundary(b::TriclinicBoundary{3, T, C, A}, rng, size) where {T, C, A}
        H = box_matrix(b)
        # Upper triangular so the first basis vector stays along x and the second in xy
        dH = SMatrix{3, 3}(randn(rng), 0.0, 0.0, randn(rng), randn(rng), 0.0,
                           randn(rng), randn(rng), randn(rng)) .* size .* H[1, 1]
        H2 = H + dH
        return TriclinicBoundary(H2[:, 1], H2[:, 2], H2[:, 3]; approx_images=A)
    end
    deformation(b0, b1) = box_matrix(b1) * inv(box_matrix(b0))
    deformation(b0::CubicBoundary, b1::CubicBoundary) = (isinf(b0.side_lengths[1]) ?
                        SMatrix{3, 3}(I) : box_matrix(b1) * inv(box_matrix(b0)))
    start_coords(b, n, rng) = [random_coord(b; rng=rng) for _ in 1:n]
    start_coords(b::CubicBoundary, n, rng) = (isinf(b.side_lengths[1]) ?
                        [3.0 .* rand(rng, SVector{3}) for _ in 1:n] :
                        [random_coord(b; rng=rng) for _ in 1:n])

    # Whenever a list is reused, every pair within the interaction cutoff at the new
    #   coordinates has to be in it, including when the box changes and when pairs move
    #   straight towards each other by close to the allowed amount
    rng = Xoshiro(1)
    n_atoms = 250
    for b0 in boundaries, box_size in (0.0, 0.005), adversarial in (false, true)
        n_reused, n_missing = 0, 0
        for trial in 1:20
            coords_0 = start_coords(b0, n_atoms, rng)
            b1 = (iszero(box_size) ? b0 : deform_boundary(b0, rng, box_size))
            F = deformation(b0, b1)
            D = length(coords_0[1])
            disp_scale = (adversarial ? 1e-5 : 0.002 + 0.015 * rand(rng))
            coords_1 = [F * c + disp_scale .* randn(rng, SVector{D}) for c in coords_0]
            if adversarial
                # A pair just outside the list cutoff moves straight towards each other
                #   by just less or just more than the allowed amount
                allowance = minimum(svdvals(F)) * r_l - r_c
                far_pairs = [(i, j) for i in 1:n_atoms for j in (i + 1):n_atoms
                             if r_l < norm(vector(coords_0[i], coords_0[j], b0)) < r_l + 0.01]
                i, j = rand(rng, far_pairs)
                dr = vector(coords_0[i], coords_0[j], b0)
                u = F * (dr / norm(dr))
                frac = (isodd(trial) ? 0.48 : 0.52)
                coords_1[i] = F * coords_0[i] + (allowance * frac / norm(u)) * u
                coords_1[j] = F * coords_0[j] - (allowance * frac / norm(u)) * u
            end
            coords_1 = wrap_coords.(coords_1, (b1,))
            if !Molly.neighbors_invalidated(coords_1, b1, coords_0, b0, r_l, r_c^2;
                                            n_threads=1)
                n_reused += 1
                needed = brute_force_pairs(coords_1, b1, r_c)
                listed = brute_force_pairs(coords_0, b0, r_l)
                n_missing += !issubset(needed, listed)
            end
        end
        @test n_reused > 0
        @test n_missing == 0
    end

    # The bound is tight for two atoms moving straight towards each other
    b = CubicBoundary(5.0)
    for (frac, expect_invalid) in ((0.999, false), (1.001, true))
        coords_0 = [SVector(1.0, 2.5, 2.5), SVector(1.0 + r_l + 1e-9, 2.5, 2.5)]
        move = SVector(frac * (r_l - r_c) / 2, 0.0, 0.0)
        coords_1 = [coords_0[1] + move, coords_0[2] - move]
        @test Molly.neighbors_invalidated(coords_1, b, coords_0, b, r_l, r_c^2) == expect_invalid
        @test (norm(vector(coords_1[1], coords_1[2], b)) <= r_c) == expect_invalid
    end
    # The two largest displacements are used rather than twice the largest
    coords_0 = [SVector(1.0, 1.0, 1.0), SVector(3.0, 3.0, 3.0), SVector(4.0, 1.0, 1.0)]
    move = SVector(0.6 * (r_l - r_c), 0.0, 0.0)
    @test !Molly.neighbors_invalidated([coords_0[1] + move, coords_0[2], coords_0[3]], b,
                                       coords_0, b, r_l, r_c^2)
    @test Molly.neighbors_invalidated([coords_0[1] + move, coords_0[2] + move, coords_0[3]], b,
                                      coords_0, b, r_l, r_c^2)
    # Without a known interaction cutoff, a changed box that is not a deformation, a skin
    #   that is not positive or different coordinates the list is always rebuilt
    @test Molly.neighbors_invalidated(coords_0, b, coords_0, b, r_l, nothing)
    @test Molly.neighbors_invalidated(coords_0, b, coords_0, b, r_l, (r_l + 0.1)^2)
    @test Molly.neighbors_invalidated(coords_0, b, coords_0, b, r_l, r_l^2)
    @test Molly.neighbors_invalidated(coords_0, b, coords_0[1:2], b, r_l, r_c^2)
    @test Molly.neighbors_invalidated(coords_0, b, nothing, b, r_l, r_c^2)
    @test Molly.neighbors_invalidated(coords_0, CubicBoundary(Inf), coords_0,
                                      CubicBoundary(20.0), r_l, r_c^2)
    @test !Molly.neighbors_invalidated(coords_0, b, coords_0, b, r_l, r_c^2)
    # NaN coordinates give a rebuild
    @test Molly.neighbors_invalidated([SVector(NaN, 1.0, 1.0), coords_0[2], coords_0[3]], b,
                                      coords_0, b, r_l, r_c^2)
    # Units, and an interaction cutoff in other units
    coords_0_u = coords_0 .* u"nm"
    b_u = CubicBoundary(5.0u"nm")
    @test !Molly.neighbors_invalidated(coords_0_u, b_u, coords_0_u, b_u, r_l * u"nm",
                                       (r_c * 10u"Å")^2)
    @test Molly.neighbors_invalidated([coords_0_u[1] + move * u"nm", coords_0_u[2] + move * u"nm",
                                       coords_0_u[3]], b_u, coords_0_u, b_u, r_l * u"nm",
                                      (r_c * 10u"Å")^2)
    # The threaded CPU check gives the same result as the serial one
    rng = Xoshiro(3)
    coords_many_0 = [random_coord(b; rng=rng) for _ in 1:20_000]
    coords_many_1 = wrap_coords.(coords_many_0 .+ [0.01 .* randn(rng, SVector{3})
                                                   for _ in 1:20_000], (b,))
    serial = Molly.max_sq_displacements(coords_many_1, coords_many_0, b, nothing, NoUnits,
                                        Float64, 1)
    threaded = Molly.max_sq_displacements(coords_many_1, coords_many_0, b, nothing, NoUnits,
                                          Float64, 4)
    @test serial == threaded
    d2_all = sort([sum(abs2, vector(c0, c1, b)) for (c0, c1) in zip(coords_many_0,
                                                                     coords_many_1)])
    @test serial == (d2_all[end], d2_all[end - 1])

    # The GPU reductions give the same displacements as the CPU one, including the generic
    #   kernel used on GPUs other than CUDA ones
    for AT in array_list[2:end]
        for T in (Float32, Float64)
            c0 = [SVector{3, T}(c) for c in coords_many_0]
            c1 = [SVector{3, T}(c) for c in coords_many_1]
            b_T = CubicBoundary(T(5.0))
            cpu = Molly.max_sq_displacements(c1, c0, b_T, nothing, NoUnits, T, 1)
            c0_d, c1_d = to_device(c0, AT), to_device(c1, AT)
            tracker = Molly.DisplacementTracker()
            for _ in 1:2 # The second time reuses the buffers
                @test Molly.max_sq_displacements(c1_d, c0_d, b_T, nothing, NoUnits, T, 1,
                                                 tracker) == cpu
            end
            generic = invoke(Molly.max_sq_displacements,
                             Tuple{AbstractGPUArray, Any, Any, Any, Any, Type{T}, Integer, Any},
                             c1_d, c0_d, b_T, nothing, NoUnits, T, 1, nothing)
            @test generic == cpu
            # A box deformation, and fewer atoms than threads in a group
            F = SVector{3, T}(1.01, 0.99, 1.0)
            b_F = CubicBoundary(T(5.0) .* F)
            @test Molly.max_sq_displacements(c1_d[1:7], c0_d[1:7], b_F, F, NoUnits, T, 1) ==
                  Molly.max_sq_displacements(c1[1:7], c0[1:7], b_F, F, NoUnits, T, 1)
        end
    end
end

@testset "Neighbor finder auto_update" begin
    n_atoms = 200
    boundary = CubicBoundary(4.0u"nm")
    r_c, skin = 0.8u"nm", 0.2u"nm"
    rng = Xoshiro(2)
    coords_start = place_atoms(n_atoms, boundary; min_dist=0.25u"nm", rng=rng)
    atoms = [Atom(mass=10.0u"g/mol", σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:n_atoms]
    lj = LennardJones(cutoff=DistanceCutoff(r_c))
    x_shift(d) = SVector(d, zero(d), zero(d))

    nf_makers = Any[
        (AT, auto) -> DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin,
                                             auto_update=auto, n_steps=10),
        (AT, auto) -> TreeNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin,
                                         auto_update=auto, n_steps=10),
        (AT, auto) -> CellListMapNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin,
                                                boundary=boundary, auto_update=auto,
                                                n_steps=10),
    ]
    test_cases = [(Array, mk) for mk in nf_makers]
    for AT in array_list[2:end]
        push!(test_cases, (AT, (AT, auto) -> DistanceNeighborFinder(n_atoms=n_atoms,
                                dist_cutoff=r_c + skin, auto_update=auto, n_steps=10,
                                array_type=AT)))
        push!(test_cases, (AT, (AT, auto) -> GPUCellListNeighborFinder(n_atoms=n_atoms,
                                dist_cutoff=r_c + skin, auto_update=auto, n_steps=10,
                                array_type=AT, ragged=false)))
    end

    for (AT, mk_nf) in test_cases
        nf = mk_nf(AT, true)
        @test nf.auto_update
        @test Molly.tracks_displacements(nf)
        @test !Molly.tracks_displacements(mk_nf(AT, false))
        @test occursin("auto_update = true", sprint(show, nf))
        sys = System(atoms=to_device(atoms, AT), coords=to_device(copy(coords_start), AT),
                     boundary=boundary, pairwise_inters=(lj,), neighbor_finder=nf)
        @test Molly.neighbor_inters_sqdist(sys) == r_c^2

        nbs = find_neighbors(sys, nf, nothing, 0, true; n_threads=1)
        @test Molly.neighbor_update_counts(nf) == (n_builds=1, n_checks=0)
        @test neighbor_pair_set(nbs) == brute_force_pairs(coords_start, boundary, r_c + skin)
        # The atoms have not moved, so the list is kept whatever the step number
        @test find_neighbors(sys, nf, nbs, 1; n_threads=1) === nbs
        @test find_neighbors(sys, nf, nbs, 10; n_threads=1) === nbs
        @test Molly.neighbor_update_counts(nf) == (n_builds=1, n_checks=2)

        # One atom moving more than half the skin, but less than the skin, is not enough
        coords = copy(coords_start)
        coords[1] = wrap_coords(coords[1] + x_shift(0.6 * skin), boundary)
        sys.coords .= to_device(coords, AT)
        @test find_neighbors(sys, nf, nbs, 2; n_threads=1) === nbs
        # Two atoms moving that far are
        coords[2] = wrap_coords(coords[2] + x_shift(0.6 * skin), boundary)
        sys.coords .= to_device(coords, AT)
        nbs_new = find_neighbors(sys, nf, nbs, 3; n_threads=1)
        @test Molly.neighbor_update_counts(nf).n_builds == 2
        @test neighbor_pair_set(nbs_new) == brute_force_pairs(coords, boundary, r_c + skin)
        nbs = nbs_new

        # A list built from scratch without being asked for, as for analysis, leaves the
        #   tracked list alone, apart from CellListMapNeighborFinder which always updates
        #   the same list
        nbs_fresh = find_neighbors(sys; n_threads=1)
        @test neighbor_pair_set(nbs_fresh) == neighbor_pair_set(nbs)
        n_builds = Molly.neighbor_update_counts(nf).n_builds
        @test find_neighbors(sys, nf, nbs, 4; n_threads=1) === nbs
        @test Molly.neighbor_update_counts(nf).n_builds == n_builds

        # A list that is not the last one built is not reused
        nbs_other = find_neighbors(sys, nf, nothing, 0, true; n_threads=1)
        @test find_neighbors(sys, nf, nbs_other, 4; n_threads=1) === nbs_other
        n_builds = Molly.neighbor_update_counts(nf).n_builds
        nbs = find_neighbors(sys, nf, nbs, 5; n_threads=1)
        # CellListMapNeighborFinder updates and returns the same list each time
        @test Molly.neighbor_update_counts(nf).n_builds ==
                    n_builds + (nf isa CellListMapNeighborFinder ? 0 : 1)
        # Forcing a rebuild rebuilds
        n_builds = Molly.neighbor_update_counts(nf).n_builds
        nbs = find_neighbors(sys, nf, nbs, 6, true; n_threads=1)
        @test Molly.neighbor_update_counts(nf).n_builds == n_builds + 1

        # Scaling the box and coordinates, as a barostat does, keeps the list since the
        #   displacements are measured relative to the deformation, unless the box
        #   shrinks enough to bring pairs from outside the list within the cutoff
        for (scale, kept) in ((1.002, true), (0.999, true), (1.3, true), (0.8, false))
            sys_scaled = System(sys; coords=to_device(from_device(sys.coords) .* scale, AT),
                                boundary=CubicBoundary(boundary.side_lengths .* scale))
            n_builds = Molly.neighbor_update_counts(nf).n_builds
            nbs_scaled = find_neighbors(sys_scaled, nf, nbs, 7; n_threads=1)
            @test (Molly.neighbor_update_counts(nf).n_builds == n_builds) == kept
            needed = brute_force_pairs(from_device(sys_scaled.coords), sys_scaled.boundary, r_c)
            @test issubset(needed, neighbor_pair_set(nbs_scaled))
            # Back to the list for the unscaled coordinates
            nbs = find_neighbors(sys, nf, nbs_scaled, 8, true; n_threads=1)
        end

        # The interactions of a system without a cutoff, or none that use the neighbor list,
        #   give a rebuild every time
        sys_nocut = System(sys; pairwise_inters=(LennardJones(),))
        @test isnothing(Molly.neighbor_inters_sqdist(sys_nocut))
        n_builds = Molly.neighbor_update_counts(nf).n_builds
        nbs = find_neighbors(sys_nocut, nf, nbs, 1; n_threads=1)
        nbs = find_neighbors(sys_nocut, nf, nbs, 2; n_threads=1)
        @test Molly.neighbor_update_counts(nf).n_builds == n_builds + 2
        sys_none = System(sys; pairwise_inters=())
        @test isnothing(Molly.neighbor_inters_sqdist(sys_none))

        # With auto_update=false the list is rebuilt every n_steps steps
        nf_steps = mk_nf(AT, false)
        @test occursin("auto_update = false", sprint(show, nf_steps))
        sys_steps = System(sys; neighbor_finder=nf_steps)
        nbs = find_neighbors(sys_steps, nf_steps, nothing, 0, true; n_threads=1)
        @test find_neighbors(sys_steps, nf_steps, nbs, 3; n_threads=1) === nbs
        nbs_10 = find_neighbors(sys_steps, nf_steps, nbs, 10; n_threads=1)
        @test Molly.neighbor_update_counts(nf_steps) == (n_builds=2, n_checks=0)
        # No reference coordinates are kept
        @test isnothing(nf_steps.tracker.coords)

        # Changing the excluded pairs means that the next call rebuilds
        if nf isa GPUCellListNeighborFinder
            nbs = find_neighbors(sys, nf, nothing, 0, true; n_threads=1)
            Molly.append_excluded_pairs!(nf, [(1, 2)])
            n_builds = Molly.neighbor_update_counts(nf).n_builds
            nbs = find_neighbors(sys, nf, nbs, 1; n_threads=1)
            @test Molly.neighbor_update_counts(nf).n_builds == n_builds + 1
        end

        # Converting the system keeps auto_update
        if AT == Array
            sys_f32 = System(sys; float_type=Float32)
            @test sys_f32.neighbor_finder.auto_update
            @test !System(sys_steps; float_type=Float32).neighbor_finder.auto_update
            @test System(sys_steps; float_type=Float32).neighbor_finder.n_steps == 10
        end
    end

    # The neighbor finder of a system set up from a file
    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...)
    sys_file = System(joinpath(data_dir, "6mrr_equil.pdb"), ff)
    @test sys_file.neighbor_finder.auto_update
    @test Molly.neighbor_inters_sqdist(sys_file) == (1.0u"nm")^2
    sys_file_steps = System(joinpath(data_dir, "6mrr_equil.pdb"), ff;
                            neighbor_finder_auto_update=false, neighbor_finder_n_steps=5)
    @test !sys_file_steps.neighbor_finder.auto_update
    @test sys_file_steps.neighbor_finder.n_steps == 5
    for nf_type in (DistanceNeighborFinder, TreeNeighborFinder)
        sys_type = System(joinpath(data_dir, "6mrr_equil.pdb"), ff; neighbor_finder_type=nf_type,
                          neighbor_finder_auto_update=false, strictness=:nowarn)
        @test sys_type.neighbor_finder isa nf_type
        @test !sys_type.neighbor_finder.auto_update
    end
    # Setting up constraints excludes the constrained pairs from a neighbor finder that has
    #   already built a list, which then can not be reused
    for nf_cons in (DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin),
                    CellListMapNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin,
                                              boundary=boundary))
        sys_uncons = System(atoms=atoms, coords=coords_start, boundary=boundary,
                            pairwise_inters=(lj,), neighbor_finder=nf_cons)
        coords_cons = copy(coords_start)
        coords_cons[2] = coords_cons[1] + SVector(0.15, 0.0, 0.0)u"nm"
        sys_uncons.coords .= coords_cons
        nbs_uncons = find_neighbors(sys_uncons, nf_cons, nothing, 0, true)
        @test (1, 2) in neighbor_pair_set(nbs_uncons)
        cons = SHAKE_RATTLE(n_atoms=n_atoms,
                            dist_constraints=[DistanceConstraint(1, 2, 0.15u"nm")])
        sys_cons = System(atoms=atoms, coords=coords_cons, boundary=boundary,
                          pairwise_inters=(lj,), neighbor_finder=nf_cons, constraints=(cons,))
        nbs_cons = find_neighbors(sys_cons, nf_cons, nbs_uncons, 1)
        @test !((1, 2) in neighbor_pair_set(nbs_cons))
    end
end

@testset "Neighbor list rebuild modes" begin
    # A light, hot Lennard-Jones fluid with a small buffer, so the atoms cross the buffer
    #   within a few steps
    r_c, skin = 0.8u"nm", 0.05u"nm"
    temp, dt = 300.0u"K", 0.002u"ps"
    n_atoms = 200
    boundary = CubicBoundary(3.0u"nm")
    mass = 4.0u"g/mol"

    function fluid_system(nf; checker=ListCompletenessCheck(r_c))
        rng = Xoshiro(10)
        coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng=rng)
        return System(
            atoms=[Atom(mass=mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:n_atoms],
            coords=coords,
            boundary=boundary,
            velocities=[random_velocity(mass, temp; rng=rng) for _ in 1:n_atoms],
            pairwise_inters=(LennardJones(cutoff=DistanceCutoff(r_c)),),
            general_inters=(checker,),
            neighbor_finder=nf,
        )
    end
    sim = VelocityVerlet(dt=dt)
    n_steps_sim = 100

    # Automatic updates: the list always has every pair within the cutoff, and is rebuilt
    #   far less often than every step
    nf_auto = DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin)
    sys_auto = fluid_system(nf_auto)
    simulate!(sys_auto, sim, n_steps_sim; n_threads=1)
    checker = sys_auto.general_inters[1]
    @test checker.n_checks == n_steps_sim + 1
    @test checker.n_incomplete == 0
    counts = Molly.neighbor_update_counts(nf_auto)
    @test 2 < counts.n_builds < n_steps_sim ÷ 2
    @test counts.n_checks == n_steps_sim

    # Updates every 10 steps miss pairs with this buffer
    nf_10 = DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin,
                                   auto_update=false, n_steps=10)
    sys_10 = fluid_system(nf_10)
    simulate!(sys_10, sim, n_steps_sim; n_threads=1)
    @test sys_10.general_inters[1].n_incomplete > 0
    @test Molly.neighbor_update_counts(nf_10) == (n_builds=(n_steps_sim ÷ 10 + 1), n_checks=0)

    # Updates every step without tracking the displacements
    nf_1 = DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin,
                                  auto_update=false, n_steps=1)
    sys_1 = fluid_system(nf_1)
    simulate!(sys_1, sim, n_steps_sim; n_threads=1)
    @test sys_1.general_inters[1].n_incomplete == 0
    @test Molly.neighbor_update_counts(nf_1) == (n_builds=(n_steps_sim + 1), n_checks=0)
    @test isnothing(nf_1.tracker.coords)
    # Automatic updates and updates every step give the same trajectory
    @test sys_auto.coords == sys_1.coords
    @test sys_auto.velocities == sys_1.velocities

    # Without a buffer the list is rebuilt every step and no displacements are tracked
    for nf_noskin in (DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c),
                      CellListMapNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c,
                                                boundary=boundary))
        sys_noskin = fluid_system(nf_noskin)
        simulate!(sys_noskin, sim, 20; n_threads=1)
        @test sys_noskin.general_inters[1].n_incomplete == 0
        @test Molly.neighbor_update_counts(nf_noskin) == (n_builds=21, n_checks=0)
        @test isnothing(nf_noskin.tracker.coords)
        @test isnothing(nf_noskin.tracker.neighbors)
    end
end

@testset "Every force calculation sees a complete list" begin
    # The simulators check the list at every set of coordinates that the interactions using
    #   it are calculated at, including the trial moves of the Monte Carlo barostat and
    #   of Monte Carlo simulation, the minimizers and the inner steps of the MTS integrators
    r_c, skin = 0.8u"nm", 0.05u"nm"
    temp, press, dt = 300.0u"K", 1.0u"bar", 0.002u"ps"
    n_atoms = 200
    boundary = CubicBoundary(3.0u"nm")
    mass = 4.0u"g/mol"

    function checked_system(; two_inters=false)
        rng = Xoshiro(11)
        coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng=rng)
        lj = LennardJones(cutoff=DistanceCutoff(r_c))
        inters = (two_inters ? (lj, SoftSphere(cutoff=DistanceCutoff(0.5u"nm"))) : (lj,))
        return System(
            atoms=[Atom(mass=mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:n_atoms],
            coords=coords,
            boundary=boundary,
            velocities=[random_velocity(mass, temp; rng=rng) for _ in 1:n_atoms],
            pairwise_inters=inters,
            general_inters=(ListCompletenessCheck(r_c),),
            neighbor_finder=DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin),
        )
    end

    min_tol = 1e-6u"kJ * mol^-1 * nm^-1"
    sims = [
        VelocityVerlet(dt=dt, coupling=AndersenThermostat(temp, 0.5u"ps")),
        VelocityVerlet(dt=dt, coupling=CRescaleBarostat(press, 0.1u"ps";
                                        compressibility=4.6e-3u"bar^-1")),
        Verlet(dt=dt),
        StormerVerlet(dt=dt),
        Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1"),
        # Large volume moves, which are checked rather than assuming the list is valid
        Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1",
                 coupling=MonteCarloBarostat(press, temp, boundary; n_steps=2,
                                             scale_factor=0.05)),
        LangevinSplitting(dt=dt, temperature=temp, friction=4.0u"g * mol^-1 * ps^-1",
                          splitting="BAOAB"),
        NoseHoover(dt=dt, temperature=temp),
        # The check is at the inner level, so the list is checked at the inner coordinates
        MTSIntegrator(dt=2dt, pi_fractions=(1, 2), gi_fractions=(2,)),
        MTSLangevinIntegrator(dt=2dt, temperature=temp, friction=1.0u"ps^-1",
                              pi_fractions=(2, 1), gi_fractions=(2,)),
        MetropolisMonteCarlo(temperature=temp, trial_moves=Molly.random_uniform_translation!,
                             trial_args=Dict(:shift_size => 0.1u"nm")),
        SteepestDescentMinimizer(max_steps=50, tol=min_tol),
        FIREMinimizer(max_steps=50, tol=min_tol),
        LBFGSMinimizer(max_steps=50, tol=min_tol),
    ]
    for sim in sims
        sys = checked_system(; two_inters=(sim isa Molly.AbstractMTSIntegrator))
        if sim isa Union{SteepestDescentMinimizer, FIREMinimizer, LBFGSMinimizer}
            simulate!(sys, sim; n_threads=1)
        else
            simulate!(sys, sim, 100; n_threads=1, rng=Xoshiro(3))
        end
        checker = sys.general_inters[1]
        @test checker.n_checks > 0
        @test checker.n_incomplete == 0
        @test Molly.neighbor_update_counts(sys.neighbor_finder).n_builds > 1
    end
end

@testset "Simulations with auto_update" begin
    # With DistanceNeighborFinder on one thread, pairs beyond the cutoff contribute exactly
    #   zero and the pairs are in a fixed order, so a list that is valid at every force
    #   evaluation gives the same trajectory as rebuilding the list before every force
    #   evaluation
    r_c, skin = 0.8u"nm", 0.05u"nm"
    temp, press, dt = 300.0u"K", 1.0u"bar", 0.002u"ps"
    function lj_system(auto_update; two_inters=false)
        rng = Xoshiro(10)
        n_atoms = 300
        boundary = CubicBoundary(3.2u"nm")
        mass = 4.0u"g/mol"
        coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm", rng=rng)
        lj = LennardJones(cutoff=DistanceCutoff(r_c))
        inters = (two_inters ? (lj, SoftSphere(cutoff=DistanceCutoff(0.5u"nm"))) : (lj,))
        return System(
            atoms=[Atom(mass=mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for _ in 1:n_atoms],
            coords=coords,
            boundary=boundary,
            velocities=[random_velocity(mass, temp; rng=rng) for _ in 1:n_atoms],
            pairwise_inters=inters,
            neighbor_finder=DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=r_c + skin,
                                                   auto_update=auto_update, n_steps=1),
        )
    end
    barostat() = CRescaleBarostat(press, 0.1u"ps"; compressibility=4.6e-3u"bar^-1")
    simulators = [
        VelocityVerlet(dt=dt, coupling=AndersenThermostat(temp, 0.5u"ps")),
        VelocityVerlet(dt=dt, coupling=barostat()),
        Verlet(dt=dt),
        Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1"),
        Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1", coupling=barostat()),
        Langevin(dt=dt, temperature=temp, friction=1.0u"ps^-1",
                 coupling=MonteCarloBarostat(press, temp, CubicBoundary(3.2u"nm"); n_steps=5,
                                             trial_find_neighbors=true)),
        LangevinSplitting(dt=dt, temperature=temp, friction=4.0u"g * mol^-1 * ps^-1",
                          splitting="BAOAB"),
        LangevinSplitting(dt=dt, temperature=temp, friction=4.0u"g * mol^-1 * ps^-1",
                          splitting="ABOBA"),
        NoseHoover(dt=dt, temperature=temp),
        MTSIntegrator(dt=2dt, pi_fractions=(1, 2)),
    ]
    for sim in simulators
        mts = sim isa MTSIntegrator
        sys_auto = lj_system(true; two_inters=mts)
        sys_ref = lj_system(false; two_inters=mts)
        # The reference rebuilds at every MTS inner step
        sim_ref = (mts ? MTSIntegrator(dt=2dt, pi_fractions=(1, 2), inner_step_neighbors=true) :
                         sim)
        simulate!(sys_auto, sim, 150; n_threads=1, rng=Xoshiro(3))
        simulate!(sys_ref, sim_ref, 150; n_threads=1, rng=Xoshiro(3))
        @test sys_auto.coords == sys_ref.coords
        @test sys_auto.velocities == sys_ref.velocities
        @test sys_auto.boundary == sys_ref.boundary
        # Far fewer rebuilds than steps, apart from the Monte Carlo barostat which is asked
        #   to rebuild the list for every trial move
        trial_rebuilds = (hasproperty(sim, :coupling) && sim.coupling isa MonteCarloBarostat)
        @test Molly.neighbor_update_counts(sys_auto.neighbor_finder).n_builds <
                    (trial_rebuilds ? 150 : 75)
    end

    # Metropolis Monte Carlo reuses the list when the moves do not invalidate it
    sys_mc = lj_system(true)
    simulate!(sys_mc, MetropolisMonteCarlo(temperature=temp,
                    trial_moves=Molly.random_uniform_translation!,
                    trial_args=Dict(:shift_size => 0.02u"nm")), 200; rng=Xoshiro(4))
    @test Molly.neighbor_update_counts(sys_mc.neighbor_finder).n_builds < 100
    pe_list = potential_energy(sys_mc, find_neighbors(sys_mc, sys_mc.neighbor_finder,
                                                      nothing, 0, true))
    sys_mc_ref = System(sys_mc; neighbor_finder=nothing,
                        pairwise_inters=(LennardJones(cutoff=DistanceCutoff(r_c),
                                                      use_neighbors=false),))
    @test pe_list ≈ potential_energy(sys_mc_ref)
end

@testset "GPU neighbor finders with auto_update" begin
    # On the GPU the summation order differs between lists, so the forces at every step
    #   are compared to those from all pairs
    for AT in array_list[2:end]
        n_atoms = 1_000
        boundary = CubicBoundary(5.0f0u"nm")
        rng = Xoshiro(5)
        coords = place_atoms(n_atoms, boundary; min_dist=0.3f0u"nm", rng=rng)
        atoms = [Atom(mass=4.0f0u"g/mol", σ=0.3f0u"nm", ϵ=0.2f0u"kJ * mol^-1")
                 for _ in 1:n_atoms]
        velocities = [random_velocity(4.0f0u"g/mol", 300.0f0u"K"; rng=rng) for _ in 1:n_atoms]
        lj_gpu = LennardJones(cutoff=DistanceCutoff(0.8f0u"nm"))
        nfs = Any[GPUCellListNeighborFinder(n_atoms=n_atoms, dist_cutoff=0.85f0u"nm",
                                            array_type=AT, ragged=false),
                  DistanceNeighborFinder(n_atoms=n_atoms, dist_cutoff=0.85f0u"nm",
                                         array_type=AT)]
        if Molly.uses_gpu_neighbor_finder(AT)
            push!(nfs, GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=0.85f0u"nm",
                                         array_type=AT))
        end
        barostat = MonteCarloBarostat(1.0f0u"bar", 300.0f0u"K", boundary; n_steps=3,
                                      scale_factor=0.02)
        for nf in nfs, coupling in (nothing, barostat)
            nf_used = deepcopy(nf)
            coupling_used = deepcopy(coupling)
            sys = System(atoms=to_device(atoms, AT), coords=to_device(coords, AT),
                         boundary=boundary, velocities=to_device(velocities, AT),
                         pairwise_inters=(lj_gpu,), neighbor_finder=nf_used)
            sys_all = System(sys; neighbor_finder=nothing,
                             pairwise_inters=(LennardJones(cutoff=DistanceCutoff(0.8f0u"nm"),
                                                           use_neighbors=false),))
            max_force_diff = Ref(0.0)
            function check_forces(s, neighbors, step_n, buffers; kwargs...)
                fs = Molly.zero_forces(s)
                Molly.forces!(fs, s, neighbors, step_n, buffers, Val(false))
                sys_all.coords .= s.coords
                sys_all.boundary = s.boundary
                fs_all = forces(sys_all)
                diff = maximum(norm.(from_device(fs) .- from_device(fs_all)))
                max_force_diff[] = max(max_force_diff[], ustrip(diff))
                return diff
            end
            sys = System(sys; loggers=(forces_check=GeneralObservableLogger(check_forces,
                                       typeof(1.0f0u"kJ * mol^-1 * nm^-1"), 1),))
            sim = Langevin(dt=0.002f0u"ps", temperature=300.0f0u"K", friction=1.0f0u"ps^-1",
                           coupling=coupling_used)
            simulate!(sys, sim, 150)
            n_builds = Molly.neighbor_update_counts(sys.neighbor_finder).n_builds
            @test 1 < n_builds < 75
            @test max_force_diff[] < 0.05
            if !isnothing(coupling)
                @test sys.boundary != boundary
            end
        end

        if Molly.uses_gpu_neighbor_finder(AT)
            # The tile list is checked by the force and energy kernels, against the
            #   coordinates it was built from in the buffers that hold it
            nf = GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=0.9f0u"nm", array_type=AT)
            sys = System(atoms=to_device(atoms, AT), coords=to_device(coords, AT),
                         boundary=boundary, pairwise_inters=(lj_gpu,), neighbor_finder=nf)
            buffers = Molly.init_buffers!(sys, 1)
            fs = Molly.zero_forces(sys)
            Molly.forces!(fs, sys, nothing, 0, buffers, Val(false))
            @test Molly.neighbor_update_counts(nf) == (n_builds=1, n_checks=0)
            Molly.forces!(fs, sys, nothing, 1, buffers, Val(false))
            potential_energy(sys, nothing, 1, buffers)
            @test Molly.neighbor_update_counts(nf) == (n_builds=1, n_checks=2)
            # One atom moving 0.6 of the skin is not a rebuild, two atoms are
            coords_moved = copy(coords)
            coords_moved[1] = wrap_coords(coords_moved[1] + SVector(0.06f0, 0.0f0, 0.0f0)u"nm",
                                          boundary)
            sys.coords .= to_device(coords_moved, AT)
            Molly.forces!(fs, sys, nothing, 2, buffers, Val(false))
            @test Molly.neighbor_update_counts(nf).n_builds == 1
            coords_moved[2] = wrap_coords(coords_moved[2] + SVector(0.06f0, 0.0f0, 0.0f0)u"nm",
                                          boundary)
            sys.coords .= to_device(coords_moved, AT)
            Molly.forces!(fs, sys, nothing, 3, buffers, Val(false))
            @test Molly.neighbor_update_counts(nf).n_builds == 2
            # Other buffers have their own reference and do not disturb these
            forces(sys)
            @test Molly.neighbor_update_counts(nf).n_builds == 3
            Molly.forces!(fs, sys, nothing, 4, buffers, Val(false))
            @test Molly.neighbor_update_counts(nf).n_builds == 3
            # A forced rebuild refreshes the tiles
            find_neighbors(sys, nf, nothing, 5, true)
            Molly.forces!(fs, sys, nothing, 5, buffers, Val(false))
            @test Molly.neighbor_update_counts(nf).n_builds == 4
            # A check that fails when it is not expected to, as the first after a rebuild,
            #   gives the forces and energy from the rebuilt list, since the kernel launched
            #   before the result is known does nothing and is launched again
            sys_ref = System(sys; neighbor_finder=nothing,
                             pairwise_inters=(LennardJones(cutoff=DistanceCutoff(0.8f0u"nm"),
                                                           use_neighbors=false),))
            function move_atoms!(is, step_n)
                for i in is
                    coords_moved[i] = wrap_coords(
                            coords_moved[i] + SVector(0.0f0, 0.06f0, 0.0f0)u"nm", boundary)
                end
                sys.coords .= to_device(coords_moved, AT)
                sys_ref.coords .= sys.coords
                return Molly.neighbor_update_counts(nf).n_builds
            end
            max_diff(fs_1, fs_2) = maximum(norm.(from_device(fs_1) .- from_device(fs_2)))
            n_builds_before = move_atoms!(3:4, 6)
            Molly.forces!(fs, sys, nothing, 6, buffers, Val(false))
            @test Molly.neighbor_update_counts(nf).n_builds == n_builds_before + 1
            @test max_diff(fs, forces(sys_ref)) < 0.05u"kJ * mol^-1 * nm^-1"
            n_builds_before = move_atoms!(5:6, 7)
            @test potential_energy(sys, nothing, 7, buffers) ≈ potential_energy(sys_ref) rtol=1e-4
            @test Molly.neighbor_update_counts(nf).n_builds == n_builds_before + 1
            # A check that is left pending is finished by the next force calculation
            n_builds_before = move_atoms!(7:8, 8)
            Molly.start_pairwise_forces_gpu!(buffers, sys, (lj_gpu,), nothing, Val(false), 8)
            Molly.forces!(fs, sys, nothing, 8, buffers, Val(false))
            @test Molly.neighbor_update_counts(nf).n_builds == n_builds_before + 1
            @test max_diff(fs, forces(sys_ref)) < 0.05u"kJ * mol^-1 * nm^-1"
            # Without a buffer the tiles are refreshed every time without tracking
            nf_noskin = GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=0.8f0u"nm",
                                          array_type=AT)
            sys_noskin = System(sys; neighbor_finder=nf_noskin)
            buffers_noskin = Molly.init_buffers!(sys_noskin, 1)
            for step_n in 0:2
                Molly.forces!(fs, sys_noskin, nothing, step_n, buffers_noskin, Val(false))
            end
            @test Molly.neighbor_update_counts(nf_noskin) == (n_builds=3, n_checks=0)
            # With auto_update=false the tiles follow the schedule of find_neighbors
            nf_steps = GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=0.9f0u"nm",
                                         array_type=AT, auto_update=false, n_steps=1)
            sys_steps = System(sys; neighbor_finder=nf_steps)
            buffers_steps = Molly.init_buffers!(sys_steps, 1)
            for step_n in 0:2
                find_neighbors(sys_steps, nf_steps, nothing, step_n)
                Molly.forces!(fs, sys_steps, nothing, step_n, buffers_steps, Val(false))
            end
            @test Molly.neighbor_update_counts(nf_steps) == (n_builds=3, n_checks=0)
            @test !occursin("unused", sprint(show, nf_steps))
            @test occursin("unused", sprint(show, nf))
        end
    end
end

@testset "ReaxFF neighbor finder updates" begin
    reaxff_dir = joinpath(data_dir, "ReaxFF")
    for AT in array_list
        reaxff = ReaxFF(
            force_field_file=joinpath(reaxff_dir, "force_fields", "Chenoweth_2008_CHO.ff"),
            float_type=Float64,
            array_type=AT,
        )
        boundary = CubicBoundary(3.5u"nm")
        # Without a skin the list is rebuilt every step and nothing is tracked
        sys = System(joinpath(reaxff_dir, "data", "methane_oxidation.pdb"), reaxff;
                     boundary=boundary, array_type=AT)
        nf = sys.neighbor_finder
        @test !nf.auto_update
        @test nf.n_steps == 1
        simulate!(sys, Langevin(dt=0.25u"fs", temperature=300.0u"K", friction=1.0u"ps^-1"),
                  5; run_loggers=false)
        @test Molly.neighbor_update_counts(nf) == (n_builds=6, n_checks=0)
        @test isnothing(nf.tracker.coords)

        # With a skin the list is rebuilt when needed, given the ReaxFF cutoffs
        reaxff_skin = ReaxFF(
            force_field_file=joinpath(reaxff_dir, "force_fields", "Chenoweth_2008_CHO.ff"),
            float_type=Float64,
            array_type=AT,
        )
        sys_skin = System(joinpath(reaxff_dir, "data", "methane_oxidation.pdb"), reaxff_skin;
                          boundary=boundary, array_type=AT, neighbor_skin=0.1u"nm")
        nf_skin = sys_skin.neighbor_finder
        @test nf_skin.auto_update
        required = Molly.neighbor_cutoff(reaxff_skin)
        @test uconvert(u"nm", nf_skin.dist_cutoff - 0.1u"nm") ≈ uconvert(u"nm", required)
        @test sqrt(Molly.neighbor_inters_sqdist(sys_skin)) ≈ uconvert(u"nm", required)
        sys_skin_ref = System(joinpath(reaxff_dir, "data", "methane_oxidation.pdb"),
                              ReaxFF(force_field_file=joinpath(reaxff_dir, "force_fields",
                                                    "Chenoweth_2008_CHO.ff"),
                                     float_type=Float64, array_type=AT);
                              boundary=boundary, array_type=AT)
        sim = Langevin(dt=0.25u"fs", temperature=300.0u"K", friction=1.0u"ps^-1")
        random_velocities!(sys_skin, 300.0u"K"; rng=Xoshiro(1))
        sys_skin_ref.velocities .= sys_skin.velocities
        simulate!(sys_skin, sim, 20; run_loggers=false, rng=Xoshiro(2))
        simulate!(sys_skin_ref, sim, 20; run_loggers=false, rng=Xoshiro(2))
        @test Molly.neighbor_update_counts(nf_skin).n_builds < 21
        @test Molly.neighbor_update_counts(nf_skin).n_checks == 20
        # The same dynamics as rebuilding every step without a skin
        @test maximum(norm.(from_device(sys_skin.coords) .- from_device(sys_skin_ref.coords))) <
                    1e-6u"nm"
    end
end
