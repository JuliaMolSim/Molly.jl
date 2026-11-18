@testset "Analysis" begin
    # Displacements and distances
    coords = [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0)]
    boundary = CubicBoundary(10.0)
    disps = displacements(coords, boundary)
    @test disps[1, 2] == SVector(1.0, 1.0, 1.0)
    @test disps[2, 1] == SVector(-1.0, -1.0, -1.0)
    dists = distances(coords, boundary)
    @test dists[1, 2] ≈ sqrt(3.0)
    @test dists[2, 1] ≈ sqrt(3.0)

    boundary_triclinic = TriclinicBoundary(
        SVector(10.0, 0.0, 0.0),
        SVector(5.0, 10.0, 0.0),
        SVector(5.0, 5.0, 10.0),
    )
    disps_triclinic = displacements(coords, boundary_triclinic)
    @test disps_triclinic[1, 2] == SVector(1.0, 1.0, 1.0)
    dists_triclinic = distances(coords, boundary_triclinic)
    @test dists_triclinic[1, 2] ≈ sqrt(3.0)

    # RMSD
    coords1 = [SVector(1.0, 1.0, 1.0), SVector(2.0, 2.0, 2.0)]
    coords2 = [SVector(2.0, 1.0, 1.0), SVector(3.0, 2.0, 2.0)]
    @test rmsd(coords1, coords2) ≈ 0.0 atol=1e-12

    coords3 = [SVector(10.0, 10.0, 10.0), SVector(11.0, 11.0, 11.0)]
    coords4 = [SVector(11.0, 10.0, 10.0), SVector(12.0, 11.0, 11.0)]
    @test rmsd(coords3, coords4) ≈ 0.0 atol=1e-12

    coords5 = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
    rot = @SMatrix [0.0 -1.0  0.0;
                    1.0  0.0  0.0;
                    0.0  0.0  1.0]
    coords6 = [rot * c for c in coords5]
    @test rmsd(coords5, coords6) ≈ 0.0 atol=1e-12

    pdb_path = joinpath(data_dir, "1ssu.pdb")
    struc = read(pdb_path, BioStructures.PDBFormat)
    cm_1 = BioStructures.coordarray(struc[1], BioStructures.calphaselector)
    cm_2 = BioStructures.coordarray(struc[2], BioStructures.calphaselector)
    coords_1 = SVector{3, Float64}.(eachcol(cm_1)) / 10 * u"nm"
    coords_2 = SVector{3, Float64}.(eachcol(cm_2)) / 10 * u"nm"
    @test rmsd(coords_1, coords_2) ≈ 2.54859467758795u"Å"
    for AT in array_list[2:end]
        @test rmsd(to_device(coords_1, AT), to_device(coords_2, AT)) ≈ 2.54859467758795u"Å"
    end

    # Radius of gyration and hydrodynamic radius
    atoms = [Atom(mass=1.0), Atom(mass=1.0)]
    coords = [SVector(1.0, 1.0, 1.0), SVector(3.0, 3.0, 3.0)]
    @test radius_gyration(coords, atoms) ≈ sqrt(3.0)

    atoms2 = [Atom(mass=1.0), Atom(mass=2.0), Atom(mass=3.0)]
    coords2 = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
    center_of_mass = (1.0 * coords2[1] + 2.0 * coords2[2] + 3.0 * coords2[3]) / 6.0
    I = 1.0 * sum(abs2, coords2[1] - center_of_mass) +
        2.0 * sum(abs2, coords2[2] - center_of_mass) +
        3.0 * sum(abs2, coords2[3] - center_of_mass)
    rg_sq = I / 6.0
    @test radius_gyration(coords2, atoms2) ≈ sqrt(rg_sq)

    coords = [SVector(0.0, 0.0, 0.0), SVector(3.0, 4.0, 0.0)]
    boundary = CubicBoundary(10.0)
    @test hydrodynamic_radius(coords, boundary) ≈ 20.0 atol=1e-12

    coords2 = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
    d12 = 1.0
    d13 = 1.0
    d23 = sqrt(2.0)
    inv_rh_formula = (1/d12 + 1/d13 + 1/d23) / 9
    @test hydrodynamic_radius(coords2, boundary) ≈ inv(inv_rh_formula) atol=1e-12

    # Radial distribution function
    # The largest peak for a face-centred cubic lattice is the nearest neighbor
    #   shell at a / √2
    a_fcc = 0.52468u"nm"
    fcc_crystal = SimpleCrystals.FCC(a_fcc, 39.948u"g/mol", SVector(4, 4, 4))
    sys_fcc = System(fcc_crystal; force_units=u"kJ * mol^-1 * nm^-1",
                     energy_units=u"kJ * mol^-1")
    bins_fcc, gr_fcc = rdf(sys_fcc.coords, sys_fcc.boundary)
    @test length(bins_fcc) == length(gr_fcc) == 200
    @test bins_fcc[argmax(gr_fcc)] ≈ a_fcc / sqrt(2) atol=0.02u"nm"
    # There are no pairs closer than the nearest neighbor distance
    @test first(gr_fcc) < 0.01 * maximum(gr_fcc)

    # Uniformly distributed coordinates give g(r) = 1 away from the boundary, and
    #   this should not depend on the number of atoms or on npoints
    for n_atoms_rdf in (500, 2_000)
        for npoints_rdf in (100, 400)
            rng_rdf = Xoshiro(4)
            boundary_rdf = CubicBoundary(6.0u"nm")
            coords_rdf = [SVector{3}(rand(rng_rdf, 3) .* 6.0u"nm") for _ in 1:n_atoms_rdf]
            bins_gas, gr_gas = rdf(coords_rdf, boundary_rdf; npoints=npoints_rdf)
            sel_gas = findall(r -> 1.0u"nm" < r < 2.5u"nm", bins_gas)
            @test all(g -> isapprox(g, 1.0; atol=0.05), gr_gas[sel_gas])
        end
    end

    # The same holds in 2D, where the shell is a circumference rather than a sphere
    rng_rdf_2D = Xoshiro(7)
    boundary_rdf_2D = RectangularBoundary(6.0u"nm")
    coords_rdf_2D = [SVector{2}(rand(rng_rdf_2D, 2) .* 6.0u"nm") for _ in 1:1_000]
    bins_2D, gr_2D = rdf(coords_rdf_2D, boundary_rdf_2D; npoints=100)
    sel_2D = findall(r -> 1.0u"nm" < r < 2.5u"nm", bins_2D)
    @test all(g -> isapprox(g, 1.0; atol=0.05), gr_2D[sel_2D])

    bb_atoms = BioStructures.collectatoms(struc[1], BioStructures.backboneselector)
    coords = SVector{3, Float64}.(eachcol(BioStructures.coordarray(bb_atoms))) / 10 * u"nm"
    bb_to_mass = Dict("C" => 12.011u"g/mol", "N" => 14.007u"g/mol", "O" => 15.999u"g/mol")
    atoms = [Atom(mass=bb_to_mass[BioStructures.element(bb_atoms[i])]) for i in eachindex(bb_atoms)]
    @test isapprox(radius_gyration(coords, atoms), 11.51225678195222u"Å"; atol=1e-6u"nm")
    boundary = CubicBoundary(10.0u"nm")
    coords_wrap = wrap_coords.(coords, (boundary,))
    @test isapprox(hydrodynamic_radius(coords_wrap, boundary), 21.00006825680275u"Å"; atol=1e-6u"nm")
end

@testset "MBAR" begin
    # Umbrella windows on a harmonic potential sampled exactly, which gives analytic
    #   free energies and PMF (reduced units, β = 1)
    rng = Xoshiro(10)
    κ0, κ, μs, n = 1.0, 5.0, range(-2.0, 2.0; length=9), 2_000
    x = [κ * μ / (κ0 + κ) + randn(rng) / sqrt(κ0 + κ) for μ in μs for _ in 1:n]
    win_of, N_k = repeat(eachindex(μs); inner=n), fill(n, length(μs))
    u = [0.5 * κ0 * xi^2 + 0.5 * κ * (xi - μ)^2 for xi in x, μ in μs]
    f, logN = Molly.iterate_mbar(u, win_of, N_k)
    # The tolerance is about 4 standard errors of the last window
    @test f ≈ @.(0.5 * κ0 * κ / (κ0 + κ) * (μs^2 - μs[1]^2)) atol=0.25

    u_target = 0.5 .* κ0 .* x .^ 2
    shifts = vec(minimum(u; dims=2))
    _, w_target, _ = Molly.mbar_weights(u .- shifts, u_target, f, N_k, logN; shifts=shifts)
    @test sum(w_target .* x .^ 2) ≈ 1 / κ0 rtol=0.05 # Equipartition in the target state

    R_k = [x[win_of .== k] for k in eachindex(μs)]
    pmf_res = Molly.pmf_with_uncertainty(u, u_target, f, N_k, logN, R_k; edges=-1.5:0.25:1.5)
    F_diff = pmf_res.F .- 0.5 .* κ0 .* pmf_res.centers .^ 2
    @test all(abs.(F_diff .- mean(F_diff)) .< 4 .* pmf_res.sigma_F)
    @test Molly.mbar_pmf(u, u_target, f, N_k, logN, x, (-1.5, 1.5, 12)).F ≈ pmf_res.F

    # With unequal sample counts and poor overlap the uncertainty of each bin still matches
    #   the spread over independent data sets
    μs_poor, N_poor = range(-2.0, 2.0; length=5), [200, 3000, 500, 100, 2000]
    neg_log_p, σ_poor = Vector{Float64}[], Vector{Float64}[]
    for _ in 1:60
        x_p = [20 * μ / 21 + randn(rng) / sqrt(21) for (μ, n) in zip(μs_poor, N_poor) for _ in 1:n]
        win_p = reduce(vcat, [fill(k, n) for (k, n) in enumerate(N_poor)])
        u_p = [0.5 * xi^2 + 10 * (xi - μ)^2 for xi in x_p, μ in μs_poor]
        f_p, logN_p = Molly.iterate_mbar(u_p, win_p, N_poor)
        res = Molly.pmf_with_uncertainty(u_p, 0.5 .* x_p .^ 2, f_p, N_poor, logN_p,
                                         [x_p[win_p .== k] for k in 1:5]; edges=-1.5:0.5:1.5)
        push!(neg_log_p, -log.(res.p))
        push!(σ_poor, res.sigma_F)
    end
    @test all(isapprox.(vec(std(reduce(hcat, neg_log_p); dims=2)),
                        vec(mean(reduce(hcat, σ_poor); dims=2)); rtol=0.3))
end

@testset "MBAR solvers" begin
    # Harmonic oscillators in d dimensions at inverse temperatures βs, sampled exactly through
    #   their energy U = |x|² / 2, which gives f_k - f_1 = (d / 2) log(β_k / β_1)
    function ladder_samples(rng, βs, d, N_k)
        U = Float64[]
        for (β, n) in zip(βs, N_k), _ in 1:n
            push!(U, sum(abs2, randn(rng, d)) / (2β))
        end
        win_of = reduce(vcat, [fill(k, n) for (k, n) in enumerate(N_k)])
        return U .* βs', win_of, (d / 2) .* log.(βs ./ βs[1])
    end
    # Largest deviation of ∑_n W[n,k] from 1 over the sampled states
    function normalization_residual(u, f, N_k)
        logD = Molly.mbar_logD(u, f, log.(Float64.(N_k)))
        return maximum(k -> abs(sum(exp.(f[k] .- u[:, k] .- logD)) - 1), findall(>(0), N_k))
    end
    rng = Xoshiro(20)

    # Well-overlapping states, where both solvers give the same free energies
    βs = exp.(0.1 .* (0:7))
    N_k = fill(500, 8)
    u, win_of, f_exact = ladder_samples(rng, βs, 20, N_k)
    f_fp, logN = Molly.iterate_mbar(u, win_of, N_k; rtol=1e-13, max_iter=100_000)
    f_newton, logN_newton = Molly.iterate_mbar(u, win_of, N_k; method=:newton)
    @test f_newton ≈ f_fp atol=1e-9
    @test f_newton[1] == 0
    @test logN_newton == logN
    @test f_newton ≈ f_exact atol=0.25
    @test normalization_residual(u, f_newton, N_k) ≤ 1e-10
    @test Molly.MBARResult(u, win_of, N_k, ones(8); method=:newton).f == f_newton
    @test_throws ArgumentError Molly.iterate_mbar(u, win_of, N_k; method=:bisection)
    @test_throws DimensionMismatch Molly.iterate_mbar(u, win_of, N_k; f_init=zeros(3))

    # A long ladder of weakly overlapping states, where the weights are normalized to atol
    βs = exp.(0.06 .* (0:29))
    N_k = fill(100, 30)
    logN = log.(Float64.(N_k))
    u, win_of, f_exact = ladder_samples(rng, βs, 1000, N_k)
    f_newton, _ = Molly.iterate_mbar(u, win_of, N_k; method=:newton)
    @test normalization_residual(u, f_newton, N_k) ≤ 1e-10
    @test f_newton ≈ f_exact atol=12
    f_loose, _ = Molly.iterate_mbar(u, win_of, N_k; method=:newton, atol=1e-3)
    @test 1e-10 < normalization_residual(u, f_loose, N_k) ≤ 1e-3

    # Warm starts: the free energies of all frames are a good start for a subset of them, as
    #   in a jackknife, and the fixed point started at the solution stays there
    _, cold = Molly.mbar_newton(u, Molly.mbar_init(u, win_of), N_k, logN)
    keep = [n for n in eachindex(win_of) if n % 10 != 0]
    N_keep = [count(==(k), win_of[keep]) for k in 1:30]
    f_keep, warm = Molly.mbar_newton(u[keep, :], f_newton, N_keep, log.(Float64.(N_keep)))
    @test warm.iterations ≤ 5 < cold.iterations
    @test warm.residual ≤ 1e-10
    @test first(Molly.iterate_mbar(u[keep, :], win_of[keep], N_keep; method=:newton,
                                   f_init=f_newton .+ 3.0)) ≈ f_keep atol=1e-12
    @test first(Molly.iterate_mbar(u[keep, :], win_of[keep], N_keep; method=:newton)) ≈ f_keep atol=1e-8
    @test first(Molly.iterate_mbar(u, win_of, N_k; f_init=f_newton)) ≈ f_newton atol=1e-6

    # States without samples, including the first, are normalized by the solved free energies
    βs = exp.(0.1 .* (0:8))
    N_k = [0, 500, 500, 500, 0, 500, 500, 500, 500]
    u, win_of, f_exact = ladder_samples(rng, βs, 20, N_k)
    f_fp, _ = Molly.iterate_mbar(u, win_of, N_k; rtol=1e-13, max_iter=100_000)
    f_newton, _ = Molly.iterate_mbar(u, win_of, N_k; method=:newton)
    @test f_newton ≈ f_fp atol=1e-9
    @test f_newton[1] == 0
    @test f_newton ≈ f_exact atol=0.4
    logD = Molly.mbar_logD(u, f_newton, log.(Float64.(N_k)))
    @test all(k -> sum(exp.(f_newton[k] .- u[:, k] .- logD)) ≈ 1, (1, 5))
    # The fixed point started away from the solution reaches the same free energies
    @test first(Molly.iterate_mbar(u, win_of, N_k; f_init=f_exact, rtol=1e-13,
                                   max_iter=100_000)) ≈ f_fp atol=1e-9
    result = Molly.MBARResult(u, win_of, N_k, ones(9); method=:newton)
    Θ = result.covariance
    @test result.f == f_newton
    @test all(j -> isfinite(Θ[1, j]) && Θ[1, 1] + Θ[j, j] - 2Θ[1, j] > 0, 2:9)
    # Weights for a target state need samples in every state
    @test_throws ArgumentError Molly.mbar_weights(
                                    Molly.MBARInput(u, u[:, 1], N_k, win_of, nothing))

    # Nearly degenerate states, two identical states and two ladders joined by a weak link,
    #   started far from the solution, where Newton steps fail and fixed-point steps take over
    βs = vcat(exp.(0.1 .* (0:4)), exp.(1.4 .+ 0.1 .* (0:4)), exp(0.4))
    N_k = fill(300, 11)
    logN = log.(Float64.(N_k))
    u, win_of, f_exact = ladder_samples(rng, βs, 20, N_k)
    f, info = Molly.mbar_newton(u, zeros(11), N_k, logN)
    @test info.fixed_point_steps > 0 # This case is chosen to cover the fallback
    @test info.residual ≤ 1e-10
    @test f[11] ≈ f[5] atol=1e-8
    @test f ≈ f_exact atol=1.2

    # Asked for a residual below rounding error, the solver stops at the floor with a warning
    #   instead of running to max_iter
    f_floor, floor_info = @test_logs (:warn, r"normalization residual") Molly.mbar_newton(
                                        u, f, N_k, logN; atol=0.0)
    @test floor_info.iterations < 20
    @test f_floor ≈ f atol=1e-9

    # Free energies of order 1e7, where steps far below the rounding of f are not applied
    #   and the floor is above the default atol: the solver stops there and says so
    βs = exp.(0.1 .* (0:9))
    N_k = fill(200, 10)
    logN = log.(Float64.(N_k))
    u, win_of, _ = ladder_samples(rng, βs, 50, N_k)
    offsets = 1e6 .* (0:9)
    f, _ = Molly.mbar_newton(u, Molly.mbar_init(u, win_of), N_k, logN)
    f_large, large_info = @test_logs (:warn, r"rounding floor") Molly.mbar_newton(
                                        u .+ offsets', Molly.mbar_init(u .+ offsets', win_of),
                                        N_k, logN)
    @test large_info.iterations < 20
    @test large_info.residual ≤ 1e-8
    @test f_large .- offsets ≈ f atol=1e-6
end

@testset "Statistical inefficiency" begin
    # An AR(1) series with coefficient φ has statistical inefficiency (1 + φ) / (1 - φ)
    rng = Xoshiro(11)
    φ, n = 0.8, 100_000
    series = zeros(n)
    for i in 2:n
        series[i] = φ * series[i - 1] + randn(rng)
    end
    si = Molly.StatisticalInefficiency(series)
    @test si.inefficiency ≈ (1 + φ) / (1 - φ) rtol=0.1
    @test si.effective_size == n ÷ si.stride
    @test Molly.StatisticalInefficiency(randn(rng, n)).inefficiency ≈ 1 atol=0.05
end
