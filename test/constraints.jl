@testset "Constraints diatomic" begin
    r_cut = 8.5u"Å"
    temp = 300.0u"K"
    atom_mass = 1.00794u"g/mol"
    n_atoms = 400
    hydrogen_data = readdlm(joinpath(data_dir, "initial_hydrogen_data.atom"); skipstart=9)
    coords_matrix = hydrogen_data[:, 2:4]
    vel_matrix = hydrogen_data[:, 5:7]

    simulators = (
        VelocityVerlet(dt=0.002u"ps"),
        Verlet(dt=0.002u"ps"),
        StormerVerlet(dt=0.002u"ps"),
        Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1"),
    )

    bond_length = 0.74u"Å"
    constraints = [DistanceConstraint(j, j + 1, bond_length) for j in 1:2:n_atoms]
    atoms = [Atom(index=j, mass=atom_mass, σ=2.8279u"Å", ϵ=0.074u"kcal* mol^-1") for j in 1:n_atoms]
    cons_shake = SHAKE_RATTLE(n_atoms, 1e-8u"Å", 1e-8u"Å^2 * ps^-1"; dist_constraints=constraints)
    cons_lincs = LINCS(masses=repeat([atom_mass], n_atoms), dist_tolerance=1e-8u"Å",
                       vel_tolerance=1e-8u"Å^2 * ps^-1", dist_constraints=constraints)

    @test length(cons_shake.clusters12) == (n_atoms ÷ 2)

    boundary = CubicBoundary(200.0u"Å")
    lj = LennardJones(cutoff=ShiftedPotentialCutoff(r_cut), use_neighbors=true)
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        dist_cutoff=1.5*r_cut,
    )
    constraints = [cons_shake, cons_lincs]

    for simulator in simulators
        for cons in constraints
            coords = [SVector(coords_matrix[j, 1]u"Å", coords_matrix[j, 2]u"Å", coords_matrix[j, 3]u"Å")
                    for j in 1:n_atoms]
            velocities = [1000 * SVector(vel_matrix[j, :]u"Å/ps"...) for j in 1:n_atoms]

            sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                velocities=velocities,
                pairwise_inters=(lj,),
                neighbor_finder=neighbor_finder,
                constraints=(cons,),
                force_units=u"kcal * mol^-1 * Å^-1",
                energy_units=u"kcal * mol^-1",
            )

            simulate!(sys, simulator, 10_000)

            @test check_position_constraints(sys, cons)
            if simulator isa VelocityVerlet
                @test check_velocity_constraints(sys, cons)
            end
        end
    end
end

@testset "Constraints triatomic" begin
    n_atoms = 30
    atom_mass = 10.0u"g/mol"
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(2.0u"nm")

    coords = place_atoms(n_atoms ÷ 3, boundary, min_dist=0.3u"nm")
    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .+ [0.13, 0.0, 0.0]u"nm")
    end
    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .+ [0.26, 0.0, 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    is = collect(1:(2 * (n_atoms ÷ 3)))
    js = collect(((n_atoms ÷ 3) + 1):n_atoms)
    constraints = [DistanceConstraint(is[idx], js[idx], bond_length) for idx in eachindex(is)]
    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm",  1e-8u"nm^2/ps"; dist_constraints=constraints)

    @test length(cons.clusters23) == (n_atoms ÷ 3)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints 4-atom" begin
    n_atoms = 40
    atom_mass = 10.0u"g/mol"
    # Central atom
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(3.0u"nm")

    coords = place_atoms(n_atoms ÷ 4, boundary, min_dist=0.3u"nm")
    for i in 1:(n_atoms ÷ 4)
        push!(coords, coords[i] .+ [0.13, 0.0, 0.0]u"nm")
    end
    for i in 1:(n_atoms ÷ 4)
        push!(coords, coords[i] .- [0.13, 0.0, 0.0]u"nm")
    end
    for i in 1:(n_atoms ÷ 4)
        push!(coords, coords[i] .+ [0.0, 0.13, 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    is = repeat(1:(n_atoms ÷ 4), 3) # Central atom in each cluster
    js = collect(((n_atoms ÷ 4) + 1):n_atoms)
    constraints = [DistanceConstraint(is[idx], js[idx], bond_length) for idx in eachindex(is)]
    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm", 1e-8u"nm^2/ps"; dist_constraints=constraints)

    @test length(cons.clusters34) == (n_atoms ÷ 4)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints angle" begin
    n_atoms = 30
    atom_mass = 10.0u"g/mol"
    # Central atoms
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(3.0u"nm")

    coords = place_atoms(n_atoms ÷ 3, boundary, min_dist=0.3u"nm")
    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .- [0.13, 0.0, 0.0]u"nm")
    end
    θ = 2π / 3
    for i in 1:(n_atoms ÷ 3)
        push!(coords, coords[i] .+ [0.13 * cos(π - θ), 0.13 * sin(π - θ), 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    js = collect(1:(n_atoms ÷ 3))
    is = collect(((n_atoms ÷ 3) + 1):(2 * (n_atoms ÷ 3)))
    ks = collect((2 * (n_atoms ÷ 3) + 1):n_atoms)
    angle_constraints = [AngleConstraint(is[idx], js[idx], ks[idx], θ, bond_length, bond_length)
                         for idx in eachindex(is)]
    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm", 1e-8u"nm^2/ps"; angle_constraints=angle_constraints)

    @test length(cons.angle_clusters) == (n_atoms ÷ 3)

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints distance and angle" begin
    n_atoms = 60
    atom_mass = 10.0u"g/mol"
    # Central atoms
    atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
    boundary = CubicBoundary(3.0u"nm")

    coords = place_atoms(20, boundary, min_dist=0.3u"nm")

    # Angle constraints
    for i in 1:(n_atoms ÷ 6) # Atoms 21 -30
        push!(coords, coords[i] .- [0.13, 0.0, 0.0]u"nm")
    end
    θ = 2π / 3
    for i in 1:(n_atoms ÷ 6) # Atoms 31 - 40
        push!(coords, coords[i] .+ [0.13 * cos(π - θ), 0.13 * sin(π - θ), 0.0]u"nm")
    end

    # Central atom constraints
    for i in ((n_atoms ÷ 6) + 1):(2 * (n_atoms ÷ 6)) # Atoms 41 - 50
        push!(coords, coords[i] .+ [0.13, 0.0, 0.0]u"nm")
    end
    for i in ((n_atoms ÷ 6) + 1):(2 * (n_atoms ÷ 6)) # Atoms 51 - 60
        push!(coords, coords[i] .+ [0.26, 0.0, 0.0]u"nm")
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )
    bond_length = 0.13u"nm"

    js = collect(1:(n_atoms ÷ 6)) # 1 - 10
    is = collect((2 * (n_atoms ÷ 6) + 1):(3 * (n_atoms ÷ 6))) # 21 - 30
    ks = collect((3 * (n_atoms ÷ 6) + 1):(4 * (n_atoms ÷ 6))) # 31 - 40
    angle_constraints = [AngleConstraint(is[idx], js[idx], ks[idx], θ, bond_length, bond_length)
                         for idx in eachindex(is)]

    is = repeat(collect(((n_atoms ÷ 6) + 1):(2 * (n_atoms ÷ 6))), 2) # 11 - 20
    js = collect((4 * (n_atoms ÷ 6) + 1):n_atoms) # 41 - 60
    distance_constraints = [DistanceConstraint(is[idx], js[idx], bond_length)
                            for idx in eachindex(is)]

    cons = SHAKE_RATTLE(n_atoms, 1e-8u"nm", 1e-8u"nm^2/ps"; dist_constraints=distance_constraints,
                        angle_constraints=angle_constraints)

    @test length(cons.angle_clusters) == length(angle_constraints)
    @test length(cons.clusters23) == 10

    sys = System(
        atoms=atoms,
        coords=coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder
    )

    old_coords = copy(sys.coords)

    for i in eachindex(sys.coords)
        sys.coords[i]     += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm"
        sys.velocities[i] += [rand()*0.01, rand()*0.01, rand()*0.01]u"nm/ps"
    end

    apply_position_constraints!(sys, old_coords)
    apply_velocity_constraints!(sys)

    @test check_position_constraints(sys, cons)
    @test check_velocity_constraints(sys, cons)
end

@testset "Constraints protein CPU/GPU" begin
    pdb_fp = joinpath(data_dir, "1ubq.pdb") # No solvent
    T = Float32
    ff = MolecularForceField(
        T,
        joinpath(ff_dir, "ff99SBildn.xml"),
    )
    boundary = CubicBoundary(T(10.0)u"nm")
    temp = T(100.0)u"K"
    minimizer = SteepestDescentMinimizer()
    simulator = VelocityVerlet(dt=T(0.001)u"ps")

    for AT in array_list
        for rigid_water in (false, true)
            sys = System(
                pdb_fp,
                ff;
                boundary=boundary,
                array_type=AT,
                constraints=:hbonds,
                rigid_water=rigid_water, # No water present
            )

            simulate!(sys, minimizer)
            random_velocities!(sys, temp)

            simulate!(sys, simulator, 20)
            simulate!(sys, simulator, 1000)

            @test check_position_constraints(sys)
            @test check_velocity_constraints(sys)
            @test check_constraints(sys)

            coords_copy = copy(sys.coords)
            coords_bump     = randn(SVector{3, T}, length(sys))u"nm"         ./ 100
            velocities_bump = randn(SVector{3, T}, length(sys))u"nm * ps^-1" ./ 100
            sys.coords     .+= to_device(coords_bump, AT)
            sys.velocities .+= to_device(velocities_bump, AT)
            @test !check_position_constraints(sys)
            @test !check_velocity_constraints(sys)
            @test !check_constraints(sys)

            apply_position_constraints!(sys, coords_copy)
            apply_velocity_constraints!(sys)
            @test check_position_constraints(sys)
            @test check_velocity_constraints(sys)
            @test check_constraints(sys)
        end
    end
end

# --- LINCS test helpers ---

function make_lincs_diatomic(; mass1=12.0, mass2=12.0, bond_length=0.15,
                               v1=SVector(0.0, 0.0, 0.0), v2=SVector(0.0, 0.0, 0.0),
                               nrec=4, niter=1)
    dc = [DistanceConstraint(1, 2, bond_length)]
    masses = [mass1, mass2]
    x = [SVector(0.0, 0.0, 0.0), SVector(bond_length, 0.0, 0.0)]
    v = [v1, v2]
    data = Molly.build_lincs_data(dc, masses; nrec, niter)
    ws = Molly.create_lincs_workspace(data)
    return x, v, data, ws
end

function make_lincs_triatomic(; masses_val=[12.0, 12.0, 12.0], bond_length=0.15,
                                x=nothing, v=nothing, nrec=4, niter=1)
    if x === nothing
        x = [SVector(0.0, 0.0, 0.0), SVector(0.15, 0.0, 0.0), SVector(0.30, 0.0, 0.0)]
    end
    if v === nothing
        v = [SVector(0.0, 0.0, 0.0) for _ in 1:3]
    end
    dc = [DistanceConstraint(1, 2, bond_length), DistanceConstraint(2, 3, bond_length)]
    data = Molly.build_lincs_data(dc, masses_val; nrec, niter)
    ws = Molly.create_lincs_workspace(data)
    return x, v, data, ws
end

function make_lincs_methane(; center_mass=12.0, outer_mass=1.008, bond_length=0.109,
                              nrec=4, niter=1)
    x = [
        SVector(0.0, 0.0, 0.0),
        SVector( 0.0629,  0.0629,  0.0629),
        SVector(-0.0629, -0.0629,  0.0629),
        SVector(-0.0629,  0.0629, -0.0629),
        SVector( 0.0629, -0.0629, -0.0629),
    ]
    v = [SVector(0.0, 0.0, 0.0) for _ in 1:5]
    masses_val = [center_mass, outer_mass, outer_mass, outer_mass, outer_mass]
    dc = [
        DistanceConstraint(1, 2, bond_length),
        DistanceConstraint(1, 3, bond_length),
        DistanceConstraint(1, 4, bond_length),
        DistanceConstraint(1, 5, bond_length),
    ]
    data = Molly.build_lincs_data(dc, masses_val; nrec, niter)
    ws = Molly.create_lincs_workspace(data)
    return x, v, data, ws
end

function make_lincs_triangle(; masses_val=[12.0, 12.0, 12.0], bond_length=0.15,
                               x=nothing, v=nothing, nrec=4, niter=2)
    if x === nothing
        x = [
            SVector(0.0, 0.0, 0.0),
            SVector(bond_length, 0.0, 0.0),
            SVector(bond_length / 2, bond_length * sqrt(3) / 2, 0.0),
        ]
    end
    if v === nothing
        v = [SVector(0.0, 0.0, 0.0) for _ in 1:3]
    end
    dc = [
        DistanceConstraint(1, 2, bond_length),
        DistanceConstraint(2, 3, bond_length),
        DistanceConstraint(3, 1, bond_length),
    ]
    data = Molly.build_lincs_data(dc, masses_val; nrec, niter)
    ws = Molly.create_lincs_workspace(data)
    return x, v, data, ws
end

function make_lincs_square(; masses_val=[12.0, 12.0, 12.0, 12.0], bond_length=0.15,
                             x=nothing, v=nothing, nrec=4, niter=2)
    if x === nothing
        x = [
            SVector(0.0, 0.0, 0.0),
            SVector(bond_length, 0.0, 0.0),
            SVector(bond_length, bond_length, 0.0),
            SVector(0.0, bond_length, 0.0),
        ]
    end
    if v === nothing
        v = [SVector(0.0, 0.0, 0.0) for _ in 1:4]
    end
    dc = [
        DistanceConstraint(1, 2, bond_length),
        DistanceConstraint(2, 3, bond_length),
        DistanceConstraint(3, 4, bond_length),
        DistanceConstraint(4, 1, bond_length),
    ]
    data = Molly.build_lincs_data(dc, masses_val; nrec, niter)
    ws = Molly.create_lincs_workspace(data)
    return x, v, data, ws
end

function lincs_check_constraints(xp, data; atol=1e-10)
    for i in eachindex(data.atom1)
        d = norm(xp[data.atom1[i]] - xp[data.atom2[i]])
        if !isapprox(d, data.lengths[i]; atol)
            return false
        end
    end
    return true
end

function lincs_center_of_mass(x, masses_val)
    total = sum(masses_val)
    return sum(masses_val[i] * x[i] for i in eachindex(x)) / total
end

function make_lincs_chain(natoms; bond_length=0.15, mass=12.0)
    x = [SVector(i * bond_length, 0.0, 0.0) for i in 0:natoms-1]
    v = [SVector(0.01 * randn(), 0.01 * randn(), 0.01 * randn()) for _ in 1:natoms]
    dc = [DistanceConstraint(i, i + 1, bond_length) for i in 1:natoms-1]
    masses_val = fill(mass, natoms)
    return x, v, dc, masses_val
end

function make_lincs_branched(nbackbone; nbranch=3, backbone_length=0.153, branch_length=0.109,
                             backbone_mass=12.0, branch_mass=1.008)
    natoms = nbackbone + nbackbone * nbranch
    x = Vector{SVector{3,Float64}}(undef, natoms)
    masses_val = Vector{Float64}(undef, natoms)
    dc = DistanceConstraint{Float64, Int}[]

    for i in 1:nbackbone
        x[i] = SVector((i - 1) * backbone_length, 0.0, 0.0)
        masses_val[i] = backbone_mass
        if i > 1
            push!(dc, DistanceConstraint(i - 1, i, backbone_length))
        end
    end

    atom_idx = nbackbone
    for i in 1:nbackbone
        for j in 1:nbranch
            atom_idx += 1
            angle = 2π * j / nbranch
            x[atom_idx] = x[i] + SVector(0.0, branch_length * cos(angle), branch_length * sin(angle))
            masses_val[atom_idx] = branch_mass
            push!(dc, DistanceConstraint(i, atom_idx, branch_length))
        end
    end

    v = [SVector(0.01 * randn(), 0.01 * randn(), 0.01 * randn()) for _ in 1:natoms]
    return x, v, dc, masses_val
end

# --- LINCS tests ---

@testset "LINCS setup" begin
    # single constraint - empty coupling
    @testset begin
        dc = [DistanceConstraint(1, 2, 0.15)]
        masses_val = [12.0, 12.0]
        data = Molly.build_lincs_data(dc, masses_val)

        @test length(data.coupling.neighbors) == 0
        @test data.coupling.range == [1, 1]
        @test data.sdiag[1] ≈ 1.0 / sqrt(1/12.0 + 1/12.0)
    end

    # chain A-B-C - one coupling pair
    @testset begin
        dc = [DistanceConstraint(1, 2, 0.15), DistanceConstraint(2, 3, 0.15)]
        masses_val = [12.0, 12.0, 12.0]
        data = Molly.build_lincs_data(dc, masses_val)

        @test data.coupling.range == [1, 2, 3]
        @test data.coupling.neighbors == [2, 1]

        invmass2 = 1.0 / 12.0
        s1 = data.sdiag[1]
        s2 = data.sdiag[2]
        expected_coef = 1.0 * invmass2 * s1 * s2
        @test data.coupling.coef[1] ≈ expected_coef
        @test data.coupling.coef[2] ≈ expected_coef
    end

    # star topology (methane-like)
    @testset begin
        dc = [
            DistanceConstraint(1, 2, 0.109),
            DistanceConstraint(1, 3, 0.109),
            DistanceConstraint(1, 4, 0.109),
            DistanceConstraint(1, 5, 0.109),
        ]
        masses_val = [12.0, 1.008, 1.008, 1.008, 1.008]
        data = Molly.build_lincs_data(dc, masses_val)

        for i in 1:4
            n_neighbors = data.coupling.range[i+1] - data.coupling.range[i]
            @test n_neighbors == 3
        end

        for n in 1:length(data.coupling.coef)
            @test data.coupling.coef[n] < 0
        end
    end

    # sdiag values
    @testset begin
        dc = [DistanceConstraint(1, 2, 0.15)]
        masses_val = [1.0, 4.0]
        data = Molly.build_lincs_data(dc, masses_val)
        @test data.sdiag[1] ≈ 1.0 / sqrt(1.0 + 0.25)
    end

    # nrec and niter defaults
    @testset begin
        dc = [DistanceConstraint(1, 2, 0.15)]
        masses_val = [12.0, 12.0]
        data = Molly.build_lincs_data(dc, masses_val)
        @test data.nrec == 4
        @test data.niter == 1

        data2 = Molly.build_lincs_data(dc, masses_val; nrec=6, niter=2)
        @test data2.nrec == 6
        @test data2.niter == 2
    end

    # workspace allocation
    @testset begin
        dc = [DistanceConstraint(1, 2, 0.15)]
        masses_val = [12.0, 12.0]
        data = Molly.build_lincs_data(dc, masses_val)
        ws = Molly.create_lincs_workspace(data)

        @test length(ws.B) == 1
        @test length(ws.rhs) == 1
        @test length(ws.sol) == 1
        @test length(ws.tmp) == 1
    end
end

@testset "LINCS solver" begin
    # nrec=0 - sol equals rhs, direct position update
    @testset begin
        x, v, _, _ = make_lincs_diatomic(; v1=SVector(0.1, 0.0, 0.0), v2=SVector(-0.1, 0.0, 0.0))
        dc = [DistanceConstraint(1, 2, 0.15)]
        masses_val = [12.0, 12.0]
        data = Molly.build_lincs_data(dc, masses_val; nrec=0, niter=0)
        ws = Molly.create_lincs_workspace(data)

        dt = 0.002
        xp = x .+ v .* dt

        diff = x[1] - x[2]
        B = normalize(diff)
        ws.B[1] = B

        proj = dot(B, xp[1] - xp[2])
        rhs_val = data.sdiag[1] * (proj - 0.15)
        ws.rhs[1] = rhs_val
        ws.sol[1] = rhs_val

        xp_before = copy(xp)
        Molly.lincs_solve!(xp, data, ws, 1.0)

        invmass = 1.0 / 12.0
        expected_delta = invmass * B * data.sdiag[1] * rhs_val
        @test xp[1] ≈ xp_before[1] - expected_delta atol=1e-16
        @test xp[2] ≈ xp_before[2] + expected_delta atol=1e-16
    end

    # increasing nrec improves constraint satisfaction
    @testset begin
        x, v, _, _ = make_lincs_triatomic(;
            v=[SVector(0.05, 0.02, 0.0), SVector(-0.03, 0.01, 0.0), SVector(0.01, -0.02, 0.0)]
        )
        dt = 0.002
        deviations = Float64[]

        for nrec in [0, 1, 2, 4, 8]
            dc = [DistanceConstraint(1, 2, 0.15), DistanceConstraint(2, 3, 0.15)]
            masses_val = [12.0, 12.0, 12.0]
            data = Molly.build_lincs_data(dc, masses_val; nrec, niter=1)
            ws = Molly.create_lincs_workspace(data)
            xp = x .+ v .* dt
            Molly.lincs_apply!(xp, x, data, ws, nothing)

            max_dev = maximum(abs(norm(xp[data.atom1[i]] - xp[data.atom2[i]]) - data.lengths[i])
                              for i in eachindex(data.atom1))
            push!(deviations, max_dev)
        end

        for i in 2:length(deviations)
            @test deviations[i] <= deviations[i-1] + 1e-15
        end
    end
end

@testset "LINCS algorithm" begin
    # diatomic equal mass
    @testset begin
        x, v, data, ws = make_lincs_diatomic(;
            v1=SVector(0.1, 0.05, 0.0), v2=SVector(-0.1, -0.05, 0.0)
        )
        dt = 0.002
        xp = x .+ v .* dt
        Molly.lincs_apply!(xp, x, data, ws, nothing)

        d = norm(xp[1] - xp[2])
        @test d ≈ 0.15 atol=1e-10
    end

    # diatomic unequal mass - constraint + CoM conservation
    @testset begin
        x, v, data, ws = make_lincs_diatomic(;
            mass1=1.0, mass2=16.0,
            v1=SVector(0.2, 0.1, 0.0), v2=SVector(-0.05, 0.02, 0.0)
        )
        masses_val = [1.0, 16.0]
        dt = 0.002
        xp = x .+ v .* dt

        com_before = lincs_center_of_mass(xp, masses_val)
        Molly.lincs_apply!(xp, x, data, ws, nothing)
        com_after = lincs_center_of_mass(xp, masses_val)

        d = norm(xp[1] - xp[2])
        @test d ≈ 0.15 atol=1e-10
        @test com_after ≈ com_before atol=1e-12
    end

    # triatomic chain
    @testset begin
        x, v, data, ws = make_lincs_triatomic(;
            v=[SVector(0.05, 0.02, 0.0), SVector(-0.03, 0.01, 0.0), SVector(0.01, -0.02, 0.0)]
        )
        dt = 0.002
        xp = x .+ v .* dt
        Molly.lincs_apply!(xp, x, data, ws, nothing)

        @test lincs_check_constraints(xp, data; atol=1e-6)
    end

    # methane star
    @testset begin
        x, v, data, ws = make_lincs_methane()
        v = [
            SVector(0.0, 0.0, 0.0),
            SVector(0.5, 0.3, -0.1),
            SVector(-0.3, 0.5, 0.2),
            SVector(0.2, -0.4, 0.5),
            SVector(-0.4, -0.2, -0.3),
        ]
        dt = 0.002
        xp = x .+ v .* dt
        Molly.lincs_apply!(xp, x, data, ws, nothing)

        @test lincs_check_constraints(xp, data; atol=1e-6)
    end

    # zero displacement is no-op
    @testset begin
        x, v, data, ws = make_lincs_diatomic()
        xp = copy(x)
        x_orig = copy(x)
        Molly.lincs_apply!(xp, x, data, ws, nothing)

        @test xp ≈ x_orig atol=1e-14
    end

    # niter > 1 improves accuracy
    @testset begin
        x, _, _, _ = make_lincs_methane()
        v = [
            SVector(0.0, 0.0, 0.0),
            SVector(1.0, 0.5, -0.2),
            SVector(-0.6, 1.0, 0.4),
            SVector(0.4, -0.8, 1.0),
            SVector(-0.8, -0.4, -0.6),
        ]
        dt = 0.002

        deviations = Float64[]
        for niter in [1, 2, 4]
            x_copy, _, data, ws = make_lincs_methane(; niter)
            xp = x .+ v .* dt
            Molly.lincs_apply!(xp, x_copy, data, ws, nothing)
            max_dev = maximum(abs(norm(xp[data.atom1[i]] - xp[data.atom2[i]]) - data.lengths[i])
                              for i in eachindex(data.atom1))
            push!(deviations, max_dev)
        end

        @test deviations[2] <= deviations[1] + 1e-15
        @test deviations[3] <= deviations[2] + 1e-15
    end

    # triangle ring
    @testset begin
        x, v, data, ws = make_lincs_triangle(;
            v=[SVector(0.05, 0.02, 0.0), SVector(-0.03, 0.04, 0.0), SVector(0.01, -0.03, 0.0)]
        )
        masses_val = [12.0, 12.0, 12.0]
        dt = 0.002
        xp = x .+ v .* dt

        com_before = lincs_center_of_mass(xp, masses_val)
        Molly.lincs_apply!(xp, x, data, ws, nothing)
        com_after = lincs_center_of_mass(xp, masses_val)

        @test lincs_check_constraints(xp, data; atol=1e-4)
        @test com_after ≈ com_before atol=1e-12
    end

    # square ring
    @testset begin
        x, v, data, ws = make_lincs_square(;
            v=[SVector(0.05, 0.02, 0.0), SVector(-0.03, 0.04, 0.0),
               SVector(0.01, -0.03, 0.0), SVector(-0.02, 0.01, 0.0)]
        )
        masses_val = [12.0, 12.0, 12.0, 12.0]
        dt = 0.002
        xp = x .+ v .* dt

        com_before = lincs_center_of_mass(xp, masses_val)
        Molly.lincs_apply!(xp, x, data, ws, nothing)
        com_after = lincs_center_of_mass(xp, masses_val)

        @test lincs_check_constraints(xp, data; atol=1e-4)
        @test com_after ≈ com_before atol=1e-12
    end
end

@testset "LINCS integration with System" begin
    r_cut = 8.5u"Å"
    temp = 300.0u"K"
    atom_mass = 1.00794u"g/mol"
    n_atoms = 400
    hydrogen_data = readdlm(joinpath(data_dir, "initial_hydrogen_data.atom"); skipstart=9)
    coords_matrix = hydrogen_data[:, 2:4]
    vel_matrix = hydrogen_data[:, 5:7]

    bond_length = 0.74u"Å"
    dist_constraints = [DistanceConstraint(j, j + 1, bond_length) for j in 1:2:n_atoms]
    atoms = [Atom(index=j, mass=atom_mass, σ=2.8279u"Å", ϵ=0.074u"kcal* mol^-1") for j in 1:n_atoms]
    atom_masses = [atom_mass for _ in 1:n_atoms]

    cons = LINCS(masses=atom_masses, dist_tolerance=1e-8u"Å", vel_tolerance=1e-8u"Å^2 * ps^-1",
                 dist_constraints=dist_constraints)

    @test length(cons.clusters) == (n_atoms ÷ 2)

    boundary = CubicBoundary(200.0u"Å")
    lj = LennardJones(cutoff=ShiftedPotentialCutoff(r_cut), use_neighbors=true)
    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        dist_cutoff=1.5*r_cut,
    )

    simulators = (
        VelocityVerlet(dt=0.002u"ps"),
        Verlet(dt=0.002u"ps"),
        StormerVerlet(dt=0.002u"ps"),
        Langevin(dt=0.002u"ps", temperature=temp, friction=1.0u"ps^-1"),
    )

    for simulator in simulators
        coords = [SVector(coords_matrix[j, 1]u"Å", coords_matrix[j, 2]u"Å", coords_matrix[j, 3]u"Å")
                  for j in 1:n_atoms]
        velocities = [1000 * SVector(vel_matrix[j, :]u"Å/ps"...) for j in 1:n_atoms]

        sys = System(
            atoms=atoms,
            coords=coords,
            boundary=boundary,
            velocities=velocities,
            pairwise_inters=(lj,),
            neighbor_finder=neighbor_finder,
            constraints=(cons,),
            force_units=u"kcal * mol^-1 * Å^-1",
            energy_units=u"kcal * mol^-1",
        )

        simulate!(sys, simulator, 10_000)

        @test check_position_constraints(sys, cons)
        if simulator isa VelocityVerlet
            @test check_velocity_constraints(sys, cons)
        end
    end
end

@testset "LINCS GPU integration" begin
    T = Float32
    n_atoms = 200
    bond_length = T(0.74)u"Å"
    atom_mass = T(1.00794)u"g/mol"
    r_cut = T(8.5)u"Å"
    temp = T(300.0)u"K"

    dist_constraints = [DistanceConstraint(j, j + 1, bond_length) for j in 1:2:n_atoms]
    atoms = [Atom(index=j, mass=atom_mass, σ=T(2.8279)u"Å", ϵ=T(0.074)u"kcal* mol^-1")
             for j in 1:n_atoms]
    atom_masses = [atom_mass for _ in 1:n_atoms]

    cons = LINCS(masses=atom_masses, dist_tolerance=T(1e-4)u"Å", vel_tolerance=T(1e-4)u"Å^2 * ps^-1",
                 dist_constraints=dist_constraints)

    boundary = CubicBoundary(T(200.0)u"Å")
    lj = LennardJones(cutoff=ShiftedPotentialCutoff(r_cut), use_neighbors=true)

    for AT in array_list[2:end]

        if Molly.uses_gpu_neighbor_finder(AT)
            neighbor_finder = GPUNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                dist_cutoff=T(1.5)*r_cut,
            )
        else
            neighbor_finder = DistanceNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                dist_cutoff=T(1.5)*r_cut,
            )
        end

        # Place molecules on a 2D grid with 5 Å spacing so adjacent molecules
        # are ~4.26 Å apart (well above σ=2.83 Å to avoid LJ blowup)
        n_mol = n_atoms ÷ 2
        n_grid = isqrt(n_mol)
        coords = [SVector{3, T}(
            T(5.0) + T(5.0) * T(((j - 1) ÷ 2) % n_grid) + (isodd(j) ? T(-0.37) : T(0.37)),
            T(5.0) + T(5.0) * T(((j - 1) ÷ 2) ÷ n_grid),
            T(5.0),
        )u"Å" for j in 1:n_atoms]

        simulator = VelocityVerlet(dt=T(0.002)u"ps")

        sys = System(
            atoms=to_device(atoms, AT),
            coords=to_device(coords, AT),
            boundary=boundary,
            pairwise_inters=(lj,),
            neighbor_finder=neighbor_finder,
            constraints=(cons,),
            force_units=u"kcal * mol^-1 * Å^-1",
            energy_units=u"kcal * mol^-1",
        )
        random_velocities!(sys, temp)

        simulate!(sys, simulator, 500)

        @test check_position_constraints(sys, cons)
        if simulator isa VelocityVerlet
            @test check_velocity_constraints(sys, cons)
        end
    end
end

@testset "LINCS angle constraints" begin
    n_molecules = 10
    n_atoms = 3 * n_molecules
    mass_O = 15.999u"g/mol"
    mass_H = 1.008u"g/mol"

    atoms = Atom[]
    for _ in 1:n_molecules
        push!(atoms, Atom(mass=mass_O, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1"))
        push!(atoms, Atom(mass=mass_H, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1"))
        push!(atoms, Atom(mass=mass_H, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1"))
    end
    atom_masses = [a.mass for a in atoms]

    boundary = CubicBoundary(3.0u"nm")
    bond_length = 0.09572u"nm"
    θ = deg2rad(104.52)

    coords = place_atoms(n_molecules, boundary, min_dist=0.3u"nm")
    full_coords = SVector{3, typeof(1.0u"nm")}[]
    for i in 1:n_molecules
        O_pos = coords[i]
        H1_pos = O_pos .+ SVector(bond_length, 0.0u"nm", 0.0u"nm")
        H2_pos = O_pos .+ SVector(bond_length * cos(θ), bond_length * sin(θ), 0.0u"nm")
        push!(full_coords, O_pos)
        push!(full_coords, H1_pos)
        push!(full_coords, H2_pos)
    end

    temp = 100.0u"K"
    velocities = [random_velocity(atoms[i].mass, temp) for i in 1:n_atoms]

    angle_constraints = [AngleConstraint(3*(i-1)+2, 3*(i-1)+1, 3*(i-1)+3, θ, bond_length, bond_length)
                         for i in 1:n_molecules]

    cons = LINCS(masses=atom_masses, dist_tolerance=1e-5u"nm", vel_tolerance=1e-5u"nm^2 * ps^-1",
                 angle_constraints=angle_constraints, nrec=8, niter=2)

    @test !isnothing(cons.angle_constraints)
    @test length(cons.angle_constraints) == n_molecules
    @test length(cons.dist_constraints) == 3 * n_molecules

    neighbor_finder = DistanceNeighborFinder(
        eligible=trues(n_atoms, n_atoms),
        n_steps=10,
        dist_cutoff=1.5u"nm",
    )

    sys = System(
        atoms=atoms,
        coords=full_coords,
        boundary=boundary,
        velocities=velocities,
        pairwise_inters=(LennardJones(use_neighbors=true),),
        constraints=(cons,),
        neighbor_finder=neighbor_finder,
    )

    simulate!(sys, VelocityVerlet(dt=0.001u"ps"), 1000)

    @test check_position_constraints(sys, cons)

    # isolation validation
    @testset begin
        dc = [DistanceConstraint(1, 4, 0.15u"nm")]
        ac = [AngleConstraint(1, 2, 3, θ, bond_length, bond_length)]
        @test_throws ArgumentError LINCS(masses=atom_masses, dist_constraints=dc, angle_constraints=ac)

        ac_overlap = [
            AngleConstraint(1, 2, 3, θ, bond_length, bond_length),
            AngleConstraint(3, 4, 5, θ, bond_length, bond_length),
        ]
        @test_throws ArgumentError LINCS(masses=atom_masses, angle_constraints=ac_overlap)
    end

    # show method
    @testset begin
        s = sprint(show, cons)
        @test occursin("distance", s)
        @test occursin("angle", s)
    end
end

@testset "LINCS benchmarks" begin
    # Warmup
    x_w, v_w, c_w, m_w = make_lincs_chain(10)
    d_w = Molly.build_lincs_data(c_w, m_w)
    ws_w = Molly.create_lincs_workspace(d_w)
    xp_w = x_w .+ v_w .* 0.002
    Molly.lincs_apply!(xp_w, x_w, d_w, ws_w, nothing)

    # apply_lincs! zero allocations
    @testset begin
        for (label, nbackbone, nbranch) in [
            ("small (7 constraints)", 2, 3),
            ("medium (399 constraints)", 100, 3),
        ]
            x, v, c, m = make_lincs_branched(nbackbone; nbranch)
            data = Molly.build_lincs_data(c, m)
            ws = Molly.create_lincs_workspace(data)
            xp = x .+ v .* 0.002

            # Warmup
            Molly.lincs_apply!(xp, x, data, ws, nothing)
            xp .= x .+ v .* 0.002

            allocs = @allocated Molly.lincs_apply!(xp, x, data, ws, nothing)
            @test allocs == 0
        end
    end

    # solve! zero allocations
    @testset begin
        x, v, c, m = make_lincs_branched(100; nbranch=3)
        data = Molly.build_lincs_data(c, m)
        ws = Molly.create_lincs_workspace(data)
        xp = x .+ v .* 0.002

        K = length(data.atom1)
        for i in 1:K
            ws.B[i] = normalize(x[data.atom1[i]] - x[data.atom2[i]])
        end
        coupling = data.coupling
        for i in 1:K
            for n in coupling.range[i]:(coupling.range[i+1]-1)
                j = coupling.neighbors[n]
                ws.blcc[n] = coupling.coef[n] * dot(ws.B[i], ws.B[j])
            end
        end
        for i in 1:K
            ws.rhs[i] = data.sdiag[i] * (dot(ws.B[i], xp[data.atom1[i]] - xp[data.atom2[i]]) - data.lengths[i])
        end
        ws.sol .= ws.rhs

        # Warmup
        Molly.lincs_solve!(xp, data, ws, 1.0)
        ws.sol .= ws.rhs
        xp .= x .+ v .* 0.002

        allocs = @allocated Molly.lincs_solve!(xp, data, ws, 1.0)
        @test allocs == 0
    end
end
