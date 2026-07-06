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
                # Verlet and Langevin are half-step integrators so this is not expected to be true
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

    constraint_algorithms = [SHAKE_RATTLE, LINCS]

    for AT in array_list
        for constraint_algorithm in constraint_algorithms
            for rigid_water in (false, true)
                sys = System(
                    pdb_fp,
                    ff;
                    boundary=boundary,
                    array_type=AT,
                    constraints=:hbonds,
                    rigid_water=rigid_water, # No water present
                    constraint_algorithm=constraint_algorithm,    
                )
                if constraint_algorithm == LINCS
                    # Increase tolerances from default
                    lincs = LINCS(
                        masses=Array(masses(sys)),
                        dist_constraints=sys.constraints[1].dist_constraints,
                        angle_constraints=sys.constraints[1].angle_constraints,
                        dist_tolerance=sys.constraints[1].dist_tolerance,
                        vel_tolerance=sys.constraints[1].vel_tolerance,
                        nrec=6,
                        niter=6,
                    )
                    sys.constraints = (Molly.setup_constraints!(lincs, sys.neighbor_finder, AT),)
                end

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
end

@testset "GPU constrained System constructor" begin
    if CuArray in array_list
        T = Float32
        AT = CuArray
        n_atoms = 2
        atom_mass = T(10.0)u"g/mol"
        bond_length = T(0.13)u"nm"
        dist_constraints = [DistanceConstraint(1, 2, bond_length)]
        coords = to_device([
            SVector(T(0.0), T(0.0), T(0.0))u"nm",
            SVector(T(0.13), T(0.0), T(0.0))u"nm",
        ], AT)
        velocities = to_device(fill(SVector(T(0.0), T(0.0), T(0.0))u"nm * ps^-1",
                                    n_atoms), AT)
        atoms = to_device([
            Atom(mass=atom_mass, σ=T(0.3)u"nm", ϵ=T(0.2)u"kJ * mol^-1")
            for _ in 1:n_atoms
        ], AT)
        boundary = CubicBoundary(T(3.0)u"nm")
        pairwise_inters = (LennardJones(use_neighbors=true),)

        constraints = (
            SHAKE_RATTLE(n_atoms, T(1e-6)u"nm", T(1e-6)u"nm^2/ps";
                         dist_constraints=dist_constraints),
            LINCS(masses=fill(atom_mass, n_atoms), dist_tolerance=T(1e-6)u"nm",
                  vel_tolerance=T(1e-6)u"nm^2/ps", dist_constraints=dist_constraints),
        )

        for cons in constraints
            neighbor_finder = GPUNeighborFinder(
                eligible=to_device(trues(n_atoms, n_atoms), AT),
                dist_cutoff=T(1.0)u"nm",
            )
            sys = System(
                atoms=atoms,
                coords=coords,
                boundary=boundary,
                velocities=velocities,
                pairwise_inters=pairwise_inters,
                constraints=(cons,),
                neighbor_finder=neighbor_finder,
            )
            sys_rebuilt = System(deepcopy(sys); general_inters=sys.general_inters)

            @test sys_rebuilt.df == sys.df
            @test length(sys_rebuilt.constraints) == length(sys.constraints)
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

function make_lincs_chain(natoms; bond_length=0.15, mass=12.0)
    x = [SVector(i * bond_length, 0.0, 0.0) for i in 0:(natoms - 1)]
    v = [SVector(0.01 * randn(), 0.01 * randn(), 0.01 * randn()) for _ in 1:natoms]
    dc = [DistanceConstraint(i, i + 1, bond_length) for i in 1:(natoms - 1)]
    return x, v, dc, fill(mass, natoms)
end

function make_lincs_branched(nbackbone; nbranch=3, backbone_length=0.153,
                             branch_length=0.109, backbone_mass=12.0,
                             branch_mass=1.008)
    natoms = nbackbone * (nbranch + 1)
    x = Vector{SVector{3, Float64}}(undef, natoms)
    masses_val = Vector{Float64}(undef, natoms)
    dc = DistanceConstraint{Float64, Int}[]

    for i in 1:nbackbone
        x[i] = SVector((i - 1) * backbone_length, 0.0, 0.0)
        masses_val[i] = backbone_mass
        i > 1 && push!(dc, DistanceConstraint(i - 1, i, backbone_length))
    end

    atom_idx = nbackbone
    for i in 1:nbackbone, j in 1:nbranch
        atom_idx += 1
        angle = 2π * j / nbranch
        x[atom_idx] = x[i] + SVector(0.0, branch_length * cos(angle),
                                    branch_length * sin(angle))
        masses_val[atom_idx] = branch_mass
        push!(dc, DistanceConstraint(i, atom_idx, branch_length))
    end

    v = [SVector(0.01 * randn(), 0.01 * randn(), 0.01 * randn()) for _ in 1:natoms]
    return x, v, dc, masses_val
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

function make_lincs_no_unit_buffer_system(coords, masses_val)
    atoms = [Atom(mass=m, σ=0.0, ϵ=0.0) for m in masses_val]
    sys = System(
        atoms=atoms,
        coords=copy(coords),
        velocities=zero.(coords),
        boundary=CubicBoundary(5.0),
        force_units=NoUnits,
        energy_units=NoUnits,
    )
    return sys, Molly.init_buffers!(sys, 1)
end

function lincs_position_context(buffers=nothing; needs_virial=true, step_n=1,
                                virial_scale=1.0)
    return Molly.ConstraintApplicationContext(
        kind=Molly.PositionConstraintApplication(),
        needs_virial=needs_virial,
        step_n=Int(step_n),
        virial_scale=virial_scale,
        buffers=buffers,
    )
end

lincs_no_virial_context() = lincs_position_context(; needs_virial=false)

function lincs_velocity_context(buffers=nothing; needs_virial=true, step_n=1,
                                virial_scale=1.0)
    return Molly.ConstraintApplicationContext(
        kind=Molly.VelocityConstraintApplication(),
        needs_virial=needs_virial,
        step_n=Int(step_n),
        virial_scale=virial_scale,
        buffers=buffers,
    )
end

lincs_no_velocity_virial_context() = lincs_velocity_context(; needs_virial=false)

shake_position_context(buffers=nothing; needs_virial=true, step_n=1,
                       virial_scale=1.0) = lincs_position_context(
    buffers; needs_virial, step_n, virial_scale,
)

shake_velocity_context(buffers=nothing; needs_virial=true, step_n=1,
                       virial_scale=1.0) = lincs_velocity_context(
    buffers; needs_virial, step_n, virial_scale,
)

@testset "constraint virial context scaling" begin
    coords = [
        SVector(0.0, 0.0, 0.0),
        SVector(0.15, 0.0, 0.0),
    ]
    sys, buffers = make_lincs_no_unit_buffer_system(coords, [12.0, 16.0])
    dt = 0.5

    pos_non_vv = Molly.position_constraint_virial_scale(sys, buffers, dt)
    vel_non_vv = Molly.velocity_constraint_virial_scale(sys, buffers, dt)

    @test pos_non_vv == inv(dt^2)
    @test vel_non_vv == inv(dt)

    for sim in (VelocityVerlet(dt=dt), DPDVelocityVerlet(dt=dt))
        @test Molly.position_constraint_virial_scale(sys, buffers, dt, sim) == 2 * pos_non_vv
        @test Molly.velocity_constraint_virial_scale(sys, buffers, dt, sim) == 2 * vel_non_vv
    end

    for sim in (Verlet(dt=dt), StormerVerlet(dt=dt),
                Langevin(dt=dt, temperature=1.0, friction=1.0))
        @test Molly.position_constraint_virial_scale(sys, buffers, dt, sim) == pos_non_vv
        @test Molly.velocity_constraint_virial_scale(sys, buffers, dt, sim) == vel_non_vv
    end

    @test Molly.position_constraint_context(buffers, sys, 1, dt, true).virial_scale == pos_non_vv
    @test Molly.velocity_constraint_context(buffers, sys, 1, dt, true).virial_scale == vel_non_vv

    masses_val = [12.0u"g/mol", 16.0u"g/mol"]
    unit_coords = [
        SVector(0.0, 0.0, 0.0)u"nm",
        SVector(0.15, 0.0, 0.0)u"nm",
    ]
    atoms = [Atom(mass=m, σ=0.0u"nm", ϵ=0.0u"kJ/mol") for m in masses_val]
    unit_sys = System(
        atoms=atoms,
        coords=unit_coords,
        velocities=zero.(unit_coords) ./ 1.0u"ps",
        boundary=CubicBoundary(5.0u"nm"),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
    )
    unit_buffers = Molly.init_buffers!(unit_sys, 1)
    unit_dt = 0.004u"ps"
    unit_pos_non_vv = Molly.position_constraint_virial_scale(unit_sys, unit_buffers, unit_dt)
    unit_vel_non_vv = Molly.velocity_constraint_virial_scale(unit_sys, unit_buffers, unit_dt)

    sim = VelocityVerlet(dt=unit_dt)
    @test Molly.position_constraint_virial_scale(unit_sys, unit_buffers, unit_dt, sim) ≈
          2 * unit_pos_non_vv
    @test Molly.velocity_constraint_virial_scale(unit_sys, unit_buffers, unit_dt, sim) ≈
          2 * unit_vel_non_vv
end

# --- SHAKE/RATTLE virial tests ---

@testset "SHAKE_RATTLE CPU virial" begin
    function make_shake_system(coords, velocities, masses_val;
                               dist_constraints=nothing, angle_constraints=nothing)
        atoms = [Atom(mass=m, σ=0.0, ϵ=0.0) for m in masses_val]
        cons = SHAKE_RATTLE(
            length(coords), 1e-10, 1e-10;
            dist_constraints=dist_constraints,
            angle_constraints=angle_constraints,
            max_iters=50,
        )
        sys = System(
            atoms=atoms,
            coords=copy(coords),
            velocities=copy(velocities),
            boundary=CubicBoundary(5.0),
            constraints=(cons,),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        return sys, sys.constraints[1]
    end

    expected_cluster_atom_inds(cluster) =
        filter(!isnothing, (
            Int(cluster.k1),
            Int(cluster.k2),
            hasproperty(cluster, :k3) ? Int(cluster.k3) : nothing,
            hasproperty(cluster, :k4) ? Int(cluster.k4) : nothing,
        ))

    function expected_constraint_virial_local(coords_ref, values_before, values_after,
                                              masses_val, scale, boundary, cons)
        expected = zeros(3, 3)
        for clusters in (cons.clusters12, cons.clusters23, cons.clusters34, cons.angle_clusters)
            for cluster in clusters
                atom_inds = expected_cluster_atom_inds(cluster)
                anchor = first(atom_inds)
                anchor_coord = coords_ref[anchor]
                for atom_i in atom_inds
                    local_coord = atom_i == anchor ? zero(coords_ref[atom_i]) :
                                  vector(anchor_coord, coords_ref[atom_i], boundary)
                    correction_force = masses_val[atom_i] *
                                       (values_after[atom_i] - values_before[atom_i]) *
                                       scale
                    expected .+= local_coord * transpose(correction_force)
                end
            end
        end
        return expected
    end

    function expected_position_constraint_virial(coords_ref, coords_before, coords_after,
                                                 masses_val, dt, boundary, cons)
        return expected_constraint_virial_local(coords_ref, coords_before, coords_after,
                                                masses_val, inv(dt^2), boundary, cons)
    end

    function expected_velocity_constraint_virial(coords, velocities_before, velocities_after,
                                                 masses_val, dt, boundary, cons)
        return expected_constraint_virial_local(coords, velocities_before, velocities_after,
                                                masses_val, inv(dt), boundary, cons)
    end

    dt = 0.5
    bond_length = 0.15

    # 2-atom position virial
    @testset "2-atom position virial" begin
        masses_val = [12.0, 16.0]
        direction = normalize(SVector(1.0, 2.0, -1.0))
        origin = SVector(0.2, -0.1, 0.05)
        old_coords = [origin, origin + bond_length * direction]
        unconstrained = [old_coords[1], old_coords[2] + 0.001 * direction]
        velocities = zero.(old_coords)
        dc = [DistanceConstraint(1, 2, bond_length)]
        sys, cons = make_shake_system(unconstrained, velocities, masses_val;
                                      dist_constraints=dc)
        buffers = Molly.init_buffers!(sys, 1)
        step_n = 31
        Molly.clear_constraint_virial!(buffers, sys, step_n)
        context = shake_position_context(buffers; step_n, virial_scale=inv(dt^2))

        apply_position_constraints!(sys, cons, old_coords; context)
        expected = expected_position_constraint_virial(old_coords, unconstrained, sys.coords,
                                                       masses_val, dt, sys.boundary, cons)

        @test check_position_constraints(sys, cons)
        @test buffers.constraint_virial_nounits ≈ expected atol=1e-12
        @test Molly.has_constraint_virial(buffers, step_n)
    end

    # 2-atom velocity virial
    @testset "2-atom velocity virial" begin
        masses_val = [12.0, 16.0]
        direction = normalize(SVector(1.0, 2.0, -1.0))
        origin = SVector(0.2, -0.1, 0.05)
        coords = [origin, origin + bond_length * direction]
        velocities = [SVector(-0.1, 0.0, 0.0), SVector(0.1, 0.0, 0.0)]
        velocities_before = copy(velocities)
        dc = [DistanceConstraint(1, 2, bond_length)]
        sys, cons = make_shake_system(coords, velocities, masses_val; dist_constraints=dc)
        buffers = Molly.init_buffers!(sys, 1)
        step_n = 32
        Molly.clear_constraint_virial!(buffers, sys, step_n)
        context = shake_velocity_context(buffers; step_n, virial_scale=inv(dt))

        apply_velocity_constraints!(sys, cons; context)
        expected = expected_velocity_constraint_virial(coords, velocities_before,
                                                       sys.velocities, masses_val, dt,
                                                       sys.boundary, cons)

        @test check_velocity_constraints(sys, cons)
        @test buffers.constraint_virial_nounits ≈ expected atol=1e-12
        @test Molly.has_constraint_virial(buffers, step_n)
    end

    # Angle-cluster position and velocity virial
    @testset "Angle-cluster position and velocity virial" begin
        masses_val = [1.0, 16.0, 1.0]
        angle = deg2rad(104.5)
        origin = SVector(0.2, -0.1, 0.05)
        d21 = SVector(1.0, 0.0, 0.0)
        d23 = SVector(cos(angle), sin(angle), 0.0)
        old_coords = [
            origin + bond_length * d21,
            origin,
            origin + bond_length * d23,
        ]
        unconstrained = old_coords .+ [
            SVector(0.0005, -0.0004, 0.0002),
            SVector(-0.0002, 0.0003, -0.0001),
            SVector(-0.0004, 0.0005, 0.0003),
        ]
        ac = [AngleConstraint(1, 2, 3, angle, bond_length, bond_length)]
        sys, cons = make_shake_system(unconstrained, zero.(old_coords), masses_val;
                                      angle_constraints=ac)
        buffers = Molly.init_buffers!(sys, 1)
        step_n = 35
        Molly.clear_constraint_virial!(buffers, sys, step_n)
        context = shake_position_context(buffers; step_n, virial_scale=inv(dt^2))

        apply_position_constraints!(sys, cons, old_coords; context)
        expected = expected_position_constraint_virial(old_coords, unconstrained, sys.coords,
                                                       masses_val, dt, sys.boundary, cons)

        @test check_position_constraints(sys, cons)
        @test buffers.constraint_virial_nounits ≈ expected atol=1e-12
    end

end

@testset "SHAKE_RATTLE GPU CPU agreement" begin
    T = Float32
    bond_length = T(0.15)
    boundary = CubicBoundary(T(5.0))
    dt = T(0.5)

    function make_shake(dist_constraints, angle_constraints, natoms)
        return SHAKE_RATTLE(
            natoms, T(1e-5), T(1e-5);
            dist_constraints=dist_constraints,
            angle_constraints=angle_constraints,
            max_iters=50,
        )
    end

    function make_shake_cpu_system(coords, velocities, masses_val;
                                   dist_constraints=nothing, angle_constraints=nothing)
        atoms = [Atom(mass=m, σ=zero(T), ϵ=zero(T)) for m in masses_val]
        return System(
            atoms=atoms,
            coords=copy(coords),
            velocities=copy(velocities),
            boundary=boundary,
            constraints=(make_shake(dist_constraints, angle_constraints, length(coords)),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    function make_shake_gpu_system(coords, velocities, masses_val, AT;
                                   dist_constraints=nothing, angle_constraints=nothing)
        atoms = [Atom(mass=m, σ=zero(T), ϵ=zero(T)) for m in masses_val]
        return System(
            atoms=to_device(atoms, AT),
            coords=to_device(copy(coords), AT),
            velocities=to_device(copy(velocities), AT),
            boundary=boundary,
            constraints=(make_shake(dist_constraints, angle_constraints, length(coords)),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    function test_shake_gpu_position!(old_coords, unconstrained_coords, masses_val, AT,
                                      step_n; dist_constraints=nothing,
                                      angle_constraints=nothing)
        zero_velocities = zero.(old_coords)
        cpu_sys = make_shake_cpu_system(
            unconstrained_coords, zero_velocities, masses_val;
            dist_constraints, angle_constraints,
        )
        gpu_sys = make_shake_gpu_system(
            unconstrained_coords, zero_velocities, masses_val, AT;
            dist_constraints, angle_constraints,
        )
        cpu_buffers = Molly.init_buffers!(cpu_sys, 1)
        gpu_buffers = Molly.init_buffers!(gpu_sys, 1)
        Molly.clear_constraint_virial!(cpu_buffers, cpu_sys, step_n)
        Molly.clear_constraint_virial!(gpu_buffers, gpu_sys, step_n)
        cpu_context = shake_position_context(cpu_buffers; step_n,
                                             virial_scale=inv(dt^2))
        gpu_context = shake_position_context(gpu_buffers; step_n,
                                             virial_scale=inv(dt^2))

        apply_position_constraints!(cpu_sys, cpu_sys.constraints[1], old_coords;
                                    context=cpu_context)
        apply_position_constraints!(gpu_sys, gpu_sys.constraints[1], to_device(old_coords, AT);
                                    context=gpu_context)

        @test from_device(gpu_sys.coords) ≈ cpu_sys.coords atol=T(1e-5)
        @test from_device(gpu_buffers.constraint_virial_nounits) ≈
              cpu_buffers.constraint_virial_nounits atol=T(1e-4)
        @test abs(cpu_buffers.constraint_virial_nounits[1, 2]) > T(1e-7)
        @test Molly.has_constraint_virial(gpu_buffers, step_n)
    end

    function test_shake_gpu_velocity!(coords, velocities, masses_val, AT, step_n;
                                      dist_constraints=nothing, angle_constraints=nothing)
        cpu_sys = make_shake_cpu_system(
            coords, velocities, masses_val;
            dist_constraints, angle_constraints,
        )
        gpu_sys = make_shake_gpu_system(
            coords, velocities, masses_val, AT;
            dist_constraints, angle_constraints,
        )
        cpu_buffers = Molly.init_buffers!(cpu_sys, 1)
        gpu_buffers = Molly.init_buffers!(gpu_sys, 1)
        Molly.clear_constraint_virial!(cpu_buffers, cpu_sys, step_n)
        Molly.clear_constraint_virial!(gpu_buffers, gpu_sys, step_n)
        cpu_context = shake_velocity_context(cpu_buffers; step_n, virial_scale=inv(dt))
        gpu_context = shake_velocity_context(gpu_buffers; step_n, virial_scale=inv(dt))

        apply_velocity_constraints!(cpu_sys, cpu_sys.constraints[1]; context=cpu_context)
        apply_velocity_constraints!(gpu_sys, gpu_sys.constraints[1]; context=gpu_context)

        @test from_device(gpu_sys.velocities) ≈ cpu_sys.velocities atol=T(1e-5)
        @test from_device(gpu_buffers.constraint_virial_nounits) ≈
              cpu_buffers.constraint_virial_nounits atol=T(1e-4)
        @test abs(cpu_buffers.constraint_virial_nounits[1, 2]) > T(1e-7)
        @test Molly.has_constraint_virial(gpu_buffers, step_n)
    end

    for AT in array_list[2:end]
        @testset "$(AT)" begin
            direction = normalize(SVector{3, T}(1.0, 2.0, -1.0))
            origin = SVector{3, T}(0.2, -0.1, 0.05)
            masses_2 = T[12.0, 16.0]
            old_2 = [origin, origin + bond_length * direction]
            unconstrained_2 = [old_2[1], old_2[2] + T(0.001) * direction]
            dc_2 = [DistanceConstraint(1, 2, bond_length)]
            velocities_2 = [
                SVector{3, T}(-0.1, 0.0, 0.0),
                SVector{3, T}(0.1, 0.0, 0.0),
            ]
            test_shake_gpu_velocity!(
                old_2, velocities_2, masses_2, AT, 39;
                dist_constraints=dc_2,
            )

            masses_angle = T[1.0, 16.0, 1.0]
            angle = T(deg2rad(104.5))
            d21 = SVector{3, T}(1.0, 0.0, 0.0)
            d23 = SVector{3, T}(cos(angle), sin(angle), 0.0)
            old_angle = [
                origin + bond_length * d21,
                origin,
                origin + bond_length * d23,
            ]
            unconstrained_angle = old_angle .+ [
                SVector{3, T}(0.0005, -0.0004, 0.0002),
                SVector{3, T}(-0.0002, 0.0003, -0.0001),
                SVector{3, T}(-0.0004, 0.0005, 0.0003),
            ]
            ac = [AngleConstraint(1, 2, 3, angle, bond_length, bond_length)]
            test_shake_gpu_position!(
                old_angle, unconstrained_angle, masses_angle, AT, 42;
                angle_constraints=ac,
            )

            gpu_sys = make_shake_gpu_system(
                unconstrained_2, zero.(old_2), masses_2, AT;
                dist_constraints=dc_2,
            )
            gpu_buffers = Molly.init_buffers!(gpu_sys, 1)
            sentinel = T(3.0)
            fill!(gpu_buffers.constraint_virial_nounits, sentinel)
            position_context = shake_position_context(gpu_buffers; needs_virial=false)
            apply_position_constraints!(
                gpu_sys, gpu_sys.constraints[1], to_device(old_2, AT);
                context=position_context,
            )
            velocity_context = shake_velocity_context(gpu_buffers; needs_virial=false)
            apply_velocity_constraints!(gpu_sys, gpu_sys.constraints[1];
                                        context=velocity_context)
            @test all(==(sentinel), from_device(gpu_buffers.constraint_virial_nounits))
        end
    end
end

mutable struct VirialSnapshotScalingCoupler
    scale::Float64
    n_steps::Int
    virial::Any
    kin_tensor::Any
    volume::Any
    snapshot_seen::Bool
end

VirialSnapshotScalingCoupler(scale; n_steps=1) =
    VirialSnapshotScalingCoupler(scale, n_steps, nothing, nothing, nothing, false)

Molly.needs_virial(c::VirialSnapshotScalingCoupler) = c.n_steps

function Molly.apply_coupling!(sys, buffers, c::VirialSnapshotScalingCoupler, sim,
                               neighbors, step_n; kwargs...)
    step_n % c.n_steps == 0 || return false
    c.snapshot_seen = Molly.has_pre_coupling_virial(buffers, step_n)
    c.virial = copy(buffers.virial)
    c.kin_tensor = copy(buffers.kin_tensor)
    c.volume = volume(sys.boundary)
    s = c.scale
    scale_coords!(sys, SMatrix{3, 3, Float64}(s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s))
    return true
end

mutable struct CustomVirialSnapshotLogger
    n_steps::Int
    history::Vector{Bool}
end

Base.values(logger::CustomVirialSnapshotLogger) = logger.history

Molly.logger_virial_interval(logger::CustomVirialSnapshotLogger) = logger.n_steps

function Molly.log_property!(logger::CustomVirialSnapshotLogger, sys, neighbors, step_n,
                             buffers; kwargs...)
    if step_n % logger.n_steps == 0
        push!(logger.history, Molly.has_pre_coupling_virial(buffers, step_n))
    end
end

@testset "Constraint virial simulator integration" begin
    dt = 0.5
    bond_length = 0.15
    direction = normalize(SVector(1.0, 2.0, -1.0))
    tangent = normalize(SVector(direction[2], -direction[1], 0.0))
    origin = SVector(0.2, -0.1, 0.05)
    coords = [origin, origin + bond_length * direction]
    velocities = [
        -0.01 * direction + 0.03 * tangent,
         0.01 * direction - 0.03 * tangent,
    ]
    masses_val = [12.0, 12.0]
    atoms = [Atom(mass=m, σ=0.0, ϵ=0.0) for m in masses_val]
    dc = [DistanceConstraint(1, 2, bond_length)]

    function simulator_constraint(kind)
        if kind == :shake
            return SHAKE_RATTLE(2, 1e-10, 1e-10; dist_constraints=dc, max_iters=50)
        elseif kind == :lincs
            return LINCS(
                masses=masses_val,
                dist_constraints=dc,
                dist_tolerance=1e-10,
                vel_tolerance=1e-10,
                iter_vel_correction=true,
            )
        end
        error("unknown constraint kind $kind")
    end

    function simulator_constraint_system(kind; loggers, coords_in=coords, velocities_in=velocities)
        return System(
            atoms=copy(atoms),
            coords=copy(coords_in),
            velocities=copy(velocities_in),
            boundary=CubicBoundary(5.0),
            constraints=(simulator_constraint(kind),),
            loggers=loggers,
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    simulators = (
        VelocityVerlet(dt=dt, remove_CM_motion=0),
        DPDVelocityVerlet(dt=dt, remove_CM_motion=0),
        Verlet(dt=dt, remove_CM_motion=0),
        StormerVerlet(dt=dt),
        Langevin(dt=dt, temperature=1.0, friction=0.0, remove_CM_motion=0),
    )

    @testset "Initial constrained virial preview" begin
        for simulator in simulators
            sys = simulator_constraint_system(
                :shake;
                loggers=(virial=VirialLogger(Matrix{Float64}, 1),),
            )
            coords_before = wrap_coords.(sys.coords, (sys.boundary,))
            velocities_before = copy(sys.velocities)

            simulate!(sys, simulator, 0; n_threads=1, rng=MersenneTwister(1234))

            initial_virial = only(values(sys.loggers.virial))
            @test maximum(abs.(initial_virial)) > 1e-8
            @test sys.coords == coords_before
            if simulator isa DPDVelocityVerlet
                @test check_velocity_constraints(sys)
            else
                @test sys.velocities == velocities_before
            end
        end

        sys = simulator_constraint_system(
            :lincs;
            loggers=(virial=VirialLogger(Matrix{Float64}, 1),),
        )
        simulate!(sys, VelocityVerlet(dt=dt, remove_CM_motion=0), 0; n_threads=1)
        @test maximum(abs.(only(values(sys.loggers.virial)))) > 1e-8
    end

    @testset "Public constrained virial and pressure use preview" begin
        sys = simulator_constraint_system(:shake; loggers=())
        coords_before = copy(sys.coords)
        velocities_before = copy(sys.velocities)

        initial_virial = virial(sys; n_threads=1)
        initial_pressure = pressure(sys; n_threads=1)

        @test maximum(abs.(initial_virial)) > 1e-8
        @test all(isfinite, initial_pressure)
        @test sys.coords == coords_before
        @test sys.velocities == velocities_before

        @test scalar_virial(sys; n_threads=1) ≈ tr(initial_virial)
        @test scalar_pressure(sys; n_threads=1) ≈ tr(initial_pressure) / 3
    end

    for simulator in simulators
        sys = simulator_constraint_system(
            :shake;
            loggers=(virial=VirialLogger(Matrix{Float64}, 1),),
        )

        simulate!(sys, simulator, 1; run_loggers=:skipzero, n_threads=1,
                  rng=MersenneTwister(1234))

        @test maximum(abs.(only(values(sys.loggers.virial)))) > 1e-8
    end

    @testset "VelocityVerlet default LINCS final velocity virial" begin
        sys = System(
            atoms=copy(atoms),
            coords=copy(coords),
            velocities=copy(velocities),
            boundary=CubicBoundary(5.0),
            constraints=(LINCS(
                masses=masses_val,
                dist_constraints=dc,
                dist_tolerance=1e-10,
                vel_tolerance=1e-10,
            ),),
            loggers=(virial=VirialLogger(Matrix{Float64}, 1),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )

        simulate!(sys, VelocityVerlet(dt=dt, remove_CM_motion=0), 1;
                  run_loggers=:skipzero, n_threads=1)

        @test check_velocity_constraints(sys)
        @test maximum(abs.(only(values(sys.loggers.virial)))) > 1e-8
    end

    sys = simulator_constraint_system(
        :shake;
        loggers=(
            pressure=PressureLogger(Matrix{Float64}, 1),
            scalar_pressure=ScalarPressureLogger(Float64, 1),
        ),
    )

    simulate!(sys, VelocityVerlet(dt=dt, remove_CM_motion=0), 1;
              run_loggers=:skipzero, n_threads=1)

    logged_pressure = only(values(sys.loggers.pressure))
    logged_scalar_pressure = only(values(sys.loggers.scalar_pressure))
    @test all(isfinite, logged_pressure)
    @test logged_scalar_pressure ≈ tr(logged_pressure) / 3

    @testset "Constrained loggers use pre-coupling virial after scaling" begin
        coupler = VirialSnapshotScalingCoupler(1.5)
        sys = simulator_constraint_system(
            :shake;
            loggers=(
                virial=VirialLogger(Matrix{Float64}, 1),
                scalar_virial=ScalarVirialLogger(Float64, 1),
                pressure=PressureLogger(Matrix{Float64}, 1),
                scalar_pressure=ScalarPressureLogger(Float64, 1),
            ),
        )

        simulate!(sys, VelocityVerlet(dt=dt, coupling=(coupler,), remove_CM_motion=0), 1;
                  run_loggers=:skipzero, n_threads=1)

        expected_pressure = (2 .* coupler.kin_tensor .+ coupler.virial) ./ coupler.volume
        post_coupling_pressure = (2 .* coupler.kin_tensor .+ coupler.virial) ./
                                 volume(sys.boundary)

        @test coupler.snapshot_seen
        @test only(values(sys.loggers.virial)) ≈ coupler.virial
        @test only(values(sys.loggers.scalar_virial)) ≈ tr(coupler.virial)
        @test only(values(sys.loggers.pressure)) ≈ expected_pressure
        @test only(values(sys.loggers.scalar_pressure)) ≈ tr(expected_pressure) / 3
        @test !isapprox(
            only(values(sys.loggers.pressure)), post_coupling_pressure; rtol=1e-8, atol=1e-8)
    end

    @testset "Pre-coupling snapshot is skipped without virial loggers" begin
        coupler = VirialSnapshotScalingCoupler(1.5)
        snapshot_observable(sys, neighbors, step_n, buffers; kwargs...) =
            Molly.has_pre_coupling_virial(buffers, step_n)
        sys = simulator_constraint_system(
            :shake;
            loggers=(
                snapshot=GeneralObservableLogger(snapshot_observable, Bool, 1),
            ),
        )

        simulate!(sys, VelocityVerlet(dt=dt, coupling=(coupler,), remove_CM_motion=0), 1;
                  run_loggers=:skipzero, n_threads=1)

        @test !coupler.snapshot_seen
        @test only(values(sys.loggers.snapshot)) == false
    end

    @testset "Custom virial logger requests pre-coupling snapshot" begin
        coupler = VirialSnapshotScalingCoupler(1.5)
        sys = simulator_constraint_system(
            :shake;
            loggers=(snapshot=CustomVirialSnapshotLogger(1, Bool[]),),
        )

        simulate!(sys, VelocityVerlet(dt=dt, coupling=(coupler,), remove_CM_motion=0), 1;
                  run_loggers=:skipzero, n_threads=1)

        @test coupler.snapshot_seen
        @test only(values(sys.loggers.snapshot)) == true
    end

    @testset "Average pressure logger requests pre-coupling pressure" begin
        coupler = VirialSnapshotScalingCoupler(1.5)
        sys = simulator_constraint_system(
            :shake;
            loggers=(
                pressure=AverageObservableLogger(Molly.pressure_wrapper, Matrix{Float64}, 1),
            ),
        )

        simulate!(sys, VelocityVerlet(dt=dt, coupling=(coupler,), remove_CM_motion=0), 1;
                  run_loggers=:skipzero, n_threads=1)

        expected_pressure = (2 .* coupler.kin_tensor .+ coupler.virial) ./ coupler.volume
        @test coupler.snapshot_seen
        @test only(sys.loggers.pressure.block_averages) ≈ expected_pressure
    end

    barostat = BerendsenBarostat(1.0, 1.0; compressibility=1.0, n_steps=1,
                                 max_scale_frac=0.001)
    sys = simulator_constraint_system(
        :shake;
        loggers=(pressure=PressureLogger(Matrix{Float64}, 1),),
    )

    simulate!(sys, VelocityVerlet(dt=dt, coupling=(barostat,), remove_CM_motion=0), 1;
              run_loggers=:skipzero, n_threads=1)

    @test length(values(sys.loggers.pressure)) == 1

end

# --- LINCS tests ---

@testset "LINCS setup" begin
    dc = [DistanceConstraint(1, 2, 0.15), DistanceConstraint(2, 3, 0.15)]
    data = Molly.build_lincs_data(dc, [12.0, 12.0, 12.0])

    @test data.coupling.range == [1, 2, 3]
    @test data.coupling.neighbors == [2, 1]
    expected_coef = (1 / 12.0) * data.sdiag[1] * data.sdiag[2]
    @test data.coupling.coef ≈ [expected_coef, expected_coef]
end

@testset "LINCS solver" begin
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
            Molly.lincs_apply!(xp, x, data, ws, CubicBoundary(5.0),
                               lincs_no_virial_context())

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
        Molly.lincs_apply!(xp, x, data, ws, CubicBoundary(5.0),
                           lincs_no_virial_context())
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
        Molly.lincs_apply!(xp, x, data, ws, CubicBoundary(5.0),
                           lincs_no_virial_context())

        @test lincs_check_constraints(xp, data; atol=1e-6)
    end

    # position constraint virial for a stretched diatomic
    @testset begin
        bond_length = 0.15
        x, _, data, ws = make_lincs_diatomic(; bond_length, nrec=0, niter=0)
        xp = [x[1], x[2] + SVector(0.001, 0.0, 0.0)]
        sys, buffers = make_lincs_no_unit_buffer_system(x, [12.0, 12.0])
        step_n = 7
        Molly.clear_constraint_virial!(buffers, sys, step_n)
        context = lincs_position_context(buffers; step_n)

        B = normalize(x[1] - x[2])
        factor = data.sdiag[1]^2 * (dot(B, xp[1] - xp[2]) - bond_length)
        expected = -bond_length * factor * (B * transpose(B))

        Molly.lincs_apply!(xp, x, data, ws, sys.boundary, context)

        @test norm(xp[1] - xp[2]) ≈ bond_length atol=1e-14
        @test ws.factor_sum[1] ≈ factor atol=1e-14
        @test buffers.constraint_virial_nounits ≈ Matrix(expected) atol=1e-14
        @test Molly.has_constraint_virial(buffers, step_n)
    end

    # unitful position virial is independent of the timestep unit spelling
    @testset begin
        function lincs_unitful_position_virial(dt)
            bond_length = 0.15f0u"nm"
            masses_val = [12.0f0u"g/mol", 12.0f0u"g/mol"]
            dc = [DistanceConstraint(1, 2, bond_length)]
            lincs = LINCS(masses=masses_val, dist_constraints=dc,
                          dist_tolerance=1.0f-8u"nm",
                          vel_tolerance=1.0f-8u"nm^2 * ps^-1",
                          nrec=0, niter=0)
            old_coords = [
                SVector(0.0f0, 0.0f0, 0.0f0)u"nm",
                SVector(0.15f0, 0.0f0, 0.0f0)u"nm",
            ]
            coords = [
                old_coords[1],
                old_coords[2] + SVector(0.001f0, 0.0f0, 0.0f0)u"nm",
            ]
            atoms = [Atom(mass=m, σ=0.0f0u"nm", ϵ=0.0f0u"kJ/mol")
                     for m in masses_val]
            sys = System(
                atoms=atoms,
                coords=copy(coords),
                velocities=zero.(coords) ./ 1.0f0u"ps",
                boundary=CubicBoundary(5.0f0u"nm"),
                constraints=(lincs,),
                force_units=u"kJ * mol^-1 * nm^-1",
                energy_units=u"kJ * mol^-1",
            )
            buffers = Molly.init_buffers!(sys, 1)
            step_n = 9
            Molly.clear_constraint_virial!(buffers, sys, step_n)
            context = Molly.position_constraint_context(buffers, sys, step_n, dt, true)

            apply_position_constraints!(sys, lincs, old_coords; context)

            return copy(buffers.constraint_virial_nounits) .* u"kJ/mol"
        end

        virial_fs = lincs_unitful_position_virial(4.0f0u"fs")
        virial_ps = lincs_unitful_position_virial(0.004f0u"ps")

        @test virial_fs ≈ virial_ps rtol=1e-6
        @test virial_ps[1, 1] ≈ -56.25f0u"kJ/mol" rtol=1e-5
        @test all(iszero, virial_ps[2:3, :])
        @test all(iszero, virial_ps[:, 2:3])
    end

    # velocity constraint virial for a separating diatomic
    @testset begin
        bond_length = 0.15
        x, _, data, ws = make_lincs_diatomic(; bond_length, nrec=0, niter=0)
        velocities = [SVector(-0.1, 0.0, 0.0), SVector(0.1, 0.0, 0.0)]
        sys, buffers = make_lincs_no_unit_buffer_system(x, [12.0, 12.0])
        step_n = 10
        Molly.clear_constraint_virial!(buffers, sys, step_n)
        context = lincs_velocity_context(buffers; step_n)

        B = normalize(x[1] - x[2])
        factor = data.sdiag[1]^2 * dot(B, velocities[1] - velocities[2])
        expected = -bond_length * factor * (B * transpose(B))

        Molly.lincs_vel_apply!(velocities, x, data, ws, sys.boundary, context)

        @test dot(B, velocities[2] - velocities[1]) ≈ 0.0 atol=1e-14
        @test ws.factor_sum[1] ≈ factor atol=1e-14
        @test buffers.constraint_virial_nounits ≈ Matrix(expected) atol=1e-14
        @test Molly.has_constraint_virial(buffers, step_n)
    end

    # needs_virial=false leaves the velocity constraint virial buffer untouched
    @testset begin
        x, _, data, ws = make_lincs_diatomic(; nrec=0, niter=0)
        velocities = [SVector(-0.1, 0.0, 0.0), SVector(0.1, 0.0, 0.0)]
        _, buffers = make_lincs_no_unit_buffer_system(x, [12.0, 12.0])
        fill!(buffers.constraint_virial_nounits, 3.0)
        context = lincs_velocity_context(buffers; needs_virial=false)

        Molly.lincs_vel_apply!(velocities, x, data, ws, CubicBoundary(5.0), context)

        @test all(==(3.0), buffers.constraint_virial_nounits)
    end

    # high-level position constraint application computes and merges scaled virial
    @testset begin
        bond_length = 0.15
        masses_val = [12.0, 12.0]
        dc = [DistanceConstraint(1, 2, bond_length)]
        lincs = LINCS(masses=masses_val, dist_constraints=dc,
                      dist_tolerance=1e-8, vel_tolerance=1e-8,
                      nrec=0, niter=0)
        old_coords = [SVector(0.0, 0.0, 0.0), SVector(bond_length, 0.0, 0.0)]
        coords = [old_coords[1], old_coords[2] + SVector(0.001, 0.0, 0.0)]
        atoms = [Atom(mass=m, σ=0.0, ϵ=0.0) for m in masses_val]
        sys = System(
            atoms=atoms,
            coords=copy(coords),
            velocities=zero.(coords),
            boundary=CubicBoundary(5.0),
            constraints=(lincs,),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
        buffers = Molly.init_buffers!(sys, 1)
        step_n = 12
        scale = 2.0
        interaction_virial = [
            1.0 0.2 0.0
            0.2 0.5 0.1
            0.0 0.1 0.25
        ]

        Molly.clear_constraint_virial!(buffers, sys, step_n)
        buffers.virial .= interaction_virial
        Molly.mark_interaction_virial!(buffers.validity, step_n)
        context = lincs_position_context(buffers; step_n, virial_scale=scale)

        B = normalize(old_coords[1] - old_coords[2])
        factor = lincs.lincs_data.sdiag[1]^2 *
                 (dot(B, coords[1] - coords[2]) - bond_length)
        expected_constraint = scale * (-bond_length * factor * (B * transpose(B)))

        apply_position_constraints!(sys, lincs, old_coords; context)
        Molly.merge_constraint_virial!(buffers, sys, step_n)

        @test norm(sys.coords[1] - sys.coords[2]) ≈ bond_length atol=1e-14
        @test buffers.constraint_virial_nounits ≈ Matrix(expected_constraint) atol=1e-14
        @test buffers.constraint_virial ≈ Matrix(expected_constraint) atol=1e-14
        @test buffers.virial ≈ interaction_virial + Matrix(expected_constraint) atol=1e-14
        @test Molly.has_total_virial(buffers, step_n)
    end

    # niter > 1 improves accuracy
    @testset begin
        v = [
            SVector(0.5, 0.2, 0.0),
            SVector(-0.3, 0.1, 0.0),
            SVector(0.1, -0.2, 0.0),
        ]
        dt = 0.002

        deviations = Float64[]
        for niter in [1, 2, 4]
            x, _, data, ws = make_lincs_triatomic(; v, niter)
            xp = x .+ v .* dt
            Molly.lincs_apply!(xp, x, data, ws, CubicBoundary(5.0),
                               lincs_no_virial_context())
            max_dev = maximum(abs(norm(xp[data.atom1[i]] - xp[data.atom2[i]]) - data.lengths[i])
                              for i in eachindex(data.atom1))
            push!(deviations, max_dev)
        end

        @test deviations[2] <= deviations[1] + 1e-15
        @test deviations[3] <= deviations[2] + 1e-15
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

@testset "LINCS GPU CPU agreement" begin
    T = Float32
    bond_length = T(0.15)
    masses_val = T[12.0, 16.0, 14.0]
    atoms = [Atom(mass=m, σ=zero(T), ϵ=zero(T)) for m in masses_val]
    boundary = CubicBoundary(T(5.0))
    d12 = normalize(SVector{3, T}(1.0, 2.0, -1.0))
    d23 = normalize(SVector{3, T}(-1.0, 1.0, 2.0))
    origin = SVector{3, T}(0.2, -0.1, 0.05)
    old_coords = [
        origin,
        origin + bond_length * d12,
        origin + bond_length * d12 + bond_length * d23,
    ]
    dist_constraints = [
        DistanceConstraint(1, 2, bond_length),
        DistanceConstraint(2, 3, bond_length),
    ]

    function make_lincs()
        return LINCS(masses=masses_val, dist_constraints=dist_constraints,
                     dist_tolerance=T(1e-5), vel_tolerance=T(1e-5),
                     nrec=4, niter=2, iter_vel_correction=true)
    end

    function make_cpu_system(coords, velocities)
        return System(
            atoms=copy(atoms),
            coords=copy(coords),
            velocities=copy(velocities),
            boundary=boundary,
            constraints=(make_lincs(),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    function make_gpu_system(coords, velocities, AT)
        return System(
            atoms=to_device(copy(atoms), AT),
            coords=to_device(copy(coords), AT),
            velocities=to_device(copy(velocities), AT),
            boundary=boundary,
            constraints=(make_lincs(),),
            force_units=NoUnits,
            energy_units=NoUnits,
        )
    end

    for AT in array_list[2:end]
        @testset "$(AT)" begin
            dt = T(0.5)

            # Position constraints and virial.
            unconstrained_coords = old_coords .+ [
                T(0.0005) * d23,
                T(-0.0007) * d12,
                T(0.0009) * (d12 - d23),
            ]
            zero_velocities = zero.(old_coords)
            cpu_sys = make_cpu_system(unconstrained_coords, zero_velocities)
            gpu_sys = make_gpu_system(unconstrained_coords, zero_velocities, AT)
            cpu_buffers = Molly.init_buffers!(cpu_sys, 1)
            gpu_buffers = Molly.init_buffers!(gpu_sys, 1)
            step_n = 21
            Molly.clear_constraint_virial!(cpu_buffers, cpu_sys, step_n)
            Molly.clear_constraint_virial!(gpu_buffers, gpu_sys, step_n)
            cpu_context = lincs_position_context(cpu_buffers; step_n,
                                                 virial_scale=inv(dt^2))
            gpu_context = lincs_position_context(gpu_buffers; step_n,
                                                 virial_scale=inv(dt^2))

            apply_position_constraints!(cpu_sys, cpu_sys.constraints[1], old_coords;
                                        context=cpu_context)
            apply_position_constraints!(gpu_sys, gpu_sys.constraints[1],
                                        to_device(old_coords, AT); context=gpu_context)

            @test from_device(gpu_sys.coords) ≈ cpu_sys.coords atol=T(1e-5)
            @test from_device(gpu_buffers.constraint_virial_nounits) ≈
                  cpu_buffers.constraint_virial_nounits atol=T(1e-4)
            @test abs(cpu_buffers.constraint_virial_nounits[1, 2]) > T(1e-7)
            @test Molly.has_constraint_virial(gpu_buffers, step_n)

            # Velocity constraints and virial.
            common_velocity = SVector{3, T}(0.04, -0.02, 0.03)
            velocities = [
                common_velocity - T(0.06) * d12,
                common_velocity + T(0.04) * d12 - T(0.05) * d23,
                common_velocity + T(0.07) * d23,
            ]
            cpu_sys = make_cpu_system(old_coords, velocities)
            gpu_sys = make_gpu_system(old_coords, velocities, AT)
            cpu_buffers = Molly.init_buffers!(cpu_sys, 1)
            gpu_buffers = Molly.init_buffers!(gpu_sys, 1)
            step_n = 22
            Molly.clear_constraint_virial!(cpu_buffers, cpu_sys, step_n)
            Molly.clear_constraint_virial!(gpu_buffers, gpu_sys, step_n)
            cpu_context = lincs_velocity_context(cpu_buffers; step_n,
                                                 virial_scale=inv(dt))
            gpu_context = lincs_velocity_context(gpu_buffers; step_n,
                                                 virial_scale=inv(dt))

            apply_velocity_constraints!(cpu_sys, cpu_sys.constraints[1];
                                        context=cpu_context)
            apply_velocity_constraints!(gpu_sys, gpu_sys.constraints[1];
                                        context=gpu_context)

            @test from_device(gpu_sys.velocities) ≈ cpu_sys.velocities atol=T(1e-5)
            @test from_device(gpu_buffers.constraint_virial_nounits) ≈
                  cpu_buffers.constraint_virial_nounits atol=T(1e-4)
            @test abs(cpu_buffers.constraint_virial_nounits[1, 2]) > T(1e-7)
            @test Molly.has_constraint_virial(gpu_buffers, step_n)

        end
    end
end

@testset "LINCS GPU block size validation" begin
    # Chain of 130 constraints sharing atoms: 1-2, 2-3, ..., 130-131
    # Forms a single connected component of 130 constraints
    block_size = 128
    atom1 = collect(1:130)
    atom2 = collect(2:131)
    @test_throws ErrorException Molly.group_constraints_for_gpu(atom1, atom2, block_size)

    # Same chain should succeed with a large enough block size
    perm = Molly.group_constraints_for_gpu(atom1, atom2, 256)
    @test length(perm) % 256 == 0
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
    Molly.lincs_apply!(xp_w, x_w, d_w, ws_w, CubicBoundary(5.0))

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
            Molly.lincs_apply!(xp, x, data, ws, CubicBoundary(5.0))
            xp .= x .+ v .* 0.002

            allocs = @allocated Molly.lincs_apply!(xp, x, data, ws, CubicBoundary(5.0))
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

@testset "Minimization with constraints" begin
    ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...)
    pe_cpu_nocons, pe_cpu_cons = 0.0u"kJ/mol", 0.0u"kJ/mol"

    for AT in array_list
        sys_nocons = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            array_type=AT,
            nonbonded_method=:pme,
        )
        sys_cons = System(
            joinpath(data_dir, "6mrr_equil.pdb"),
            ff;
            array_type=AT,
            nonbonded_method=:pme,
            constraints=:hbonds,
            rigid_water=true,
        )

        sim = SteepestDescentMinimizer()
        simulate!(sys_nocons, sim)
        simulate!(sys_cons, sim)
        pe_nocons = potential_energy(sys_nocons)
        pe_cons   = potential_energy(sys_cons)
        if AT == Array
            pe_cpu_nocons = pe_nocons
            pe_cpu_cons   = pe_cons
        else
            @test pe_nocons ≈ pe_cpu_nocons
            @test pe_cons   ≈ pe_cpu_cons
        end

        @test rmsd(sys_nocons.coords[1:1170], sys_cons.coords[1:1170]) < 0.01u"nm"
        @test pe_nocons < -150_000u"kJ/mol"
        @test pe_cons   < -150_000u"kJ/mol"
    end
end
