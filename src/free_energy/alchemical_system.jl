export
    AbsoluteFESystem,
    RelativeFESystem

### Checks for hybrid system ###
# same atoms in environment in both A and B
function check_env_alignment(sysA, sysB, env_A, env_B)
    for (iA, iB) in zip(env_A, env_B)
        dA, dB = sysA.atoms_data[iA], sysB.atoms_data[iB]
        if dA.atom_name != dB.atom_name || dA.res_name != dB.res_name
            throw(ArgumentError("environment atom $iA of system A ($(dA.res_name) $(dA.atom_name)) " *
                "is paired with atom $iB of system B ($(dB.res_name) $(dB.atom_name)); the " *
                "environment atoms of the two systems have to be the same atoms in the same order"))
        end
    end
    return nothing
end

# Env-Uniq bonds -> should be Core-Uniq
function check_unique_env_bonds(sys, unique_inds, ligand, to_A, state, host_lists)
    unique_set = Set(unique_inds)
    bonds = Tuple{Int, Int}[]
    for il in host_lists(sys)
        if il isa InteractionList2Atoms && !(eltype(il.inters) <:
                    Union{LennardJones14, LennardJones14λ, EwaldExclusion})
            append!(bonds, ((Int(i), Int(j)) for (i, j) in zip(il.is, il.js)))
        end
    end
    for ca in sys.constraints
        dist_cs, angle_cs = constraint_lists(ca)
        append!(bonds, ((Int(c.i), Int(c.j)) for c in dist_cs))
        append!(bonds, ((Int(c.i), Int(c.j)) for c in angle_cs))
        append!(bonds, ((Int(c.j), Int(c.k)) for c in angle_cs))
    end
    for (i, j) in bonds
        for (u, e) in ((i, j), (j, i))
            if u in unique_set && !(e in ligand)
                throw(ArgumentError("atom $u of system $state is unique to it but is bonded to " *
                    "the environment atom $(to_A(e)) (numbered as in system A), so it can not be " *
                    "switched off; add the environment atom to the core of the mapping and to the " *
                    "core map"))
            end
        end
    end
    return nothing
end

### Neighborfinder functions ###
function neighbor_exclusions(nf::GPUNeighborFinder)
    excluded = collect(zip(from_device(nf.excluded_i), from_device(nf.excluded_j)))
    special  = collect(zip(from_device(nf.special_i ), from_device(nf.special_j )))
    return excluded, special
end

neighbor_exclusions(nf::NoNeighborFinder) = (Tuple{Int32, Int32}[], Tuple{Int32, Int32}[])
neighbor_exclusions(nf) = dense_masks_to_pair_lists(from_device(nf.eligible), from_device(nf.special))

### Constraints ###
# Core atom can't have different constraint lengths or angles between A and B
# If detected, core atom should become a unique atom
function unmap_changing_constraints!(mapping, core_mapAB, sysA, sysB)
    core = mapping["core"]
    dists_A = constrained_distances(sysA, Set(core))
    dists_B = constrained_distances(sysB, Set(core_mapAB[i] for i in core))
    core_mapBA = Dict(core_mapAB[i] => i for i in core)
    pairs = union(keys(dists_A), (minmax(core_mapBA[x], core_mapBA[y]) for (x, y) in keys(dists_B)))
    is_H(i) = sysA.atoms_data[i].element == "H" || sysB.atoms_data[core_mapAB[i]].element == "H"
    moved = Int[]
    for (i, j) in pairs
        dist_B = get(dists_B, minmax(core_mapAB[i], core_mapAB[j]), nothing)
        get(dists_A, (i, j), nothing) == dist_B && continue
        if is_H(i)
            push!(moved, i)
        elseif is_H(j)
            push!(moved, j)
        else
            throw(ArgumentError("the constraint between core atoms $i and $j of system A changes " *
                                "between the end states and neither is a hydrogen, change the mapping"))
        end
    end
    if !allunique(moved)
        throw(ArgumentError("a core hydrogen is in several constraints that change between the " *
                            "end states, check the mapping"))
    end
    for i in sort!(moved)
        filter!(!=(i), core)
        push!(mapping["unique_A"], i)
        push!(mapping["unique_B"], core_mapAB[i])
        delete!(core_mapAB, i)
    end
    return moved
end

function constrained_distances(sys, atoms)
    dists = Dict{Tuple{Int, Int}, Any}()
    for ca in sys.constraints
        dist_cs, angle_cs = constraint_lists(ca)
        pairs = [((c.i, c.j), c.dist) for c in dist_cs]
        for c in angle_cs
            push!(pairs, ((c.i, c.j), c.dist_ij), ((c.j, c.k), c.dist_jk), ((c.i, c.k), c.dist_ik))
        end
        for ((i, j), dist) in pairs
            if i in atoms && j in atoms
                dists[minmax(Int(i), Int(j))] = dist
            end
        end
    end
    return dists
end

function constraint_lists(ca::SHAKE_RATTLE)
    dist_constraints = DistanceConstraint[]
    for clusters in (ca.clusters12, ca.clusters23, ca.clusters34)
        for cluster in from_device(clusters)
            for (i, j, dist) in cluster_interactions(cluster)
                push!(dist_constraints, DistanceConstraint(i, j, dist))
            end
        end
    end
    angle_constraints = [AngleConstraint(c.k2, c.k1, c.k3,
                                         acos((c.dist12^2 + c.dist13^2 - c.dist23^2) /
                                              (2 * c.dist12 * c.dist13)), c.dist12, c.dist13) 
                            for c in from_device(ca.angle_clusters)
                                if c isa AngleClusterData]
    return dist_constraints, angle_constraints
end

function constraint_lists(ca::LINCS)
    angle_constraints = something(ca.angle_constraints, AngleConstraint[])
    angle_atoms = Set(a for c in angle_constraints for a in (c.i, c.j, c.k))
    dist_constraints = [c for c in something(ca.dist_constraints, DistanceConstraint[])
                        if !(c.i in angle_atoms && c.j in angle_atoms)]
    return dist_constraints, angle_constraints
end

function rebuild_constraints(ca::SHAKE_RATTLE, dist_constraints, angle_constraints, masses)
    return SHAKE_RATTLE(n_atoms=length(masses), dist_tolerance=ca.dist_tolerance,
                        vel_tolerance=ca.vel_tolerance, dist_constraints=dist_constraints,
                        angle_constraints=angle_constraints, gpu_block_size=ca.gpu_block_size,
                        max_iters=ca.max_iters)
end

function rebuild_constraints(ca::LINCS, dist_constraints, angle_constraints, masses)
    return LINCS(masses=masses, dist_tolerance=ca.dist_tolerance, vel_tolerance=ca.vel_tolerance,
                 dist_constraints=dist_constraints, angle_constraints=angle_constraints,
                 n_rec=ca.lincs_data.n_rec, n_iter=ca.lincs_data.n_iter,
                 iter_vel_correction=ca.iter_vel_correction, gpu_block_size=ca.gpu_block_size)
end

### Functions to change interactions to λ versions ###
# Specific interactions
function to_lambda_inter_list(inter_list, scheduler, AT, FT, lj_sc)
    inters_cpu = from_device(inter_list.inters)
    isempty(inters_cpu) && return inter_list
    if !(eltype(inters_cpu) <: LennardJones14) && !hasmethod(to_lambda_function, Tuple{eltype(inters_cpu)})
        @warn "no to_lambda_function for $(eltype(inters_cpu)), leaving it unscaled; its " *
              "virial will not follow the alchemical coupling"
        return inter_list
    end
    new_inters = [lambda_specific(x, scheduler, FT, lj_sc) for x in inters_cpu]
    IT = typeof(inter_list).name.wrapper
    fields = getfield.((inter_list,), fieldnames(typeof(inter_list)))
    return IT(fields[1:(end - 3)]..., to_device(new_inters, AT), inter_list.types,
              inter_list.data)
end

lambda_specific(inter, scheduler, FT, lj_sc) = to_lambda_function(inter; λ_mixing=MinimumMixing(),
                                                                  scheduler=scheduler)
lambda_specific(inter::LennardJones14, scheduler, FT, lj_sc) = to_lambda_function(inter, lj_sc;
        λ_mixing=MinimumMixing(), scheduler=scheduler, float_type=FT)

lambda_single(interA, interB, scheduler, lj_sc) = to_lambda_function_single(interA, interB;
                                                                            scheduler=scheduler)
lambda_single(interA::LennardJones14, interB::Nothing, scheduler, lj_sc) =
    to_lambda_function_single(interA, interB, lj_sc; scheduler=scheduler)
lambda_single(interA::Nothing, interB::LennardJones14, scheduler, lj_sc) =
    to_lambda_function_single(interA, interB, lj_sc; scheduler=scheduler)
    
lambda_single_A(inter, scheduler, lj_sc, deleted) = lambda_single(inter, nothing, scheduler, lj_sc)
function lambda_single_A(inter::LennardJones14, scheduler, lj_sc, deleted)
    lj14 = lambda_single(inter, nothing, scheduler, lj_sc)
    return deleted ? update_lambda_function(lj14, LennardJones14(inter.σ14_mixed, zero(inter.ϵ14_mixed),
                                                                  inter.weight_14)) : lj14
end

# Pairwise interactions
const softcore_dic = Dict(:beutler => BeutlerSoftCore(),
                          :gapsys  => GapsysSoftCore(),
                          :scaled  => ScaledSoftCore())
                          
function check_softcores(LJsoftcore, Csoftcore)
    if !(LJsoftcore in (:beutler, :gapsys, :scaled))
        throw(ArgumentError("LJsoftcore is $(repr(LJsoftcore)), use :beutler, :gapsys or :scaled"))
    end
    if !(Csoftcore in (:beutler, :gapsys, :scaled))
        throw(ArgumentError("Csoftcore is $(repr(Csoftcore)), use :beutler, :gapsys or :scaled"))
    end
end

function lambda_pairwise_inters(sys, scheduler, LJsoftcore, Csoftcore, FT)
    return map(sys.pairwise_inters) do inter
        softcore = softcore_dic[inter isa LennardJones ? LJsoftcore : Csoftcore]
        to_lambda_function(inter, softcore; scheduler=scheduler, λ_mixing=MinimumMixing(),
                           float_type=FT)
    end
end

# General interactions
function lambda_general_inters(sys, atoms, boundary, scheduler, global_λ, two_grid)
    FT, AT = float_type(sys), array_type(sys)
    inters = []
    for inter in sys.general_inters
        if inter isa PME && two_grid
            push!(inters, PME_λ(inter.dist_cutoff, to_device(atoms, AT), boundary;
                                grad_safe=inter.grad_safe, error_tol=inter.error_tol,
                                fixed_charges=false, scheduler=scheduler, λ=FT(global_λ)))
        elseif inter isa PME
            push!(inters, PME(inter.dist_cutoff, to_device(atoms, AT), boundary;
                              grad_safe=inter.grad_safe, error_tol=inter.error_tol,
                              fixed_charges=false, scheduler=scheduler))
        elseif inter isa LJDispersionCorrection
            push!(inters, LJDispersionCorrectionλ(to_device(atoms, AT), inter.dist_cutoff,
                                                  scheduler, MinimumMixing(), inter.σ_mix,
                                                  inter.ϵ_mix))
        else
            @warn "Currently $inter is not implemented for alchemical simulations"
        end
    end
    return tuple(inters...)
end

### General functions for settings and setup ###
function alchemical_settings(sys, temp)
    FT = float_type(sys)
    units = (sys.energy_units != NoUnits)
    temp = (units ? FT(ustrip(u"K", temp))u"K" : FT(ustrip(temp)))
    grad_safe = sys.grad_safe
    return FT, units, temp, array_type(sys), grad_safe
end

function random_velocity_or_zero(mass, temp)
    return iszero(mass) ? zero(random_velocity(oneunit(mass), temp)) : random_velocity(mass, temp)
end

remap_virtual_site(v::VirtualSite, m) = VirtualSite(v.type, m(v.atom_ind), m(v.atom_1),
        (v.atom_2 == 0 ? 0 : m(v.atom_2)), (v.atom_3 == 0 ? 0 : m(v.atom_3)), v.weight_1,
        v.weight_2, v.weight_3, v.weight_12, v.weight_13, v.weight_cross)

function alchemical_system(sys_ref, atoms, coords, data, boundary, temp; kwargs...)
    AT = array_type(sys_ref)
    velocities = [random_velocity_or_zero(a.mass, temp) for a in atoms]
    return System(; atoms=to_device(atoms, AT), coords=to_device(coords, AT), atoms_data=data,
                  boundary=boundary, velocities=to_device(velocities, AT),
                  force_units=sys_ref.force_units, energy_units=sys_ref.energy_units, 
                  grad_safe=sys_ref.grad_safe, kwargs...)
end

"""
    AbsoluteFESystem(sys, global_λ, mapping; temp=298.0u"K",
                     scheduler=DefaultLambdaScheduler(dual=true), loggers=(),
                     LJsoftcore=:gapsys, Csoftcore=:gapsys)

Set up an alchemical system for an absolute free energy calculation, in which the atoms in
`mapping` are decoupled from the rest of `sys`.

The mapped atoms are fully coupled at `global_λ = 0` and fully decoupled at `global_λ = 1`.
The atom order of `sys` is kept. Interactions are replaced by their λ-dependent versions and
random velocities are generated at `temp`.

# Arguments
- `sys`: the [`System`](@ref) containing the atoms to decouple.
- `global_λ`: the global λ of the system, from 0 (coupled) to 1 (decoupled).
- `mapping`: a vector with the indices of the atoms to decouple.
- `temp=298.0u"K"`: the temperature used to generate random velocities.
- `scheduler=DefaultLambdaScheduler(dual=true)`: the lambda scheduler that turns `global_λ`
    into the couplings of the steric, electrostatic and bonded interactions, for example
    [`DefaultLambdaScheduler`](@ref), [`GROMACSLambdaABFEScheduler`](@ref) or
    [`OpenFEScheduler`](@ref). Absolute systems require dual topology (`dual=true`).
- `loggers=()`: the loggers that record properties of interest during a simulation.
- `LJsoftcore=:gapsys`: the Lennard-Jones soft core, `:beutler`
    ([Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1)), `:gapsys`
    ([Gapsys et al. 2012](https://doi.org/10.1021/ct300220p)) or `:scaled`, where the potential
    is scaled directly by λ. The 1-4 Lennard-Jones interactions of force fields with separate
    1-4 parameters (e.g. CHARMM) use the same soft core.
- `Csoftcore=:gapsys`: the Coulomb soft core, `:beutler`, `:gapsys` or `:scaled`, where the
    potential is scaled directly by λ.
"""
function AbsoluteFESystem(sys::System, global_λ, mapping; 
                        temp = 298.0u"K", 
                        scheduler=DefaultLambdaScheduler(dual=true),
                        loggers=(),
                        LJsoftcore=:gapsys,
                        Csoftcore=:gapsys
                        )
    FT, units, temp, AT, grad_safe = alchemical_settings(sys, temp)
    check_softcores(LJsoftcore, Csoftcore)
    lj_sc = softcore_dic[LJsoftcore]
    sys_atoms  = from_device(sys.atoms)
    sys_coords = from_device(sys.coords)

    if !scheduler.dual
        throw(ArgumentError("absolute free energy systems require dual topology, " *
                            "use a scheduler with dual=true"))
    end

    # Initialize data groups for new system
    Atoms        = []
    Data         = []
    Coords       = []
    Interactions = []
    Boundary = deepcopy(sys.boundary)

    # The atom order is kept: the mapped atoms are decoupled, the others are the environment
    for i in 1:length(sys_atoms)
        a = sys_atoms[i]
        if i in mapping
            push!(Atoms, Atom(index=i, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                λ=FT(global_λ), alch_role=DeleteRole))
        else
            push!(Atoms, Atom(index=i, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                λ=FT(1.0), alch_role=EnvRole))
        end
        push!(Data, sys.atoms_data[i])
        push!(Coords, sys_coords[i])
    end

    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)

    pairwise_inters = lambda_pairwise_inters(sys, scheduler, LJsoftcore, Csoftcore, FT)
    general_inters = lambda_general_inters(sys, Atoms, Boundary, scheduler, global_λ,
                                           scheduler isa GROMACSLambdaABFEScheduler)
    SpecificInteraction = Any[]
    for inter_list in sys.specific_inter_lists
        if inter_list isa InteractionList2Atoms && inter_list.data isa EwaldExclusionData
            d = inter_list.data
            push!(SpecificInteraction, InteractionList2Atoms(
                inter_list.is, inter_list.js, inter_list.inters, inter_list.types,
                EwaldExclusionData(d.dist_cutoff; error_tol=d.error_tol, ϵr=d.ϵr,
                                   scheduler=scheduler, λ_mix=d.λ_mixing),
            ))
        else
            push!(SpecificInteraction, to_lambda_inter_list(inter_list, scheduler, AT, FT, lj_sc))
        end
    end

    return alchemical_system(sys, Atoms, Coords, Data, Boundary, temp;
        topology=sys.topology,
        virtual_sites=to_device(sys.virtual_sites, AT),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=to_device.(tuple(SpecificInteraction...), AT),
        neighbor_finder=sys.neighbor_finder,
        constraints=sys.constraints,
        general_inters=general_inters,
        loggers=loggers,
    )
end

"""
    RelativeFESystem(sysA, sysB, global_λ, mapping, core_mapAB; temp=298.0u"K",
                     scheduler=DefaultLambdaScheduler(dual=true), loggers=(),
                     LJsoftcore=:gapsys, Csoftcore=:gapsys)

Set up a hybrid system for a relative free energy calculation that transforms system A
(`global_λ = 0`) into system B (`global_λ = 1`).

Core atoms are interpolated between their parameters in A and B, unique A atoms are decoupled
and unique B atoms are coupled. The environment, every atom of `sysA` that is neither core nor
unique A, is taken from `sysA`. The atoms of the returned system are ordered as core, unique A,
unique B and environment. In dual topology each core atom is added twice, with the parameters
of A and as a massless copy with the parameters of B. Interactions are replaced by their
λ-dependent versions and random velocities are generated at `temp`.

# Arguments
- `sysA`: the [`System`](@ref) of state A, which also provides the environment.
- `sysB`: the [`System`](@ref) of state B, which provides the unique B atoms and the B
    parameters of the core atoms.
- `global_λ`: the global λ of the system, from 0 (state A) to 1 (state B).
- `mapping`: a dictionary of atom index vectors, where `"core"` and `"unique_A"` index into
    `sysA` and `"unique_B"` into `sysB`. An `"env"` entry is added to it. An atom unique to one
    end state can not be bonded to an atom outside the mapping, as it is switched off at the
    other end state; such an atom belongs in `"core"`, where only its parameters change, which
    is how a covalently bound ligand is set up.
- `core_mapAB`: a dictionary from each core atom index in `sysA` to its index in `sysB`.
- `temp=298.0u"K"`: the temperature used to generate random velocities.
- `scheduler=DefaultLambdaScheduler(dual=true)`: the lambda scheduler that turns `global_λ`
    into the couplings of the steric, electrostatic and bonded interactions, for example
    [`DefaultLambdaScheduler`](@ref), [`GROMACSLambdaRBFEScheduler`](@ref) or
    [`OpenFEScheduler`](@ref). `dual=true` uses dual topology (energy scaling) and `dual=false`
    single topology (parameter scaling), which does not support `intraLJ=true`.
- `loggers=()`: the loggers that record properties of interest during a simulation.
- `LJsoftcore=:gapsys`: the Lennard-Jones soft core, `:beutler`
    ([Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1)), `:gapsys`
    ([Gapsys et al. 2012](https://doi.org/10.1021/ct300220p)) or `:scaled`, where the potential
    is scaled directly by λ. The 1-4 Lennard-Jones interactions of force fields with separate
    1-4 parameters (e.g. CHARMM) use the same soft core.
- `Csoftcore=:gapsys`: the Coulomb soft core, `:beutler`, `:gapsys` or `:scaled`, where the
    potential is scaled directly by λ.
"""
function RelativeFESystem(sysA::System, sysB::System, global_λ, mapping, core_mapAB; 
                        temp = 298.0u"K", 
                        scheduler=DefaultLambdaScheduler(dual=true),
                        loggers=(),
                        LJsoftcore=:gapsys,
                        Csoftcore=:gapsys
                        )
    FT, units, temp, AT, grad_safe = alchemical_settings(sysA, temp)
    check_softcores(LJsoftcore, Csoftcore)
    lj_sc = softcore_dic[LJsoftcore]
    # To-do: Currently not implemented a 1-4 intramolecular LJ interaction for single topology that
    #   is not decoupled
    if !scheduler.dual && scheduler.intraLJ
        throw(ArgumentError("intraLJ=true requires dual topology, " *
                            "use a scheduler with dual=true or intraLJ=false"))
    end

    atomsA, coordsA = from_device(sysA.atoms), from_device(sysA.coords)
    atomsB, coordsB = from_device(sysB.atoms), from_device(sysB.coords)
    host_lists(sys) = [to_device(il, Array) for il in sys.specific_inter_lists
                       if !(eltype(il.inters) <: EwaldExclusion)]

    # Check/add mappings by including mapping between environment A and environment B
    mapping, core_mapAB = deepcopy(mapping), copy(core_mapAB)
    ligand_A = Set([mapping["core"]; mapping["unique_A"]])
    ligand_B = Set([mapping["unique_B"]; [core_mapAB[i] for i in mapping["core"]]])
    env_A = [i for i in 1:length(atomsA) if !(i in ligand_A)]
    env_B = [i for i in 1:length(atomsB) if !(i in ligand_B)]
    if length(env_B) != length(env_A)
        throw(ArgumentError("system B has $(length(env_B)) environment atoms but system A has " *
                            "$(length(env_A)), they should be the same"))
    end
    check_env_alignment(sysA, sysB, env_A, env_B)
    env_BA = Dict(zip(env_B, env_A))

    # Unique atoms should be bonded to core atoms and not directly to environment atoms,
    #   however, if found, env atoms are turned into core atoms.
    check_unique_env_bonds(sysA, mapping["unique_A"], ligand_A, identity, "A", host_lists)
    check_unique_env_bonds(sysB, mapping["unique_B"], ligand_B, i -> env_BA[i], "B", host_lists)

    # Core atoms that have different constraint lengths or angles between sys A and sys B will
    #   become unique atoms as interpolating between constraint length or angles is not possible
    moved = unmap_changing_constraints!(mapping, core_mapAB, sysA, sysB)
    if !isempty(moved)
        @info "$(length(moved)) core atoms of system A are unique atoms because their constraint " *
              "changes between the end states: $moved"
    end
    mapping["env"] = env_A
    unique_A = Set(mapping["unique_A"])
    mapping_A = Dict()
    mapping_B = Dict()
    unique_groups = Dict("sysA"=>[], "sysB"=>[],"core"=>[])

    # Initialize data groups for new system
    Atoms        = []
    Virtual      = []
    vsA          = from_device(sysA.virtual_sites)
    weight_cross = (isempty(vsA) ? 0.0u"nm^-1" : zero(first(vsA).weight_cross))
    Data         = []
    Coords       = []
    Interactions = []
    Boundary = deepcopy(sysA.boundary)

    # Add all the atoms with new numbering (save a mapping to adjust the interactions list later)
    # Add first core, then unique and then environment atoms
    counter = 1
    res_n = 0
    res_number = 0
    chain = ""
    for i in mapping["core"]
        aA = atomsA[i]
        dA = sysA.atoms_data[i]
        cA = coordsA[i]
        aB = atomsB[core_mapAB[i]]
        dB = sysB.atoms_data[core_mapAB[i]]
        cB = coordsB[core_mapAB[i]]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=aA.atom_type, mass=aA.mass, charge=aA.charge, σ=aA.σ, ϵ=aA.ϵ, 
                                                λ=FT(global_λ), alch_role=CoreDRole))
            if chain!=dA.chain_id
                chain = dA.chain_id
                res_n = 0
                res_number = 0
            end
            if dA.res_number!=res_n
                res_number +=1
                res_n = dA.res_number
            end
            push!(Data, AtomData(atom_type=dA.atom_type, atom_name=dA.atom_name, res_number=res_number,
                                        res_name=dA.res_name, chain_id=dA.chain_id, element=dA.element, hetero_atom=dA.hetero_atom))
            push!(Coords, cA)
            mapping_A[i] = counter
            push!(unique_groups["sysA"], counter)
            counter += 1

            push!(Atoms, Atom(index=counter, atom_type=aB.atom_type, mass=(units ? FT(0.0)u"g/mol" : FT(0.0)), charge=aB.charge, σ=aB.σ, ϵ=aB.ϵ, 
                                                λ=FT(global_λ), alch_role=CoreIRole))
            push!(Virtual, OneParticleSite(counter, counter-1, weight_cross))
            push!(Data, AtomData(atom_type=dB.atom_type, atom_name=dB.atom_name, res_number=dB.res_number,
                                        res_name=dB.res_name, chain_id=dB.chain_id, element=dB.element, hetero_atom=dB.hetero_atom))
            push!(Coords, zero(cB))
            mapping_B[core_mapAB[i]] = counter
            push!(unique_groups["sysB"], counter)
            counter += 1
        else
            push!(Atoms, Atom(index=counter, atom_type=aA.atom_type, mass=aA.mass, charge=(aA.charge, aB.charge), σ=(aA.σ, aB.σ), ϵ=(aA.ϵ, aB.ϵ),
                                                λ=FT(global_λ), alch_role=CoreRole))
            if chain!=dA.chain_id
                chain = dA.chain_id
                res_n = 0
                res_number = 0
            end
            if dA.res_number!=res_n
                res_number +=1
                res_n = dA.res_number
            end
            push!(Data, AtomData(atom_type=dA.atom_type, atom_name=dA.atom_name, res_number=res_number,
                                        res_name=dA.res_name, chain_id=dA.chain_id, element=dA.element, hetero_atom=dA.hetero_atom))
            push!(Coords, cA)
            mapping_A[i] = counter
            mapping_B[core_mapAB[i]] = counter
            push!(unique_groups["core"], counter)
            counter += 1
        end
    end
    for i in mapping["unique_A"]
        a = atomsA[i]
        d = sysA.atoms_data[i]
        c = coordsA[i]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                                λ=FT(global_λ), alch_role=DeleteRole))
        else
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=(a.charge, zero(a.charge)), 
                                    σ=(a.σ, a.σ), ϵ=(a.ϵ, zero(a.ϵ)), 
                                    λ=FT(global_λ), alch_role=DeleteRole))
        end
        if chain!=d.chain_id
            chain = d.chain_id
            res_n = 0
            res_number = 0
        end
        if d.res_number!=res_n
            res_number +=1
            res_n = d.res_number
        end
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_A[i] = counter
        push!(unique_groups["sysA"], counter)
        counter += 1
    end
    for i in mapping["unique_B"]
        a = atomsB[i]
        d = sysB.atoms_data[i]
        c = coordsB[i]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                                λ=FT(global_λ), alch_role=InsertRole))
        else
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=(zero(a.charge), a.charge), 
                                    σ=(a.σ, a.σ), ϵ=(zero(a.ϵ),   a.ϵ),
                                    λ=FT(global_λ), alch_role=InsertRole))
        end
        if chain!=d.chain_id
            chain = d.chain_id
            res_n = 0
            res_number = 0
        end
        if d.res_number!=res_n
            res_number +=1
            res_n = d.res_number
        end
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_B[i] = counter
        push!(unique_groups["sysB"], counter)
        counter += 1
    end
    for i in mapping["env"]
        a = atomsA[i]
        d = sysA.atoms_data[i]
        c = coordsA[i]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                                λ=FT(1.0), alch_role=EnvRole))
        else
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=(a.charge, a.charge), σ=(a.σ, a.σ), ϵ=(a.ϵ, a.ϵ),
                                    λ=FT(1.0), alch_role=EnvRole))
        end
        if chain!=d.chain_id
            chain = d.chain_id
            res_n = 0
            res_number = 0
        end
        if d.res_number!=res_n
            res_number +=1
            res_n = d.res_number
        end
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_A[i] = counter
        counter += 1
    end

    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    if scheduler.dual
        Virtual = Vector{typeof(Virtual[1])}(Virtual)
    end
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)

    # Trackers for interactions for single Topology
    if !scheduler.dual
        single_top_lambda_arrays = Dict{Any, Any}() 
        single_top_atom_maps = Dict{Any, Dict{Tuple, Int}}()
    end

    # Loop through interactions in system A and add them to new system
    # for dual topology the parameters are floats
    # for single topology the parameters are tuples with (A,nothing)
    for interaction in host_lists(sysA)
        IT = typeof(interaction).name.wrapper
        IIT = typeof(interaction.inters[1]).name.wrapper
        P = hasfield(typeof(interaction.inters[1]), :proper) ? interaction.inters[1].proper : nothing
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        field_types = fieldtypes(typeof(interaction))[1:end-1]
        
        if scheduler.dual
            converted_type = typeof(lambda_specific(interaction.inters[1], scheduler, FT, lj_sc))
        else
            converted_type = typeof(lambda_single(interaction.inters[1], nothing, scheduler, lj_sc))
        end
        
        field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
        tmp = [T() for T in field_types]
        
        if !scheduler.dual && !haskey(single_top_lambda_arrays,(IT,IIT,P))
            single_top_lambda_arrays[(IT,IIT,P)] = tmp[end-1]
            single_top_atom_maps[(IT,IIT,P)] = Dict{Tuple, Int}()
        end
        
        for field_tuple in zip(field_names[1:end-1]...)
            n_fields = length(field_tuple)
            if all(haskey(mapping_A, field_tuple[i]) for i in 1:n_fields-2)
                mapped_atoms = Int[]

                for i in 1:n_fields-2
                    val = mapping_A[field_tuple[i]]
                    push!(tmp[i], val)
                    push!(mapped_atoms, val) # Track mapped atoms for sysB lookup
                end
                
                if scheduler.dual
                    push!(tmp[n_fields-1], lambda_specific(field_tuple[end-1], scheduler, FT, lj_sc))
                else
                    deleted = any(in(unique_A), field_tuple[1:(n_fields - 2)])
                    push!(tmp[n_fields-1], lambda_single_A(field_tuple[end-1], scheduler, lj_sc, deleted))
                    single_top_atom_maps[(IT,IIT,P)][Tuple(mapped_atoms)] = length(tmp[n_fields-1])
                end
                
                push!(tmp[n_fields], field_tuple[end])
            end
        end
        Interactions = push!(Interactions, IT(tmp..., interaction.data))
    end

    # Loop through interactions in system B and add them to new system
    # for dual topology the parameters are floats
    # for single topology the parameters are tuples that are either updated (A,B) or (0,B) is only occuring in system B
    for interaction in host_lists(sysB)
        IT = typeof(interaction).name.wrapper
        IIT = typeof(interaction.inters[1]).name.wrapper
        P = hasfield(typeof(interaction.inters[1]), :proper) ? interaction.inters[1].proper : nothing
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        field_types = fieldtypes(typeof(interaction))[1:end-1]
        
        if scheduler.dual
            converted_type = typeof(lambda_specific(interaction.inters[1], scheduler, FT, lj_sc))
            field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
            tmp = [T() for T in field_types]
            
            for field_tuple in zip(field_names[1:end-1]...)
                n_fields = length(field_tuple)
                # The terms of the ligand of B, also those with the environment: the core copies of B
                #   take over from those of A along λ. The environment atoms are those of A.
                atoms_B = field_tuple[1:(n_fields - 2)]
                if any(i -> haskey(mapping_B, i), atoms_B) &&
                        all(i -> haskey(mapping_B, i) || haskey(env_BA, i), atoms_B)
                    for (k, i) in enumerate(atoms_B)
                        push!(tmp[k], haskey(mapping_B, i) ? mapping_B[i] : mapping_A[env_BA[i]])
                    end
                    push!(tmp[n_fields-1], lambda_specific(field_tuple[end-1], scheduler, FT, lj_sc))
                    push!(tmp[n_fields], field_tuple[end])
                end
            end
            Interactions = push!(Interactions, IT(tmp..., interaction.data))
        else
            converted_type = typeof(lambda_single(nothing, interaction.inters[1], scheduler, lj_sc))
            field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
            
            tmp_unique_B = [T() for T in field_types]
            has_unique = false
            
            lambda_array_A = get(single_top_lambda_arrays, (IT,IIT,P), nothing)
            atom_map_A = get(single_top_atom_maps, (IT,IIT,P), Dict{Tuple, Int}())
            
            for field_tuple in zip(field_names[1:end-1]...)
                n_fields = length(field_tuple)
                mapped_atoms = Int[]
                
                if !any(haskey(mapping_B, field_tuple[i]) for i in 1:n_fields-2)
                    continue
                end
                for i in 1:n_fields-2
                    if haskey(mapping_B, field_tuple[i])
                        push!(mapped_atoms, mapping_B[field_tuple[i]])
                    elseif haskey(env_BA, field_tuple[i])
                        push!(mapped_atoms, mapping_A[env_BA[field_tuple[i]]])
                    end
                end

                mapped_tuple = Tuple(mapped_atoms)
                
                if haskey(atom_map_A, mapped_tuple)
                    idx = atom_map_A[mapped_tuple]
                    lambda_array_A[idx] = update_lambda_function(lambda_array_A[idx], field_tuple[end-1])
                else
                    has_unique = true
                    for i in 1:n_fields-2
                        push!(tmp_unique_B[i], mapped_tuple[i])
                    end
                    push!(tmp_unique_B[n_fields-1], lambda_single(nothing, field_tuple[end-1], scheduler, lj_sc))
                    push!(tmp_unique_B[n_fields], field_tuple[end])
                end

            end
            
            if has_unique
                Interactions = push!(Interactions, IT(tmp_unique_B..., interaction.data))
            end
        end
    end

    Interactions = merge(Interactions)

    # A pair of atoms is excluded, or special, if it is so in either end state
    map_A(i) = mapping_A[i]
    map_B(i) = (haskey(mapping_B, i) ? mapping_B[i] : mapping_A[env_BA[i]])
    n_atoms = length(Coords)
    eligible = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    special = falses(n_atoms, n_atoms)
    for (sys_x, map_x) in ((sysA, map_A), (sysB, map_B))
        excluded_x, special_x = neighbor_exclusions(sys_x.neighbor_finder)
        for (i, j) in excluded_x
            eligible[map_x(i), map_x(j)] = false
            eligible[map_x(j), map_x(i)] = false
        end
        for (i, j) in special_x
            special[map_x(i), map_x(j)] = true
            special[map_x(j), map_x(i)] = true
        end
    end

    # The unique alchemical groups are excluded from interating
    pairs = collect(Iterators.product(unique_groups["sysA"], unique_groups["sysB"]))
    for (i, j) in pairs
        eligible[i, j] = false
        eligible[j, i] = false
    end

    neighbor_finder_type = sysA.neighbor_finder
    dist_cutoff_nf = sysA.neighbor_finder.dist_cutoff

    if neighbor_finder_type == NoNeighborFinder
        nf = NoNeighborFinder()
    elseif neighbor_finder_type == GPUNeighborFinder && uses_gpu_neighbor_finder(AT) && !grad_safe
        excluded_pairs, special_pairs = dense_masks_to_pair_lists(eligible, special)
        n_steps_reorder = sysA.neighbor_finder.n_steps_reorder
        nf = GPUNeighborFinder(n_atoms=n_atoms, dist_cutoff=dist_cutoff_nf,
                               excluded_pairs=excluded_pairs, special_pairs=special_pairs,
                               n_steps_reorder=n_steps_reorder, device_vector_type=AT{Int32, 1})
    elseif neighbor_finder_type == DistanceNeighborFinder &&
                (AT <: AbstractGPUArray || has_infinite_boundary(Boundary))
        n_steps = sysA.neighbor_finder.n_steps
        nf = DistanceNeighborFinder(eligible=to_device(eligible, AT), special=to_device(special, AT),
                                    n_steps=n_steps, dist_cutoff=dist_cutoff_nf)
    elseif neighbor_finder_type == CellListMapNeighborFinder && !(AT <: AbstractGPUArray)
        n_steps = sysA.neighbor_finder.n_steps
        nf = CellListMapNeighborFinder(eligible=eligible, special=special,
                                        n_steps=n_steps, boundary=Boundary, x0=Coords,
                                        dist_cutoff=dist_cutoff_nf)
    else
        n_steps = sysA.neighbor_finder.n_steps
        nf = neighbor_finder_type(
            eligible=to_device(eligible, AT),
            special=to_device(special, AT),
            n_steps=n_steps,
            dist_cutoff=dist_cutoff_nf,
        )
    end

    pairwise_inters = lambda_pairwise_inters(sysA, scheduler, LJsoftcore, Csoftcore, FT)
    # GROMACS lambda scheduler use different PME with two charge grids, the use of two grids
    #   is only possible when insert and delete are scaled linearly at the same time I = (1-D).
    general_inters = lambda_general_inters(sysA, Atoms, Boundary, scheduler, global_λ,
                                           scheduler isa GROMACSLambdaRBFEScheduler)

    # The Ewald exclusions follow the hybrid masks
    for inter in sysA.general_inters
        if inter isa PME
            excluded_pairs = find_excluded_pairs(eligible, special)
            exclusion_data = EwaldExclusionData(FT(inter.dist_cutoff); error_tol=FT(inter.error_tol), scheduler=scheduler)
            ewald_exclusions = InteractionList2Atoms(
                to_device([ep[1] for ep in excluded_pairs], AT),
                to_device([ep[2] for ep in excluded_pairs], AT),
                to_device(fill(EwaldExclusion(), length(excluded_pairs)), AT),
                fill("", length(excluded_pairs)),
                exclusion_data,
            )
            push!(Interactions, ewald_exclusions)
        end
    end

    # The virtual sites of system A, unique virtual sites of B, core (virtual) atoms of B mappend 
    #   to atoms of A
    # To-do: @Joe please check this part
    core_mapBA = Dict(j => i for (i, j) in core_mapAB)
    map_B_real(i) = (haskey(core_mapBA, i) ? mapping_A[core_mapBA[i]] : map_B(i))
    unique_B = Set(mapping["unique_B"])
    vsA = [remap_virtual_site(v, map_A) for v in vsA]
    vsB = [remap_virtual_site(v, map_B_real) for v in from_device(sysB.virtual_sites)
           if v.atom_ind in unique_B]
    virtual_sites = [vsA..., vsB..., Virtual...]

    # Constraints of system A + unique constraints of system B.
    if length(sysA.constraints) != length(sysB.constraints) ||
            any(nameof(typeof(ca)) !== nameof(typeof(cb))
                for (ca, cb) in zip(sysA.constraints, sysB.constraints))
        throw(ArgumentError("both systems need to be set up with the same constraint algorithms, " *
                            "system A has $(nameof.(typeof.(sysA.constraints))) and system B has " *
                            "$(nameof.(typeof.(sysB.constraints)))"))
    end

    constraints = []
    remap_angle_constraint(c, m) = AngleConstraint(m(c.i), m(c.j), m(c.k),
                        acos((c.dist_ij^2 + c.dist_jk^2 - c.dist_ik^2) / (2 * c.dist_ij * c.dist_jk)),
                        c.dist_ij, c.dist_jk)
    angle_pairs(c) = (Set([c.i, c.j]), Set([c.j, c.k]), Set([c.i, c.k]))
    for ca in sysA.constraints
        dist_cs, angle_cs = constraint_lists(ca)
        dist_cs = [DistanceConstraint(map_A(c.i), map_A(c.j), c.dist) for c in dist_cs]
        angle_cs = [remap_angle_constraint(c, map_A) for c in angle_cs]
        constrained = Set(Set([c.i, c.j]) for c in dist_cs)
        foreach(c -> push!(constrained, angle_pairs(c)...), angle_cs)
        for caB in sysB.constraints
            dist_csB, angle_csB = constraint_lists(caB)
            for c in dist_csB
                i, j = map_B_real(c.i), map_B_real(c.j)
                if !(Set([i, j]) in constrained)
                    push!(dist_cs, DistanceConstraint(i, j, c.dist))
                    push!(constrained, Set([i, j]))
                end
            end
            # Adding an AngleConstraint that's in system B on a core atom, might result
            #   in an error if the core atom already has a distance constraint.
            #   So, a constrained angle of B is added as its three distance constraints. Similarly, to how 
            #   GROMACS represents every constrained angle (`h-angles`) and as LINCS does internally.
            # To-do: @Joe please check this part
            for c in angle_csB, dc in to_distance_constraints(remap_angle_constraint(c, map_B_real))
                if !(Set([dc.i, dc.j]) in constrained)
                    push!(dist_cs, dc)
                    push!(constrained, Set([dc.i, dc.j]))
                end
            end
        end
        if !isempty(dist_cs) || !isempty(angle_cs)
            push!(constraints, rebuild_constraints(ca, (isempty(dist_cs) ? nothing : dist_cs),
                                                   (isempty(angle_cs) ? nothing : angle_cs),
                                                   mass.(Atoms)))
        end
    end

    # Build topology based on system A topology, but add unique atoms of B and
    #   virtual sites for core atoms of B
    if isnothing(sysA.topology)
        topology = nothing
    else
        bonded = [(map_A(i), map_A(j)) for (i, j) in sysA.topology.bonded_atoms]
        if !isnothing(sysB.topology)
            unique_B_set = Set(mapping["unique_B"])
            append!(bonded, [(map_B(i), map_B(j)) for (i, j) in sysB.topology.bonded_atoms
                             if Int(i) in unique_B_set || Int(j) in unique_B_set])
        end
        append!(bonded, [(v.atom_ind, v.atom_1) for v in virtual_sites])
        topology = MolecularTopology(first.(bonded), last.(bonded), n_atoms)
    end

    return alchemical_system(sysA, Atoms, Coords, Data, Boundary, temp;
        topology=topology,
        constraints=Tuple(constraints),
        virtual_sites=(isempty(virtual_sites) ? [] : to_device([virtual_sites...], AT)),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=to_device.(tuple(Interactions...), AT),
        neighbor_finder=nf,
        general_inters=general_inters,
        loggers=loggers,
    )
end