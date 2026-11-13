using Molly
using Molly: alchemical_settings, check_softcores, check_scheduler, softcore_dic,
             check_env_alignment, check_unique_env_bonds, constrained_distances, constraint_lists,
             rebuild_constraints, lambda_specific, lambda_pairwise_inters, lambda_general_inters,
             neighbor_exclusions, remap_virtual_site, alchemical_system, from_device, to_device,
             EwaldExclusionData, AbstractEwald, find_excluded_pairs,
             uses_gpu_neighbor_finder, has_infinite_boundary, to_distance_constraints,
             MinimumMixing, λ_params_function, scale_sterics, volume, LJDispersionCorrectionλ,
             disp_corr_cutoff, disp_corr_factors, lj_disp_corr_types, disp_corr_pe,
             disp_corr_virial!

const AbstractGPUArray = Molly.GPUArrays.AbstractGPUArray
const AtomsCalculators = Molly.AtomsCalculators

include(joinpath(@__DIR__, "probability_shed.jl"))

# The alchemical atoms of each system (`groups[n]`, it behaves as that vector), and the counter-ions
#   shared by several systems: hybrid atom => the systems it serves, its λ is the sum of theirs
struct HybridGroups <: AbstractVector{Vector{Int}}
    systems::Vector{Vector{Int}}
    pools::Vector{Pair{Int, Vector{Int}}}
end
Base.size(g::HybridGroups) = size(g.systems)
Base.getindex(g::HybridGroups, n::Int) = g.systems[n]

# Every atom whose interactions depend on λ
alchemical_atoms(groups::HybridGroups) = [reduce(vcat, groups.systems); first.(groups.pools)]

# The length of a constraint can not be interpolated, so the constraints between core atoms have
#   to be the same in every system
function check_core_constraints(sys1, sysn, core, core_map_n, n)
    dists_1 = constrained_distances(sys1, Set(core))
    dists_n = constrained_distances(sysn, Set(core_map_n[i] for i in core))
    back = Dict(core_map_n[i] => i for i in core)
    dists_n = Dict(minmax(back[i], back[j]) => d for ((i, j), d) in dists_n)
    for pair in union(keys(dists_1), keys(dists_n))
        d1, dn = get(dists_1, pair, nothing), get(dists_n, pair, nothing)
        if isnothing(d1) || isnothing(dn) || !isapprox(d1, dn; rtol=1e-6)
            throw(ArgumentError("the constraint between core atoms $(pair[1]) and $(pair[2]) of " *
                "system 1 is different in system $n, map the atom that changes as a unique atom " *
                "of every system instead"))
        end
    end
end

# A switched-off atom should hold on to the real atoms by one bond and one angle. Its other angles
#   around the core atom it is bonded to, and the Urey-Bradley terms that go with them, restrain
#   the angles between the real atoms. This returns a function that tells whether a term of `sys`
#   (interaction and atom indices) is such a redundant term, which is then scaled with λ.
function redundant_terms(sys, unique, soft, ligand, host_lists)
    unique = Set(unique)
    held = setdiff(unique, soft)
    # Molly stores a Urey-Bradley term as a bond, the angles tell which atoms are really bonded
    neighbours = Dict{Int, Set{Int}}()
    for il in host_lists(sys)
        eltype(il.inters) <: HarmonicAngle || continue
        for (i, j, k) in zip(il.is, il.js, il.ks), (a, b) in ((i, j), (j, i), (j, k), (k, j))
            push!(get!(neighbours, Int(a), Set{Int}()), Int(b))
        end
    end
    # The kept angle of a unique atom goes to a core atom if there is one, else to the environment
    anchor = Dict{Int, Int}()
    for x in held, a in get(neighbours, x, ())
        a in unique && continue
        others = [b for b in neighbours[a] if !(b in unique)]
        isempty(others) || (anchor[x] = argmin(b -> (!(b in ligand), b), others))
    end
    return function (inter, inds)
        if inter isa HarmonicBond
            i, j = Int.(inds)
            return !(j in get(neighbours, i, ())) && ((i in unique) != (j in unique))
        elseif inter isa HarmonicAngle
            i, j, k = Int.(inds)
            j in unique && return false
            x, b = (i in held) ? (i, k) : (k, i)
            return haskey(anchor, x) && !(b in unique) && anchor[x] != b
        end
        return false
    end
end

# `mapping["core"]` are the core atoms of system 1, `mapping["unique_sys\$n"]` the unique atoms of
#   system n and `core_map[n]` maps the core atoms of system 1 to those of system n (n >= 2).
#   `mapping["soft_sys\$n"]` is optional: unique atoms of system n whose bonded terms with core
#   atoms are scaled with λ, to open a ring that the unique atoms close through the core.
#   `scale_redundant=false` keeps every angle and Urey-Bradley term of a unique atom on at every
#   λ, as RelativeFESystem does.
#   `scaled_torsions[n]` are proper torsions of system n (four atom indices each) that are scaled
#   with λ whatever the scheduler does with the other torsions, as LaDyBUGS scales a chosen list.
#   `global_λs[n]` is the fraction system n is turned on, they sum to one.
#   `ion_pools` are counter-ions for charge-changing systems, environment atoms of system 1 (in
#   every system): ion => the systems it serves. Its λ is the sum of theirs, so it is on as much as
#   they are; the systems of one ion have to be of one site, so that the sum is at most one. An ion
#   with no systems is off in every state.
function MultiFESystem(systems, global_λs, mapping, core_map;
                        temp = 298.0u"K",
                        scheduler=ProbabilityLambdaScheduler(dual=true),
                        loggers=(),
                        LJsoftcore=:gapsys,
                        Csoftcore=:gapsys,
                        scale_redundant=true,
                        scaled_torsions=Dict{Int, Vector{NTuple{4, Int}}}(),
                        ion_pools=Pair{Int, Vector{Int}}[],
                        ion_role=CorePRole
                        )
    sysA = systems[1]
    n_sys = length(systems)
    FT, units, temp, AT, grad_safe = alchemical_settings(sysA, temp)
    check_softcores(LJsoftcore, Csoftcore)
    check_scheduler(scheduler, sysA)
    if !(scheduler isa ProbabilityLambdaScheduler) || !scheduler.dual
        throw(ArgumentError("scheduler must be a ProbabilityLambdaScheduler(dual=true)"))
    end
    if length(global_λs) != n_sys || !isapprox(sum(global_λs), 1)
        throw(ArgumentError("global_λs needs one λ for each of the $n_sys systems and they have " *
                            "to sum to one, got $global_λs"))
    end
    lj_sc = softcore_dic[LJsoftcore]

    atoms_list = [from_device(sys.atoms) for sys in systems]
    coords_list = [from_device(sys.coords) for sys in systems]
    host_lists(sys) = [to_device(il, Array) for il in sys.specific_inter_lists
                       if !(eltype(il.inters) <: EwaldExclusion)]

    # The environment of each system is everything outside its ligand, and has to be the same
    #   atoms in the same order in every system
    core = mapping["core"]
    core_maps = [n == 1 ? Dict(i => i for i in core) : core_map[n] for n in 1:n_sys]
    core_back = [Dict(core_maps[n][i] => i for i in core) for n in 1:n_sys]
    uniques = [mapping["unique_sys$n"] for n in 1:n_sys]
    ligands = [Set([uniques[n]; [core_maps[n][i] for i in core]]) for n in 1:n_sys]
    envs = [[i for i in 1:length(atoms_list[n]) if !(i in ligands[n])] for n in 1:n_sys]
    for n in 2:n_sys
        if length(envs[1]) != length(envs[n])
            throw(ArgumentError("system $n has $(length(envs[n])) environment atoms but system 1 has " *
                                "$(length(envs[1])), they should be the same"))
        end
        check_env_alignment(sysA, systems[n], envs[1], envs[n])
        check_core_constraints(sysA, systems[n], core, core_maps[n], n)
    end
    env_maps = [Dict(zip(envs[n], envs[1])) for n in 1:n_sys]
    for (i, ns) in ion_pools
        if !(i in envs[1]) || !issubset(ns, 1:n_sys)
            throw(ArgumentError("ion pool $i => $ns: the ion has to be an environment atom of system 1 " *
                                "and the systems among 1:$n_sys"))
        end
    end

    # Unique atoms should be bonded to core atoms and not directly to environment atoms. Molly's
    #   check reads the bonds out of the pair lists it is given and skips the 1-4 and Ewald
    #   exclusion ones itself
    for n in 1:n_sys
        check_unique_env_bonds(systems[n], uniques[n], ligands[n], i -> env_maps[n][i], "$n", host_lists)
    end

    # A core atom that is a virtual site in one system has to be one in all of them, since the
    #   copies of a core atom share their position
    vs_list = [from_device(sys.virtual_sites) for sys in systems]
    vs_inds = [Set(v.atom_ind for v in vs) for vs in vs_list]
    weight_cross = (isempty(vs_list[1]) ? 0.0u"nm^-1" : zero(first(vs_list[1]).weight_cross))
    for n in 2:n_sys, i in core
        if (i in vs_inds[1]) != (core_maps[n][i] in vs_inds[n])
            throw(ArgumentError("core atom $i of system 1 is a virtual site in only one of system " *
                "1 and system $n, make it a unique atom of every system instead"))
        end
    end

    # Initialize data groups for new system
    Atoms        = []
    Virtual      = []
    Data         = []
    Coords       = []
    Interactions = []
    Boundary = deepcopy(sysA.boundary)
    mappings = [Dict() for _ in 1:n_sys]     # atom of system n -> atom of the new system
    groups = [Int[] for _ in 1:n_sys]        # the alchemical atoms of each system

    # Residues are renumbered from one within each chain
    res_n = 0
    res_number = 0
    chain = ""
    function renumbered(d)
        if chain != d.chain_id
            chain = d.chain_id
            res_n = 0
            res_number = 0
        end
        if d.res_number != res_n
            res_number += 1
            res_n = d.res_number
        end
        return res_number
    end

    function add_atom!(n, i, λ, role; real=true, in_group=(role != EnvRole))
        a, d = atoms_list[n][i], systems[n].atoms_data[i]
        counter = length(Atoms) + 1
        push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=(real ? a.mass : zero(a.mass)),
                          charge=a.charge, σ=a.σ, ϵ=a.ϵ, λ=FT(λ), alch_role=role))
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name,
                             res_number=(real ? renumbered(d) : d.res_number), res_name=d.res_name,
                             chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, real ? coords_list[n][i] : zero(coords_list[n][i]))
        mappings[n][i] = counter
        in_group && push!(groups[n], counter)
        return counter
    end

    # Add all the atoms with new numbering (save a mapping to adjust the interactions list later)
    # Add first core, then unique and then environment atoms
    # A core atom has one copy for each system: that of system 1 is the real atom, the others are
    #   massless and sit on top of it
    for i in core
        real_copy = add_atom!(1, i, global_λs[1], CorePRole)
        for n in 2:n_sys
            site = add_atom!(n, core_maps[n][i], global_λs[n], CorePRole; real=false)
            # A core virtual site keeps the definition it has in system n, added below
            if !(core_maps[n][i] in vs_inds[n])
                push!(Virtual, OneParticleSite(site, real_copy, weight_cross))
            end
        end
    end
    for n in 1:n_sys, i in uniques[n]
        add_atom!(n, i, global_λs[n], i in get(mapping, "soft_sys$n", ()) ? SoftRole : ProbRole)
    end
    # A pooled ion is an atom of no system, with the summed λ of its systems. It takes the role of a
    #   core copy, so that all its interactions scale with λ, also with `intraLJ` (which keeps those
    #   between unique atoms on); or, with `ion_role=ChargeRole`, only its charge does
    pool_of = Dict(ion_pools)
    pools = Pair{Int, Vector{Int}}[]
    for i in envs[1]
        if haskey(pool_of, i)
            push!(pools, add_atom!(1, i, sum(global_λs[pool_of[i]]), ion_role; in_group=false) => pool_of[i])
        else
            add_atom!(1, i, 1, EnvRole)
        end
    end

    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)

    # Atom of system n -> atom of the new system; the environment atoms are those of system 1
    hybrid(n) = i -> (haskey(mappings[n], i) ? mappings[n][i] : mappings[1][env_maps[n][i]])
    # The same, but with the core atoms as the real atoms of system 1 instead of their copies
    hybrid_real(n) = i -> (haskey(core_back[n], i) ? mappings[1][core_back[n][i]] : hybrid(n)(i))

    redundant = [redundant_terms(systems[n], uniques[n], get(mapping, "soft_sys$n", Int[]),
                                 ligands[n], host_lists) for n in 1:n_sys]
    scheduler_scaled = always_scaled(scheduler)
    listed = [Set(get(scaled_torsions, n, NTuple{4, Int}[])) for n in 1:n_sys]
    is_listed(n, inter, inds) = inter isa PeriodicTorsion &&
                                (Int.(inds) in listed[n] || reverse(Int.(inds)) in listed[n])

    # All interactions of system 1, and of the other systems those of the ligand, also the ones
    #   with the environment
    for (n, sys) in enumerate(systems), interaction in host_lists(sys)
        IT = typeof(interaction).name.wrapper
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        field_types = fieldtypes(typeof(interaction))[1:end-1]

        converted_type = typeof(lambda_specific(interaction.inters[1], scheduler, FT, lj_sc))
        field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
        tmp = [T() for T in field_types]

        for field_tuple in zip(field_names[1:end-1]...)
            n_fields = length(field_tuple)
            inds = field_tuple[1:(n_fields - 2)]
            if n == 1 || any(i -> haskey(mappings[n], i), inds)
                for (k, i) in enumerate(inds)
                    push!(tmp[k], hybrid(n)(i))
                end
                scaled = (scale_redundant && redundant[n](field_tuple[end-1], inds)) ||
                         is_listed(n, field_tuple[end-1], inds)
                sched = scaled ? scheduler_scaled : scheduler
                push!(tmp[n_fields-1], lambda_specific(field_tuple[end-1], sched, FT, lj_sc))
                push!(tmp[n_fields], field_tuple[end])
            end
        end
        push!(Interactions, IT(tmp..., interaction.data))
    end

    Interactions = Molly.merge_interactions(Interactions)

    # A pair of atoms is excluded, or special, if it is so in any of the systems
    #   Have to take information from neighbourfinder, since constraints drop
    #   the interaction from the list of interactions.
    n_atoms = length(Coords)
    eligible = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    special = falses(n_atoms, n_atoms)
    for (n, sys) in enumerate(systems)
        map_n = hybrid(n)
        excluded_n, special_n = neighbor_exclusions(sys.neighbor_finder)
        for (i, j) in excluded_n
            eligible[map_n(i), map_n(j)] = false
            eligible[map_n(j), map_n(i)] = false
        end
        for (i, j) in special_n
            special[map_n(i), map_n(j)] = true
            special[map_n(j), map_n(i)] = true
        end
    end

    # The alchemical groups of different systems are excluded from interacting
    for n1 in 1:n_sys, n2 in 1:(n1 - 1)
        for i in groups[n1], j in groups[n2]
            eligible[i, j] = false
            eligible[j, i] = false
        end
    end

    # The neighbour finder of the end states, as `System` setup picks it: the type is compared with
    #   `isa`, since the finders are parametric; `nothing` is no finder (as Molly's alchemical_system.jl)
    nf_A = sysA.neighbor_finder
    if isnothing(nf_A)
        nf = nothing
    elseif nf_A isa GPUNeighborFinder && uses_gpu_neighbor_finder(AT) && !grad_safe
        nf = GPUNeighborFinder(eligible=eligible, special=special, dist_cutoff=nf_A.dist_cutoff,
                               n_steps=nf_A.n_steps, array_type=AT)
    elseif nf_A isa DistanceNeighborFinder &&
                (AT <: AbstractGPUArray || has_infinite_boundary(Boundary))
        nf = DistanceNeighborFinder(eligible=eligible, special=special, n_steps=nf_A.n_steps,
                                    dist_cutoff=nf_A.dist_cutoff, array_type=AT)
    elseif nf_A isa CellListMapNeighborFinder && !(AT <: AbstractGPUArray)
        nf = CellListMapNeighborFinder(eligible=eligible, special=special, n_steps=nf_A.n_steps,
                                       boundary=Boundary, x0=Coords, dist_cutoff=nf_A.dist_cutoff)
    elseif nf_A isa GPUCellListNeighborFinder && AT <: AbstractGPUArray
        nf = GPUCellListNeighborFinder(eligible=eligible, special=special, n_steps=nf_A.n_steps,
                                       dist_cutoff=nf_A.dist_cutoff, array_type=AT,
                                       max_neighbors=nf_A.max_neighbors, ragged=nf_A.ragged)
    else
        # Another finder, or one that does not suit the device, keeps the masks and the settings
        nf = typeof(nf_A).name.wrapper(
            eligible=to_device(eligible, AT),
            special=to_device(special, AT),
            n_steps=nf_A.n_steps,
            dist_cutoff=nf_A.dist_cutoff,
        )
    end

    pairwise_inters = lambda_pairwise_inters(sysA, scheduler, LJsoftcore, Csoftcore, FT)
    # Molly's dispersion correction does not know which substituents do not see each other
    general_inters = map(lambda_general_inters(sysA, Atoms, Boundary, scheduler, first(global_λs))) do inter
        inter isa LJDispersionCorrectionλ || return inter
        isempty(ion_pools) || throw(ArgumentError("the dispersion correction does not know the ion pools"))
        MultiLJDispersionCorrection(Atoms, groups, inter.cutoff, inter.σ_mix, inter.ϵ_mix;
                                    scheduler=scheduler)
    end

    # The Ewald exclusions follow the hybrid masks
    for inter in sysA.general_inters
        if inter isa AbstractEwald
            excluded_pairs = find_excluded_pairs(eligible, special)
            exclusion_data = EwaldExclusionData(FT(inter.dist_cutoff); error_tol=FT(inter.error_tol),
                                    ϵr=(inter isa PME ? FT(inter.ϵr) : one(FT)), scheduler=scheduler)
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

    # The virtual sites of system 1, and of the other systems those of the ligand. A site of a
    #   core copy is placed relative to the real atoms, which lie at the same positions
    virtual_sites = Any[remap_virtual_site(v, hybrid(1)) for v in vs_list[1]]
    for n in 2:n_sys
        append!(virtual_sites, [remap_virtual_site(v, hybrid_real(n), hybrid(n)) for v in vs_list[n]
                                if v.atom_ind in ligands[n]])
    end
    virtual_sites = [virtual_sites..., Virtual...]

    # Constraints of system 1 + the constraints of the unique atoms of the other systems
    for n in 2:n_sys
        if length(sysA.constraints) != length(systems[n].constraints) ||
                any(nameof(typeof(ca)) !== nameof(typeof(cb))
                    for (ca, cb) in zip(sysA.constraints, systems[n].constraints))
            throw(ArgumentError("all systems need to be set up with the same constraint algorithms, " *
                                "system 1 has $(nameof.(typeof.(sysA.constraints))) and system $n has " *
                                "$(nameof.(typeof.(systems[n].constraints)))"))
        end
    end

    constraints = []
    remap_angle_constraint(c, m) = AngleConstraint(m(c.i), m(c.j), m(c.k),
                        acos((c.dist_ij^2 + c.dist_jk^2 - c.dist_ik^2) / (2 * c.dist_ij * c.dist_jk)),
                        c.dist_ij, c.dist_jk)
    angle_pairs(c) = (Set([c.i, c.j]), Set([c.j, c.k]), Set([c.i, c.k]))
    for ca in sysA.constraints
        dist_cs, angle_cs = constraint_lists(ca)
        dist_cs = [DistanceConstraint(hybrid(1)(c.i), hybrid(1)(c.j), c.dist) for c in dist_cs]
        angle_cs = [remap_angle_constraint(c, hybrid(1)) for c in angle_cs]
        constrained = Set(Set([c.i, c.j]) for c in dist_cs)
        foreach(c -> push!(constrained, angle_pairs(c)...), angle_cs)
        for n in 2:n_sys, ca_n in systems[n].constraints
            map_n = hybrid_real(n)
            dist_cs_n, angle_cs_n = constraint_lists(ca_n)
            for c in dist_cs_n
                i, j = map_n(c.i), map_n(c.j)
                if !(Set([i, j]) in constrained)
                    push!(dist_cs, DistanceConstraint(i, j, c.dist))
                    push!(constrained, Set([i, j]))
                end
            end
            # An angle constraint of system n stays one when none of its three pairs is
            #   constrained already, which is what LINCS needs. A pair of a core atom that is
            #   constrained in system 1 makes the triangle coupled, so there it falls back to three
            #   distance constraints, as GROMACS represents `h-angles` and as LINCS does internally.
            for c in angle_cs_n
                c_n = remap_angle_constraint(c, map_n)
                if !any(p in constrained for p in angle_pairs(c_n))
                    push!(angle_cs, c_n)
                    foreach(p -> push!(constrained, p), angle_pairs(c_n))
                    continue
                end
                for dc in to_distance_constraints(c_n)
                    if !(Set([dc.i, dc.j]) in constrained)
                        push!(dist_cs, dc)
                        push!(constrained, Set([dc.i, dc.j]))
                    end
                end
            end
        end
        if !isempty(dist_cs) || !isempty(angle_cs)
            push!(constraints, rebuild_constraints(ca, (isempty(dist_cs) ? nothing : dist_cs),
                                                   (isempty(angle_cs) ? nothing : angle_cs),
                                                   mass.(Atoms)))
        end
    end

    # Build topology based on system 1 topology, but add unique atoms of the other systems and
    #   the virtual sites
    if isnothing(sysA.topology)
        topology = nothing
    else
        bonded = [(hybrid(1)(i), hybrid(1)(j)) for (i, j) in sysA.topology.bonded_atoms]
        for n in 2:n_sys
            isnothing(systems[n].topology) && continue
            unique_n = Set(uniques[n])
            append!(bonded, [(hybrid(n)(i), hybrid(n)(j)) for (i, j) in systems[n].topology.bonded_atoms
                             if Int(i) in unique_n || Int(j) in unique_n])
        end
        append!(bonded, [(v.atom_ind, v.atom_1) for v in virtual_sites])
        topology = MolecularTopology(first.(bonded), last.(bonded), n_atoms)
    end

    sys = alchemical_system(sysA, Atoms, Coords, Data, Boundary, temp;
        topology=topology,
        constraints=Tuple(constraints),
        virtual_sites=(isempty(virtual_sites) ? [] : to_device([virtual_sites...], AT)),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=to_device.(tuple(Interactions...), AT),
        neighbor_finder=nf,
        general_inters=general_inters,
        loggers=loggers,
    )
    return sys, HybridGroups(groups, pools)
end

# Set the λ of each system in place, without building the hybrid system again. Only the λ of the
#   alchemical atoms and the dispersion correction depend on it.
function set_lambdas!(sys, groups::HybridGroups, global_λs)
    if length(global_λs) != length(groups) || !isapprox(sum(global_λs), 1)
        throw(ArgumentError("global_λs needs one λ for each of the $(length(groups)) systems and " *
                            "they have to sum to one, got $global_λs"))
    end
    FT = float_type(sys)
    # A new vector rather than a write into the old one, so a system copied from another keeps its λ
    atoms = copy(from_device(sys.atoms))
    set_λ(i, λ) = (a = atoms[i]; atoms[i] = Atom(a.index, a.atom_type, a.mass, a.charge, a.σ, a.ϵ, FT(λ), a.alch_role))
    for (group, λ) in zip(groups, global_λs), i in group
        set_λ(i, λ)
    end
    for (i, ns) in groups.pools
        set_λ(i, sum(global_λs[ns]))
    end
    sys.atoms = to_device(atoms, array_type(sys))
    sys.general_inters = map(sys.general_inters) do inter
        inter isa MultiLJDispersionCorrection ? MultiLJDispersionCorrection(atoms, inter) : inter
    end
    return sys
end

# -------------------------------------------------------------------------------------------------
# The long range Lennard-Jones correction of Molly's `LJDispersionCorrectionλ` for a system with
#   several substituents. `groups[n]` are the alchemical atoms of substituent n: atoms of different
#   substituents do not interact, so their pairs are left out. The factors are computed from the λ
#   of the atoms, and `classes` is kept so that they can be computed again after λ changes.
struct MultiLJDispersionCorrection{F6, F12, EC, C, S, E, SCH, CL}
    factor_6::F6
    factor_12::F12
    energy_const::EC
    cutoff::C
    σ_mix::S
    ϵ_mix::E
    scheduler::SCH
    classes::CL
end

# The atoms the correction cannot tell apart, as (σ, ϵ, role, substituent) with a count and one
#   representative atom. None of the four changes after the hybrid is built, so the classing is
#   done once: λ does change, but every atom of a substituent carries the same λ, so reading the
#   representative is enough. `set_lambdas!` rebuilds this correction at every λ and LaDyBUGS
#   visits 141 states per cycle, so classing all the atoms again each time dominated the cost.
function dispersion_classes(atoms_cpu, groups)
    group_of = zeros(Int, length(atoms_cpu))
    for (n, group) in enumerate(groups)
        group_of[group] .= n
    end
    at = atoms_cpu[1]
    Key = Tuple{typeof(at.σ), typeof(at.ϵ), Int32, Int}
    index = Dict{Key, Int}()
    keys_seen, counts, first_atom = Key[], Int[], Int[]
    for (i, a) in enumerate(atoms_cpu)
        key = (a.σ, a.ϵ, a.alch_role, group_of[i])
        j = get!(index, key, length(counts) + 1)
        if j > length(counts)
            push!(keys_seen, key); push!(counts, 0); push!(first_atom, i)
        elseif atoms_cpu[first_atom[j]].λ != a.λ
            error("atoms $i and $(first_atom[j]) share σ, ϵ, role and substituent but have " *
                  "different λ, which the dispersion correction assumes cannot happen")
        end
        counts[j] += 1
    end
    return [(σ=k[1], ϵ=k[2], role=k[3], group=k[4], count=counts[j], atom=first_atom[j])
            for (j, k) in enumerate(keys_seen)]
end

function correction(atoms_cpu, classes, cutoff, σ_mix, ϵ_mix, scheduler)
    cutoff_used = disp_corr_cutoff(cutoff)
    at = atoms_cpu[1]
    term_6_example  = at.ϵ * at.σ^6
    term_12_example = at.ϵ * at.σ^12
    ϵσ6_unit  = unit(term_6_example)
    ϵσ12_unit = unit(term_12_example)
    F6, F12, EC = lj_disp_corr_types(cutoff_used, at)   # the types Molly returns the factors in

    λs = [atoms_cpu[c.atom].λ for c in classes]
    rep(j) = Atom(σ=classes[j].σ, ϵ=classes[j].ϵ, λ=λs[j], alch_role=classes[j].role)

    # Accumulate pair sums in Float64 for precision. The coupling is applied once for each pair of
    #   classes, not to the counts
    ϵσ6_sum, ϵσ12_sum = 0.0, 0.0
    for a in eachindex(classes), b in 1:a
        ga, gb = classes[a].group, classes[b].group
        ga != 0 && gb != 0 && ga != gb && continue
        na, nb = classes[a].count, classes[b].count
        n_pair = (a == b ? (na * (na + 1)) / 2 : na * nb)
        λ, _, _, σ, ϵ = λ_params_function(scheduler, MinimumMixing(), σ_mix, ϵ_mix, rep(a), rep(b), false)
        ϵσ6_sum  += Float64(ustrip(ϵσ6_unit,  λ * ϵ * σ^6 )) * n_pair
        ϵσ12_sum += Float64(ustrip(ϵσ12_unit, λ * ϵ * σ^12)) * n_pair
    end

    # The copies of a core atom count as one particle, since their λ sum to one
    n_eff = Float64(sum(c.count * first(scale_sterics(scheduler, λs[j], c.role, Val(true)))
                        for (j, c) in enumerate(classes)))
    n_pairs = n_eff * (n_eff + 1) / 2
    ϵσ6_mean  = (iszero(n_pairs) ? 0.0 : ϵσ6_sum  / n_pairs) * ϵσ6_unit
    ϵσ12_mean = (iszero(n_pairs) ? 0.0 : ϵσ12_sum / n_pairs) * ϵσ12_unit

    # Molly turns the pair averages into the factors, so the cutoff shapes it supports and the
    #   constant a shifted potential adds are handled there rather than repeated here
    factor_6, factor_12, energy_const = disp_corr_factors(cutoff_used, ϵσ6_mean, ϵσ12_mean, n_eff,
                                                          F6, F12, EC)
    return MultiLJDispersionCorrection(factor_6, factor_12, energy_const, cutoff_used,
                                       σ_mix, ϵ_mix, scheduler, classes)
end

function MultiLJDispersionCorrection(atoms, groups, cutoff, σ_mix, ϵ_mix; scheduler)
    atoms_cpu = from_device(atoms)
    return correction(atoms_cpu, dispersion_classes(atoms_cpu, groups), cutoff, σ_mix, ϵ_mix,
                      scheduler)
end

# The same correction for atoms with a new λ
MultiLJDispersionCorrection(atoms, inter::MultiLJDispersionCorrection) =
    correction(from_device(atoms), inter.classes, inter.cutoff, inter.σ_mix, inter.ϵ_mix,
               inter.scheduler)

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys,
                                                    inter::MultiLJDispersionCorrection; kwargs...)
    return disp_corr_pe(inter, sys)
end

# No net force on any atom, but the tail does contribute to the virial
AtomsCalculators.@generate_interface function AtomsCalculators.forces!(
        fs, sys, inter::MultiLJDispersionCorrection;
        buffers=nothing,
        needs_vir=false,
        kwargs...)
    needs_vir && disp_corr_virial!(buffers, sys, inter)
    return fs
end
