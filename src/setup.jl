# Read files to set up a system
# See OpenMM source code
# See http://manual.gromacs.org/documentation/2016/user-guide/file-formats.html

export
    place_atoms,
    place_diatomics,
    is_any_atom,
    is_heavy_atom,
    add_position_restraints

"""
    place_atoms(n_atoms, boundary; min_dist=nothing, max_attempts=100, rng=Random.default_rng())

Generate random coordinates.

Obtain `n_atoms` coordinates in bounding box `boundary` where no two
points are closer than `min_dist`, accounting for periodic boundary conditions.
The keyword argument `max_attempts` determines the number of failed tries after
which to stop placing atoms.
Can not be used if one or more dimensions has infinite boundaries.
"""
function place_atoms(n_atoms::Integer,
                     boundary;
                     min_dist=zero(length_type(boundary)),
                     max_attempts::Integer=100,
                     rng=Random.default_rng())
    if has_infinite_boundary(boundary)
        throw(ArgumentError("one or more dimension has infinite boundaries, boundary is $boundary"))
    end
    dims = AtomsBase.n_dimensions(boundary)
    max_atoms = volume(boundary) / (min_dist ^ dims)
    if n_atoms > max_atoms
        throw(ArgumentError("boundary $boundary too small for $n_atoms atoms with minimum distance $min_dist"))
    end
    min_dist_sq = min_dist ^ 2
    coords = SArray[]
    sizehint!(coords, n_atoms)
    failed_attempts = 0
    while length(coords) < n_atoms
        new_coord = random_coord(boundary; rng=rng)
        okay = true
        if min_dist > zero(min_dist)
            for coord in coords
                if sum(abs2, vector(coord, new_coord, boundary)) < min_dist_sq
                    okay = false
                    failed_attempts += 1
                    break
                end
            end
        end
        if okay
            push!(coords, new_coord)
            failed_attempts = 0
        elseif failed_attempts >= max_attempts
            error("failed to place atom $(length(coords) + 1) after $max_attempts (max_attempts) tries")
        end
    end
    return [coords...]
end

"""
    place_diatomics(n_molecules, boundary, bond_length; min_dist=nothing,
                    max_attempts=100, aligned=false, rng=Random.default_rng())

Generate random diatomic molecule coordinates.

Obtain coordinates for `n_molecules` diatomics in bounding box `boundary`
where no two points are closer than `min_dist` and the bond length is `bond_length`,
accounting for periodic boundary conditions.
The keyword argument `max_attempts` determines the number of failed tries after
which to stop placing atoms.
The keyword argument `aligned` determines whether the bonds all point the same direction
(`true`) or random directions (`false`).
Can not be used if one or more dimensions has infinite boundaries.
"""
function place_diatomics(n_molecules::Integer,
                         boundary,
                         bond_length;
                         min_dist=zero(length_type(boundary)),
                         max_attempts::Integer=100,
                         aligned::Bool=false,
                         rng=Random.default_rng())
    if has_infinite_boundary(boundary)
        throw(ArgumentError("one or more dimension has infinite boundaries, boundary is $boundary"))
    end
    dims = AtomsBase.n_dimensions(boundary)
    max_molecules = volume(boundary) / ((min_dist + bond_length) ^ dims)
    if n_molecules > max_molecules
        throw(ArgumentError("boundary $boundary too small for $n_molecules diatomics with minimum distance $min_dist"))
    end
    min_dist_sq = min_dist ^ 2
    coords = SArray[]
    sizehint!(coords, 2 * n_molecules)
    failed_attempts = 0
    while length(coords) < (n_molecules * 2)
        new_coord_a = random_coord(boundary; rng=rng)
        if aligned
            shift = SVector{dims}([bond_length, [zero(bond_length) for d in 1:(dims - 1)]...])
        else
            shift = bond_length * normalize(randn(rng, SVector{dims, typeof(ustrip(bond_length))}))
        end
        new_coord_b = copy(new_coord_a) + shift
        okay = true
        if min_dist > zero(min_dist)
            for coord in coords
                if sum(abs2, vector(coord, new_coord_a, boundary)) < min_dist_sq ||
                        sum(abs2, vector(coord, new_coord_b, boundary)) < min_dist_sq
                    okay = false
                    failed_attempts += 1
                    break
                end
            end
        end
        if okay
            push!(coords, new_coord_a)
            push!(coords, new_coord_b)
            failed_attempts = 0
        elseif failed_attempts >= max_attempts
            error("failed to place atom $(length(coords) + 1) after $max_attempts (max_attempts) tries")
        end
    end
    # Second atom in each molecule may be outside boundary
    return wrap_coords.([coords...], (boundary,))
end

get_res_id(res) = (
    Chemfiles.id(res),
    "chainid" in Chemfiles.list_properties(res) ? Chemfiles.property(res, "chainid") : "X",
)

# Return the residue name with N or C added for terminal residues
function residue_name(res, res_id_to_standard::Dict, rename_terminal_res::Bool=true)
    res_id = get_res_id(res)
    res_name = Chemfiles.name(res)
    if rename_terminal_res && res_id_to_standard[res_id]
        prev_res_id = (res_id[1] - 1, res_id[2])
        next_res_id = (res_id[1] + 1, res_id[2])
        if !haskey(res_id_to_standard, prev_res_id) || !res_id_to_standard[prev_res_id]
            res_name = "N" * res_name
        elseif !haskey(res_id_to_standard, next_res_id) || !res_id_to_standard[next_res_id]
            res_name = "C" * res_name
        end
    end
    return res_name
end

atom_types_to_string(atom_types...) = join(map(at -> at == "" ? "-" : at, atom_types), "/")

atom_types_to_tuple(atom_types) = tuple(map(at -> at == "-" ? "" : at, split(atom_types, "/"))...)

const standard_res_names = [keys(BioStructures.threeletter_to_aa)..., "HID", "HIE", "HIP"]

# Versions of Chemfiles functions that don't make a copy
function chemfiles_residue(top, ai)
    ptr_raw = Chemfiles.lib.chfl_residue_from_topology(Chemfiles.__ptr(top), UInt64(ai))
    ptr = Chemfiles.@__check_ptr(ptr_raw)
    return Chemfiles.Residue(Chemfiles.CxxPointer(ptr, is_const=true))
end

function chemfiles_residue_for_atom(top, ai)
    ptr = Chemfiles.lib.chfl_residue_for_atom(Chemfiles.__ptr(top), UInt64(ai))
    return Chemfiles.Residue(Chemfiles.CxxPointer(ptr, is_const=true))
end

function chemfiles_name(top, ai)
    ptr_raw = Chemfiles.lib.chfl_atom_from_topology(Chemfiles.__ptr(top), UInt64(ai))
    ptr = Chemfiles.@__check_ptr(ptr_raw)
    at = Chemfiles.Atom(Chemfiles.CxxPointer(ptr, is_const=false))
    return Chemfiles.name(at)
end

# Creates a Dict representation of the system Chains -> Residues -> Graphs
# It is useful to have all the necessary data in one hashable object
# It also ensures that the correct names are used for downstream template matching
function canonicalize_system(top, resname_replacements, atomname_replacements)
    canon_system = Dict{String, Dict{Int, ResidueGraph}}()

    for ri in 1:Chemfiles.count_residues(top)
        res = chemfiles_residue(top, ri-1)
        res_id = get_res_id(res)
        res_name = Chemfiles.name(res)
        res_name = (haskey(resname_replacements, res_name) ? resname_replacements[res_name] : res_name)
        atom_inds_zero = Int.(Chemfiles.atoms(res))
        atom_inds = atom_inds_zero .+ 1
        atom_names = Molly.chemfiles_name.((top,), atom_inds_zero)
        atom_elements = Symbol[]
        for atom_idx in atom_inds_zero
            atom = Chemfiles.Atom(top, atom_idx)
            an = Int(Chemfiles.atomic_number(atom))
            # Chemfiles treats e.g. "C" but not "C2" as carbon
            # Here we search for elements after removing numbers, so "C2" is treated as carbon
            # Cases that are ambiguous, such as "CA" with calcium, are not assigned (i.e. X)
            if iszero(an)
                atom_name_nonum = replace(Chemfiles.name(atom), r"\d+" => "")
                element_symbol = Symbol(atom_name_nonum)
                if haskey(PeriodicTable.elements, element_symbol)
                    an = PeriodicTable.elements[element_symbol].number
                end
            end
            if iszero(an) # Extra particle returns 0 from chemfiles
                push!(atom_elements, :X)
            else
                elm_str = PeriodicTable.elements[an].symbol
                push!(atom_elements, Symbol(elm_str))
            end
        end
        if haskey(atomname_replacements, res_name)
            lookup = atomname_replacements[res_name]
            atom_names = [(haskey(lookup, a) ? (lookup[a], aidx+1) : (a, aidx+1))
                          for (a, aidx) in zip(atom_names, atom_inds_zero)]
        else
            atom_names = [(a, aidx+1) for (a, aidx) in zip(atom_names, atom_inds_zero)]
        end

        rgraph = ResidueGraph(res_name, atom_inds, [a[1] for a in atom_names], atom_elements,
                              Tuple{Int,Int}[], fill(0, length(atom_inds)))

        if !haskey(canon_system, res_id[2])
            canon_system[res_id[2]] = Dict(res_id[1] => rgraph)
        else
            canon_system[res_id[2]][res_id[1]] = rgraph
        end
    end
    return canon_system
end

function resolve_bond(ff::MolecularForceField, t1::AbstractString, t2::AbstractString)
    # Exact types first, both orders
    key = (t1, t2)
    if haskey(ff.bond_resolver.cache, key)
        return ff.bond_resolver.cache[key]
    end

    cand = Int[]
    append!(cand, get(ff.bond_resolver.idx, (:type,  t1, t2), Int[]))
    append!(cand, get(ff.bond_resolver.idx, (:type,  t2, t1), Int[]))
    c1 = get(ff.class_of, t1, "")
    c2 = get(ff.class_of, t2, "")
    append!(cand, get(ff.bond_resolver.idx, (:class, c1, c2), Int[]))
    append!(cand, get(ff.bond_resolver.idx, (:class, c2, c1), Int[]))
    append!(cand, get(ff.bond_resolver.idx, (:wild,  "", ""), Int[]))

    best = nothing
    bestspec = Int8(-1)
    for i in cand
        r = ff.bond_resolver.rules[i]
        if (matches(r.p1, t1, ff.class_of) && matches(r.p2, t2, ff.class_of)) ||
           (matches(r.p1, t2, ff.class_of) && matches(r.p2, t1, ff.class_of))
            if r.specificity > bestspec
                bestspec = r.specificity
                best = r.params
            end
        end
    end
    # Symmetric caching
    ff.bond_resolver.cache[(t1, t2)] = best
    ff.bond_resolver.cache[(t2, t1)] = best
    return best
end

function resolve_angle(ff::MolecularForceField, t1::AbstractString, t2::AbstractString,
                       t3::AbstractString)
    key = (t1, t2, t3)
    if haskey(ff.angle_resolver.cache, key)
        return ff.angle_resolver.cache[key]
    end

    cand = Int[]
    append!(cand, get(ff.angle_resolver.idx, (:type,  t2), Int[]))
    append!(cand, get(ff.angle_resolver.idx, (:class, get(ff.class_of, t2, "")), Int[]))
    append!(cand, get(ff.angle_resolver.idx, (:wild,  ""), Int[]))

    best = nothing
    bestspec = Int8(-1)
    for i in cand
        r = ff.angle_resolver.rules[i]
        if matches(r.p1,t1,ff.class_of) && matches(r.p2,t2,ff.class_of) &&
                                                    matches(r.p3,t3,ff.class_of)
            if r.specificity > bestspec
                bestspec = r.specificity
                best = r.params
            end
        end
        # Neighbor-reversed
        if matches(r.p1,t3,ff.class_of) && matches(r.p2,t2,ff.class_of) &&
                                                    matches(r.p3,t1,ff.class_of)
            if r.specificity > bestspec
                bestspec = r.specificity
                best = r.params
            end
        end
    end
    # Symmetric caching
    ff.angle_resolver.cache[(t1, t2, t3)] = best
    ff.angle_resolver.cache[(t3, t2, t1)] = best
    return best
end

function resolve_proper_torsion(ff::MolecularForceField, t1::AbstractString, t2::AbstractString,
                                t3::AbstractString, t4::AbstractString)
    # OpenMM-style lazy resolution via resolver
    p,  pspec  = find_proper_match(t1, t2, t3, t4; resolver=ff.torsion_resolver, class_of=ff.class_of)
    pr, prspec = find_proper_match(t4, t3, t2, t1; resolver=ff.torsion_resolver, class_of=ff.class_of)

    if !isnothing(p) && isnothing(pr)
        return (p, (t1, t2, t3, t4))
    elseif isnothing(p) && !isnothing(pr)
        return (pr, (t4, t3, t2, t1))
    elseif !isnothing(p) && !isnothing(pr)
        ret = (pspec > prspec ? (p, (t1, t2, t3, t4)) : (pr, (t4, t3, t2, t1)))
        return ret
    else
        return (nothing, ("", "", "", ""))
    end
end

function resolve_improper_torsion(ff::MolecularForceField, t1::AbstractString, t2::AbstractString,
                                  t3::AbstractString, t4::AbstractString)
    # Resolver scans all 6 permutations internally and caches the winner
    p = find_improper_match(t1, t2, t3, t4; resolver=ff.torsion_resolver, class_of=ff.class_of)
    if isnothing(p)
        return (nothing, ("", "", "", ""))
    end

    # Recover matched permutation from cache to return the oriented key
    ic = ff.torsion_resolver.improper_cache
    cache_hit = get(ic, (t1, t2, t3, t4), :miss)
    if cache_hit == :miss
        return (p, (t1, t2, t3, t4)) # Fallback
    else
        perm, _ = cache_hit
        src = (t1, t2, t3, t4)
        key = (src[perm[1]], src[perm[2]], src[perm[3]], src[perm[4]])
        return (p, key)
    end
end

# Map an atom name in a residue to the global atom index
function atom_name_to_global_i(atom_name, template_atoms, rgraph_atom_inds, matches)
    atom_name_ind = findfirst(isequal(atom_name), template_atoms)
    return rgraph_atom_inds[findfirst(isequal(atom_name_ind), matches)]
end

function add_virtual_sites!(virtual_sites, template, rgraph, matches)
    for vst in template.virtual_sites
        atom_ind = atom_name_to_global_i(vst.name       , template.atoms, rgraph.atom_inds, matches)
        atom_1   = atom_name_to_global_i(vst.atom_name_1, template.atoms, rgraph.atom_inds, matches)
        atom_2   = atom_name_to_global_i(vst.atom_name_2, template.atoms, rgraph.atom_inds, matches)
        atom_3   = atom_name_to_global_i(vst.atom_name_3, template.atoms, rgraph.atom_inds, matches)
        vs = VirtualSite(vst.type, atom_ind, atom_1, atom_2, atom_3, vst.weight_1, vst.weight_2,
                         vst.weight_3, vst.weight_12, vst.weight_13, vst.weight_cross)
        push!(virtual_sites, vs)
    end
    return virtual_sites
end

"""
    System(coordinate_file, force_field; <keyword arguments>)

Read a coordinate file in a file format readable by Chemfiles and apply a
force field to it.

Atom names should exactly match residue templates - no searching of residue
templates is carried out.

    System(coordinate_file, topology_file; <keyword arguments>)
    System(T, coordinate_file, topology_file; <keyword arguments>)

Read a Gromacs coordinate file and a Gromacs topology file with all
includes collapsed into one file.

Gromacs file reading should be considered experimental.

# Arguments
- `boundary=nothing`: the bounding box used for simulation, read from the
    file by default.
- `velocities=nothing`: the velocities of the atoms in the system, set to
    zero by default.
- `loggers=()`: the loggers that record properties of interest during a
    simulation.
- `units::Bool=true`: whether to use Unitful quantities.
- `array_type=Array`: the array type for the simulation, for example
    use `CuArray` or `ROCArray` for GPU support.
- `dist_cutoff=1.0u"nm"`: cutoff distance for long-range interactions.
- `dist_buffer=0.2u"nm"`: distance added to `dist_cutoff` when calculating
    neighbors every few steps. Not relevant if [`GPUNeighborFinder`](@ref) is
    used since the neighbors are calculated each step.
- `constraints=:none`: which constraints to apply during the simulation, options
    are `:none`, `:hbonds` (bonds involving hydrogen), `:allbonds` and `:hangles`
    (all bonds plus H-X-H and H-O-X angles). Note that not all options may be
    supported depending on the bonding topology.
- `rigid_water=false`: whether to constrain the bonds and angle in water
    molecules. Applied on top of `constraints`, so `constraints=:hangles` and
    `rigid_water=false` gives rigid water.
- `nonbonded_method=:none`: method for long range interaction summation,
    options are `:none` (short range only), `:cutoff` (reaction field method),
    `:pme` (particle mesh Ewald summation) and `:ewald` (Ewald summation, slow).
- `ewald_error_tol=0.0005`: the error tolerance for Ewald summation, used when
    `nonbonded_method` is `:pme` or `:ewald`.
- `approximate_pme=true`: whether to use a fast approximation to the erfc
    function, used when `nonbonded_method` is `:pme`.
- `center_coords::Bool=true`: whether to center the coordinates in the
    simulation box.
- `neighbor_finder_type`: which neighbor finder to use, default is
    [`CellListMapNeighborFinder`](@ref) on CPU, [`GPUNeighborFinder`](@ref)
    on CUDA compatible GPUs and [`DistanceNeighborFinder`](@ref) on non-CUDA
    compatible GPUs.
- `data=nothing`: arbitrary data associated with the system.
- `implicit_solvent=:none`: the implicit solvent model to use, options are
    `:none`, `:obc1`, `:obc2` and `:gbn2`.
- `kappa=0.0u"nm^-1"`: the kappa value for the implicit solvent model if one
    is used.
- `disulfide_bonds=true`: whether or not to look for disulfide bonds between CYS
    residues in the structure file and add them to the topology. Uses geometric
    arguments to assign them.
- `grad_safe=false`: should be set to `true` if the system is going to be used
    with Enzyme.jl and `nonbonded_method` is `:pme`.
- `strictness=:warn`: determines behavior when encountering possible problems,
    options are `:warn` to emit warnings, `:nowarn` to suppress warnings or
    `:error` to error.
"""
function System(coord_file::AbstractString,
                force_field::MolecularForceField;
                boundary=nothing,
                velocities=nothing,
                loggers=(),
                units::Bool=true,
                array_type::Type{AT}=Array,
                dist_cutoff=(units ? 1.0u"nm" : 1.0),
                dist_buffer=(units ? 0.2u"nm" : 0.2),
                constraints=:none,
                rigid_water=false,
                nonbonded_method=:none,
                ewald_error_tol=0.0005,
                approximate_pme=true,
                center_coords::Bool=true,
                neighbor_finder_type=nothing,
                data=nothing,
                implicit_solvent=:none,
                kappa=0.0u"nm^-1",
                disulfide_bonds=true,
                grad_safe::Bool=false,
                strictness=:warn,
                rename_terminal_res=nothing) where {AT <: AbstractArray}
    if !isnothing(rename_terminal_res)
        @info "rename_terminal_res is no longer required and will be removed in a future breaking release"
    end
    check_strictness(strictness)
    if dist_buffer < zero(dist_buffer)
        throw(ArgumentError("dist_buffer ($dist_buffer) should not be less than zero"))
    end
    dist_neighbors = dist_cutoff + dist_buffer
    T = typeof(force_field.weight_14_coulomb)
    IC = (units ? typeof(zero(T) * u"nm^-1") : T)

    resname_replacements  = force_field.residue_name_replacements
    atomname_replacements = force_field.atom_name_replacements
    standard_bonds        = force_field.standard_bonds

    # Read structure
    traj = Chemfiles.Trajectory(coord_file)
    frame = Chemfiles.read(traj)
    top = Chemfiles.Topology(frame)
    n_atoms = size(top)

    # Boundary
    if isnothing(boundary)
        boundary_used = boundary_from_chemfiles(Chemfiles.UnitCell(frame), T,
                                                            (units ? u"nm" : NoUnits))
    else
        boundary_used = boundary
    end
    min_box_side = minimum(box_sides(boundary_used))
    if min_box_side < (2 * dist_cutoff)
        err_str = "Minimum box side ($min_box_side) is less than 2 * dist_cutoff " *
                  "($(2 * dist_cutoff)), this can lead to unphysical simulations" *
                  "since multiple copies of the same atom are seen but only one is " *
                  "considered due to the minimum image convention"
        report_issue(err_str, strictness)
    end

    # Units and coordinates
    if units
        coords = [T.(SVector{3}(col)u"nm" / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    else
        coords = [T.(SVector{3}(col) / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    end
    if center_coords
        coords = coords .- (mean(coords),) .+ (box_center(boundary_used),)
    end
    coords = wrap_coords.(coords, (boundary_used,))

    canonical_system = canonicalize_system(top, resname_replacements, atomname_replacements)

    top_bonds = create_bonds!(canonical_system, standard_bonds)
    if disulfide_bonds
        top_bonds = create_disulfide_bonds(coords, boundary_used, canonical_system, top_bonds)
    end
    top_bonds = read_extra_bonds!(canonical_system, top, top_bonds)

    template_names = keys(force_field.residues)
    # Match each residue graph to a template and assign atom types/charges
    atom_type_of = Vector{String}(undef, n_atoms)
    charge_of = Vector{Union{T, Missing}}(undef, n_atoms)
    element_of = Vector{String}(undef, n_atoms)
    use_charge_from_residue = ("charge" in force_field.attributes_from_residue)

    virtual_sites = VirtualSite{T, IC}[]
    for (chain, resids) in canonical_system
        for (res_id, rgraph) in resids
            matched = false
            if rgraph.res_name in template_names
                template = force_field.residues[rgraph.res_name]
                matches = match_residue_to_template(rgraph, template)
                if isnothing(matches)
                    for (templ_name, template) in force_field.residues
                        # Dont check it again
                        if rgraph.res_name == templ_name
                            continue
                        end
                        matches = match_residue_to_template(rgraph, template)
                        if !isnothing(matches)
                            matched = true
                            for (r_i, m_i) in enumerate(matches)
                                global_idx = rgraph.atom_inds[r_i]
                                atom_type_of[global_idx] = template.types[m_i]
                                charge_of[global_idx] = template.charges[m_i]
                                element_of[global_idx] = force_field.atom_types[template.types[m_i]].element
                            end
                            add_virtual_sites!(virtual_sites, template, rgraph, matches)
                            break
                        end
                    end
                else
                    matched = true
                    for (r_i, m_i) in enumerate(matches)
                        global_idx = rgraph.atom_inds[r_i]
                        atom_type_of[global_idx] = template.types[m_i]
                        charge_of[global_idx] = template.charges[m_i]
                        element_of[global_idx] = force_field.atom_types[template.types[m_i]].element
                    end
                    add_virtual_sites!(virtual_sites, template, rgraph, matches)
                end
            else
                for (templ_name, template) in force_field.residues
                    matches = match_residue_to_template(rgraph, template)
                    if !isnothing(matches)
                        matched = true
                        for (r_i, m_i) in enumerate(matches)
                            global_idx = rgraph.atom_inds[r_i]
                            atom_type_of[global_idx] = template.types[m_i]
                            charge_of[global_idx] = template.charges[m_i]
                            element_of[global_idx] = force_field.atom_types[template.types[m_i]].element
                        end
                        add_virtual_sites!(virtual_sites, template, rgraph, matches)
                        break
                    end
                end
            end
            if !matched
                throw(ArgumentError("could not match residue $(rgraph.res_name) to any of " *
                                    "the provided templates, make sure that the atoms match " *
                                    "and have elements assigned"))
            end
        end
    end
    virtual_sites_type = (length(virtual_sites) > 0 ? virtual_sites : [])

    adj           = build_adjacency(n_atoms, top_bonds)
    top_angles    = build_angles(adj, top_bonds)
    top_torsions  = build_torsions(adj, top_angles)
    top_impropers = build_impropers(adj)

    # Allocate interaction lists and particles
    atoms_abst = Atom[]
    atoms_data = AtomData[]
    bonds_il   = InteractionList2Atoms(HarmonicBond)
    angles_il  = InteractionList3Atoms(HarmonicAngle)
    tors_il    = InteractionList4Atoms(PeriodicTorsion)
    imps_il    = InteractionList4Atoms(PeriodicTorsion)
    eligible = trues(n_atoms, n_atoms)
    special  = falses(n_atoms, n_atoms)
    torsion_n_terms = 6
    weight_14_coulomb, weight_14_lj = force_field.weight_14_coulomb, force_field.weight_14_lj

    # Atoms
    for ai in 1:n_atoms
        atype = atom_type_of[ai]
        at = force_field.atom_types[atype]
        if (units && at.σ < zero(T)u"nm") || (!units && at.σ < zero(T))
            error("atom $ai type $atype has unset σ or ϵ")
        end
        if use_charge_from_residue
            ch = charge_of[ai]
            if ismissing(ch)
                error("atom $ai type $atype has charge missing from residue template")
            end
        else
            ch = force_field.atom_types[atype].charge
            if ismissing(ch)
                error("atom $ai type $atype has charge missing")
            end
        end
        push!(atoms_abst, Atom(index=ai, mass=at.mass, charge=ch, σ=at.σ, ϵ=at.ϵ))

        res = residue_from_atom_idx(ai, canonical_system)
        res_cfl = chemfiles_residue_for_atom(top, ai - 1)
        if "is_standard_pdb" in Chemfiles.list_properties(res_cfl)
            hetero = !Chemfiles.property(res_cfl, "is_standard_pdb")
        else
            hetero = false
        end
        push!(atoms_data, AtomData(atom_type=atype, atom_name=atom_name_from_index(ai, canonical_system),
                                   res_number=resnum_from_atom_idx(ai, canonical_system), res_name=res.res_name,
                                   chain_id=chain_from_atom_idx(ai, canonical_system), element=element_of[ai], hetero_atom=hetero))
        eligible[ai, ai] = false
    end
    atoms = to_device([atoms_abst...], AT)

    # Bonds
    for (i, j) in top_bonds
        t1, t2 = atom_type_of[i], atom_type_of[j]
        hb = resolve_bond(force_field, t1, t2)
        if isnothing(hb)
            throw(ArgumentError("no bond parameters found for ($t1, $t2)"))
        end
        push!(bonds_il.is, i)
        push!(bonds_il.js, j)
        push!(bonds_il.types, atom_types_to_string(t1,t2))
        push!(bonds_il.inters, hb)
        eligible[i, j] = false
        eligible[j, i] = false
    end

    # Angles
    for (i, j, k) in top_angles
        t1,t2,t3 = atom_type_of[i], atom_type_of[j], atom_type_of[k]
        ha = resolve_angle(force_field, t1,t2,t3)
        if isnothing(ha)
            throw(ArgumentError("no angle parameters found for ($t1, $t2, $t3)"))
        end
        push!(angles_il.is,i)
        push!(angles_il.js,j)
        push!(angles_il.ks,k)
        push!(angles_il.types, atom_types_to_string(t1,t2,t3))
        push!(angles_il.inters, ha)
        eligible[i, k] = false
        eligible[k, i] = false
    end

    # Virtual sites share all the non-bonded exclusions of, and are excluded from,
    #   their parent atoms
    for vs in virtual_sites
        i = vs.atom_ind
        for j in (vs.atom_1, vs.atom_2, vs.atom_3)
            if !iszero(j)
                for k in 1:n_atoms
                    if !eligible[j, k]
                        eligible[i, k] = false
                        eligible[k, i] = false
                    end
                end
                eligible[i, j] = false
                eligible[j, i] = false
            end
        end
    end

    # Proper torsions
    for (i,j,k,l) in top_torsions
        t1,t2,t3,t4 = atom_type_of[i], atom_type_of[j], atom_type_of[k], atom_type_of[l]
        tt, key = resolve_proper_torsion(force_field, t1, t2, t3, t4)
        isnothing(tt) && continue

        n_terms = length(tt.periodicities)
        for s in 1:torsion_n_terms:n_terms
            e = min(s+torsion_n_terms-1, n_terms)
            push!(tors_il.is, i)
            push!(tors_il.js, j)
            push!(tors_il.ks, k)
            push!(tors_il.ls, l)
            push!(tors_il.types, atom_types_to_string(key...))
            push!(tors_il.inters, PeriodicTorsion(periodicities=tt.periodicities[s:e],
                                                phases=tt.phases[s:e], ks=tt.ks[s:e], proper=true))
        end
        special[i, l] = true
        special[l, i] = true
    end

    # Impropers (Amber ordering)
    for (c, j, k, l) in top_impropers
        t1, t2, t3, t4 = atom_type_of[c], atom_type_of[j], atom_type_of[k], atom_type_of[l]

        # Resolve improper params and oriented key (central first)
        tt, key = resolve_improper_torsion(force_field, t1,t2,t3,t4)
        isnothing(tt) && continue

        # Recover metadata from resolver cache
        ic = force_field.torsion_resolver.improper_cache
        hit = get(ic, (t1, t2, t3, t4), :miss)
        ordering::String = "default"
        has_wild::Bool = false
        if hit !== :miss
            perm, ridx = hit
            r = force_field.torsion_resolver.rules[ridx]
            ordering = r.ordering
            has_wild = r.has_wildcard

            # Reorder indices based on how atoms were permuted
            src_atoms = (c, j, k, l)
            j = src_atoms[perm[2]]
            k = src_atoms[perm[3]]
            l = src_atoms[perm[4]]

            # refresh types after remapping
            t2, t3, t4 = atom_type_of[j], atom_type_of[k], atom_type_of[l]
        end

        # topology indices for current j,k,l
        r2 = resnum_from_atom_idx(j, canonical_system)
        r3 = resnum_from_atom_idx(k, canonical_system)
        r4 = resnum_from_atom_idx(l, canonical_system)

        res2 = residue_from_atom_idx(j, canonical_system)
        res3 = residue_from_atom_idx(k, canonical_system)
        res4 = residue_from_atom_idx(l, canonical_system)

        ta2 = findfirst(isequal(j), res2.atom_inds)
        ta3 = findfirst(isequal(k), res3.atom_inds)
        ta4 = findfirst(isequal(l), res4.atom_inds)

        e2 = Symbol(element_of[j])
        e3 = Symbol(element_of[k])
        e4 = Symbol(element_of[l])

        if ordering == "amber"
            # OpenMM amber branch, with/without wildcards
            if !has_wild
                if t2 == t4 && (r2 > r4 || (r2 == r4 && ta2 > ta4))
                    (j,   l)   = (l,   j)
                    (r2,  r4)  = (r4,  r2)
                    (ta2, ta4) = (ta4, ta2)
                end
                if t3 == t4 && (r3 > r4 || (r3 == r4 && ta3 > ta4))
                    (k,   l)   = (l,   k)
                    (r3,  r4)  = (r4,  r3)
                    (ta3, ta4) = (ta4, ta3)
                end
                if t2 == t3 && (r2 > r3 || (r2 == r3 && ta2 > ta3))
                    (j, k) = (k, j)
                end
            else
                if e2 == e4 && (r2 > r4 || (r2 == r4 && ta2 > ta4))
                    (j,   l)   = (l,   j)
                    (r2,  r4)  = (r4,  r2)
                    (ta2, ta4) = (ta4, ta2)
                end
                if e3 == e4 && (r3 > r4 || (r3 == r4 && ta3 > ta4))
                    (k,   l)   = (l,   k)
                    (r3,  r4)  = (r4,  r3)
                    (ta3, ta4) = (ta4, ta3)
                end
                if r2 > r3 || (r2 == r3 && ta2 > ta3)
                    (j, k) = (k, j)
                end
            end
        elseif ordering == "charmm"
            # If wildcards were used then apply the same Amber tie-break, else unambiguous
            if has_wild
                if e2 == e4 && (r2 > r4 || (r2 == r4 && ta2 > ta4))
                    (j,   l)   = (l,   j)
                    (r2,  r4)  = (r4,  r2)
                    (ta2, ta4) = (ta4, ta2)
                end
                if e3 == e4 && (r3 > r4 || (r3 == r4 && ta3 > ta4))
                    (k,   l)   = (l,   k)
                    (r3,  r4)  = (r4,  r3)
                    (ta3, ta4) = (ta4, ta3)
                end
            end
        elseif ordering == "smirnoff"
            # Add the trefoil set
            a1, a2, a3, a4 = c, j, k, l
            for (x1, x2, x3, x4) in ((a1,a2,a3,a4),
                                     (a1,a3,a4,a2),
                                     (a1,a4,a2,a3))
                p1, p2, cen, p3 = x2, x3, x1, x4
                push!(imps_il.is, p1)
                push!(imps_il.js, p2)
                push!(imps_il.ks, cen)
                push!(imps_il.ls, p3)
                push!(imps_il.types, atom_types_to_string(key...))
                push!(imps_il.inters, PeriodicTorsion(periodicities=tt.periodicities,
                                            phases=tt.phases, ks=tt.ks, proper=false))
            end
            continue # Skip the single-add fallback below
        else
            # ordering == "default"
            # Only if a wildcard is present
            if has_wild
                # Mirror the permutation on the current topology atoms (c,j,k,l)
                src_atoms = (c, j, k, l)

                # We need the two peripheral atoms in positions 2 and 3, and the remaining
                #   peripheral in 4
                a1 = src_atoms[perm[2]]
                a2 = src_atoms[perm[3]]
                a4 = src_atoms[perm[4]]

                # Elements and masses for tie-break
                e_a1 = Symbol(element_of[a1])
                e_a2 = Symbol(element_of[a2])
                m_a1 = force_field.atom_types[atom_type_of[a1]].mass
                m_a2 = force_field.atom_types[atom_type_of[a2]].mass

                # 1) If same element, lower atom index first
                # 2) Else, prefer carbon; else heavier mass first
                if e_a1 == e_a2
                    if a1 > a2
                        (a1, a2) = (a2, a1)
                    end
                elseif !(e_a1 == :C) && (e_a2 == :C || m_a1 < m_a2)
                    (a1, a2) = (a2, a1)
                end

                # Reassign current triplet to ordered pair and remaining peripheral
                j, k, l = a1, a2, a4
            end
            # If no wildcard leave j, k, l as-is
        end

        push!(imps_il.is, j)
        push!(imps_il.js, k)
        push!(imps_il.ks, c)
        push!(imps_il.ls, l)
        push!(imps_il.types, atom_types_to_string(key...))
        push!(imps_il.inters, PeriodicTorsion(periodicities=tt.periodicities,
                                              phases=tt.phases, ks=tt.ks, proper=false))
    end

    tors_pad = [PeriodicTorsion(periodicities=t.periodicities, phases=t.phases, ks=t.ks,
                                proper=t.proper, n_terms=torsion_n_terms) for t in tors_il.inters]
    imps_pad = [PeriodicTorsion(periodicities=t.periodicities, phases=t.phases, ks=t.ks,
                                proper=t.proper, n_terms=torsion_n_terms) for t in imps_il.inters]

    return System(T, AT, atoms, coords, boundary_used, velocities,
                  atoms_data, virtual_sites_type, loggers, data, bonds_il, angles_il, tors_il,
                  imps_il, tors_pad, imps_pad, eligible, special, units, dist_cutoff,
                  constraints, rigid_water, nonbonded_method, ewald_error_tol, approximate_pme,
                  neighbor_finder_type, implicit_solvent, kappa, grad_safe, dist_neighbors,
                  weight_14_lj, weight_14_coulomb, strictness)
end

function element_from_mass(atom_mass, element_names, element_masses)
    atom_mass_nounits = ustrip(atom_mass)
    el = "?"
    for (el_name, el_mass) in zip(element_names, element_masses)
        if isapprox(atom_mass_nounits * u"u", el_mass; atol=0.01u"u")
            el = el_name
            break
        end
    end
    return el
end

function System(T::Type,
                coord_file::AbstractString,
                top_file::AbstractString;
                boundary=nothing,
                velocities=nothing,
                loggers=(),
                units::Bool=true,
                array_type::Type{AT}=Array,
                dist_cutoff=(units ? 1.0u"nm" : 1.0),
                dist_buffer=(units ? 0.2u"nm" : 0.2),
                constraints=:none,
                rigid_water=false,
                nonbonded_method=:none,
                ewald_error_tol=0.0005,
                approximate_pme=true,
                center_coords::Bool=true,
                neighbor_finder_type=nothing,
                data=nothing,
                implicit_solvent=:none,
                kappa=0.0u"nm^-1",
                grad_safe::Bool=false) where AT <: AbstractArray
    if dist_buffer < zero(dist_buffer)
        throw(ArgumentError("dist_buffer ($dist_buffer) should not be less than zero"))
    end
    dist_neighbors = dist_cutoff + dist_buffer

    # Read force field and topology file
    atomtypes = Dict{String, Atom}()
    bondtypes = Dict{String, HarmonicBond}()
    angletypes = Dict{String, HarmonicAngle}()
    torsiontypes = Dict{String, RBTorsion}()
    atomnames = Dict{String, String}()

    name = "?"
    atoms_abst = Atom[]
    atoms_data = AtomData[]
    bonds = InteractionList2Atoms(HarmonicBond)
    pairs = Tuple{Int, Int}[]
    angles = InteractionList3Atoms(HarmonicAngle)
    possible_torsions = Tuple{Int, Int, Int, Int}[]
    torsions = InteractionList4Atoms(RBTorsion)
    impropers = InteractionList4Atoms(RBTorsion)
    torsion_n_terms = 6
    weight_14_lj, weight_14_coulomb = T(0.5), T(0.5)

    if units
        force_units = u"kJ * mol^-1 * nm^-1"
        energy_units = u"kJ * mol^-1"
    else
        force_units = NoUnits
        energy_units = NoUnits
    end

    element_names  = [el.symbol      for el in PeriodicTable.elements]
    element_masses = [el.atomic_mass for el in PeriodicTable.elements]

    current_field = ""
    for l in eachline(top_file)
        sl = strip(l)
        if iszero(length(sl)) || startswith(sl, ';')
            continue
        end
        if startswith(sl, '[') && endswith(sl, ']')
            current_field = strip(sl[2:end-1])
            continue
        end
        c = split(rstrip(first(split(sl, ";", limit=2))), r"\s+")
        if current_field == "bondtypes"
            if units
                bondtype = HarmonicBond(parse(T, c[5])u"kJ * mol^-1 * nm^-2", parse(T, c[4])u"nm")
            else
                bondtype = HarmonicBond(parse(T, c[5]), parse(T, c[4]))
            end
            bondtypes["$(c[1])/$(c[2])"] = bondtype
            bondtypes["$(c[2])/$(c[1])"] = bondtype
        elseif current_field == "angletypes"
            # Convert θ0 to radians
            if units
                angletype = HarmonicAngle(parse(T, c[6])u"kJ * mol^-1", deg2rad(parse(T, c[5])))
            else
                angletype = HarmonicAngle(parse(T, c[6]), deg2rad(parse(T, c[5])))
            end
            angletypes["$(c[1])/$(c[2])/$(c[3])"] = angletype
            angletypes["$(c[3])/$(c[2])/$(c[1])"] = angletype
        elseif current_field == "dihedraltypes" && c[1] != "#define"
            # Convert back to OPLS types
            f4 = parse(T, c[10]) / -4
            f3 = parse(T, c[9]) / -2
            f2 = 4 * f4 - parse(T, c[8])
            f1 = 3 * f3 - 2 * parse(T, c[7])
            if units
                torsiontype = RBTorsion((f1)u"kJ * mol^-1", (f2)u"kJ * mol^-1",
                                        (f3)u"kJ * mol^-1", (f4)u"kJ * mol^-1")
            else
                torsiontype = RBTorsion(f1, f2, f3, f4)
            end
            torsiontypes["$(c[1])/$(c[2])/$(c[3])/$(c[4])"] = torsiontype
        elseif current_field == "atomtypes" && length(c) >= 8
            atomname = uppercase(c[2])
            atomnames[c[1]] = atomname
            # Take the first version of each atom type only
            if !haskey(atomtypes, atomname)
                if units
                    atomtypes[atomname] = Atom(
                        mass=parse(T, c[4])u"g/mol",
                        charge=parse(T, c[5]),
                        σ=parse(T, c[7])u"nm",
                        ϵ=parse(T, c[8])u"kJ * mol^-1",
                    )
                else
                    atomtypes[atomname] = Atom(
                        mass=parse(T, c[4]),
                        charge=parse(T, c[5]),
                        σ=parse(T, c[7]),
                        ϵ=parse(T, c[8]),
                    )
                end
            end
        elseif current_field == "atoms"
            attype = atomnames[c[2]]
            ch = parse(T, c[7])
            if units
                atom_mass = parse(T, c[8])u"g/mol"
            else
                atom_mass = parse(T, c[8])
            end
            atom_index = length(atoms_abst) + 1
            el = element_from_mass(atom_mass, element_names, element_masses)
            push!(atoms_abst, Atom(index=atom_index, mass=atom_mass, charge=ch, σ=atomtypes[attype].σ,
                                ϵ=atomtypes[attype].ϵ))
            push!(atoms_data, AtomData(atom_type=attype, atom_name=c[5], res_number=parse(Int, c[3]),
                                        res_name=c[4], element=el))
        elseif current_field == "bonds"
            i, j = parse.(Int, c[1:2])
            bn = "$(atoms_data[i].atom_type)/$(atoms_data[j].atom_type)"
            bondtype = bondtypes[bn]
            push!(bonds.is, i)
            push!(bonds.js, j)
            push!(bonds.types, bn)
            push!(bonds.inters, HarmonicBond(k=bondtype.k, r0=bondtype.r0))
        elseif current_field == "pairs"
            push!(pairs, (parse(Int, c[1]), parse(Int, c[2])))
        elseif current_field == "angles"
            i, j, k = parse.(Int, c[1:3])
            an = "$(atoms_data[i].atom_type)/$(atoms_data[j].atom_type)/$(atoms_data[k].atom_type)"
            angletype = angletypes[an]
            push!(angles.is, i)
            push!(angles.js, j)
            push!(angles.ks, k)
            push!(angles.types, an)
            push!(angles.inters, HarmonicAngle(k=angletype.k, θ0=angletype.θ0))
        elseif current_field == "dihedrals"
            i, j, k, l = parse.(Int, c[1:4])
            push!(possible_torsions, (i, j, k, l))
        elseif current_field == "system"
            name = rstrip(first(split(sl, ";", limit=2)))
        end
    end

    # Add torsions based on wildcard torsion types
    for inds in possible_torsions
        at_types = [atoms_data[x].atom_type for x in inds]
        desired_key = join(at_types, "/")
        if haskey(torsiontypes, desired_key)
            d = torsiontypes[desired_key]
            push!(torsions.is, inds[1])
            push!(torsions.js, inds[2])
            push!(torsions.ks, inds[3])
            push!(torsions.ls, inds[4])
            push!(torsions.types, desired_key)
            push!(torsions.inters, RBTorsion(f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4))
        else
            best_score = 0
            best_key = ""
            for k in keys(torsiontypes)
                c = split(k, "/")
                for a in (c, reverse(c))
                    valid = true
                    score = 0
                    for (i, v) in enumerate(a)
                        if v == at_types[i]
                            score += 1
                        elseif v != "X"
                            valid = false
                            break
                        end
                    end
                    if valid && (score > best_score)
                        best_score = score
                        best_key = k
                    end
                end
            end
            # If a wildcard match is found, add a new specific torsion type
            if best_key != ""
                d = torsiontypes[best_key]
                push!(torsions.is, inds[1])
                push!(torsions.js, inds[2])
                push!(torsions.ks, inds[3])
                push!(torsions.ls, inds[4])
                push!(torsions.types, best_key)
                push!(torsions.inters, RBTorsion(f1=d.f1, f2=d.f2, f3=d.f3, f4=d.f4))
            end
        end
    end

    # Read coordinate file and add solvent atoms
    lines = readlines(coord_file)

    if isnothing(boundary)
        box_size_vals = SVector{3}(parse.(T, split(strip(lines[end]), r"\s+")))
        box_size = (units ? (box_size_vals)u"nm" : box_size_vals)
        boundary_used = CubicBoundary(box_size)
    else
        boundary_used = boundary
    end
    min_box_side = minimum(box_sides(boundary_used))
    if min_box_side < (2 * dist_cutoff)
        @warn "Minimum box side ($min_box_side) is less than 2 * dist_cutoff " *
              "($(2 * dist_cutoff)), this can lead to unphysical simulations" *
              "since multiple copies of the same atom are seen but only one is " *
              "considered due to the minimum image convention"
    end

    coords_abst = SArray[]
    for (i, l) in enumerate(lines[3:end-1])
        coord = SVector(parse(T, l[21:28]), parse(T, l[29:36]), parse(T, l[37:44]))
        if units
            push!(coords_abst, (coord)u"nm")
        else
            push!(coords_abst, coord)
        end

        # Some atoms are not specified explicitly in the topology so are added here
        if i > length(atoms_abst)
            atname = strip(l[11:15])
            attype = replace(atname, r"\d+" => "")
            temp_charge = atomtypes[attype].charge
            if attype == "CL" # Temp hack to fix charges
                temp_charge = T(-1.0)
            end
            atom_mass = atomtypes[attype].mass
            atom_index = length(atoms_abst) + 1
            el = element_from_mass(atom_mass, element_names, element_masses)
            push!(atoms_abst, Atom(index=atom_index, mass=atom_mass, charge=temp_charge,
                                σ=atomtypes[attype].σ, ϵ=atomtypes[attype].ϵ))
            push!(atoms_data, AtomData(atom_type=attype, atom_name=atname, res_number=parse(Int, l[1:5]),
                                        res_name=strip(l[6:10]), element=el))

            # Add O-H bonds and H-O-H angle in water
            if atname == "OW"
                bondtype = bondtypes["OW/HW"]
                push!(bonds.is, i)
                push!(bonds.js, i + 1)
                push!(bonds.types, "OW/HW")
                push!(bonds.inters, HarmonicBond(k=bondtype.k, r0=bondtype.r0))
                push!(bonds.is, i)
                push!(bonds.js, i + 2)
                push!(bonds.types, "OW/HW")
                push!(bonds.inters, HarmonicBond(k=bondtype.k, r0=bondtype.r0))
                angletype = angletypes["HW/OW/HW"]
                push!(angles.is, i + 1)
                push!(angles.js, i)
                push!(angles.ks, i + 2)
                push!(angles.types, "HW/OW/HW")
                push!(angles.inters, HarmonicAngle(k=angletype.k, θ0=angletype.θ0))
            end
        end
    end
    atoms = to_device([atoms_abst...], AT)

    # Calculate matrix of pairs eligible for non-bonded interactions
    n_atoms = length(coords_abst)
    eligible = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end
    for (i, j) in zip(bonds.is, bonds.js)
        eligible[i, j] = false
        eligible[j, i] = false
    end
    for (i, k) in zip(angles.is, angles.ks)
        # Assume bonding is already specified
        eligible[i, k] = false
        eligible[k, i] = false
    end

    # Calculate matrix of pairs eligible for halved non-bonded interactions
    # This applies to specified pairs in the topology file, usually 1-4 bonded
    special = falses(n_atoms, n_atoms)
    for (i, j) in pairs
        special[i, j] = true
        special[j, i] = true
    end

    coords = [coords_abst...]
    if center_coords
        coords = coords .- (mean(coords),) .+ (box_center(boundary_used),)
    end
    coords = wrap_coords.(coords, (boundary_used,))

    torsion_inters_pad = torsions.inters
    improper_inters_pad = impropers.inters
    virtual_sites = []
    strictness = :warn

    return System(T, AT, atoms, coords, boundary_used, velocities, atoms_data, virtual_sites,
                  loggers, data, bonds, angles, torsions, impropers, torsion_inters_pad,
                  improper_inters_pad, eligible, special, units, dist_cutoff, constraints,
                  rigid_water, nonbonded_method, ewald_error_tol, approximate_pme,
                  neighbor_finder_type, implicit_solvent, kappa, grad_safe, dist_neighbors,
                  weight_14_lj, weight_14_coulomb, strictness)
end

function System(coord_file::AbstractString, top_file::AbstractString; kwargs...)
    return System(DefaultFloat, coord_file, top_file; kwargs...)
end

const water_residue_names = ("SOL", "WAT", "HOH", "H2O")

# H-X-H and H-O-X angles, where X is any element
function is_h_angle(atoms_data, i, j, k)
    el_i, el_j, el_k = atoms_data[i].element, atoms_data[j].element, atoms_data[k].element
    return (el_i == "H" && el_k == "H") || (el_j == "O" && (el_i == "H" || el_k == "H"))
end

function find_bond_r0(bonds_all, i, j)
    for (bi, bj, inter) in zip(bonds_all.is, bonds_all.js, bonds_all.inters)
        if (bi, bj) == (i, j) || (bj, bi) == (i, j)
            return inter.r0
        end
    end
    error("atoms $i and $j are in an angle constraint but the bond cannot be found")
end

function exchange_constraints(T, bonds_all, angles_all, atoms_data, constraints_type,
                              rigid_water, units, strictness)
    if (constraints_type == :none && !rigid_water) || iszero(length(bonds_all.is))
        return (), bonds_all, angles_all
    end

    bonds = InteractionList2Atoms(HarmonicBond)
    angles = InteractionList3Atoms(HarmonicAngle)
    dist_constraints, angle_constraints = [], []
    angle_dist_pairs = Set{Tuple{Int, Int}}()

    for (i, j, k, inter, type) in zip(angles_all.is, angles_all.js, angles_all.ks,
                                      angles_all.inters, angles_all.types)
        if (constraints_type == :hangles && is_h_angle(atoms_data, i, j, k)) ||
                (rigid_water && atoms_data[i].res_name in water_residue_names)
            r0_ij = find_bond_r0(bonds_all, i, j)
            r0_jk = find_bond_r0(bonds_all, j, k)
            push!(angle_constraints, AngleConstraint(i, j, k, inter.θ0, r0_ij, r0_jk))
            push!(angle_dist_pairs, (min(i, j), max(i, j)))
            push!(angle_dist_pairs, (min(j, k), max(j, k)))
        else
            push!(angles.is, i)
            push!(angles.js, j)
            push!(angles.ks, k)
            push!(angles.inters, inter)
            push!(angles.types, type)
        end
    end

    for (i, j, inter, type) in zip(bonds_all.is, bonds_all.js, bonds_all.inters, bonds_all.types)
        if constraints_type in (:allbonds, :hangles) ||
                (constraints_type == :hbonds && (atoms_data[i].element == "H" || atoms_data[j].element == "H")) ||
                (rigid_water && atoms_data[i].res_name in water_residue_names)
            if !((min(i, j), max(i, j)) in angle_dist_pairs)
                # Only add distance constraints that are not part of an angle constraint
                push!(dist_constraints, DistanceConstraint(i, j, inter.r0))
            end
        else
            push!(bonds.is, i)
            push!(bonds.js, j)
            push!(bonds.inters, inter)
            push!(bonds.types, type)
        end
    end

    if length(dist_constraints) > 0 || length(angle_constraints) > 0
        shake = SHAKE_RATTLE(
            length(atoms_data),
            (units ? T(1e-6)u"nm" : T(1e-6)),
            (units ? T(1e-6)u"nm^2 * ps^-1" : T(1e-6));
            dist_constraints=[dist_constraints...],
            angle_constraints=[angle_constraints...],
            strictness=strictness,
        )
        constraints = (shake,)
    else
        constraints = ()
    end
    return constraints, bonds, angles
end

function System(T, AT, atoms, coords, boundary_used, velocities, atoms_data, virtual_sites,
                loggers, data, bonds_all, angles_all, torsions, impropers, torsion_inters_pad,
                improper_inters_pad, eligible, special, units, dist_cutoff, constraints_type,
                rigid_water, nonbonded_method, ewald_error_tol, approximate_pme,
                neighbor_finder_type, implicit_solvent, kappa, grad_safe, dist_neighbors,
                weight_14_lj, weight_14_coulomb, strictness)
    coords_dev = to_device(coords, AT)
    using_neighbors = (neighbor_finder_type != NoNeighborFinder)
    lj = LennardJones(
        cutoff=DistanceCutoff(T(dist_cutoff)),
        use_neighbors=using_neighbors,
        weight_special=weight_14_lj,
    )
    if nonbonded_method == :none
        coul = Coulomb(
            cutoff=DistanceCutoff(T(dist_cutoff)),
            use_neighbors=using_neighbors,
            weight_special=weight_14_coulomb,
            coulomb_const=(units ? T(coulomb_const) : T(ustrip(coulomb_const))),
        )
        general_inters_ewald = ()
    elseif nonbonded_method == :cutoff
        coul = CoulombReactionField(
            dist_cutoff=T(dist_cutoff),
            solvent_dielectric=T(crf_solvent_dielectric),
            use_neighbors=using_neighbors,
            weight_special=weight_14_coulomb,
            coulomb_const=(units ? T(coulomb_const) : T(ustrip(coulomb_const))),
        )
        general_inters_ewald = ()
    elseif nonbonded_method in (:ewald, :pme)
        coul = CoulombEwald(
            dist_cutoff=T(dist_cutoff),
            error_tol=T(ewald_error_tol),
            use_neighbors=using_neighbors,
            weight_special=weight_14_coulomb,
            coulomb_const=(units ? T(coulomb_const) : T(ustrip(coulomb_const))),
            approximate_erfc=approximate_pme,
        )
        if nonbonded_method == :ewald
            ewald = Ewald(
                T(dist_cutoff);
                error_tol=T(ewald_error_tol),
                eligible=eligible,
                special=special,
            )
        else
            ewald = PME(
                T(dist_cutoff),
                atoms,
                boundary_used;
                error_tol=T(ewald_error_tol),
                eligible=eligible,
                special=special,
                grad_safe=grad_safe,
            )
        end
        general_inters_ewald = (ewald,)
    else
        throw(ArgumentError("unknown non-bonded method \"$nonbonded_method\", options are " *
                            ":none, :cutoff, :pme and :ewald"))
    end
    pairwise_inters = (lj, coul)

    # For the purposes of assigning molecules, add connections from atoms to virtual sites
    bonds_all_vs_is, bonds_all_vs_js = copy(bonds_all.is), copy(bonds_all.js)
    for vs in virtual_sites
        for ai in (vs.atom_1, vs.atom_2, vs.atom_3)
            if !iszero(ai)
                push!(bonds_all_vs_is, ai)
                push!(bonds_all_vs_js, vs.atom_ind)
            end
        end
    end
    if length(bonds_all_vs_is) > 0
        topology = MolecularTopology(bonds_all_vs_is, bonds_all_vs_js, length(coords_dev))
    else
        topology = nothing
    end

    constraints, bonds, angles = exchange_constraints(T, bonds_all, angles_all, atoms_data,
                                            constraints_type, rigid_water, units, strictness)

    # Only add present interactions and ensure that array types are concrete
    specific_inter_array = []
    if length(bonds.is) > 0
        push!(specific_inter_array, InteractionList2Atoms(
            to_device(bonds.is, AT),
            to_device(bonds.js, AT),
            to_device([bonds.inters...], AT),
            bonds.types,
        ))
    end
    if length(angles.is) > 0
        push!(specific_inter_array, InteractionList3Atoms(
            to_device(angles.is, AT),
            to_device(angles.js, AT),
            to_device(angles.ks, AT),
            to_device([angles.inters...], AT),
            angles.types,
        ))
    end
    if length(torsions.is) > 0
        push!(specific_inter_array, InteractionList4Atoms(
            to_device(torsions.is, AT),
            to_device(torsions.js, AT),
            to_device(torsions.ks, AT),
            to_device(torsions.ls, AT),
            to_device(torsion_inters_pad, AT),
            torsions.types,
        ))
    end
    if length(impropers.is) > 0
        push!(specific_inter_array, InteractionList4Atoms(
            to_device(impropers.is, AT),
            to_device(impropers.js, AT),
            to_device(impropers.ks, AT),
            to_device(impropers.ls, AT),
            to_device(improper_inters_pad, AT),
            impropers.types,
        ))
    end
    specific_inter_lists = tuple(specific_inter_array...)

    if neighbor_finder_type == NoNeighborFinder
        neighbor_finder = NoNeighborFinder()
    elseif neighbor_finder_type in (nothing, GPUNeighborFinder) && uses_gpu_neighbor_finder(AT)
        neighbor_finder = GPUNeighborFinder(
            eligible=to_device(eligible, AT),
            dist_cutoff=T(dist_cutoff), # Neighbors are computed each step so no buffer is needed
            special=to_device(special, AT),
        )
    elseif neighbor_finder_type in (nothing, DistanceNeighborFinder) &&
                (AT <: AbstractGPUArray || has_infinite_boundary(boundary_used))
        neighbor_finder = DistanceNeighborFinder(
            eligible=to_device(eligible, AT),
            special=to_device(special, AT),
            n_steps=10,
            dist_cutoff=T(dist_neighbors),
        )
    elseif neighbor_finder_type in (nothing, CellListMapNeighborFinder) && !(AT <: AbstractGPUArray)
        neighbor_finder = CellListMapNeighborFinder(
            eligible=eligible,
            special=special,
            n_steps=10,
            x0=coords,
            unit_cell=boundary_used,
            dist_cutoff=T(dist_neighbors),
        )
    else
        neighbor_finder = neighbor_finder_type(
            eligible=to_device(eligible, AT),
            special=to_device(special, AT),
            n_steps=10,
            dist_cutoff=T(dist_neighbors),
        )
    end

    if isnothing(velocities)
        if units
            vels = zero(ustrip_vec.(coords_dev))u"nm * ps^-1"
        else
            vels = zero(coords_dev)
        end
    else
        vels = to_device(velocities, AT)
    end

    if implicit_solvent != :none
        if implicit_solvent in (:obc1, :obc2)
            general_inters_is = (ImplicitSolventOBC(atoms, atoms_data, bonds;
                                 kappa=kappa, use_OBC2=(implicit_solvent == :obc2)),)
        elseif implicit_solvent == :gbn2
            general_inters_is = (ImplicitSolventGBN2(atoms, atoms_data, bonds; kappa=kappa),)
        else
            throw(ArgumentError("unknown implicit solvent model $implicit_solvent, " *
                                "options are :none, :obc1, :obc2 and :gbn2"))
        end
    else
        general_inters_is = ()
    end
    general_inters = (general_inters_ewald..., general_inters_is...)

    k = (units ? Unitful.Na * Unitful.k : ustrip(u"kJ * K^-1 * mol^-1", Unitful.Na * Unitful.k))
    virtual_sites_dev = (length(virtual_sites) > 0 ? to_device(virtual_sites, AT) : virtual_sites)

    return System(
        atoms=atoms,
        coords=coords_dev,
        boundary=boundary_used,
        velocities=vels,
        atoms_data=atoms_data,
        topology=topology,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        constraints=constraints,
        virtual_sites=virtual_sites_dev,
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
        k=k,
        data=data,
        strictness=strictness,
    )
end

"""
    is_any_atom(at, at_data)

Placeholder function that returns `true`, used to select any [`Atom`](@ref).
"""
is_any_atom(at, at_data) = true

"""
    is_heavy_atom(at, at_data)

Determines whether an [`Atom`](@ref) is a heavy atom, i.e. any element other than hydrogen.
"""
function is_heavy_atom(at, at_data)
    if isnothing(at_data) || at_data.element in ("?", "")
        return mass(at) > 1.01u"g/mol"
    else
        return !(at_data.element in ("H", "D"))
    end
end

"""
    add_position_restraints(sys, k; atom_selector=is_any_atom, restrain_coords=sys.coords)

Return a copy of a [`System`](@ref) with [`HarmonicPositionRestraint`](@ref)s added to restrain the
atoms.

The force constant `k` can be a single value or an array of equal length to the number of atoms
in the system.
The `atom_selector` function takes in each atom and atom data and determines whether to restrain
that atom.
For example, [`is_heavy_atom`](@ref) means non-hydrogen atoms are restrained.
"""
function add_position_restraints(sys::System{<:Any, AT},
                                 k;
                                 atom_selector::Function=is_any_atom,
                                 restrain_coords=sys.coords) where AT
    k_array = isa(k, AbstractArray) ? k : fill(k, length(sys))
    if length(k_array) != length(sys)
        throw(ArgumentError("the system has $(length(sys)) atoms but there are $(length(k_array)) k values"))
    end
    is = Int32[]
    types = String[]
    inters = HarmonicPositionRestraint[]
    atoms_data = (length(sys.atoms_data) > 0 ? sys.atoms_data : fill(nothing, length(sys)))
    for (i, (at, at_data, k_res, x0)) in enumerate(zip(from_device(sys.atoms), atoms_data, k_array,
                                                       from_device(restrain_coords)))
        if atom_selector(at, at_data)
            push!(is, i)
            push!(types, "")
            push!(inters, HarmonicPositionRestraint(k_res, x0))
        end
    end
    restraints = InteractionList1Atoms(to_device(is, AT), to_device([inters...], AT), types)
    sis = (sys.specific_inter_lists..., restraints)
    return System(
        atoms=deepcopy(sys.atoms),
        coords=copy(sys.coords),
        boundary=deepcopy(sys.boundary),
        velocities=copy(sys.velocities),
        atoms_data=deepcopy(sys.atoms_data),
        topology=deepcopy(sys.topology),
        pairwise_inters=deepcopy(sys.pairwise_inters),
        specific_inter_lists=sis,
        general_inters=deepcopy(sys.general_inters),
        constraints=deepcopy(sys.constraints),
        virtual_sites=deepcopy(sys.virtual_sites),
        neighbor_finder=deepcopy(sys.neighbor_finder),
        loggers=deepcopy(sys.loggers),
        force_units=sys.force_units,
        energy_units=sys.energy_units,
        k=sys.k,
        data=sys.data,
    )
end
