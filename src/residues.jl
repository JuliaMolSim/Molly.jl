# Deal with residues

# Struct to carry the information necessary to represent the residue templates
#   defined in the force field XML files
struct ResidueTemplate{T, IC}
    name::String
    atoms::Vector{String}
    elements::Vector{Symbol}
    types::Vector{String}
    virtual_sites::Vector{VirtualSiteTemplate{T, IC}}
    bonds::Vector{Tuple{Int, Int}}
    external_bonds::Vector{Int} # Count of external connections per atom
    allowed_patches::Vector{String}
    charges::Vector{T}
    extras::BitVector # Marks extra particles
end

struct ResidueTemplatePatch{T}
    pname::String
    add_atoms::Vector{Tuple{String, String, T}}
    change_atoms::Vector{Tuple{String, String, T}}
    remove_atoms::Vector{String}
    add_bonds::Vector{Tuple{String, String}}
    remove_bonds::Vector{Tuple{String, String}}
    add_external_bonds::Vector{String}
    remove_external_bonds::Vector{String}
    apply_to_residues::Vector{String}
end

# Equivalent to the struct above, but this represents a residue read from the
#   provided structure files
struct ResidueGraph
    res_name::String # Includes N/C prefix if terminal
    atom_inds::Vector{Int} # Global 1-based
    atom_names::Vector{String}
    elements::Vector{Symbol}
    bonds::Vector{Tuple{Int, Int}} # Local intra-residue bonds
    external_bonds::Vector{Int} # Count of external connections per atom
end

function atom_name_from_index(atom_idx, canon_system)
    for (chain, resids) in canon_system
        for (res_id, rgraph) in resids
            if !(atom_idx in rgraph.atom_inds)
                continue
            else
                local_idx = findfirst(isequal(atom_idx), rgraph.atom_inds)
                return rgraph.atom_names[local_idx]
            end
        end
    end
end

function residue_from_atom_idx(atom_idx, canon_system)
    for (chain, resids) in canon_system
        for (res_id, rgraph) in resids
            if atom_idx in rgraph.atom_inds
                return rgraph
            end
        end
    end
end

function resnum_from_atom_idx(atom_idx, canon_system)
    for (chain, resids) in canon_system
        for (res_id, rgraph) in resids
            if atom_idx in rgraph.atom_inds
                return res_id
            end
        end
    end
end

function chain_from_atom_idx(atom_idx, canon_system)
    for (chain, resids) in canon_system
        for (res_id, rgraph) in resids
            if atom_idx in rgraph.atom_inds
                return chain
            end
        end
    end
end

# Fill d with every attribute value of each <Atom> mapping to the canonical name
function parse_atoms(residue::EzXML.Node, d::Dict{String, String})
    for atom in findall("Atom", residue)
        canon = atom["name"]
        for attr in eachattribute(atom)
            d[attr.content] = canon
        end
    end
    return d
end

# Loads atom and residue name replacements, given a table of commonly
#   used alternative namings in PDB files
function load_replacements(; xmlpath=nothing, resname_replacements=nothing,
                           atomname_replacements=nothing)
    if isnothing(resname_replacements)
        resname_replacements = Dict{String,String}()
        atomname_replacements = Dict{String, Dict{String, String}}()
    end

    if isnothing(xmlpath)
        xmlpath = normpath(@__DIR__, "..", "data", "force_fields", "pdbNames.xml")
    end

    doc = readxml(xmlpath)
    root = doc.root

    allResidues         = Dict{String,String}()
    proteinResidues     = Dict{String,String}()
    nucleicAcidResidues = Dict{String,String}()

    # First pass
    for residue in findall("Residue", root)
        rname = residue["name"]
        if rname == "All"
            parse_atoms(residue, allResidues)
        elseif rname == "Protein"
            parse_atoms(residue, proteinResidues)
        elseif rname == "Nucleic"
            parse_atoms(residue, nucleicAcidResidues)
        end
    end

    # Merge "All" into specific groups
    for (k, v) in allResidues
        proteinResidues[k] = v
        nucleicAcidResidues[k] = v
    end

    # Second pass
    for residue in findall("Residue", root)
        rname = residue["name"]

        # Map residue aliases (name and any alt*)
        for attr in eachattribute(residue)
            aname = attr.name
            if aname == "name" || startswith(aname, "alt")
                resname_replacements[attr.content] = rname
            end
        end

        # Select base atom map by type
        if haskey(residue, "type")
            rtype = residue["type"]
            if rtype == "Protein"
                atoms = copy(proteinResidues)
            elseif rtype == "Nucleic"
                atoms = copy(nucleicAcidResidues)
            else
                atoms = copy(allResidues)
            end
        else
            atoms = copy(allResidues)
        end

        parse_atoms(residue, atoms)
        atomname_replacements[rname] = atoms
    end

    return resname_replacements, atomname_replacements
end

# Loads the standard topology for the residues
function load_bond_definitions(; xmlpath=nothing, standardBonds=nothing)
    if isnothing(xmlpath)
        xmlpath = normpath(@__DIR__, "..", "data", "force_fields", "residues.xml")
    end
    if isnothing(standardBonds)
        standardBonds = Dict{String, Vector{Tuple{String, String}}}()
    end

    doc = readxml(xmlpath)
    root = doc.root

    for residue in findall("Residue", root)
        bonds = Tuple{String, String}[]
        standardBonds[residue["name"]] = bonds
        for bond in findall("Bond", residue)
            push!(bonds, (bond["from"], bond["to"]))
        end
    end
    return standardBonds
end

# Builds the topology of the system read from a structure file given the
#   template bonds
function create_bonds!(canon_sys, standard_bonds)
    bonds = Tuple{Int, Int}[]

    for (chain, resids) in canon_sys
        n_resids = length(resids)
        atom_maps = Dict{Int, Dict{String, Int}}()

        for (resnum, rgraph) in resids
            atomMap = Dict{String, Int}()
            atom_names = rgraph.atom_names
            atom_inds  = rgraph.atom_inds
            for (name, idx) in zip(atom_names, atom_inds)
                atomMap[name] = idx
            end
            atom_maps[resnum] = atomMap
        end

        for (resnum, rgraph) in resids
            i = resnum
            res_name = rgraph.res_name

            if haskey(standard_bonds, res_name)
                for bond in standard_bonds[res_name]
                    external = false
                    if startswith(bond[1], "-") && i > 1
                        external = true
                        fromResidue = i - 1
                        fromAtom = bond[1][2:end]

                        ext_ind = findfirst(isequal(fromAtom), resids[fromResidue].atom_names)
                        resids[fromResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(isequal(bond[2]), rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1
                    elseif startswith(bond[1], "+") && i < n_resids
                        external = true
                        fromResidue = i + 1
                        fromAtom = bond[1][2:end]

                        ext_ind = findfirst(isequal(fromAtom), resids[fromResidue].atom_names)
                        resids[fromResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(isequal(bond[2]), rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1
                    else
                        fromResidue = i
                        fromAtom = bond[1]
                    end

                    if startswith(bond[2], "-") && i > 1
                        external = true
                        toResidue = i - 1
                        toAtom = bond[2][2:end]

                        ext_ind = findfirst(isequal(toAtom), resids[toResidue].atom_names)
                        resids[toResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(isequal(bond[1]), rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1
                    elseif startswith(bond[2], "+") && i < n_resids
                        external = true
                        toResidue = i + 1
                        toAtom = bond[2][2:end]

                        ext_ind = findfirst(isequal(toAtom), resids[toResidue].atom_names)
                        resids[toResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(isequal(bond[1]), rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1
                    else
                        toResidue = i
                        toAtom = bond[2]
                    end

                    if fromAtom in keys(atom_maps[fromResidue]) && toAtom in keys(atom_maps[toResidue])
                        atom1 = atom_maps[fromResidue][fromAtom]
                        atom2 = atom_maps[toResidue][toAtom]
                        pair = (atom1 < atom2 ? (atom1, atom2) : (atom2, atom1))
                        if !(pair in bonds)
                            push!(bonds, pair)
                            if !external
                                i_local = findfirst(isequal(fromAtom), rgraph.atom_names)
                                j_local = findfirst(isequal(toAtom),   rgraph.atom_names)
                                pair_local = (i_local < j_local ? (i_local, j_local) : (j_local, i_local))
                                push!(rgraph.bonds, pair_local)
                            end
                        end
                    end
                end
            end
        end
    end

    return sort(bonds)
end

# Builds disulfide bonds given some geometric criteria
function create_disulfide_bonds(coords, boundary, canon_system, bonds)
    function is_cysx(rgraph::ResidueGraph)
        names = rgraph.atom_names
        return ("SG" in names && !("HG" in names))
    end

    function is_disulfide_bonded(atom_idx)
        for b in bonds
            atom_name_i = atom_name_from_index(b[1], canon_system)
            atom_name_j = atom_name_from_index(b[2], canon_system)
            if atom_idx in b && atom_name_i == "SG" && atom_name_j == "SG"
                return true
            end
        end
        return false
    end

    cysx = ResidueGraph[]
    for (chain, resids) in canon_system
        for (res_idx, rgraph) in resids
            if rgraph.res_name == "CYS" && is_cysx(rgraph)
                push!(cysx, rgraph)
            end
        end
    end

    n_cysx = length(cysx)
    for (cys_idx, cysi) in enumerate(cysx)
        sg1_idx = findfirst(isequal("SG"), cysi.atom_names)
        atom_idx = cysi.atom_inds[sg1_idx]
        pos1 = coords[atom_idx]

        candidate_distance = (unit(eltype(coords[1])) == NoUnits ? 0.3 : 0.3u"nm")
        candidate_atom = nothing
        cysj_valid = nothing
        sg2_idx_valid = nothing
        atom_jdx_valid = nothing

        for cys_jdx in (cys_idx+1):n_cysx
            cysj = cysx[cys_jdx]
            sg2_idx = findfirst(isequal("SG"), cysj.atom_names)
            atom_jdx = cysj.atom_inds[sg2_idx]
            pos2 = coords[atom_jdx]
            vec = vector(pos1, pos2, boundary)
            dst = norm(vec)

            if dst < candidate_distance && !is_disulfide_bonded(atom_idx)
                cysj_valid = cysj
                sg2_idx_valid = sg2_idx
                atom_jdx_valid = atom_jdx
                candidate_distance = dst
                candidate_atom = atom_jdx
            end
        end
        if !isnothing(candidate_atom)
            cysi.external_bonds[sg1_idx] += 1
            cysj_valid.external_bonds[sg2_idx_valid] += 1
            pair = (atom_idx < atom_jdx_valid ? (atom_idx, atom_jdx_valid) : (atom_jdx_valid, atom_idx))
            push!(bonds, pair)
        end
    end

    sort!(bonds)
    return bonds
end

# Add bonds only if they have not been added by the previous steps
function read_extra_bonds!(canonical_system, top, top_bonds)
    chfl_bonds = Vector{Int}[is .+ 1 for is in eachcol(Int.(Chemfiles.bonds(top)))]
    for (i, j) in chfl_bonds
        res_i = residue_from_atom_idx(i, canonical_system)
        res_j = residue_from_atom_idx(j, canonical_system)
        pair = (i < j ? (i, j) : (j, i))
        local_idx = findfirst(isequal(i), res_i.atom_inds)
        local_jdx = findfirst(isequal(j), res_j.atom_inds)
        if res_i == res_j
            local_pair = (local_idx < local_jdx ? (local_idx, local_jdx) : (local_jdx, local_idx))
            if !(pair in top_bonds)
                push!(top_bonds, pair)
                if !(local_pair in res_i.bonds)
                    push!(res_i.bonds, local_pair)
                end
            end
        else
            if !(pair in top_bonds)
                res_i.external_bonds[local_idx] += 1
                res_j.external_bonds[local_jdx] += 1
                push!(top_bonds, pair)
            end
        end
    end
    return sort!(unique!(top_bonds))
end

# Template matching step, follows the OpenMM procedure.
# In general, it first checks if the residue to be matched
# has the same signature (N elements, bonds per atom) than its template.
# If not, residues do not match. If residue and template share signature,
# the residue graphs are compared through a depth-first search
# (dfs helper in this method), ensuring that their topologies are the same.
function match_residue_to_template(res::ResidueGraph,
                                   tpl::ResidueTemplate;
                                   ignoreExternalBonds::Bool=false,
                                   ignoreExtraParticles::Bool=false)::Union{Vector{Int}, Nothing}
    # 0) Define extra-particle predicates
    is_extra_res(i) = (res.elements[i] == :X)
    is_extra_tpl(j) = tpl.extras[j]

    # 1) Select atoms to consider
    if ignoreExtraParticles
        res_keep = findall(i -> !is_extra_res(i), eachindex(res.atom_names))
        tpl_keep = findall(j -> !is_extra_tpl(j), eachindex(tpl.atoms))
    else
        res_keep = collect(eachindex(res.atom_names))
        tpl_keep = collect(eachindex(tpl.atoms))
    end

    numAtoms = length(res_keep)
    if numAtoms != length(tpl_keep)
        return nothing
    end
    if numAtoms == 0
        return Int[] # Both empty after filtering → vacuous match
    end

    # 2) Build local index maps (kept-only)
    res_old2new = Dict{Int, Int}(res_keep[k] => k for k in 1:numAtoms)
    tpl_old2new = Dict{Int, Int}(tpl_keep[k] => k for k in 1:numAtoms)
    tpl_new2old = copy(tpl_keep) # Inverse map to original template indices

    # 3) Build adjacency among kept atoms and external-bond counts
    # Residue: local bonds are given in res.bonds over original local indices
    res_adj = [Int[] for _ in 1:numAtoms]
    for (i, j) in res.bonds
        (haskey(res_old2new, i) && haskey(res_old2new, j)) || continue
        ii, jj = res_old2new[i], res_old2new[j]
        push!(res_adj[ii], jj)
        push!(res_adj[jj], ii)
    end
    if ignoreExternalBonds
        res_ext = fill(0, numAtoms)
    else
        res_ext = [res.external_bonds[res_keep[k]] for k in 1:numAtoms]
    end

    # Template: build adjacency from tpl.bonds, but only within kept atoms
    tpl_adj = [Int[] for _ in 1:numAtoms]
    for (i, j) in tpl.bonds
        (haskey(tpl_old2new, i) && haskey(tpl_old2new, j)) || continue
        ii, jj = tpl_old2new[i], tpl_old2new[j]
        push!(tpl_adj[ii], jj)
        push!(tpl_adj[jj], ii)
    end
    if ignoreExternalBonds
        tpl_ext = fill(0, numAtoms)
    else
        tpl_ext = [tpl.external_bonds[tpl_keep[k]] for k in 1:numAtoms]
    end

    # 4) Quick type-count screen: (element or :X, degree, ext) multiplicities must match
    # Residue keys
    res_keys = Tuple{Symbol, Int, Int}[]
    for i in 1:numAtoms
        key = (res.elements[res_keep[i]], length(res_adj[i]), res_ext[i])
        push!(res_keys, key)
    end
    # Template keys
    tpl_keys = Tuple{Symbol, Int, Int}[]
    for k in 1:numAtoms
        # Use template element symbol, but treat extras specially in candidate stage
        key = (tpl.elements[tpl_keep[k]], length(tpl_adj[k]), tpl_ext[k])
        push!(tpl_keys, key)
    end
    # Compare multisets
    sort!(res_keys)
    sort!(tpl_keys)
    if res_keys != tpl_keys
        return nothing
    end

    # 5) Candidate template atoms for each residue atom
    # OpenMM's exactNameMatch: if residue atom is extra and there exists a template extra
    # with same name, enforce name equality. Otherwise extra can map to any template extra.
    # Non-extra must match element exactly and template must be non-extra.
    candidates = Vector{Vector{Int}}(undef, numAtoms)

    for i in 1:numAtoms
        ri_old = res_keep[i]
        r_el   = res.elements[ri_old]
        r_name = res.atom_names[ri_old]
        r_deg  = length(res_adj[i])
        r_ext  = res_ext[i]
        r_is_extra = (r_el == :X)

        exactNameMatch = (r_is_extra && any(is_extra_tpl(j) && tpl.atoms[j] == r_name for j in tpl_keep))

        cands = Int[]
        for (k, tj_old) in enumerate(tpl_keep)
            t_el   = tpl.elements[tj_old]
            t_name = tpl.atoms[tj_old]
            t_deg  = length(tpl_adj[k])
            t_ext  = tpl_ext[k]
            t_is_extra = is_extra_tpl(tj_old)

            # Element/name gate
            if r_is_extra
                # Residue extra → template must be extra
                t_is_extra || continue
                if exactNameMatch && t_name != r_name
                    continue
                end
            else
                # Residue real element → template must be non-extra and element equal
                t_is_extra && continue
                t_el == r_el || continue
            end

            # Degree and external-bond checks
            r_deg == t_deg || continue
            (ignoreExternalBonds || (r_ext == t_ext)) || continue

            push!(cands, k) # Store template new-index k
        end
        # Early prune: if no candidates for a residue atom, fail
        isempty(cands) && return nothing
        candidates[i] = cands
    end
    # 6) Heuristic search order: fewest candidates first, then neighbors of chosen
    atomsToOrder = Set(1:numAtoms)
    searchOrder = Int[]
    neighbor_heap = Int[] # An unordered list of candidate neighbors

    while !isempty(atomsToOrder)
        if isempty(neighbor_heap)
            # Pick global minimum by candidate count among remaining
            nextAtom = argmin(i -> length(candidates[i]), collect(atomsToOrder))
        else
            # Pick the neighbor with fewest candidates
            sort!(neighbor_heap, by=(i -> length(candidates[i])))
            nextAtom = neighbor_heap[1]
            filter!(i -> i != nextAtom, neighbor_heap)
        end
        push!(searchOrder, nextAtom)
        delete!(atomsToOrder, nextAtom)
        # push its neighbors
        for nb in res_adj[nextAtom]
            if nb in atomsToOrder && !(nb in neighbor_heap)
                push!(neighbor_heap, nb)
            end
        end
    end

    inverseSearchOrder = zeros(Int, numAtoms)
    for (pos, i) in enumerate(searchOrder)
        inverseSearchOrder[i] = pos
    end

    # Reorder adjacency and candidates by searchOrder, and relabel neighbor indices to search positions
    res_adj_ord = Vector{Vector{Int}}(undef, numAtoms)
    cand_ord    = Vector{Vector{Int}}(undef, numAtoms)
    for pos in 1:numAtoms
        i = searchOrder[pos]
        res_adj_ord[pos] = [inverseSearchOrder[j] for j in res_adj[i]]
        cand_ord[pos] = candidates[i]
    end

    # 7) Recursive backtracking with bond-consistency
    matches_tpl = fill(0, numAtoms) # At position pos, matched template new-index
    used_tpl = falses(numAtoms)

    function dfs(pos::Integer)
        pos > numAtoms && return true
        # Try candidates for this residue position
        for t_new in cand_ord[pos]
            if used_tpl[t_new]
                continue
            end
            # Check bond consistency with already assigned neighbors
            ok = true
            for nb_pos in res_adj_ord[pos]
                if nb_pos < pos # Already assigned
                    t_nb = matches_tpl[nb_pos]
                    # Must be bonded in template
                    # t_new is connected to t_nb if each appears in adjacency of the other
                    # We have tpl_adj in new-index space
                    if !(t_nb in tpl_adj[t_new])
                        ok = false
                        break
                    end
                end
            end
            ok || continue
            # Assign and recurse
            matches_tpl[pos] = t_new
            used_tpl[t_new] = true
            if dfs(pos + 1)
                return true
            end
            used_tpl[t_new] = false
        end
        return false
    end

    if !dfs(1)
        return nothing
    end

    # 8) Return mapping back to original template indices, in original residue-kept order
    # We need mapping for residue atoms in kept-order, not search-order.
    # matches_tpl is in search-order -> invert back:
    matches_tpl_in_res_order = similar(matches_tpl)
    for pos in 1:numAtoms
        i = searchOrder[pos]
        matches_tpl_in_res_order[i] = matches_tpl[pos]
    end

    # Convert template new-indices to original template indices
    return [tpl_new2old[t_new] for t_new in matches_tpl_in_res_order]
end

# Global adjacency from bonds
function build_adjacency(natoms::Integer, bonds::Vector{NTuple{2, Int}})
    adj = [Int[] for _ in 1:natoms]
    @inbounds for (i, j) in bonds
        push!(adj[i], j)
        push!(adj[j], i)
    end
    for a in adj
        unique!(a)
        sort!(a)
    end
    return adj
end

# Builds the angles (i, j, k) from the adjacency matrix and bonds
function build_angles(adj::Vector{Vector{Int}}, bonds)
    angles = Vector{NTuple{3, Int}}()
    for bond in bonds
        for atom in adj[bond[1]]
            if atom != bond[2]
                if atom < bond[2]
                    push!(angles, (atom, bond[1], bond[2]))
                else
                    push!(angles, (bond[2], bond[1], atom))
                end
            end
        end
        for atom in adj[bond[2]]
            if atom != bond[1]
                if atom > bond[1]
                    push!(angles, (bond[1], bond[2], atom))
                else
                    push!(angles, (atom, bond[2], bond[1]))
                end
            end
        end
    end
    return sort!(unique!(angles))
end

# Builds proper torsion (i, j, k, l) from adjacency and angles
function build_torsions(adj::Vector{Vector{Int}}, angles::Vector{NTuple{3, Int}})
    tors = Vector{NTuple{4, Int}}()
    for angle in angles
        for atom in adj[angle[1]]
            if !(atom in angle)
                if atom < angle[3]
                    push!(tors, (atom, angle[1], angle[2], angle[3]))
                else
                    push!(tors, (angle[3], angle[2], angle[1], atom))
                end
            end
        end
        for atom in adj[angle[3]]
            if !(atom in angle)
                if atom > angle[1]
                    push!(tors, (angle[1], angle[2], angle[3], atom))
                else
                    push!(tors, (atom, angle[3], angle[2], angle[1]))
                end
            end
        end
    end
    return sort!(unique!(tors))
end

# Helper to make combinations, needed for impropers
function combinations_of(vec::Vector, n::Integer)
    if n < 0 || n > length(vec)
        throw(ArgumentError("n must be between 0 and length(vec)"))
    end
    result = Vector{Vector{eltype(vec)}}()
    inds = collect(1:n)
    L = length(vec)
    while true
        push!(result, vec[inds])
        k = n
        while k ≥ 1 && inds[k] == L - n + k
            k -= 1
        end
        if k == 0
            break
        end
        inds[k] += 1
        for i in k+1:n
            inds[i] = inds[i-1] + 1
        end
    end
    return result
end

# Builds the improper torsion (i, j, k, l) given the adjacency matrix
function build_impropers(adj::Vector{Vector{Int}})
    top_impropers = Tuple{Int, Int, Int, Int}[]
    for (i, bonded_to) in enumerate(adj)
        if length(bonded_to) > 2
            for subset in combinations_of(bonded_to, 3)
                push!(top_impropers, (i, subset[1], subset[2], subset[3]))
            end
        end
    end
    return top_impropers
end

function find_bond_ind(i, j, bonds)
    return findfirst(bij -> ((bij[1] == i && bij[2] == j) || (bij[1] == j && bij[2] == i)), bonds)
end

function shift_bond_ind(bi, i)
    if bi < i
        return bi
    else # bi > i due to previous checks
        return bi - 1
    end
end

shift_bond_inds(bij, i) = (shift_bond_ind(bij[1], i), shift_bond_ind(bij[2], i))

function apply_residue_patch(residue, patch, patch_res_name, res_name, patch_name, atom_types)
    atoms           = copy(residue.atoms)
    elements        = copy(residue.elements)
    types           = copy(residue.types)
    virtual_sites   = copy(residue.virtual_sites)
    bonds           = copy(residue.bonds)
    external_bonds  = copy(residue.external_bonds)
    partial_charges = copy(residue.charges)
    extras          = copy(residue.extras)

    for (atom_name, atom_type, partial_charge) in patch.add_atoms
        i = findfirst(isequal(atom_name), atoms)
        if !isnothing(i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name already present"
            return nothing
        end
        el = atom_types[atom_type].element
        push!(atoms, atom_name)
        push!(elements, element_string_to_symbol(el))
        push!(types, atom_type)
        push!(external_bonds, 0)
        push!(partial_charges, partial_charge)
        push!(extras, 0)
    end

    for (atom_name, atom_type, partial_charge) in patch.change_atoms
        i = findfirst(isequal(atom_name), atoms)
        if isnothing(i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name missing"
            return nothing
        end
        types[i] = atom_type
        partial_charges[i] = partial_charge
    end

    for (atom_name_1, atom_name_2) in patch.remove_bonds
        # This comes before remove_atoms as one of the atoms may be removed later
        i = findfirst(isequal(atom_name_1), atoms)
        if isnothing(i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name_1 missing"
            return nothing
        end
        j = findfirst(isequal(atom_name_2), atoms)
        if isnothing(j)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name_2 missing"
            return nothing
        end
        bond_i = find_bond_ind(i, j, bonds)
        if isnothing(bond_i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "bond between $atom_name_1 and $atom_name_2 missing"
            return nothing
        end
        deleteat!(bonds, bond_i)
    end

    for atom_name in patch.remove_atoms
        i = findfirst(isequal(atom_name), atoms)
        if isnothing(i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name missing"
            return nothing
        end
        deleteat!(atoms, i)
        deleteat!(elements, i)
        deleteat!(types, i)
        deleteat!(external_bonds, i)
        deleteat!(partial_charges, i)
        deleteat!(extras, i)
        if any(bij -> (bij[1] == i || bij[2] == i), bonds)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name can't be removed as it is part of a bond"
        end
        bonds .= shift_bond_inds.(bonds, i)
    end

    for (atom_name_1, atom_name_2) in patch.add_bonds
        i = findfirst(isequal(atom_name_1), atoms)
        if isnothing(i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name_1 missing"
            return nothing
        end
        j = findfirst(isequal(atom_name_2), atoms)
        if isnothing(j)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name_2 missing"
            return nothing
        end
        bond_i = find_bond_ind(i, j, bonds)
        if !isnothing(bond_i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "bond between $atom_name_1 and $atom_name_2 already present"
            return nothing
        end
        push!(bonds, (i, j))
    end

    for atom_name in patch.add_external_bonds
        i = findfirst(isequal(atom_name), atoms)
        if isnothing(i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name missing"
            return nothing
        end
        external_bonds[i] += 1
    end

    for atom_name in patch.remove_external_bonds
        i = findfirst(isequal(atom_name), atoms)
        if isnothing(i)
            @warn "Can't apply patch $patch_name to residue template $res_name: " *
                  "atom name $atom_name missing"
            return nothing
        end
        external_bonds[i] = max(external_bonds[i] - 1, 0)
    end

    return ResidueTemplate(
        patch_res_name, atoms, elements, types, virtual_sites,
        bonds, external_bonds, String[], partial_charges, extras,
    )
end
