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
    charges::Vector{Union{T, Missing}}
    extras::BitVector # Marks extra particles
end

struct ResiduePatchTemplate
    pname::String
    add_atoms::Vector{Tuple{String, String, Any}}
    change_atoms::Vector{Tuple{String, String, Any}}
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

# Match a residue read from a structure file to a residue template in the force field
# A template with the same name as the residue is preferred, otherwise all templates are
#   checked in name order so that the assignment does not depend on dictionary ordering
# Returns the template and the mapping from residue atoms to template atoms, or
#   (nothing, nothing) if no template matches
function match_residue(rgraph::ResidueGraph, force_field, sorted_template_names,
                       chain, res_id, strictness)
    templates = force_field.residues
    if haskey(templates, rgraph.res_name)
        template = templates[rgraph.res_name]
        matches = match_residue_to_template(rgraph, template)
        isnothing(matches) || return template, matches
    end

    matched_names, matched_lists = String[], Vector{Int}[]
    for templ_name in sorted_template_names
        templ_name == rgraph.res_name && continue # Already checked above
        matches = match_residue_to_template(rgraph, templates[templ_name])
        if !isnothing(matches)
            push!(matched_names, templ_name)
            push!(matched_lists, matches)
        end
    end
    length(matched_names) == 0 && return nothing, nothing

    # Multiple matching templates are only unambiguous if they assign the same atom
    #   types and charges to every atom
    if length(matched_names) > 1
        name_1, matches_1 = matched_names[1], matched_lists[1]
        templ_1 = templates[name_1]
        ambiguous = String[]
        for (name_2, matches_2) in zip(matched_names[2:end], matched_lists[2:end])
            templ_2 = templates[name_2]
            same = all(zip(matches_1, matches_2)) do (m1, m2)
                templ_1.types[m1] == templ_2.types[m2] &&
                    isequal(templ_1.charges[m1], templ_2.charges[m2])
            end
            same || push!(ambiguous, name_2)
        end
        if length(ambiguous) > 0
            err_str = "residue $(rgraph.res_name) (residue number $res_id of chain " *
                      "\"$chain\") matches multiple residue templates that assign " *
                      "different parameters: $(join([name_1; ambiguous], ", ")). " *
                      "Template $name_1 was used, rename the residue in the structure " *
                      "file to select a different one."
            report_issue(err_str, strictness)
        end
    end
    return templates[matched_names[1]], matched_lists[1]
end

function count_occurrences(items)
    counts = Dict{eltype(items), Int}()
    for item in items
        counts[item] = get(counts, item, 0) + 1
    end
    return counts
end

element_counts(elements) = count_occurrences(collect(Symbol, elements))

# counts1 - counts2, keeping negative values (missing from counts1)
function counts_subtract(counts1::Dict{K, Int}, counts2::Dict{K, Int}) where K
    diff = copy(counts1)
    for (k, v) in counts2
        diff[k] = get(diff, k, 0) - v
    end
    return diff
end

element_label(el::Symbol) = (el == :X ? "extra site" : String(el))

function format_count(el, n)
    return "$n $(element_label(el)) atom" * (n == 1 ? "" : "s")
end

function format_bond_count(key, n)
    return "$(element_label(key[1]))-$(element_label(key[2])) bond" * (n == 1 ? "" : "s")
end

function join_messages(messages)
    msgs = collect(messages)
    length(msgs) == 0 && return ""
    length(msgs) == 1 && return msgs[1]
    return join(msgs[1:(end - 1)], ", ") * " and " * msgs[end]
end

# Describe how a residue differs from a template given the difference in counts
function format_diff_message(diffs, formatter)
    missing_keys = sort([(k, -v) for (k, v) in diffs if v < 0], by=first)
    extra_keys   = sort([(k,  v) for (k, v) in diffs if v > 0], by=first)
    messages = String[]
    if length(missing_keys) > 0
        push!(messages, "is missing " *
              join_messages(formatter(k, n) for (k, n) in missing_keys))
    end
    if length(extra_keys) > 0
        push!(messages, "has " *
              join_messages(formatter(k, n) for (k, n) in extra_keys) * " too many")
    end
    return join_messages(messages)
end

# Score templates by how closely their atom counts match, optionally ignoring
#   hydrogens and extra sites
# Templates the residue is missing atoms from are favored over templates where the
#   residue has extra atoms
function best_matching_templates(template_diffs, template_names, heavy_only)
    best_names, best_score = String[], nothing
    for name in template_names
        all_diffs = template_diffs[name]
        if heavy_only
            diffs = [(k, v) for (k, v) in all_diffs if !(k in (:H, :D, :X))]
        else
            diffs = collect(all_diffs)
        end
        score = (any(v -> v > 0, values(all_diffs)), sum(abs(v) for (_, v) in diffs; init=0))
        if isnothing(best_score) || score <= best_score
            if score != best_score
                empty!(best_names)
                best_score = score
            end
            push!(best_names, name)
        end
    end
    return best_names, best_score
end

# Pick the template with the name closest to the residue name for reporting
function pick_best_match(best_names, res_name)
    length(best_names) == 1 && return best_names[1]
    sorted_names = sort(best_names)
    res_name in sorted_names && return res_name
    # Prefer the longest shared prefix, e.g. NALA/CALA for ALA
    score(name) = length(res_name) == 0 ? 0 :
                  count(i -> i <= length(name) && name[i] == res_name[i],
                        eachindex(res_name))
    return argmax(score, sorted_names)
end

# Element pair key for a bond, sorted so the order of the atoms does not matter
bond_key(el1::Symbol, el2::Symbol) = (el1 <= el2 ? (el1, el2) : (el2, el1))

# Diagnose why a residue read from a structure file did not match any of the residue
#   templates in the force field, following the approach used by OpenMM
# Returns a sentence explaining the most likely cause
function residue_match_error(rgraph::ResidueGraph, templates)
    length(templates) == 0 && return "The force field contains no residue templates."
    template_names = sort(collect(keys(templates)))

    res_counts = element_counts(rgraph.elements)
    supported = Set{Symbol}()
    for name in template_names
        union!(supported, templates[name].elements)
    end
    unsupported = sort([el for el in keys(res_counts) if !(el in supported)])
    if length(unsupported) > 0
        msg = join_messages(element_label(el) * " atoms" for el in unsupported)
        return "The residue contains $msg, which are not supported by any template in " *
               "the force field."
    end

    template_diffs = Dict(name => counts_subtract(res_counts,
                                                  element_counts(templates[name].elements))
                          for name in template_names)

    # Compare heavy atom counts, then all atom counts
    best_names, best_score = best_matching_templates(template_diffs, template_names, true)
    if length(best_names) > 0 && !iszero(best_score[2])
        best = pick_best_match(best_names, rgraph.res_name)
        return "The set of atoms is similar to $best, but the residue " *
               format_diff_message(template_diffs[best], format_count) * "."
    end
    best_names, best_score = best_matching_templates(template_diffs, best_names, false)
    if length(best_names) > 0 && !iszero(best_score[2])
        best = pick_best_match(best_names, rgraph.res_name)
        diffs = template_diffs[best]
        extra_msg = ""
        if get(diffs, :H, 0) < 0 && all(v -> v >= 0, [v for (k, v) in diffs if k != :H])
            extra_msg = " Hydrogens can be added with a tool such as OpenMM Modeller, " *
                        "PDBFixer or gmx pdb2gmx."
        end
        return "The set of heavy atoms matches $best, but the residue " *
               format_diff_message(diffs, format_count) * "." * extra_msg
    end

    # Atom counts match, so compare the bonds within the residue
    res_bond_counts = count_occurrences([bond_key(rgraph.elements[i], rgraph.elements[j])
                                        for (i, j) in rgraph.bonds])
    bond_diffs = Dict{String, Dict{Tuple{Symbol, Symbol}, Int}}()
    for name in best_names
        tpl = templates[name]
        tpl_counts = count_occurrences([bond_key(tpl.elements[i], tpl.elements[j])
                                       for (i, j) in tpl.bonds])
        bond_diffs[name] = counts_subtract(res_bond_counts, tpl_counts)
    end
    bond_best_names, bond_best_score = best_matching_templates(bond_diffs, best_names, false)
    if length(bond_best_names) > 0 && !iszero(bond_best_score[2])
        best = pick_best_match(bond_best_names, rgraph.res_name)
        if length(rgraph.bonds) == 0
            return "The set of atoms matches $best, but the residue has no bonds between " *
                   "its atoms. If the structure was read from a PDB file it may contain " *
                   "non-standard residue or atom names, or be missing CONECT records."
        end
        return "The set of atoms matches $best, but the residue " *
               format_diff_message(bond_diffs[best], format_bond_count) * "."
    end

    # Atoms and internal bonds match, so compare the bonds to other residues
    res_ext_counts = Dict{Symbol, Int}()
    for (i, n_ext) in enumerate(rgraph.external_bonds)
        n_ext > 0 && (res_ext_counts[rgraph.elements[i]] =
                      get(res_ext_counts, rgraph.elements[i], 0) + n_ext)
    end
    ext_diffs = Dict{String, Dict{Symbol, Int}}()
    for name in bond_best_names
        tpl = templates[name]
        tpl_counts = Dict{Symbol, Int}()
        for (i, n_ext) in enumerate(tpl.external_bonds)
            n_ext > 0 && (tpl_counts[tpl.elements[i]] =
                          get(tpl_counts, tpl.elements[i], 0) + n_ext)
        end
        ext_diffs[name] = counts_subtract(res_ext_counts, tpl_counts)
    end
    ext_best_names, ext_best_score = best_matching_templates(ext_diffs, bond_best_names, false)
    if length(ext_best_names) > 0 && !iszero(ext_best_score[2])
        best = pick_best_match(ext_best_names, rgraph.res_name)
        diffs = ext_diffs[best]
        extra_msg = (all(v -> v <= 0, values(diffs)) ?
                     " Is the chain missing a terminal capping group, or are bonds to " *
                     "neighboring residues missing?" : "")
        return "The atoms and bonds in the residue match $best, but the set of atoms " *
               "bonded to other residues " * format_diff_message(diffs, format_count) *
               "." * extra_msg
    end

    if length(ext_best_names) > 0
        return "The atoms and bonds in the residue match " * join_messages(ext_best_names) *
               ", but the connectivity is different."
    end
    return "This may mean that the structure file is missing atoms or bonds, or that the " *
           "wrong force field is being used."
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

# Builds CMAP torsions (i, j, k, l, m) from adjacency and torsions
function build_cmaps(adj::Vector{Vector{Int}}, tors::Vector{NTuple{4, Int}})
    cmap = Vector{NTuple{5, Int}}()
    for tor in tors
        for atom in adj[tor[1]]
            if !(atom in tor)
                push!(cmap, (atom, tor[1], tor[2], tor[3], tor[4]))
            end
        end
        for atom in adj[tor[3]]
            if !(atom in tor)
                push!(cmap, (tor[1], tor[2], tor[3], tor[4], atom))
            end
        end
    end
    return sort!(unique!(cmap))
end

# Helper to make combinations, needed for impropers
function combinations_of(vec::Vector, n::Integer)
    if n < 0 || n > length(vec)
        error("n must be between 0 and length(vec)")
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

function apply_residue_patch(residue, patch, patch_res_name, res_name, patch_name,
                             atom_types, strictness)
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
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name already present"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
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
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        types[i] = atom_type
        partial_charges[i] = partial_charge
    end

    for (atom_name_1, atom_name_2) in patch.remove_bonds
        # This comes before remove_atoms as one of the atoms may be removed later
        i = findfirst(isequal(atom_name_1), atoms)
        if isnothing(i)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name_1 missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        j = findfirst(isequal(atom_name_2), atoms)
        if isnothing(j)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name_2 missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        bond_i = find_bond_ind(i, j, bonds)
        if isnothing(bond_i)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "bond between $atom_name_1 and $atom_name_2 missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        deleteat!(bonds, bond_i)
    end

    for atom_name in patch.remove_atoms
        i = findfirst(isequal(atom_name), atoms)
        if isnothing(i)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        deleteat!(atoms, i)
        deleteat!(elements, i)
        deleteat!(types, i)
        deleteat!(external_bonds, i)
        deleteat!(partial_charges, i)
        deleteat!(extras, i)
        if any(bij -> (bij[1] == i || bij[2] == i), bonds)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name can't be removed as it is part of a bond"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
        end
        bonds .= shift_bond_inds.(bonds, i)
    end

    for (atom_name_1, atom_name_2) in patch.add_bonds
        i = findfirst(isequal(atom_name_1), atoms)
        if isnothing(i)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name_1 missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        j = findfirst(isequal(atom_name_2), atoms)
        if isnothing(j)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name_2 missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        bond_i = find_bond_ind(i, j, bonds)
        if !isnothing(bond_i)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "bond between $atom_name_1 and $atom_name_2 already present"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        push!(bonds, (i, j))
    end

    for atom_name in patch.add_external_bonds
        i = findfirst(isequal(atom_name), atoms)
        if isnothing(i)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        external_bonds[i] += 1
    end

    for atom_name in patch.remove_external_bonds
        i = findfirst(isequal(atom_name), atoms)
        if isnothing(i)
            err_str = "Can't apply patch $patch_name to residue template $res_name: " *
                      "atom name $atom_name missing"
            report_issue(err_str, strictness; error_type=ForceFieldXMLError)
            return nothing
        end
        external_bonds[i] = max(external_bonds[i] - 1, 0)
    end

    return ResidueTemplate(
        patch_res_name, atoms, elements, types, virtual_sites,
        bonds, external_bonds, String[], partial_charges, extras,
    )
end
