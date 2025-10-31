# --- Elements dictionary ---
const ELEMENT_SYMBOLS = Dict{String,Symbol}(
    "H"=>:H,  "He"=>:He,
    "Li"=>:Li,"Be"=>:Be,"B"=>:B,"C"=>:C,"N"=>:N,"O"=>:O,"F"=>:F,"Ne"=>:Ne,
    "Na"=>:Na,"Mg"=>:Mg,"Al"=>:Al,"Si"=>:Si,"P"=>:P,"S"=>:S,"Cl"=>:Cl,"Ar"=>:Ar,
    "K"=>:K,  "Ca"=>:Ca,"Sc"=>:Sc,"Ti"=>:Ti,"V"=>:V,"Cr"=>:Cr,"Mn"=>:Mn,"Fe"=>:Fe,"Co"=>:Co,"Ni"=>:Ni,"Cu"=>:Cu,"Zn"=>:Zn,
    "Ga"=>:Ga,"Ge"=>:Ge,"As"=>:As,"Se"=>:Se,"Br"=>:Br,"Kr"=>:Kr,
    "Rb"=>:Rb,"Sr"=>:Sr,"Y"=>:Y,"Zr"=>:Zr,"Nb"=>:Nb,"Mo"=>:Mo,"Tc"=>:Tc,"Ru"=>:Ru,"Rh"=>:Rh,"Pd"=>:Pd,"Ag"=>:Ag,"Cd"=>:Cd,
    "In"=>:In,"Sn"=>:Sn,"Sb"=>:Sb,"Te"=>:Te,"I"=>:I,"Xe"=>:Xe,
    "Cs"=>:Cs,"Ba"=>:Ba,"La"=>:La,"Ce"=>:Ce,"Pr"=>:Pr,"Nd"=>:Nd,"Pm"=>:Pm,"Sm"=>:Sm,"Eu"=>:Eu,"Gd"=>:Gd,"Tb"=>:Tb,"Dy"=>:Dy,"Ho"=>:Ho,"Er"=>:Er,"Tm"=>:Tm,"Yb"=>:Yb,"Lu"=>:Lu,
    "Hf"=>:Hf,"Ta"=>:Ta,"W"=>:W,"Re"=>:Re,"Os"=>:Os,"Ir"=>:Ir,"Pt"=>:Pt,"Au"=>:Au,"Hg"=>:Hg,
    "Tl"=>:Tl,"Pb"=>:Pb,"Bi"=>:Bi,"Po"=>:Po,"At"=>:At,"Rn"=>:Rn,
    "Fr"=>:Fr,"Ra"=>:Ra,"Ac"=>:Ac,"Th"=>:Th,"Pa"=>:Pa,"U"=>:U,"Np"=>:Np,"Pu"=>:Pu,"Am"=>:Am,"Cm"=>:Cm,"Bk"=>:Bk,"Cf"=>:Cf,"Es"=>:Es,"Fm"=>:Fm,"Md"=>:Md,"No"=>:No,"Lr"=>:Lr,
    "Rf"=>:Rf,"Db"=>:Db,"Sg"=>:Sg,"Bh"=>:Bh,"Hs"=>:Hs,"Mt"=>:Mt,"Ds"=>:Ds,"Rg"=>:Rg,"Cn"=>:Cn,"Nh"=>:Nh,"Fl"=>:Fl,"Mc"=>:Mc,"Lv"=>:Lv,"Ts"=>:Ts,"Og"=>:Og,
    "?"=>:X
)

const PERIODIC_TABLE = Symbol[
    :H,  :He,
    :Li, :Be, :B,  :C,  :N,  :O,  :F,  :Ne,
    :Na, :Mg, :Al, :Si, :P,  :S,  :Cl, :Ar,
    :K,  :Ca, :Sc, :Ti, :V,  :Cr, :Mn, :Fe, :Co, :Ni, :Cu, :Zn,
    :Ga, :Ge, :As, :Se, :Br, :Kr,
    :Rb, :Sr, :Y,  :Zr, :Nb, :Mo, :Tc, :Ru, :Rh, :Pd, :Ag, :Cd,
    :In, :Sn, :Sb, :Te, :I,  :Xe,
    :Cs, :Ba, :La, :Ce, :Pr, :Nd, :Pm, :Sm, :Eu, :Gd, :Tb, :Dy, :Ho, :Er, :Tm, :Yb, :Lu,
    :Hf, :Ta, :W,  :Re, :Os, :Ir, :Pt, :Au, :Hg,
    :Tl, :Pb, :Bi, :Po, :At, :Rn,
    :Fr, :Ra, :Ac, :Th, :Pa, :U,  :Np, :Pu, :Am, :Cm, :Bk, :Cf, :Es, :Fm, :Md, :No, :Lr,
    :Rf, :Db, :Sg, :Bh, :Hs, :Mt, :Ds, :Rg, :Cn, :Nh, :Fl, :Mc, :Lv, :Ts, :Og
]

"""
"""
struct ResidueTemplate{T}
    name::String
    atoms::Vector{String}                 # atom names
    elements::Vector{Symbol}              # element symbols
    types::Vector{String}                 # atom types
    bonds::Vector{Tuple{Int,Int}}         # internal bonds
    external_bonds::Vector{Int}           # count of external connections per atom
    charges::Vector{T}
    extras::BitVector                     # marks extra particles
end

"""
"""
struct ResidueGraph
    res_name::String              # includes N/C prefix if terminal
    atom_inds::Vector{Int}        # global 1-based
    atom_names::Vector{String}
    elements::Vector{Symbol}      # atom elements
    bonds::Vector{Tuple{Int,Int}} # local intra-res bonds
    external_bonds::Vector{Int}   # per-atom external degree
end

# Fill `map` with every attribute value of each <Atom> mapping to the canonical name
function parse_atoms(residue::EzXML.Node, map::Dict{String,String})
    for atom in findall("Atom", residue)
        canon = atom["name"]
        for attr in eachattribute(atom)
            map[attr.content] = canon
        end
    end
    return map
end

function load_replacements(; xmlpath = nothing,
                             resname_replacements  = nothing,
                             atomname_replacements = nothing)

    if isnothing(resname_replacements)
        resname_replacements = Dict{String,String}()
    end
    
    if isnothing(atomname_replacements)
        atomname_replacements = Dict{String,Dict{String,String}}()
    end

    if isnothing(xmlpath)
        xmlpath = normpath(@__DIR__, "..", "data/force_fields/pdbNames.xml")
    end
    
    doc  = readxml(xmlpath)
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
        proteinResidues[k]     = v
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
        atoms = if haskey(residue, "type")
            rtype = residue["type"]
            rtype == "Protein" ? copy(proteinResidues) :
            rtype == "Nucleic" ? copy(nucleicAcidResidues) :
                                 copy(allResidues)
        else
            copy(allResidues)
        end

        parse_atoms(residue, atoms)
        atomname_replacements[rname] = atoms
    end

    return resname_replacements, atomname_replacements
end

function load_bond_definitions(; xmlpath = nothing, standardBonds = nothing)

    if isnothing(xmlpath)
        xmlpath = normpath(@__DIR__, "..", "data/force_fields/residues.xml")
    end
    if isnothing(standardBonds)
        standardBonds = Dict{String,Vector{Tuple{String, String}}}()
    end

    doc  = readxml(xmlpath)
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

function create_bonds(top, canon_sys, standard_bonds,
                      resname_replacements)

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
                        fromResidue = i-1
                        fromAtom = bond[1][2:end]
                        
                        ext_ind = findfirst(x->x==fromAtom, resids[fromResidue].atom_names)
                        resids[fromResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(x->x==bond[2], rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1

                    elseif startswith(bond[1], "+") && i < n_resids
                        external = true
                        fromResidue = i+1
                        fromAtom = bond[1][2:end]

                        ext_ind = findfirst(x->x==fromAtom, resids[fromResidue].atom_names)
                        resids[fromResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(x->x==bond[2], rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1

                    else
                        fromResidue = i
                        fromAtom = bond[1]
                    end

                    if startswith(bond[2], "-") && i > 1
                        external = true
                        toResidue = i-1
                        toAtom = bond[2][2:end]

                        ext_ind = findfirst(x->x==toAtom, resids[toResidue].atom_names)
                        resids[toResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(x->x==bond[1], rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1

                    elseif startswith(bond[2], "+") && i < n_resids
                        external = true
                        toResidue = i+1
                        toAtom = bond[2][2:end]
                        
                        ext_ind = findfirst(x->x==toAtom, resids[toResidue].atom_names)
                        resids[toResidue].external_bonds[ext_ind] += 1
                        ext_ind = findfirst(x->x==bond[1], rgraph.atom_names)
                        rgraph.external_bonds[ext_ind] += 1

                    else
                        toResidue = i
                        toAtom = bond[2]
                    end

                    if fromAtom ∈ keys(atom_maps[fromResidue]) && toAtom ∈ keys(atom_maps[toResidue])
                        atom1 = atom_maps[fromResidue][fromAtom]
                        atom2 = atom_maps[toResidue][toAtom]
                        pair = atom1 < atom2 ? (atom1, atom2) : (atom2, atom1) 
                        if !(pair ∈ bonds)
                            push!(bonds, pair)
                            if !external
                                
                                i_local = findfirst(x -> x == fromAtom, rgraph.atom_names)
                                j_local = findfirst(x -> x == toAtom,   rgraph.atom_names)

                                pair_local = i_local < j_local ? (i_local, j_local) : (j_local, i_local)

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

function atom_name_by_index(atom_idx, canon_system)

    for (chain, resids) in canon_system

        for (res_id, rgraph) in resids

            if !(atom_idx ∈ rgraph.atom_inds)
                continue
            else
                local_idx = findfirst(i -> i == atom_idx, rgraph.atom_inds)
                return rgraph.atom_names[local_idx]
            end

        end
    end

end

function create_disulfide_bonds(coords, boundary, canon_system, bonds)

    function is_cysx(rgraph::ResidueGraph)
        names = rgraph.atom_names
        return "SG" in names && !("HG" in names)
    end
    
    function is_disulfide_bonded(atom_idx)
        for b in bonds
            atom_name_i = atom_name_by_index(b[1], canon_system)
            atom_name_j = atom_name_by_index(b[2], canon_system)
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
    #= atom_names = [[name for name in rg.atom_names] for rg in cysx] =#
    n_cysx = length(cysx)
    for (cys_idx, cysi) in enumerate(cysx)
        
        sg1_idx  = findfirst(x -> x == "SG", cysi.atom_names)
        atom_idx = cysi.atom_inds[sg1_idx]
        pos1     = coords[atom_idx]

        candidate_distance =  unit(eltype(coords[1])) == NoUnits ? 0.3 : 0.3u"nm"
        candidate_atom     = nothing 

        for cys_jdx in cys_idx:n_cysx
            cysj = cysx[cys_jdx]
            sg2_idx  = findfirst(x -> x == "SG", cysj.atom_names)
            atom_jdx = cysj.atom_inds[sg2_idx]
            pos2     = coords[atom_jdx]

            vec = vector(pos1, pos2, boundary)
            dst = norm(vec)

            if dst < candidate_distance && !is_disulfide_bonded(atom_idx)
                candidate_distance = dst
                candidate_atom = atom_jdx
            end

            if !isnothing(candidate_atom)
                pair = atom_idx < atom_jdx ? (atom_idx, atom_jdx) : (atom_jdx, atom_idx)
                push!(bonds, pair)
            end
        end
    end

    sort!(bonds)
    return bonds

end

function residue_from_atom_idx(atom_idx, canon_system)

    for (chain, resids) in canon_system
        for (res_id, rgraph) in resids
            if atom_idx ∈ rgraph.atom_inds
                return rgraph
            end
        end
    end

end

function read_connect_bonds(pdbfile, bonds, canon_system)
    filtered = String[]
    open(pdbfile) do io
        for line in eachline(io)
            if startswith(line, "CONECT")
                push!(filtered, line)
                fields = split(line)[2:end]
                fromAtom = parse(Int, fields[1])
                atom_name_i = atom_name_by_index(fromAtom, canon_system)
                res_i       = residue_from_atom_idx(fromAtom, canon_system)
                for toAtom in fields[2:end]
                    toAtom = parse(Int, toAtom)
                    atom_name_j = atom_name_by_index(toAtom, canon_system)
                    res_j       = residue_from_atom_idx(toAtom, canon_system)
                    pair = fromAtom < toAtom ? (fromAtom, toAtom) : (toAtom, fromAtom)
                    if !(pair ∈ bonds)
                        push!(bonds, pair)
                    end
                    if res_i == res_j
                        local_i = findfirst(n -> n == atom_name_i, res_i.atom_names)
                        local_j = findfirst(n -> n == atom_name_j, res_i.atom_names)
                        local_pair = local_i < local_j ? (local_i, local_j) : (local_j, local_i)
                        if !(local_pair ∈ res_i.bonds)
                            push!(res_i.bonds, local_pair)
                        end
                    end
                end
            end
        end
    end
    return bonds
end

"""
    match_residue_to_template(res::ResidueGraph,
                              tpl::ResidueTemplate;
                              ignoreExternalBonds::Bool=false,
                              ignoreExtraParticles::Bool=false) -> Union{Vector{Int},Nothing}

Replicates OpenMM's `matchResidueToTemplate`.

- Extras in residue are atoms with `res.elements[i] == :X`.
- Extras in template are `tpl.extras[j] == true`.
- If `ignoreExtraParticles`, extras are removed from both sides before matching.
- If `ignoreExternalBonds`, external bond counts are ignored.

Returns a vector `match` of length equal to the number of residue atoms kept
(after optional extra filtering). `match[i]` is the **template atom index (1-based, original template indexing)**
corresponding to the `i`-th kept residue atom. Returns `nothing` if no match exists.
"""
function match_residue_to_template(res::ResidueGraph,
                                   tpl::ResidueTemplate;
                                   ignoreExternalBonds::Bool=false,
                                   ignoreExtraParticles::Bool=false)::Union{Vector{Int}, Nothing}

    # --- 0) Define extra-particle predicates ---
    is_extra_res(i) = res.elements[i] == :X
    is_extra_tpl(j) = tpl.extras[j]

    # --- 1) Select atoms to consider (apply ignoreExtraParticles) ---
    res_keep = ignoreExtraParticles ? findall(i -> !is_extra_res(i), eachindex(res.atom_names)) : collect(eachindex(res.atom_names))
    tpl_keep = ignoreExtraParticles ? findall(j -> !is_extra_tpl(j), eachindex(tpl.atoms)) : collect(eachindex(tpl.atoms))

    numAtoms = length(res_keep)
    if numAtoms != length(tpl_keep)
        return nothing
    end
    if numAtoms == 0
        return Int[]  # both empty after filtering → vacuous match
    end

    # --- 2) Build local index maps (kept-only) ---
    res_old2new = Dict{Int,Int}(res_keep[k] => k for k in 1:numAtoms)
    tpl_old2new = Dict{Int,Int}(tpl_keep[k] => k for k in 1:numAtoms)
    tpl_new2old = copy(tpl_keep)  # inverse map to original template indices

    # --- 3) Build adjacency among kept atoms and external-bond counts ---
    # Residue: local bonds are given in res.bonds over original local indices.
    res_adj = [Int[] for _ in 1:numAtoms]
    for (i,j) in res.bonds
        (haskey(res_old2new, i) && haskey(res_old2new, j)) || continue
        ii = res_old2new[i]; jj = res_old2new[j]
        push!(res_adj[ii], jj); push!(res_adj[jj], ii)
    end
    res_ext = ignoreExternalBonds ? fill(0, numAtoms) : [res.external_bonds[res_keep[k]] for k in 1:numAtoms]

    # Template: build adjacency from tpl.bonds, but only within kept atoms.
    tpl_adj = [Int[] for _ in 1:numAtoms]
    for (i,j) in tpl.bonds
        (haskey(tpl_old2new, i) && haskey(tpl_old2new, j)) || continue
        ii = tpl_old2new[i]; jj = tpl_old2new[j]
        push!(tpl_adj[ii], jj); push!(tpl_adj[jj], ii)
    end
    tpl_ext = ignoreExternalBonds ? fill(0, numAtoms) : [tpl.external_bonds[tpl_keep[k]] for k in 1:numAtoms]

    # --- 4) Quick type-count screen: (element or :X, degree, ext) multiplicities must match ---
    # Residue keys
    res_keys = Tuple{Symbol,Int,Int}[]
    for i in 1:numAtoms
        key = (res.elements[res_keep[i]], length(res_adj[i]), res_ext[i])
        push!(res_keys, key)
    end
    # Template keys
    tpl_keys = Tuple{Symbol,Int,Int}[]
    for k in 1:numAtoms
        # Use template element symbol, but treat extras specially in candidate stage
        key = (tpl.elements[tpl_keep[k]], length(tpl_adj[k]), tpl_ext[k])
        push!(tpl_keys, key)
    end
    # Compare multisets
    sort!(res_keys); sort!(tpl_keys)
    if res_keys != tpl_keys
        return nothing
    end

    # --- 5) Candidate template atoms for each residue atom ---
    # OpenMM's exactNameMatch: if residue atom is extra and there exists a template extra
    # with same name, enforce name equality. Otherwise extra can map to any template extra.
    # Non-extra must match element exactly and template must be non-extra.
    candidates = Vector{Vector{Int}}(undef, numAtoms)
    # Precompute template-extra presence by name
    tpl_extra_name_set = Set{String}(tpl.atoms[j] for j in tpl_keep if is_extra_tpl(j))

    for i in 1:numAtoms
        ri_old = res_keep[i]
        r_el   = res.elements[ri_old]
        r_name = res.atom_names[ri_old]
        r_deg  = length(res_adj[i])
        r_ext  = res_ext[i]
        r_is_extra = (r_el == :X)

        exactNameMatch = r_is_extra && any(is_extra_tpl(j) && tpl.atoms[j] == r_name for j in tpl_keep)

        cands = Int[]
        for (k, tj_old) in enumerate(tpl_keep)
            t_el   = tpl.elements[tj_old]
            t_name = tpl.atoms[tj_old]
            t_deg  = length(tpl_adj[k])
            t_ext  = tpl_ext[k]
            t_is_extra = is_extra_tpl(tj_old)

            # Element/name gate
            if r_is_extra
                # residue extra → template must be extra
                t_is_extra || continue
                if exactNameMatch && t_name != r_name
                    continue
                end
            else
                # residue real element → template must be non-extra and element equal
                t_is_extra && continue
                t_el == r_el || continue
            end

            # Degree and external-bond checks
            r_deg == t_deg || continue
            ignoreExternalBonds || (r_ext == t_ext) || continue

            push!(cands, k)  # store template new-index k
        end
        # Early prune: if no candidates for a residue atom, fail
        isempty(cands) && return nothing
        candidates[i] = cands
    end

    # --- 6) Heuristic search order: fewest candidates first, then neighbors of chosen ---
    atomsToOrder = Set(1:numAtoms)
    searchOrder = Int[]
    neighbor_heap = Int[]  # acts as an unordered list of candidate neighbors

    while !isempty(atomsToOrder)
        if isempty(neighbor_heap)
            # pick global minimum by candidate count among remaining
            nextAtom = argmin(i -> length(candidates[i]), collect(atomsToOrder))
        else
            # pick the neighbor with fewest candidates
            sort!(neighbor_heap, by=i -> length(candidates[i]))
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
        cand_ord[pos]    = candidates[i]
    end

    # --- 7) Recursive backtracking with bond-consistency ---
    matches_tpl = fill(0, numAtoms)   # at position pos, matched template new-index
    used_tpl    = falses(numAtoms)

    function dfs(pos::Int)::Bool
        pos > numAtoms && return true
        # Try candidates for this residue position
        for t_new in cand_ord[pos]
            if used_tpl[t_new]
                continue
            end
            # Check bond consistency with already assigned neighbors
            ok = true
            for nb_pos in res_adj_ord[pos]
                if nb_pos < pos  # already assigned
                    t_nb = matches_tpl[nb_pos]
                    # must be bonded in template
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
            if dfs(pos+1)
                return true
            end
            used_tpl[t_new] = false
        end
        return false
    end

    if !dfs(1)
        return nothing
    end

    # --- 8) Return mapping back to original template indices, in original residue-kept order ---
    # We need mapping for residue atoms in kept-order, not search-order.
    # matches_tpl is in search-order; invert back:
    matches_tpl_in_res_order = similar(matches_tpl)
    for pos in 1:numAtoms
        i = searchOrder[pos]
        matches_tpl_in_res_order[i] = matches_tpl[pos]
    end

    # Convert template new-indices to original template indices
    return [tpl_new2old[t_new] for t_new in matches_tpl_in_res_order]
end

# ---- Global adjacency from bonds ----
function build_adjacency(natoms::Int, bonds::Vector{NTuple{2,Int}})
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

# ---- Angles (i,j,k) with j as center; unique by i<k ----
function build_angles(adj::Vector{Vector{Int}}, bonds)
    angles = Vector{NTuple{3,Int}}()
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


function build_torsions(adj::Vector{Vector{Int}}, angles::Vector{NTuple{3,Int}})
    tors = Vector{NTuple{4,Int}}()
    for angle in angles
        for atom in adj[angle[1]]
            if !(atom ∈ angle)
                if atom < angle[3]
                    push!(tors, (atom, angle[1], angle[2], angle[3]))
                else
                    push!(tors, (angle[3], angle[2], angle[1], atom))
                end
            end
        end
        for atom in adj[angle[3]]
            if !(atom ∈ angle)
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

function combinations_of(vec::Vector, n::Int)
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

# ---- Impropers (c, j, k, l) with c as center; i<j<k for uniqueness ----
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