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
    res_id::Tuple{Int,String}
    res_name::String              # includes N/C prefix if terminal
    atom_inds::Vector{Int}        # global 1-based
    atom_names::Vector{String}
    bonds::Vector{Tuple{Int,Int}} # local intra-res bonds
    external_bonds::Vector{Int}   # per-atom external degree
end

# Renaming rule (prefix N or C for termini).
function residue_name_with_term_flag(res, res_id_to_standard::Dict{Tuple{Int,String},Bool})
    res_id = get_res_id(res)
    base = Chemfiles.name(res)
    name = base
    if get(res_id_to_standard, res_id, false)
        prev_id = (res_id[1]-1, res_id[2])
        next_id = (res_id[1]+1, res_id[2])
        is_prev_std = get(res_id_to_standard, prev_id, false)
        is_next_std = get(res_id_to_standard, next_id, false)
        if !is_prev_std
            name = "N"*base
        elseif !is_next_std
            name = "C"*base
        end
    end
    return name
end

# Build map of (resid,chain) → is_standard
function compute_res_standard_map(top::Chemfiles.Topology)
    m = Dict{Tuple{Int,String},Bool}()
    for ri in 1:Chemfiles.count_residues(top)
        res   = chemfiles_residue(top, ri-1)
        rid   = get_res_id(res)
        rname = Chemfiles.name(res)
        m[rid] = (rname in standard_res_names)
    end
    return m
end

"""
    build_residue_graphs(top::Chemfiles.Topology)
        -> Dict{Tuple{Int,String},ResidueGraph}

Construct residue-local graphs and external-bond counts from a Chemfiles topology.
Global atom indices are 1-based. Bond lists use local indices within each residue.
"""
function build_residue_graphs(top::Chemfiles.Topology)
    n_atoms = size(top)

    # Build map for terminal detection (standard vs not).
    res_id_to_standard = compute_res_standard_map(top)

    res_graphs = Dict{Tuple{Int,String},ResidueGraph}()
    atom_to_res = fill(((0,""), 0), n_atoms)  # ((res_id,chain), local_idx)

    # First pass: residues, local membership
    for ri in 1:Chemfiles.count_residues(top)
        res       = chemfiles_residue(top, ri-1)
        rid       = get_res_id(res)
        rname     = residue_name_with_term_flag(res, res_id_to_standard) # N/C prefix if terminal
        atoms0    = Int.(Chemfiles.atoms(res))           # 0-based
        atom_inds = atoms0 .+ 1                          # 1-based
        atom_names = chemfiles_name.((top,), atoms0)

        for (li, ai) in enumerate(atom_inds)
            atom_to_res[ai] = (rid, li)
        end

        res_graphs[rid] = ResidueGraph(rid, rname, atom_inds, atom_names,
                                       Tuple{Int,Int}[], fill(0, length(atom_inds)))
    end

    # Second pass: start from Chemfiles bonds
    bonds_mat = Int.(Chemfiles.bonds(top))  # 2×NB, 0-based
    for col in eachcol(bonds_mat)
        i = col[1]+1; j = col[2]+1
        (rid_i, li) = atom_to_res[i]
        (rid_j, lj) = atom_to_res[j]
        if rid_i == ((0,""),0) || rid_j == ((0,""),0)
            continue
        end

        if rid_i == rid_j
            rg = res_graphs[rid_i]
            push!(rg.bonds, li < lj ? (li, lj) : (lj, li))
        else
            res_graphs[rid_i].external_bonds[li] += 1
            res_graphs[rid_j].external_bonds[lj] += 1
        end
    end

    # Canonicalize bond lists
    for rg in values(res_graphs)
        unique!(rg.bonds)
        sort!(rg.bonds)
    end

    return res_graphs
end

# Heuristics to detect "extra particle" names and infer element.
# Adjust to your conventions. If you already tag extras elsewhere, replace these.
name_startswith_ep(nm::String) = startswith(nm, "EP") || startswith(nm, "VS") || false

function infer_element_from_name(nm::String)::Symbol
    # Try simple OpenMM-like convention: first letter(s) capital, then digits
    # Fall back to :X
    i = findfirst(isletter, nm)
    if i === nothing
        return :X
    end
    c = nm[i]
    if i < lastindex(nm) && islowercase(nm[nextind(nm,i)])
        sym = string(c, nm[nextind(nm,i)])
        return get(ELEMENT_SYMBOLS, sym, get(ELEMENT_SYMBOLS, string(c), :X))
    else
        return get(ELEMENT_SYMBOLS, string(c), :X)
    end
end

"""
    atom_degrees(natoms::Int, bonds::Vector{Tuple{Int,Int}}) -> Vector{Int}

Compute per-atom internal degree from intra-residue bonds.
"""
function atom_degrees(natoms::Int, bonds::Vector{Tuple{Int,Int}})
    deg = fill(0, natoms)
    @inbounds for (i,j) in bonds
        deg[i] += 1
        deg[j] += 1
    end
    return deg
end

# Adjacency sets
function adjacency_sets(n::Int, bonds::Vector{Tuple{Int,Int}})
    adj = [Int[] for _ in 1:n]
    @inbounds for (i,j) in bonds
        push!(adj[i], j); push!(adj[j], i)
    end
    # keep unique and sorted
    for a in adj
        unique!(a); sort!(a)
    end
    return adj
end


"""
    template_signature(t::ResidueTemplate; include_external::Bool=false) -> String

Canonical, hashed signature:
- Non-extra atoms: multiset of `(element, degree)` or `(element, degree, external)` if requested.
- Extra particles: exact atom-name multiset.
"""
function template_signature(t::ResidueTemplate; include_external::Bool=false)
    n = length(t.atoms)
    deg = atom_degrees(n, t.bonds)

    nonextra_parts = String[]
    extra_parts    = String[]

    @inbounds for i in 1:n
        if t.extras[i]
            # exact-name marker for extra particles
            push!(extra_parts, "X:" * t.atoms[i])
        else
            if include_external
                push!(nonextra_parts, string(t.elements[i], "/", deg[i], "/", t.external_bonds[i]))
            else
                push!(nonextra_parts, string(t.elements[i], "/", deg[i]))
            end
        end
    end

    sort!(nonextra_parts)
    sort!(extra_parts)

    return "NE[" * join(nonextra_parts, ",") * "]|EX[" * join(extra_parts, ",") * "]"
end

function residue_signature(rg::ResidueGraph)
    n = length(rg.atom_names)
    deg = atom_degrees(n, rg.bonds)
    parts = String[]
    @inbounds for i in 1:n
        el = infer_element_from_name(rg.atom_names[i])
        push!(parts, string(el, "/", deg[i]))
    end
    sort!(parts)
    return "NE[" * join(parts, ",") * "]|EX[]"
end

"""
    build_template_index(residues::Dict{String,ResidueTemplate};
                         include_external::Bool=false)
        -> Dict{String,Vector{String}}

Map signature => template names sharing that signature.
"""
function build_template_index(residues::Dict{String,ResidueTemplate{T}};
                              include_external::Bool=false) where T
    idx = Dict{String,Vector{String}}()
    for (rname, tmpl) in residues
        sig = template_signature(tmpl; include_external)
        push!(get!(idx, sig, String[]), rname)
    end
    return idx
end


"""
    element_multiset_residue(rg) -> Dict{Symbol,Int}

Infer element counts from atom names only.
"""
function element_multiset_residue(rg::ResidueGraph)
    counts = Dict{Symbol,Int}()
    @inbounds for nm in rg.atom_names
        el = infer_element_from_name(nm)   # your existing helper
        counts[el] = get(counts, el, 0) + 1
    end
    counts
end

"""
    element_multiset_template(tmpl) -> Dict{Symbol,Int}
"""
function element_multiset_template(tmpl::ResidueTemplate)
    counts = Dict{Symbol,Int}()
    @inbounds for el in tmpl.elements
        counts[el] = get(counts, el, 0) + 1
    end
    counts
end

candidate_templates(rg::ResidueGraph, idx::Dict{String,Vector{String}}) =
    get(idx, residue_signature(rg), String[])

"""
    candidate_templates_by_elements(rg, templates; allow_superset=false)

Return template names whose element multiset matches the residue.
If `allow_superset=true`, also accept templates that have additional H only.
"""
function candidate_templates_by_elements(rg::ResidueGraph,
                                         templates::Dict{String, ResidueTemplate{T}};
                                         allow_superset::Bool=false) where T
    rc = element_multiset_residue(rg)
    names = String[]
    for (nm, tmpl) in templates
        tc = element_multiset_template(tmpl)
        if !allow_superset
            if rc == tc
                push!(names, nm)
            end
        else
            # require tmpl has at least the residue counts, and any excess are H only
            ok = true
            for (el, n) in rc
                if get(tc, el, -1) != n
                    ok = false; break
                end
            end
            if ok
                extra = Dict{Symbol,Int}((el => tc[el] - get(rc, el, 0)) for el in keys(tc))
                delete!(extra, :H)
                if all(v -> v == 0, values(extra))
                    push!(names, nm)
                end
            end
        end
    end
    return names
end

"""
    partial_match_residue_to_template(rg, tmpl; ignore_external_bonds=true, ignore_extra_particles=true)
Return (res2tpl::Dict{Int,Int}, unmatched_tpl::Vector{Int}) if residue can embed in template,
else (nothing, nothing). Allows residue to be a subgraph of the template.
"""
function partial_match_residue_to_template(rg::ResidueGraph, tmpl::ResidueTemplate;
        ignore_external_bonds::Bool=true,
        ignore_extra_particles::Bool=true)

    res_keep = collect(1:length(rg.atom_names))                    # keep all residue atoms
    tpl_keep = ignore_extra_particles ? findall(!, tmpl.extras) : collect(1:length(tmpl.atoms))

    n_res, n_tpl = length(res_keep), length(tpl_keep)

    # residue view
    res_old2new = Dict(i=>k for (k,i) in enumerate(res_keep))
    res_bonds = Tuple{Int,Int}[]
    keep = Set(res_keep)
    for (i,j) in rg.bonds
        if i in keep && j in keep
            ii = res_old2new[i]; jj = res_old2new[j]
            push!(res_bonds, ii<jj ? (ii,jj) : (jj,ii))
        end
    end
    res_deg = atom_degrees(n_res, res_bonds)
    res_ext = ignore_external_bonds ? fill(0, n_res) : [rg.external_bonds[res_keep[i]] for i in 1:n_res]
    res_el  = [infer_element_from_name(rg.atom_names[res_keep[i]]) for i in 1:n_res]
    res_adj = adjacency_sets(n_res, res_bonds)

    # template view
    tpl_old2new = Dict(i=>k for (k,i) in enumerate(tpl_keep))
    tpl_bonds = Tuple{Int,Int}[]
    keepT = Set(tpl_keep)
    for (i,j) in tmpl.bonds
        if i in keepT && j in keepT
            ii = tpl_old2new[i]; jj = tpl_old2new[j]
            push!(tpl_bonds, ii<jj ? (ii,jj) : (jj,ii))
        end
    end
    tpl_deg = atom_degrees(n_tpl, tpl_bonds)
    tpl_ext = ignore_external_bonds ? fill(0, n_tpl) : [tmpl.external_bonds[tpl_keep[i]] for i in 1:n_tpl]
    tpl_el  = [tmpl.elements[tpl_keep[i]] for i in 1:n_tpl]
    tpl_adj = adjacency_sets(n_tpl, tpl_bonds)

    # candidates: residue atom i can map to template atom j if attributes compatible
    cand = [Int[] for _ in 1:n_res]
    for i in 1:n_res
        for j in 1:n_tpl
            if res_el[i]==tpl_el[j] && res_deg[i] <= tpl_deg[j] && (ignore_external_bonds || res_ext[i] <= tpl_ext[j])
                push!(cand[i], j)
            end
        end
        isempty(cand[i]) && return nothing, nothing
    end

    order = sortperm(1:n_res; by=i->(length(cand[i]), -res_deg[i]))
    used_tpl = fill(false, n_tpl)
    map_res2tpl = fill(0, n_res)

    function dfs(k::Int)::Bool
        k > n_res && return true
        i = order[k]
        for j in cand[i]
            used_tpl[j] && continue
            ok = true
            for n in res_adj[i]
                m = map_res2tpl[n]
                if m!=0 && !(m in tpl_adj[j]); ok=false; break; end
            end
            ok || continue
            map_res2tpl[i]=j; used_tpl[j]=true
            dfs(k+1) && return true
            map_res2tpl[i]=0; used_tpl[j]=false
        end
        return false
    end

    dfs(1) || return nothing, nothing

    res2tpl = Dict(res_keep[i] => tpl_keep[map_res2tpl[i]] for i in 1:n_res)
    unmatched_tpl = [tpl_keep[j] for j in 1:n_tpl if !used_tpl[j]]
    return res2tpl, unmatched_tpl
end

"""
    match_residue_to_template(rg::ResidueGraph, tmpl::ResidueTemplate;
                              ignore_external_bonds::Bool=false,
                              ignore_extra_particles::Bool=false)
        -> Union{Dict{Int,Int},Nothing}

Backtracking graph match from residue `rg` to template `tmpl`.

Constraints:
- Non-extra atoms must match element.
- Internal degree must match.
- External-bond count must match unless `ignore_external_bonds`.
- Extra particles must match by exact atom name unless `ignore_extra_particles`.
- Bonds in `rg` must map to bonds in `tmpl`.

Returns a mapping of local residue indices → template indices, or `nothing` if no match.
"""
function match_residue_to_template(rg::ResidueGraph, tmpl::ResidueTemplate;
                                   ignore_external_bonds::Bool=false,
                                   ignore_extra_particles::Bool=false)

    # --- Prepare residue view (optionally drop extras) ---
    # Identify extras in residue by element :X (or any policy you use)
    res_is_extra = [name_startswith_ep(n) || false for n in rg.atom_names]  # fallback false
    if ignore_extra_particles
        res_keep = findall(!, res_is_extra)
        tpl_keep = findall(!, tmpl.extras)
    else
        res_keep = collect(1:length(rg.atom_names))
        tpl_keep = collect(1:length(tmpl.atoms))
        # quick cardinality check for extras by exact name
        if count(identity, res_is_extra) != count(identity, tmpl.extras)
            return nothing
        end
    end

    # Quick size check
    if length(res_keep) != length(tpl_keep)
        return nothing
    end
    n_keep = length(res_keep)

    # --- Build local indexing for residue subset ---
    res_old2new = Dict{Int,Int}(i => k for (k,i) in enumerate(res_keep))
    res_new2old = res_keep

    # Residue intra-bonds restricted to kept atoms
    res_bonds_kept = Tuple{Int,Int}[]
    begin
        keep_set = Set(res_keep)
        for (i,j) in rg.bonds
            if i in keep_set && j in keep_set
                ii = res_old2new[i]
                jj = res_old2new[j]
                push!(res_bonds_kept, ii < jj ? (ii,jj) : (jj,ii))
            end
        end
    end

    # Residue degrees and external counts
    res_deg  = atom_degrees(n_keep, res_bonds_kept)
    res_ext  = ignore_external_bonds ? fill(0, n_keep) :
               [rg.external_bonds[res_new2old[i]] for i in 1:n_keep]
    res_elem = Vector{Symbol}(undef, n_keep)
    res_name = Vector{String}(undef, n_keep)
    for i in 1:n_keep
        res_elem[i] = infer_element_from_name(rg.atom_names[res_new2old[i]])
        res_name[i] = rg.atom_names[res_new2old[i]]
    end

    # --- Build template view on kept atoms ---
    tpl_old2new = Dict{Int,Int}(i => k for (k,i) in enumerate(tpl_keep))
    tpl_new2old = tpl_keep
    tpl_bonds_kept = Tuple{Int,Int}[]
    begin
        keep_set = Set(tpl_keep)
        for (i,j) in tmpl.bonds
            if i in keep_set && j in keep_set
                ii = tpl_old2new[i]; jj = tpl_old2new[j]
                push!(tpl_bonds_kept, ii < jj ? (ii,jj) : (jj,ii))
            end
        end
    end

    tpl_deg  = atom_degrees(n_keep, tpl_bonds_kept)
    tpl_ext  = ignore_external_bonds ? fill(0, n_keep) :
               [tmpl.external_bonds[tpl_new2old[i]] for i in 1:n_keep]
    tpl_elem = [tmpl.elements[tpl_new2old[i]] for i in 1:n_keep]
    tpl_name = [tmpl.atoms[tpl_new2old[i]]     for i in 1:n_keep]
    tpl_is_extra = [tmpl.extras[tpl_new2old[i]] for i in 1:n_keep]

    # --- Quick multiset filters on non-extras ---
    nonextra_idx = ignore_extra_particles ? collect(1:n_keep) :
                    [i for i in 1:n_keep if !tpl_is_extra[i]]
    # Multiset equality on tuples
    multiset_res = sort!(string.(res_elem[nonextra_idx], "/", res_deg[nonextra_idx], "/", res_ext[nonextra_idx]))
    multiset_tpl = sort!(string.(tpl_elem[nonextra_idx], "/", tpl_deg[nonextra_idx], "/", tpl_ext[nonextra_idx]))
    if multiset_res != multiset_tpl
        return nothing
    end
    # Extra particles must match names 1:1 if not ignoring
    if !ignore_extra_particles
        res_extra_names = sort!([res_name[i] for i in 1:n_keep if name_startswith_ep(res_name[i])])
        tpl_extra_names = sort!([tpl_name[i] for i in 1:n_keep if tpl_is_extra[i]])
        if res_extra_names != tpl_extra_names
            return nothing
        end
    end

    # --- Adjacency sets for fast bond checks ---
    res_adj = adjacency_sets(n_keep, res_bonds_kept)
    tpl_adj = adjacency_sets(n_keep, tpl_bonds_kept)

    # --- Candidate template indices per residue atom ---
    candidates = Vector{Vector{Int}}(undef, n_keep)
    for i in 1:n_keep
        if !ignore_extra_particles && name_startswith_ep(res_name[i])
            # exact-name match against template extras
            c = Int[]
            for j in 1:n_keep
                if tpl_is_extra[j] && tpl_name[j] == res_name[i]
                    push!(c, j)
                end
            end
            candidates[i] = c
            if isempty(c)
                return nothing
            end
        else
            # Non-extra matching by attributes
            c = Int[]
            for j in 1:n_keep
                if (!tpl_is_extra[j]) && (res_elem[i] == tpl_elem[j]) && (res_deg[i]  == tpl_deg[j])  && (ignore_external_bonds || res_ext[i] == tpl_ext[j])
                    push!(c, j)
                end
            end
            candidates[i] = c
            if isempty(c)
                return nothing
            end
        end
    end

    # --- Variable ordering: fewest candidates first, then by larger degree ---
    order = collect(1:n_keep)
    sort!(order; by = i -> (length(candidates[i]), -res_deg[i]))

    # --- Backtracking ---
    mapping_res2tpl = fill(0, n_keep)       # 0 = unassigned
    used_tpl        = fill(false, n_keep)

    function dfs(k::Int)::Bool
        if k > n_keep
            return true
        end
        i = order[k]
        # Prefer neighbors of already-matched atoms to propagate constraints
        cand = candidates[i]
        # Heuristic: try candidates that are adjacent to already matched neighbors first
        neigh_matched = [mapping_res2tpl[n] for n in res_adj[i] if mapping_res2tpl[n] != 0]
        sort!(cand; by = j -> -count(==(true), [n in tpl_adj[j] for n in neigh_matched]))
        for j in cand
            if used_tpl[j]
                continue
            end
            # Check bond consistency with already mapped neighbors
            ok = true
            for n in res_adj[i]
                m = mapping_res2tpl[n]
                if m != 0
                    if !(m in tpl_adj[j])  # residue has bond i-n, template must have j-m
                        ok = false
                        break
                    end
                end
            end
            if !ok
                continue
            end
            # Assign
            mapping_res2tpl[i] = j
            used_tpl[j] = true
            if dfs(k+1) 
                return true
            end
            # Undo
            mapping_res2tpl[i] = 0
            used_tpl[j] = false
        end
        return false
    end

    if !dfs(1)
        return nothing
    end

    # Build mapping in original residue local indices
    res_map = Dict{Int,Int}()
    for i_new in 1:n_keep
        j_new = mapping_res2tpl[i_new]
        i_old = res_new2old[i_new]
        j_old = tpl_new2old[j_new]
        res_map[i_old] = j_old
    end
    return res_map
end

"""
    diff_residue_vs_template(rg, tmpl, res2tpl)
Return bonds between already-present residue atoms that are required by the template but missing.
Atoms are not added.
"""
function diff_residue_vs_template(rg::ResidueGraph, tmpl::ResidueTemplate, res2tpl::Dict{Int,Int})
    tpl2res = Dict{Int,Int}(v=>k for (k,v) in res2tpl)
    have = Set{Tuple{Int,Int}}( (i<j ? (i,j) : (j,i)) for (i,j) in rg.bonds )

    missing_bonds = Tuple{Int,Int}[]
    for (ti,tj) in tmpl.bonds
        ri = get(tpl2res, ti, 0); rj = get(tpl2res, tj, 0)
        if ri!=0 && rj!=0
            ij = ri<rj ? (ri,rj) : (rj,ri)
            if !(ij in have); push!(missing_bonds, ij); end
        end
    end
    return missing_bonds
end

"""
    apply_missing_bonds!(rg, missing_bonds)
Add any missing intra-residue bonds between existing atoms.
"""
function apply_missing_bonds!(rg::ResidueGraph, missing_bonds::Vector{Tuple{Int,Int}})
    have = Set{Tuple{Int,Int}}(rg.bonds)
    for (i,j) in missing_bonds
        ij = i<j ? (i,j) : (j,i)
        (ij in have) && continue
        push!(rg.bonds, ij); push!(have, ij)
    end
    sort!(rg.bonds); unique!(rg.bonds)
    return rg
end

"""
    repair_intra_residue_bonds_from_templates!(rg, templates, tmpl_index)
Attempt to fix missing intra-residue bonds using residue templates only.
No atom additions. No coordinates. Returns true if any bonds were added.
"""
function repair_intra_residue_bonds_from_templates!(rg::ResidueGraph,
                                                    templates::Dict{String,ResidueTemplate{T}}) where T
    # choose candidates by residue signature
    # Replace signature-based pruning:
    cands = candidate_templates_by_elements(rg, templates; allow_superset=false)

    # Then do template-guided bond repair:
    for tname in cands
        tmpl = templates[tname]
        res2tpl, _ = partial_match_residue_to_template(rg, tmpl;
                        ignore_external_bonds=true, ignore_extra_particles=true)
        res2tpl === nothing && continue
        miss = diff_residue_vs_template(rg, tmpl, res2tpl)
        isempty(miss) && continue
        apply_missing_bonds!(rg, miss)
        break
    end
end

# ---- Bonds from residue graphs -> global set ----
function build_global_bonds(res_graphs)::Vector{NTuple{2,Int}}
    est = sum(length(rg.bonds) for rg in values(res_graphs))
    bonds = Vector{NTuple{2,Int}}(undef, 0); sizehint!(bonds, est)
    seen = Set{NTuple{2,Int}}()
    for rg in values(res_graphs)
        aid = rg.atom_inds
        for (i,j) in rg.bonds
            a = aid[i]; b = aid[j]
            p = a < b ? (a,b) : (b,a)
            if !(p in seen)
                push!(bonds, p); push!(seen, p)
            end
        end
    end
    return bonds
end

# ---- Global adjacency from bonds ----
function build_adjacency(natoms::Int, bonds::Vector{NTuple{2,Int}})
    adj = [Int[] for _ in 1:natoms]
    @inbounds for (i,j) in bonds
        push!(adj[i], j); push!(adj[j], i)
    end
    for a in adj
        unique!(a); sort!(a)
    end
    return adj
end

# ---- Angles (i,j,k) with j as center; unique by i<k ----
function build_angles(adj::Vector{Vector{Int}})
    angles = Vector{NTuple{3,Int}}()
    for j in 1:length(adj)
        nbr = adj[j]
        ln = length(nbr)
        ln < 2 && continue
        for u in 1:ln-1, v in u+1:ln
            i = nbr[u]; k = nbr[v]
            push!(angles, i < k ? (i,j,k) : (k,j,i))
        end
    end
    unique!(angles); return angles
end

# ---- Proper torsions (i,j,k,l); unique and orientation-consistent ----
function build_torsions(adj::Vector{Vector{Int}}, bonds::Vector{NTuple{2,Int}})
    tors = Vector{NTuple{4,Int}}()
    # iterate over each bond j-k once
    for (j,k) in bonds
        # neighbors excluding the bonded partner
        nj = (n for n in adj[j] if n != k)
        nk = (n for n in adj[k] if n != j)
        for i in nj, l in nk
            t = (i,j,k,l)
            # canonicalize: if (j>k) flip and reverse ends
            if j > k
                t = (l,k,j,i)
            elseif j == k
                continue
            end
            # also dedup mirrored ends by ordering (i,l)
            if t[1] > t[4]
                t = (t[4], t[3], t[2], t[1])
            end
            push!(tors, t)
        end
    end
    unique!(tors); return tors
end

# ---- Impropers (i, c, j, k) with c as center; i<j<k for uniqueness ----
function build_impropers(adj::Vector{Vector{Int}})
    imps = Vector{NTuple{4,Int}}()
    for c in 1:length(adj)
        nbr = adj[c]
        ln = length(nbr)
        ln < 3 && continue
        sort!(nbr)
        for a in 1:ln-2, b in a+1:ln-1, d in b+1:ln
            i, j, k = nbr[a], nbr[b], nbr[d]
            push!(imps, (i, c, j, k))
        end
    end
    unique!(imps); return imps
end