# Read in a force field

export
    MolecularForceField

@enum SpecKind::UInt8 WILD=0 TYPE=1 CLASS=2

struct AtomPattern
    kind::SpecKind
    val::String
end

function matches(p::AtomPattern, t::String, type_to_class::Dict{String, String})
    if p.kind == WILD
        return true
    elseif p.kind == TYPE
        return t == p.val
    else # CLASS
        cls = get(type_to_class, t, "")
        return cls == p.val
    end
end

spec_score(ap::AtomPattern) = (ap.kind==TYPE ? 2 : (ap.kind==CLASS ? 1 : 0))

function pattern_from_attrs(n::EzXML.Node, typekey::AbstractString, classkey::AbstractString)
    if haskey(n, typekey)
        if haskey(n, classkey)
            error("a <$(n.name)> tag in the force field specifies both \"$typekey\" and " *
                  "\"$classkey\" for the same atom, only one of an atom type and an atom " *
                  "class can be given")
        end
        v = n[typekey]
        return (isempty(v) ? AtomPattern(WILD, "") : AtomPattern(TYPE, v))
    elseif haskey(n, classkey)
        v = n[classkey]
        return (isempty(v) ? AtomPattern(WILD, "") : AtomPattern(CLASS, v))
    else
        return AtomPattern(WILD, "")
    end
end

struct AtomType{C, M, S, E}
    type::String
    class::String
    element::String
    charge::Union{C, Missing}
    mass::M
    σ::S
    ϵ::E
    σ14::Union{S, Missing}
    ϵ14::Union{E, Missing}
end

struct PeriodicTorsionType{T, E}
    periodicities::Vector{Int}
    phases::Vector{T}
    ks::Vector{E}
    proper::Bool
end

struct HarmonicTorsionType{T, E}
    k::E
    θ0::T
end

struct CMAPTorsionType{T}
    size::Int
    energy::Vector{T}
end

struct NBFixPair{S, E}
    type1::String
    type2::String
    class1::String
    class2::String
    σ::S
    ϵ::E
end

# These are used to materialise the bonded terms present
# in the structure file, comparing them to the rules defined
# in the force field. Should be agnostic to class/type definitions.

# Bonds
struct BondRule{K,D}
    p1::AtomPattern
    p2::AtomPattern
    params::HarmonicBond{K, D}
    specificity::UInt8
end

struct BondResolver{K,D}
    rules::Vector{BondRule{K, D}}
    # Indexes: ordered type pair, ordered class pair, and a broad bin
    idx::Dict{Tuple{Symbol, String, String}, Vector{Int}}
    cache::Dict{Tuple{String, String}, Union{HarmonicBond{K, D}, Nothing}}
end

# Angles
struct AngleRule{K,D}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    params::HarmonicAngle{K, D}
    specificity::UInt8
end

# Urey-Bradley angle bond term
struct UreyBradleyRule{K, D}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    params::HarmonicBond{K, D}
    specificity::UInt8
end

struct AngleResolver{KA, DA, K, D}
    rules::Vector{Union{AngleRule{KA, DA}, UreyBradleyRule{K, D}}}
    # Index by center atom
    idx::Dict{Tuple{Symbol, String}, Vector{Int}} # (:type|:class|:wild, key)
    angle_cache::Dict{NTuple{3, String}, Union{HarmonicAngle{KA, DA}, Nothing}}
    urey_cache::Dict{NTuple{3, String}, Union{HarmonicBond{K, D}, Nothing}}
end

# Torsions
struct TorsionRule{T,E}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    p4::AtomPattern
    proper::Bool
    ordering::String   # "default" | "charmm" | "amber" | "smirnoff"
    has_wildcard::Bool # Any position is WILD
    params::PeriodicTorsionType{T,E}
    specificity::UInt8 # TYPE=2, CLASS=1, WILD=0, used to bias towards specific definitions
end

struct HarmonicTorsionRule{T,E}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    p4::AtomPattern
    proper::Bool
    has_wildcard::Bool # Any position is WILD
    params::HarmonicTorsionType{T,E}
    specificity::UInt8 # TYPE=2, CLASS=1, WILD=0, used to bias towards specific definitions
end

struct TorsionResolver{T,E}
    rules::Vector{Union{TorsionRule{T, E}, HarmonicTorsionRule{T, E}}}

    # Candidate lists keyed by type1 or class1 for impropers, and by type2 or class2 for propers
    impropers_by_type1::Dict{String, Vector{Int}}
    impropers_by_class1::Dict{String, Vector{Int}}
    propers_by_type2::Dict{String, Vector{Int}}
    propers_by_class2::Dict{String, Vector{Int}}
    wild_impropers::Vector{Int} # p1.kind == WILD
    wild_propers::Vector{Int}   # p2.kind == WILD

    # Caches
    # Propers: unordered signature {(t1,t2,t3,t4),(t4,t3,t2,t1)} -> rule index or :miss
    proper_cache::Dict{Tuple{NTuple{4, String}, NTuple{4, String}}, Any}
    # Impropers: ordered (t1,t2,t3,t4) -> (perm_indices::NTuple{4,Int}, rule_index) or :miss
    improper_cache::Dict{NTuple{4, String}, Any}
end

struct CMAPRule{E}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    p4::AtomPattern
    p5::AtomPattern
    has_wildcard::Bool
    params::CMAPTorsionType{E}
    specificity::UInt8 # TYPE=2, CLASS=1, WILD=0, used to bias towards specific definitions
end

struct CMAPResolver{E}
    rules::Vector{CMAPRule{E}}
    cache::Dict{Tuple{String,String,String,String,String}, CMAPTorsionType{E}}
end

# Proper torsions: lookup with cache
function find_proper_match(t1::AbstractString, t2::AbstractString, t3::AbstractString,
                           t4::AbstractString; resolver::TorsionResolver{T,E},
                           type_to_class::Dict{String, String}) where {T, E}
    # Unordered signature cache
    sig = ((t1, t2, t3, t4), (t4, t3, t2, t1))
    pc = resolver.proper_cache
    if haskey(pc, sig)
        v = pc[sig]
        if v == :miss
            return (nothing, nothing)
        else
            return (resolver.rules[v::Int].params, resolver.rules[v::Int].specificity)
        end
    end

    # Candidates by central atom 2 (type → class → wild)
    cand = Int[]
    c2 = type_to_class[t2]
    append!(cand, get(resolver.propers_by_type2,  t2, Int[]))
    append!(cand, get(resolver.propers_by_class2, c2, Int[]))
    append!(cand, resolver.wild_propers)

    best = 0
    bestspec = Int8(-1)
    # Try forward and reverse; prefer non-wildcard, otherwise highest specificity
    for (q1,q2,q3,q4) in ((t1,t2,t3,t4), (t4,t3,t2,t1))
        for i in cand
            r = resolver.rules[i]
            r.proper || continue
            if matches(r.p1, q1, type_to_class) && matches(r.p2, q2, type_to_class) &&
               matches(r.p3, q3, type_to_class) && matches(r.p4, q4, type_to_class)
                if !r.has_wildcard
                    pc[sig] = i
                    return (r.params, r.specificity)
                elseif r.specificity > bestspec
                    bestspec = r.specificity
                    best = i
                end
            end
        end
    end

    if best == 0
        pc[sig] = :miss
        return (nothing, nothing)
    else
        pc[sig] = best
        return (resolver.rules[best].params, resolver.rules[best].specificity)
    end
end

# Impropers: lookup with 6-permutation scan and cache
function find_improper_match(t1::AbstractString, t2::AbstractString, t3::AbstractString,
                             t4::AbstractString; resolver::TorsionResolver{T, E},
                             type_to_class::Dict{String, String}) where {T, E}
    key = (t1, t2, t3, t4)
    ic = resolver.improper_cache
    if haskey(ic, key)
        v = ic[key]
        if v == :miss
            return nothing
        else
            return resolver.rules[(v::Tuple{NTuple{4, Int}, Int})[2]].params
        end
    end

    # Candidates by central atom 1 (type → class → wild)
    cand = Int[]
    c1 = type_to_class[t1]
    append!(cand, get(resolver.impropers_by_type1,  t1, Int[]))
    append!(cand, get(resolver.impropers_by_class1, c1, Int[]))
    append!(cand, resolver.wild_impropers)

    best = 0
    bestperm = (1,2,3,4)
    bestspec = Int8(-1)

    for (p2,p3,p4,perm) in (
        (t2,t3,t4,(1,2,3,4)),
        (t2,t4,t3,(1,2,4,3)),
        (t3,t2,t4,(1,3,2,4)),
        (t3,t4,t2,(1,3,4,2)),
        (t4,t2,t3,(1,4,2,3)),
        (t4,t3,t2,(1,4,3,2))
    )
        for i in cand
            r = resolver.rules[i]
            r.proper && continue
            matches(r.p1, t1, type_to_class) || continue # If key does not match do not even bother
            if matches(r.p2, p2, type_to_class) && matches(r.p3, p3, type_to_class) &&
                                                        matches(r.p4, p4, type_to_class)
                if !r.has_wildcard
                    ic[key] = (perm, i)
                    return r.params
                elseif r.specificity > bestspec
                    bestspec, best, bestperm = r.specificity, i, perm
                end
            end
        end
    end

    if best == 0
        ic[key] = :miss
        return nothing
    else
        ic[key] = (bestperm, best)
        return resolver.rules[best].params
    end
end

element_string_to_symbol(el) = (el == "?" ? :X : Symbol(el))

# Version of get for EzXML objects
get_ezxml(collection, key, default) = (haskey(collection, key) ? collection[key] : default)

function check_lj_params(σ, ϵ)
    σ < zero(σ) && error("σ value $σ must be non-negative")
    ϵ < zero(ϵ) && error("ϵ value $ϵ must be non-negative")
end

# Read a required attribute from an XML tag, giving an error naming the tag,
#   the attribute and the file when it is not present
function xml_attr(node::EzXML.Node, key::AbstractString, ff_file)
    if !haskey(node, key)
        found = join(("\"$(a.name)\"" for a in attributes(node)), ", ")
        found = (isempty(found) ? "no attributes" : "attributes $found")
        error("a <$(node.name)> tag in force field file $ff_file is missing the required " *
              "\"$key\" attribute, it has $found")
    end
    return node[key]
end

# Parse a required numeric attribute from an XML tag, giving an error naming the tag,
#   the attribute and the file when it is not present or cannot be parsed
function parse_attr(::Type{T}, node::EzXML.Node, key::AbstractString, ff_file) where T
    str = xml_attr(node, key, ff_file)
    val = tryparse(T, strip(str))
    if isnothing(val)
        error("could not parse the \"$key\" attribute of a <$(node.name)> tag in force " *
              "field file $ff_file as $T, found \"$str\"")
    end
    return val
end

function parse_bool_attr(node::EzXML.Node, key::AbstractString, ff_file)
    str = strip(lowercase(xml_attr(node, key, ff_file)))
    str in ("true" , "1") && return true
    str in ("false", "0") && return false
    error("could not parse the \"$key\" attribute of a <$(node.name)> tag in force " *
          "field file $ff_file as a boolean, found \"$str\"")
end

# Having this as a function allows recursion to support <Include> tags
# Modifies most arguments
function read_ff_xml!(ff_file, ff_param_array, atom_types, atom_type_order, attributes_from_residue,
                      residues, residue_overrides, patches, bond_rule_specs, angle_rule_specs,
                      torsion_rule_specs, cmap_rules, custor_rule_specs, nb_atom_classes,
                      ljforce_atom_classes, nbfix_pairs, urey_rule_specs, units, strictness,
                      T, IC)
    if !isfile(ff_file)
        throw(ArgumentError("force field XML file $ff_file does not exist"))
    end
    ff_xml = parsexml(read(ff_file))
    ff = root(ff_xml)
    if ff.name != "ForceField"
        throw(ArgumentError("file $ff_file does not have a ForceField top level " *
                            "tag, found $(ff.name)"))
    end

    has_lj_force = any(entry -> entry.name == "LennardJonesForce", eachelement(ff))
    has_custom_nb_force = any(entry -> entry.name == "CustomNonbondedForce", eachelement(ff))
    if has_lj_force && has_custom_nb_force
        error("file $ff_file contains both LennardJonesForce and CustomNonbondedForce tags " *
              "which is not supported")
    end

    for entry in eachelement(ff)
        entry_name = entry.name
        if entry_name == "Include"
            xml_fp = joinpath(dirname(ff_file), xml_attr(entry, "file", ff_file))
            read_ff_xml!(xml_fp, ff_param_array, atom_types, atom_type_order,
                         attributes_from_residue, residues, residue_overrides, patches,
                         bond_rule_specs, angle_rule_specs, torsion_rule_specs, cmap_rules,
                         custor_rule_specs, nb_atom_classes, ljforce_atom_classes, nbfix_pairs,
                         urey_rule_specs, units, strictness, T, IC)

        elseif entry_name == "AtomTypes"
            for atom_type in eachelement(entry)
                at_type  = xml_attr(atom_type, "name" , ff_file)
                at_class = xml_attr(atom_type, "class", ff_file)
                element = get_ezxml(atom_type, "element", "?")
                ch = missing # This is set later
                atom_mass = add_units(parse_attr(T, atom_type, "mass", ff_file), u"g/mol", units)
                σ = add_units(T(-1), u"nm", units)
                ϵ = add_units(T(-1), u"kJ * mol^-1", units)
                if haskey(atom_types, at_type)
                    error("atom type $at_type is defined twice in the force field XML file(s)")
                end
                atom_types[at_type] = AtomType{T, typeof(atom_mass), typeof(σ), typeof(ϵ)}(
                    at_type, at_class, element, ch, atom_mass, σ, ϵ, missing, missing)
                push!(atom_type_order, at_type)
            end

        elseif entry_name == "Residues"
            for residue in eachelement(entry)
                rname = xml_attr(residue, "name", ff_file)
                atoms, types = String[], String[]
                charges = Union{T, Missing}[]
                elements = Symbol[]
                virtual_sites = VirtualSiteTemplate{T, IC}[]
                external_bonds_name = String[]
                externals = Int[]
                allowed_patches = String[]
                extras = BitVector()
                bonds_by_name = Tuple{String,String}[]

                for re in eachelement(residue)
                    if re.name == "Atom"
                        at_type = xml_attr(re, "type", ff_file)
                        at_name = xml_attr(re, "name", ff_file)
                        q = (haskey(re, "charge") ? parse_attr(T, re, "charge", ff_file) : missing)
                        if !haskey(atom_types, at_type)
                            error("atom \"$at_name\" in residue template $rname in force " *
                                  "field file $ff_file has type \"$at_type\", which is not " *
                                  "defined in an <AtomTypes> entry read so far. Atom types " *
                                  "have to be defined before the residue templates that " *
                                  "use them, so give the files that define atom types first")
                        end
                        if at_name in atoms
                            error("residue template $rname in force field file $ff_file " *
                                  "contains multiple atoms named \"$at_name\"")
                        end
                        push!(atoms, at_name)
                        push!(types, at_type)
                        push!(charges, q)
                        push!(externals, 0)
                        at = atom_types[at_type]
                        push!(extras, (at.element == "?") || (at.class == "EP"))
                        push!(elements, element_string_to_symbol(at.element))
                    elseif re.name == "Bond"
                        if haskey(re, "atomName1")
                            an1 = re["atomName1"]
                        else
                            # Allow the deprecated "from/to" syntax
                            an1 = atoms[parse_attr(Int, re, "from", ff_file) + 1]
                        end
                        if haskey(re, "atomName2")
                            an2 = re["atomName2"]
                        else
                            # Allow the deprecated "from/to" syntax
                            an2 = atoms[parse_attr(Int, re, "to", ff_file) + 1]
                        end
                        push!(bonds_by_name, (an1, an2))
                    elseif re.name == "ExternalBond"
                        if haskey(re, "atomName")
                            an = re["atomName"]
                        else
                            an = atoms[parse_attr(Int, re, "from", ff_file) + 1]
                        end
                        push!(external_bonds_name, an)
                    elseif re.name == "AllowPatch"
                        push!(allowed_patches, xml_attr(re, "name", ff_file))
                    elseif re.name == "VirtualSite"
                        vs_type = xml_attr(re, "type", ff_file)
                        if haskey(re, "siteName")
                            vs_name = re["siteName"]
                        else
                            # Allow the deprecated "index/atom1/atom2/atom3" syntax
                            vs_name = atoms[parse_attr(Int, re, "index", ff_file) + 1]
                        end
                        if haskey(re, "atomName1")
                            atom_name_1 = re["atomName1"]
                        else
                            atom_name_1 = atoms[parse_attr(Int, re, "atom1", ff_file) + 1]
                        end
                        if haskey(re, "atomName2")
                            atom_name_2 = re["atomName2"]
                        else
                            atom_name_2 = atoms[parse_attr(Int, re, "atom2", ff_file) + 1]
                        end
                        if vs_type == "average2"
                            weight_1 = parse_attr(T, re, "weight1", ff_file)
                            weight_2 = parse_attr(T, re, "weight2", ff_file)
                            vs = VirtualSiteTemplate(2, vs_name, atom_name_1, atom_name_2,
                                    "", weight_1, weight_2, zero(T), zero(T), zero(T), zero(IC))
                            push!(virtual_sites, vs)
                        elseif vs_type == "average3"
                            if haskey(re, "atomName3")
                                atom_name_3 = re["atomName3"]
                            else
                                atom_name_3 = atoms[parse_attr(Int, re, "atom3", ff_file) + 1]
                            end
                            weight_1 = parse_attr(T, re, "weight1", ff_file)
                            weight_2 = parse_attr(T, re, "weight2", ff_file)
                            weight_3 = parse_attr(T, re, "weight3", ff_file)
                            vs = VirtualSiteTemplate(3, vs_name, atom_name_1, atom_name_2,
                                    atom_name_3, weight_1, weight_2, weight_3, zero(T),
                                    zero(T), zero(IC))
                            push!(virtual_sites, vs)
                        elseif vs_type == "outOfPlane"
                            if haskey(re, "atomName3")
                                atom_name_3 = re["atomName3"]
                            else
                                atom_name_3 = atoms[parse_attr(Int, re, "atom3", ff_file) + 1]
                            end
                            weight_12 = parse_attr(T, re, "weight12", ff_file)
                            weight_13 = parse_attr(T, re, "weight13", ff_file)
                            weight_cross = add_units(
                                parse_attr(T, re, "weightCross", ff_file), u"nm^-1", units)
                            vs = VirtualSiteTemplate(4, vs_name, atom_name_1, atom_name_2,
                                    atom_name_3, zero(T), zero(T), zero(T), weight_12,
                                    weight_13, weight_cross)
                            push!(virtual_sites, vs)
                        elseif vs_type == "localCoords"
                            report_issue(
                                "Virtual site type $vs_type not currently supported, ignoring",
                                strictness,
                            )
                        else
                            report_issue(
                                "Unrecognised virtual site type $vs_type, ignoring",
                                strictness,
                            )
                        end
                    end
                end

                vs_atom_names = Set(vs.name for vs in virtual_sites)
                for a1_a2 in bonds_by_name
                    for nm in a1_a2
                        if nm in vs_atom_names
                            error("virtual site $nm in residue $rname appears in a bond")
                        end
                    end
                end
                for nm in external_bonds_name
                    if nm in vs_atom_names
                        error("virtual site $nm in residue $rname appears in an external bond")
                    end
                end

                name_to_idx = Dict(a => i for (i,a) in enumerate(atoms))
                bonds = Tuple{Int, Int}[]
                for (a1, a2) in bonds_by_name
                    for a in (a1, a2)
                        if !haskey(name_to_idx, a)
                            error("a <Bond> tag in residue template $rname in force field " *
                                  "file $ff_file refers to atom \"$a\", which is not one " *
                                  "of the atoms of the template ($(join(atoms, ", ")))")
                        end
                    end
                    i, j = name_to_idx[a1], name_to_idx[a2]
                    push!(bonds, (i < j ? (i, j) : (j, i)))
                end
                for nm in external_bonds_name
                    if haskey(name_to_idx, nm)
                        externals[name_to_idx[nm]] += 1
                    end
                end
                override = (haskey(residue, "override") ?
                            parse_attr(Int, residue, "override", ff_file) : 0)
                if haskey(residues, rname)
                    existing_override = residue_overrides[rname]
                    if override < existing_override
                        continue # The existing template takes precedence
                    elseif override == existing_override
                        error("residue template $rname with the same override level " *
                              "$override is defined twice in the force field XML file(s), " *
                              "the second definition is in $ff_file")
                    end
                end
                residue_overrides[rname] = override
                residues[rname] = ResidueTemplate(rname, atoms, elements, types, virtual_sites,
                                        bonds, externals, allowed_patches, charges, extras)
            end

        elseif entry_name == "Patches"
            for patch in eachelement(entry)
                pname = xml_attr(patch, "name", ff_file)
                if haskey(patch, "residues") && patch["residues"] != "1"
                    err_str = "Residue patches altering multiple templates not currently " *
                              "supported, ignoring patch $pname"
                    report_issue(err_str, strictness)
                    continue
                end

                add_atoms = Tuple{String, String, Any}[]
                change_atoms = Tuple{String, String, Any}[]
                remove_atoms = String[]
                add_bonds = Tuple{String, String}[]
                remove_bonds = Tuple{String, String}[]
                add_external_bonds = String[]
                remove_external_bonds = String[]
                apply_to_residues = String[]

                for pa in eachelement(patch)
                    if pa.name == "AddAtom"
                        q = (haskey(pa, "charge") ?
                             parse_attr(T, pa, "charge", ff_file) : missing)
                        push!(add_atoms, (xml_attr(pa, "name", ff_file),
                                          xml_attr(pa, "type", ff_file), q))
                    elseif pa.name == "ChangeAtom"
                        q = (haskey(pa, "charge") ?
                             parse_attr(T, pa, "charge", ff_file) : missing)
                        push!(change_atoms, (xml_attr(pa, "name", ff_file),
                                             xml_attr(pa, "type", ff_file), q))
                    elseif pa.name == "RemoveAtom"
                        push!(remove_atoms, xml_attr(pa, "name", ff_file))
                    elseif pa.name == "AddBond"
                        push!(add_bonds, (xml_attr(pa, "atomName1", ff_file),
                                          xml_attr(pa, "atomName2", ff_file)))
                    elseif pa.name == "RemoveBond"
                        push!(remove_bonds, (xml_attr(pa, "atomName1", ff_file),
                                             xml_attr(pa, "atomName2", ff_file)))
                    elseif pa.name == "AddExternalBond"
                        push!(add_external_bonds, xml_attr(pa, "atomName", ff_file))
                    elseif pa.name == "RemoveExternalBond"
                        push!(remove_external_bonds, xml_attr(pa, "atomName", ff_file))
                    elseif pa.name == "ApplyToResidue"
                        push!(apply_to_residues, xml_attr(pa, "name", ff_file))
                    end
                end
                patches[pname] = ResiduePatchTemplate(pname, add_atoms, change_atoms,
                                    remove_atoms, add_bonds, remove_bonds, add_external_bonds,
                                    remove_external_bonds, apply_to_residues)
            end

        elseif entry_name == "HarmonicBondForce"
            for bond in eachelement(entry)
                k = add_units(parse_attr(T, bond, "k", ff_file),
                              u"kJ * mol^-1 * nm^-2", units)
                r0 = add_units(parse_attr(T, bond, "length", ff_file), u"nm", units)
                p1 = pattern_from_attrs(bond, "type1", "class1")
                p2 = pattern_from_attrs(bond, "type2", "class2")
                push!(bond_rule_specs, (:bond_rule, p1, p2, HarmonicBond(k,r0)))
            end

        elseif entry_name == "HarmonicAngleForce"
            for ang in eachelement(entry)
                k = add_units(parse_attr(T, ang, "k", ff_file), u"kJ * mol^-1", units)
                θ0 = parse_attr(T, ang, "angle", ff_file)
                p1 = pattern_from_attrs(ang, "type1", "class1")
                p2 = pattern_from_attrs(ang, "type2", "class2")
                p3 = pattern_from_attrs(ang, "type3", "class3")
                push!(angle_rule_specs, (:angle_rule, p1, p2, p3, HarmonicAngle(k, θ0)))
            end

        elseif entry_name == "PeriodicTorsionForce"
            ff_param_array[1] = get_ezxml(entry, "ordering", ff_param_array[1])
            local_ordering    = get_ezxml(entry, "ordering", "default")
            for torsion in eachelement(entry)
                proper = torsion.name == "Proper"
                periodicities = Int[]
                phases = T[]
                ks = (units ? typeof(T(1u"kJ * mol^-1"))[] : T[])
                i = 1
                while haskey(torsion, "periodicity$i")
                    push!(periodicities, parse_attr(Int, torsion, "periodicity$i", ff_file))
                    push!(phases, parse_attr(T, torsion, "phase$i", ff_file))
                    push!(ks, add_units(parse_attr(T, torsion, "k$i", ff_file),
                                        u"kJ * mol^-1", units))
                    i += 1
                end

                p1 = pattern_from_attrs(torsion, "type1", "class1")
                p2 = pattern_from_attrs(torsion, "type2", "class2")
                p3 = pattern_from_attrs(torsion, "type3", "class3")
                p4 = pattern_from_attrs(torsion, "type4", "class4")

                has_wildcard = (p1.kind == WILD || p2.kind == WILD || p3.kind == WILD || p4.kind == WILD)
                spec = UInt8(spec_score(p1) + spec_score(p2) + spec_score(p3) + spec_score(p4))
                params_any = (:params, periodicities, phases, ks, proper)
                push!(torsion_rule_specs, (:torsion_rule, p1, p2, p3, p4, spec, params_any,
                                           local_ordering, has_wildcard))
            end

        elseif entry_name == "CMAPTorsionForce"
            maps = []
            for cmap in eachelement(entry)
                if cmap.name == "Map"
                    tmp_map = add_units(parse.(T, split(cmap.content)), u"kJ * mol^-1", units)
                    push!(maps, tmp_map)
                elseif cmap.name == "Torsion"
                    map_n = parse_attr(Int, cmap, "map", ff_file) + 1 # Zero-indexed
                    p1 = pattern_from_attrs(cmap, "type1", "class1")
                    p2 = pattern_from_attrs(cmap, "type2", "class2")
                    p3 = pattern_from_attrs(cmap, "type3", "class3")
                    p4 = pattern_from_attrs(cmap, "type4", "class4")
                    p5 = pattern_from_attrs(cmap, "type5", "class5")

                    has_wildcard = (p1.kind==WILD || p2.kind==WILD || p3.kind==WILD ||
                                    p4.kind==WILD || p5.kind==WILD)
                    cmap_tt = CMAPTorsionType(Int(sqrt(length(maps[map_n]))), maps[map_n])
                    cmap_spec = UInt8(spec_score(p1) + spec_score(p2) + spec_score(p3) +
                                      spec_score(p4) + spec_score(p5))
                    push!(
                        cmap_rules,
                        CMAPRule(p1, p2, p3, p4, p5, has_wildcard, cmap_tt, cmap_spec),
                    )
                end
            end

        elseif entry_name == "CustomTorsionForce"
            if xml_attr(entry, "energy", ff_file) != "k*(theta-theta0)^2"
                err_str = "CustomTorsionForce without energy=\"k*(theta-theta0)^2\" not " *
                          "currently supported, ignoring"
                report_issue(err_str, strictness)
                continue
            end
            for torsion in eachelement(entry)
                # Assume PerTorsionParameter entries are k and theta0
                if torsion.name == "Improper"
                    k = add_units(parse_attr(T, torsion, "k", ff_file), u"kJ * mol^-1", units)
                    θ0 = parse_attr(T, torsion, "theta0", ff_file)

                    p1 = pattern_from_attrs(torsion, "type1", "class1")
                    p2 = pattern_from_attrs(torsion, "type2", "class2")
                    p3 = pattern_from_attrs(torsion, "type3", "class3")
                    p4 = pattern_from_attrs(torsion, "type4", "class4")

                    has_wildcard = (p1.kind==WILD || p2.kind==WILD || p3.kind==WILD || p4.kind==WILD)
                    spec = UInt8(spec_score(p1)+spec_score(p2)+spec_score(p3)+spec_score(p4))
                    params_any = (:params, k, θ0)
                    push!(
                        custor_rule_specs,
                        (:custom_rule, p1, p2, p3, p4, spec, params_any, has_wildcard),
                    )
                elseif torsion.name == "Proper"
                    err_str = "CustomTorsionForce with Proper entries not " *
                              "currently supported, ignoring"
                    report_issue(err_str, strictness)
                    continue
                end
            end

        elseif entry_name == "NonbondedForce"
            if haskey(entry, "useDispersionCorrection")
                dispersion_correction = parse_bool_attr(entry, "useDispersionCorrection",
                                                        ff_file)
                if !isnothing(ff_param_array[12]) && dispersion_correction != ff_param_array[12]
                    error("multiple NonbondedForce/LennardJonesForce entries with " *
                          "different useDispersionCorrection")
                end
                ff_param_array[12] = dispersion_correction
            end
            if haskey(entry, "coulomb14scale")
                w = parse_attr(T, entry, "coulomb14scale", ff_file)
                if ff_param_array[3] && w != ff_param_array[2]
                    error("multiple NonbondedForce entries with different coulomb14scale")
                end
                ff_param_array[2] = w
                ff_param_array[3] = true
            end
            if haskey(entry, "lj14scale")
                w = parse_attr(T, entry, "lj14scale", ff_file)
                if ff_param_array[5] && w != ff_param_array[4]
                    error("multiple NonbondedForce entries with different lj14scale")
                end
                ff_param_array[4] = w
                ff_param_array[5] = true
            end
            for atom_or_attr in eachelement(entry)
                if atom_or_attr.name == "Atom"
                    ch = (haskey(atom_or_attr, "charge") ?
                          parse_attr(T, atom_or_attr, "charge", ff_file) : missing)
                    σ = add_units(parse_attr(T, atom_or_attr, "sigma", ff_file), u"nm", units)
                    ϵ = add_units(parse_attr(T, atom_or_attr, "epsilon", ff_file),
                                  u"kJ * mol^-1", units)
                    check_lj_params(σ, ϵ)
                    if haskey(atom_or_attr, "class")
                        push!(nb_atom_classes, AtomType{T, T, typeof(σ), typeof(ϵ)}(
                                "", xml_attr(atom_or_attr, "class", ff_file), "", ch,
                                zero(T), σ, ϵ,
                                missing, missing))
                    else
                        atom_type = xml_attr(atom_or_attr, "type", ff_file)
                        if haskey(atom_types, atom_type)
                            at = atom_types[atom_type]
                            atom_types[atom_type] = AtomType{T, typeof(at.mass), typeof(σ), typeof(ϵ)}(
                                at.type, at.class, at.element, ch, at.mass, σ, ϵ, missing, missing)
                        end
                    end
                elseif atom_or_attr.name == "UseAttributeFromResidue"
                    use_attr = xml_attr(atom_or_attr, "name", ff_file)
                    if !(use_attr in attributes_from_residue)
                        push!(attributes_from_residue, use_attr)
                    end
                    if use_attr != "charge"
                        err_str = "UseAttributeFromResidue only supported for charge, " *
                                    "ignoring $use_attr"
                        report_issue(err_str, strictness)
                    end
                end
            end

        elseif entry_name == "LennardJonesForce"
            if haskey(entry, "useDispersionCorrection")
                dispersion_correction = parse_bool_attr(entry, "useDispersionCorrection", ff_file)
                if !isnothing(ff_param_array[12]) && dispersion_correction != ff_param_array[12]
                    error("multiple NonbondedForce/LennardJonesForce entries with " *
                          "different useDispersionCorrection")
                end
                ff_param_array[12] = dispersion_correction
            end
            if haskey(entry, "lj14scale")
                w = parse_attr(T, entry, "lj14scale", ff_file)
                if ff_param_array[7] && w != ff_param_array[6]
                    error("multiple LennardJonesForce entries with different lj14scale")
                end
                ff_param_array[6] = w
                ff_param_array[7] = true
            end
            for atom_or_nbfix in eachelement(entry)
                if atom_or_nbfix.name == "Atom"
                    if haskey(atom_or_nbfix, "sigma14")
                        σ14 = add_units(parse_attr(T, atom_or_nbfix, "sigma14", ff_file),
                                        u"nm", units)
                    else
                        σ14 = missing
                    end
                    if haskey(atom_or_nbfix, "epsilon14")
                        ϵ14 = add_units(parse_attr(T, atom_or_nbfix, "epsilon14", ff_file),
                                        u"kJ * mol^-1", units)
                    else
                        ϵ14 = missing
                    end
                    σ = add_units(parse_attr(T, atom_or_nbfix, "sigma", ff_file), u"nm", units)
                    ϵ = add_units(parse_attr(T, atom_or_nbfix, "epsilon", ff_file),
                                  u"kJ * mol^-1", units)
                    check_lj_params(σ, ϵ)
                    if haskey(atom_or_nbfix, "class")
                        push!(ljforce_atom_classes, AtomType{T, T, typeof(σ), typeof(ϵ)}(
                                "", xml_attr(atom_or_nbfix, "class", ff_file), "",
                                zero(T), zero(T),
                                σ, ϵ, σ14, ϵ14))
                    else
                        atom_type = xml_attr(atom_or_nbfix, "type", ff_file)
                        if haskey(atom_types, atom_type)
                            at = atom_types[atom_type]
                            # Re-use charge from NonbondedForce entry if present
                            atom_types[atom_type] = AtomType{T, typeof(at.mass), typeof(σ), typeof(ϵ)}(
                                at.type, at.class, at.element, at.charge, at.mass, σ, ϵ, σ14, ϵ14)
                        end
                    end
                elseif atom_or_nbfix.name == "NBFixPair"
                    if haskey(atom_or_nbfix, "type1")
                        type1 = xml_attr(atom_or_nbfix, "type1", ff_file)
                        type2 = xml_attr(atom_or_nbfix, "type2", ff_file)
                        class1, class2 = "", ""
                    else
                        type1, type2 = "", ""
                        class1 = xml_attr(atom_or_nbfix, "class1", ff_file)
                        class2 = xml_attr(atom_or_nbfix, "class2", ff_file)
                    end
                    σ = add_units(parse_attr(T, atom_or_nbfix, "sigma", ff_file), u"nm", units)
                    ϵ = add_units(parse_attr(T, atom_or_nbfix, "epsilon", ff_file),
                                  u"kJ * mol^-1", units)
                    check_lj_params(σ, ϵ)
                    push!(nbfix_pairs, NBFixPair(type1, type2, class1, class2, σ, ϵ))
                end
            end

        elseif entry_name == "CustomNonbondedForce"
            dexp_definition = "sqrt(epsilon1*epsilon2)*(((beta*exp(alpha))/(alpha-beta))*exp(-alpha*(r/((2^(1/6))*((sigma1+sigma2)/2))))-((alpha*exp(beta))/(alpha-beta))*exp(-beta*(r/((2^(1/6))*((sigma1+sigma2)/2)))))"
            if get_ezxml(entry, "energy", "") == dexp_definition &&
                            get_ezxml(entry, "bondCutoff", "") == "3"
                ff_param_array[13] = true
                for element in eachelement(entry)
                    if element.name == "GlobalParameter"
                        if xml_attr(element, "name", ff_file) == "alpha"
                            ff_param_array[9] && error("Multiple alpha values for double exponential alpha")
                            ff_param_array[8] = parse_attr(T, element, "defaultValue", ff_file)
                            ff_param_array[9] = true
                        elseif xml_attr(element, "name", ff_file) == "beta"
                            ff_param_array[11] && error("Multiple alpha values for double exponential beta")
                            ff_param_array[10] = parse_attr(T, element, "defaultValue", ff_file)
                            ff_param_array[11] = true
                        else
                            err_str = "CustomNonbondedForce with global parameters other than " *
                                      "\"alpha\" and \"beta\" not supported, ignoring parameter"
                            report_issue(err_str, strictness)
                        end
                    elseif element.name == "Atom"
                        σ = add_units(parse_attr(T, element, "sigma", ff_file), u"nm", units)
                        ϵ = add_units(parse_attr(T, element, "epsilon", ff_file),
                                      u"kJ * mol^-1", units)
                        check_lj_params(σ, ϵ)
                        if haskey(element, "class")
                            # This array can be used since CustomNonbondedForce and
                            #   LennardJonesForce cannot both be present
                            push!(ljforce_atom_classes, AtomType{T, T, typeof(σ), typeof(ϵ)}(
                                    "", xml_attr(element, "class", ff_file), "",
                                    zero(T), zero(T), σ, ϵ,
                                    missing, missing))
                        else
                            atom_type = xml_attr(element, "type", ff_file)
                            if haskey(atom_types, atom_type)
                                at = atom_types[atom_type]
                                # Re-use charge from NonbondedForce entry if present
                                atom_types[atom_type] = AtomType{T, typeof(at.mass), typeof(σ), typeof(ϵ)}(
                                    at.type, at.class, at.element, at.charge, at.mass, σ, ϵ, missing, missing)
                            end
                        end
                    end
                end
            else
                err_str = "CustomNonbondedForce without energy=\"$dexp_definition\" " *
                          "and bondCutoff=\"3\" not supported, ignoring"
                report_issue(err_str, strictness)
            end

        elseif entry_name == "AmoebaUreyBradleyForce"
            for ang in eachelement(entry)
                k = add_units(2 * parse_attr(T, ang, "k", ff_file), u"kJ * mol^-1 * nm^-2", units)
                r0 = add_units(parse_attr(T, ang, "d", ff_file), u"nm", units)
                p1 = pattern_from_attrs(ang, "type1", "class1")
                p2 = pattern_from_attrs(ang, "type2", "class2")
                p3 = pattern_from_attrs(ang, "type3", "class3")
                push!(urey_rule_specs, (:urey_rule, p1, p2, p3, HarmonicBond(k, r0)))
            end

        elseif entry_name in (
                    "RBTorsionForce", "GBSAOBCForce", "CustomBondForce", "CustomAngleForce",
                    "CustomGBForce", "CustomHbondForce",
                    "CustomManyParticleForce", "DrudeForce", "HippoNonbondedForce",
                    "AmoebaBondForce", "AmoebaAngleForce", "AmoebaOutOfPlaneBendForce",
                    "AmoebaTorsionForce", "AmoebaPiTorsionForce", "AmoebaStretchTorsionForce",
                    "AmoebaAngleTorsionForce", "AmoebaTorsionTorsionForce",
                    "AmoebaStretchBendForce", "AmoebaVdwForce", "AmoebaMultipoleForce", 
                    "AmoebaWcaDispersionForce", "AmoebaGeneralizedKirkwoodForce",
                )
            report_issue("$entry_name not currently supported, ignoring", strictness)

        elseif entry_name != "Info" # Info contains metadata
            report_issue("Ignoring unknown XML entry $entry_name", strictness)
        end
    end
end

"""
    MolecularForceField(ff_files...; units=true, custom_residue_templates=nothing,
                        custom_renaming_scheme=nothing, float_type=Float64,
                        strictness=:warn)

A molecular force field.

Read one or more OpenMM force field XML files by passing them to the constructor.
See the [OpenMM documentation](https://docs.openmm.org/latest/userguide/application/06_creating_ffs.html)
for how these files are formatted.

In order to assign force field parameters to the atoms in the simulation, the
residues determined from a structure file are matched to templates provided by
the force field file, as well as a template dictionary in XML format, which defines
the standard topology (bonds) of the residues to be found in the simulation.
At the moment, Molly provides a dictionary for all standard amino acids, nucleic acids and
water for this purpose.
If the system to be simulated contains other molecules, their template topologies must be
defined either through `CONECT` records in the PDB file or by providing an extra
custom template file to the `custom_residue_templates` keyword argument.

`float_type` should generally be `Float64` since the float type of a [`System`](@ref)
is determined later when creating the [`System`](@ref).
Behavior with unsupported files is determined by the `strictness` keyword argument.
This can be `:warn` to emit warnings, `:nowarn` to suppress warnings or `:error` to error.
"""
struct MolecularForceField{T, G, NB, M, D, DA, E, K, KA, C}
    atom_types::Dict{String, AtomType{T, M, D, E}}
    atom_type_order::Vector{String}
    residues::Dict{String, ResidueTemplate{T, C}}
    torsion_order::String
    weight_14_coulomb::T
    weight_14_lj::T
    global_params::G
    dispersion_correction::Bool
    custom_nonbonded::Bool
    nbfix_pairs::NB
    attributes_from_residue::Vector{String}
    residue_name_replacements::Dict{String,String}
    atom_name_replacements::Dict{String, Dict{String, String}}
    standard_bonds::Dict{String, Vector{Tuple{String, String}}}
    type_to_class::Dict{String, String}
    class_to_types::Dict{String, Vector{String}}
    bond_resolver::BondResolver{K, D}
    angle_resolver::AngleResolver{KA, DA, K, D}
    torsion_resolver::TorsionResolver{T, E}
    cmap_resolver::CMAPResolver{E}
    units::Bool
end

function MolecularForceField(ff_files::AbstractString...; units::Bool=true,
                             custom_residue_templates=nothing, custom_renaming_scheme=nothing,
                             float_type=Float64, strictness=default_strictness())
    check_strictness(strictness)
    T = float_type
    if units
        M  = typeof(T(1u"g/mol"))
        D  = typeof(T(1u"nm"))
        DA = typeof(T(1))
        E  = typeof(T(1u"kJ * mol^-1"))
        K  = typeof(T(1u"kJ * mol^-1 * nm^-2"))
        KA = typeof(T(1u"kJ * mol^-1"))
        IC = typeof(T(1u"nm^-1"))
    else
        M, D, DA, E, K, KA, IC = T, T, T, T, T, T, T
    end
    atom_types = Dict{String, AtomType}()

    # Array to allow mutation in read_ff_xml!
    # Torsion order, 
    # weight_14_coulomb, weight_14_coulomb_set, weight_14_lj, weight_14_lj_set,
    # weight_14_lj_ljforce, weight_14_lj_ljforce_set, 
    # double_exp_alpha, dexp_alpha_set, double_exp_beta, dexp_beta_set, 
    # dispersion_correction, custom_nonbonded_set
    ff_param_array = ["",
                      one(T), false, one(T), false, 
                      one(T), false, 
                      zero(T), false, zero(T), false,
                      nothing, false]
    attributes_from_residue = String[]
    residues = Dict{String, ResidueTemplate}()
    # Residue templates can be replaced by ones with a higher override level
    residue_overrides = Dict{String, Int}()
    patches = Dict{String, ResiduePatchTemplate}()

    atom_type_order = String[]
    bond_rule_specs    = [] # Accumulators for pattern rules
    angle_rule_specs   = []
    urey_rule_specs    = []
    torsion_rule_specs = []
    custor_rule_specs  = []
    cmap_rules = CMAPRule{E}[]
    nb_atom_classes, ljforce_atom_classes = AtomType[], AtomType[]
    nbfix_pairs = NBFixPair[]

    for ff_file in ff_files
        read_ff_xml!(ff_file, ff_param_array, atom_types, atom_type_order, attributes_from_residue,
                     residues, residue_overrides, patches, bond_rule_specs, angle_rule_specs,
                     torsion_rule_specs, cmap_rules, custor_rule_specs, nb_atom_classes,
                     ljforce_atom_classes, nbfix_pairs, urey_rule_specs, units, strictness,
                     T, IC)
    end
    torsion_order = ff_param_array[1]
    weight_14_coulomb = ff_param_array[2]
    # Use LennardJonesForce 1-4 weighting if present
    weight_14_lj = (ff_param_array[7] ? ff_param_array[6] : ff_param_array[4])
    dispersion_correction = if isnothing(ff_param_array[12])
        !ff_param_array[13]
    else
        ff_param_array[12]
    end

    double_exp_alpha = (ff_param_array[9] ? ff_param_array[8] : zero(T))
    double_exp_beta  = (ff_param_array[11] ? ff_param_array[10] : zero(T))

    global_params = [double_exp_alpha, double_exp_beta]
    G = typeof(global_params)
    if ff_param_array[13] && count(at -> at.ϵ > zero(at.ϵ), nb_atom_classes) > 0
        error("if CustomNonbondedForce is used, all atoms must have a NonbondedForce " *
              "ϵ of zero since the Lennard-Jones potential is not used")
    end

    # Apply residue patches
    for res_name in collect(keys(residues)) # Collect required since residues changes
        patches_to_apply = copy(residues[res_name].allowed_patches)
        for patch_name in keys(patches)
            for rn in patches[patch_name].apply_to_residues
                if rn == res_name
                    push!(patches_to_apply, patch_name)
                    break
                end
            end
        end
        patches_to_apply = collect(Set(patches_to_apply))

        for patch_name in patches_to_apply
            patch_res_name = ""
            suffix = 0
            free_name_found = false
            while !free_name_found
                suffix_str = (iszero(suffix) ? "" : "_$suffix")
                patch_res_name = "$(res_name)_$patch_name$suffix_str"
                if !haskey(residues, patch_res_name)
                    free_name_found = true
                end
                suffix += 1
            end

            patched_res = apply_residue_patch(residues[res_name], patches[patch_name],
                                    patch_res_name, res_name, patch_name, atom_types, strictness)
            if !isnothing(patched_res) # Invalid patches warn and return nothing
                residues[patch_res_name] = patched_res
            end
        end
    end

    resname_replacements, atomname_replacements = load_replacements()
    standard_bonds = load_bond_definitions()

    if !isnothing(custom_renaming_scheme)
        resname_replacements, atomname_replacements = load_replacements(
            xmlpath=custom_renaming_scheme,
            resname_replacements=resname_replacements,
            atomname_replacements=atomname_replacements,
        )
    end
    if !isnothing(custom_residue_templates)
        standard_bonds = load_bond_definitions(
            xmlpath=custom_residue_templates,
            standardBonds=standard_bonds,
        )
    end

    nbfix_pairs_conc = [nbfix_pairs...]
    NB = typeof(nbfix_pairs_conc)

    # Build class maps once
    type_to_class = Dict{String, String}(t => atom_types[t].class for t in keys(atom_types))
    class_to_types = Dict{String, Vector{String}}()
    for (t, at) in atom_types
        push!(get!(class_to_types, at.class, String[]), t)
    end

    # Assign parameters to atom types from classes
    for ac in nb_atom_classes
        if haskey(class_to_types, ac.class)
            for t in class_to_types[ac.class]
                at = atom_types[t]
                atom_types[t] = AtomType{T, typeof(at.mass), typeof(ac.σ), typeof(ac.ϵ)}(
                    at.type, at.class, at.element, ac.charge, at.mass, ac.σ, ac.ϵ, ac.σ14, ac.ϵ14)
            end
        end
    end

    for ac in ljforce_atom_classes
        if haskey(class_to_types, ac.class)
            for t in class_to_types[ac.class]
                at = atom_types[t]
                # Re-use charge from NonbondedForce entry if present
                atom_types[t] = AtomType{T, typeof(at.mass), typeof(ac.σ), typeof(ac.ϵ)}(
                    at.type, at.class, at.element, at.charge, at.mass, ac.σ, ac.ϵ, ac.σ14, ac.ϵ14)
            end
        end
    end

    at_missing_params = sort(collect(filter(t -> atom_types[t].σ < zero(atom_types[t].σ),
                                            keys(atom_types))))
    if length(at_missing_params) > 0
        n_missing = length(at_missing_params)
        shown = join(at_missing_params[1:min(n_missing, 20)], ", ")
        n_missing > 20 && (shown *= " and $(n_missing - 20) more")
        error("$n_missing atom types have not had σ and ϵ set in a NonbondedForce, " *
              "LennardJonesForce or CustomNonbondedForce entry: $shown. Every atom type " *
              "in the force field files needs non-bonded parameters, so check that all " *
              "the required XML files are given and that the atom types and classes in " *
              "them match")
    end

    # Bonds resolver
    bond_rules = BondRule{K,D}[]
    bidx = Dict{Tuple{Symbol, String, String}, Vector{Int}}()
    for spec in bond_rule_specs
        _, p1, p2, hb = spec
        push!(bond_rules, BondRule{K, D}(p1, p2, hb, UInt8(spec_score(p1) + spec_score(p2))))
        i = length(bond_rules)
        # Index both orientations
        for (a, b) in ((p1, p2), (p2, p1))
            if a.kind == TYPE && b.kind == TYPE
                push!(get!(bidx, (:type,  a.val, b.val), Int[]), i)
            elseif a.kind == CLASS && b.kind == CLASS
                push!(get!(bidx, (:class, a.val, b.val), Int[]), i)
            else
                push!(get!(bidx, (:wild,  "", ""), Int[]), i)
            end
        end
    end
    bond_resolver = BondResolver{K, D}(
        bond_rules,
        bidx,
        Dict{Tuple{String, String}, Union{HarmonicBond{K, D}, Nothing}}(),
    )

    # Angles resolver
    angle_rules = Union{AngleRule{KA, DA}, UreyBradleyRule{K, D}}[]
    aidx = Dict{Tuple{Symbol, String}, Vector{Int}}()
    for spec in angle_rule_specs
        _, p1, p2, p3, ha = spec
        sscore = UInt8(spec_score(p1) + spec_score(p2) + spec_score(p3))
        push!(angle_rules, AngleRule{KA, DA}(p1, p2, p3, ha, sscore))
        i = length(angle_rules)
        # Central indexing, use p2 as key
        if p2.kind == TYPE
            push!(get!(aidx, (:type,  p2.val), Int[]), i)
        elseif p2.kind == CLASS
            push!(get!(aidx, (:class, p2.val), Int[]), i)
        else
            push!(get!(aidx, (:wild, ""), Int[]), i)
        end
    end

    for spec in urey_rule_specs
        _, p1, p2, p3, hb = spec
        sscore = UInt8(spec_score(p1) + spec_score(p2) + spec_score(p3))
        push!(angle_rules, UreyBradleyRule{K, D}(p1, p2, p3, hb, sscore))
        i = length(angle_rules)
        # Central indexing: use p2 as key
        if p2.kind==TYPE
            push!(get!(aidx, (:type,  p2.val), Int[]), i)
        elseif p2.kind==CLASS
            push!(get!(aidx, (:class, p2.val), Int[]), i)
        else
            push!(get!(aidx, (:wild,  ""), Int[]), i)
        end
    end

    angle_resolver = AngleResolver{KA, DA, K, D}(
        angle_rules,
        aidx,
        Dict{NTuple{3, String}, Union{HarmonicAngle{KA, DA}, Nothing}}(),
        Dict{NTuple{3, String}, Union{HarmonicBond{K, D}, Nothing}}(),
    )

    # Torsions resolver
    torsion_rules = Union{TorsionRule{T, E}, HarmonicTorsionRule{T, E}}[]

    # Candidate lists
    propers_by_type2   = Dict{String, Vector{Int}}()
    propers_by_class2  = Dict{String, Vector{Int}}()
    impropers_by_type1 = Dict{String, Vector{Int}}()
    impropers_by_class1= Dict{String, Vector{Int}}()
    wild_propers   = Int[]
    wild_impropers = Int[]

    for (idx_spec, item) in enumerate(torsion_rule_specs)
        _, p1, p2, p3, p4, sscore, params_any, ordering, wildcard = item
        _, periodicities, phases, ks, proper = params_any
        params = PeriodicTorsionType{T, E}(periodicities, phases, ks, proper)

        push!(
            torsion_rules,
            TorsionRule{T, E}(p1, p2, p3, p4, proper, ordering, wildcard, params, sscore),
        )
        ridx = length(torsion_rules)

        # OpenMM-style candidate lists
        if proper
            if p2.kind == TYPE
                push!(get!(propers_by_type2, p2.val, Int[]), ridx)
            elseif p2.kind == CLASS
                push!(get!(propers_by_class2, p2.val, Int[]), ridx)
            else
                push!(wild_propers, ridx)
            end
        else
            if p1.kind == TYPE
                push!(get!(impropers_by_type1, p1.val, Int[]), ridx)
            elseif p1.kind == CLASS
                push!(get!(impropers_by_class1, p1.val, Int[]), ridx)
            else
                push!(wild_impropers, ridx)
            end
        end
    end

    for (idx_spec, item) in enumerate(custor_rule_specs)
        _, p1, p2, p3, p4, sscore, params_any, wildcard = item
        _, k, θ0 = params_any
        params = HarmonicTorsionType{T, E}(k, θ0)

        push!(
            torsion_rules,
            HarmonicTorsionRule{T, E}(p1, p2, p3, p4, false, wildcard, params, sscore),
        )
        ridx = length(torsion_rules)

        # OpenMM-style candidate lists
        if p1.kind==TYPE
            push!(get!(impropers_by_type1,  p1.val, Int[]), ridx)
        elseif p1.kind==CLASS
            push!(get!(impropers_by_class1, p1.val, Int[]), ridx)
        else
            push!(wild_impropers, ridx)
        end
    end

    torsion_resolver = TorsionResolver{T, E}(
        torsion_rules,
        impropers_by_type1,
        impropers_by_class1,
        propers_by_type2,
        propers_by_class2,
        wild_impropers,
        wild_propers,
        Dict{Tuple{NTuple{4, String}, NTuple{4, String}} ,Any}(), # Proper cache
        Dict{NTuple{4, String}, Any}(),                           # Improper cache
    )

    cmap_resolver = CMAPResolver{E}(
        cmap_rules,
        Dict{Tuple{String,String,String,String,String}, CMAPTorsionType{E}}(),
    )

    return MolecularForceField{T, G, NB, M, D, DA, E, K, KA, IC}(
        atom_types, atom_type_order, residues, torsion_order, weight_14_coulomb, weight_14_lj,
        global_params, dispersion_correction, ff_param_array[13], nbfix_pairs_conc,
        attributes_from_residue, resname_replacements, atomname_replacements, standard_bonds,
        type_to_class, class_to_types, bond_resolver, angle_resolver, torsion_resolver,
        cmap_resolver, units,
    )
end

function Base.show(io::IO, ff::MolecularForceField)
    print(io, "MolecularForceField with ", length(ff.atom_types), " atom types and ",
            length(ff.residues), " residue templates")
end
