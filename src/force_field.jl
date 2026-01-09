# Read in a force field

export
    MolecularForceField

@enum SpecKind::UInt8 WILD=0 TYPE=1 CLASS=2

struct AtomPattern
    kind::SpecKind
    val::String
end

function matches(p::AtomPattern, t::String, class_of::Dict{String, String})
    if p.kind === WILD
        return true
    elseif p.kind === TYPE
        return t === p.val
    else # CLASS
        # Guard missing entries
        cls = get(class_of, t, "")
        return cls === p.val
    end
end

spec_score(ap::AtomPattern) = (ap.kind==TYPE ? 2 : ap.kind==CLASS ? 1 : 0)

function pattern_from_attrs(n::EzXML.Node, typekey::AbstractString, classkey::AbstractString)
    if haskey(n, typekey)
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
end

struct PeriodicTorsionType{T, E}
    periodicities::Vector{Int}
    phases::Vector{T}
    ks::Vector{E}
    proper::Bool
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

struct AngleResolver{K,D}
    rules::Vector{AngleRule{K, D}}
    # Index by center atom
    idx::Dict{Tuple{Symbol, String}, Vector{Int}} # (:type|:class|:wild, key)
    cache::Dict{NTuple{3, String}, Union{HarmonicAngle{K, D}, Nothing}}
end

# Torsions
struct TorsionRule{T,E}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    p4::AtomPattern
    proper::Bool
    ordering::String   # "default" | "charmm" | "amber" | "smirnoff"
    has_wildcard::Bool # any position is WILD
    params::PeriodicTorsionType{T,E}
    specificity::UInt8 # TYPE=2, CLASS=1, WILD=0, used to bias towards specific definitions
end

struct TorsionResolver{T,E}
    rules::Vector{TorsionRule{T, E}}

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

# Proper torsions: lookup with cache
function find_proper_match(t1::AbstractString, t2::AbstractString, t3::AbstractString,
                           t4::AbstractString; resolver::TorsionResolver{T,E},
                           class_of::Dict{String, String}) where {T, E}
    # Unordered signature cache
    sig = ((t1, t2, t3, t4), (t4, t3, t2, t1))
    pc = resolver.proper_cache
    if haskey(pc, sig)
        v = pc[sig]
        if v === :miss
            return (nothing, nothing)
        else
            return (resolver.rules[v::Int].params, resolver.rules[v::Int].specificity)
        end
    end

    # Candidates by central atom 2 (type → class → wild)
    cand = Int[]
    c2 = class_of[t2]
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
            if matches(r.p1, q1, class_of) && matches(r.p2, q2, class_of) &&
               matches(r.p3, q3, class_of) && matches(r.p4, q4, class_of)
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
                             class_of::Dict{String, String}) where {T, E}
    key = (t1, t2, t3, t4)
    ic = resolver.improper_cache
    if haskey(ic, key)
        v = ic[key]
        if v === :miss
            return nothing
        else
            return resolver.rules[(v::Tuple{NTuple{4, Int}, Int})[2]].params
        end
    end

    # Candidates by central atom 1 (type → class → wild)
    cand = Int[]
    c1 = class_of[t1]
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
            matches(r.p1, t1, class_of) || continue # If key does not match do not even bother
            if matches(r.p2, p2, class_of) && matches(r.p3, p3, class_of) && matches(r.p4, p4, class_of)
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

"""
    MolecularForceField(ff_files...; units=true, custom_residue_templates=nothing,
                        custom_renaming_scheme=nothing)
    MolecularForceField(T, ff_files...; units=true, custom_residue_templates=nothing,
                        custom_renaming_scheme=nothing)
    MolecularForceField(atom_types, residue_types, bond_types, angle_types,
                        torsion_types, torsion_order, weight_14_coulomb,
                        weight_14_lj, attributes_from_residue,
                        residue_name_replacements, atom_name_replacements,
                        standard_bonds)

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
custom template file with the `custom_residue_templates` keyword.
"""
struct MolecularForceField{T, M, D, DA, E, K, KA}
    atom_types::Dict{String, AtomType{T, M, D, E}}
    residues::Dict{String, ResidueTemplate{T}}
    torsion_order::String
    weight_14_coulomb::T
    weight_14_lj::T
    attributes_from_residue::Vector{String}
    residue_name_replacements::Dict{String,String}
    atom_name_replacements::Dict{String, Dict{String, String}}
    standard_bonds::Dict{String, Vector{Tuple{String, String}}}
    class_of::Dict{String, String} # Type -> class
    bond_resolver::BondResolver{K, D}
    angle_resolver::AngleResolver{KA, DA}
    torsion_resolver::TorsionResolver{T, E}
end

function MolecularForceField(T::Type, ff_files::AbstractString...; units::Bool=true,
                             custom_residue_templates=nothing, custom_renaming_scheme=nothing)
    atom_types = Dict{String, AtomType}()
    torsion_order = ""

    weight_14_coulomb, weight_14_lj = one(T), one(T)
    weight_14_coulomb_set, weight_14_lj_set = false, false
    attributes_from_residue = String[]
    residues = Dict{String, ResidueTemplate}()
    patches = Dict{String, ResidueTemplatePatch}()
    type_info = Dict{String, Tuple{String, String}}() # Type => (element, class)

    resname_replacements, atomname_replacements = load_replacements()
    standard_bonds = load_bond_definitions()

    if !isnothing(custom_renaming_scheme)
        resname_replacements, atomname_replacements = load_replacements(
            xmlpath=custom_residue_templates,
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

    # Accumulators for pattern rules
    bond_rule_specs   = Any[]
    angle_rule_specs  = Any[]
    torsion_rule_spec = Any[]
    nb_class_updates  = Any[]

    for ff_file in ff_files
        ff_xml = parsexml(read(ff_file))
        ff = root(ff_xml)

        for entry in eachelement(ff)
            entry_name = entry.name

            if entry_name == "AtomTypes"
                for atom_type in eachelement(entry)
                    at_type  = atom_type["name"]
                    at_class = atom_type["class"]
                    element = (haskey(atom_type, "element") ? atom_type["element"] : "?")
                    ch = missing
                    atom_mass = (units ? parse(T, atom_type["mass"])u"g/mol" : parse(T, atom_type["mass"]))
                    σ = (units ? T(-1u"nm") : T(-1))
                    ϵ = (units ? T(-1u"kJ * mol^-1") : T(-1))
                    atom_types[at_type] = AtomType{T, typeof(atom_mass), typeof(σ), typeof(ϵ)}(
                        at_type, at_class, element, ch, atom_mass, σ, ϵ)
                    type_info[at_type] = (element, at_class)
                end

            elseif entry_name == "Residues"
                for residue in eachelement(entry)
                    rname = residue["name"]
                    atoms, types = String[], String[]
                    charges = T[]
                    elements = Symbol[]
                    external_bonds_name = String[]
                    externals = Int[]
                    allowed_patches = String[]
                    extras = BitVector()
                    bonds_by_name = Tuple{String,String}[]

                    for re in eachelement(residue)
                        if re.name == "Atom"
                            an = re["name"]
                            tp = re["type"]
                            q = parse(T, re["charge"])
                            push!(atoms, an)
                            push!(types, tp)
                            push!(charges, q)
                            push!(externals, 0)
                            tel, tclass = type_info[tp]
                            push!(extras, (tel == "?") || (tclass == "EP"))
                            push!(elements, element_string_to_symbol(tel))
                        elseif re.name == "Bond"
                            push!(bonds_by_name, (re["atomName1"], re["atomName2"]))
                        elseif re.name == "ExternalBond"
                            push!(external_bonds_name, re["atomName"])
                        elseif re.name == "AllowPatch"
                            push!(allowed_patches, re["name"])
                        elseif re.name == "VirtualSite"
                            throw(ArgumentError("virtual sites not currently supported"))
                        end
                    end

                    name_to_idx = Dict(a => i for (i,a) in enumerate(atoms))
                    bonds = Tuple{Int, Int}[]
                    for (a1, a2) in bonds_by_name
                        i, j = name_to_idx[a1], name_to_idx[a2]
                        push!(bonds, (i < j ? (i, j) : (j, i)))
                    end
                    for nm in external_bonds_name
                        if haskey(name_to_idx, nm)
                            externals[name_to_idx[nm]] += 1
                        end
                    end
                    residues[rname] = ResidueTemplate(rname, atoms, elements, types, bonds,
                                                      externals, allowed_patches, charges, extras)
                end

            elseif entry_name == "Patches"
                for patch in eachelement(entry)
                    pname = patch["name"]
                    if haskey(patch, "residues") && patch["residues"] != "1"
                        @warn "Residue patches altering multiple templates not currently " *
                              "supported; ignoring patch $pname"
                        continue
                    end

                    add_atoms = Tuple{String, String, T}[]
                    change_atoms = Tuple{String, String, T}[]
                    remove_atoms = String[]
                    add_bonds = Tuple{String, String}[]
                    remove_bonds = Tuple{String, String}[]
                    add_external_bonds = String[]
                    remove_external_bonds = String[]
                    apply_to_residues = String[]

                    for pa in eachelement(patch)
                        if pa.name == "AddAtom"
                            push!(add_atoms, (pa["name"], pa["type"], parse(T, pa["charge"])))
                        elseif pa.name == "ChangeAtom"
                            push!(change_atoms, (pa["name"], pa["type"], parse(T, pa["charge"])))
                        elseif pa.name == "RemoveAtom"
                            push!(remove_atoms, pa["name"])
                        elseif pa.name == "AddBond"
                            push!(add_bonds, (pa["atomName1"], pa["atomName2"]))
                        elseif pa.name == "RemoveBond"
                            push!(remove_bonds, (pa["atomName1"], pa["atomName2"]))
                        elseif pa.name == "AddExternalBond"
                            push!(add_external_bonds, pa["atomName"])
                        elseif pa.name == "RemoveExternalBond"
                            push!(remove_external_bonds, pa["atomName"])
                        elseif pa.name == "ApplyToResidue"
                            push!(apply_to_residues, pa["name"])
                        end
                    end
                    patches[pname] = ResidueTemplatePatch(pname, add_atoms, change_atoms,
                                        remove_atoms, add_bonds, remove_bonds, add_external_bonds,
                                        remove_external_bonds, apply_to_residues)
                end

            elseif entry_name == "HarmonicBondForce"
                for bond in eachelement(entry)
                    k  = (units ? parse(T, bond["k"])u"kJ * mol^-1 * nm^-2" : parse(T, bond["k"]))
                    r0 = (units ? parse(T, bond["length"])u"nm" : parse(T, bond["length"]))
                    p1 = pattern_from_attrs(bond, "type1","class1")
                    p2 = pattern_from_attrs(bond, "type2","class2")
                    push!(bond_rule_specs, (:bond_rule, p1,p2, HarmonicBond(k,r0)))
                end

            elseif entry_name == "HarmonicAngleForce"
                for ang in eachelement(entry)
                    k  = (units ? parse(T, ang["k"])u"kJ * mol^-1" : parse(T, ang["k"]))
                    θ0 = parse(T, ang["angle"])
                    p1 = pattern_from_attrs(ang, "type1","class1")
                    p2 = pattern_from_attrs(ang, "type2","class2")
                    p3 = pattern_from_attrs(ang, "type3","class3")
                    push!(angle_rule_specs, (:angle_rule, p1, p2, p3, HarmonicAngle(k, θ0)))
                end

            elseif entry_name == "PeriodicTorsionForce"
                torsion_order  = (haskey(entry, "ordering") ? entry["ordering"] : torsion_order)
                local_ordering = (haskey(entry, "ordering") ? entry["ordering"] : "default")
                for torsion in eachelement(entry)
                    proper = torsion.name == "Proper"
                    periodicities = Int[]
                    phases = T[]
                    ks = (units ? typeof(T(1u"kJ * mol^-1"))[] : T[])
                    i = 1
                    while haskey(torsion, "periodicity$i")
                        push!(periodicities, parse(Int, torsion["periodicity$i"]))
                        push!(phases, parse(T,   torsion["phase$i"]))
                        push!(ks, (units ? parse(T, torsion["k$i"])u"kJ * mol^-1" : parse(T, torsion["k$i"])))
                        i += 1
                    end

                    p1 = pattern_from_attrs(torsion, "type1", "class1")
                    p2 = pattern_from_attrs(torsion, "type2", "class2")
                    p3 = pattern_from_attrs(torsion, "type3", "class3")
                    p4 = pattern_from_attrs(torsion, "type4", "class4")

                    has_wildcard = (p1.kind == WILD || p2.kind == WILD || p3.kind == WILD || p4.kind == WILD)
                    spec = UInt8(spec_score(p1) + spec_score(p2) + spec_score(p3) + spec_score(p4))
                    params_any = (:params, periodicities, phases, ks, proper)
                    push!(torsion_rule_spec, (:torsion_rule, p1, p2, p3, p4, spec, params_any,
                                              local_ordering, has_wildcard))
                end

            elseif entry_name == "NonbondedForce"
                if haskey(entry, "coulomb14scale")
                    w = parse(T, entry["coulomb14scale"])
                    if weight_14_coulomb_set && w != weight_14_coulomb
                        error("multiple NonbondedForce entries with different coulomb14scale")
                    end
                    weight_14_coulomb = w
                    weight_14_coulomb_set = true
                end
                if haskey(entry, "lj14scale")
                    w = parse(T, entry["lj14scale"])
                    if weight_14_lj_set && w != weight_14_lj
                        error("multiple NonbondedForce entries with different lj14scale")
                    end
                    weight_14_lj = w
                    weight_14_lj_set = true
                end
                for atom_or_attr in eachelement(entry)
                    if atom_or_attr.name == "Atom"
                        ch = (haskey(atom_or_attr, "charge") ? parse(T, atom_or_attr["charge"]) : missing)
                        σ = (units ? parse(T, atom_or_attr["sigma"])u"nm" : parse(T, atom_or_attr["sigma"]))
                        ϵ = (units ? parse(T, atom_or_attr["epsilon"])u"kJ * mol^-1" : parse(T, atom_or_attr["epsilon"]))
                        if haskey(atom_or_attr, "class")
                            push!(nb_class_updates, (:nb_class, atom_or_attr["class"], ch, σ, ϵ))
                        else
                            atom_type = atom_or_attr["type"]
                            if haskey(atom_types, atom_type)
                                at = atom_types[atom_type]
                                atom_types[atom_type] = AtomType{T, typeof(at.mass), typeof(σ), typeof(ϵ)}(
                                    at.type, at.class, at.element, ch, at.mass, σ, ϵ)
                            end
                        end
                    elseif atom_or_attr.name == "UseAttributeFromResidue"
                        if !(atom_or_attr["name"] in attributes_from_residue)
                            push!(attributes_from_residue, atom_or_attr["name"])
                        end
                        if atom_or_attr["name"] != "charge"
                            @warn "UseAttributeFromResidue only supported for charge; ignoring"
                        end
                    end
                end

            elseif entry_name == "Include"
                @warn "File includes not currently supported; ignoring"
            elseif entry_name in ("RBTorsionForce","CMAPTorsionForce","GBSAOBCForce",
                                  "CustomBondForce","CustomAngleForce","CustomTorsionForce",
                                  "CustomNonbondedForce","CustomGBForce","CustomHbondForce",
                                  "CustomManyParticleForce","LennardJonesForce")
                @warn "$entry_name not currently supported; ignoring"
            end
        end
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
                                              patch_res_name, res_name, patch_name, atom_types)
            if !isnothing(patched_res) # Invalid patches warn and return nothing
                residues[patch_res_name] = patched_res
            end
        end
    end

    # Units parametric types
    if units
        M  = typeof(T(1u"g/mol"))
        D  = typeof(T(1u"nm"))
        DA = typeof(T(1))
        E  = typeof(T(1u"kJ * mol^-1"))
        K  = typeof(T(1u"kJ * mol^-1 * nm^-2"))
        KA = typeof(T(1u"kJ * mol^-1"))
    else
        M, D, DA, E, K, KA = T, T, T, T, T, T
    end

    # Build class maps once
    class_of = Dict{String,String}(t => atom_types[t].class for t in keys(atom_types))
    types_by_class = Dict{String,Vector{String}}()
    for (t, at) in atom_types
        push!(get!(types_by_class, at.class, String[]), t)
    end

    # Apply nonbonded class-based updates
    for item in nb_class_updates
        _, cls, ch, σ, ϵ = item
        for t in get(types_by_class, cls, String[])
            at = atom_types[t]
            atom_types[t] = AtomType{T, typeof(at.mass), typeof(σ), typeof(ϵ)}(
                at.type, at.class, at.element, ch, at.mass, σ, ϵ)
        end
    end

    # Bonds resolver
    bond_rules = BondRule{K,D}[]
    bidx = Dict{Tuple{Symbol, String, String}, Vector{Int}}()
    for spec in bond_rule_specs
        _, p1::AtomPattern, p2::AtomPattern, hb::HarmonicBond{K,D} = spec
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
    angle_rules = AngleRule{KA, DA}[]
    aidx = Dict{Tuple{Symbol, String}, Vector{Int}}()
    for spec in angle_rule_specs
        _, p1::AtomPattern, p2::AtomPattern, p3::AtomPattern, ha::HarmonicAngle{KA,DA} = spec
        push!(angle_rules, AngleRule{KA, DA}(p1, p2, p3, ha,
                                    UInt8(spec_score(p1) + spec_score(p2) + spec_score(p3))))
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
    angle_resolver = AngleResolver{KA, DA}(
        angle_rules,
        aidx,
        Dict{NTuple{3, String}, Union{HarmonicAngle{KA, DA}, Nothing}}(),
    )

    # Torsions resolver
    torsion_rules = TorsionRule{T, E}[]

    # Candidate lists
    propers_by_type2   = Dict{String, Vector{Int}}()
    propers_by_class2  = Dict{String, Vector{Int}}()
    impropers_by_type1 = Dict{String, Vector{Int}}()
    impropers_by_class1= Dict{String, Vector{Int}}()
    wild_propers   = Int[]
    wild_impropers = Int[]

    for (idx_spec, item) in enumerate(torsion_rule_spec)
        if item[1] === :torsion_rule
            _, p1, p2, p3, p4, spec, params_any, ordering, wildcard = item
            _, periodicities, phases, ks, proper = params_any
            params = PeriodicTorsionType{T, E}(periodicities, phases, ks, proper)

            push!(torsion_rules, TorsionRule{T, E}(p1, p2, p3, p4, proper, ordering,
                                                   wildcard, params, spec))
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
    end

    torsion_resolver = TorsionResolver{T, E}(
        torsion_rules,
        impropers_by_type1,
        impropers_by_class1,
        propers_by_type2,
        propers_by_class2,
        wild_impropers,
        wild_propers,
        Dict{Tuple{NTuple{4, String}, NTuple{4, String}} ,Any}(),  # Proper_cache
        Dict{NTuple{4, String}, Any}(),                            # Improper_cache
    )

    return MolecularForceField{T, M, D, DA, E, K, KA}(
        atom_types, residues, torsion_order, weight_14_coulomb, weight_14_lj,
        attributes_from_residue, resname_replacements, atomname_replacements, standard_bonds,
        class_of, bond_resolver, angle_resolver, torsion_resolver,
    )
end

function MolecularForceField(ff_files::AbstractString...; kwargs...)
    return MolecularForceField(DefaultFloat, ff_files...; kwargs...)
end

function Base.show(io::IO, ff::MolecularForceField)
    print(io, "MolecularForceField with ", length(ff.atom_types), " atom types and ",
            length(ff.residues), " residue templates")
end
