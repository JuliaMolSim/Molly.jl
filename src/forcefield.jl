export 
    MolecularForceField

@enum SpecKind::UInt8 WILD=0 TYPE=1 CLASS=2

struct AtomPattern
    kind::SpecKind
    val::String  # ignored when kind==WILD
end

@inline function matches(p::AtomPattern, t::String, class_of::Dict{String,String})

    p.kind==WILD  && return true
    p.kind==TYPE  && return t == p.val
    p.kind==CLASS && return class_of[t] == p.val
    return false
end

spec_score(ap::AtomPattern) = ap.kind==TYPE ? 2 : ap.kind==CLASS ? 1 : 0

@inline function pattern_from_attrs(n::EzXML.Node, typekey::AbstractString, classkey::AbstractString)
    if haskey(n, typekey)
        v = n[typekey]
        return isempty(v) ? AtomPattern(WILD, "") : AtomPattern(TYPE, v)
    elseif haskey(n, classkey)
        v = n[classkey]
        return isempty(v) ? AtomPattern(WILD, "") : AtomPattern(CLASS, v)
    else
        return AtomPattern(WILD, "")
    end
end

struct AtomType{C, M, S, E}
    type::String
    class::String # Currently this is not used
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

# Bonds
struct BondRule{K,D}
    p1::AtomPattern
    p2::AtomPattern
    params::HarmonicBond{K,D}
    specificity::UInt8
end
struct BondResolver{K,D}
    rules::Vector{BondRule{K,D}}
    # indexes: ordered type pair, ordered class pair, and a broad bin
    idx::Dict{Tuple{Symbol,String,String}, Vector{Int}}
    cache::Dict{Tuple{String,String}, Union{HarmonicBond{K,D},Nothing}}
end

# Angles
struct AngleRule{K,D}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    params::HarmonicAngle{K,D}
    specificity::UInt8
end
struct AngleResolver{K,D}
    rules::Vector{AngleRule{K,D}}
    # index primarily by central pair (p2 with p1/p3 hints)
    idx::Dict{Tuple{Symbol,String,String}, Vector{Int}}  # (:type|:class|:wild, v2, key)
    cache::Dict{NTuple{3,String}, Union{HarmonicAngle{K,D},Nothing}}
end

# Torsions
struct TorsionRule{T,E}
    p1::AtomPattern
    p2::AtomPattern
    p3::AtomPattern
    p4::AtomPattern
    proper::Bool
    params::PeriodicTorsionType{T,E}
    specificity::UInt8
end
struct TorsionResolver{T,E}
    rules::Vector{TorsionRule{T,E}}
    center_index::Dict{Tuple{Symbol,String,String}, Vector{Int}} # (:type|:class|:wild, v2, v3)
    cache::Dict{NTuple{4,String}, Union{PeriodicTorsionType{T,E},Nothing}}
end

function find_torsion_params(
    t1::String,t2::String,t3::String,t4::String;
    resolver::TorsionResolver{T,E},
    class_of::Dict{String,String}
) where {T,E}
    key = (t1,t2,t3,t4)
    if haskey(resolver.cache, key)
        return resolver.cache[key]
    end
    cand = Int[]
    # exact type pair
    append!(cand, get(resolver.center_index, (:type,  t2, t3), Int[]))
    # exact class pair
    c2, c3 = class_of[t2], class_of[t3]
    append!(cand, get(resolver.center_index, (:class, c2, c3), Int[]))
    # broad bucket
    append!(cand, get(resolver.center_index, (:wild,  "", ""), Int[]))

    best_spec = Int8(-1)
    best = nothing
    @inbounds for i in cand
        r = resolver.rules[i]
        if matches(r.p1,t1,class_of) & matches(r.p2,t2,class_of) &
           matches(r.p3,t3,class_of) & matches(r.p4,t4,class_of)
            if r.specificity > best_spec
                best_spec = r.specificity
                best = r.params
            end
        end
    end
    resolver.cache[key] = best
    return best
end

"""
    MolecularForceField(ff_files...; units=true, custom_residue_templates = nothing, custom_renaming_scheme = nothing)
    MolecularForceField(T, ff_files...; units=true)
    MolecularForceField(atom_types, residue_types, bond_types, angle_types,
                        torsion_types, torsion_order, weight_14_coulomb,
                        weight_14_lj, attributes_from_residue,
                        residue_name_replacements, atom_name_replacements,
                        standard_bonds)

A molecular force field. 

Read one or more OpenMM force field XML files by passing them to the constructor.

In order to  assign force field parameters to the atoms in the simulation, the
residues determined from a structure file are matched to templates provided by
the force field file, as well as a template dictionary in .xml format, which defines
the standard topology (bonds) of the residues to be found in the simulation. At the
moment, Molly provides a dictionary for all standard aminoacids, nucleic acids and
water for this purpose. If the system to be simulated contains other molecules, their
template topologies must be defined either through `CONNECT` records in the .pdb file 
or by prviding an extra custom template file with the `custom_residue_templates` keyword.

"""
struct MolecularForceField{T, M, D, DA, E, K, KA}
    atom_types::Dict{String, AtomType{T, M, D, E}}
    residues::Dict{String, ResidueTemplate{T}}
    bond_types::Dict{Tuple{String, String}, HarmonicBond{K, D}}
    angle_types::Dict{Tuple{String, String, String}, HarmonicAngle{E, T}}
    torsion_types::Dict{Tuple{String, String, String, String}, PeriodicTorsionType{T, E}}  # exact-type rules
    torsion_order::String
    weight_14_coulomb::T
    weight_14_lj::T
    attributes_from_residue::Vector{String}
    residue_name_replacements::Dict{String,String}
    atom_name_replacements::Dict{String,Dict{String,String}}
    standard_bonds::Dict{String,Vector{Tuple{String, String}}}

    # Class/torsion machinery
    class_of::Dict{String,String}                # type -> class
    bond_resolver::BondResolver{K, D}
    angle_resolver::AngleResolver{KA, DA}
    torsion_resolver::TorsionResolver{T,E}       # wildcard/class rules and index
end

function MolecularForceField(T::Type, ff_files::AbstractString...; units::Bool=true,
                             custom_residue_templates = nothing,
                             custom_renaming_scheme   = nothing)

    atom_types = Dict{String, AtomType}()
    bond_types = Dict{Tuple{String, String}, HarmonicBond}()
    angle_types = Dict{Tuple{String, String, String}, HarmonicAngle}()
    torsion_types = Dict{Tuple{String, String, String, String}, PeriodicTorsionType}()
    torsion_order = ""

    weight_14_coulomb, weight_14_lj = one(T), one(T)
    weight_14_coulomb_set, weight_14_lj_set = false, false
    attributes_from_residue = String[]
    residues = Dict{String,ResidueTemplate}()
    type_info = Dict{String,Tuple{String,String}}()  # type => (element, class)

    resname_replacements, atomname_replacements = load_replacements()
    standard_bonds = load_bond_definitions()

    if !isnothing(custom_renaming_scheme)
        resname_replacements, atomname_replacements = load_replacements(;
            xmlpath = custom_residue_templates,
            resname_replacements  = resname_replacements,
            atomname_replacements = atomname_replacements
        )
    end
    if !isnothing(custom_residue_templates)
        standard_bonds = load_bond_definitions(;
            xmlpath = custom_residue_templates,
            standardBonds = standard_bonds
        )
    end

    # Bond rule accumulator
    bond_rule_specs   = Any[]
    # Angle rules
    angle_rule_specs  = Any[]
    # Torsion rules
    torsion_rule_spec = Any[]
    center_index_any = Dict{Tuple{Symbol,String,String}, Vector{Int}}()

    @inline function push_center!(idx::Int, p2::AtomPattern, p3::AtomPattern)
        if p2.kind == TYPE && p3.kind == TYPE
            push!(get!(center_index_any, (:type,  p2.val, p3.val), Int[]), idx)
        elseif p2.kind == CLASS && p3.kind == CLASS
            push!(get!(center_index_any, (:class, p2.val, p3.val), Int[]), idx)
        else
            # mixed or wildcard
            push!(get!(center_index_any, (:wild, "", ""), Int[]), idx)
        end
    end

    for ff_file in ff_files
        ff_xml = parsexml(read(ff_file))
        ff = root(ff_xml)

        for entry in eachelement(ff)
            entry_name = entry.name

            if entry_name == "AtomTypes"
                for atom_type in eachelement(entry)
                    at_type = atom_type["name"]
                    at_class = atom_type["class"]
                    element = haskey(atom_type, "element") ? atom_type["element"] : "?"
                    ch = missing
                    atom_mass = units ? parse(T, atom_type["mass"])u"g/mol" : parse(T, atom_type["mass"])
                    σ = units ? T(-1u"nm") : T(-1)
                    ϵ = units ? T(-1u"kJ * mol^-1") : T(-1)
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
                            push!(elements, tel == "?" ? :X : Symbol(tel))

                        elseif re.name == "Bond"
                            push!(bonds_by_name, (re["atomName1"], re["atomName2"]))
                        elseif re.name == "ExternalBond"
                            push!(external_bonds_name, re["atomName"])
                        end
                    end

                    name_to_idx = Dict(a => i for (i,a) in enumerate(atoms))
                    bonds = Tuple{Int,Int}[]
                    for (a1,a2) in bonds_by_name
                        i = name_to_idx[a1]
                        j = name_to_idx[a2]
                        push!(bonds, i < j ? (i,j) : (j,i))
                    end

                    for nm in external_bonds_name
                        if haskey(name_to_idx, nm)
                            externals[name_to_idx[nm]] += 1
                        end
                    end
                    residues[rname] = ResidueTemplate(rname, atoms, elements, types, bonds, externals, charges, extras)
                end
            elseif entry_name == "HarmonicBondForce"
                for bond in eachelement(entry)
                    k = units ? parse(T, bond["k"])u"kJ * mol^-1 * nm^-2" : parse(T, bond["k"])
                    r0 = units ? parse(T, bond["length"])u"nm" : parse(T, bond["length"])
                    if haskey(bond, "class1") || haskey(bond, "type1")
                        p1 = pattern_from_attrs(bond, "type1","class1")
                        p2 = pattern_from_attrs(bond, "type2","class2")
                        push!(bond_rule_specs, (:bond_rule, p1,p2, HarmonicBond(k,r0)))
                    else
                        # keep exact types for O(1) path
                        t1 = bond["type1"]
                        t2 = bond["type2"]
                        bond_types[(t1,t2)] = HarmonicBond(k,r0)
                    end
                end

            elseif entry_name == "HarmonicAngleForce"
                for ang in eachelement(entry)
                    k  = units ? parse(T, ang["k"])u"kJ * mol^-1" : parse(T, ang["k"])
                    θ0 = parse(T, ang["angle"])
                    if haskey(ang,"class1") || haskey(ang,"type1")
                        p1 = pattern_from_attrs(ang, "type1","class1")
                        p2 = pattern_from_attrs(ang, "type2","class2")
                        p3 = pattern_from_attrs(ang, "type3","class3")
                        push!(angle_rule_specs, (:angle_rule, p1,p2,p3, HarmonicAngle(k,θ0)))
                    else
                        t1 = ang["type1"]
                        t2 = ang["type2"]
                        t3 = ang["type3"]
                        angle_types[(t1,t2,t3)] = HarmonicAngle(k,θ0)
                    end
                end

            elseif entry_name == "PeriodicTorsionForce"
                torsion_order = haskey(entry, "ordering") ? entry["ordering"] : "default"

                for torsion in eachelement(entry)
                    proper = torsion.name == "Proper"
                    periodicities = Int[]
                    phases = T[]
                    ks = units ? typeof(T(1u"kJ * mol^-1"))[] : T[]
                    i = 1
                    while haskey(torsion, "periodicity$i")
                        push!(periodicities, parse(Int, torsion["periodicity$i"]))
                        push!(phases,       parse(T,   torsion["phase$i"]))
                        push!(ks, units ? parse(T, torsion["k$i"])u"kJ * mol^-1" : parse(T, torsion["k$i"]))
                        i += 1
                    end

                    # Gather any provided type/class strings; "" means wildcard
                    v1 = haskey(torsion,"type1") ? torsion["type1"] : (haskey(torsion,"class1") ? torsion["class1"] : "")
                    v2 = haskey(torsion,"type2") ? torsion["type2"] : (haskey(torsion,"class2") ? torsion["class2"] : "")
                    v3 = haskey(torsion,"type3") ? torsion["type3"] : (haskey(torsion,"class3") ? torsion["class3"] : "")
                    v4 = haskey(torsion,"type4") ? torsion["type4"] : (haskey(torsion,"class4") ? torsion["class4"] : "")

                    # Exact all-type rule can go straight into dict for O(1) later
                    if haskey(torsion,"type1") && haskey(torsion,"type2") && haskey(torsion,"type3") && haskey(torsion,"type4")
                       torsion_types[(v1,v2,v3,v4)] = PeriodicTorsionType(periodicities, phases, ks, proper)
                    end

                    # build patterns per position using attribute presence, not string membership
                    p1 = pattern_from_attrs(torsion, "type1", "class1")
                    p2 = pattern_from_attrs(torsion, "type2", "class2")
                    p3 = pattern_from_attrs(torsion, "type3", "class3")
                    p4 = pattern_from_attrs(torsion, "type4", "class4")

                    spec = UInt8(spec_score(p1)+spec_score(p2)+spec_score(p3)+spec_score(p4))
                    # store the rule for lazy resolution
                    params_any = (:params, periodicities, phases, ks, proper)
                    idx = length(torsion_rule_spec) + 1
                    push!(torsion_rule_spec, (:torsion_rule, p1,p2,p3,p4, spec, params_any))
                    push_center!(idx, p2, p3) 
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
                        ch = haskey(atom_or_attr, "charge") ? parse(T, atom_or_attr["charge"]) : missing
                        σ = units ? parse(T, atom_or_attr["sigma"])u"nm" : parse(T, atom_or_attr["sigma"])
                        ϵ = units ? parse(T, atom_or_attr["epsilon"])u"kJ * mol^-1" : parse(T, atom_or_attr["epsilon"])

                        if haskey(atom_or_attr, "class")
                            # Defer expansion until we know types_by_class
                            push!(torsion_rule_spec, (:nb_class, atom_or_attr["class"], ch, σ, ϵ))
                        else
                            atom_type = atom_or_attr["type"]
                            if haskey(atom_types, atom_type)
                                partial_type = atom_types[atom_type]
                                atom_types[atom_type] = AtomType{T, typeof(partial_type.mass), typeof(σ), typeof(ϵ)}(
                                    partial_type.type, partial_type.class, partial_type.element,
                                    ch, partial_type.mass, σ, ϵ)
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

            elseif entry_name == "Patches"
                @warn "Residue patches not currently supported; ignoring"
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

    # Units parametric types
    if units
        M = typeof(T(1u"g/mol"))
        D = typeof(T(1u"nm"))
        DA = typeof(T(1)) 
        E = typeof(T(1u"kJ * mol^-1"))
        K = typeof(T(1u"kJ * mol^-1 * nm^-2"))
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

    # Finalize Nonbonded class-based updates efficiently
    for item in torsion_rule_spec
        kind = item[1]
        if kind === :nb_class
            _, cls, ch, σ, ϵ = item
            for t in get(types_by_class, cls, String[])
                at = atom_types[t]
                atom_types[t] = AtomType{T, typeof(at.mass), typeof(σ), typeof(ϵ)}(
                    at.type, at.class, at.element, ch, at.mass, σ, ϵ)
            end
        end
    end

    # Bonds resolver
    bond_rules = BondRule{K,D}[]
    bidx = Dict{Tuple{Symbol,String,String}, Vector{Int}}()
    for spec in bond_rule_specs
        _, p1::AtomPattern, p2::AtomPattern, hb::HarmonicBond{K,D} = spec
        push!(bond_rules, BondRule{K,D}(p1, p2, hb, UInt8(spec_score(p1)+spec_score(p2))))
        i = length(bond_rules)
        # index both orientations
        for (a,b) in ((p1,p2),(p2,p1))
            if a.kind==TYPE && b.kind==TYPE
                push!(get!(bidx, (:type,  a.val, b.val), Int[]), i)
            elseif a.kind==CLASS && b.kind==CLASS
                push!(get!(bidx, (:class, a.val, b.val), Int[]), i)
            else
                push!(get!(bidx, (:wild,  "", ""), Int[]), i)
            end
        end
    end
    bond_resolver = BondResolver{K,D}(bond_rules, bidx, Dict{Tuple{String,String},Union{HarmonicBond{K,D},Nothing}}())

    # Angles resolver
    angle_rules = AngleRule{KA,DA}[]
    aidx = Dict{Tuple{Symbol,String,String}, Vector{Int}}()
    for spec in angle_rule_specs
        _, p1::AtomPattern, p2::AtomPattern, p3::AtomPattern, ha::HarmonicAngle{E,T} = spec
        push!(angle_rules, AngleRule{E,T}(p1,p2,p3, ha, UInt8(spec_score(p1)+spec_score(p2)+spec_score(p3))))
        i = length(angle_rules)
        # central indexing: use p2 as the key; neighbor hint "key" groups mixed patterns
        if p2.kind==TYPE
            push!(get!(aidx, (:type,  p2.val, ""), Int[]), i)
        elseif p2.kind==CLASS
            push!(get!(aidx, (:class, p2.val, ""), Int[]), i)
        else
            push!(get!(aidx, (:wild,  "", ""), Int[]), i)
        end
    end
    angle_resolver = AngleResolver{KA,DA}(angle_rules, aidx, Dict{NTuple{3,String},Union{HarmonicAngle{KA,DA},Nothing}}())

    # Torsion resolver
    torsion_rules = TorsionRule{T,E}[]
    tindex = Dict{Tuple{Symbol,String,String}, Vector{Int}}()
    for (idx, item) in enumerate(torsion_rule_spec)
        if item[1] === :torsion_rule
            _, p1,p2,p3,p4, spec, params_any = item
            _, periodicities, phases, ks, proper = params_any
            params = PeriodicTorsionType{T,E}(periodicities, phases, ks, proper)
            push!(torsion_rules, TorsionRule{T,E}(p1,p2,p3,p4, proper, params, spec))

            # re-index center bins
            p2i, p3i = p2::AtomPattern, p3::AtomPattern
            if p2i.kind==TYPE && p3i.kind==TYPE
                push!(get!(tindex, (:type,  p2i.val, p3i.val), Int[]), length(torsion_rules))
            elseif p2i.kind==CLASS && p3i.kind==CLASS
                push!(get!(tindex, (:class, p2i.val, p3i.val), Int[]), length(torsion_rules))
            else
                push!(get!(tindex, (:wild, "", ""), Int[]), length(torsion_rules))
            end
        end
    end
    torsion_resolver = TorsionResolver{T,E}(torsion_rules, tindex, Dict{NTuple{4,String}, Union{PeriodicTorsionType{T,E},Nothing}}())

     return MolecularForceField{T, M, D, DA, E, K, KA}(
        atom_types, residues, bond_types, angle_types,
        torsion_types, torsion_order, weight_14_coulomb, weight_14_lj, attributes_from_residue,
        resname_replacements, atomname_replacements, standard_bonds,
        class_of, bond_resolver, angle_resolver, torsion_resolver
    )
end

function MolecularForceField(ff_files::AbstractString...; kwargs...)
    return MolecularForceField(DefaultFloat, ff_files...; kwargs...)
end

function Base.show(io::IO, ff::MolecularForceField)
    print(io, "MolecularForceField with ", length(ff.atom_types), " atom types and",
            length(ff.residues), " residues.")
end