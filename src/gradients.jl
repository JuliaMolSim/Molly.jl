# Utilities for taking gradients through simulations

export
    extract_parameters,
    inject_gradients

"""
    extract_parameters(system, force_field)

Form a `Dict` of all parameters in a [`System`](@ref), allowing gradients to be tracked.
"""
function extract_parameters(sys, ff)
    params_dic = Dict()

    for at_data in sys.atoms_data
        key_prefix = "atom_$(at_data.atom_type)_"
        if !haskey(params_dic, key_prefix * "mass")
            at = ff.atom_types[at_data.atom_type]
            params_dic[key_prefix * "mass"] = at.mass
            params_dic[key_prefix * "σ"   ] = at.σ
            params_dic[key_prefix * "ϵ"   ] = at.ϵ
        end
    end

    for inter in values(sys.pairwise_inters)
        if inter isa LennardJones
            key_prefix = "inter_LJ_"
            params_dic[key_prefix * "weight_14"] = inter.weight_14
            params_dic[key_prefix * "weight_solute_solvent"] = inter.weight_solute_solvent
        elseif inter isa Coulomb
            key_prefix = "inter_CO_"
            params_dic[key_prefix * "weight_14"] = inter.weight_14
            params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
        elseif inter isa CoulombReactionField
            key_prefix = "inter_CRF_"
            params_dic[key_prefix * "dist_cutoff"] = inter.dist_cutoff
            params_dic[key_prefix * "solvent_dielectric"] = inter.solvent_dielectric
            params_dic[key_prefix * "weight_14"] = inter.weight_14
            params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
        end
    end

    for inter in values(sys.specific_inter_lists)
        if interaction_type(inter) <: HarmonicBond
            for bond_type in inter.types
                key_prefix = "inter_HB_$(bond_type)_"
                if !haskey(params_dic, key_prefix * "k")
                    bond = ff.bond_types[atom_types_to_tuple(bond_type)]
                    params_dic[key_prefix * "k" ] = bond.k
                    params_dic[key_prefix * "r0"] = bond.r0
                end
            end
        elseif interaction_type(inter) <: HarmonicAngle
            for angle_type in inter.types
                key_prefix = "inter_HA_$(angle_type)_"
                if !haskey(params_dic, key_prefix * "k")
                    angle = ff.angle_types[atom_types_to_tuple(angle_type)]
                    params_dic[key_prefix * "k" ] = angle.k
                    params_dic[key_prefix * "θ0"] = angle.θ0
                end
            end
        elseif interaction_type(inter) <: PeriodicTorsion
            for (torsion_type, torsion_inter) in zip(inter.types, Array(inter.inters))
                if torsion_inter.proper
                    key_prefix = "inter_PT_$(torsion_type)_"
                else
                    key_prefix = "inter_IT_$(torsion_type)_"
                end
                if !haskey(params_dic, key_prefix * "phase_1")
                    torsion = ff.torsion_types[atom_types_to_tuple(torsion_type)]
                    for i in 1:length(torsion.phases)
                        params_dic[key_prefix * "phase_$i"] = torsion.phases[i]
                        params_dic[key_prefix * "k_$i"    ] = torsion.ks[i]
                    end
                end
            end
        end
    end

    return params_dic
end

"""
    inject_gradients(sys, params_dic)

Add parameters from a dictionary to a [`System`](@ref).
Allows gradients for individual parameters to be tracked.
Returns atoms, pairwise interactions, specific interaction lists and general
interactions.
"""
function inject_gradients(sys, params_dic, gpu::Bool=isa(sys.coords, CuArray))
    if gpu
        atoms_grad = CuArray(inject_atom.(Array(sys.atoms), sys.atoms_data, (params_dic,)))
    else
        atoms_grad = inject_atom.(sys.atoms, sys.atoms_data, (params_dic,))
    end
    if length(sys.pairwise_inters) > 0
        pis_grad = inject_interaction.(sys.pairwise_inters, (params_dic,))
    else
        pis_grad = sys.pairwise_inters
    end
    if length(sys.specific_inter_lists) > 0
        sis_grad = inject_interaction_list.(sys.specific_inter_lists, (params_dic,), gpu)
    else
        sis_grad = sys.specific_inter_lists
    end
    if length(sys.general_inters) > 0
        gis_grad = inject_interaction.(sys.general_inters, (params_dic,), (sys,))
    else
        gis_grad = sys.general_inters
    end
    return atoms_grad, pis_grad, sis_grad, gis_grad
end

# get function errors with AD
dict_get(dic, key, default) = haskey(dic, key) ? dic[key] : default

function inject_atom(at, at_data, params_dic)
    key_prefix = "atom_$(at_data.atom_type)_"
    Atom(
        at.index,
        at.charge, # Residue-specific
        dict_get(params_dic, key_prefix * "mass"  , at.mass  ),
        dict_get(params_dic, key_prefix * "σ"     , at.σ     ),
        dict_get(params_dic, key_prefix * "ϵ"     , at.ϵ     ),
        at.solute,
    )
end

function inject_interaction_list(inter::InteractionList1Atoms, params_dic, gpu)
    if gpu
        inters_grad = CuArray(inject_interaction.(Array(inter.inters), inter.types, (params_dic,)))
    else
        inters_grad = inject_interaction.(inter.inters, inter.types, (params_dic,))
    end
    InteractionList1Atoms(inter.is, inter.types, inters_grad)
end

function inject_interaction_list(inter::InteractionList2Atoms, params_dic, gpu)
    if gpu
        inters_grad = CuArray(inject_interaction.(Array(inter.inters), inter.types, (params_dic,)))
    else
        inters_grad = inject_interaction.(inter.inters, inter.types, (params_dic,))
    end
    InteractionList2Atoms(inter.is, inter.js, inter.types, inters_grad)
end

function inject_interaction_list(inter::InteractionList3Atoms, params_dic, gpu)
    if gpu
        inters_grad = CuArray(inject_interaction.(Array(inter.inters), inter.types, (params_dic,)))
    else
        inters_grad = inject_interaction.(inter.inters, inter.types, (params_dic,))
    end
    InteractionList3Atoms(inter.is, inter.js, inter.ks, inter.types, inters_grad)
end

function inject_interaction_list(inter::InteractionList4Atoms, params_dic, gpu)
    if gpu
        inters_grad = CuArray(inject_interaction.(Array(inter.inters), inter.types, (params_dic,)))
    else
        inters_grad = inject_interaction.(inter.inters, inter.types, (params_dic,))
    end
    InteractionList4Atoms(inter.is, inter.js, inter.ks, inter.ls,
                          inter.types, inters_grad)
end

function inject_interaction(inter::LennardJones{S, C, W, WS, F, E}, params_dic) where {S, C, W, WS, F, E}
    key_prefix = "inter_LJ_"
    LennardJones{S, C, W, WS, F, E}(
        inter.cutoff,
        inter.nl_only,
        inter.lorentz_mixing,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_14),
        dict_get(params_dic, key_prefix * "weight_solute_solvent", inter.weight_solute_solvent),
        inter.force_units,
        inter.energy_units,
    )
end

function inject_interaction(inter::Coulomb, params_dic)
    key_prefix = "inter_CO_"
    Coulomb(
        inter.cutoff,
        inter.nl_only,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_14),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
        inter.force_units,
        inter.energy_units,
    )
end

function inject_interaction(inter::CoulombReactionField, params_dic)
    key_prefix = "inter_CRF_"
    CoulombReactionField(
        dict_get(params_dic, key_prefix * "dist_cutoff", inter.dist_cutoff),
        dict_get(params_dic, key_prefix * "solvent_dielectric", inter.solvent_dielectric),
        inter.nl_only,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_14),
        dict_get(params_dic, key_prefix * "coulomb_const", inter.coulomb_const),
        inter.force_units,
        inter.energy_units,
    )
end

function inject_interaction(inter::HarmonicBond, inter_type, params_dic)
    key_prefix = "inter_HB_$(inter_type)_"
    HarmonicBond(
        dict_get(params_dic, key_prefix * "k" , inter.k ),
        dict_get(params_dic, key_prefix * "r0", inter.r0),
    )
end

function inject_interaction(inter::HarmonicAngle, inter_type, params_dic)
    key_prefix = "inter_HA_$(inter_type)_"
    HarmonicAngle(
        dict_get(params_dic, key_prefix * "k" , inter.k ),
        dict_get(params_dic, key_prefix * "θ0", inter.θ0),
    )
end

function inject_interaction(inter::PeriodicTorsion{N, T, E}, inter_type, params_dic) where {N, T, E}
    if inter.proper
        key_prefix = "inter_PT_$(inter_type)_"
    else
        key_prefix = "inter_IT_$(inter_type)_"
    end
    PeriodicTorsion{N, T, E}(
        inter.periodicities,
        ntuple(i -> dict_get(params_dic, key_prefix * "phase_$i", inter.phases[i]), N),
        ntuple(i -> dict_get(params_dic, key_prefix * "k_$i"    , inter.ks[i]    ), N),
        inter.proper,
    )
end

function inject_interaction(inter::ImplicitSolventGBN2, params_dic, sys)
    key_prefix = "inter_GB_"
    bond_index = findfirst(sil -> eltype(sil.inters) <: HarmonicBond, sys.specific_inter_lists)

    element_to_radius = Dict{String, DefaultFloat}() # Units here made the gradients vanish
    for k in keys(mbondi2_element_to_radius)
        element_to_radius[k] = dict_get(params_dic, key_prefix * "radius_" * k,
                                        ustrip(mbondi2_element_to_radius[k]))
    end
    element_to_screen = empty(gbn2_element_to_screen)
    for k in keys(gbn2_element_to_screen)
        element_to_screen[k] = dict_get(params_dic, key_prefix * "screen_" * k, gbn2_element_to_screen[k])
    end
    atom_params = empty(gbn2_atom_params)
    for k in keys(gbn2_atom_params)
        atom_params[k] = dict_get(params_dic, key_prefix * "params_" * k, gbn2_atom_params[k])
    end

    ImplicitSolventGBN2(
        sys.atoms,
        sys.atoms_data,
        sys.specific_inter_lists[bond_index];
        solvent_dielectric=dict_get(params_dic, key_prefix * "solvent_dielectric", inter.solvent_dielectric),
        solute_dielectric=dict_get(params_dic, key_prefix * "solute_dielectric", inter.solute_dielectric),
        kappa=dict_get(params_dic, key_prefix * "kappa", ustrip(inter.kappa))u"nm^-1",
        offset=dict_get(params_dic, key_prefix * "offset", ustrip(inter.offset))u"nm",
        dist_cutoff=inter.dist_cutoff,
        probe_radius=dict_get(params_dic, key_prefix * "probe_radius", ustrip(inter.probe_radius))u"nm",
        sa_factor=dict_get(params_dic, key_prefix * "sa_factor", ustrip(inter.sa_factor))u"kJ * mol^-1 * nm^-2",
        use_ACE=inter.use_ACE,
        neck_scale=dict_get(params_dic, key_prefix * "neck_scale", inter.neck_scale),
        neck_cut=dict_get(params_dic, key_prefix * "neck_cut", ustrip(inter.neck_cut))u"nm",
        element_to_radius=element_to_radius,
        element_to_screen=element_to_screen,
        atom_params=atom_params,
    )
end
