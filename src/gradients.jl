# Utilities for taking gradients through simulations

export
    extract_parameters,
    inject_gradients

"""
    extract_parameters(system, force_field)

Form a `Dict` of all parameters in a `System`, allowing gradients
to be tracked.
"""
function extract_parameters(sys::System, ff::OpenMMForceField)
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

    for inter in values(sys.general_inters)
        if inter isa LennardJones
            key_prefix = "inter_LJ_"
            params_dic[key_prefix * "weight_14"] = inter.weight_14
            params_dic[key_prefix * "weight_solute_solvent"] = inter.weight_solute_solvent
        elseif inter isa CoulombReactionField
            key_prefix = "inter_CRF_"
            params_dic[key_prefix * "dist_cutoff"] = inter.dist_cutoff
            params_dic[key_prefix * "solvent_dielectric"] = inter.solvent_dielectric
            params_dic[key_prefix * "weight_14"] = inter.weight_14
            params_dic[key_prefix * "coulomb_const"] = inter.coulomb_const
        end
    end

    for inter in values(sys.specific_inter_lists)
        if Molly.interaction_type(inter) <: HarmonicBond
            for bond_type in inter.types
                key_prefix = "inter_HB_$(bond_type)_"
                if !haskey(params_dic, key_prefix * "b0")
                    bond = ff.bond_types[atom_types_to_tuple(bond_type)]
                    params_dic[key_prefix * "b0"] = bond.b0
                    params_dic[key_prefix * "kb"] = bond.kb
                end
            end
        elseif Molly.interaction_type(inter) <: HarmonicAngle
            for angle_type in inter.types
                key_prefix = "inter_HA_$(angle_type)_"
                if !haskey(params_dic, key_prefix * "th0")
                    angle = ff.angle_types[atom_types_to_tuple(angle_type)]
                    params_dic[key_prefix * "th0"] = angle.th0
                    params_dic[key_prefix * "cth"] = angle.cth
                end
            end
        elseif Molly.interaction_type(inter) <: PeriodicTorsion
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

Add parameters from a dictionary to a `System`.
Allows gradients for individual parameters to be tracked.
Returns atoms, general interactions and specific interaction lists.
"""
function inject_gradients(sys, params_dic, gpu::Bool=isa(sys.coords, CuArray))
    if gpu
        atoms_grad = cu(inject_atom.(Array(sys.atoms), sys.atoms_data, (params_dic,)))
    else
        atoms_grad = inject_atom.(sys.atoms, sys.atoms_data, (params_dic,))
    end
    gis_grad = inject_interaction.(sys.general_inters, (params_dic,))
    sis_grad = inject_interaction_list.(sys.specific_inter_lists, (params_dic,), gpu)
    return atoms_grad, gis_grad, sis_grad
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

function inject_interaction_list(inter::InteractionList2Atoms, params_dic, gpu)
    if gpu
        inters_grad = cu(inject_interaction.(Array(inter.inters), inter.types, (params_dic,)))
    else
        inters_grad = inject_interaction.(inter.inters, inter.types, (params_dic,))
    end
    InteractionList2Atoms(inter.is, inter.js, inter.types, inters_grad)
end

function inject_interaction_list(inter::InteractionList3Atoms, params_dic, gpu)
    if gpu
        inters_grad = cu(inject_interaction.(Array(inter.inters), inter.types, (params_dic,)))
    else
        inters_grad = inject_interaction.(inter.inters, inter.types, (params_dic,))
    end
    InteractionList3Atoms(inter.is, inter.js, inter.ks, inter.types, inters_grad)
end

function inject_interaction_list(inter::InteractionList4Atoms, params_dic, gpu)
    if gpu
        inters_grad = cu(inject_interaction.(Array(inter.inters), inter.types, (params_dic,)))
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
        dict_get(params_dic, key_prefix * "b0", inter.b0),
        dict_get(params_dic, key_prefix * "kb", inter.kb),
    )
end

function inject_interaction(inter::HarmonicAngle, inter_type, params_dic)
    key_prefix = "inter_HA_$(inter_type)_"
    HarmonicAngle(
        dict_get(params_dic, key_prefix * "th0", inter.th0),
        dict_get(params_dic, key_prefix * "cth", inter.cth),
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
