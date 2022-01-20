# Utilities for taking gradients through simulations

export inject_gradients

"""
    inject_gradients(sys, params_dic, gpu=false)

Add parameters from a dictionary to a `System`.
Allows gradients for individual parameters to be tracked.
Returns atoms, general interactions and specific interaction lists.
"""
function inject_gradients(sys, params_dic, gpu::Bool=false)
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
        dict_get(params_dic, key_prefix * "charge", at.charge),
        dict_get(params_dic, key_prefix * "mass"  , at.mass  ),
        dict_get(params_dic, key_prefix * "σ"     , at.σ     ),
        dict_get(params_dic, key_prefix * "ϵ"     , at.ϵ     ),
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

function inject_interaction(inter::LennardJones{S, C, W, F, E}, params_dic) where {S, C, W, F, E}
    key_prefix = "inter_LJ_"
    LennardJones{S, C, W, F, E}(
        inter.cutoff,
        inter.nl_only,
        inter.lorentz_mixing,
        dict_get(params_dic, key_prefix * "weight_14", inter.weight_14),
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
    key_prefix = "inter_PT_$(inter_type)_"
    PeriodicTorsion{N, T, E}(
        inter.periodicities,
        ntuple(i -> dict_get(params_dic, key_prefix * "phase_$i", inter.phases[i]), N),
        ntuple(i -> dict_get(params_dic, key_prefix * "k_$i"    , inter.ks[i]    ), N),
    )
end
