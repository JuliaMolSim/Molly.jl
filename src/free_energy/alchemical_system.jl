export
    AbsoluteFESystem,
    RelativeFESystem

"""
    to_lambda_inter_list(inter_list, scheduler, AT)

Convert every entry of a specific interaction list to its λ counterpart.

Types with no `to_lambda_function` method (`FENEBond`, `HarmonicPositionRestraint`, and the
1-4 pair lists, whose conversion takes a soft core argument) are passed through unchanged
with a warning rather than aborting the whole setup. They then fall under the default
`virial_lambda_factor`, i.e. their virial is left unscaled.
"""
function to_lambda_inter_list(inter_list, scheduler, AT)
    inters_cpu = from_device(inter_list.inters)
    isempty(inters_cpu) && return inter_list
    if !hasmethod(to_lambda_function, Tuple{eltype(inters_cpu)})
        @warn "no to_lambda_function for $(eltype(inters_cpu)), leaving it unscaled; its " *
              "virial will not follow the alchemical coupling"
        return inter_list
    end
    new_inters = [to_lambda_function(x; λ_mixing=MinimumMixing(), scheduler=scheduler)
                  for x in inters_cpu]
    IT = typeof(inter_list).name.wrapper
    fields = getfield.((inter_list,), fieldnames(typeof(inter_list)))
    # The field order is (is, js, …, inters, types, data), so everything before the last
    # three is the per-interaction atom index vectors, which are unchanged here
    return IT(fields[1:(end - 3)]..., to_device(new_inters, AT), inter_list.types,
              inter_list.data)
end

const softcore_dic = Dict("none" => DefaultSoftCore(),
                          "beutler" => BeutlerSoftCore(),
                          "gapsys" => GapsysSoftCore(), 
                          "scaled" => ScaledSoftCore())

"""
    AbsoluteFESystem(sys, global_λ, mapping; temp = 298.0u"K", units=true,
                        scheduler=DefaultLambdaScheduler(dual=true), loggers=(),
                        array_type=Array, float_type=Float32, LJsoftcore="gapsys",
                        Csoftcore="gapsys")

Sets up a absolute free energy system, where atoms are transformed into alchemical atoms based on the
provided indexes in the mapping. 

# Arguments
- `sys`: The reference system in which the solute is present.
- `global_λ`: The global λ for the system.
- `mapping`: A array with atom indexes for the solute that is annihilated.
- `temp = 298.0u"K"`: Temperature to generate random velocities for the system, standard is 298K.
- `units` = true``:  whether to use Unitful quantities.
- `scheduler = DefaultLambdaScheduler(dual=true)`: Lambda scheduler used to transform global λ into
lambda for sterics, electrostatics and bonded interactions. Options include: default, linear, openfe,
    NAMD, quarters, electrostatics scaled. See the plots of the schedulers on the examples page.
- `loggers = ()`:  the loggers that record properties of interest during a
    simulation.
- `array_type = Array`: the array type for the simulation, for example
    use `CuArray` or `ROCArray` for GPU support.
- `float_type = Float32`: 
- `LJsoftcore = "gapsys"`: which softcore type to use for LennardJones potential, options are: "none" = regular LJ,
"beutler" = softcore described in [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1).,
"gapsys" = softcore described in [Gapsys et al. 2012](https://doi.org/10.1021/ct300220p),
"scaled" = potential directly scaled by λ.
- `Csoftcore = "gapsys"`: which softcore type to use for Coulomb potential, see options above.
"""

function AbsoluteFESystem(sys::System, global_λ, mapping; 
                        temp = 298.0u"K", 
                        units=true,
                        scheduler=DefaultLambdaScheduler(dual=true),
                        loggers=(),
                        array_type=Array,
                        float_type=Float32,
                        LJsoftcore="gapsys",
                        Csoftcore="gapsys"
                        )
    # Collect correct datatypes and fill in missing gaps in the mappings
    FT = float_type
    if float_type!=typeof(global_λ)
        @warn "Float type of global_λ does not match float_type argument. Using float_type of global_λ."
        FT = typeof(global_λ)
    end
    if float_type!=typeof(ustrip(temp))
        temp = units ? FT(ustrip(temp))u"K" : FT(ustrip(temp))
    end
    AT = array_type
    # The atoms and coords are rebuilt one at a time below, so bring them to the host once
    # rather than indexing a device array per atom
    sys_atoms  = from_device(sys.atoms)
    sys_coords = from_device(sys.coords)
    S = typeof(sys_atoms[1].σ)
    E = typeof(sys_atoms[1].ϵ)
    C = typeof(sys_atoms[1].charge)

    if !scheduler.dual
        @error "Current parameters scaling for absolute free energy setup is not available, set scheduler(dual=true)"
    end

    # Initialize data groups for new system
    Atoms        = []
    Data         = []
    Coords       = []
    Interactions = []
    Boundary = deepcopy(sys.boundary)

    # Add all the atoms with new numbering (save a mapping to adjust the interactions list later)
    # Add first core, then unique and then environment atoms
    counter = 1
    res_n = 0
    res_number = 0
    chain = ""
    # Add unique atoms from system A
    for i in 1:length(sys_atoms)
        a = sys_atoms[i]
        d = sys.atoms_data[i]
        c = sys_coords[i]
        if i in mapping
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                λ=FT(global_λ), alch_role=DeleteRole))
        else
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                λ=FT(1.0), alch_role=EnvRole))
        end

        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        counter += 1
    end

    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)

    # Pairwise Interactions
    PairInteraction = []
    for inter in sys.pairwise_inters
        if inter isa LennardJones
            push!(PairInteraction, to_lambda_function(inter, softcore_dic[LJsoftcore]; 
                                                        scheduler=scheduler,
                                                        λ_mixing=MinimumMixing(),
                                                        float_type=FT))
        else
            push!(PairInteraction, to_lambda_function(inter, softcore_dic[Csoftcore]; 
                                                        scheduler=scheduler,
                                                        λ_mixing=MinimumMixing(),
                                                        float_type=FT))
        end
    end

    # General interactions
    #
    # The scheduler decides which reciprocal sum the whole system uses. GROMACSLambdaABFEScheduler
    # has linear λ curves, which is what the two grid scheme needs, so it selects `PME_λ`;
    # every other scheduler selects the OpenFE/OpenMM `PME`. The short range and exclusion
    # terms dispatch on the same scheduler through `ewald_pair_qq`, so all three Ewald terms
    # follow from this one choice and cannot be mismatched.
    GenerInteraction = []
    has_ewald = false
    for inter in sys.general_inters
        if inter isa PME
            has_ewald = true
            if scheduler isa GROMACSLambdaABFEScheduler
                push!(GenerInteraction, PME_λ(inter.dist_cutoff, to_device(Atoms, AT), Boundary,
                                        grad_safe=inter.grad_safe;
                                        error_tol=inter.error_tol, fixed_charges=false,
                                        scheduler=scheduler, λ=global_λ),
                            )
            else
                push!(GenerInteraction, PME(inter.dist_cutoff, to_device(Atoms, AT), Boundary,
                                        grad_safe=inter.grad_safe;
                                        error_tol=inter.error_tol, fixed_charges=false,
                                        scheduler=scheduler),
                            )
            end
        elseif inter isa LJDispersionCorrection
            push!(GenerInteraction, LJDispersionCorrectionλ(to_device(Atoms, AT), inter.dist_cutoff,
                            scheduler, MinimumMixing(), inter.σ_mix, inter.ϵ_mix))
        else
            @warn "Currently $inter is not implemented for alchemical simulations"
        end
    end

    # The Ewald real space, exclusion and mesh terms are three halves of one interaction and
    # only add up to a screened Coulomb sum when all three are scaled by the same λ. The mesh
    # scales each atom's charge through the scheduler and so always annihilates the solute's
    # intramolecular electrostatics; `intraC=true` would keep them switched on in the
    # pairwise term alone, leaving the exclusion correction subtracting a reciprocal
    # contribution the mesh never put there.
    if has_ewald && scheduler.intraC
        @warn "scheduler has intraC=true but the system uses a reciprocal sum; the solute's " *
              "intramolecular electrostatics cannot be decoupled independently of the mesh. " *
              "Use intraC=false (annihilation) with PME or PME_λ."
    end

    # Every specific interaction is converted to its λ counterpart, as `RelativeFESystem`
    # already does. Beyond keeping the two constructors consistent, this is what lets the
    # virial tell a self-scaling interaction from a geometry-preserving one by type alone:
    # after this an alchemical atom only ever carries λ types. Atom indices are unchanged in
    # an absolute setup, so the index and type vectors carry straight over.
    SpecificInteraction = Any[]
    for inter_list in sys.specific_inter_lists
        if inter_list isa InteractionList2Atoms && inter_list.data isa EwaldExclusionData
            d = inter_list.data
            push!(SpecificInteraction, InteractionList2Atoms(
                inter_list.is, inter_list.js, inter_list.inters, inter_list.types,
                EwaldExclusionData(d.dist_cutoff; error_tol=d.error_tol, ϵr=d.ϵr,
                                   scheduler=scheduler, λ_mix=d.λ_mixing),
            ))
        else
            push!(SpecificInteraction, to_lambda_inter_list(inter_list, scheduler, AT))
        end
    end

    # Create all interaction tuples
    pairwise_inters = tuple(PairInteraction...)
    general_inters = tuple(GenerInteraction...)
    specific_inter_lists = tuple(SpecificInteraction...)

    # Setup new system
    vels_gpu = [random_velocity(a.mass, temp) for a in Atoms]

    # For the purposes of assigning molecules, add connections from atoms to virtual sites
    bonds_all = sys.specific_inter_lists[1]
    # `bond_graph` iterates the index arrays on the host, so these must not stay on device
    bonds_all_vs_is, bonds_all_vs_js = from_device(bonds_all.is), from_device(bonds_all.js)

    if length(bonds_all_vs_is) > 0
        topology = MolecularTopology(bonds_all_vs_is, bonds_all_vs_js, length(Coords))
    else
        topology = nothing
    end

    sys_final = System(
        atoms=to_device(Atoms, AT),
        coords=to_device(Coords, AT),
        atoms_data=Data,
        boundary=Boundary,
        topology=topology,
        velocities=to_device(vels_gpu, AT),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=to_device.(specific_inter_lists,AT),
        neighbor_finder=sys.neighbor_finder,
        constraints=sys.constraints,
        general_inters=general_inters,
        loggers=loggers,
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
    )

    return sys_final
end

# System A is the main reference system from which the environment atoms are taken
# System B is only used for the Unique system B atoms and the parameters for Core atoms

"""
    RelativeFESystem(sysA, sysB, global_λ, mapping, core_mapAB; temp = 298.0u"K", units=true,
                        scheduler=DefaultLambdaScheduler(dual=true), loggers=(),
                        array_type=Array, float_type=Float32, LJsoftcore="gapsys",
                        Csoftcore="gapsys")

Sets up a relative free energy system, where atoms are interpolated between system A and system B. The mapping provides
the atom indices for the core atoms, unique A and unique B atoms. 

# Arguments
- `sysA`: System A is the main reference system from which the environment atoms are taken.,
- `sysB`: System B is only used for the Unique system B atoms and the parameters for Core atoms
- `global_λ`: The global λ for the system.
- `mapping`: A dictionary with arrays of atom indexes for "core", "unique_A", "unique_B".
- `core_mapAB`: A dictionary of mapping for atom indices for core atoms in system A and matching atom indices
for core atoms in B.
- `temp = 298.0u"K"`: Temperature to generate random velocities for the system, standard is 298K.
- `units` = true``:  whether to use Unitful quantities.
- `scheduler = DefaultLambdaScheduler(dual=true)`: Lambda scheduler used to transform global λ into
lambda for sterics, electrostatics and bonded interactions. Options include: default, linear, openfe,
    NAMD, quarters, electrostatics scaled. See the plots of the schedulers on the examples page.
- `loggers = ()`:  the loggers that record properties of interest during a
    simulation.
- `array_type = Array`: the array type for the simulation, for example
    use `CuArray` or `ROCArray` for GPU support.
- `float_type = Float32`: 
- `LJsoftcore = "gapsys"`: which softcore type to use for LennardJones potential, options are: "none" = regular LJ,
"beutler" = softcore described in [Beutler et al. 1994](https://doi.org/10.1016/0009-2614(94)00397-1).,
"gapsys" = softcore described in [Gapsys et al. 2012](https://doi.org/10.1021/ct300220p),
"scaled" = potential directly scaled by λ.
- `Csoftcore = "gapsys"`: which softcore type to use for Coulomb potential, see options above.
"""

function RelativeFESystem(sysA::System, sysB::System, global_λ, mapping, core_mapAB; 
                        temp = 298.0u"K", 
                        units=true,
                        scheduler=DefaultLambdaScheduler(dual=true),
                        loggers=(),
                        array_type=Array,
                        float_type=Float32,
                        LJsoftcore="gapsys",
                        Csoftcore="gapsys"
                        )
    # Collect correct datatypes and fill in missing gaps in the mappings
    FT = float_type
    if float_type!=typeof(global_λ)
        @warn "Float type of global_λ does not match float_type argument. Using float_type of global_λ."
        FT = typeof(global_λ)
    end
    if float_type!=typeof(ustrip(temp))
        temp = units ? FT(ustrip(temp))u"K" : FT(ustrip(temp))
    end
    AT = array_type
    S = typeof(sysA.atoms[1].σ)
    E = typeof(sysA.atoms[1].ϵ)
    C = typeof(sysA.atoms[1].charge)
    env = []
    for i in 1:length(sysA.atoms)
        if !(i in mapping["unique_A"]) && !(i in mapping["core"])
            push!(env, i)
        end
    end
    mapping["env"] = env
    mapping_A = Dict()
    mapping_B = Dict()
    unique_groups = Dict("sysA"=>[], "sysB"=>[],"core"=>[])

    # Initialize data groups for new system
    Atoms        = []
    Virtual      = []
    Data         = []
    Coords       = []
    Interactions = []
    Boundary = deepcopy(sysA.boundary)

    # Add all the atoms with new numbering (save a mapping to adjust the interactions list later)
    # Add first core, then unique and then environment atoms
    counter = 1
    res_n = 0
    res_number = 0
    chain = ""
    # Add core atoms
    for i in mapping["core"]
        aA = sysA.atoms[i]
        dA = sysA.atoms_data[i]
        cA = sysA.coords[i]
        aB = sysB.atoms[core_mapAB[i]]
        dB = sysB.atoms_data[core_mapAB[i]]
        cB = sysB.coords[core_mapAB[i]]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=aA.atom_type, mass=aA.mass, charge=aA.charge, σ=aA.σ, ϵ=aA.ϵ, 
                                                λ=FT(global_λ), alch_role=CoreDRole))
            if chain!=d.chain_id
                chain = d.chain_id
                res_n = 0
                res_number = 0
            end
            if d.res_number!=res_n
                res_number +=1
                res_n = d.res_number
            end
            push!(Data, AtomData(atom_type=dA.atom_type, atom_name=dA.atom_name, res_number=res_number,
                                        res_name=dA.res_name, chain_id=dA.chain_id, element=dA.element, hetero_atom=dA.hetero_atom))
            push!(Coords, cA)
            mapping_A[i] = counter
            push!(unique_groups["sysA"], counter)
            counter += 1

            push!(Atoms, Atom(index=counter, atom_type=aB.atom_type, mass=(units ? FT(0.0)u"g/mol" : FT(0.0)), charge=aB.charge, σ=aB.σ, ϵ=aB.ϵ, 
                                                λ=FT(global_λ), alch_role=CoreIRole))
            push!(Virtual, OneParticleSite(counter, counter-1))
            push!(Data, AtomData(atom_type=dB.atom_type, atom_name=dB.atom_name, res_number=dB.res_number,
                                        res_name=dB.res_name, chain_id=dB.chain_id, element=dB.element, hetero_atom=dB.hetero_atom))
            push!(Coords, zero(cB))
            mapping_B[core_mapAB[i]] = counter
            push!(unique_groups["sysB"], counter)
            counter += 1
        else
            push!(Atoms, Atom(index=counter, atom_type=aA.atom_type, mass=aA.mass, charge=(aA.charge, aB.charge), σ=(aA.σ, aB.σ), ϵ=(aA.ϵ, aB.ϵ),
                                                λ=FT(global_λ), alch_role=CoreRole))
            if chain!=dA.chain_id
                chain = dA.chain_id
                res_n = 0
                res_number = 0
            end
            if dA.res_number!=res_n
                res_number +=1
                res_n = dA.res_number
            end
            push!(Data, AtomData(atom_type=dA.atom_type, atom_name=dA.atom_name, res_number=res_number,
                                        res_name=dA.res_name, chain_id=dA.chain_id, element=dA.element, hetero_atom=dA.hetero_atom))
            push!(Coords, cA)
            mapping_A[i] = counter
            mapping_B[core_mapAB[i]] = counter
            push!(unique_groups["core"], counter)
            counter += 1
        end
    end
    # Add unique atoms from system A
    for i in mapping["unique_A"]
        a = sysA.atoms[i]
        d = sysA.atoms_data[i]
        c = sysA.coords[i]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                                λ=FT(global_λ), alch_role=DeleteRole))
        else
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=(a.charge, zero(a.charge)), 
                                    σ=(a.σ, a.σ), ϵ=(a.ϵ, zero(a.ϵ)), 
                                    λ=FT(global_λ), alch_role=DeleteRole))
        end
        if chain!=d.chain_id
            chain = d.chain_id
            res_n = 0
            res_number = 0
        end
        if d.res_number!=res_n
            res_number +=1
            res_n = d.res_number
        end
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_A[i] = counter
        push!(unique_groups["sysA"], counter)
        counter += 1
    end

    # Add unique atoms from system B
    for i in mapping["unique_B"]
        a = sysB.atoms[i]
        d = sysB.atoms_data[i]
        c = sysB.coords[i]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                                λ=T(global_λ), alch_role=InsertRole))
        else
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=(zero(a.charge), a.charge), 
                                    σ=(a.σ, a.σ), ϵ=(zero(a.ϵ),   a.ϵ),
                                    λ=FT(global_λ), alch_role=InsertRole))
        end
        if chain!=d.chain_id
            chain = d.chain_id
            res_n = 0
            res_number = 0
        end
        if d.res_number!=res_n
            res_number +=1
            res_n = d.res_number
        end
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_B[i] = counter
        push!(unique_groups["sysB"], counter)
        counter += 1
    end

    # Add environment atoms from system A
    for i in mapping["env"]
        a = sysA.atoms[i]
        d = sysA.atoms_data[i]
        c = sysA.coords[i]
        if scheduler.dual
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=a.charge, σ=a.σ, ϵ=a.ϵ, 
                                                λ=FT(1.0), alch_role=EnvRole))
        else
            push!(Atoms, Atom(index=counter, atom_type=a.atom_type, mass=a.mass, charge=(a.charge, a.charge), σ=(a.σ, a.σ), ϵ=(a.ϵ, a.ϵ),
                                    λ=FT(1.0), alch_role=EnvRole))
        end
        if chain!=d.chain_id
            chain = d.chain_id
            res_n = 0
            res_number = 0
        end
        if d.res_number!=res_n
            res_number +=1
            res_n = d.res_number
        end
        push!(Data, AtomData(atom_type=d.atom_type, atom_name=d.atom_name, res_number=res_number,
                                    res_name=d.res_name, chain_id=d.chain_id, element=d.element, hetero_atom=d.hetero_atom))
        push!(Coords, c)
        mapping_A[i] = counter
        counter += 1
    end

    # Ensure all arrays have correct typing
    Atoms = Vector{typeof(Atoms[1])}(Atoms)
    if scheduler.dual
        Virtual = Vector{typeof(Virtual[1])}(Virtual)
    end
    Coords = Vector{typeof(Coords[1])}(Coords)
    Data = Vector{typeof(Data[1])}(Data)

    # Trackers for interactions for single Topology
    if !scheduler.dual
        single_top_lambda_arrays = Dict{Any, Any}() 
        single_top_atom_maps = Dict{Any, Dict{Tuple, Int}}()
    end

    # Loop through interactions in system A and add them to new system
    # for dual topology the variables are floats
    # for single topology the variables are tuples with (A,nothing)
    for interaction in sysA.specific_inter_lists
        IT = typeof(interaction).name.wrapper
        IIT = typeof(interaction.inters[1]).name.wrapper
        P = hasfield(typeof(interaction.inters[1]), :proper) ? interaction.inters[1].proper : nothing
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        field_types = fieldtypes(typeof(interaction))[1:end-1]
        interaction.inters[1] isa EwaldExclusion && continue
        
        if scheduler.dual
            converted_type = typeof(to_lambda_function(interaction.inters[1]; scheduler=scheduler))
        else
            converted_type = typeof(to_lambda_function_single(interaction.inters[1], nothing; scheduler=scheduler))
        end
        
        field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
        tmp = [T() for T in field_types]
        
        if !scheduler.dual && !haskey(single_top_lambda_arrays,(IT,IIT,P))
            single_top_lambda_arrays[(IT,IIT,P)] = tmp[end-1]
            single_top_atom_maps[(IT,IIT,P)] = Dict{Tuple, Int}()
        end
        
        for field_tuple in zip(field_names[1:end-1]...)
            n_fields = length(field_tuple)
            if all(haskey(mapping_A, field_tuple[i]) for i in 1:n_fields-2)
                mapped_atoms = Int[]

                for i in 1:n_fields-2
                    val = mapping_A[field_tuple[i]]
                    push!(tmp[i], val)
                    push!(mapped_atoms, val) # Track mapped atoms for sysB lookup
                end
                
                if scheduler.dual
                    push!(tmp[n_fields-1], to_lambda_function(field_tuple[end-1]; scheduler=scheduler))
                else
                    push!(tmp[n_fields-1], to_lambda_function_single(field_tuple[end-1], nothing; scheduler=scheduler))
                    single_top_atom_maps[(IT,IIT,P)][Tuple(mapped_atoms)] = length(tmp[n_fields-1])
                end
                
                push!(tmp[n_fields], field_tuple[end])
            end
        end
        Interactions = push!(Interactions, IT(tmp..., nothing))
    end

    # Loop through interactions in system A and add them to new system
    # for dual topology the variables are floats
    # for single topology the variables are tuples that are either updated (A,B) or (0,B) is only occuring in system B
    for interaction in sysB.specific_inter_lists
        IT = typeof(interaction).name.wrapper
        IIT = typeof(interaction.inters[1]).name.wrapper
        P = hasfield(typeof(interaction.inters[1]), :proper) ? interaction.inters[1].proper : nothing
        field_names = getfield.((interaction,), fieldnames(typeof(interaction)))
        field_types = fieldtypes(typeof(interaction))[1:end-1]
        interaction.inters[1] isa EwaldExclusion && continue
        
        if scheduler.dual
            converted_type = typeof(to_lambda_function(interaction.inters[1]; scheduler=scheduler))
            field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
            tmp = [T() for T in field_types]
            
            for field_tuple in zip(field_names[1:end-1]...)
                n_fields = length(field_tuple)
                if all(haskey(mapping_B, field_tuple[i]) for i in 1:n_fields-2)
                    for i in 1:n_fields-2
                        idx = haskey(mapping_B, field_tuple[i]) ? mapping_B[field_tuple[i]] : mapping_A[field_tuple[i]]
                        push!(tmp[i], idx)
                    end
                    push!(tmp[n_fields-1], to_lambda_function(field_tuple[end-1]; scheduler=scheduler))
                    push!(tmp[n_fields], field_tuple[end])
                end
            end
            Interactions = push!(Interactions, IT(tmp..., nothing))
        else
            converted_type = typeof(to_lambda_function_single(nothing, interaction.inters[1]; scheduler=scheduler))
            field_types = [field_types[1:end-2]..., Vector{converted_type}, field_types[end]]
            
            tmp_unique_B = [T() for T in field_types]
            has_unique = false
            
            lambda_array_A = get(single_top_lambda_arrays, (IT,IIT,P), nothing)
            atom_map_A = get(single_top_atom_maps, (IT,IIT,P), Dict{Tuple, Int}())
            
            for field_tuple in zip(field_names[1:end-1]...)
                n_fields = length(field_tuple)
                mapped_atoms = Int[]
                
                if !any(haskey(mapping_B, field_tuple[i]) for i in 1:n_fields-2)
                    continue
                end
                for i in 1:n_fields-2
                    if haskey(mapping_B, field_tuple[i])
                        push!(mapped_atoms, mapping_B[field_tuple[i]])
                    elseif haskey(mapping_A, field_tuple[i])
                        push!(mapped_atoms, mapping_A[field_tuple[i]])
                    end
                end

                mapped_tuple = Tuple(mapped_atoms)
                
                if haskey(atom_map_A, mapped_tuple)
                    idx = atom_map_A[mapped_tuple]
                    lambda_array_A[idx] = update_lambda_function(lambda_array_A[idx], field_tuple[end-1])
                else
                    has_unique = true
                    for i in 1:n_fields-2
                        push!(tmp_unique_B[i], mapped_tuple[i])
                    end
                    push!(tmp_unique_B[n_fields-1], to_lambda_function_single(nothing, field_tuple[end-1]; scheduler=scheduler))
                    push!(tmp_unique_B[n_fields], field_tuple[end])
                end

            end
            
            if has_unique
                Interactions = push!(Interactions, IT(tmp_unique_B..., nothing))
            end
        end
    end

    Interactions = merge(Interactions)

    # Pairwise Interactions
    n_atoms = length(Coords)
    eligible = trues(n_atoms, n_atoms)
    for i in 1:n_atoms
        eligible[i, i] = false
    end

    for (i, j) in zip(Interactions[1].is, Interactions[1].js)
        eligible[i, j] = false
        eligible[j, i] = false
    end

    for (i, k) in zip(Interactions[2].is, Interactions[2].ks)
        eligible[i, k] = false
        eligible[k, i] = false
    end

    pairs = collect(Iterators.product(unique_groups["sysA"], unique_groups["sysB"]))
    for (i, j) in pairs
        eligible[i, j] = false
        eligible[j, i] = false
    end

    special = falses(n_atoms, n_atoms)
    for (i, l) in zip(Interactions[3].is, Interactions[3].ls)
        special[i, l] = true
        special[l, i] = true
    end

    if AT <:AbstractGPUArray
        nf = GPUNeighborFinder(eligible=to_device(eligible, AT), special=to_device(special, AT), n_steps_reorder=10,
                                dist_cutoff=sysA.neighbor_finder.dist_cutoff)
    else
        nf = CellListMapNeighborFinder(eligible=to_device(eligible, AT), special=to_device(special, AT),
                                        n_steps=10, boundary=Boundary, x0=Coords,
                                        dist_cutoff=sysA.neighbor_finder.dist_cutoff)
    end

    PairInteraction = []
    for inter in sysA.pairwise_inters
        if inter isa LennardJones
            push!(PairInteraction, to_lambda_function(inter, softcore_dic[LJsoftcore]; 
                                                        scheduler=scheduler,
                                                        λ_mixing=MinimumMixing(),
                                                        float_type=FT))
        else
            push!(PairInteraction, to_lambda_function(inter, softcore_dic[Csoftcore]; 
                                                        scheduler=scheduler,
                                                        λ_mixing=MinimumMixing(),
                                                        float_type=FT))
        end
    end


    # General interactions
    GenerInteraction = []
    for inter in sysA.general_inters
        if inter isa PME
            # As in `AbsoluteFESystem`, the scheduler picks the reciprocal sum. A relative
            # system also has `InsertRole` atoms, and one mesh weight only serves both roles
            # when they are mirror images, `s_I == 1 - s_D` — which the linear RBFE scheduler
            # provides. Anything else uses `PME`.
            if scheduler isa GROMACSLambdaRBFEScheduler
                push!(GenerInteraction, PME_λ(inter.dist_cutoff, to_device(Atoms, AT), Boundary,
                                            grad_safe=inter.grad_safe;
                                            error_tol=inter.error_tol, fixed_charges=false,
                                            scheduler=scheduler, λ=global_λ),
                            )
            else
                push!(GenerInteraction, PME(inter.dist_cutoff, to_device(Atoms, AT), Boundary,
                                            grad_safe=inter.grad_safe;
                                            error_tol=inter.error_tol, fixed_charges=false,
                                            scheduler=scheduler),
                            )
            end
            excluded_pairs = find_excluded_pairs(eligible, special)
            exclusion_data = EwaldExclusionData(FT(inter.dist_cutoff); error_tol=FT(inter.error_tol), scheduler=scheduler)
            ewald_exclusions = InteractionList2Atoms(
                to_device([ep[1] for ep in excluded_pairs], AT),
                to_device([ep[2] for ep in excluded_pairs], AT),
                to_device(fill(EwaldExclusion(), length(excluded_pairs)), AT),
                fill("", length(excluded_pairs)),
                exclusion_data,
            )
            push!(Interactions, ewald_exclusions)

        elseif inter isa LJDispersionCorrection
            push!(GenerInteraction, LJDispersionCorrectionλ(to_device(Atoms, AT), inter.dist_cutoff,
                            scheduler, MinimumMixing(), inter.σ_mix, inter.ϵ_mix))
        end
    end

    # Create all interaction tuples
    specific_inter_lists = tuple(Interactions...)
    pairwise_inters = tuple(PairInteraction...)
    general_inters = tuple(GenerInteraction...)

    # Setup new system
    vels_gpu = [random_velocity(a.mass, temp) for a in Atoms]

    sys_final = System(
        atoms=to_device(Atoms, AT),
        coords=to_device(Coords, AT),
        atoms_data=Data,
        boundary=Boundary,
        # constraints=sys
        virtual_sites=to_device(Virtual,AT),
        velocities=to_device(vels_gpu, AT),
        pairwise_inters=pairwise_inters,
        specific_inter_lists=to_device.(specific_inter_lists,AT),
        neighbor_finder=nf,
        general_inters=general_inters,
        loggers=loggers,
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
    )

    return sys_final
end