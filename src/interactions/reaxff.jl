export
    ReaxFF2008,
    ReaxFF

abstract type AbstractReaxFFVersion end

"""
    ReaxFF2008()

Selects the 2008 ReaxFF functional form used by [`ReaxFF`](@ref)
([Chenoweth et al. 2008](https://doi.org/10.1021/jp709896w)).
This is the default and currently supported ReaxFF version.
"""
struct ReaxFF2008 <: AbstractReaxFFVersion end


struct GeneralParameters{T}
    p_boc1::T 
    p_boc2::T  
    p_coa2::T 
    p_trip4::T 
    p_trip3::T 
    k_c2::T 
    p_ovun6::T 
    p_trip2::T 
    p_ovun7::T 
    p_ovun8::T 
    p_trip1::T 
    lower_taper_radius::T  
    upper_taper_radius::T 
    p_val6::T 
    p_lp1::T 
    p_val9::T 
    p_val10::T 
    p_pen2::T 
    p_pen3::T 
    p_pen4::T 
    p_tor2::T 
    p_tor3::T 
    p_tor4::T 
    p_cot2::T 
    p_vdW1::T 
    cutoff_bond_order::T  
    p_coa4::T 
    p_ovun4::T 
    p_ovun3::T 
    p_val8::T 
    p_coa3::T 
end

function GeneralParameters(; p_boc1::T, p_boc2::T, p_coa2::T,  p_trip4::T,  p_trip3::T,  k_c2::T,  p_ovun6::T,  p_trip2::T,  
                             p_ovun7::T,  p_ovun8::T,  p_trip1::T,  lower_taper_radius::T, upper_taper_radius::T, p_val6::T, 
                             p_lp1::T, p_val9::T, p_val10::T, p_pen2::T, p_pen3::T, p_pen4::T, p_tor2::T, p_tor3::T, 
                             p_tor4::T, p_cot2::T, p_vdW1::T, cutoff_bond_order::T,  p_coa4::T, p_ovun4::T, p_ovun3::T, 
                             p_val8::T, p_coa3::T) where {T}
    return GeneralParameters{T}(
        p_boc1, p_boc2, p_coa2, p_trip4, p_trip3, k_c2, p_ovun6, p_trip2, p_ovun7, 
        p_ovun8, p_trip1, lower_taper_radius, upper_taper_radius, p_val6,
        p_lp1, p_val9, p_val10, p_pen2, p_pen3, p_pen4, p_tor2, p_tor3,
        p_tor4, p_cot2, p_vdW1, cutoff_bond_order, p_coa4, p_ovun4, p_ovun3,
        p_val8, p_coa3,
    )
end


struct AtomParameters{T}
    ro_sigma::T
    Val::T
    atom_mass::T
    r_vdW::T
    D_ij::T
    gamma::T
    ro_pi::T
    Val_e::T
    alpha::T
    gamma_w::T
    Val_angle::T
    p_ovun5::T
    chi_EEM::T
    eta_EEM::T
    ro_pipi::T
    p_lp2::T
    Heat_increment::T
    p_boc4::T
    p_boc3::T
    p_boc5::T
    p_ovun2::T
    p_val3::T
    Val_boc::T
    p_val5::T
end

function AtomParameters(; ro_sigma::T, Val::T, atom_mass::T, r_vdW::T, D_ij::T, gamma::T, ro_pi::T, Val_e::T,
                          alpha::T, gamma_w::T, Val_angle::T, p_ovun5::T, chi_EEM::T, eta_EEM::T, ro_pipi::T, p_lp2::T,
                          Heat_increment::T, p_boc4::T, p_boc3::T, p_boc5::T, p_ovun2::T, p_val3::T, Val_boc::T, p_val5::T) where {T}
    return AtomParameters{T}(
        ro_sigma, Val, atom_mass, r_vdW, D_ij, gamma, ro_pi, Val_e,
        alpha, gamma_w, Val_angle, p_ovun5, chi_EEM, eta_EEM, ro_pipi, p_lp2,
        Heat_increment, p_boc4, p_boc3, p_boc5, p_ovun2, p_val3, Val_boc, p_val5,
    )
end


struct BondParameters{T}
    De_sigma::T
    De_pi::T
    De_pipi::T
    p_be1::T
    p_bo5::T
    ot_corr::T
    p_bo6::T
    p_ovun1::T
    p_be2::T
    p_bo3::T
    p_bo4::T
    p_bo1::T
    p_bo2::T
    ovcoord::T
    is_defined::Bool
end

function BondParameters(; De_sigma::T, De_pi::T, De_pipi::T, p_be1::T, p_bo5::T, ot_corr::T, p_bo6::T, p_ovun1::T,
                          p_be2::T, p_bo3::T, p_bo4::T, p_bo1::T, p_bo2::T, ovcoord::T, is_defined::Bool) where {T}
    return BondParameters{T}(
        De_sigma, De_pi, De_pipi, p_be1, p_bo5, ot_corr, p_bo6, p_ovun1,
        p_be2, p_bo3, p_bo4, p_bo1, p_bo2, ovcoord, is_defined
    )
end

function Base.zero(::Type{BondParameters{T}}) where T
    return BondParameters{T}(
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), false
    )
end


struct OffDiagonalParameters{T}
    D_ij::T
    r_vdW::T
    alpha::T
    ro_sigma::T
    ro_pi::T
    ro_pipi::T
    gamma_w_inv_pow::T
    gamma_ij_inv_pow::T
    is_defined::Bool
end

function OffDiagonalParameters(; D_ij::T, r_vdW::T, alpha::T, ro_sigma::T, ro_pi::T, ro_pipi::T, gamma_w_inv_pow::T, gamma_ij_inv_pow::T, is_defined::Bool) where {T}
    return OffDiagonalParameters{T}(
        D_ij, r_vdW, alpha, ro_sigma, ro_pi, ro_pipi, gamma_w_inv_pow, gamma_ij_inv_pow, is_defined
    )
end

function Base.zero(::Type{OffDiagonalParameters{T}}) where T
    return OffDiagonalParameters{T}(
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), false
    )
end


struct AngleEntry
    index::Int
    num_entries::Int
end

function Base.zero(::Type{AngleEntry})
    return AngleEntry(0, 0)
end

struct AngleParameters{T}
    Theta_oo::T
    p_val1::T
    p_val2::T
    p_coa1::T
    p_val7::T
    p_pen1::T
    p_val4::T
    is_defined::Bool
end

function AngleParameters(; Theta_oo::T, p_val1::T, p_val2::T, p_coa1::T, p_val7::T, p_pen1::T, p_val4::T, is_defined::Bool) where {T}
    return AngleParameters{T}(
        Theta_oo, p_val1, p_val2, p_coa1, p_val7, p_pen1, p_val4, is_defined
    )
end

function Base.zero(::Type{AngleParameters{T}}) where T
    return AngleParameters{T}(
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), false
    )
end

function flatten_angle_parameters(
    angp::Array{Union{Vector{AngleParameters{T}}, Nothing}, 3},
    ::Type{T}
) where T
    dims = size(angp)

    angle_entries = zeros(AngleEntry, dims)
    flat_list = AngleParameters{T}[]

    index = 1

    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        params = angp[i, j, k]
        if !isnothing(params) && !isempty(params)
            num_params = length(params)
            angle_entries[i, j, k] = AngleEntry(index, num_params)
            append!(flat_list, params)
            index += num_params
        end
    end

    return angle_entries, flat_list
end


struct TorsionParameters{T}
    V1::T
    V2::T
    V3::T
    p_tor1::T
    p_cot1::T
    is_defined::Bool
end

function TorsionParameters(; V1::T, V2::T, V3::T, p_tor1::T, p_cot1::T, is_defined::Bool) where {T}
    return TorsionParameters{T}(
        V1, V2, V3, p_tor1, p_cot1, is_defined
    )
end

function Base.zero(::Type{TorsionParameters{T}}) where T
    return TorsionParameters{T}(
        zero(T), zero(T), zero(T), zero(T), zero(T), false
    )
end


struct HBondParameters{T}
    r_hb::T
    p_hb1::T
    p_hb2::T
    p_hb3::T
    is_defined::Bool
end

function HBondParameters(; r_hb::T, p_hb1::T, p_hb2::T, p_hb3::T, is_defined) where {T}
    return HBondParameters{T}(
        r_hb, p_hb1, p_hb2, p_hb3, is_defined
    )
end

function Base.zero(::Type{HBondParameters{T}}) where T
    return HBondParameters{T}(
        zero(T), zero(T), zero(T), zero(T), false
    )
end


struct GlobalParameters{T}
    bo_cutoff::T            # cutoff threshold for uncorrected bond orders
    bond_dist_cutoff::T     # close neighbor list finder distance
    valence_cutoff::T       # cutoff threshold for i-j and j-k bond orders in valence angles
    valence_prod_cutoff::T  # cutoff threshold for product of i-j and j-k bond orders in valence angles
    hbond_bo_cutoff::T      # cutoff threshold for i-j bond order in hydrogen bonds
    hbond_dist_cutoff::T
end

function GlobalParameters(; bo_cutoff::T, bond_dist_cutoff::T, valence_cutoff::T, valence_prod_cutoff::T, hbond_bo_cutoff::T, hbond_dist_cutoff::T) where {T}
    return GlobalParameters{T}(
        bo_cutoff, bond_dist_cutoff, valence_cutoff, valence_prod_cutoff, hbond_bo_cutoff, hbond_dist_cutoff
    )
end


struct TaperParameters{T}
    Tap_0::T
    Tap_1::T
    Tap_2::T
    Tap_3::T
    Tap_4::T
    Tap_5::T
    Tap_6::T
    Tap_7::T
end

function TaperParameters(; Tap_0::T, Tap_1::T, Tap_2::T, Tap_3::T, Tap_4::T, Tap_5::T, Tap_6::T, Tap_7::T) where {T}
    return TaperParameters{T}(Tap_0, Tap_1, Tap_2, Tap_3, Tap_4, Tap_5, Tap_6, Tap_7)
end


struct ReaxFF2008Params{T}
    atom_map::Dict{String, Int}
    angle_parameters::Array{Union{Vector{AngleParameters{T}}, Nothing}, 3}
end


struct DeviceParams{GP, AP, BP, OP, AE, AGP, TP, HP, GLP, TAP, HD, HA}
    general_parameters::GP
    atom_parameters::AP
    bond_parameters::BP
    off_diagonal_parameters::OP
    angle_entries::AE
    angle_parameters::AGP
    torsion_parameters::TP
    h_bond_parameters::HP
    global_parameters::GLP
    taper_parameters::TAP
    h_bond_donors::HD
    h_bond_acceptors::HA
end


mutable struct ReaxFF2008State{T, VI, MI, MT, VT}
    n_atoms::Int
    sys_atid::VI
    nbrs::MI
    fill_pos::VI
    BO::MT
    BO_π::MT
    BO_ππ::MT
    coordn::VT
    Δ::VT
    Δ_boc::VT
    Δ_e::VT
    Δ_angle::VT
    Δ_lp::VT
    n_lp::VT
    energy::T
    forces::MT
    dn_lp_dΔi::VT
    dBO′_dr::MT
    dBOπ′_dr::MT
    dBOππ′_dr::MT
    dBO_dBO′::MT
    dBO_dΔi′::MT
    dBO_dΔj′::MT
    dBOπ_dBOπ′::MT
    dBOπ_dBO′::MT
    dBOπ_dΔi′::MT
    dBOπ_dΔj′::MT
    dBOππ_dBOππ′::MT
    dBOππ_dBO′::MT
    dBOππ_dΔi′::MT
    dBOππ_dΔj′::MT
    S_dE_dBO::MT
    S_dE_dBOπ::MT
    S_dE_dBOππ::MT
    S_dE_dΔ::VT
    energy_components::VT
    reverse_nbrs::MI
    Δ_e_mod2::VT
    thread_atom_energies_1::VT
    thread_atom_energies_2::VT
    thread_atom_energies_3::VT
    charges_sys_precision::VT
    valid_torsions::MI      # shape=(4, capacity) containing j, idx_k, idx_i, idx_l
    n_valid_torsions::VI    # 1-element device vector
    valid_angles::MI        # shape=(4, capacity) containing j, idx_i, idx_k, angle parameter index   
    n_valid_angles::VI      # 1-element device vector
    valid_hbonds::MI        # shape=(3, capacity) containing j, k, idx_i
    n_valid_hbonds::VI      # 1-element device vector
end


function empty_state(
    ::Type{T},
    ::Type{AT},
) where {T<:AbstractFloat, AT}

    VI = AT{Int, 1}
    MI = AT{Int, 2}
    MT = AT{T, 2}
    VT = AT{T, 1}

    return ReaxFF2008State(
        0,
        VI(undef, 0),
        MI(undef, 0, 0),
        VI(undef, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        VT(undef, 0), 
        VT(undef, 0),
        VT(undef, 0),
        VT(undef, 0),
        VT(undef, 0),
        VT(undef, 0),
        VT(undef, 0),
        zero(T),
        MT(undef, 0, 0),
        VT(undef, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        MT(undef, 0, 0),
        VT(undef, 0),
        VT(undef, 0),
        MI(undef, 0, 0),
        VT(undef, 0),
        VT(undef, 0),
        VT(undef, 0),
        VT(undef, 0),
        VT(undef, 0),
        MI(undef, 4, 0),
        VI(undef, 1),
        MI(undef, 4, 0),
        VI(undef, 1),
        MI(undef, 3, 0),
        VI(undef, 1)
    )
end


"""
    ReaxFF(; force_field_file, version=ReaxFF2008(), float_type=Float64,
            array_type=Array, smoothness_correction=false,
            bond_dist_cutoff=0.5, hbond_dist_cutoff=0.75,
            valence_cutoff=0.001, valence_prod_cutoff=0.00001,
            hbond_bo_cutoff=0.01)

Reactive force field (ReaxFF) implemented as an AtomsCalculators.jl calculator.

`force_field_file` specifies the path to a ReaxFF parameter file. `version` selects
the functional form; currently only [`ReaxFF2008`](@ref) is supported.
`float_type` sets the precision of the force-field parameters and internal ReaxFF calculations (Float64 strongly recommended).
`array_type` sets CPU (`Array`) or GPU storage and the device to run on, e.g. `CuArray` after loading CUDA.jl.

`bond_dist_cutoff` and `hbond_dist_cutoff` set the distance cutoffs for
bond construction and hydrogen-bond interactions, respectively. They accept
lengths with units; unitless values are interpreted as nanometers.

`valence_cutoff` sets the bond-order threshold for angle and torsion terms.
`valence_prod_cutoff` sets the bond-order product threshold for angle terms
when `smoothness_correction=false`.
`hbond_bo_cutoff` sets the donor–hydrogen bond-order threshold for hydrogen bonds.
These three thresholds are dimensionless.

`smoothness_correction=true` enables the Furman-Wales smoothness
corrections to bond orders and energy terms ([Furman and Wales 2019](https://doi.org/10.1021/acs.jpclett.9b02810)).
The default is `false`.

Use `System(coord_file_path, reaxff; ...)` to set up coordinates, neighbor finding,
and the associated [`EEM`](@ref) charge equilibration automatically. Use a fresh
`ReaxFF` instance for each system, since it stores a system-specific cached state.

This interaction is only compatible with 3D systems. The virial is not yet supported.
"""
mutable struct ReaxFF{V<:AbstractReaxFFVersion, T, AT, DP, S, SC}
    version::V
    params::ReaxFF2008Params{T}
    device_params::DP
    state::S
    use_units::Union{Bool, Nothing}
    sys_coord_unit::Union{Unitful.Units, Nothing}
    system_assigned::Bool
    smoothness_correction::Val{SC}
end


"""
    System(coord_file_path::AbstractString, reaxff::ReaxFF; eem_tolerance=1e-6,
           eem_max_iterations=200, eem_float_type=Float64, initial_charges=nothing,
           neighbor_skin=0.0, neighbor_refresh_n_steps=1, boundary=nothing,
           array_type=Array, velocities=nothing, units=true, loggers=())

Construct a ReaxFF system from a coordinate file supported by Chemfiles.jl.
The file must identify each atom's element, and every element must be present in
the force field file. This constructor adds both [`EEM`](@ref) and [`ReaxFF`](@ref) to
the system's general interactions and selects a neighbor finder automatically.

`eem_float_type` sets the charge-equilibration precision independently of the
ReaxFF precision (Float64 recommended). `eem_tolerance` and `eem_max_iterations` control convergence;
failure to converge raises an error. `initial_charges` contains the initial
atomic partial charges in units of elementary charge and defaults to zeros. Its sum sets
the total system charge that is conserved by charge equilibration.

`neighbor_skin` is a non-negative distance added to the largest ReaxFF distance
cutoff. It accepts lengths with units; unitless values are interpreted as
nanometers, regardless of `units`. `neighbor_refresh_n_steps`
sets the scheduled neighbor finder rebuild interval and must be at least one. An interval
greater than one requires a positive skin distance. Select a skin large enough to cover
atomic motion between rebuilds.

`array_type` must match that specified in the ReaxFF instance. The boundary is read from the file
unless `boundary` is supplied, and velocities default to zero. Supplied boundaries
and velocities must match the ReaxFF floating-point type and the `units` setting.
With `units=true`, quantities carry units; with `units=false`, numerical values
use nm, ps, g/mol, and kJ/mol units. `loggers` are passed to the system unchanged.

Use a fresh `ReaxFF` instance for each system, since it stores system-specific
cached state.
"""
function System(coord_file_path::AbstractString,
                reaxff::ReaxFF{<:AbstractReaxFFVersion, T, ATR};
                eem_tolerance=1e-6,
                eem_max_iterations=200,
                eem_float_type::Type{QT}=Float64,
                initial_charges=nothing,
                neighbor_skin=0.0,
                neighbor_refresh_n_steps::Integer=1,
                boundary=nothing,
                array_type::Type{AT}=Array,
                velocities=nothing,
                units::Bool=true,
                loggers=(),
) where {T<:AbstractFloat, AT, ATR, QT<:AbstractFloat}
    if AT != ATR
        throw(ArgumentError("Array Type Mismatch: ReaxFF object specified array_type $ATR but trying to construct System with array_type $AT"))
    end

    trajectory = Chemfiles.Trajectory(coord_file_path)
    previous_callback = Chemfiles.WARNING_CALLBACK
    frame = try
        Chemfiles.set_warning_callback() do message
            startswith(message, "PDB reader: found unexpected, non-standard atom") && return
            previous_callback(message)
        end
        Chemfiles.read(trajectory)
    finally
        Chemfiles.set_warning_callback(previous_callback)
    end

    n_atoms = length(frame)

    atoms = Atom[]
    atoms_data = AtomData[]

    if units
        coords = [T.(SVector{3}(col)u"nm" / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    else
        coords = [T.(SVector{3}(col) / 10.0) for col in eachcol(Chemfiles.positions(frame))]
    end

    # QEq float precision
    initial_q = if isnothing(initial_charges)
        fill(zero(QT), n_atoms)
    else
        q_cpu = from_device(initial_charges)
        if length(q_cpu) != n_atoms
            throw(ArgumentError("Length of initial_charges must equal the number of atoms"))
        end
        QT.(q_cpu) 
    end

    at_map = reaxff.params.atom_map
    atmp = from_device(reaxff.device_params.atom_parameters)
    for at_index in 1:n_atoms
        atom = Chemfiles.Atom(frame, at_index-1)
        atom_type = Chemfiles.type(atom)
        if !haskey(at_map, atom_type)
            atom_type = uppercasefirst(lowercase(atom_type))
        end
        if !haskey(at_map, atom_type)
            throw(ArgumentError("Atom Not Found: atom <$atom_type> in coord file not defined in ReaxFF force field file"))
        end

        atom_mass = T(atmp[at_map[atom_type]].atom_mass)
        if units
            atom_mass = atom_mass*u"g/mol"
        end
        
        push!(atoms, Atom(
            index=at_index, 
            mass=atom_mass, 
            charge=T(initial_q[at_index])   # store charge in System float precision
        ))
        push!(atoms_data, AtomData(element=atom_type))
    end

    atoms = to_device([atoms...], AT)
    coords = to_device(coords, AT)

    if isnothing(boundary)
        boundary_used = boundary_from_chemfiles(Chemfiles.UnitCell(frame), T,
                                                (units ? u"nm" : NoUnits))
    else
        if float_type(boundary) != T
            throw(ArgumentError("Float Type Mismatch: float type of boundary does not match float type T specified when constructing ReaxFF instance")) 
        end

        if units
            if !(length_type(boundary) <: Unitful.Quantity)
                throw(ArgumentError("Unit Mismatch: units set to true but boundary is unitless"))
            end
        else
            if length_type(boundary) <: Unitful.Quantity
                throw(ArgumentError("Unit Mismatch: units set to false but boundary has units"))
            end
        end
        boundary_used = boundary
    end

    if isnothing(velocities)
        if units
            vels = zero(ustrip_vec.(coords))u"nm * ps^-1"
        else
            vels = zero(coords)
        end
    else
        if typeof(one(eltype(eltype(velocities)))) != T
            throw(ArgumentError("Float Type Mismatch: float type of velocities does not match float type T specified when constructing ReaxFF instance")) 
        end

        if units
            if !(eltype(eltype(velocities)) <: Unitful.Quantity)
                throw(ArgumentError("Unit Mismatch: units set to true but velocities are unitless"))
            end
        else
            if eltype(eltype(velocities)) <: Unitful.Quantity
                throw(ArgumentError("Unit Mismatch: units set to false but velocities have units"))
            end
        end

        vels = to_device(velocities, AT)
    end

    # create neighbor finder
    neighbor_skin_angstrom = if unit(neighbor_skin) == NoUnits
        10 * neighbor_skin
    else
        ustrip(u"Å", neighbor_skin)
    end

    if neighbor_skin_angstrom < zero(neighbor_skin_angstrom)
        throw(ArgumentError("neighbor_skin must be non-negative"))
    end

    if neighbor_refresh_n_steps < 1
        throw(ArgumentError("neighbor_refresh_n_steps must be at least 1"))
    end

    if neighbor_refresh_n_steps > 1 && iszero(neighbor_skin_angstrom)
        throw(ArgumentError("neighbor_refresh_n_steps > 1 requires a positive neighbor skin"))
    end

    required_cutoff = max(
        reaxff.device_params.global_parameters.bond_dist_cutoff,
        reaxff.device_params.global_parameters.hbond_dist_cutoff,
        reaxff.device_params.general_parameters.upper_taper_radius
    ) 
    check_cutoff_box_size(
        units ? required_cutoff * u"Å" : required_cutoff / T(10),
        boundary_used,
        default_strictness(),
    )
    neighbor_list_cutoff = required_cutoff + T(neighbor_skin_angstrom)

    if units
        far_dist_cutoff = uconvert(u"nm", neighbor_list_cutoff*u"Å")
    else
        # assume unitless coords in nm
        far_dist_cutoff = neighbor_list_cutoff / T(10.0)
    end

    backend = get_backend(coords)
    
    nf = if backend isa KernelAbstractions.GPU && gpu_cell_list_suitable(boundary_used, far_dist_cutoff; force_float32=true)
        GPUCellListNeighborFinder(
            n_steps = neighbor_refresh_n_steps,
            force_float32 = true,
            dist_cutoff = far_dist_cutoff,
            output = :geometric_pairs,
        )
    elseif backend isa KernelAbstractions.GPU
        DistanceNeighborFinder(
            n_atoms = n_atoms,
            array_type = AT,
            n_steps = neighbor_refresh_n_steps,
            dist_cutoff = far_dist_cutoff
        )
    else
        # CellListNeighborFinder requires perpendicualr distance between cell planes to be greater than twice the cutoff
        # If the boundary is too small, use DistanceNeighborFinder on CPU instead
        eligible = to_device(trues(n_atoms, n_atoms), AT)
        if minimum(cell_list_box_widths(boundary_used)) > 2 * far_dist_cutoff
            finder = CellListMapNeighborFinder(
                eligible = eligible,
                n_steps = neighbor_refresh_n_steps,
                dist_cutoff = far_dist_cutoff,
                boundary = boundary_used,
                x0 = coords
            )
            finder
        else
            DistanceNeighborFinder(
                eligible = eligible,
                n_steps = neighbor_refresh_n_steps,
                dist_cutoff = far_dist_cutoff
            )
        end
    end

    # create charge equilibration object
    eem = EEM(
        reaxff, 
        atoms_data; 
        charges=initial_q,  # QEq float precision
        tolerance=QT(eem_tolerance), 
        eem_float_type=eem_float_type,
        max_iterations=eem_max_iterations
    )

    sys = System(
        atoms=atoms,
        atoms_data=atoms_data,
        coords=coords,
        boundary=boundary_used,
        velocities=vels,
        general_inters=(eem, reaxff),
        neighbor_finder=nf,
        loggers=loggers,
        force_units=(units ? u"kJ * mol^-1 * nm^-1" : NoUnits),
        energy_units=(units ? u"kJ * mol^-1" : NoUnits),
    )

    if reaxff.system_assigned
        @warn "This ReaxFF instance has already been assigned to another system. " *
              "Create a separate ReaxFF instance for each system to avoid using incorrect cached state."
    end
    reaxff.system_assigned = true

    return sys
end


function _set_force_field_file_reaxff_params(
    file::AbstractString,
    ::Type{T},
    ::Type{AT};
    bond_dist_cutoff,
    hbond_dist_cutoff,
    valence_cutoff,
    valence_prod_cutoff,
    hbond_bo_cutoff,
) where {T<:AbstractFloat, AT}

    open(file, "r") do f
        readline(f)     # Comment line, ignored

        # General parameters block
        l = strip(readline(f))
        items = split(l)

        num_general_params = items[1]
        if num_general_params != "39"
            throw(ArgumentError("Number of general parameters must be 39"))
        end

        unused_general = [14, 19, 23, 27, 35, 36, 37, 38]     # indices of general parameters that are not used

        genp_tmp = T[]

        for i in 1:39
            l = strip(readline(f))

            if i in unused_general
                continue
            end

            items = split(l)
            val = parse(T, items[1])
            push!(genp_tmp, val)
        end

        general_parameters = GeneralParameters{T}(genp_tmp...)

        # Atom parameters block
        l = strip(readline(f))
        if !occursin("Nr of atoms", l)
            throw(ArgumentError("Expected 'Nr of atoms' header, got: $l"))
        end

        num_atom_types = parse(Int, split(l)[1])

        for _ in 1:3        # skip column headers
            readline(f)
        end

        unused_atom = [13, 16, 23, 24, 27, 30, 31, 32]

        atom_map = Dict{String, Int}()

        atom_parameters = AtomParameters{T}[]
        atmp_tmp = T[]

        for i in 1:num_atom_types
            l = strip(readline(f))     # first line contains atom name followed by 8 parameters
            items = split(l)
            if length(items) != 9
                throw(ArgumentError("Expected atom name followed by 8 parameters, got: $l"))
            end

            atom_id = items[1]
            atom_map[atom_id] = i
            empty!(atmp_tmp)

            counter = 0

            for param in items[2:end]
                counter += 1
                p = parse(T, param)
                push!(atmp_tmp, p)
            end

            for _ in 1:3
                l = strip(readline(f))
                items = split(l)
                if length(items) != 8
                    throw(ArgumentError("Expected 8 atom parameters, got: $l"))
                end

                for param in items
                    counter += 1

                    if counter in unused_atom
                        continue
                    end

                    p = parse(T, param)
                    push!(atmp_tmp, p)
                end
            end

            atmp = AtomParameters{T}(atmp_tmp...)
            push!(atom_parameters, atmp)
        end


        # Bond parameters block
        l = strip(readline(f))
        if !occursin("Nr of bonds", l)
            throw(ArgumentError("Expected 'Nr of bonds' header, got: $l"))
        end

        num_bonds = parse(Int, split(l)[1])

        readline(f) # skip column headers

        unused_bond = [12, 16]
        bond_parameters = fill(zero(BondParameters{T}), num_atom_types, num_atom_types)
        bndp_tmp = T[]

        for _ in 1:num_bonds
            l = strip(readline(f))     # first line contains at1, at2, followed by 8 parameters
            items = split(l)
            if length(items) != 10
                throw(ArgumentError("Expected 2 atom types followed by 8 bond parameters, got: $l"))
            end
            empty!(bndp_tmp)

            at_1 = parse(Int, items[1])
            at_2 = parse(Int, items[2])

            counter = 0

            for param in items[3:end]
                counter += 1

                if counter in unused_bond
                    continue
                end

                p = parse(T, param)
                push!(bndp_tmp, p)
            end

            l = strip(readline(f))
            items = split(l)
            if length(items) != 8  # second line contains 8 parameters
                throw(ArgumentError("Expected 8 bond parameters, got: $l"))
            end

            for param in items
                counter += 1

                if counter in unused_bond
                    continue
                end

                p = parse(T, param)
                push!(bndp_tmp, p)
            end

            push!(bndp_tmp, true)
            bndp = BondParameters{T}(bndp_tmp...)
            bond_parameters[at_1, at_2] = bndp
            bond_parameters[at_2, at_1] = bndp
        end


        # Off diagonal terms parameters block is optional
        # Next block could be either off-diagonal terms OR angle terms
        l = strip(readline(f))

        off_diagonal_parameters = fill(zero(OffDiagonalParameters{T}), num_atom_types, num_atom_types)  # empty dict if off diagonal params block missing
        ofdp_tmp = T[]

        if occursin("Nr of off-diagonal terms", l)
            num_off_diagonal = parse(Int, split(l)[1])

            for _ in 1:num_off_diagonal
                l = strip(readline(f))     # each line contains at1, at2, followed by 6 parameters
                items = split(l)
                if length(items) != 8
                    throw(ArgumentError("Expected 2 atom types followed by 6 off-diagonal parameters, got: $l"))
                end
                empty!(ofdp_tmp)

                at_1 = parse(Int, items[1])
                at_2 = parse(Int, items[2])

                atmp_i = atom_parameters[at_1]
                atmp_j = atom_parameters[at_2]

                mixed_values = (
                    sqrt(atmp_i.D_ij * atmp_j.D_ij),
                    sqrt(atmp_i.r_vdW * atmp_j.r_vdW),
                    sqrt(atmp_i.alpha * atmp_j.alpha),
                    T(0.5) * (atmp_i.ro_sigma + atmp_j.ro_sigma),
                    T(0.5) * (atmp_i.ro_pi + atmp_j.ro_pi),
                    T(0.5) * (atmp_i.ro_pipi + atmp_j.ro_pipi),
                )

                for (param, mixed_value) in zip(items[3:end], mixed_values)
                    p = parse(T, param)
                    push!(ofdp_tmp, p > zero(T) ? p : mixed_value)
                end

                gamma_w_inv_pow = (one(T) / sqrt(atom_parameters[at_1].gamma_w * atom_parameters[at_2].gamma_w))^general_parameters.p_vdW1
                push!(ofdp_tmp, gamma_w_inv_pow)

                gamma_prod = atom_parameters[at_1].gamma * atom_parameters[at_2].gamma
                gamma_ij_inv_pow = one(T) / (gamma_prod * sqrt(gamma_prod))
                push!(ofdp_tmp, gamma_ij_inv_pow)

                push!(ofdp_tmp, true)
                ofdp = OffDiagonalParameters{T}(ofdp_tmp...)
                off_diagonal_parameters[at_1, at_2] = ofdp
                off_diagonal_parameters[at_2, at_1] = ofdp
            end

            l = strip(readline(f))      # read first line of angle parameters block
        end

        for at_1 in 1:num_atom_types
            for at_2 in 1:num_atom_types
                if !off_diagonal_parameters[at_1, at_2].is_defined
                    empty!(ofdp_tmp)
                    push!(ofdp_tmp, sqrt(atom_parameters[at_1].D_ij * atom_parameters[at_2].D_ij))
                    push!(ofdp_tmp, sqrt(atom_parameters[at_1].r_vdW * atom_parameters[at_2].r_vdW))
                    push!(ofdp_tmp, sqrt(atom_parameters[at_1].alpha * atom_parameters[at_2].alpha))
                    push!(ofdp_tmp, T(0.5) * (atom_parameters[at_1].ro_sigma + atom_parameters[at_2].ro_sigma))
                    push!(ofdp_tmp, T(0.5) * (atom_parameters[at_1].ro_pi + atom_parameters[at_2].ro_pi))
                    push!(ofdp_tmp, T(0.5) * (atom_parameters[at_1].ro_pipi + atom_parameters[at_2].ro_pipi))

                    gamma_w_inv_pow = (one(T) / sqrt(atom_parameters[at_1].gamma_w * atom_parameters[at_2].gamma_w))^general_parameters.p_vdW1
                    push!(ofdp_tmp, gamma_w_inv_pow)

                    gamma_prod = atom_parameters[at_1].gamma * atom_parameters[at_2].gamma
                    gamma_ij_inv_pow = one(T) / (gamma_prod * sqrt(gamma_prod))
                    push!(ofdp_tmp, gamma_ij_inv_pow)

                    push!(ofdp_tmp, true)
                    ofdp = OffDiagonalParameters{T}(ofdp_tmp...)
                    off_diagonal_parameters[at_1, at_2] = ofdp
                    off_diagonal_parameters[at_2, at_1] = ofdp
                end
            end
        end

        # Angle parameters block
        if !occursin("Nr of angles", l)
            throw(ArgumentError("Expected 'Nr of angles' header, got: $l"))
        end

        num_angles = parse(Int, split(l)[1])

        angle_parameters = Array{Union{Vector{AngleParameters{T}}, Nothing}}(nothing, num_atom_types, num_atom_types, num_atom_types)
        angp_tmp = T[]

        for _ in 1:num_angles
            l = strip(readline(f))     # each line contains at1, at2, at3, followed by 7 parameters
            items = split(l)
            if length(items) != 10
                throw(ArgumentError("Expected 3 atom types followed by 7 angle parameters, got: $l"))
            end
            empty!(angp_tmp)

            at_1 = parse(Int, items[1])
            at_2 = parse(Int, items[2])
            at_3 = parse(Int, items[3])

            for param in items[4:end]
                p = parse(T, param)
                push!(angp_tmp, p)
            end

            push!(angp_tmp, true)
            angp = AngleParameters{T}(angp_tmp...)

            if isnothing(angle_parameters[at_1, at_2, at_3])
                angle_parameters[at_1, at_2, at_3] = AngleParameters{T}[]
                angle_parameters[at_3, at_2, at_1] = AngleParameters{T}[]
            end

            push!(angle_parameters[at_1, at_2, at_3], angp)
            if at_1 != at_3
                push!(angle_parameters[at_3, at_2, at_1], angp)
            end
        end


        # Torsion terms parameters block
        l = strip(readline(f))
        if !occursin("Nr of torsions", l)
            throw(ArgumentError("Expected 'Nr of torsions' header, got: $l"))
        end

        num_torsions = parse(Int, split(l)[1])

        torsion_parameters = fill(zero(TorsionParameters{T}), num_atom_types, num_atom_types, num_atom_types, num_atom_types)
        torp_tmp = T[]

        seen = zeros(Int, num_atom_types, num_atom_types, num_atom_types, num_atom_types)

        for _ in 1:num_torsions
            l = strip(readline(f))     # each line contains at1, at2, at3, at4, followed by 5 parameters, followed by 2 unused parameters
            items = split(l)
            if length(items) != 11
                throw(ArgumentError("Expected 11 fields in torsion parameter line, got: $l"))
            end
            empty!(torp_tmp)

            at_1 = parse(Int, items[1])
            at_2 = parse(Int, items[2])
            at_3 = parse(Int, items[3])
            at_4 = parse(Int, items[4])

            if at_2 <= 0 || at_3 <= 0
                throw(ArgumentError("Central atom type indices in torsion entry must be positive, got: $l"))
            end

            for param in items[5:end-2]
                p = parse(T, param)
                push!(torp_tmp, p)
            end

            push!(torp_tmp, true)
            torp = TorsionParameters{T}(torp_tmp...)

            if at_1 > 0 && at_4 > 0         # All atoms types defined, store forward and reverse orders
                seen[at_1, at_2, at_3, at_4] = 1
                seen[at_4, at_3, at_2, at_1] = 1

                torsion_parameters[at_1, at_2, at_3, at_4] = torp
                torsion_parameters[at_4, at_3, at_2, at_1] = torp

            else        # First and last atoms undefined (0), loop over all possible pairs and store if not already defined previously
                for z in 1:num_atom_types
                    for w in 1:num_atom_types
                        if seen[w, at_2, at_3, z] == 0
                            seen[w, at_2, at_3, z] = 1
                            torsion_parameters[w, at_2, at_3, z] = torp
                        end

                        if seen[z, at_3, at_2, w] == 0
                            seen[z, at_3, at_2, w] = 1
                            torsion_parameters[z, at_3, at_2, w] = torp
                        end
                    end
                end
            end
        end


        # Hydrogen bonds terms parameters block
        # This block is optional => check if end-of-file reached
        h_bond_parameters = fill(zero(HBondParameters{T}), num_atom_types, num_atom_types, num_atom_types)
        hbdp_tmp = T[]
        h_bond_donors = zeros(Bool, num_atom_types)
        h_bond_acceptors = zeros(Bool, num_atom_types)

        if !eof(f)
            l = strip(readline(f))
            if !isempty(l)
                if !occursin("Nr of hydrogen bonds", l)
                    throw(ArgumentError("Expected 'Nr of hydrogen bonds' header, got: $l"))
                end

                num_h_bonds = parse(Int, split(l)[1])

                for _ in 1:num_h_bonds
                    l = strip(readline(f))     # each line contains at1, at2, at3, followed by 4 parameters
                    items = split(l)
                    if length(items) != 7
                        throw(ArgumentError("Expected 3 atom types followed by 4 hydrogen bond parameters, got: $l"))
                    end
                    empty!(hbdp_tmp)

                    at_1 = parse(Int, items[1])
                    at_2 = parse(Int, items[2])
                    at_3 = parse(Int, items[3])

                    if !(0 < at_1 <= num_atom_types && 0 < at_2 <= num_atom_types && 0 < at_3 <= num_atom_types)
                        throw(ArgumentError("Hydrogen bond atom type indices must be in 1:$num_atom_types, got: $l"))
                    end

                    h_bond_donors[at_1] = true
                    h_bond_acceptors[at_3] = true

                    for param in items[4:end]
                        p = parse(T, param)
                        push!(hbdp_tmp, p)
                    end

                    push!(hbdp_tmp, true)
                    hbdp = HBondParameters{T}(hbdp_tmp...)
                    h_bond_parameters[at_1, at_2, at_3] = hbdp
                end
            else
                @warn "No ReaxFF hydrogen bond parameters specified"
            end
        else
            @warn "No ReaxFF hydrogen bond parameters specified"
        end

        if !eof(f) && !isempty(strip(readline(f)))
            throw(ArgumentError("Unexpected content after ReaxFF force field parameters"))
        end

        # Global cutoff parameters
        global_parameters = GlobalParameters(
            bo_cutoff = general_parameters.cutoff_bond_order * T(0.01),
            bond_dist_cutoff = T(bond_dist_cutoff),
            valence_cutoff = T(valence_cutoff),
            valence_prod_cutoff = T(valence_prod_cutoff),
            hbond_bo_cutoff = T(hbond_bo_cutoff),
            hbond_dist_cutoff = T(hbond_dist_cutoff),
        )

        # Calculate taper correction terms
        R_cut = general_parameters.upper_taper_radius

        Tap = [one(T), zero(T), zero(T), zero(T)]  # Tap_0 through Tap_3

        R_cut_pow = R_cut^4
        push!(Tap, T(-35.0) / R_cut_pow)
        R_cut_pow *= R_cut
        push!(Tap, T(84.0) / R_cut_pow)
        R_cut_pow *= R_cut
        push!(Tap, T(-70.0) / R_cut_pow)
        R_cut_pow *= R_cut
        push!(Tap, T(20.0) / R_cut_pow)

        taper_parameters = TaperParameters{T}(Tap...)

        # Create combined ReaxFF parameters struct
        params = ReaxFF2008Params{T}(
            atom_map,
            angle_parameters
        )

        angle_entries, angle_parameters_flat = flatten_angle_parameters(angle_parameters, T)

        device_params = DeviceParams(
            general_parameters,
            to_device(atom_parameters, AT),
            to_device(bond_parameters, AT),
            to_device(off_diagonal_parameters, AT),
            to_device(angle_entries, AT),
            to_device(angle_parameters_flat, AT),
            to_device(torsion_parameters, AT),
            to_device(h_bond_parameters, AT),
            global_parameters,
            taper_parameters,
            to_device(h_bond_donors, AT),
            to_device(h_bond_acceptors, AT)
        )

        return params, device_params
    end
end


function ReaxFF(; version::V=ReaxFF2008(), 
                  force_field_file::AbstractString,
                  float_type::Type{T}=Float64,
                  array_type::Type{AT}=Array,
                  smoothness_correction::Bool=false,
                  bond_dist_cutoff=0.5,
                  hbond_dist_cutoff=0.75,
                  valence_cutoff=0.001,
                  valence_prod_cutoff=0.00001,
                  hbond_bo_cutoff=0.01,
) where {V<:AbstractReaxFFVersion, T<:AbstractFloat, AT}

    bond_dist_cutoff = T(unit(bond_dist_cutoff) == NoUnits ? 10 * bond_dist_cutoff : ustrip(u"Å", bond_dist_cutoff))
    hbond_dist_cutoff = T(unit(hbond_dist_cutoff) == NoUnits ? 10 * hbond_dist_cutoff : ustrip(u"Å", hbond_dist_cutoff))
    
    params, device_params = _set_force_field_file_reaxff_params(
        force_field_file, T, AT;
        bond_dist_cutoff, hbond_dist_cutoff,
        valence_cutoff, valence_prod_cutoff, hbond_bo_cutoff,
    )

    state = empty_state(T, AT) 

    DP = typeof(device_params)
    S = typeof(state)

    inter = ReaxFF{V, T, AT, DP, S, smoothness_correction}(
        version, params, device_params, state, nothing, nothing, false, Val(smoothness_correction),
    )

    return inter
end


function _init_reaxff_state!(
    inter::ReaxFF{ReaxFF2008, T, AT},
    sys::System
) where {T, AT}
    n_atoms = length(sys.atoms)

    sys_atid = zeros(Int, n_atoms)

    at_map = inter.params.atom_map

    for (idx, atom_data) in enumerate(sys.atoms_data)
        element = atom_data.element
        atid_i = at_map[element]
        sys_atid[idx] = atid_i
    end

    bond_dist_cutoff = inter.device_params.global_parameters.bond_dist_cutoff

    volume_near = 4.0 * π * bond_dist_cutoff^3 / 3.0

    max_density = 0.20      # number of atoms per Å^3
    buffer_factor = 1.1

    near_nbrs_max_est = min(n_atoms-1, ceil(Int, volume_near * max_density * buffer_factor))

    nbrs = zeros(Int, near_nbrs_max_est, n_atoms)
    fill_pos = zeros(Int, n_atoms)

    BO = zeros(T, near_nbrs_max_est, n_atoms)
    BO_π = zeros(T, near_nbrs_max_est, n_atoms)
    BO_ππ = zeros(T, near_nbrs_max_est, n_atoms)
    
    dBO′_dr = zeros(T, near_nbrs_max_est, n_atoms)
    dBOπ′_dr = zeros(T, near_nbrs_max_est, n_atoms)
    dBOππ′_dr = zeros(T, near_nbrs_max_est, n_atoms)

    coordn = zeros(T, n_atoms)
    Δ = zeros(T, n_atoms)
    Δ_boc = zeros(T, n_atoms)
    Δ_e = zeros(T, n_atoms)
    Δ_angle = zeros(T, n_atoms)
    Δ_lp = zeros(T, n_atoms)
    n_lp = zeros(T, n_atoms)

    energy = zero(T)
    forces = zeros(T, 3, n_atoms)    

    dn_lp_dΔi = zeros(T, n_atoms)

    dBO_dBO′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBO_dΔi′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBO_dΔj′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOπ_dBOπ′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOπ_dBO′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOπ_dΔi′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOπ_dΔj′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOππ_dBOππ′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOππ_dBO′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOππ_dΔi′ = zeros(T, near_nbrs_max_est, n_atoms)
    dBOππ_dΔj′ = zeros(T, near_nbrs_max_est, n_atoms)

    S_dE_dBO = zeros(T, near_nbrs_max_est, n_atoms)
    S_dE_dBOπ = zeros(T, near_nbrs_max_est, n_atoms)
    S_dE_dBOππ = zeros(T, near_nbrs_max_est, n_atoms)
    S_dE_dΔ = zeros(T, n_atoms)

    energy_components = zeros(T, 13)

    reverse_nbrs = zeros(Int, near_nbrs_max_est, n_atoms)

    Δ_e_mod2 = zeros(T, n_atoms)

    thread_atom_energies_1 = zeros(T, n_atoms)
    thread_atom_energies_2 = zeros(T, n_atoms)
    thread_atom_energies_3 = zeros(T, n_atoms)

    charges_sys_precision = zeros(T, n_atoms)

    valid_torsions = zeros(Int, 4, max(10 * n_atoms, 1))    # 10*n_atoms is initial guess for capacity, but size can grow later
    n_valid_torsions = zeros(Int, 1)
    valid_angles = zeros(Int, 4, max(10 * n_atoms, 1))
    n_valid_angles = zeros(Int, 1)
    valid_hbonds = zeros(Int, 3, max(10 * n_atoms, 1))
    n_valid_hbonds = zeros(Int, 1)

    inter.state = ReaxFF2008State(
        n_atoms,
        to_device(sys_atid, AT),
        to_device(nbrs, AT),
        to_device(fill_pos, AT),
        to_device(BO, AT),
        to_device(BO_π, AT),
        to_device(BO_ππ, AT),
        to_device(coordn, AT),
        to_device(Δ, AT),
        to_device(Δ_boc, AT),
        to_device(Δ_e, AT),
        to_device(Δ_angle, AT),
        to_device(Δ_lp, AT),
        to_device(n_lp, AT),
        energy,
        to_device(forces, AT),
        to_device(dn_lp_dΔi, AT),
        to_device(dBO′_dr, AT), 
        to_device(dBOπ′_dr, AT), 
        to_device(dBOππ′_dr, AT),
        to_device(dBO_dBO′, AT),
        to_device(dBO_dΔi′, AT),
        to_device(dBO_dΔj′, AT),
        to_device(dBOπ_dBOπ′, AT),
        to_device(dBOπ_dBO′, AT),
        to_device(dBOπ_dΔi′, AT),
        to_device(dBOπ_dΔj′, AT),
        to_device(dBOππ_dBOππ′, AT),
        to_device(dBOππ_dBO′, AT),
        to_device(dBOππ_dΔi′, AT),
        to_device(dBOππ_dΔj′, AT),
        to_device(S_dE_dBO, AT),
        to_device(S_dE_dBOπ, AT),
        to_device(S_dE_dBOππ, AT),
        to_device(S_dE_dΔ, AT),
        to_device(energy_components, AT),
        to_device(reverse_nbrs, AT),
        to_device(Δ_e_mod2, AT),
        to_device(thread_atom_energies_1, AT),
        to_device(thread_atom_energies_2, AT),
        to_device(thread_atom_energies_3, AT),
        to_device(charges_sys_precision, AT),
        to_device(valid_torsions, AT),
        to_device(n_valid_torsions, AT),
        to_device(valid_angles, AT),
        to_device(n_valid_angles, AT),
        to_device(valid_hbonds, AT),
        to_device(n_valid_hbonds, AT)
    )
end


function _clear_reaxff_state!(
    inter::ReaxFF{ReaxFF2008, T}
) where T
    fill!(inter.state.fill_pos, 0)
    fill!(inter.state.coordn, zero(T))
    fill!(inter.state.forces, zero(T))
    fill!(inter.state.S_dE_dBO, zero(T))
    fill!(inter.state.S_dE_dBOπ, zero(T))
    fill!(inter.state.S_dE_dBOππ, zero(T))
    fill!(inter.state.S_dE_dΔ, zero(T))
    fill!(inter.state.energy_components, zero(T))
    
    inter.state.energy = zero(T)
end


@inline function write_force_BO(
    coords,
    boundary,
    i,
    j,
    idx_j,
    dE_dBO,
    dE_dBOπ,
    dE_dBOππ,
    derivatives,
    forces,
    nbrs,
    fill_pos
)
    dBO_dBO′ = derivatives.dBO_dBO′
    dBO_dΔi′ = derivatives.dBO_dΔi′
    dBO_dΔj′ = derivatives.dBO_dΔj′
    dBO′_dr = derivatives.dBO′_dr

    dBOπ_dBOπ′ = derivatives.dBOπ_dBOπ′
    dBOπ_dBO′ = derivatives.dBOπ_dBO′
    dBOπ_dΔi′ = derivatives.dBOπ_dΔi′
    dBOπ_dΔj′ = derivatives.dBOπ_dΔj′
    dBOπ′_dr = derivatives.dBOπ′_dr

    dBOππ_dBOππ′ = derivatives.dBOππ_dBOππ′
    dBOππ_dBO′ = derivatives.dBOππ_dBO′
    dBOππ_dΔi′ = derivatives.dBOππ_dΔi′
    dBOππ_dΔj′ = derivatives.dBOππ_dΔj′
    dBOππ′_dr = derivatives.dBOππ′_dr

    r_ij_norm = normalize(vector(coords[j], coords[i], boundary))

    coef_BO = dBO_dBO′[idx_j, i] * dBO′_dr[idx_j, i]
    coef_BOπ = dBOπ_dBOπ′[idx_j, i] * dBOπ′_dr[idx_j, i] +
               dBOπ_dBO′[idx_j, i] * dBO′_dr[idx_j, i]
    coef_BOππ = dBOππ_dBOππ′[idx_j, i] * dBOππ′_dr[idx_j, i] +
             dBOππ_dBO′[idx_j, i] * dBO′_dr[idx_j, i]

    dE_dBO_tot = dE_dBO * coef_BO + dE_dBOπ * coef_BOπ + dE_dBOππ * coef_BOππ
    force_tmp = dE_dBO_tot .* r_ij_norm

    for n in 1:3
        Atomix.@atomic forces[n, i] += force_tmp[n]
        Atomix.@atomic forces[n, j] -= force_tmp[n]
    end

    @inbounds for idx_k in 1:fill_pos[i]
        k = nbrs[idx_k, i]

        r_ik_norm = normalize(vector(coords[k], coords[i], boundary))

        coef_BO = dBO_dΔi′[idx_j, i] * dBO′_dr[idx_k, i]
        coef_BOπ = dBOπ_dΔi′[idx_j, i] * dBO′_dr[idx_k, i]
        coef_BOππ = dBOππ_dΔi′[idx_j, i] * dBO′_dr[idx_k, i]

        dE_dΔi_tot = dE_dBO * coef_BO + dE_dBOπ * coef_BOπ + dE_dBOππ * coef_BOππ
        force_tmp = dE_dΔi_tot .* r_ik_norm

        for n in 1:3
            Atomix.@atomic forces[n, i] += force_tmp[n]
            Atomix.@atomic forces[n, k] -= force_tmp[n]
        end
    end

    @inbounds for idx_k in 1:fill_pos[j]
        k = nbrs[idx_k, j]

        r_jk_norm = normalize(vector(coords[k], coords[j], boundary))

        coef_BO = dBO_dΔj′[idx_j, i] * dBO′_dr[idx_k, j]
        coef_BOπ = dBOπ_dΔj′[idx_j, i] * dBO′_dr[idx_k, j]
        coef_BOππ = dBOππ_dΔj′[idx_j, i] * dBO′_dr[idx_k, j]

        dE_dΔj_tot = dE_dBO * coef_BO + dE_dBOπ * coef_BOπ + dE_dBOππ * coef_BOππ
        force_tmp = dE_dΔj_tot .* r_jk_norm

        for n in 1:3
            Atomix.@atomic forces[n, j] += force_tmp[n]
            Atomix.@atomic forces[n, k] -= force_tmp[n]
        end
    end
end


@inline function write_force_Θ(
    forces,
    i,
    j,
    k,
    dE_dΘ,
    dΘ_dri,
    dΘ_drj,
    dΘ_drk
)
    force_i = dE_dΘ .* dΘ_dri
    force_j = dE_dΘ .* dΘ_drj
    force_k = dE_dΘ .* dΘ_drk

    for n in 1:3
        Atomix.@atomic forces[n, i] += force_i[n]
        Atomix.@atomic forces[n, j] += force_j[n]
        Atomix.@atomic forces[n, k] += force_k[n]
    end
end


@inline function write_force_ω(
    forces,
    i,
    j,
    k,
    l,
    dE_dω,
    dω_dri,
    dω_drj,
    dw_drk,
    dw_drl
)
    force_i = dE_dω .* dω_dri
    force_j = dE_dω .* dω_drj
    force_k = dE_dω .* dw_drk
    force_l = dE_dω .* dw_drl

    for n in 1:3
        Atomix.@atomic forces[n, i] += force_i[n]
        Atomix.@atomic forces[n, j] += force_j[n]
        Atomix.@atomic forces[n, k] += force_k[n]
        Atomix.@atomic forces[n, l] += force_l[n]
    end
end


function write_force_rij(
    inter::ReaxFF{ReaxFF2008},
    i,
    j,
    dE_drij,
    r_ij_unit_vec
)
    forces = inter.state.forces

    f = dE_drij .* r_ij_unit_vec

    for n in 1:3
        Atomix.@atomic forces[n, i] += f[n]
        Atomix.@atomic forces[n, j] -= f[n]
    end
end


@kernel function resolve_forces_kernel(
    fill_pos,
    nbrs,
    coords,
    boundary,
    S_dE_dBO,
    S_dE_dBOπ,
    S_dE_dBOππ,
    S_dE_dΔ,
    forces,
    derivatives
)
    ti = @index(Global, Linear)

    @inbounds begin
        max_nbrs = size(nbrs, 1)
        
        # map linear GPU index onto column major nbrs matrix
        i = (ti - 1) ÷ max_nbrs + 1
        idx_j = (ti - 1) % max_nbrs + 1

        if idx_j <= fill_pos[i]
            j = nbrs[idx_j, i]

            if j > i
                write_force_BO(
                    coords,
                    boundary,
                    i,
                    j,
                    idx_j,
                    S_dE_dBO[idx_j, i] +  S_dE_dΔ[i] + S_dE_dΔ[j],
                    S_dE_dBOπ[idx_j, i],
                    S_dE_dBOππ[idx_j, i],
                    derivatives,
                    forces,
                    nbrs,
                    fill_pos
                )
            end
        end
    end
end


function resolve_forces!(
    inter::ReaxFF{ReaxFF2008},
    sys::System,
    backend;
    n_threads::Integer=Threads.nthreads(),
)
    fill_pos = inter.state.fill_pos
    nbrs = inter.state.nbrs
    S_dE_dBO = inter.state.S_dE_dBO
    S_dE_dBOπ = inter.state.S_dE_dBOπ
    S_dE_dBOππ = inter.state.S_dE_dBOππ
    S_dE_dΔ = inter.state.S_dE_dΔ

    derivatives = (
        dBO_dBO′ = inter.state.dBO_dBO′,
        dBO_dΔi′ = inter.state.dBO_dΔi′,
        dBO_dΔj′ = inter.state.dBO_dΔj′,
        dBO′_dr = inter.state.dBO′_dr,
        dBOπ_dBOπ′ = inter.state.dBOπ_dBOπ′,
        dBOπ_dBO′ = inter.state.dBOπ_dBO′,
        dBOπ_dΔi′ = inter.state.dBOπ_dΔi′,
        dBOπ_dΔj′ = inter.state.dBOπ_dΔj′,
        dBOπ′_dr = inter.state.dBOπ′_dr,
        dBOππ_dBOππ′ = inter.state.dBOππ_dBOππ′,
        dBOππ_dBO′ = inter.state.dBOππ_dBO′,
        dBOππ_dΔi′ = inter.state.dBOππ_dΔi′,
        dBOππ_dΔj′ = inter.state.dBOππ_dΔj′,
        dBOππ′_dr = inter.state.dBOππ′_dr
    )

    kernel = resolve_forces_kernel(backend)
    kernel(
        fill_pos,
        nbrs,
        sys.coords,
        sys.boundary,
        S_dE_dBO,
        S_dE_dBOπ,
        S_dE_dBOππ,
        S_dE_dΔ,
        inter.state.forces,
        derivatives,
        ndrange=length(nbrs),
        workgroupsize=backend_workgroupsize(backend, length(nbrs), n_threads)
    )
end


@inline function smoothness_taper(x::T, x_min, x_max) where {T}
    if x <= x_min
        return zero(T), zero(T)
    elseif x >= x_max
        return one(T), zero(T)
    end

    width_inv = inv(x_max - x_min)
    t = (x - x_min) * width_inv

    value = t^4 * (35 - 84*t + 70*t^2 - 20*t^3)
    derivative = 140 * t^3 * (1 - t)^3 * width_inv

    return value, derivative
end


@inline function bond_order(
    atmp,
    bndp,
    odgp,
    dr::T, 
    atid_i::Int, 
    atid_j::Int,
    bo_cutoff,
    ::Val{smoothness_correction},
) where {T, smoothness_correction}
    atmp_i = atmp[atid_i]   
    atmp_j = atmp[atid_j]    
    bndp_ij = bndp[atid_i, atid_j]

    ro_σ_i = atmp_i.ro_sigma
    ro_σ_j = atmp_j.ro_sigma

    ro_π_i = atmp_i.ro_pi
    ro_π_j = atmp_j.ro_pi

    ro_ππ_i = atmp_i.ro_pipi
    ro_ππ_j = atmp_j.ro_pipi

    odgp_ij = odgp[atid_i, atid_j]
    
    if ro_σ_i > zero(T) && ro_σ_j > zero(T) && odgp_ij.ro_sigma > zero(T)
        p_bo1 = bndp_ij.p_bo1
        p_bo2 = bndp_ij.p_bo2
        ro_σ_bond = odgp_ij.ro_sigma

        BO′_σ_arg = p_bo1 * (dr / ro_σ_bond)^p_bo2
        BO′_σ_exp = exp(BO′_σ_arg)
        BO′_σ_shift = (one(T) + bo_cutoff) * BO′_σ_exp
        BO′_σ = BO′_σ_shift - bo_cutoff
        dBOσ′_dr = BO′_σ_shift * p_bo2 * BO′_σ_arg / dr

        if smoothness_correction
            taper, dtaper = smoothness_taper(BO′_σ, bo_cutoff, 4 * bo_cutoff)
            dBOσ′_dr *= taper + BO′_σ * dtaper
            BO′_σ *= taper
        end
    else
        BO′_σ = zero(T)
        dBOσ′_dr = zero(T)
    end

    if ro_π_i > zero(T) && ro_π_j > zero(T) && odgp_ij.ro_pi > zero(T)
        p_bo3 = bndp_ij.p_bo3
        p_bo4 = bndp_ij.p_bo4
        ro_π_bond = odgp_ij.ro_pi

        BO′_π_arg = p_bo3 * (dr / ro_π_bond)^p_bo4
        BO′_π = exp(BO′_π_arg)
        dBOπ′_dr = BO′_π * p_bo4 * BO′_π_arg / dr 

        if smoothness_correction
            taper, dtaper = smoothness_taper(BO′_π, bo_cutoff, 4 * bo_cutoff)
            dBOπ′_dr *= taper + BO′_π * dtaper
            BO′_π *= taper
        end
    else
        BO′_π = zero(T)
        dBOπ′_dr = zero(T)
    end

    if ro_ππ_i > zero(T) && ro_ππ_j > zero(T) && odgp_ij.ro_pipi > zero(T)
        p_bo5 = bndp_ij.p_bo5
        p_bo6 = bndp_ij.p_bo6
        ro_ππ_bond = odgp_ij.ro_pipi

        BO′_ππ_arg = p_bo5 * (dr / ro_ππ_bond)^p_bo6
        BO′_ππ = exp(BO′_ππ_arg)
        dBOππ′_dr = BO′_ππ * p_bo6 * BO′_ππ_arg / dr

        if smoothness_correction
            taper, dtaper = smoothness_taper(BO′_ππ, bo_cutoff, 4 * bo_cutoff)
            dBOππ′_dr *= taper + BO′_ππ * dtaper
            BO′_ππ *= taper
        end
    else
        BO′_ππ = zero(T)
        dBOππ′_dr = zero(T)
    end

    BO′ = BO′_σ + BO′_π + BO′_ππ

    dBO′_dr = dBOσ′_dr + dBOπ′_dr + dBOππ′_dr

    return BO′, BO′_π, BO′_ππ, dBO′_dr, dBOπ′_dr, dBOππ′_dr
end


@inline function bond_order_correction(
    BO′::T,
    BO′_π::T,
    BO′_ππ::T,
    Δ′_i::T,
    Δ′_j::T,
    Δ′_boc_i::T,
    Δ′_boc_j::T,
    atid_i::Int,
    atid_j::Int,
    i,
    j,
    idx_i,
    idx_j,
    atmp,
    genp,
    bndp,
    dBO_dBO′,
    dBO_dΔi′,
    dBO_dΔj′,
    dBOπ_dBOπ′,
    dBOπ_dBO′,
    dBOπ_dΔi′,
    dBOπ_dΔj′,
    dBOππ_dBOππ′,
    dBOππ_dBO′,
    dBOππ_dΔi′,
    dBOππ_dΔj′,
    BO_mat,
    BO_π_mat, 
    BO_ππ_mat,
    ::Val{smoothness_correction},
) where {T, smoothness_correction}
    
    atmp_i = atmp[atid_i]   
    atmp_j = atmp[atid_j]

    val_i = atmp_i.Val
    val_j = atmp_j.Val

    p_boc1 = genp.p_boc1
    p_boc2 = genp.p_boc2

    p_boc3 = sqrt(atmp_i.p_boc3 * atmp_j.p_boc3)
    p_boc4 = sqrt(atmp_i.p_boc4 * atmp_j.p_boc4)
    p_boc5 = sqrt(atmp_i.p_boc5 * atmp_j.p_boc5)

    if bndp[atid_i, atid_j].ovcoord >= T(0.001)
        df2_exp1_arg = -p_boc1 * Δ′_i
        df2_exp2_arg = -p_boc1 * Δ′_j

        max_arg = max(df2_exp1_arg, df2_exp2_arg)

        f3_inv1 = inv(p_boc2)
        df3_exp1 = exp(-p_boc2 * Δ′_i)
        df3_exp2 = exp(-p_boc2 * Δ′_j)
        df3_inv1 = inv(T(0.5) * (df3_exp1 + df3_exp2))
        df3_pre = -f3_inv1 * df3_inv1 * T(0.5)
        df3_mul1 = -df3_exp1 * p_boc2
        df3_mul2 = -df3_exp2 * p_boc2 

        f3 = -f3_inv1 * log(T(0.5) * (df3_exp1 + df3_exp2))

        if max_arg < T(40.0)
            df2_exp1 = exp(df2_exp1_arg)
            df2_exp2 = exp(df2_exp2_arg)

            df2_mul1 = -df2_exp1 * p_boc1
            df2_mul2 = -df2_exp2 * p_boc1
        
            f2 = df2_exp1 + df2_exp2

            f1_s1 = val_i + f2
            f1_s2 = f1_s1 + f3
            f1_inv1 = inv(f1_s2)

            f1_s3 = val_j + f2
            f1_s4 = f1_s3 + f3
            f1_inv2 = inv(f1_s4)

            df1_inv1 = inv(f1_s2 * f1_s2)
            df1_inv2 = inv(f1_s4 * f1_s4)

            df1_dBO′ = zero(T)
            
            df2_dΔi′ = df2_mul1
            df3_dΔi′ = df3_pre * df3_mul1
            df1_dΔi′ = T(0.5) * ((df2_dΔi′ * f1_s2 - f1_s1 * (df2_dΔi′ + df3_dΔi′)) * df1_inv1 + 
                            (df2_dΔi′ * f1_s4 - f1_s3 * (df2_dΔi′ + df3_dΔi′)) * df1_inv2)

            df2_dΔj′ = df2_mul2
            df3_dΔj′ = df3_pre * df3_mul2
            df1_dΔj′ = T(0.5) * ((df2_dΔj′ * f1_s2 - f1_s1 * (df2_dΔj′ + df3_dΔj′)) * df1_inv1 + 
                            (df2_dΔj′ * f1_s4 - f1_s3 * (df2_dΔj′ + df3_dΔj′)) * df1_inv2)

            f1 = T(0.5) * ((f1_s1 * f1_inv1) + (f1_s3 * f1_inv2))

        else
            exp_max_arg = exp(-max_arg)
            f2_mod_exp1 = exp(df2_exp1_arg - max_arg)
            f2_mod_exp2 = exp(df2_exp2_arg - max_arg)
            f2_mod = f2_mod_exp1 + f2_mod_exp2
            val_i_mod = val_i * exp_max_arg
            val_j_mod = val_j * exp_max_arg
            f3_mod = f3 * exp_max_arg
            f1_inv1 = inv(val_i_mod + f2_mod + f3_mod)
            f1_inv2 = inv(val_j_mod + f2_mod + f3_mod)

            f1 = T(0.5) * ((val_i_mod + f2_mod) * f1_inv1 + (val_j_mod + f2_mod) * f1_inv2)
            df1_dBO′ = zero(T)

            df2_dΔi′_mod = -p_boc1 * f2_mod_exp1
            df3_dΔi′_mod = df3_pre * df3_mul1 * exp_max_arg
            df1_dΔi′ = T(0.5) * ((df2_dΔi′_mod * (val_i_mod + f2_mod + f3_mod) - (val_i_mod + f2_mod) * (df2_dΔi′_mod + df3_dΔi′_mod)) * f1_inv1 * f1_inv1 + 
                                     (df2_dΔi′_mod * (val_j_mod + f2_mod + f3_mod)  - (val_j_mod + f2_mod) * (df2_dΔi′_mod + df3_dΔi′_mod)) * f1_inv2 * f1_inv2)

            df2_dΔj′_mod = -p_boc1 * f2_mod_exp2
            df3_dΔj′_mod = df3_pre * df3_mul2 * exp_max_arg
            df1_dΔj′ = T(0.5) * ((df2_dΔj′_mod * (val_i_mod + f2_mod + f3_mod) - (val_i_mod + f2_mod) * (df2_dΔj′_mod + df3_dΔj′_mod)) * f1_inv1 * f1_inv1 + 
                                     (df2_dΔj′_mod * (val_j_mod + f2_mod + f3_mod)  - (val_j_mod + f2_mod) * (df2_dΔj′_mod + df3_dΔj′_mod)) * f1_inv2 * f1_inv2)
        end

    else
        f1 = one(T)
        df1_dBO′ = zero(T)
        df1_dΔi′ = zero(T)
        df1_dΔj′ = zero(T)
    end
    
    if bndp[atid_i, atid_j].ot_corr > T(0.001)
        f4_exp1 = exp(-p_boc3 * (p_boc4 * BO′ * BO′ - Δ′_boc_i) + p_boc5)
        f5_exp1 = exp(-p_boc3 * (p_boc4 * BO′ * BO′ - Δ′_boc_j) + p_boc5)

        df4_inv1 = inv(one(T) + f4_exp1)

        df5_inv1 = inv(one(T) + f5_exp1)
        
        f4 = df4_inv1
        f5 = df5_inv1

        df4_dBO′ = -(df4_inv1 * df4_inv1) * f4_exp1 * (-p_boc3) * (T(2.0) * p_boc4 * BO′)
        df5_dBO′ = -(df5_inv1 * df5_inv1) * f5_exp1 * (-p_boc3) * (T(2.0) * p_boc4 * BO′)

        df4_dΔi′ = -(df4_inv1 * df4_inv1) * f4_exp1 * p_boc3
        df5_dΔi′ = zero(T)

        df4_dΔj′ = zero(T)
        df5_dΔj′ = -(df5_inv1 * df5_inv1) * f5_exp1 * p_boc3
    else
        f4 = one(T)
        f5 = one(T)
        
        df4_dBO′ = zero(T)
        df5_dBO′ = zero(T)
        df4_dΔi′ = zero(T)
        df5_dΔi′ = zero(T)
        df4_dΔj′ = zero(T)
        df5_dΔj′ = zero(T)
    end

    
    f1f4f5 = f1 * f4 * f5
    f1f1f4f5 = f1 * f1f4f5
    f1_sq = f1 * f1
    f4f5 = f4 * f5

    BO = BO′ * f1f4f5
    BO_π = BO′_π * f1f1f4f5
    BO_ππ = BO′_ππ * f1f1f4f5
    BO_σ = BO - BO_π - BO_ππ

    dBOπ_dBOπ′_tm = f1f1f4f5 
    dBOπ_dBO′_tm  = BO′_π * (T(2.0) * f1 * df1_dBO′ * f4f5 + f1_sq * df4_dBO′ * f5 + f1_sq * f4 * df5_dBO′)
    dBOπ_dΔi′_tm  = BO′_π * (T(2.0) * f1 * df1_dΔi′ * f4f5 + f1_sq * df4_dΔi′ * f5 + f1_sq * f4 * df5_dΔi′)
    dBOπ_dΔj′_tm  = BO′_π * (T(2.0) * f1 * df1_dΔj′ * f4f5 + f1_sq * df4_dΔj′ * f5 + f1_sq * f4 * df5_dΔj′)

    dBOππ_dBOππ′_tm = f1f1f4f5
    dBOππ_dBO′_tm   = BO′_ππ * (T(2.0) * f1 * df1_dBO′ * f4f5 + f1_sq * df4_dBO′ * f5 + f1_sq * f4 * df5_dBO′)
    dBOππ_dΔi′_tm   = BO′_ππ * (T(2.0) * f1 * df1_dΔi′ * f4f5 + f1_sq * df4_dΔi′ * f5 + f1_sq * f4 * df5_dΔi′)
    dBOππ_dΔj′_tm   = BO′_ππ * (T(2.0) * f1 * df1_dΔj′ * f4f5 + f1_sq * df4_dΔj′ * f5 + f1_sq * f4 * df5_dΔj′)

    dBO_dBO′_tm = f1f4f5 + BO′ * (df1_dBO′ * f4f5 + f1 * df4_dBO′ * f5 + f1 * f4 * df5_dBO′)
    dBO_dΔi′_tm = BO′ * (df1_dΔi′ * f4f5 + f1 * df4_dΔi′ * f5 + f1 * f4 * df5_dΔi′)
    dBO_dΔj′_tm = BO′ * (df1_dΔj′ * f4f5 + f1 * df4_dΔj′ * f5 + f1 * f4 * df5_dΔj′) 

    if !smoothness_correction
        BO_threshold = T(1e-10)
        if BO < BO_threshold
            BO = zero(T)
        end
        if BO_σ < BO_threshold
            BO_σ = zero(T)
        end
        if BO_π < BO_threshold
            BO_π = zero(T)
        end
        if BO_ππ < BO_threshold
            BO_ππ = zero(T)
        end
    end

    dBO_dBO′[idx_j, i] = dBO_dBO′[idx_i, j] = dBO_dBO′_tm
    dBOπ_dBOπ′[idx_j, i] = dBOπ_dBOπ′[idx_i, j] = dBOπ_dBOπ′_tm
    dBOπ_dBO′[idx_j, i] = dBOπ_dBO′[idx_i, j] = dBOπ_dBO′_tm
    dBOππ_dBOππ′[idx_j, i] = dBOππ_dBOππ′[idx_i, j] = dBOππ_dBOππ′_tm
    dBOππ_dBO′[idx_j, i] = dBOππ_dBO′[idx_i, j] = dBOππ_dBO′_tm

    dBO_dΔi′[idx_j, i] = dBO_dΔi′_tm
    dBO_dΔi′[idx_i, j] = dBO_dΔj′_tm

    dBO_dΔj′[idx_j, i] = dBO_dΔj′_tm
    dBO_dΔj′[idx_i, j] = dBO_dΔi′_tm

    dBOπ_dΔi′[idx_j, i] = dBOπ_dΔi′_tm
    dBOπ_dΔi′[idx_i, j] = dBOπ_dΔj′_tm

    dBOπ_dΔj′[idx_j, i] = dBOπ_dΔj′_tm
    dBOπ_dΔj′[idx_i, j] = dBOπ_dΔi′_tm

    dBOππ_dΔi′[idx_j, i] = dBOππ_dΔi′_tm
    dBOππ_dΔi′[idx_i, j] = dBOππ_dΔj′_tm

    dBOππ_dΔj′[idx_j, i] = dBOππ_dΔj′_tm
    dBOππ_dΔj′[idx_i, j] = dBOππ_dΔi′_tm
    
    BO_mat[idx_j, i]    = BO_mat[idx_i, j]    = BO
    BO_π_mat[idx_j, i]  = BO_π_mat[idx_i, j]  = BO_π
    BO_ππ_mat[idx_j, i] = BO_ππ_mat[idx_i, j] = BO_ππ

    return BO, BO_σ, BO_π, BO_ππ
end


@kernel function bond_kernel(
    nbrs_list,
    bndp,
    atid,
    conv_factor::T,
    coords,
    boundary,
    fill_pos,
    nbrs,
    reverse_nbrs,
    BO′,
    BO′_π,
    BO′_ππ,
    dBO′_dr,
    dBOπ′_dr,
    dBOππ′_dr,
    coordn′,
    bo_cutoff,
    atmp,
    odgp,
    bond_dist_cutoff_sq,
    smoothness_correction,
) where T
    ti = @index(Global, Linear)
    
    @inbounds begin
        i, j, _ = nbrs_list[ti]

        qry = bndp[atid[i], atid[j]]

        if qry.is_defined 
            r_vec_ij = map(x -> ustrip(x) * conv_factor, vector(coords[j], coords[i], boundary))
            dr_sq = sum(abs2, r_vec_ij)

            if dr_sq <= bond_dist_cutoff_sq
                dr = sqrt(dr_sq)

                bo′, bo′_π, bo′_ππ, dBO′_dr_tm, dBOπ′_dr_tm, dBOππ′_dr_tm = bond_order(
                    atmp,
                    bndp,
                    odgp,
                    dr, 
                    atid[i], 
                    atid[j],
                    bo_cutoff,
                    smoothness_correction,
                )
    
                if bo′ > zero(T) 
                    fp_i = Atomix.@atomic fill_pos[i] += 1
                    fp_j = Atomix.@atomic fill_pos[j] += 1
                    if fp_i > size(nbrs, 1) || fp_j > size(nbrs, 1)
                        error("ReaxFF bonded-neighbor capacity exceeded")
                    end
    
                    nbrs[fp_i, i] = j
                    nbrs[fp_j, j] = i
    
                    reverse_nbrs[fp_i, i] = fp_j
                    reverse_nbrs[fp_j, j] = fp_i
    
                    BO′[fp_i, i] = bo′
                    BO′[fp_j, j] = bo′
                    BO′_π[fp_i, i] = bo′_π
                    BO′_π[fp_j, j] = bo′_π
                    BO′_ππ[fp_i, i] = bo′_ππ
                    BO′_ππ[fp_j, j] = bo′_ππ
    
                    dBO′_dr[fp_i, i] = dBO′_dr_tm
                    dBO′_dr[fp_j, j] = dBO′_dr_tm
                    dBOπ′_dr[fp_i, i] = dBOπ′_dr_tm
                    dBOπ′_dr[fp_j, j] = dBOπ′_dr_tm
                    dBOππ′_dr[fp_i, i] = dBOππ′_dr_tm
                    dBOππ′_dr[fp_j, j] = dBOππ′_dr_tm
    
                    Atomix.@atomic coordn′[i] += bo′
                    Atomix.@atomic coordn′[j] += bo′
                end
            end
        end
    end
end


@kernel function update_Δ_prime_kernel(
    coordn′,
    atmp,
    atid,
    Δ′,
    Δ′_boc
)
    ti = @index(Global, Linear)

    @inbounds begin
        val_i = atmp[atid[ti]].Val
        val_boc_i = atmp[atid[ti]].Val_boc
        c = coordn′[ti]
        Δ′[ti] = c - val_i
        Δ′_boc[ti] = c - val_boc_i
    end
end


@kernel function bond_correction_kernel(
    atid,
    fill_pos,
    nbrs,
    reverse_nbrs,
    coordn,
    bndp,
    S_dE_dBO,
    S_dE_dBOπ,
    S_dE_dBOππ,
    atmp,
    genp,
    dBO_dBO′,
    dBO_dΔi′,
    dBO_dΔj′,
    dBOπ_dBOπ′,
    dBOπ_dBO′,
    dBOπ_dΔi′,
    dBOπ_dΔj′,
    dBOππ_dBOππ′,
    dBOππ_dBO′,
    dBOππ_dΔi′,
    dBOππ_dΔj′,
    BO_mat,
    BO_π_mat, 
    BO_ππ_mat,
    pair_energies::AbstractArray{T},
    BO′,
    BO′_π,
    BO′_ππ,
    Δ′,
    Δ′_boc,
    smoothness_correction,
) where T
    i = @index(Global, Linear)

    @inbounds begin
        for idx_j in 1:fill_pos[i]
            j = nbrs[idx_j, i]

            j > i || continue

            idx_i = reverse_nbrs[idx_j, i]
            
            bo, bo_σ, bo_π, bo_ππ = bond_order_correction(
                BO′[idx_j, i],
                BO′_π[idx_j, i],
                BO′_ππ[idx_j, i],
                Δ′[i],
                Δ′[j],
                Δ′_boc[i],
                Δ′_boc[j],
                atid[i],
                atid[j],
                i, 
                j,
                idx_i,
                idx_j,
                atmp,
                genp,
                bndp,
                dBO_dBO′,
                dBO_dΔi′,
                dBO_dΔj′,
                dBOπ_dBOπ′,
                dBOπ_dBO′,
                dBOπ_dΔi′,
                dBOπ_dΔj′,
                dBOππ_dBOππ′,
                dBOππ_dBO′,
                dBOππ_dΔi′,
                dBOππ_dΔj′,
                BO_mat,
                BO_π_mat, 
                BO_ππ_mat,
                smoothness_correction,
            )

            Atomix.@atomic coordn[i] += bo        
            Atomix.@atomic coordn[j] += bo

            bndp_ij = bndp[atid[i], atid[j]]

            De_σ  = bndp_ij.De_sigma
            De_π  = bndp_ij.De_pi
            De_ππ = bndp_ij.De_pipi
            p_be1 = bndp_ij.p_be1
            p_be2 = bndp_ij.p_be2

            dEBond_in = (bo_σ)^p_be2
            dEbond_exp1 = -De_σ * exp(p_be1 * (one(T) - dEBond_in))

            dEBond_dBOσ = dEbond_exp1 * (one(T) - p_be1 * p_be2 * dEBond_in)
            dEBond_dBO = dEBond_dBOσ
            dEBond_dBOπ = -De_π - dEBond_dBOσ
            dEBond_dBOππ = -De_ππ - dEBond_dBOσ

            S_dE_dBO[idx_j, i] += dEBond_dBO
            S_dE_dBO[idx_i, j] += dEBond_dBO
            S_dE_dBOπ[idx_j, i] += dEBond_dBOπ
            S_dE_dBOπ[idx_i, j] += dEBond_dBOπ
            S_dE_dBOππ[idx_j, i] += dEBond_dBOππ
            S_dE_dBOππ[idx_i, j] += dEBond_dBOππ

            bnd_engy = dEbond_exp1 * bo_σ  - De_π * bo_π - De_ππ * bo_ππ
            
            pair_energies[i] += bnd_engy
        end
    end
end


@kernel function update_Δ_kernel(
    coordn,
    atmp,
    atid,
    Δ,
    Δ_e,
    Δ_angle,
    Δ_boc
)
    ti = @index(Global, Linear)

    @inbounds begin
        atmp_i = atmp[atid[ti]]

        val       = atmp_i.Val
        val_e     = atmp_i.Val_e
        val_angle = atmp_i.Val_angle
        val_boc   = atmp_i.Val_boc

        c = coordn[ti]

        Δ[ti] = c - val
        Δ_e[ti] = c - val_e
        Δ_angle[ti] = c - val_angle
        Δ_boc[ti] = c - val_boc
    end
end


function bonds!(
    inter::ReaxFF{ReaxFF2008, T},
    sys::System,
    neighbors::Union{NeighborList, GPUCellListNeighborList},
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T
    atmp = inter.device_params.atom_parameters
    bndp = inter.device_params.bond_parameters
    atid = inter.state.sys_atid 
    nbrs = inter.state.nbrs
    reverse_nbrs = inter.state.reverse_nbrs
    fill_pos = inter.state.fill_pos
    odgp = inter.device_params.off_diagonal_parameters
    genp = inter.device_params.general_parameters

    coordn′ = inter.state.coordn
    BO′     = inter.state.BO
    BO′_π   = inter.state.BO_π
    BO′_ππ  = inter.state.BO_ππ

    dBO′_dr = inter.state.dBO′_dr
    dBOπ′_dr = inter.state.dBOπ′_dr
    dBOππ′_dr = inter.state.dBOππ′_dr

    S_dE_dBO = inter.state.S_dE_dBO
    S_dE_dBOπ = inter.state.S_dE_dBOπ
    S_dE_dBOππ = inter.state.S_dE_dBOππ

    bo_cutoff  = inter.device_params.global_parameters.bo_cutoff

    # assume unit-less coords has units of nm
    conv_factor = inter.use_units ? T(ustrip(uconvert(u"Å", one(T) * inter.sys_coord_unit))) : T(10.0)

    bond_dist_cutoff_sq = inter.device_params.global_parameters.bond_dist_cutoff^2

    b_kernel = bond_kernel(backend)
    b_kernel(
        neighbors.list,
        bndp,
        atid,
        conv_factor,
        sys.coords,
        sys.boundary,
        fill_pos,
        nbrs,
        reverse_nbrs,
        BO′,
        BO′_π,
        BO′_ππ,
        dBO′_dr,
        dBOπ′_dr,
        dBOππ′_dr,
        coordn′,
        bo_cutoff,
        atmp,
        odgp,
        bond_dist_cutoff_sq,
        inter.smoothness_correction;
        ndrange=length(neighbors),
        workgroupsize=backend_workgroupsize(backend, length(neighbors), n_threads)
    )
    synchronize(backend)

    Δ′ = inter.state.Δ         # uncorrected overcoordination
    Δ′_boc = inter.state.Δ_boc     # second uncorrected overcoordination

    update_Δ′_kernel = update_Δ_prime_kernel(backend)
    update_Δ′_kernel(
        coordn′,
        atmp,
        atid,
        Δ′,
        Δ′_boc;
        ndrange=length(coordn′),
        workgroupsize=backend_workgroupsize(backend, length(coordn′), n_threads)
    )

    coordn = inter.state.coordn
    fill!(coordn, zero(T))     # will now hold corrected coordinations

    n_atoms = inter.state.n_atoms

    dBO_dBO′ = inter.state.dBO_dBO′
    dBO_dΔi′ = inter.state.dBO_dΔi′
    dBO_dΔj′ = inter.state.dBO_dΔj′
    dBOπ_dBOπ′ = inter.state.dBOπ_dBOπ′
    dBOπ_dBO′ = inter.state.dBOπ_dBO′
    dBOπ_dΔi′ = inter.state.dBOπ_dΔi′
    dBOπ_dΔj′ = inter.state.dBOπ_dΔj′
    dBOππ_dBOππ′ = inter.state.dBOππ_dBOππ′
    dBOππ_dBO′ = inter.state.dBOππ_dBO′
    dBOππ_dΔi′ = inter.state.dBOππ_dΔi′
    dBOππ_dΔj′ = inter.state.dBOππ_dΔj′
    BO_mat     = inter.state.BO
    BO_π_mat   = inter.state.BO_π
    BO_ππ_mat  = inter.state.BO_ππ

    pair_energies = inter.state.thread_atom_energies_1
    fill!(pair_energies, zero(T))

    bc_kernel = bond_correction_kernel(backend)
    bc_kernel(
        atid,
        fill_pos,
        nbrs,
        reverse_nbrs,
        coordn,
        bndp,
        S_dE_dBO,
        S_dE_dBOπ,
        S_dE_dBOππ,
        atmp,
        genp,
        dBO_dBO′,
        dBO_dΔi′,
        dBO_dΔj′,
        dBOπ_dBOπ′,
        dBOπ_dBO′,
        dBOπ_dΔi′,
        dBOπ_dΔj′,
        dBOππ_dBOππ′,
        dBOππ_dBO′,
        dBOππ_dΔi′,
        dBOππ_dΔj′,
        BO_mat,   
        BO_π_mat, 
        BO_ππ_mat,
        pair_energies,
        BO′,
        BO′_π,
        BO′_ππ,
        Δ′,
        Δ′_boc,
        inter.smoothness_correction;
        ndrange=n_atoms,
        workgroupsize=backend_workgroupsize(backend, n_atoms, n_threads)
    )

    Δ       = inter.state.Δ        
    Δ_e     = inter.state.Δ_e 
    Δ_angle = inter.state.Δ_angle 
    Δ_boc   = inter.state.Δ_boc 

    update_Δ = update_Δ_kernel(backend)
    update_Δ(
        coordn,
        atmp,
        atid,
        Δ,
        Δ_e,
        Δ_angle,
        Δ_boc;
        ndrange=length(coordn),
        workgroupsize=backend_workgroupsize(backend, length(coordn), n_threads)
    )

    E_bond = sum(pair_energies)
    inter.state.energy += E_bond
    scalar_setindex!(inter.state.energy_components, E_bond, 1)
end


@kernel function lone_pairs_kernel(
    atmp,
    atid,
    Δ_e,
    Δ_e_mod2,
    genp,
    n_lp,
    Δ_lp,
    dn_lp_dΔi,
    fill_pos,
    S_dE_dΔ,
    atom_energies::AbstractArray{T}
) where T
    i = @index(Global, Linear)

    is_float32 = T == Float32

    @inbounds begin
        atmp_i = atmp[atid[i]]

        val   = atmp_i.Val
        val_e = atmp_i.Val_e

        Δ_e_i = Δ_e[i]

        n_lp_opt = T(0.5) * (val_e - val)

        lp_trc = trunc(Int, Δ_e_i / T(2.0))
        Δ_e_mod2_i = Δ_e_i - (T(2.0) * lp_trc)
        Δ_e_mod2[i] = Δ_e_mod2_i

        p_lp1 = genp.p_lp1
        p_lp2 = atmp_i.p_lp2

        lp_arg1 = T(2.0) + Δ_e_mod2_i
        lp_exp1 = exp(-p_lp1 * lp_arg1^2)

        n_lp_i = -lp_trc + lp_exp1
        n_lp[i] = n_lp_i

        delta_lp_i = n_lp_opt - n_lp_i
        Δ_lp[i] = delta_lp_i
        
        lp_exp2_arg = T(-75.0) * delta_lp_i

        if is_float32
            if lp_exp2_arg > T(80.0)
                inv1 = zero(T)
            elseif lp_exp2_arg < T(-80.0)
                inv1 = one(T)
            else
                inv1 = inv(one(T) + exp(lp_exp2_arg))
            end
        else 
            inv1 = inv(one(T) + exp(lp_exp2_arg))
        end

        dn_lp_i = lp_exp1 * (T(2.0) * -p_lp1) * lp_arg1
        dn_lp_dΔi[i] = dn_lp_i

        dE_dΔi = p_lp2 * -dn_lp_i * inv1 * (one(T) + (one(T) - inv1) * T(75.0) * delta_lp_i)

        if fill_pos[i] > 0
            S_dE_dΔ[i] += dE_dΔi
            lp_engy = p_lp2 * delta_lp_i * inv1
            atom_energies[i] += lp_engy
        end
    end
end


function lone_pairs!(
    inter::ReaxFF{ReaxFF2008, T},
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T
    genp = inter.device_params.general_parameters
    atmp = inter.device_params.atom_parameters
    atid = inter.state.sys_atid 

    Δ_lp = inter.state.Δ_lp     # empty
    Δ_e = inter.state.Δ_e       # filled
    n_lp = inter.state.n_lp
    Δ_e_mod2 = inter.state.Δ_e_mod2

    n_atoms = inter.state.n_atoms
    dn_lp_dΔi = inter.state.dn_lp_dΔi
    S_dE_dΔ = inter.state.S_dE_dΔ
    fill_pos = inter.state.fill_pos

    atom_energies = inter.state.thread_atom_energies_1
    fill!(atom_energies, zero(T))

    kernel = lone_pairs_kernel(backend)
    kernel(
        atmp,
        atid,
        Δ_e,
        Δ_e_mod2,
        genp,
        n_lp,
        Δ_lp,
        dn_lp_dΔi,
        fill_pos,
        S_dE_dΔ,
        atom_energies;
        ndrange=n_atoms,
        workgroupsize=backend_workgroupsize(backend, n_atoms, n_threads)
    )
    synchronize(backend)
    E_lp = sum(atom_energies)
    inter.state.energy += E_lp
    scalar_setindex!(inter.state.energy_components, E_lp, 3)
end


@kernel function ovun_kernel(
    fill_pos,
    nbrs,
    bndp,
    atid,
    Δ,
    Δ_lp,
    BO_π,
    BO_ππ,
    BO,
    genp,
    atmp,
    reverse_nbrs,
    S_dE_dBO,
    S_dE_dBOπ,
    S_dE_dBOππ,
    dn_lp_dΔi,
    S_dE_dΔ,
    overcoordination_energies::AbstractArray{T},
    undercoordination_energies::AbstractArray{T}
) where T
    i = @index(Global, Linear)

    @inbounds begin
        p_ovun3 = genp.p_ovun3
        p_ovun4 = genp.p_ovun4
        p_ovun6 = genp.p_ovun6
        p_ovun7 = genp.p_ovun7
        p_ovun8 = genp.p_ovun8

        S1 = zero(T)
        S2 = zero(T)

        for idx_j in 1:fill_pos[i]
            j = nbrs[idx_j, i]

            p_ovun1  = bndp[atid[i], atid[j]].p_ovun1
            De_sigma = bndp[atid[i], atid[j]].De_sigma

            S1 += (Δ[j] - Δ_lp[j]) * (BO_π[idx_j, i] + BO_ππ[idx_j, i])
            S2 += p_ovun1 * De_sigma * BO[idx_j, i]
        end

        atmp_i = atmp[atid[i]]
        Val_i   = atmp_i.Val
        p_ovun2 = atmp_i.p_ovun2
        p_ovun5 = atmp_i.p_ovun5

        # Overcoordination energy
        ov_exp1 = exp(p_ovun4 * S1)
        inv2 = inv(one(T) + p_ovun3 * ov_exp1)
        delta_lpcorr = Δ[i] - (Δ_lp[i] * inv2)

        ov_sm1 = delta_lpcorr + Val_i
        inv3 = inv(ov_sm1 + T(1e-8))
        ov_exp2 = exp(p_ovun2 * delta_lpcorr)
        inv4 = inv(one(T) + ov_exp2)

        ovcoord_engy = S2 * inv3 * delta_lpcorr * inv4
        overcoordination_energies[i] += ovcoord_engy

        # Undercoordination energy
        ov_exp4 = exp(-p_ovun2 * delta_lpcorr)
        inv5 = inv(one(T) + ov_exp4)

        ov_povun8_exp = p_ovun7 * exp(p_ovun8 * S1)
        inv6 = inv(one(T) + ov_povun8_exp)

        inv5_inv6 = inv5 * inv6
        inv5_sq_inv6 = inv5 * inv5_inv6

        ov_exp3 = exp(p_ovun6 * delta_lpcorr)
        undcoord_engy = -p_ovun5 * (one(T) - ov_exp3) * inv5_inv6
        undercoordination_energies[i] += undcoord_engy

        dE_over_prod = inv3 * delta_lpcorr * inv4
        dΔ_lpcorr_prod = Δ_lp[i] * inv2 * inv2 * p_ovun3 * ov_exp1 * p_ovun4
        dE_over_prod2 = S2 * inv3 * inv4 * (one(T) - delta_lpcorr * ov_exp2 * inv4 * p_ovun2)
   
        dE_under_tm1 = -ov_exp3 * p_ovun6 * (one(T) + ov_exp4) + 
                        (one(T) - ov_exp3) * ov_exp4 * p_ovun2
        dE_under_tm2 = (one(T) - ov_exp3) * inv5_inv6 * -inv6 * ov_povun8_exp * p_ovun8
        dE_under_tm3 =  S2 * inv3 * inv4 * (one(T) -
                            delta_lpcorr * inv4 * ov_exp2 * p_ovun2 -
                            inv3 * delta_lpcorr
                        )
 
        for idx_j in 1:fill_pos[i]
            j = nbrs[idx_j, i]

            idx_i = reverse_nbrs[idx_j, i]

            p_ovun1  = bndp[atid[i], atid[j]].p_ovun1
            De_sigma = bndp[atid[i], atid[j]].De_sigma

            dΔ_lpcorr_dBOπ = dΔ_lpcorr_prod * (Δ[j] - Δ_lp[j])

            dE_over_dBO = (p_ovun1 * De_sigma) * dE_over_prod   
            dE_over_dBOπ = -S2 * inv3 * dE_over_prod * dΔ_lpcorr_dBOπ + dΔ_lpcorr_dBOπ * dE_over_prod2
    
            dE_under_dBOπ = -p_ovun5 * (
                                dE_under_tm1 * dΔ_lpcorr_dBOπ * inv5_sq_inv6 + 
                                dE_under_tm2 * (Δ[j] - Δ_lp[j])
                            )

            dE_tot_sum = dE_over_dBOπ + dE_under_dBOπ
            Atomix.@atomic S_dE_dBO[idx_j, i] += dE_over_dBO
            Atomix.@atomic S_dE_dBO[idx_i, j] += dE_over_dBO
            Atomix.@atomic S_dE_dBOπ[idx_j, i] += dE_tot_sum
            Atomix.@atomic S_dE_dBOπ[idx_i, j] += dE_tot_sum
            Atomix.@atomic S_dE_dBOππ[idx_j, i] += dE_tot_sum
            Atomix.@atomic S_dE_dBOππ[idx_i, j] += dE_tot_sum

            BO_sum = BO_π[idx_j, i] + BO_ππ[idx_j, i]

            dΔ_lpcorr_dΔj = dΔ_lpcorr_prod * (one(T) + dn_lp_dΔi[j]) * BO_sum

            dE_over_dΔj = dE_under_tm3 * dΔ_lpcorr_dΔj

            dE_under_tm5 = dE_under_tm1 * dΔ_lpcorr_dΔj
            dE_under_dΔj = -p_ovun5 * (
                                dE_under_tm5 * inv5_sq_inv6 + 
                                dE_under_tm2 * (one(T) + dn_lp_dΔi[j]) * BO_sum
                            )

            Atomix.@atomic S_dE_dΔ[j] += dE_over_dΔj + dE_under_dΔj
        end

        dΔ_lpcorr_dΔi = one(T) + dn_lp_dΔi[i] * inv2

        dE_over_dΔi = dE_under_tm3 * dΔ_lpcorr_dΔi
        dE_under_dΔi = -p_ovun5 * dE_under_tm1 * dΔ_lpcorr_dΔi * inv5_sq_inv6

        Atomix.@atomic S_dE_dΔ[i] += dE_over_dΔi + dE_under_dΔi
    end
end


function over_under_coordination!(
    inter::ReaxFF{ReaxFF2008, T},
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T
    nbrs = inter.state.nbrs
    atid = inter.state.sys_atid 
    bndp = inter.device_params.bond_parameters
    atmp = inter.device_params.atom_parameters
    genp = inter.device_params.general_parameters
    n_atoms = inter.state.n_atoms
    fill_pos = inter.state.fill_pos

    Δ = inter.state.Δ
    Δ_lp = inter.state.Δ_lp
    BO = inter.state.BO
    BO_π = inter.state.BO_π
    BO_ππ = inter.state.BO_ππ

    overcoordination_energies  = inter.state.thread_atom_energies_1
    undercoordination_energies = inter.state.thread_atom_energies_2
    fill!(overcoordination_energies, zero(T))
    fill!(undercoordination_energies, zero(T))

    dn_lp_dΔi = inter.state.dn_lp_dΔi

    S_dE_dBO = inter.state.S_dE_dBO
    S_dE_dBOπ = inter.state.S_dE_dBOπ
    S_dE_dBOππ = inter.state.S_dE_dBOππ
    S_dE_dΔ = inter.state.S_dE_dΔ

    reverse_nbrs = inter.state.reverse_nbrs

    kernel = ovun_kernel(backend)
    kernel(
        fill_pos,
        nbrs,
        bndp,
        atid,
        Δ,
        Δ_lp,
        BO_π,
        BO_ππ,
        BO,
        genp,
        atmp,
        reverse_nbrs,
        S_dE_dBO,
        S_dE_dBOπ,
        S_dE_dBOππ,
        dn_lp_dΔi,
        S_dE_dΔ,
        overcoordination_energies,
        undercoordination_energies;
        ndrange=n_atoms,
        workgroupsize=backend_workgroupsize(backend, n_atoms, n_threads)
    )

    E_overcoordination = sum(overcoordination_energies)
    E_undercoordination = sum(undercoordination_energies)
    inter.state.energy += E_overcoordination
    inter.state.energy += E_undercoordination
    scalar_setindex!(inter.state.energy_components,
                    E_overcoordination + E_undercoordination, 2)
end


@inline function calculate_theta_angle(
    r_ij::SVector{3},
    r_kj::SVector{3},
    ::Type{T}
) where T
    vec_dot = dot(r_ij, r_kj)
    norm_ij = norm(r_ij)
    norm_kj = norm(r_kj)
    inv_norm_mul = inv(norm_ij * norm_kj)
    cos_ang = vec_dot * inv_norm_mul

    theta = acos_bound(cos_ang)  # radians

    sin2 = one(T) - cos_ang^2
    sin2 = max(sin2, T(1e-12))
    sin_ang = sqrt(sin2)

    pre_f = -inv(sin_ang) * inv_norm_mul
    dtheta_qt1 = vec_dot * inv(norm_ij)^2
    dtheta_qt2 = vec_dot * inv(norm_kj)^2

    dtheta_dri = pre_f * (r_kj - (dtheta_qt1 * r_ij))
    dtheta_drj = -pre_f * (r_ij + r_kj - (dtheta_qt1 * r_ij) - (dtheta_qt2 * r_kj))
    dtheta_drk = pre_f * (r_ij - (dtheta_qt2 * r_kj))

    return theta, dtheta_dri, dtheta_drj, dtheta_drk
end


# identifies all valid angles with defined angle parameters and which satisfy bond order constraints
# writes j, idx_i, idx_k, angle parameter index for each valid angle
@kernel function construct_valid_angles!(
    valid_angles,
    n_valid_angles,
    fill_pos,
    nbrs,
    BO,
    valence_cutoff,
    valence_prod_cutoff,
    angle_entries,
    angp,
    atid,
    ::Val{smoothness_correction},
) where {smoothness_correction}
    ti = @index(Global, Linear)

    @inbounds begin
        max_nbrs = size(nbrs, 1)

        # one bond j-i per thread (in angle i-j-k)
        j = (ti - 1) ÷ max_nbrs + 1
        idx_i = (ti - 1) % max_nbrs + 1

        if idx_i < fill_pos[j]  # must be at least one neighbor after i
            i = nbrs[idx_i, j]

            BO_ij_raw = BO[idx_i, j]
            BO_ij = BO_ij_raw - valence_cutoff

            if BO_ij > 0
                for idx_k in (idx_i + 1):fill_pos[j]
                    k = nbrs[idx_k, j]

                    type_i, type_k = minmax(atid[i], atid[k])
                    ang_info = angle_entries[type_i, atid[j], type_k]
                    start_index = ang_info.index
                    end_index = start_index + ang_info.num_entries - 1

                    if start_index > 0
                        BO_jk_raw = BO[idx_k, j]
                        BO_jk = BO_jk_raw - valence_cutoff

                        if (BO_jk > zero(BO_jk)) && (smoothness_correction || BO_ij_raw * BO_jk_raw > valence_prod_cutoff)
                            for p_idx in start_index:end_index
                                angp_ijk = angp[p_idx]

                                if angp_ijk.p_val1 > 0.001
                                    valid_angle_idx = Atomix.@atomic n_valid_angles[1] += 1

                                    # if overflow, continue counting so can resize later
                                    if valid_angle_idx <= size(valid_angles, 2)
                                        valid_angles[1, valid_angle_idx] = j
                                        valid_angles[2, valid_angle_idx] = idx_i
                                        valid_angles[3, valid_angle_idx] = idx_k
                                        valid_angles[4, valid_angle_idx] = p_idx
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

end


@kernel function valence_angles_kernel(
    valid_angles,
    atmp,
    atid,
    genp,
    angp,
    angle_entries,
    Δ_e_mod2,
    n_lp,
    dn_lp_dΔi,
    Δ_angle,
    Δ,
    Δ_boc,
    coordn,
    fill_pos,
    nbrs,
    reverse_nbrs,
    BO_π,
    BO_ππ,
    BO,
    valence_cutoff,
    valence_prod_cutoff,
    coords,
    boundary,
    conv_factor::T,
    deg_to_rad,
    val_energies,
    pen_energies,
    coa_energies,
    S_dE_dBO,
    S_dE_dBOπ,
    S_dE_dBOππ,
    S_dE_dΔ,
    forces,
    ::Val{smoothness_correction},
) where {T, smoothness_correction}
    ti = @index(Global, Linear)

    @inbounds begin
        j     = valid_angles[1, ti]
        idx_i = valid_angles[2, ti]
        idx_k = valid_angles[3, ti]
        p_idx = valid_angles[4, ti]

        p_val6 = genp.p_val6
        p_val8 = genp.p_val8
        p_val9 = genp.p_val9
        p_val10 = genp.p_val10
        p_pen2 = genp.p_pen2
        p_pen3 = genp.p_pen3
        p_pen4 = genp.p_pen4
        p_coa2 = genp.p_coa2
        p_coa3 = genp.p_coa3
        p_coa4 = genp.p_coa4

        p_val3 = atmp[atid[j]].p_val3
        p_val5 = atmp[atid[j]].p_val5

        SBO_sum = zero(T)
        SBO_prdt = one(T)

        if Δ_e_mod2[j] >= 0
            n_lp_ef = zero(T)
            dn_lp_dΔi_ef = zero(T)
        else
            n_lp_ef = n_lp[j]
            dn_lp_dΔi_ef = dn_lp_dΔi[j]
        end

        sbo_sm1 = -Δ_angle[j] - p_val8 * n_lp_ef

        for idx_n in 1:fill_pos[j]
            n = nbrs[idx_n, j]

            SBO_sum += (BO_π[idx_n, j] + BO_ππ[idx_n, j])
            SBO_prdt *= exp(-(BO[idx_n, j]^8))
        end

        SBO = SBO_sum + (one(T) - SBO_prdt) * sbo_sm1

        dSBO_pref = SBO_prdt * T(8.0) * sbo_sm1
        dSBO_dΔj = (SBO_prdt - one(T)) * (one(T) + p_val8 * dn_lp_dΔi_ef)

        if SBO <= zero(T)
            SBO2 = zero(T)
            dSBO2 = zero(T)
        elseif SBO < 1.0
            SBO2 = SBO^p_val9
            dSBO2 = p_val9 * SBO^(p_val9 - one(T))
        elseif SBO < 2.0
            SBO2 = T(2.0) - (T(2.0) - SBO)^p_val9
            dSBO2 = p_val9 * (T(2.0) - SBO)^(p_val9 - one(T))
        else
            SBO2 = T(2.0)
            dSBO2 = zero(T)
        end

        val_exp1 = exp(p_val6 * Δ_angle[j])

        i = nbrs[idx_i, j]

        idx_j_i = reverse_nbrs[idx_i, j]

        BO_ij_raw = BO[idx_i, j]
        BO_ij = BO_ij_raw - valence_cutoff
        if BO_ij > zero(T)

            r_vec_ij = map(x -> ustrip(x) * conv_factor, vector(coords[j], coords[i], boundary))

            k = nbrs[idx_k, j]

            ang_info = angle_entries[atid[i], atid[j], atid[k]]
            start_index = ang_info.index

            if start_index > 0
                idx_j_k = reverse_nbrs[idx_k, j]

                BO_jk_raw = BO[idx_k, j]
                BO_jk = BO_jk_raw - valence_cutoff
                if (BO_jk > zero(T)) && (smoothness_correction || BO_ij_raw * BO_jk_raw > valence_prod_cutoff)
                    val_exp2 = exp(-p_val10 * (T(2.0) - SBO2))
                    val_sm1 = p_val5 - one(T)
                    val_sm2 = T(2.0) + val_exp1

                    r_vec_kj = map(x -> ustrip(x) * conv_factor, vector(coords[j], coords[k], boundary))
                        
                    Theta_ijk, dtheta_dri, dtheta_drj, dtheta_drk = calculate_theta_angle(
                        r_vec_ij,
                        r_vec_kj,
                        T
                    )

                    angp_ijk = angp[p_idx]

                    if angp_ijk.p_val1 > 0.001
                        if smoothness_correction
                            taper_ij, dtaper_ij = smoothness_taper(BO_ij_raw, valence_cutoff, 4 * valence_cutoff)
                            taper_jk, dtaper_jk = smoothness_taper(BO_jk_raw, valence_cutoff, 4 * valence_cutoff)
                            taper = taper_ij * taper_jk
                        end

                        # Angle energy
                        Theta_oo = angp_ijk.Theta_oo
                        p_val1   = angp_ijk.p_val1
                        p_val2   = angp_ijk.p_val2
                        p_val4   = angp_ijk.p_val4
                        p_val7   = angp_ijk.p_val7

                        Theta_o_degrees = T(180.0) - Theta_oo * (one(T) - val_exp2)
                        Theta_o = Theta_o_degrees * deg_to_rad

                        dTheta_o_dΔj = Theta_oo * val_exp2 * p_val10 * dSBO2 * dSBO_dΔj * deg_to_rad
        
                        val_exp3 = exp(-p_val7 * Δ_angle[j])
                        val_arg_3 = one(T) + val_exp1 + val_exp3
                        val_inv1 = inv(val_arg_3)
                        
                        f8 = p_val5 - (val_sm1 * val_sm2 * val_inv1)
                        df8_dΔj = -(p_val5 - one(T)) * (val_exp1 * p_val6 * val_arg_3 - val_sm2 * (val_exp1 * p_val6 + val_exp3 * -p_val7)) * val_inv1 * val_inv1

                        val_arg1 = p_val3 * (BO_ij)^(p_val4 - one(T))   
                        val_arg2 = p_val3 * (BO_jk)^(p_val4 - one(T))
                        f7_exp1 = exp(-val_arg1 * BO_ij)
                        f7_exp2 = exp(-val_arg2 * BO_jk)

                        f7_ij = one(T) - f7_exp1
                        f7_jk = one(T) - f7_exp2
                        df7_ij_dBO = f7_exp1 * val_arg1 * p_val4
                        df7_jk_dBO = f7_exp2 * val_arg2 * p_val4

                        if smoothness_correction
                            f7_ij = -expm1(-val_arg1 * BO_ij)
                            f7_jk = -expm1(-val_arg2 * BO_jk)
                            df7_ij_dBO = df7_ij_dBO * taper_ij + f7_ij * dtaper_ij
                            df7_jk_dBO = df7_jk_dBO * taper_jk + f7_jk * dtaper_jk
                            f7_ij *= taper_ij
                            f7_jk *= taper_jk
                        end

                        f7_ij_f7_jk = f7_ij * f7_jk
                        f7_ij_f7_jk_f8 = f7_ij_f7_jk * f8

                        engy_exp1 = exp(-p_val2 * (Theta_o - Theta_ijk)^2)
                        engy_prd1 = p_val1 * (one(T) - engy_exp1)
                        angle_engy = f7_ij_f7_jk_f8 * engy_prd1


                        Atomix.@atomic val_energies[j] += angle_engy

                        # Penalty energy
                        p_pen1 = angp_ijk.p_pen1
                        
                        pen_exp1 = exp(-p_pen3 * Δ[j])
                        pen_exp2 = exp(p_pen4 * Δ[j])
                        pen_sm1 = one(T) + pen_exp1 + pen_exp2
                        pen_inv1 = inv(pen_sm1)
                        f9 = (T(2.0) + pen_exp1) * pen_inv1
                        df9_dΔj = (pen_exp1 * -p_pen3 * pen_sm1 - (T(2.0) + pen_exp1) * (pen_exp1 * -p_pen3 + pen_exp2 * p_pen4)) * pen_inv1 * pen_inv1

                        pen_pref = p_pen1 * exp(-p_pen2 * (BO_ij - T(2.0))^2) * exp(-p_pen2 * (BO_jk - T(2.0))^2)

                        penalty_engy = f9 * pen_pref

                        if smoothness_correction
                            penalty_engy_raw = penalty_engy
                            pen_pref *= taper
                            penalty_engy *= taper
                        end

                        Atomix.@atomic pen_energies[j] += penalty_engy

                        # Three-body conjugation term
                        p_coa1 = angp_ijk.p_coa1
                        
                        coa_exp1 = exp(p_coa2 * Δ_boc[j])
                        inv_tbc1 = inv(one(T) + coa_exp1)
                        coa_arg1 = -BO_ij + coordn[i]
                        coa_arg2 = -BO_jk + coordn[k]
                        exp_tbc2 = exp(-p_coa3 * coa_arg1^2)
                        exp_tbc3 = exp(-p_coa3 * coa_arg2^2)
                        exp_tbc4 = exp(-p_coa4 * (BO_ij - T(1.5))^2)
                        exp_tbc5 = exp(-p_coa4 * (BO_jk - T(1.5))^2)
                        coa_pref = p_coa1 * inv_tbc1 * exp_tbc2 * exp_tbc3 * exp_tbc4 * exp_tbc5

                        if smoothness_correction
                            coa_pref_raw = coa_pref
                            coa_pref *= taper
                        end

                        Atomix.@atomic coa_energies[j] += coa_pref

                        # Write forces for E_val, E_pen and E_coa
                        dE_pen_pref = penalty_engy * -p_pen2 * T(2.0)
                        dE_pen_dBOij = dE_pen_pref * (BO_ij - T(2.0))
                        dE_pen_dBOjk = dE_pen_pref * (BO_jk - T(2.0))

                        if smoothness_correction
                            dE_pen_dBOij += penalty_engy_raw * dtaper_ij * taper_jk
                            dE_pen_dBOjk += penalty_engy_raw * taper_ij * dtaper_jk
                        end

                        dE_val_dBOij = df7_ij_dBO * f7_jk * f8 * engy_prd1
                        dE_val_dBOjk = f7_ij * df7_jk_dBO * f8 * engy_prd1

                        dE_coa_dBOij = coa_pref * T(2.0) * (p_coa3 * coa_arg1 - p_coa4 * (BO_ij - T(1.5)))
                        dE_coa_dBOjk = coa_pref * T(2.0) * (p_coa3 * coa_arg2 - p_coa4 * (BO_jk - T(1.5)))

                        if smoothness_correction
                            dE_coa_dBOij += coa_pref_raw * dtaper_ij * taper_jk
                            dE_coa_dBOjk += coa_pref_raw * taper_ij * dtaper_jk
                        end

                        dE_tot_sum1 = dE_val_dBOij + dE_pen_dBOij + dE_coa_dBOij
                        Atomix.@atomic S_dE_dBO[idx_i, j] += dE_tot_sum1
                        Atomix.@atomic S_dE_dBO[idx_j_i, i] += dE_tot_sum1

                        dE_tot_sum2 = dE_val_dBOjk + dE_pen_dBOjk + dE_coa_dBOjk
                        Atomix.@atomic S_dE_dBO[idx_k, j] += dE_tot_sum2
                        Atomix.@atomic S_dE_dBO[idx_j_k, k] += dE_tot_sum2
                        
                        dE_val_prod = p_val1 * engy_exp1 * p_val2 * T(2.0) * (Theta_o - Theta_ijk)

                        dE_val_dΔj = f7_ij_f7_jk * (df8_dΔj * engy_prd1 +
                                    f8 * dE_val_prod * dTheta_o_dΔj)

                        dE_pen_dΔj = df9_dΔj * pen_pref

                        dE_coa_dΔj = coa_pref * -inv_tbc1 * coa_exp1 * p_coa2
                        dE_coa_dΔi = coa_pref * -p_coa3 * T(2.0) * coa_arg1
                        dE_coa_dΔk = coa_pref * -p_coa3 * T(2.0) * coa_arg2

                        Atomix.@atomic S_dE_dΔ[j] += dE_val_dΔj + dE_pen_dΔj + dE_coa_dΔj
                        Atomix.@atomic S_dE_dΔ[i] += dE_coa_dΔi
                        Atomix.@atomic S_dE_dΔ[k] += dE_coa_dΔk

                        dE_val_dTheta_ijk = f7_ij_f7_jk_f8 * -dE_val_prod

                        write_force_Θ(
                            forces,
                            i,
                            j,
                            k,
                            dE_val_dTheta_ijk,
                            dtheta_dri,
                            dtheta_drj,
                            dtheta_drk
                        )

                        dE_val_prod2 = dE_val_prod * Theta_oo * val_exp2 * p_val10 * dSBO2 * deg_to_rad

                        for idx_n in 1:fill_pos[j]
                            n = nbrs[idx_n, j]

                            idx_j_n = reverse_nbrs[idx_n, j]

                            dE_val_dBO = f7_ij_f7_jk_f8 * dE_val_prod2 * dSBO_pref * BO[idx_n, j]^7 
                            dE_val_dBOπ = f7_ij_f7_jk_f8 * dE_val_prod2
                            
                            Atomix.@atomic S_dE_dBO[idx_n, j] += dE_val_dBO
                            Atomix.@atomic S_dE_dBO[idx_j_n, n] += dE_val_dBO
                            Atomix.@atomic S_dE_dBOπ[idx_n, j]  += dE_val_dBOπ
                            Atomix.@atomic S_dE_dBOπ[idx_j_n, n] += dE_val_dBOπ
                            Atomix.@atomic S_dE_dBOππ[idx_n, j]  += dE_val_dBOπ
                            Atomix.@atomic S_dE_dBOππ[idx_j_n, n] += dE_val_dBOπ   
                        end
                    end
                end
            end
        end
    end
end


function valence_angles!(
    inter::ReaxFF{ReaxFF2008, T},
    sys::System,
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T
    genp = inter.device_params.general_parameters
    atmp = inter.device_params.atom_parameters
    atid = inter.state.sys_atid
    angp = inter.device_params.angle_parameters
    angle_entries = inter.device_params.angle_entries
    fill_pos = inter.state.fill_pos
    BO = inter.state.BO
    BO_π = inter.state.BO_π
    BO_ππ = inter.state.BO_ππ
    Δ = inter.state.Δ
    Δ_boc = inter.state.Δ_boc
    Δ_angle = inter.state.Δ_angle
    n_lp = inter.state.n_lp
    nbrs = inter.state.nbrs
    coordn = inter.state.coordn
    Δ_e_mod2 = inter.state.Δ_e_mod2
    dn_lp_dΔi = inter.state.dn_lp_dΔi
    S_dE_dBO = inter.state.S_dE_dBO
    S_dE_dBOπ = inter.state.S_dE_dBOπ
    S_dE_dBOππ = inter.state.S_dE_dBOππ
    S_dE_dΔ = inter.state.S_dE_dΔ

    valence_cutoff = inter.device_params.global_parameters.valence_cutoff
    valence_prod_cutoff = inter.device_params.global_parameters.valence_prod_cutoff
    reverse_nbrs = inter.state.reverse_nbrs

    val_energies = inter.state.thread_atom_energies_1
    pen_energies = inter.state.thread_atom_energies_2
    coa_energies = inter.state.thread_atom_energies_3
    fill!(val_energies, zero(T))
    fill!(pen_energies, zero(T))
    fill!(coa_energies, zero(T))

    deg_to_rad = T(3.14159265 / 180.0)

    conv_factor = inter.use_units ? T(ustrip(uconvert(u"Å", 1.0 * inter.sys_coord_unit))) : T(10.0)

    fill!(inter.state.n_valid_angles, 0)

    valid_angles_kernel! = construct_valid_angles!(backend)
    if !isempty(nbrs)
        valid_angles_kernel!(
            inter.state.valid_angles,
            inter.state.n_valid_angles,
            fill_pos,
            nbrs,
            BO,
            valence_cutoff,
            valence_prod_cutoff,
            angle_entries,
            angp,
            atid,
            inter.smoothness_correction;
            ndrange=length(nbrs),
            workgroupsize=backend_workgroupsize(backend, length(nbrs), n_threads)
        )
        synchronize(backend)
    end

    n_valid_angles = scalar_getindex(inter.state.n_valid_angles, 1)

    if n_valid_angles > size(inter.state.valid_angles, 2)
        new_size = max(n_valid_angles, 2 * size(inter.state.valid_angles, 2))
        inter.state.valid_angles = similar(inter.state.valid_angles, 4, new_size)
        fill!(inter.state.n_valid_angles, 0)
        valid_angles_kernel!(
            inter.state.valid_angles,
            inter.state.n_valid_angles,
            fill_pos,
            nbrs,
            BO,
            valence_cutoff,
            valence_prod_cutoff,
            angle_entries,
            angp,
            atid,
            inter.smoothness_correction;
            ndrange=length(nbrs),
            workgroupsize=backend_workgroupsize(backend, length(nbrs), n_threads)
        )
        synchronize(backend)
        n_valid_angles = scalar_getindex(inter.state.n_valid_angles, 1)
    end

    if n_valid_angles > 0
        kernel = valence_angles_kernel(backend)
        kernel(
            inter.state.valid_angles,
            atmp,
            atid,
            genp,
            angp,
            angle_entries,
            Δ_e_mod2,
            n_lp,
            dn_lp_dΔi,
            Δ_angle,
            Δ,
            Δ_boc,
            coordn,
            fill_pos,
            nbrs,
            reverse_nbrs,
            BO_π,
            BO_ππ,
            BO,
            valence_cutoff,
            valence_prod_cutoff,
            sys.coords,
            sys.boundary,
            conv_factor::T,
            deg_to_rad,
            val_energies,
            pen_energies,
            coa_energies,
            S_dE_dBO,
            S_dE_dBOπ,
            S_dE_dBOππ,
            S_dE_dΔ,
            inter.state.forces,
            inter.smoothness_correction,
            ndrange=n_valid_angles,
            workgroupsize=backend_workgroupsize(backend, n_valid_angles, n_threads)
        )
    end

    E_val = sum(val_energies)
    E_pen = sum(pen_energies)
    E_coa = sum(coa_energies)
    inter.state.energy += E_val
    inter.state.energy += E_pen
    inter.state.energy += E_coa
    scalar_setindex!(inter.state.energy_components, E_val, 4)
    scalar_setindex!(inter.state.energy_components, E_pen, 5)
    scalar_setindex!(inter.state.energy_components, E_coa, 6)
end


@inline function calculate_omega_angle(
    r_i::SVector{3}, 
    r_j::SVector{3}, 
    r_k::SVector{3},
    r_l::SVector{3},
    conv_factor::T,
    boundary::AbstractBoundary,
) where T
    r_ij = map(x -> ustrip(x) * conv_factor, vector(r_j, r_i, boundary))
    r_jk = map(x -> ustrip(x) * conv_factor, vector(r_k, r_j, boundary))
    r_kl = map(x -> ustrip(x) * conv_factor, vector(r_l, r_k, boundary))

    cross_ij_jk = r_ij × r_jk
    cross_jk_kl = r_jk × r_kl

    r_jk_sq = dot(r_jk, r_jk)
    r_jk_norm = sqrt(r_jk_sq)

    y_val = r_jk_norm * dot(cross_ij_jk, r_kl)
    x_val = dot(cross_ij_jk, cross_jk_kl)

    hyp_sq = x_val^2 + y_val^2
    hyp_inv = inv(sqrt(max(hyp_sq, T(1e-20))))

    sin_ω = y_val * hyp_inv
    cos_ω = x_val * hyp_inv

    inv_r_jk_sq = inv(r_jk_sq)
    inv_norm_sq_cross_ij_jk = inv(max(dot(cross_ij_jk, cross_ij_jk), T(1e-12)))
    inv_norm_sq_cross_jk_kl = inv(max(dot(cross_jk_kl, cross_jk_kl), T(1e-12)))

    dω_dri = r_jk_norm * cross_ij_jk * inv_norm_sq_cross_ij_jk
    dω_drl = -r_jk_norm * cross_jk_kl * inv_norm_sq_cross_jk_kl
    v = inv_r_jk_sq * (-dot(r_ij, r_jk) * dω_dri + dot(r_kl, r_jk) * dω_drl)
    dω_drj =  v - dω_dri
    dω_drk = -v - dω_drl

    return sin_ω, cos_ω, dω_dri, dω_drj, dω_drk, dω_drl
end


# identifies all valid torsions with defined torsion parameters and which satisfy bond order constraints
# writes j, idx_k, idx_i, idx_l into valid_torsions for each valid i-j-k-l torsion
@kernel function construct_valid_torsions!(
    valid_torsions,
    n_valid_torsions,
    fill_pos,
    nbrs,
    BO,
    valence_cutoff,
    torp,
    atid,
    ::Val{smoothness_correction},
) where {smoothness_correction}
    ti = @index(Global, Linear)

    @inbounds begin
        max_nbrs = size(nbrs, 1)

        # one central bond j-k per thread
        j = (ti - 1) ÷ max_nbrs + 1
        idx_k = (ti - 1) % max_nbrs + 1
        
        if idx_k <= fill_pos[j]
            k = nbrs[idx_k, j]

            if k > j    # avoid double counting
                BO_jk_raw = BO[idx_k, j]
                BO_jk = BO_jk_raw - valence_cutoff

                if BO_jk > zero(BO_jk)
                    # loop through neighbors i of j
                    for idx_i in 1:fill_pos[j]
                        i = nbrs[idx_i, j]

                        if i != k   # atom i must not be atom k in i-j-k
                            BO_ij_raw = BO[idx_i, j]
                            BO_ij = BO_ij_raw - valence_cutoff

                            if BO_ij > zero(BO_ij)
                                # loop through neighbors l of k
                                for idx_l in 1:fill_pos[k]
                                    l = nbrs[idx_l, k]

                                    if l != j && l != i   # atom l must not be i or j in i-j-k-l
                                        # check if valid torsion
                                        torp_ijkl = torp[atid[i], atid[j], atid[k], atid[l]]

                                        if torp_ijkl.is_defined
                                            BO_kl_raw = BO[idx_l, k]
                                            BO_kl = BO_kl_raw - valence_cutoff

                                            if BO_kl > zero(BO_kl) && (smoothness_correction || BO_ij_raw * BO_jk_raw * BO_kl_raw > valence_cutoff)
                                                valid_torsion_idx = Atomix.@atomic n_valid_torsions[1] += 1

                                                if valid_torsion_idx <= size(valid_torsions, 2)
                                                    valid_torsions[1, valid_torsion_idx] = j
                                                    valid_torsions[2, valid_torsion_idx] = idx_k
                                                    valid_torsions[3, valid_torsion_idx] = idx_i
                                                    valid_torsions[4, valid_torsion_idx] = idx_l
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


@kernel function torsions_kernel(
    valid_torsions,
    nbrs,
    BO,
    valence_cutoff,
    reverse_nbrs,
    Δ_angle,
    genp,
    conv_factor::T,
    coords,
    boundary,
    torp,
    atid,
    torsion_energies,
    conj_energies,
    forces,
    S_dE_dΔ,
    S_dE_dBO,
    S_dE_dBOπ,
    BO_π,
    ::Val{smoothness_correction},
) where {T, smoothness_correction}
    ti = @index(Global, Linear)

    @inbounds begin
        j     = valid_torsions[1, ti]
        idx_k = valid_torsions[2, ti]
        idx_i = valid_torsions[3, ti]
        idx_l = valid_torsions[4, ti]

        p_tor2 = genp.p_tor2
        p_tor3 = genp.p_tor3
        p_tor4 = genp.p_tor4
        p_cot2 = genp.p_cot2

        k = nbrs[idx_k, j]
        if k > j 
            BO_jk_raw = BO[idx_k, j]
            BO_jk = BO_jk_raw - valence_cutoff
            if BO_jk > zero(T)

                idx_j_k = reverse_nbrs[idx_k, j]

                f11_ang_sum = Δ_angle[j] + Δ_angle[k]

                f10_exp2 = exp(-p_tor2 * BO_jk)
                f11_exp1 = exp(-p_tor3 * f11_ang_sum)
                f11_exp2 = exp(p_tor4 * f11_ang_sum)
                f11_sum1 = one(T) + f11_exp1 + f11_exp2
                f11_inv1 = inv(f11_sum1)

                f11 = (T(2.0) + f11_exp1) * f11_inv1

                inv_f11_exp1 = inv(f11_exp1)
                exp_quot = f11_exp2 * inv_f11_exp1
                df11_sum = inv_f11_exp1 + one(T) + exp_quot
                inv_df11_sum = inv(df11_sum)
                
                df11_dΔjk = ((-p_tor3 * df11_sum) - (T(2.0) * inv_f11_exp1 + one(T)) * (-p_tor3 + p_tor4 * exp_quot)) * inv_df11_sum * inv_df11_sum


                f12_exp2 = exp(-p_cot2 * (BO_jk - T(1.5))^2)

                r_vec_kj = map(x -> ustrip(x) * conv_factor, vector(coords[j], coords[k], boundary))

                i = nbrs[idx_i, j]
                if i != k 
                    BO_ij_raw = BO[idx_i, j]
                    BO_ij = BO_ij_raw - valence_cutoff
                    if BO_ij > zero(T)
                        idx_j_i = reverse_nbrs[idx_i, j]

                        r_vec_ij = map(x -> ustrip(x) * conv_factor, vector(coords[j], coords[i], boundary))
                        
                        Θ_ijk, dΘ_ijk_dri_buf, dΘ_ijk_drj_buf, dΘ_ijk_drk_buf = calculate_theta_angle(
                            r_vec_ij, 
                            r_vec_kj,
                            T
                        )

                        sin_Θ_ijk, cos_Θ_ijk = sincos(Θ_ijk)

                        f10_exp1 = exp(-p_tor2 * BO_ij)
                        f12_exp1 = exp(-p_cot2 * (BO_ij - T(1.5))^2)

                        l = nbrs[idx_l, k]
                        if (l != j && l != i) 

                            torp_ijkl = torp[atid[i], atid[j], atid[k], atid[l]]
                            # !isnothing(torp_ijkl) || continue
                            if torp_ijkl.is_defined
                                BO_kl_raw = BO[idx_l, k]
                                BO_kl = BO_kl_raw - valence_cutoff
                                if BO_kl > zero(T)
                                    if smoothness_correction || BO_ij_raw * BO_jk_raw * BO_kl_raw > valence_cutoff
                                        idx_k_l = reverse_nbrs[idx_l, k]

                                        if smoothness_correction
                                            taper_ij, dtaper_ij = smoothness_taper(BO_ij_raw, valence_cutoff, 4 * valence_cutoff)
                                            taper_jk, dtaper_jk = smoothness_taper(BO_jk_raw, valence_cutoff, 4 * valence_cutoff)
                                            taper_kl, dtaper_kl = smoothness_taper(BO_kl_raw, valence_cutoff, 4 * valence_cutoff)
                                            taper = taper_ij * taper_jk * taper_kl
                                        end

                                        V1 = torp_ijkl.V1
                                        V2 = torp_ijkl.V2
                                        V3 = torp_ijkl.V3
                                        p_tor1 = torp_ijkl.p_tor1
                                        p_cot1 = torp_ijkl.p_cot1

                                        r_vec_lk = map(x -> ustrip(x) * conv_factor, vector(coords[k], coords[l], boundary))

                                        Θ_jkl, dΘ_jkl_drj_buf, dΘ_jkl_drk_buf, dΘ_jkl_drl_buf = calculate_theta_angle(
                                            -r_vec_kj,
                                            r_vec_lk,
                                            T
                                        )

                                        sin_Θ_jkl, cos_Θ_jkl = sincos(Θ_jkl)

                                        f10_exp3 = exp(-p_tor2 * BO_kl)
                                        f10_sum1 = one(T) - f10_exp1
                                        f10_sum2 = one(T) - f10_exp2
                                        f10_sum3 = one(T) - f10_exp3

                                        f10 = f10_sum1 * f10_sum2 * f10_sum3

                                        sin_ω, cos_ω, dω_dri, dω_drj, dω_drk, dω_drl = calculate_omega_angle(
                                            coords[i],
                                            coords[j],
                                            coords[k],
                                            coords[l],
                                            conv_factor,
                                            boundary,
                                        )

                                        cos_2ω = T(2.0) * cos_ω^2 - one(T)
                                        cos_3ω = T(4.0) * cos_ω^3 - T(3.0) * cos_ω
                                        sin_2ω = T(2.0) * sin_ω * cos_ω
                                        sin_3ω = T(3.0) * sin_ω - T(4.0) * sin_ω^3

                                        sin_prod = sin_Θ_ijk * sin_Θ_jkl

                                        tor_sum1 = T(2.0) - BO_π[idx_k, j] - f11
                                        tor_exp1 = exp(p_tor1 * tor_sum1^2)
                                        tor_V = T(0.5) * (V1 * (one(T) + cos_ω) + V2 * tor_exp1 * (one(T) - cos_2ω) + V3 * (one(T) + cos_3ω))

                                        sin_prod_tor_V = sin_prod * tor_V
                                        torsion_engy = f10 * sin_prod_tor_V

                                        if smoothness_correction
                                            torsion_engy_raw = torsion_engy
                                            f10 *= taper
                                            torsion_engy *= taper
                                        end
                                        
                                        Atomix.@atomic torsion_energies[j] += torsion_engy

                                        f12_exp3 = exp(-p_cot2 * (BO_kl - T(1.5))^2) 
                                        f12 = f12_exp1 * f12_exp2 * f12_exp3
                                        conj_cos_2 = cos_ω^2 - one(T)
                                        conj_sum1 = one(T) + conj_cos_2 * sin_prod
                                        conj_engy = f12 * p_cot1 * conj_sum1

                                        if smoothness_correction
                                            conj_engy_raw = conj_engy
                                            f12 *= taper
                                            conj_engy *= taper
                                        end

                                        Atomix.@atomic conj_energies[j] += conj_engy 

                                        # Write forces of E_tors and E_conj
                                        dE_tors_pref = f10 * sin_prod * T(0.5) * V2 * tor_exp1 * (one(T) - cos_2ω) * p_tor1 * T(2.0) * tor_sum1
                                        dE_tors_dΔjk = dE_tors_pref * -df11_dΔjk
                                        
                                        Atomix.@atomic S_dE_dΔ[j] += dE_tors_dΔjk
                                        Atomix.@atomic S_dE_dΔ[k] += dE_tors_dΔjk

                                        p_tor_sin_prod_tor_V = p_tor2 * sin_prod_tor_V

                                        dE_tors_dBOij = f10_exp1 * f10_sum2 * f10_sum3 * p_tor_sin_prod_tor_V
                                        dE_tors_dBOjk = f10_sum1 * f10_exp2 * f10_sum3 * p_tor_sin_prod_tor_V
                                        dE_tors_dBOkl = f10_sum1 * f10_sum2 * f10_exp3 * p_tor_sin_prod_tor_V

                                        if smoothness_correction
                                            dE_tors_dBOij = dE_tors_dBOij * taper + torsion_engy_raw * dtaper_ij * taper_jk * taper_kl
                                            dE_tors_dBOjk = dE_tors_dBOjk * taper + torsion_engy_raw * taper_ij * dtaper_jk * taper_kl
                                            dE_tors_dBOkl = dE_tors_dBOkl * taper + torsion_engy_raw * taper_ij * taper_jk * dtaper_kl
                                        end

                                        dE_tors_dBOπjk = -dE_tors_pref

                                        dE_conj_pref = conj_engy * -p_cot2 * T(2.0)
                                        dE_conj_dBOij = dE_conj_pref * (BO_ij - T(1.5))
                                        dE_conj_dBOjk = dE_conj_pref * (BO_jk - T(1.5))
                                        dE_conj_dBOkl = dE_conj_pref * (BO_kl - T(1.5))

                                        if smoothness_correction
                                            dE_conj_dBOij += conj_engy_raw * dtaper_ij * taper_jk * taper_kl
                                            dE_conj_dBOjk += conj_engy_raw * taper_ij * dtaper_jk * taper_kl
                                            dE_conj_dBOkl += conj_engy_raw * taper_ij * taper_jk * dtaper_kl
                                        end

                                        dE_tot_sum1 = dE_tors_dBOij + dE_conj_dBOij
                                        Atomix.@atomic S_dE_dBO[idx_i, j] += dE_tot_sum1
                                        Atomix.@atomic S_dE_dBO[idx_j_i, i] += dE_tot_sum1

                                        dE_tot_sum2 = dE_tors_dBOjk + dE_conj_dBOjk
                                        Atomix.@atomic S_dE_dBO[idx_k, j] += dE_tot_sum2
                                        Atomix.@atomic S_dE_dBO[idx_j_k, k] += dE_tot_sum2
                                        Atomix.@atomic S_dE_dBOπ[idx_k, j]  += dE_tors_dBOπjk
                                        Atomix.@atomic S_dE_dBOπ[idx_j_k, k] += dE_tors_dBOπjk

                                        dE_tot_sum3 = dE_tors_dBOkl + dE_conj_dBOkl
                                        Atomix.@atomic S_dE_dBO[idx_l, k] += dE_tot_sum3
                                        Atomix.@atomic S_dE_dBO[idx_k_l, l] += dE_tot_sum3


                                        cos_ijk_sin_jkl = cos_Θ_ijk * sin_Θ_jkl
                                        sin_ijk_cos_jkl = sin_Θ_ijk * cos_Θ_jkl

                                        dE_tors_dΘijk = f10 * cos_ijk_sin_jkl * tor_V
                                        dE_tors_dΘjkl = f10 * sin_ijk_cos_jkl * tor_V

                                        dE_conj_pref = f12 * p_cot1 * conj_cos_2
                                        dE_conj_dΘijk = dE_conj_pref * cos_ijk_sin_jkl
                                        dE_conj_dΘjkl = dE_conj_pref * sin_ijk_cos_jkl

                                        write_force_Θ(
                                            forces,
                                            i,
                                            j,
                                            k,
                                            dE_tors_dΘijk + dE_conj_dΘijk,
                                            dΘ_ijk_dri_buf,
                                            dΘ_ijk_drj_buf,
                                            dΘ_ijk_drk_buf
                                        )
                                        write_force_Θ(
                                            forces,
                                            j,
                                            k,
                                            l,
                                            dE_tors_dΘjkl + dE_conj_dΘjkl,
                                            dΘ_jkl_drj_buf,
                                            dΘ_jkl_drk_buf,
                                            dΘ_jkl_drl_buf
                                        )

                                        dE_tors_dω = f10 * sin_prod * T(0.5) * (
                                                        V1 * -sin_ω + 
                                                        V2 * tor_exp1 * sin_2ω * T(2.0) +
                                                        V3 * -sin_3ω * T(3.0)
                                                    )

                                        dE_conj_dω = f12 * p_cot1 * (T(2.0) * cos_ω * -sin_ω * sin_prod)

                                        write_force_ω(
                                            forces,
                                            i,
                                            j,
                                            k,
                                            l,
                                            dE_tors_dω + dE_conj_dω,
                                            dω_dri,
                                            dω_drj,
                                            dω_drk,
                                            dω_drl
                                        )
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


function torsions!(
    inter::ReaxFF{ReaxFF2008, T},
    sys::System,
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T
    nbrs = inter.state.nbrs
    fill_pos = inter.state.fill_pos

    genp = inter.device_params.general_parameters
    torp = inter.device_params.torsion_parameters

    Δ_angle = inter.state.Δ_angle
    BO = inter.state.BO
    
    BO_π = inter.state.BO_π
    S_dE_dBO = inter.state.S_dE_dBO
    S_dE_dBOπ = inter.state.S_dE_dBOπ
    S_dE_dΔ = inter.state.S_dE_dΔ

    atid = inter.state.sys_atid
    valence_cutoff = inter.device_params.global_parameters.valence_cutoff
    reverse_nbrs = inter.state.reverse_nbrs
 
    torsion_energies = inter.state.thread_atom_energies_1
    conj_energies = inter.state.thread_atom_energies_2
    fill!(torsion_energies, zero(T))
    fill!(conj_energies, zero(T))

    conv_factor = inter.use_units ? T(ustrip(uconvert(u"Å", 1.0 * inter.sys_coord_unit))) : T(10.0)

    fill!(inter.state.n_valid_torsions, 0)

    # this continues counting even when capacity of valid_torsions is reached => reports exact required size
    valid_torsions_kernel! = construct_valid_torsions!(backend)
    if !isempty(nbrs)
        valid_torsions_kernel!(
            inter.state.valid_torsions,
            inter.state.n_valid_torsions,
            fill_pos,
            nbrs,
            BO,
            valence_cutoff,
            torp,
            atid,
            inter.smoothness_correction;
            ndrange=length(nbrs),
            workgroupsize=backend_workgroupsize(backend, length(nbrs), n_threads)
        )
        synchronize(backend)
    end

    n_valid_torsions = scalar_getindex(inter.state.n_valid_torsions, 1)

    # if valid_torsions matrix is too small, rebuild with larger size
    if n_valid_torsions > size(inter.state.valid_torsions, 2)
        new_size = max(n_valid_torsions, 2 * size(inter.state.valid_torsions, 2))
        inter.state.valid_torsions = similar(inter.state.valid_torsions, 4, new_size)
        fill!(inter.state.n_valid_torsions, 0)
        valid_torsions_kernel!(
            inter.state.valid_torsions,
            inter.state.n_valid_torsions,
            fill_pos,
            nbrs,
            BO,
            valence_cutoff,
            torp,
            atid,
            inter.smoothness_correction;
            ndrange=length(nbrs),
            workgroupsize=backend_workgroupsize(backend, length(nbrs), n_threads)
        )
        synchronize(backend)
        n_valid_torsions = scalar_getindex(inter.state.n_valid_torsions, 1)
    end

    if n_valid_torsions > 0
        kernel = torsions_kernel(backend)
        kernel(
            inter.state.valid_torsions,
            nbrs,
            BO,
            valence_cutoff,
            reverse_nbrs,
            Δ_angle,
            genp,
            conv_factor::T,
            sys.coords,
            sys.boundary,
            torp,
            atid,
            torsion_energies,
            conj_energies,
            inter.state.forces,
            S_dE_dΔ,
            S_dE_dBO,
            S_dE_dBOπ,
            BO_π,
            inter.smoothness_correction;
            ndrange=n_valid_torsions,
            workgroupsize=backend_workgroupsize(backend, n_valid_torsions, n_threads)
        )
    end
    
    E_tors = sum(torsion_energies)
    E_conj = sum(conj_energies)
    inter.state.energy += E_tors
    inter.state.energy += E_conj
    scalar_setindex!(inter.state.energy_components, E_tors, 8)
    scalar_setindex!(inter.state.energy_components, E_conj, 9)
end


@kernel function corrections_kernel(
    atid,
    genp,
    C_id,
    O_id,
    fill_pos,
    nbrs,
    reverse_nbrs,
    BO,
    c2_correction,
    Δ,
    S_dE_dBO,
    S_dE_dΔ,
    triple_bond_correction,
    coordn,
    c2_energies::AbstractArray{T},
    trip_energies::AbstractArray{T},
    ::Val{smoothness_correction},
) where {T, smoothness_correction}
    i = @index(Global, Linear)

    @inbounds begin
        if (atid[i] == C_id || atid[i] == O_id) 
            k_c2 = genp.k_c2 
            p_trip1 = genp.p_trip1
            p_trip2 = genp.p_trip2
            p_trip3 = genp.p_trip3
            p_trip4 = genp.p_trip4

            for idx_j in 1:fill_pos[i]
                j = nbrs[idx_j, i]

                if j > i 
                    idx_i = reverse_nbrs[idx_j, i]

                    BO_ij = BO[idx_j, i]

                    # C2 correction
                    if c2_correction && atid[i] == C_id && atid[j] == C_id
                        c2_arg_i = BO_ij - Δ[i] - (T(0.04) * Δ[i]^4)
                        c2_arg_j = BO_ij - Δ[j] - (T(0.04) * Δ[j]^4)

                        if c2_arg_i > 3.0
                            c2_tm1 = k_c2 * (c2_arg_i - T(3.0))

                            c2_energies[i] += c2_tm1 * (c2_arg_i - T(3.0))

                            dE_c2_dBOij = T(2.0) * c2_tm1
                            dE_c2_dΔi = T(2.0) * c2_tm1 * (T(-1.0) - T(0.16) * Δ[i]^3)

                            S_dE_dBO[idx_j, i] += dE_c2_dBOij
                            S_dE_dBO[idx_i, j] += dE_c2_dBOij

                            Atomix.@atomic S_dE_dΔ[i] += dE_c2_dΔi
                        end

                        if c2_arg_j > 3.0
                            c2_tm2 = k_c2 * (c2_arg_j - T(3.0))

                            c2_energies[i] += c2_tm2 * (c2_arg_j - T(3.0))

                            dE_c2_dBOij = T(2.0) * c2_tm2
                            dE_c2_dΔj = T(2.0) * c2_tm2 * (T(-1.0) - T(0.16) * Δ[j]^3)

                            S_dE_dBO[idx_j, i] += dE_c2_dBOij
                            S_dE_dBO[idx_i, j] += dE_c2_dBOij

                            Atomix.@atomic S_dE_dΔ[j] += dE_c2_dΔj
                        end    
                    end

                    # Triple bond energy correction
                    if triple_bond_correction && BO_ij >= (smoothness_correction ? T(0.9) : T(1.0)) && ((atid[i] == C_id && atid[j] == O_id) || (atid[i] == O_id && atid[j] == C_id))
                        trip_exp1 = exp(-p_trip2 * (BO_ij - T(2.5))^2)
                        trip_exp2 = exp(-p_trip4 * (coordn[i] - BO_ij))
                        trip_exp3 = exp(-p_trip4 * (coordn[j] - BO_ij))
                        trip_exp4 = exp(p_trip3 * (Δ[i] + Δ[j]))
                        trip_denom = one(T) + T(25.0) * trip_exp4

                        denom_inv = inv(trip_denom)

                        exp_sum = trip_exp2 + trip_exp3
                        trip_tm2 = exp_sum * denom_inv
                        trip_tm3 = p_trip1 * trip_exp1

                        if smoothness_correction
                            taper, dtaper = smoothness_taper(BO_ij, T(0.9), one(T))
                            trip_engy_raw = trip_tm3 * trip_tm2
                            trip_tm3 *= taper
                        end

                        trip_energies[i] += trip_tm3 * trip_tm2
                        
                        dE_trip_dBOij = trip_tm3 * (-p_trip2 * T(2.0) * (BO_ij - T(2.5)) * trip_tm2 +
                                                    p_trip4 * exp_sum * denom_inv)

                        if smoothness_correction
                            dE_trip_dBOij += trip_engy_raw * dtaper
                        end
                        
                        pref_1 = trip_tm3 * inv(trip_denom) * inv(trip_denom)
                        uv′ = exp_sum * T(25.0) * trip_exp4 * p_trip3
                        dE_trip_dΔi = pref_1 * (trip_exp2 * -p_trip4 * trip_denom - uv′)
                        dE_trip_dΔj = pref_1 * (trip_exp3 * -p_trip4 * trip_denom - uv′)

                        S_dE_dBO[idx_j, i] += dE_trip_dBOij
                        S_dE_dBO[idx_i, j] += dE_trip_dBOij

                        Atomix.@atomic S_dE_dΔ[i] += dE_trip_dΔi
                        Atomix.@atomic S_dE_dΔ[j] += dE_trip_dΔj
                    end
                end
            end
        end
    end
end


function corrections!(
    inter::ReaxFF{ReaxFF2008, T},
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T
    atom_map = inter.params.atom_map
    C_id = get(atom_map, "C", -1)

    if C_id == -1
        return
    end

    nbrs = inter.state.nbrs
    fill_pos = inter.state.fill_pos
    atid = inter.state.sys_atid
    BO = inter.state.BO
    Δ = inter.state.Δ
    genp = inter.device_params.general_parameters
    n_atoms = inter.state.n_atoms
    coordn = inter.state.coordn

    k_c2 = genp.k_c2 
    c2_correction = k_c2 > 0.001

    O_id = get(atom_map, "O", -1)

    triple_bond_correction = O_id > 0

    S_dE_dBO = inter.state.S_dE_dBO
    S_dE_dΔ = inter.state.S_dE_dΔ
    reverse_nbrs = inter.state.reverse_nbrs

    c2_energies = inter.state.thread_atom_energies_1
    trip_energies = inter.state.thread_atom_energies_2
    fill!(c2_energies, zero(T))
    fill!(trip_energies, zero(T))

    kernel = corrections_kernel(backend)
    kernel(
        atid,
        genp,
        C_id,
        O_id,
        fill_pos,
        nbrs,
        reverse_nbrs,
        BO,
        c2_correction,
        Δ,
        S_dE_dBO,
        S_dE_dΔ,
        triple_bond_correction,
        coordn,
        c2_energies,
        trip_energies,
        inter.smoothness_correction;
        ndrange=n_atoms,
        workgroupsize=backend_workgroupsize(backend, n_atoms, n_threads)
    )

    E_c2 = sum(c2_energies)
    E_trip = sum(trip_energies)
    inter.state.energy += E_c2
    inter.state.energy += E_trip
    scalar_setindex!(inter.state.energy_components,
                    scalar_getindex(inter.state.energy_components, 1) + E_trip, 1)
    scalar_setindex!(inter.state.energy_components,
                    scalar_getindex(inter.state.energy_components, 3) + E_c2, 3)
end


# identifies all valid hydrogen bonds and which satisfy bond order constraints
# writes j, k, idx_i into valid_hbonds for each valid i-j --- k hydrogen bond
@kernel function construct_valid_hbonds!(
    valid_hbonds,
    n_valid_hbonds,
    far_nbrs_list,
    h_id,
    fill_pos,
    nbrs,
    BO,
    hbond_bo_cutoff,
    conv_factor::T,
    coords,
    boundary,
    hbdp,
    atid,
    hbond_dist_cutoff,
    h_bond_donors,
    h_bond_acceptors
) where T
    ti = @index(Global, Linear)

    cutoff_sq = hbond_dist_cutoff * hbond_dist_cutoff

    @inbounds begin
        atom_a, atom_b, _ = far_nbrs_list[ti]

        # i-j --- k  <==>  donor-H --- acceptor
        if atid[atom_a] == h_id && h_bond_acceptors[atid[atom_b]]
            j = atom_a
            k = atom_b
        elseif atid[atom_b] == h_id && h_bond_acceptors[atid[atom_a]]
            j = atom_b
            k = atom_a
        else
            j = zero(atom_a)
            k = zero(atom_b)
        end

        if !iszero(j)
            r_vec_jk = map(x -> ustrip(x) * conv_factor, vector(coords[k], coords[j], boundary))

            r_jk_sq = r_vec_jk[1]^2 + r_vec_jk[2]^2 + r_vec_jk[3]^2

            if r_jk_sq < cutoff_sq
                atid_j = atid[j]
                atid_k = atid[k]

                # find donors by checking atoms bonded to H
                for idx_i in 1:fill_pos[j]
                    i = nbrs[idx_i, j]

                    # donor and acceptor must not be same atom
                    i != k || continue

                    atid_i = atid[i]
                    h_bond_donors[atid_i] || continue

                    BO_ij = BO[idx_i, j]
                    BO_ij >= hbond_bo_cutoff || continue

                    hbdp_ijk = hbdp[atid_i, atid_j, atid_k]
                    hbdp_ijk.is_defined || continue

                    valid_hbond_idx = Atomix.@atomic n_valid_hbonds[1] += 1

                    if valid_hbond_idx <= size(valid_hbonds, 2)
                        valid_hbonds[1, valid_hbond_idx] = j
                        valid_hbonds[2, valid_hbond_idx] = k
                        valid_hbonds[3, valid_hbond_idx] = idx_i
                    end
                end
            end
        end
    end
end


# one thread per interaction
@kernel function hbond_kernel(
    valid_hbonds,
    nbrs,
    BO,
    reverse_nbrs,
    conv_factor::T,
    coords,
    boundary,
    hbdp,
    atid,
    S_dE_dBO,
    forces,
    hb_energies,
    hbond_bo_cutoff,
    hbond_dist_cutoff,
    ::Val{smoothness_correction},
) where {T, smoothness_correction}
    ti = @index(Global, Linear)

    @inbounds begin
        j     = valid_hbonds[1, ti]
        k     = valid_hbonds[2, ti]
        idx_i = valid_hbonds[3, ti]

        i = nbrs[idx_i, j]

        c_j = coords[j]
        atid_i = atid[i]
        atid_j = atid[j]
        atid_k = atid[k]

        BO_ij = BO[idx_i, j]
        hbdp_ijk = hbdp[atid_i, atid_j, atid_k]

        r_vec_jk = map(x -> ustrip(x) * conv_factor, vector(coords[k], c_j, boundary))

        r_jk_sq = r_vec_jk[1]^2 + r_vec_jk[2]^2 + r_vec_jk[3]^2
        r_jk = sqrt(r_jk_sq)
        inv_r_jk = inv(r_jk)
        r_vec_jk_unit = r_vec_jk / r_jk

        energy_pair = zero(T)
        f_j_x = zero(T)
        f_j_y = zero(T)
        f_j_z = zero(T)

        idx_j = reverse_nbrs[idx_i, j]

        r_vec_ij = map(x -> ustrip(x) * conv_factor, vector(c_j, coords[i], boundary))

        r_vec_kj = -r_vec_jk
        ij_dot_kj = dot(r_vec_ij, r_vec_kj)
        norm_ij_sq = dot(r_vec_ij, r_vec_ij)
        inv_norm_ij = inv(sqrt(norm_ij_sq))
        inv_norm_mul = inv_norm_ij * inv_r_jk

        cos_Θ_ijk = clamp(ij_dot_kj * inv_norm_mul, -one(T), one(T))
        sin_half_θ_sq = T(0.5) * (one(T) - cos_Θ_ijk)           # sin^2(θ/2)  

        p_hb1 = hbdp_ijk.p_hb1
        p_hb2 = hbdp_ijk.p_hb2
        p_hb3 = hbdp_ijk.p_hb3
        r_hb = hbdp_ijk.r_hb

        hbond_exp1 = exp(-p_hb2 * BO_ij)

        inv_r_hb = inv(r_hb)

        hbond_exp2 = exp(
            -p_hb3 * (
                r_hb * inv_r_jk +
                r_jk * inv_r_hb -
                T(2.0)
            )
        )

        if smoothness_correction
            taper_ij, dtaper_ij = smoothness_taper(BO_ij, hbond_bo_cutoff, 4 * hbond_bo_cutoff)
            taper_jk, dtaper_jk = smoothness_taper(r_jk, T(0.9) * hbond_dist_cutoff, hbond_dist_cutoff)
            taper_jk = one(T) - taper_jk
            dtaper_jk = -dtaper_jk
            taper = taper_ij * taper_jk
        end

        hbond_prefactor = p_hb1 * hbond_exp2
        hbond_tm2 = hbond_prefactor * sin_half_θ_sq * sin_half_θ_sq

        hbond_engy = hbond_tm2 * (one(T) - hbond_exp1)

        if smoothness_correction
            hbond_engy_raw = hbond_engy
            hbond_prefactor *= taper
            hbond_tm2 *= taper
            hbond_engy *= taper
        end
        
        energy_pair += hbond_engy

        angular_prefactor = -hbond_prefactor *  (one(T) - hbond_exp1) * sin_half_θ_sq * inv_norm_mul
        dtheta_qt1 = ij_dot_kj / norm_ij_sq
        dtheta_qt2 = ij_dot_kj * inv_r_jk * inv_r_jk

        dE_dΘ_i = angular_prefactor * (r_vec_kj - dtheta_qt1 * r_vec_ij)
        dE_dΘ_j = -angular_prefactor * (r_vec_ij + r_vec_kj - dtheta_qt1 * r_vec_ij - dtheta_qt2 * r_vec_kj)
        dE_dΘ_k = angular_prefactor * (r_vec_ij - dtheta_qt2 * r_vec_kj)

        dE_hbond_dBOij = hbond_tm2 * hbond_exp1 * p_hb2

        dE_hbond_drjk = hbond_engy * -p_hb3 * (-r_hb * inv_r_jk * inv_r_jk + inv_r_hb)

        if smoothness_correction
            dE_hbond_dBOij += hbond_engy_raw * dtaper_ij * taper_jk
            dE_hbond_drjk += hbond_engy_raw * taper_ij * dtaper_jk
        end

        dE_hbond_drjk_vec = dE_hbond_drjk * r_vec_jk_unit

        Atomix.@atomic forces[1, k] += dE_dΘ_k[1] - dE_hbond_drjk_vec[1]
        Atomix.@atomic forces[2, k] += dE_dΘ_k[2] - dE_hbond_drjk_vec[2]
        Atomix.@atomic forces[3, k] += dE_dΘ_k[3] - dE_hbond_drjk_vec[3]

        Atomix.@atomic forces[1, i] += dE_dΘ_i[1]
        Atomix.@atomic forces[2, i] += dE_dΘ_i[2]
        Atomix.@atomic forces[3, i] += dE_dΘ_i[3]

        f_j_x += dE_dΘ_j[1] + dE_hbond_drjk_vec[1]
        f_j_y += dE_dΘ_j[2] + dE_hbond_drjk_vec[2]
        f_j_z += dE_dΘ_j[3] + dE_hbond_drjk_vec[3]

        Atomix.@atomic S_dE_dBO[idx_i, j] += dE_hbond_dBOij
        Atomix.@atomic S_dE_dBO[idx_j, i] += dE_hbond_dBOij

        Atomix.@atomic forces[1, j] += f_j_x
        Atomix.@atomic forces[2, j] += f_j_y
        Atomix.@atomic forces[3, j] += f_j_z

        Atomix.@atomic hb_energies[j] += energy_pair
    end
end


function hydrogen_bonds!(
    inter::ReaxFF{ReaxFF2008, T},
    sys::System,
    far_nbrs::Union{NeighborList, GPUCellListNeighborList},
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T
    atom_map = inter.params.atom_map
    h_id = get(atom_map, "H", -1)

    if h_id == -1
        return
    end

    fill_pos = inter.state.fill_pos
    nbrs = inter.state.nbrs
    BO = inter.state.BO
    reverse_nbrs = inter.state.reverse_nbrs
    atid = inter.state.sys_atid

    hbdp = inter.device_params.h_bond_parameters
    h_bond_donors = inter.device_params.h_bond_donors
    h_bond_acceptors = inter.device_params.h_bond_acceptors
    hbond_bo_cutoff = inter.device_params.global_parameters.hbond_bo_cutoff
    hbond_dist_cutoff = inter.device_params.global_parameters.hbond_dist_cutoff

    conv_factor = inter.use_units ? T(ustrip(uconvert(u"Å", 1.0 * inter.sys_coord_unit))) : T(10.0)

    hb_energies = inter.state.thread_atom_energies_1
    fill!(hb_energies, zero(T))

    fill!(inter.state.n_valid_hbonds, 0)

    # this continues counting even when capacity of valid hbonds is reached => reports exact required size
    valid_hbonds_kernel! = construct_valid_hbonds!(backend)
    if length(far_nbrs) > 0
        valid_hbonds_kernel!(
            inter.state.valid_hbonds,
            inter.state.n_valid_hbonds,
            far_nbrs.list,
            h_id,
            fill_pos,
            nbrs,
            BO,
            hbond_bo_cutoff,
            conv_factor,
            sys.coords,
            sys.boundary,
            hbdp,
            atid,
            hbond_dist_cutoff,
            h_bond_donors,
            h_bond_acceptors;
            ndrange=length(far_nbrs),
            workgroupsize=backend_workgroupsize(backend, length(far_nbrs), n_threads)
        )
        synchronize(backend)
    end

    n_valid_hbonds = scalar_getindex(inter.state.n_valid_hbonds, 1)

    # if valid hbonds matrix is too small, rebuild with larger size
    if n_valid_hbonds > size(inter.state.valid_hbonds, 2)
        new_size = max(n_valid_hbonds, 2 * size(inter.state.valid_hbonds, 2))
        inter.state.valid_hbonds = similar(inter.state.valid_hbonds, 3, new_size)
        fill!(inter.state.n_valid_hbonds, 0)
        valid_hbonds_kernel!(
            inter.state.valid_hbonds,
            inter.state.n_valid_hbonds,
            far_nbrs.list,
            h_id,
            fill_pos,
            nbrs,
            BO,
            hbond_bo_cutoff,
            conv_factor,
            sys.coords,
            sys.boundary,
            hbdp,
            atid,
            hbond_dist_cutoff,
            h_bond_donors,
            h_bond_acceptors;
            ndrange=length(far_nbrs),
            workgroupsize=backend_workgroupsize(backend, length(far_nbrs), n_threads)
        )
        synchronize(backend)
        n_valid_hbonds = scalar_getindex(inter.state.n_valid_hbonds, 1)
    end

    if n_valid_hbonds > 0
        calculate_h_bonds_kernel = hbond_kernel(backend)
        calculate_h_bonds_kernel(
            inter.state.valid_hbonds,
            nbrs,
            BO,
            reverse_nbrs,
            conv_factor,
            sys.coords,
            sys.boundary,
            hbdp,
            atid,
            inter.state.S_dE_dBO,
            inter.state.forces,
            hb_energies,
            hbond_bo_cutoff,
            hbond_dist_cutoff,
            inter.smoothness_correction,
            ndrange=n_valid_hbonds,
            workgroupsize=backend_workgroupsize(backend, n_valid_hbonds, n_threads)
        )
        synchronize(backend)
    end

    E_hb = sum(hb_energies)
    inter.state.energy += E_hb
    scalar_setindex!(inter.state.energy_components, E_hb, 7)
end


@kernel function non_bonded_kernel(
    far_nbrs_list,
    coords,
    boundary,
    conv_factor,
    tp,
    odgp,
    atid,
    p_vdW1,
    p_vdW1_inv,
    q,
    C::T,
    forces,
    vdW_energies,
    coulomb_energies,
    upper_taper_radius
) where T
    ti = @index(Global, Linear)

    @inbounds begin
        ni = far_nbrs_list[ti]
        i, j, _ = ni

        r_vec_ij = map(x -> ustrip(x) * conv_factor, vector(coords[j], coords[i], boundary))
        r_ij = norm(r_vec_ij)

        if r_ij <= upper_taper_radius
            r_ij_unit = r_vec_ij / r_ij
    
            Tap = muladd(tp.Tap_7, r_ij, tp.Tap_6)
            Tap = muladd(Tap, r_ij, tp.Tap_5)
            Tap = muladd(Tap, r_ij, tp.Tap_4)
            Tap = muladd(Tap, r_ij, tp.Tap_3)
            Tap = muladd(Tap, r_ij, tp.Tap_2)
            Tap = muladd(Tap, r_ij, tp.Tap_1)
            Tap = muladd(Tap, r_ij, tp.Tap_0)
    
            dTap = muladd(T(7.0), tp.Tap_7 * r_ij, T(6.0) * tp.Tap_6)
            dTap = muladd(dTap, r_ij, T(5.0) * tp.Tap_5)
            dTap = muladd(dTap, r_ij, T(4.0) * tp.Tap_4)
            dTap = muladd(dTap, r_ij, T(3.0) * tp.Tap_3)
            dTap = muladd(dTap, r_ij, T(2.0) * tp.Tap_2)
            dTap = muladd(dTap, r_ij, tp.Tap_1)
    
            odgp_ij = odgp[atid[i], atid[j]]
    
            # van der Walls interactions
            D_ij = odgp_ij.D_ij
            alpha = odgp_ij.alpha
            r_vdW = T(2.0) * odgp_ij.r_vdW
    
            gamma_w_inv_pow = odgp_ij.gamma_w_inv_pow
    
            r_ij_pow = r_ij^p_vdW1
            f13_sum1 = r_ij_pow + gamma_w_inv_pow
            f13 = f13_sum1^p_vdW1_inv
    
            df13_drij = f13 * r_ij_pow * inv(f13_sum1 * r_ij)
    
            r_vdW_inv = inv(r_vdW)
            vdW_sum1 = one(T) - f13 * r_vdW_inv
            
            vdW_exp2 = exp(T(0.5) * alpha * vdW_sum1)
            vdW_exp1 = vdW_exp2 * vdW_exp2
    
            vdW_tm2 = D_ij * (vdW_exp1 - T(2.0) * vdW_exp2)
            vdW_engy = Tap * vdW_tm2
    
            Atomix.@atomic vdW_energies[i] += vdW_engy
    
            vdW_prd1 = r_vdW_inv * df13_drij
    
            dE_vdW_drij = dTap * vdW_tm2 + 
                            Tap * D_ij * alpha * vdW_prd1 * (vdW_exp2 - vdW_exp1)
    
            # Coulomb energy
            gamma_ij_inv_pow = odgp_ij.gamma_ij_inv_pow
    
            q_prod = q[i] * q[j]
            denom_sum = r_ij^3 + gamma_ij_inv_pow
            coulomb_inv = inv(cbrt(denom_sum))
    
            coulomb_engy = Tap * C * q_prod * coulomb_inv
            Atomix.@atomic coulomb_energies[i] += coulomb_engy
    
            dE_coulomb_drij = C * q_prod * coulomb_inv * (
                                dTap -
                                Tap * inv(denom_sum) * r_ij^2
                            )
            
            tot_dE_drij = (dE_vdW_drij + dE_coulomb_drij) * r_ij_unit
    
            for n in 1:3
                Atomix.@atomic forces[n, i] += tot_dE_drij[n]
                Atomix.@atomic forces[n, j] -= tot_dE_drij[n]
            end
        end
    end
end


function update_reaxff_charges!(
    inter::ReaxFF{ReaxFF2008, T},
    sys
) where T
    inter.state.charges_sys_precision .= T.(charge.(sys.atoms))
end


function non_bonded!(
    inter::ReaxFF{ReaxFF2008, T},
    sys::System,
    far_nbrs::Union{NeighborList, GPUCellListNeighborList},
    backend;
    n_threads::Integer=Threads.nthreads(),
) where T

    tp = inter.device_params.taper_parameters
    genp = inter.device_params.general_parameters
    odgp = inter.device_params.off_diagonal_parameters
    atid = inter.state.sys_atid
    upper_taper_radius = inter.device_params.general_parameters.upper_taper_radius

    update_reaxff_charges!(inter, sys)
    q = inter.state.charges_sys_precision

    p_vdW1 = genp.p_vdW1
    p_vdW1_inv = inv(p_vdW1)

    C = T(332.06371)    # legacy constant

    conv_factor = inter.use_units ? T(ustrip(uconvert(u"Å", 1.0 * inter.sys_coord_unit))) : T(10.0)

    vdW_energies = inter.state.thread_atom_energies_1
    coulomb_energies = inter.state.thread_atom_energies_2
    fill!(vdW_energies, zero(T))
    fill!(coulomb_energies, zero(T))

    synchronize(backend)
    nb_kernel = non_bonded_kernel(backend)
    nb_kernel(
        far_nbrs.list,
        sys.coords,
        sys.boundary,
        conv_factor,
        tp,
        odgp,
        atid,
        p_vdW1,
        p_vdW1_inv,
        q,
        C,
        inter.state.forces,
        vdW_energies,
        coulomb_energies,
        upper_taper_radius;
        ndrange=length(far_nbrs),
        workgroupsize=backend_workgroupsize(backend, length(far_nbrs), n_threads)
    )
    synchronize(backend)

    E_vdW = sum(vdW_energies)
    E_coulomb = sum(coulomb_energies)
    inter.state.energy += E_vdW + E_coulomb

    scalar_setindex!(inter.state.energy_components, E_vdW, 10)
    scalar_setindex!(inter.state.energy_components, E_coulomb, 11)
end


# ReaxFF concrete subtype of AbstractEEMCoupling
struct ReaxFFTaperedCoulomb{T, TP, OD, QT} <: AbstractEEMCoupling
    cutoff::T
    taper_parameters::TP
    off_diagonal_parameters::OD
    coulomb_const::QT
end

# allows coupling::ReaxFFTaperedCoulomb to be passed into KernelAbstractions @kernel
Adapt.@adapt_structure ReaxFFTaperedCoulomb

# convenience constructor for ReaxFFTaperedCoulomb
# only requires reaxff object (even before System creation)
function ReaxFFTaperedCoulomb(
    reaxff::ReaxFF{ReaxFF2008, T},
    eem_float_type::Type{QT}
) where {T, QT}
    return ReaxFFTaperedCoulomb(
        reaxff.device_params.general_parameters.upper_taper_radius,
        reaxff.device_params.taper_parameters,
        reaxff.device_params.off_diagonal_parameters,
        QT(14.4)    # Coulomb constant in eV compatible units, legacy ReaxFF
    )
end

# units of r passed into eem_hij should be Angstroms
function eem_length_unit(::ReaxFFTaperedCoulomb)
    return u"Å"
end

function eem_energy_conversion(energy_eV, sys, coupling::ReaxFFTaperedCoulomb{T}) where T
    energy_kcal_mol = energy_eV * (T(332.06371) / coupling.coulomb_const) * u"kcal/mol"
    target_units = sys.energy_units == NoUnits ? u"kJ/mol" : sys.energy_units
    energy_sys = uconvert(target_units, energy_kcal_mol)
    return float_type(sys)(sys.energy_units == NoUnits ? ustrip(energy_sys) : energy_sys)
end

# calculate off-diagonal elements of H_ij matrix using ReaxFF's tapered + shielded Coulomb potential
# r should be in Angstroms
@inline function eem_hij(
    coupling::ReaxFFTaperedCoulomb{T, TP, OD, QT},
    r,
    type_i,
    type_j,
) where {T, TP, OD, QT}
    if r > coupling.cutoff
        return zero(QT)
    end

    tp = coupling.taper_parameters

    Tap = muladd(tp.Tap_7, r, tp.Tap_6)
    Tap = muladd(Tap, r, tp.Tap_5)
    Tap = muladd(Tap, r, tp.Tap_4)
    Tap = muladd(Tap, r, tp.Tap_3)
    Tap = muladd(Tap, r, tp.Tap_2)
    Tap = muladd(Tap, r, tp.Tap_1)
    Tap = muladd(Tap, r, tp.Tap_0)

    odgp_ij = coupling.off_diagonal_parameters[type_i, type_j]
    gamma_ij_inv_pow = odgp_ij.gamma_ij_inv_pow
    denom_sum = r^3 + gamma_ij_inv_pow
    coulomb_inv = inv(cbrt(denom_sum))

    h = Tap * coupling.coulomb_const * coulomb_inv

    return h
end

"""
    EEM(reaxff::ReaxFF, atoms_data; charges=nothing, tolerance=1e-6,
        eem_float_type=Float64, max_iterations=200)

Construct [`EEM`](@ref) charge equilibration using a ReaxFF force field's
electronegativities, hardnesses, and tapered, shielded Coulomb interaction.

`atoms_data` contains an entry for each atom with an `element` field matching the
force field. Arrays are placed on the same device as `reaxff`, with solver precision
set by `eem_float_type`. `charges` gives the initial numerical charges in
elementary-charge units and defaults to zeros; their sum sets the conserved total
charge. `tolerance` and `max_iterations` control the EEM solver.

`System(coord_file_path, reaxff; ...)` calls this constructor automatically.
"""
function EEM(
    reaxff::ReaxFF{ReaxFF2008, T, AT},
    atoms_data;
    charges=nothing,
    tolerance=1e-6,
    eem_float_type::Type{QT}=Float64,
    max_iterations=200
) where {T, AT, QT<:AbstractFloat}
    if !isnothing(charges)
        if length(charges) != length(atoms_data)
            throw(ArgumentError("Length of charges array must equal length of atoms_data array"))
        end
    end

    coupling = ReaxFFTaperedCoulomb(reaxff, QT)

    type_ids = [reaxff.params.atom_map[ad.element] for ad in atoms_data]    # type id for every atom in system

    atmp_cpu = from_device(reaxff.device_params.atom_parameters)

    # store EEM parameters per atom type
    χ = QT[p.chi_EEM for p in atmp_cpu]
    η = QT[2.0 * p.eta_EEM for p in atmp_cpu]   # force field value is half the physical value

    q = isnothing(charges) ? nothing : to_device(QT.(charges), AT)

    return EEM(
        coupling,
        to_device(type_ids, AT),
        to_device(χ, AT),
        to_device(η, AT);
        charges=q,
        tolerance=QT(tolerance),
        max_iterations=max_iterations
    )
end


function print_energy_components(
    inter::ReaxFF{ReaxFF2008}
)
    ec = from_device(inter.state.energy_components)

    println("-----------------------------------------")
    println("###             Energy                ###")
    println("-----------------------------------------")
    println("E_bond:            $(ec[1])")
    println("E_atom:            $(ec[2])")
    println("E_lone_pair:       $(ec[3])")
    println("E_valence_angle:   $(ec[4])")
    println("E_pen:             $(ec[5])")
    println("E_coa:             $(ec[6])")
    println("E_hbond:           $(ec[7])")
    println("E_torsion:         $(ec[8])")
    println("E_conjugation:     $(ec[9])")
    println("E_vdW:             $(ec[10])")
    println("E_coulomb:         $(ec[11])")
    println("-----------------------------------------")
    println("Total:             $(sum(ec))")
    println("-----------------------------------------")
end


function compute_reaxff_energy_forces!(
    inter::ReaxFF{ReaxFF2008, T},
    sys::System,
    neighbors::Union{NeighborList, GPUCellListNeighborList},
    backend;
    n_threads::Integer=Threads.nthreads(),
) where {T}
    
    _clear_reaxff_state!(inter)
    bonds!(inter, sys, neighbors, backend; n_threads=n_threads)
    lone_pairs!(inter, backend; n_threads=n_threads)
    over_under_coordination!(inter, backend; n_threads=n_threads)
    valence_angles!(inter, sys, backend; n_threads=n_threads)
    torsions!(inter, sys, backend; n_threads=n_threads)
    corrections!(inter, backend; n_threads=n_threads)
    hydrogen_bonds!(inter, sys, neighbors, backend; n_threads=n_threads)
    non_bonded!(inter, sys, neighbors, backend; n_threads=n_threads)
    resolve_forces!(inter, sys, backend; n_threads=n_threads)

    return inter.state.energy
end


get_float_type(::ReaxFF{V, T}) where {V<:AbstractReaxFFVersion, T} = T


@kernel function force_convert_tally!(
    fs, 
    inter_state_forces, 
    unit_conv
)
    ti = @index(Global, Linear)

    @inbounds begin
        fx = inter_state_forces[1, ti] * unit_conv
        fy = inter_state_forces[2, ti] * unit_conv
        fz = inter_state_forces[3, ti] * unit_conv
                        
        fs[ti] += SVector{3}(fx, fy, fz)
    end
end


AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(sys, inter::ReaxFF; neighbors, n_threads::Integer=Threads.nthreads(), kwargs...)
    if isnothing(inter.state) || length(sys.atoms) != inter.state.n_atoms
        inter.use_units = length_type(sys.boundary) <: Unitful.Quantity
        inter.sys_coord_unit = unit(eltype(eltype(sys.coords)))
        _init_reaxff_state!(inter, sys)
    end

    backend = get_backend(sys.coords)

    energy = compute_reaxff_energy_forces!(inter, sys, neighbors, backend; n_threads=n_threads)

    if inter.use_units
        return energy*u"kcal * mol^-1"
    else
        # assume unitless path uses kJ/mol
        return ustrip(uconvert(u"kJ/mol", energy * u"kcal/mol"))
    end
end


function refresh_reaxff_shielding(pair, atom_i, atom_j, p_vdW1)
    gamma_w = sqrt(atom_i.gamma_w * atom_j.gamma_w)
    gamma_prod = atom_i.gamma * atom_j.gamma

    return OffDiagonalParameters(
        pair.D_ij, pair.r_vdW, pair.alpha,
        pair.ro_sigma, pair.ro_pi, pair.ro_pipi,
        inv(gamma_w)^p_vdW1, inv(gamma_prod * sqrt(gamma_prod)), pair.is_defined,
    )
end


#=
    Molly.enzyme_forces_and_energy(sys, inter::ReaxFF, eem::EEM; neighbors)

Return `(forces, energy)` for the ReaxFF and EEM interactions, for use in loss
functions differentiated with Enzyme.jl.

`forces` is a numerical `3 × N` matrix, with one atom per column, in kJ/mol/nm.
`energy` is a numerical scalar in kJ/mol, including the EEM self-energy.
Both outputs have units stripped, regardless of the system's units setting.
Other interactions in `sys` are not included.

Supply the ReaxFF and EEM instances belonging to `sys` and a neighbor list valid
for its current coordinates. The function updates the shielding and EEM
coefficients from the current parameters, equilibrates charges, and evaluates
energy and forces. It mutates `sys.atoms` and both interactions' parameters and
cached state. Cutoff and taper caches must be updated separately if those
parameters are changed.

Call once to initialize the interaction state before constructing Enzyme shadow
objects.
=#
function enzyme_forces_and_energy(sys, inter::ReaxFF, eem::EEM; neighbors)
    atom_parameters = inter.device_params.atom_parameters
    off_diagonal_parameters = inter.device_params.off_diagonal_parameters
    p_vdW1 = inter.device_params.general_parameters.p_vdW1

    off_diagonal_parameters .= refresh_reaxff_shielding.(
        off_diagonal_parameters,
        atom_parameters,
        reshape(atom_parameters, 1, :),
        p_vdW1,
    )

    eem.χ .= (p -> p.chi_EEM).(atom_parameters)
    eem.η .= (p -> 2 * p.eta_EEM).(atom_parameters)

    backend = get_backend(eem.state.q)
    kernel! = update_QEq_diagonal!(backend)
    kernel!(
        eem.state.H_diag, eem.state.H_diag_inv,
        eem.η, eem.χ, eem.state.mχ, eem.type_ids;
        ndrange=eem.n_atoms,
    )
    synchronize(backend)

    fill!(eem.state.q, sum(eem.state.q) / eem.n_atoms)
    eem.state.history_count = 0

    energy = AtomsCalculators.potential_energy(sys, eem; neighbors)
    energy += AtomsCalculators.potential_energy(sys, inter; neighbors)

    if inter.use_units
        energy = ustrip(uconvert(u"kJ/mol", energy))
    end

    forces = inter.state.forces .* get_float_type(inter)(-41.84)
    return forces, energy
end


AtomsCalculators.@generate_interface function AtomsCalculators.forces!(fs, sys, inter::ReaxFF; neighbors, n_threads::Integer=Threads.nthreads(), kwargs...)
    get(kwargs, :needs_vir, false) && error("ReaxFF virial calculation is not implemented")
    if isnothing(inter.state) || length(sys.atoms) != inter.state.n_atoms
        inter.use_units = length_type(sys.boundary) <: Unitful.Quantity
        inter.sys_coord_unit = unit(eltype(eltype(sys.coords)))

        _init_reaxff_state!(inter, sys)
    end

    backend = get_backend(sys.coords)

    compute_reaxff_energy_forces!(inter, sys, neighbors, backend; n_threads=n_threads)

    T = get_float_type(inter)

    if inter.use_units
        fs_unit = unit(eltype(eltype(fs)))
        unit_conv = T(uconvert(fs_unit, T(-1.0)u"kcal * mol^-1 * Å^-1"))
    else
        # assume unit-less forces are in kJ nm^-1 mol^-1
        unit_conv = T(ustrip(uconvert(u"kJ * mol^-1 * nm^-1", T(-1.0)u"kcal * mol^-1 * Å^-1")))
    end

    force_tally_kernel = force_convert_tally!(backend)
    force_tally_kernel(
        fs,
        inter.state.forces,
        unit_conv;
        ndrange=inter.state.n_atoms,
        workgroupsize=backend_workgroupsize(backend, inter.state.n_atoms, n_threads)
    )

    return fs
end


### Extra energy and force functions for tests below ###

# returns component level energies in kcal/mol
function compute_energy_components(
    sys,
    inter::ReaxFF,
)
    inter.use_units = length_type(sys.boundary) <: Unitful.Quantity
    inter.sys_coord_unit = unit(eltype(eltype(sys.coords)))

    _init_reaxff_state!(inter, sys)

    neighbors = find_neighbors(
        sys,
        sys.neighbor_finder,
        nothing,
        0,
        true
    )

    backend = get_backend(sys.coords)

    compute_reaxff_energy_forces!(inter, sys, neighbors, backend)

    ec = from_device(inter.state.energy_components)

    E_bond = ec[1]
    E_atom = ec[2]
    E_lone_pair = ec[3]
    E_valence_angle = ec[4]
    E_pen = ec[5]
    E_coa = ec[6]
    E_hbond = ec[7]
    E_torsion = ec[8]
    E_conjugation = ec[9]
    E_vdW = ec[10]
    E_Qeq_coulomb = ec[11]

    E_total = sum(ec)

    return (
        E_bond,
        E_atom,
        E_lone_pair,
        E_valence_angle,
        E_pen,
        E_coa,
        E_hbond,
        E_torsion,
        E_conjugation,
        E_vdW,
        E_Qeq_coulomb,
        E_total
    )
end

# Returns force in kcal/mol/nm for tests
function compute_force(sys, inter::ReaxFF)
    inter.use_units = length_type(sys.boundary) <: Unitful.Quantity
    inter.sys_coord_unit = unit(eltype(eltype(sys.coords)))

    _init_reaxff_state!(inter, sys)

    neighbors = find_neighbors(
        sys,
        sys.neighbor_finder,
        nothing,
        0,
        true
    )

    backend = get_backend(sys.coords)

    compute_reaxff_energy_forces!(inter, sys, neighbors, backend)

    f = [-10*convert(Vector,i) for i in eachcol(from_device(inter.state.forces))]
    return f
end
