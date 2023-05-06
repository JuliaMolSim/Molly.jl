
function System(crystal::Crystal{D};
        pairwise_inters=(),
        specific_inter_lists=(),
        general_inters=(),
        constraints=(),
        velocities=nothing,
        neighbor_finder=NoNeighborFinder(),
        loggers=(),
        force_units=u"kJ * mol^-1 * nm^-1",
        energy_units=u"kJ * mol^-1",
        k=Unitful.k) where D


    atoms = [Atom(index=i, charge=a.charge, mass=a.mass) for (i,a) in enumerate(crystal.atoms)]

    #not really sure what the member variables here should be mapped to
    atoms_data = [AtomData(, atom_name = SimpleCrystals.element_name(a),) for (i,a) in enumerate(crystal.atoms)]

    coords = SimpleCrystals.position(atoms, :)

    side_lengths = norm.(eachrow(bounding_box(crystal)))
    if any(typeof(crystal.lattice.crystal_family) .<: [CubicLattice, OrthorhombicLattice, TetragonalLattice])
        boundary = CubicBoundary(side_lengths...)
    elseif any(typeof(crystal.lattice.crystal_family) .<: [SquareLattice, RectangularLattice])
        boundary = RectangularBoundary(side_lengths...)
    else if D == 2 #Honeycomb, Hex2D, & Oblique
        error("$(crystal.lattice.crystal_family) is not supported as it would need a 2D triclinic boundary")
    else
        boundary = TriclinicBoundary(side_lengths, crystal.lattice_angles)
    end

    #Call original constructor
    return System(
        atoms = atoms,
        atoms_data = atoms_data,
        pairwise_inters = pairwise_inters,
        specific_inter_lists = specific_inter_lists,
        general_inters = general_inters,
        constraints=constraints,
        coords = coords,
        velocities = velocities,
        boundary = boundary,
        neighbor_finder = neighbor_finder,
        loggers = loggers,
        force_units = force_units,
        energy_units = energy_units,
        k = k,
        gpu_diff_safe = gpu_diff_safe=isa(coords, CuArray),
    )

end