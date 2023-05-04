export
    nothing

 function LoadSimpleCrystalSystem(crystal::Crystal{D}) where D


    atoms = [Atom(index=i, charge=a.charge, mass=a.mass) for (i,a) in enumerate(crystal.atoms)]

    #not really sure what the member variables here should be mapped to
    atoms_data = [AtomData(, atom_name = SimpleCrystals.element_name(a),) for (i,a) in enumerate(crystal.atoms)]

    coords = SimpleCrystals.position(atoms, :)

    #why did I name it lattice angle in 2D that is annoying ...
    # I also have multiple things called Cubic -- BravaisLattice and the actual crystal, can it resolve types??
    # Need triclinic 2D to support oblique and honeycomb
    side_lengths = norm.(bounding_box(crystal))
    if typeof(crystal.lattice.crystal_family) == Cubic
        boundary = CubicBoundary(side_lengths...)
    elseif typeof(crystal.lattice.crystal_family) == Square
        boundary = RectangularBoundary()
    else
        boundary = TriclinicBoundary()
    end

    # I feel like a system constructor that takes a Crystal object is the easiest way to do this
    System(
        atoms = atoms,
        atoms_data = atoms_data,
        pairwise_inters = nothing,
        specific_inter_lists = nothing,
        general_inters = nothing,
        coords = coords,
        velocities = nothing,
        boundary = boundary,
        neighbor_finder = nothing,
        loggers = nothing,
        force_units = nothing,
        energy_units = nothing,
        k = nothing,
        gpu_diff_safe = nothing,
    )
 end