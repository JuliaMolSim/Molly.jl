# Run molecular dynamics
# See https://www.saylor.org/site/wp-content/uploads/2011/06/MA221-6.1.pdf for
#   integration algorithm - used shorter second version
# See https://udel.edu/~arthij/MD.pdf for information on forces
# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See Gromacs manual for other aspects of forces

export
    simulate!

"The constant for Coulomb interaction, 1/(4*π*ϵ0*ϵr)."
const coulomb_const = 138.935458 / 70.0 # Treat ϵr as 70 for now

"Square of the neighbour list cutoff in nm."
const sqdist_cutoff = 1.5 ^ 2

"3D acceleration values, e.g. for an atom, in nm/(ps^2)."
mutable struct Acceleration
    x::Float64
    y::Float64
    z::Float64
end

"Update coordinates of all atoms and bound to the bounding box."
function update_coordinates!(coords::Vector{Coordinates},
                    velocities::Vector{Velocity},
                    accels::Vector{Acceleration},
                    timestep::Real,
                    box_size::Real)
    for (i, c) in enumerate(coords)
        c.x += velocities[i].x*timestep + 0.5*accels[i].x*timestep^2
        while (c.x >= box_size) c.x -= box_size end
        while (c.x < 0.0) c.x += box_size end

        c.y += velocities[i].y*timestep + 0.5*accels[i].y*timestep^2
        while (c.y >= box_size) c.y -= box_size end
        while (c.y < 0.0) c.y += box_size end

        c.z += velocities[i].z*timestep + 0.5*accels[i].z*timestep^2
        while (c.z >= box_size) c.z -= box_size end
        while (c.z < 0.0) c.z += box_size end
    end
    return coords
end

"Update velocities of all atoms using the accelerations."
function update_velocities!(velocities::Vector{Velocity},
                    accels_t::Vector{Acceleration},
                    accels_t_dt::Vector{Acceleration},
                    timestep::Real)
    for (i, v) in enumerate(velocities)
        v.x += 0.5*(accels_t[i].x+accels_t_dt[i].x)*timestep
        v.y += 0.5*(accels_t[i].y+accels_t_dt[i].y)*timestep
        v.z += 0.5*(accels_t[i].z+accels_t_dt[i].z)*timestep
    end
    return velocities
end

"Vector between two coordinate values, accounting for the bounding box."
function vector1D(c1::Real, c2::Real, box_size::Real)
    if c1 < c2
        return (c2 - c1) < (c1 - c2 + box_size) ? (c2 - c1) : (c2 - c1 - box_size)
    else
        return (c1 - c2) < (c2 - c1 + box_size) ? (c2 - c1) : (c2 - c1 + box_size)
    end
end

"3D vector between two `Coordinates`, accounting for the bounding box."
vector(coords_one::Coordinates, coords_two::Coordinates, box_size::Real) = [
        vector1D(coords_one.x, coords_two.x, box_size),
        vector1D(coords_one.y, coords_two.y, box_size),
        vector1D(coords_one.z, coords_two.z, box_size)]

"Square distance between two `Coordinates`, accounting for the bounding box."
sqdist(coords_one::Coordinates, coords_two::Coordinates, box_size::Real) =
        vector1D(coords_one.x, coords_two.x, box_size) ^ 2 +
        vector1D(coords_one.y, coords_two.y, box_size) ^ 2 +
        vector1D(coords_one.z, coords_two.z, box_size) ^ 2

"Force on each atom in a covalent bond."
function forcebond(coords_one::Coordinates,
                coords_two::Coordinates,
                bondtype::Bondtype,
                box_size::Real)
    ab = vector(coords_one, coords_two, box_size)
    c = bondtype.kb * (norm(ab) - bondtype.b0)
    fa = c*normalize(ab)
    fb = -fa
    return fa..., fb...
end

# Sometimes domain error occurs for acos if the float is > 1.0 or < 1.0
acosbound(x::Real) = acos(max(min(x, 1.0), -1.0))

"Force on each atom in a bond angle."
function forceangle(coords_one::Coordinates,
                coords_two::Coordinates,
                coords_three::Coordinates,
                angletype::Angletype,
                box_size::Real)
    ba = vector(coords_two, coords_one, box_size)
    bc = vector(coords_two, coords_three, box_size)
    pa = normalize(ba × (ba × bc))
    pc = normalize(-bc × (ba × bc))
    angle_term = -angletype.cth * (acosbound(dot(ba, bc) / (norm(ba) * norm(bc))) - angletype.th0)
    fa = (angle_term / norm(ba)) * pa
    fc = (angle_term / norm(bc)) * pc
    fb = -fa - fc
    return fa..., fb..., fc...
end

"Force on each atom in a dihedral torsion angle."
function forcedihedral(coords_one::Coordinates,
                coords_two::Coordinates,
                coords_three::Coordinates,
                coords_four::Coordinates,
                dihedraltype::Dihedraltype,
                box_size::Real)
    ba = vector(coords_two, coords_one, box_size)
    bc = vector(coords_two, coords_three, box_size)
    dc = vector(coords_four, coords_three, box_size)
    p1 = normalize(ba × bc)
    p2 = normalize(-dc × -bc)
    θ = atan2(dot((-ba × bc) × (bc × -dc), normalize(bc)), dot(-ba × bc, bc × -dc))
    angle_term = 0.5*(dihedraltype.f1*sin(θ) - 2*dihedraltype.f2*sin(2*θ) + 3*dihedraltype.f3*sin(3*θ))
    fa = (angle_term / (norm(ba) * sin(acosbound(dot(ba, bc) / (norm(ba) * norm(bc)))))) * p1
    fd = (angle_term / (norm(dc) * sin(acosbound(dot(bc, dc) / (norm(bc) * norm(dc)))))) * p2
    oc = 0.5 * bc
    tc = -(oc × fd + 0.5 * (-dc × fd) + 0.5 * (ba × fa))
    fc = (1 / dot(oc, oc)) * (tc × oc)
    fb = -fa - fc -fd
    return fa..., fb..., fc..., fd...
end

"Force on each atom due to Lennard Jones attractive/repulsive potential."
function forcelennardjones end

@fastmath @inbounds function forcelennardjones(coords_one::Coordinates,
                coords_two::Coordinates,
                atom_one::Atom,
                atom_two::Atom,
                box_size::Real)
    σ = sqrt(atom_one.σ * atom_two.σ)
    ϵ = sqrt(atom_one.ϵ * atom_two.ϵ)
    dx = vector1D(coords_one.x, coords_two.x, box_size)
    dy = vector1D(coords_one.y, coords_two.y, box_size)
    dz = vector1D(coords_one.z, coords_two.z, box_size)
    r2 = dx*dx + dy*dy + dz*dz
    six_term = (σ^2 / r2)^3
    # Limit this to 100 as a fudge to stop it exploding
    f = min(((24 * ϵ) / (r2)) * (2 * six_term^2 - six_term), 100.0)
    return -f*dx, -f*dy, -f*dz, f*dx, f*dy, f*dz
end

"Force on each atom due to electrostatic Coulomb potential."
function forcecoulomb end

@fastmath @inbounds function forcecoulomb(coords_one::Coordinates,
                coords_two::Coordinates,
                charge_one::Real,
                charge_two::Real,
                box_size::Real)
    dx = vector1D(coords_one.x, coords_two.x, box_size)
    dy = vector1D(coords_one.y, coords_two.y, box_size)
    dz = vector1D(coords_one.z, coords_two.z, box_size)
    f = (coulomb_const * charge_one * charge_two) / sqrt((dx*dx + dy*dy + dz*dz)^3)
    return -f*dx, -f*dy, -f*dz, f*dx, f*dy, f*dz
end

"Update accelerations of all atoms using the bonded and non-bonded forces."
function update_accelerations!(accels::Vector{Acceleration},
                universe::Universe,
                forcefield::Forcefield)
    # Clear accelerations
    for i in 1:length(accels)
        accels[i].x = 0.0
        accels[i].y = 0.0
        accels[i].z = 0.0
    end

    # Bonded forces
    # Covalent bond forces
    #temp_atomid = 59
    #temp_bond_sum = 0.0
    atoms = universe.molecule.atoms
    for b in universe.molecule.bonds
        if haskey(forcefield.bondtypes, "$(atoms[b.atom_i].attype)/$(atoms[b.atom_j].attype)")
            bondtype = forcefield.bondtypes["$(atoms[b.atom_i].attype)/$(atoms[b.atom_j].attype)"]
        else
            bondtype = forcefield.bondtypes["$(atoms[b.atom_j].attype)/$(atoms[b.atom_i].attype)"]
        end
        d1x, d1y, d1z, d2x, d2y, d2z = forcebond(
            universe.coords[b.atom_i], universe.coords[b.atom_j],
            bondtype, universe.box_size)
        accels[b.atom_i].x += d1x
        accels[b.atom_i].y += d1y
        accels[b.atom_i].z += d1z
        accels[b.atom_j].x += d2x
        accels[b.atom_j].y += d2y
        accels[b.atom_j].z += d2z
        #b.atom_i == temp_atomid ? temp_bond_sum += d1x : nothing
        #b.atom_j == temp_atomid ? temp_bond_sum += d2x : nothing
    end
    #print("Bond: $(round(temp_bond_sum, 2)), ")

    # Angle forces
    #temp_angle_sum = 0.0
    for a in universe.molecule.angles
        if haskey(forcefield.angletypes, "$(atoms[a.atom_i].attype)/$(atoms[a.atom_j].attype)/$(atoms[a.atom_k].attype)")
            angletype = forcefield.angletypes["$(atoms[a.atom_i].attype)/$(atoms[a.atom_j].attype)/$(atoms[a.atom_k].attype)"]
        else
            angletype = forcefield.angletypes["$(atoms[a.atom_k].attype)/$(atoms[a.atom_j].attype)/$(atoms[a.atom_i].attype)"]
        end
        d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z = forceangle(
            universe.coords[a.atom_i], universe.coords[a.atom_j],
            universe.coords[a.atom_k], angletype, universe.box_size)
        accels[a.atom_i].x += d1x
        accels[a.atom_i].y += d1y
        accels[a.atom_i].z += d1z
        accels[a.atom_j].x += d2x
        accels[a.atom_j].y += d2y
        accels[a.atom_j].z += d2z
        accels[a.atom_k].x += d3x
        accels[a.atom_k].y += d3y
        accels[a.atom_k].z += d3z
        #a.atom_i == temp_atomid ? temp_angle_sum += d1x : nothing
        #a.atom_j == temp_atomid ? temp_angle_sum += d2x : nothing
        #a.atom_k == temp_atomid ? temp_angle_sum += d3x : nothing
    end
    #print("Angle: $(round(temp_angle_sum, 2)), ")

    # Dihedral forces
    #temp_tor_sum = 0.0
    for d in universe.molecule.dihedrals
        dihedraltype = forcefield.dihedraltypes["$(atoms[d.atom_i].attype)/$(atoms[d.atom_j].attype)/$(atoms[d.atom_k].attype)/$(atoms[d.atom_l].attype)"]
        d1x, d1y, d1z, d2x, d2y, d2z, d3x, d3y, d3z, d4x, d4y, d4z = forcedihedral(
            universe.coords[d.atom_i], universe.coords[d.atom_j],
            universe.coords[d.atom_k], universe.coords[d.atom_l],
            dihedraltype, universe.box_size)
        accels[d.atom_i].x += d1x
        accels[d.atom_i].y += d1y
        accels[d.atom_i].z += d1z
        accels[d.atom_j].x += d2x
        accels[d.atom_j].y += d2y
        accels[d.atom_j].z += d2z
        accels[d.atom_k].x += d3x
        accels[d.atom_k].y += d3y
        accels[d.atom_k].z += d3z
        accels[d.atom_l].x += d4x
        accels[d.atom_l].y += d4y
        accels[d.atom_l].z += d4z
        #d.atom_i == temp_atomid ? temp_tor_sum += d1x : nothing
        #d.atom_j == temp_atomid ? temp_tor_sum += d2x : nothing
        #d.atom_k == temp_atomid ? temp_tor_sum += d3x : nothing
        #d.atom_l == temp_atomid ? temp_tor_sum += d4x : nothing
    end
    #print("Torsion: $(round(temp_tor_sum, 2)), ")

    # Non-bonded forces
    #temp_lj_sum = 0.0
    #temp_el_sum = 0.0
    for (i, j, d) in universe.neighbour_list
        a1 = universe.molecule.atoms[i]
        a2 = universe.molecule.atoms[j]

        # Lennard Jones forces
        d1x, d1y, d1z, d2x, d2y, d2z = forcelennardjones(
            universe.coords[i], universe.coords[j], a1, a2, universe.box_size)
        accels[i].x += d ? 0.5*d1x : d1x
        accels[i].y += d ? 0.5*d1y : d1y
        accels[i].z += d ? 0.5*d1z : d1z
        accels[j].x += d ? 0.5*d2x : d2x
        accels[j].y += d ? 0.5*d2y : d2y
        accels[j].z += d ? 0.5*d2z : d2z
        #i == temp_atomid ? temp_lj_sum += d1x : nothing
        #j == temp_atomid ? temp_lj_sum += d2x : nothing

        # Electrostatic forces
        d1x, d1y, d1z, d2x, d2y, d2z = forcecoulomb(
            universe.coords[i], universe.coords[j], a1.charge, a2.charge, universe.box_size)
        accels[i].x += d ? 0.5*d1x : d1x
        accels[i].y += d ? 0.5*d1y : d1y
        accels[i].z += d ? 0.5*d1z : d1z
        accels[j].x += d ? 0.5*d2x : d2x
        accels[j].y += d ? 0.5*d2y : d2y
        accels[j].z += d ? 0.5*d2z : d2z
        #i == temp_atomid ? temp_el_sum += d1x : nothing
        #j == temp_atomid ? temp_el_sum += d2x : nothing
    end
    #print("LJ: $(round(temp_lj_sum, 2)), ")
    #print("Coulomb: $(round(temp_el_sum, 2)), ")

    return accels
end

"Update list of close atoms between which non-bonded forces are calculated."
function update_neighbours!(universe::Universe)
    empty!(universe.neighbour_list)
    for i in 1:length(universe.coords)
        for j in 1:(i-1)
            if universe.molecule.nb_matrix[i,j] &&
                    sqdist(universe.coords[i], universe.coords[j], universe.box_size) < sqdist_cutoff
                push!(universe.neighbour_list, (i, j, universe.molecule.nb_pairs[i, j]))
            end
        end
    end
    return universe
end

"Initialise empty `Acceleration`s."
empty_accelerations(n_atoms::Int) = [Acceleration(0.0, 0.0, 0.0) for i in 1:n_atoms]

"Simulate molecular dynamics."
function simulate!(s::Simulation, n_steps::Int)
    report("Starting simulation")
    n_atoms = length(s.universe.coords)
    update_neighbours!(s.universe)
    a_t = update_accelerations!(empty_accelerations(n_atoms), s.universe, s.forcefield)
    a_t_dt = empty_accelerations(n_atoms)
    #out_prefix = "pdbs_5XER"
    #writepdb("$out_prefix/snap_0.pdb", s.universe)
    @showprogress for i in 1:n_steps
        update_coordinates!(s.universe.coords, s.universe.velocities, a_t,
                            s.timestep, s.universe.box_size)
        update_accelerations!(a_t_dt, s.universe, s.forcefield)
        update_velocities!(s.universe.velocities, a_t, a_t_dt, s.timestep)
        # Update neighbour list every 10 steps
        if i % 10 == 0
            update_neighbours!(s.universe)
            #writepdb("$out_prefix/snap_$(Int(i/10)).pdb", s.universe)
        end
        a_t = a_t_dt
        s.steps_made += 1
    end
    report("Simulation finished")
    return s
end

simulate!(s::Simulation) = simulate!(s, s.n_steps-s.steps_made)
