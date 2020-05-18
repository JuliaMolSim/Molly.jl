var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Molly.jl-1",
    "page": "Home",
    "title": "Molly.jl",
    "category": "section",
    "text": "(Image: Travis build status) (Image: AppVeyor build status) (Image: Coverage Status) (Image: Documentation)Much of science can be explained by the movement and interaction of molecules. Molecular dynamics (MD) is a computational technique used to explore these phenomena, particularly for biological macromolecules. Molly.jl is a pure Julia implementation of MD.At the minute the package is a proof of concept for MD in Julia. It is not production ready. It can simulate a system of atoms with arbitrary interactions as defined by the user. It can also read in pre-computed Gromacs topology and coordinate files with the OPLS-AA forcefield and run MD on proteins with given parameters. In theory it can do this for any regular protein, but in practice this is untested. Implemented features include:Interface to allow definition of new forces, thermostats etc.\nNon-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential.\nBonded interactions - covalent bonds, bond angles, dihedral angles.\nAndersen thermostat.\nVelocity Verlet integration.\nExplicit solvent.\nPeriodic boundary conditions in a cubic box.\nNeighbour list to speed up calculation of non-bonded forces.\nAutomatic multithreading.Features not yet implemented include:Protein force fields other than OPLS-AA.\nWater models.\nEnergy minimisation.\nOther temperature or pressure coupling methods.\nProtein preparation - solvent box, add hydrogens etc.\nTrajectory/topology file format readers/writers.\nTrajectory analysis.\nGPU compatibility.\nUnit tests."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Julia v1.0 or later is required. Install from the Julia REPL. Enter the package mode by pressing ] and run add https://github.com/jgreener64/Molly.jl."
},

{
    "location": "index.html#Usage-1",
    "page": "Home",
    "title": "Usage",
    "category": "section",
    "text": "Some examples are given here, see the documentation for more.Simulation of an ideal gas:using Molly\n\nn_atoms = 100\nbox_size = 2.0 # nm\ntemperature = 298 # K\nmass = 10.0 # Relative atomic mass\n\natoms = [Atom(mass=mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms]\ncoords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]\nvelocities = [velocity(mass, temperature) for i in 1:n_atoms]\ngeneral_inters = Dict(\"LJ\" => LennardJones())\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=velocities,\n    temperature=temperature,\n    box_size=box_size,\n    thermostat=AndersenThermostat(1.0),\n    loggers=[TemperatureLogger(100)],\n    timestep=0.002, # ps\n    n_steps=100_000\n)\n\nsimulate!(s)Simulation of a protein:using Molly\n\ntimestep = 0.0002 # ps\ntemperature = 298 # K\nn_steps = 5000\n\natoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(\n            joinpath(dirname(pathof(Molly)), \"..\", \"data\", \"5XER\", \"gmx_top_ff.top\"),\n            joinpath(dirname(pathof(Molly)), \"..\", \"data\", \"5XER\", \"gmx_coords.gro\"))\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    specific_inter_lists=specific_inter_lists,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=[velocity(a.mass, temperature) for a in atoms],\n    temperature=temperature,\n    box_size=box_size,\n    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),\n    thermostat=AndersenThermostat(1.0),\n    loggers=[TemperatureLogger(10), StructureWriter(10, \"traj_5XER_1ps.pdb\")],\n    timestep=timestep,\n    n_steps=n_steps\n)\n\nsimulate!(s)The above 1 ps simulation looks something like this when you output view it in VMD: (Image: MD simulation)"
},

{
    "location": "index.html#Plans-1",
    "page": "Home",
    "title": "Plans",
    "category": "section",
    "text": "I plan to work on this in my spare time, but progress will be slow. MD could provide a nice use case for Julia - I think a reasonably featured and performant MD program could be written in fewer than 1,000 lines of code for example. The development of auto-differentiation packages in Julia opens up interesting avenues for differentiable molecular simulations. Julia is also a well-suited language for trajectory analysis.Contributions are very welcome - see the roadmap issue for more."
},

{
    "location": "docs.html#",
    "page": "Documentation",
    "title": "Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "docs.html#Molly-documentation-1",
    "page": "Documentation",
    "title": "Molly documentation",
    "category": "section",
    "text": "These docs are work in progressMolly takes a modular approach to molecular simulation. To run a simulation you create a Simulation object and run simulate! on it. The different components of the simulation can be used as defined by the package, or you can define your own versions."
},

{
    "location": "docs.html#Simulating-an-ideal-gas-1",
    "page": "Documentation",
    "title": "Simulating an ideal gas",
    "category": "section",
    "text": "Let\'s look at the simulation of an ideal gas to start with. First, we\'ll need some atoms with the relevant parameters defined.using Molly\n\nn_atoms = 100\nmass = 10.0\natoms = [Atom(mass=mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms]Next, we\'ll need some starting coordinates and velocities.box_size = 2.0 # nm\ncoords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]\n\ntemperature = 298 # K\nvelocities = [velocity(mass, temperature) for i in 1:n_atoms]We store the coordinates and velocities as static arrays for performance. They can be of any number of dimensions. Now we can define our dictionary of general interactions, i.e. those between most or all atoms. Because we have defined the relevant parameters for the atoms, we can use the built-in Lennard Jones type.general_inters = Dict(\"LJ\" => LennardJones())Finally, we can define and run the simulation. We use an Andersen thermostat to keep a constant temperature, and we log the temperature and coordinates every 100 steps.s = Simulation(\n    simulator=VelocityVerlet(), # Use velocity Verlet integration\n    atoms=atoms,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=velocities,\n    temperature=temperature,\n    box_size=box_size,\n    thermostat=AndersenThermostat(1.0), # Coupling constant of 1.0\n    loggers=[TemperatureLogger(100), CoordinateLogger(100)],\n    timestep=0.002, # ps\n    n_steps=100_000\n)\n\nsimulate!(s)By default the simulation is run in parallel on the number of threads available to Julia, but this can be turned off by giving the keyword argument parallel=false to simulate!. We can get a quick look at the simulation by plotting the coordinate and temperature loggers (in the future ideally this will be one easy plot command using recipes).using Plots\n\ncoords = s.loggers[2].coords\ntemps = s.loggers[1].temperatures\n\nsplitcoords(coord) = [c[1] for c in coord], [c[2] for c in coord], [c[3] for c in coord]\n\n@gif for (i, coord) in enumerate(coords)\n    l = @layout [a b{0.7h}]\n\n    cx, cy, cz = splitcoords(coord)\n    p = scatter(cx, cy, cz,\n        xlims=(0, box_size),\n        ylims=(0, box_size),\n        zlims=(0, box_size),\n        layout=l,\n        legend=false\n    )\n\n    plot!(p[2],\n        temps[1:i],\n        xlabel=\"Frame\",\n        ylabel=\"Temperature / K\",\n        xlims=(1, i),\n        ylims=(0.0, maximum(temps[1:i])),\n        legend=false\n    )\nend(Image: LJ simulation)"
},

{
    "location": "docs.html#Simulating-diatomic-molecules-1",
    "page": "Documentation",
    "title": "Simulating diatomic molecules",
    "category": "section",
    "text": "If we want to define specific interactions between atoms, for example bonds, we can do. Using the same atom definitions as before, let\'s set up the coordinates so that paired atoms are 1 Å apart.coords = [box_size .* rand(SVector{3}) for i in 1:(n_atoms / 2)]\nfor i in 1:length(coords)\n    push!(coords, coords[i] .+ [0.1, 0.0, 0.0])\nend\n\nvelocities = [velocity(mass, temperature) for i in 1:n_atoms]Now we can use the built-in bond type to place a harmonic constraint between paired atoms. The arguments are the indices of the two atoms in the bond, the equilibrium distance and the force constant.bonds = [Bond((i * 2) - 1, i * 2, 0.1, 300_000) for i in 1:(n_atoms / 2)]\n\nspecific_inter_lists = Dict(\"Bonds\" => bonds)This time, we are also going to use a neighbour list to speed up the Lennard Jones calculation. We can use the built-in distance neighbour finder. The arguments are a 2D array of eligible interactions, the number of steps between each update and the cutoff in nm to be classed as a neighbour.neighbour_finder = DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 1.2)Now we can simulate as before.s = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    specific_inter_lists=specific_inter_lists,\n    general_inters=Dict(\"LJ\" => LennardJones(true)), # true means we are using the neighbour list for this interaction\n    coords=coords,\n    velocities=velocities,\n    temperature=temperature,\n    box_size=box_size,\n    neighbour_finder=neighbour_finder,\n    thermostat=AndersenThermostat(1.0),\n    loggers=[TemperatureLogger(100), CoordinateLogger(100)],\n    timestep=0.002,\n    n_steps=100_000\n)\n\nsimulate!(s)This time when we view the trajectory we can add lines to show the bonds.using LinearAlgebra\n\ncoords = s.loggers[2].coords\ntemps = s.loggers[1].temperatures\n\nconnections = [((i * 2) - 1, i * 2) for i in 1:Int(n_atoms / 2)]\n\n@gif for (i, coord) in enumerate(coords)\n    l = @layout [a b{0.7h}]\n\n    cx, cy, cz = splitcoords(coord)\n    p = scatter(cx, cy, cz,\n        xlims=(0, box_size),\n        ylims=(0, box_size),\n        zlims=(0, box_size),\n        layout=l,\n        legend=false\n    )\n\n    for (a1, a2) in connections\n        if norm(coord[a1] - coord[a2]) < (box_size / 2)\n            plot!(p[1],\n                [cx[a1], cx[a2]],\n                [cy[a1], cy[a2]],\n                [cz[a1], cz[a2]],\n                linecolor=\"lightblue\"\n            )\n        end\n    end\n\n    plot!(p[2],\n        temps[1:i],\n        xlabel=\"Frame\",\n        ylabel=\"Temperature / K\",\n        xlims=(1, i),\n        ylims=(0.0, maximum(temps[1:i])),\n        legend=false\n    )\nend(Image: Diatomic simulation)"
},

{
    "location": "docs.html#Simulating-a-protein-in-the-OPLS-AA-forcefield-1",
    "page": "Documentation",
    "title": "Simulating a protein in the OPLS-AA forcefield",
    "category": "section",
    "text": "In progress"
},

{
    "location": "docs.html#Defining-your-own-forces-1",
    "page": "Documentation",
    "title": "Defining your own forces",
    "category": "section",
    "text": "In progress"
},

{
    "location": "api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api.html#Molly.AndersenThermostat",
    "page": "API",
    "title": "Molly.AndersenThermostat",
    "category": "type",
    "text": "Rescale random velocities according to the Andersen thermostat.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Angle",
    "page": "API",
    "title": "Molly.Angle",
    "category": "type",
    "text": "A bond angle between three atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Angletype",
    "page": "API",
    "title": "Molly.Angletype",
    "category": "type",
    "text": "Gromacs angle type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Atom",
    "page": "API",
    "title": "Molly.Atom",
    "category": "type",
    "text": "An atom and its associated information.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Atomtype",
    "page": "API",
    "title": "Molly.Atomtype",
    "category": "type",
    "text": "Gromacs atom type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Bond",
    "page": "API",
    "title": "Molly.Bond",
    "category": "type",
    "text": "A bond between two atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Bondtype",
    "page": "API",
    "title": "Molly.Bondtype",
    "category": "type",
    "text": "Gromacs bond type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.CoordinateLogger",
    "page": "API",
    "title": "Molly.CoordinateLogger",
    "category": "type",
    "text": "Log the coordinates throughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Coulomb",
    "page": "API",
    "title": "Molly.Coulomb",
    "category": "type",
    "text": "The Coulomb electrostatic interaction.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Dihedral",
    "page": "API",
    "title": "Molly.Dihedral",
    "category": "type",
    "text": "A dihedral torsion angle between four atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Dihedraltype",
    "page": "API",
    "title": "Molly.Dihedraltype",
    "category": "type",
    "text": "Gromacs dihedral type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.DistanceNeighbourFinder",
    "page": "API",
    "title": "Molly.DistanceNeighbourFinder",
    "category": "type",
    "text": "Find close atoms by distance.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.GeneralInteraction",
    "page": "API",
    "title": "Molly.GeneralInteraction",
    "category": "type",
    "text": "A general interaction that will apply to all atom pairs.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Interaction",
    "page": "API",
    "title": "Molly.Interaction",
    "category": "type",
    "text": "An interaction between atoms that contributes to forces on the atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.LennardJones",
    "page": "API",
    "title": "Molly.LennardJones",
    "category": "type",
    "text": "The Lennard-Jones 6-12 interaction.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Logger",
    "page": "API",
    "title": "Molly.Logger",
    "category": "type",
    "text": "A way to record a property, e.g. the temperature, throughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.NeighbourFinder",
    "page": "API",
    "title": "Molly.NeighbourFinder",
    "category": "type",
    "text": "A way to find near atoms to save on simulation time.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.NoNeighbourFinder",
    "page": "API",
    "title": "Molly.NoNeighbourFinder",
    "category": "type",
    "text": "Placeholder neighbour finder that does nothing.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.NoThermostat",
    "page": "API",
    "title": "Molly.NoThermostat",
    "category": "type",
    "text": "Placeholder thermostat that does nothing.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Simulation",
    "page": "API",
    "title": "Molly.Simulation",
    "category": "type",
    "text": "The data associated with a molecular simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Simulator",
    "page": "API",
    "title": "Molly.Simulator",
    "category": "type",
    "text": "A type of simulation to run, e.g. leap-frog integration or energy minimisation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.SpecificInteraction",
    "page": "API",
    "title": "Molly.SpecificInteraction",
    "category": "type",
    "text": "A specific interaction between sets of specific atoms, e.g. a bond angle.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.StructureWriter",
    "page": "API",
    "title": "Molly.StructureWriter",
    "category": "type",
    "text": "Write 3D output structures to the PDB file format throughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.TemperatureLogger",
    "page": "API",
    "title": "Molly.TemperatureLogger",
    "category": "type",
    "text": "Log the temperature throughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Thermostat",
    "page": "API",
    "title": "Molly.Thermostat",
    "category": "type",
    "text": "A way to keep the temperature of a simulation constant.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.VelocityVerlet",
    "page": "API",
    "title": "Molly.VelocityVerlet",
    "category": "type",
    "text": "The velocity Verlet integrator.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.apply_thermostat!-Tuple{Simulation,AndersenThermostat}",
    "page": "API",
    "title": "Molly.apply_thermostat!",
    "category": "method",
    "text": "Apply a thermostat to modify a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.calc_accelerations-Tuple{Simulation}",
    "page": "API",
    "title": "Molly.calc_accelerations",
    "category": "method",
    "text": "Calculate accelerations of all atoms using the bonded and non-bonded forces.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.find_neighbours!-Tuple{Simulation,DistanceNeighbourFinder,Integer}",
    "page": "API",
    "title": "Molly.find_neighbours!",
    "category": "method",
    "text": "Update list of close atoms between which non-bonded forces are calculated.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.log_property!-Tuple{TemperatureLogger,Simulation,Integer}",
    "page": "API",
    "title": "Molly.log_property!",
    "category": "method",
    "text": "Log a property thoughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.maxwellboltzmann-Tuple{Real,Real}",
    "page": "API",
    "title": "Molly.maxwellboltzmann",
    "category": "method",
    "text": "Draw from the Maxwell-Boltzmann distribution.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.readinputs-Tuple{AbstractString,AbstractString}",
    "page": "API",
    "title": "Molly.readinputs",
    "category": "method",
    "text": "Read a Gromacs topology flat file, i.e. all includes collapsed into one file.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.simulate!-Tuple{Simulation,VelocityVerlet,Integer}",
    "page": "API",
    "title": "Molly.simulate!",
    "category": "method",
    "text": "Simulate molecular dynamics.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.temperature-Tuple{Simulation}",
    "page": "API",
    "title": "Molly.temperature",
    "category": "method",
    "text": "Calculate the temperature of a system from the kinetic energy of the atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.update_accelerations!",
    "page": "API",
    "title": "Molly.update_accelerations!",
    "category": "function",
    "text": "Update the accelerations in response to a given interation type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.update_coordinates!-Union{Tuple{T}, Tuple{Simulation,Array{T,1}}} where T",
    "page": "API",
    "title": "Molly.update_coordinates!",
    "category": "method",
    "text": "Update coordinates of all atoms and bound to the bounding box.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.update_velocities!-Union{Tuple{T}, Tuple{Simulation,Array{T,1},Array{T,1}}} where T",
    "page": "API",
    "title": "Molly.update_velocities!",
    "category": "method",
    "text": "Update velocities of all atoms using the accelerations.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.vector1D-Tuple{Real,Real,Real}",
    "page": "API",
    "title": "Molly.vector1D",
    "category": "method",
    "text": "Vector between two coordinate values, accounting for the bounding box.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.velocity-Tuple{Real,Real}",
    "page": "API",
    "title": "Molly.velocity",
    "category": "method",
    "text": "Generate a random velocity from the Maxwell-Boltzmann distribution.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly-API-1",
    "page": "API",
    "title": "Molly API",
    "category": "section",
    "text": "The API reference can be found here. Molly also re-exports StaticArrays.jl, making the likes of SVector available when you call using Molly.Order   = [:module, :type, :constant, :function, :macro]Modules = [Molly]\nPrivate = false\nOrder   = [:module, :type, :constant, :function, :macro]"
},

]}
