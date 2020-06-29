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
    "text": "(Image: Travis build status) (Image: AppVeyor build status) (Image: Coverage Status) (Image: Documentation)Much of science can be explained by the movement and interaction of molecules. Molecular dynamics (MD) is a computational technique used to explore these phenomena, from noble gases to biological macromolecules. Molly.jl is a pure Julia package for MD, and for the simulation of physical systems more broadly.At the minute the package is a proof of concept for MD in Julia. It is not production ready. It can simulate a system of atoms with arbitrary interactions as defined by the user. Implemented features include:Interface to allow definition of new forces, simulators, thermostats, neighbour finders, loggers etc.\nRead in pre-computed Gromacs topology and coordinate files with the OPLS-AA forcefield and run MD on proteins with given parameters. In theory it can do this for any regular protein, but in practice this is untested.\nNon-bonded interactions - Lennard-Jones Van der Waals/repulsion force, electrostatic Coulomb potential, gravitational potential, soft sphere potential.\nBonded interactions - covalent bonds, bond angles, torsion angles.\nAndersen thermostat.\nVelocity Verlet and velocity-free Verlet integration.\nExplicit solvent.\nPeriodic boundary conditions in a cubic box.\nNeighbour list to speed up calculation of non-bonded forces.\nAutomatic multithreading.\nSome analysis functions, e.g. RDF.\nRun with Float64 or Float32.\nPhysical agent-based modelling.\nVisualise simulations as animations.Features not yet implemented include:Protein force fields other than OPLS-AA.\nWater models.\nEnergy minimisation.\nOther temperature or pressure coupling methods.\nCell-based neighbour list.\nProtein preparation - solvent box, add hydrogens etc.\nTrajectory/topology file format readers/writers.\nQuantum mechanical modelling.\nGPU compatibility.\nHigh test coverage."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Julia v1.0 or later is required. Install from the Julia REPL. Enter the package mode by pressing ] and runadd https://github.com/JuliaMolSim/Molly.jl"
},

{
    "location": "index.html#Usage-1",
    "page": "Home",
    "title": "Usage",
    "category": "section",
    "text": "Some examples are given here, see the documentation for more on how to use the package.Simulation of an ideal gas:using Molly\n\nn_atoms = 100\nbox_size = 2.0 # nm\ntemp = 298 # K\nmass = 10.0 # Relative atomic mass\n\natoms = [Atom(mass=mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms]\ncoords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]\nvelocities = [velocity(mass, temp) for i in 1:n_atoms]\ngeneral_inters = (LennardJones(),)\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=velocities,\n    temperature=temp,\n    box_size=box_size,\n    thermostat=AndersenThermostat(1.0),\n    loggers=Dict(\"temp\" => TemperatureLogger(100)),\n    timestep=0.002, # ps\n    n_steps=10_000\n)\n\nsimulate!(s)Simulation of a protein:using Molly\n\ntimestep = 0.0002 # ps\ntemp = 298 # K\nn_steps = 5_000\n\natoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(\n            joinpath(dirname(pathof(Molly)), \"..\", \"data\", \"5XER\", \"gmx_top_ff.top\"),\n            joinpath(dirname(pathof(Molly)), \"..\", \"data\", \"5XER\", \"gmx_coords.gro\"))\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    specific_inter_lists=specific_inter_lists,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=[velocity(a.mass, temp) for a in atoms],\n    temperature=temp,\n    box_size=box_size,\n    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),\n    thermostat=AndersenThermostat(1.0),\n    loggers=Dict(\"temp\" => TemperatureLogger(10),\n                    \"writer\" => StructureWriter(10, \"traj_5XER_1ps.pdb\")),\n    timestep=timestep,\n    n_steps=n_steps\n)\n\nsimulate!(s)The above 1 ps simulation looks something like this when you view it in VMD: (Image: MD simulation)"
},

{
    "location": "index.html#Plans-1",
    "page": "Home",
    "title": "Plans",
    "category": "section",
    "text": "I plan to work on this in my spare time, but progress will be slow. MD could provide a nice use case for Julia - I think a reasonably featured and performant MD program could be written in fewer than 1,000 lines of code for example. The development of auto-differentiation packages in Julia opens up interesting avenues for differentiable molecular simulations (there is an experimental branch on this repo). Julia is also a well-suited language for trajectory analysis.Contributions are very welcome - see the roadmap issue for more."
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
    "text": "Molly takes a modular approach to molecular simulation. To run a simulation you create a Simulation object and call simulate! on it. The different components of the simulation can be used as defined by the package, or you can define your own versions. An important principle of the package is that your custom components, particularly force functions, should be easy to define and just as performant as the built-in versions.This documentation will first introduce the main features of the package with some examples, then will give details on each component of a simulation. There are further examples in the Molly examples section. For more information on specific types or functions, see the Molly API section or call ?function_name in Julia."
},

{
    "location": "docs.html#Simulating-a-gas-1",
    "page": "Documentation",
    "title": "Simulating a gas",
    "category": "section",
    "text": "Let\'s look at the simulation of a gas acting under the Lennard-Jones potential to start with. First, we\'ll need some atoms with the relevant parameters defined.using Molly\n\nn_atoms = 100\nmass = 10.0\natoms = [Atom(mass=mass, σ=0.3, ϵ=0.2) for i in 1:n_atoms]Next, we\'ll need some starting coordinates and velocities.box_size = 2.0 # nm\ncoords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]\n\ntemp = 100 # K\nvelocities = [velocity(mass, temp) for i in 1:n_atoms]We store the coordinates and velocities as static arrays for performance. They can be of any number of dimensions and of any number type, e.g. Float64 or Float32. Now we can define our general interactions, i.e. those between most or all atoms. Because we have defined the relevant parameters for the atoms, we can use the built-in Lennard Jones type.general_inters = (LennardJones(),)Finally, we can define and run the simulation. We use an Andersen thermostat to keep a constant temperature, and we log the temperature and coordinates every 10 steps.s = Simulation(\n    simulator=VelocityVerlet(), # Use velocity Verlet integration\n    atoms=atoms,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=velocities,\n    temperature=temp,\n    box_size=box_size,\n    thermostat=AndersenThermostat(1.0), # Coupling constant of 1.0\n    loggers=Dict(\"temp\" => TemperatureLogger(10),\n                    \"coords\" => CoordinateLogger(10)),\n    timestep=0.002, # ps\n    n_steps=1_000\n)\n\nsimulate!(s)By default the simulation is run in parallel on the number of threads available to Julia, but this can be turned off by giving the keyword argument parallel=false to simulate!. An animation of the stored coordinates using can be saved using visualize, which is available when Makie.jl is imported.using Makie\n\nvisualize(s.loggers[\"coords\"], box_size, \"sim_lj.gif\")(Image: LJ simulation)"
},

{
    "location": "docs.html#Simulating-diatomic-molecules-1",
    "page": "Documentation",
    "title": "Simulating diatomic molecules",
    "category": "section",
    "text": "If we want to define specific interactions between atoms, for example bonds, we can do. Using the same atom definitions as before, let\'s set up the coordinates so that paired atoms are 1 Å apart.coords = [box_size .* rand(SVector{3}) for i in 1:(n_atoms / 2)]\nfor i in 1:length(coords)\n    push!(coords, coords[i] .+ [0.1, 0.0, 0.0])\nend\n\nvelocities = [velocity(mass, temp) for i in 1:n_atoms]Now we can use the built-in bond type to place a harmonic constraint between paired atoms. The arguments are the indices of the two atoms in the bond, the equilibrium distance and the force constant.bonds = [HarmonicBond(i, Int(i + n_atoms / 2), 0.1, 300_000.0) for i in 1:Int(n_atoms / 2)]\n\nspecific_inter_lists = (bonds,)This time, we are also going to use a neighbour list to speed up the Lennard Jones calculation. We can use the built-in distance neighbour finder. The arguments are a 2D array of eligible interactions, the number of steps between each update and the cutoff in nm to be classed as a neighbour.neighbour_finder = DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 1.2)Now we can simulate as before.s = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    specific_inter_lists=specific_inter_lists,\n    general_inters=(LennardJones(true),), # true means we are using the neighbour list for this interaction\n    coords=coords,\n    velocities=velocities,\n    temperature=temp,\n    box_size=box_size,\n    neighbour_finder=neighbour_finder,\n    thermostat=AndersenThermostat(1.0),\n    loggers=Dict(\"temp\" => TemperatureLogger(10),\n                    \"coords\" => CoordinateLogger(10)),\n    timestep=0.002,\n    n_steps=1_000\n)\n\nsimulate!(s)This time when we view the trajectory we can add lines to show the bonds.visualize(s.loggers[\"coords\"], box_size, \"sim_diatomic.gif\",\n            connections=[(i, Int(i + n_atoms / 2)) for i in 1:Int(n_atoms / 2)],\n            markersize=0.05, linewidth=5.0)(Image: Diatomic simulation)"
},

{
    "location": "docs.html#Simulating-gravity-1",
    "page": "Documentation",
    "title": "Simulating gravity",
    "category": "section",
    "text": "Molly is geared primarily to molecular simulation, but can also be used to simulate other physical systems. Let\'s set up a gravitational simulation. This example also shows the use of Float32 and a 2D simulation.atoms = [Atom(mass=1.0f0), Atom(mass=1.0f0)]\ncoords = [SVector(0.3f0, 0.5f0), SVector(0.7f0, 0.5f0)]\nvelocities = [SVector(0.0f0, 1.0f0), SVector(0.0f0, -1.0f0)]\ngeneral_inters = (Gravity(false, 1.5f0),)\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=velocities,\n    box_size=1.0f0,\n    loggers=Dict(\"coords\" => CoordinateLogger(Float32, 10, dims=2)),\n    timestep=0.002f0,\n    n_steps=2000\n)\n\nsimulate!(s)When we view the simulation we can use some extra options:visualize(s.loggers[\"coords\"], 1.0f0, \"sim_gravity.gif\",\n            trails=4, framerate=15, color=[:orange, :lightgreen],\n            markersize=0.05)(Image: Gravity simulation)"
},

{
    "location": "docs.html#Simulating-a-protein-1",
    "page": "Documentation",
    "title": "Simulating a protein",
    "category": "section",
    "text": "Molly has a rudimentary parser of Gromacs topology and coordinate files. Data for a protein can be read into the same data structures as above and simulated in the same way. Currently, the OPLS-AA forcefield is implemented.atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(\n            joinpath(dirname(pathof(Molly)), \"..\", \"data\", \"5XER\", \"gmx_top_ff.top\"),\n            joinpath(dirname(pathof(Molly)), \"..\", \"data\", \"5XER\", \"gmx_coords.gro\"))\n\ntemp = 298\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    specific_inter_lists=specific_inter_lists,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=[velocity(a.mass, temp) for a in atoms],\n    temperature=temp,\n    box_size=box_size,\n    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),\n    thermostat=AndersenThermostat(1.0),\n    loggers=Dict(\"temp\" => TemperatureLogger(10),\n                    \"writer\" => StructureWriter(10, \"traj_5XER_1ps.pdb\")),\n    timestep=0.0002,\n    n_steps=5_000\n)\n\nsimulate!(s)"
},

{
    "location": "docs.html#Agent-based-modelling-1",
    "page": "Documentation",
    "title": "Agent-based modelling",
    "category": "section",
    "text": "Agent-based modelling (ABM) is conceptually similar to molecular dynamics. Julia has Agents.jl for ABM, but Molly can also be used to simulate arbitrary agent-based systems in continuous space. Here we simulate a toy SIR model for disease spread. This example shows how atom properties can be mutable, i.e. change during the simulation, and includes custom forces and loggers (see below for more).@enum Status susceptible infected recovered\n\n# Custom atom type\nmutable struct Person\n    status::Status\n    mass::Float64\n    σ::Float64\n    ϵ::Float64\nend\n\n# Custom GeneralInteraction\nstruct SIRInteraction <: GeneralInteraction\n    nl_only::Bool\n    dist_infection::Float64\n    prob_infection::Float64\n    prob_recovery::Float64\nend\n\n# Custom force function\nfunction Molly.force!(forces, inter::SIRInteraction,\n                        s::Simulation,\n                        i::Integer,\n                        j::Integer)\n    if i == j\n        # Recover randomly\n        if s.atoms[i].status == infected && rand() < inter.prob_recovery\n            s.atoms[i].status = recovered\n        end\n    elseif (s.atoms[i].status == infected && s.atoms[j].status == susceptible) ||\n                (s.atoms[i].status == susceptible && s.atoms[j].status == infected)\n        # Infect close people randomly\n        dr = vector(s.coords[i], s.coords[j], s.box_size)\n        r2 = sum(abs2, dr)\n        if r2 < inter.dist_infection ^ 2 && rand() < inter.prob_infection\n            s.atoms[i].status = infected\n            s.atoms[j].status = infected\n        end\n    end\n    return nothing\nend\n\n# Custom Logger\nstruct SIRLogger <: Logger\n    n_steps::Int\n    fracs_sir::Vector{Vector{Float64}}\nend\n\n# Custom logging function\nfunction Molly.log_property!(logger::SIRLogger, s::Simulation, step_n::Integer)\n    if step_n % logger.n_steps == 0\n        counts_sir = [\n            count(p -> p.status == susceptible, s.atoms),\n            count(p -> p.status == infected   , s.atoms),\n            count(p -> p.status == recovered  , s.atoms)\n        ]\n        push!(logger.fracs_sir, counts_sir ./ length(s.atoms))\n    end\nend\n\ntemp = 0.01\ntimestep = 0.02\nbox_size = 10.0\nn_steps = 1_000\nn_people = 500\nn_starting = 2\natoms = [Person(i <= n_starting ? infected : susceptible, 1.0, 0.1, 0.02) for i in 1:n_people]\ncoords = [box_size .* rand(SVector{2}) for i in 1:n_people]\nvelocities = [velocity(1.0, temp, dims=2) for i in 1:n_people]\ngeneral_inters = (LennardJones = LennardJones(true), SIR = SIRInteraction(false, 0.5, 0.06, 0.01))\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=velocities,\n    temperature=temp,\n    box_size=box_size,\n    neighbour_finder=DistanceNeighbourFinder(trues(n_people, n_people), 10, 2.0),\n    thermostat=AndersenThermostat(5.0),\n    loggers=Dict(\"coords\" => CoordinateLogger(10, dims=2),\n                    \"SIR\" => SIRLogger(10, [])),\n    timestep=timestep,\n    n_steps=n_steps\n)\n\nsimulate!(s)\n\nvisualize(s.loggers[\"coords\"], box_size, \"sim_agent.gif\")(Image: Agent simulation)We can use the logger to plot the fraction of people susceptible (blue), infected (orange) and recovered (green) over the course of the simulation: (Image: Fraction SIR)"
},

{
    "location": "docs.html#Forces-1",
    "page": "Documentation",
    "title": "Forces",
    "category": "section",
    "text": "Forces define how different parts of the system interact. In Molly they are separated into two types. GeneralInteractions are present between all or most atoms, and account for example for non-bonded terms. SpecificInteractions are present between specific atoms, and account for example for bonded terms.The available general interactions are:LennardJones.\nSoftSphere.\nCoulomb.\nGravity.The available specific interactions are:HarmonicBond.\nHarmonicAngle.\nTorsion.To define your own GeneralInteraction, first define the struct:struct MyGeneralInter <: GeneralInteraction\n    nl_only::Bool\n    # Any other properties, e.g. constants for the interaction\nendThe nl_only property is required and determines whether the neighbour list is used to omit distant atoms (true) or whether all atom pairs are always considered (false). Next, you need to define the force! function acting between a pair of atoms. For example:function Molly.force!(forces,\n                        inter::MyGeneralInter,\n                        s::Simulation,\n                        i::Integer,\n                        j::Integer)\n    dr = vector(s.coords[i], s.coords[j], s.box_size)\n\n    # Replace this with your force calculation\n    # A positive force causes the atoms to move apart\n    f = 0.0\n\n    fdr = f * normalize(dr)\n    forces[i] -= fdr\n    forces[j] += fdr\n    return nothing\nendIf you need to obtain the vector from atom i to atom j, use the vector function. This gets the vector between the closest images of atoms i and j accounting for the periodic boundary conditions. The Simulation is available so atom properties or velocities can be accessed, e.g. s.atoms[i].σ or s.velocities[i]. This form of the function can also be used to define three-atom interactions by looping a third variable k up to j in the force! function. Typically the force function is where most computation time is spent during the simulation, so consider optimising this function if you want high performance.To use your custom force, add it to the list of general interactions:general_inters = (MyGeneralInter(true),)Then create and run a Simulation as above. Note that you can also use named tuples instead of tuples if you want to access interactions by name:general_inters = (MyGeneralInter = MyGeneralInter(true),)For performance reasons it is best to avoid containers with abstract type parameters, such as Vector{GeneralInteraction}.To define your own SpecificInteraction, first define the struct:struct MySpecificInter <: SpecificInteraction\n    # Any number of atoms involved in the interaction\n    i::Int\n    j::Int\n    # Any other properties, e.g. a bond distance corresponding to the energy minimum\nendNext, you need to define the force! function. For example:function Molly.force!(forces, inter::MySpecificInter, s::Simulation)\n    dr = vector(s.coords[inter.i], s.coords[inter.j], s.box_size)\n\n    # Replace this with your force calculation\n    # A positive force causes the atoms to move apart\n    f = 0.0\n\n    fdr = f * normalize(dr)\n    forces[inter.i] += fdr\n    forces[inter.j] -= fdr\n    return nothing\nendThe example here is between two atoms but can be adapted for any number of atoms. To use your custom force, add it to the specific interaction lists:specific_inter_lists = ([MySpecificInter(1, 2), MySpecificInter(3, 4)],)"
},

{
    "location": "docs.html#Simulators-1",
    "page": "Documentation",
    "title": "Simulators",
    "category": "section",
    "text": "Simulators define what type of simulation is run. This could be anything from a simple energy minimisation to complicated replica exchange MD. The available simulators are:VelocityVerlet.\nVelocityFreeVerlet.To define your own Simulator, first define the struct:struct MySimulator <: Simulator\n    # Any properties, e.g. an implicit solvent friction constant\nendThen, define the function that carries out the simulation. This example shows some of the helper functions you can use:function Molly.simulate!(s::Simulation,\n                            simulator::MySimulator,\n                            n_steps::Integer;\n                            parallel::Bool=true)\n    # Find neighbours like this\n    neighbours = find_neighbours(s, nothing, s.neighbour_finder, 0,\n                                    parallel=parallel)\n\n    # Show a progress bar like this, if you have imported ProgressMeter\n    @showprogress for step_n in 1:n_steps\n        # Apply the loggers like this\n        for logger in values(s.loggers)\n            log_property!(logger, s, step_n)\n        end\n\n        # Calculate accelerations like this\n        accels_t = accelerations(s, neighbours, parallel=parallel)\n\n        # Ensure coordinates stay within the simulation box like this\n        for i in 1:length(s.coords)\n            s.coords[i] = adjust_bounds.(s.coords[i], s.box_size)\n        end\n\n        # Apply the thermostat like this\n        apply_thermostat!(s, s.thermostat)\n\n        # Find new neighbours like this\n        neighbours = find_neighbours(s, neighbours, s.neighbour_finder, step_n,\n                                        parallel=parallel)\n\n        # Increment the step counter like this\n        s.n_steps_made[1] += 1\n    end\n    return s\nend"
},

{
    "location": "docs.html#Thermostats-1",
    "page": "Documentation",
    "title": "Thermostats",
    "category": "section",
    "text": "Thermostats control the temperature over a simulation. The available thermostats are:AndersenThermostat.To define your own Thermostats, first define the struct:struct MyThermostat <: Thermostat\n    # Any properties, e.g. a coupling constant\nendThen, define the function that implements the thermostat every timestep:function apply_thermostat!(s::Simulation, thermostat::MyThermostat)\n    # Do something to the simulation, e.g. scale the velocities\n    return s\nendThe functions velocity, maxwellboltzmann and temperature may be useful here. To use your custom thermostat, give it as the thermostat argument when creating the Simulation."
},

{
    "location": "docs.html#Neighbour-finders-1",
    "page": "Documentation",
    "title": "Neighbour finders",
    "category": "section",
    "text": "Neighbour finders find close atoms periodically throughout the simulation, saving on computation time by allowing the force calculation between distant atoms to be omitted. The available neighbour finders are:DistanceNeighbourFinder.To define your own NeighbourFinder, first define the struct:struct MyNeighbourFinder <: NeighbourFinder\n    nb_matrix::BitArray{2}\n    n_steps::Int\n    # Any other properties, e.g. a distance cutoff\nendAn example of useful properties is used here: a matrix indicating atom pairs eligible for non-bonded interactions, and a value determining how many timesteps occur between each evaluation of the neighbour finder. Then, define the neighbour finding function that is called every step by the simulator:function find_neighbours(s::Simulation,\n                            current_neighbours,\n                            nf::MyNeighbourFinder,\n                            step_n::Integer;\n                            parallel::Bool=true)\n    if step_n % nf.n_steps == 0\n        neighbours = Tuple{Int, Int}[]\n        # Add to neighbours\n        return neighbours\n    else\n        return current_neighbours\n    end\nend"
},

{
    "location": "docs.html#Loggers-1",
    "page": "Documentation",
    "title": "Loggers",
    "category": "section",
    "text": "Loggers record properties of the simulation to allow monitoring and analysis. The available loggers are:TemperatureLogger.\nCoordinateLogger.\nStructureWriter.To define your own Logger, first define the struct:struct MyLogger <: Logger\n    n_steps::Int\n    # Any other properties, e.g. an Array to record values during the trajectory\nendThen, define the logging function that is called every step by the simulator:function Molly.log_property!(logger::MyLogger, s::Simulation, step_n::Integer)\n    if step_n % logger.n_steps == 0\n        # Record some property or carry out some action\n    end\nendThe use of n_steps is optional and is an example of how to record a property every n steps through the simulation. To use your custom logger, add it to the dictionary of loggers:loggers = Dict(\"mylogger\" => MyLogger(10))"
},

{
    "location": "docs.html#Analysis-1",
    "page": "Documentation",
    "title": "Analysis",
    "category": "section",
    "text": "Molly contains some tools for analysing the results of simulations. The available analysis functions are:visualize.\nrdf.\ndistances.\ndisplacements.Julia is a language well-suited to implementing all kinds of analysis for molecular simulations."
},

{
    "location": "examples.html#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": ""
},

{
    "location": "examples.html#Molly-examples-1",
    "page": "Examples",
    "title": "Molly examples",
    "category": "section",
    "text": "The best examples for learning how the package works are in the Molly documentation section. Here we give further examples, showing what you can do with the package. Each is a self-contained block of code. Made something cool yourself? Make a PR to add it to this page."
},

{
    "location": "examples.html#Making-and-breaking-bonds-1",
    "page": "Examples",
    "title": "Making and breaking bonds",
    "category": "section",
    "text": "There is an example of mutable atom properties in the main docs, but what if you want to make and break bonds during the simulation? In this case you can use a GeneralInteraction to make, break and apply the bonds. The partners of the atom can be stored in the atom type. We make a Logger to record when the bonds are present, allowing us to visualize them with the connection_frames keyword argument to visualize (this can take a while to plot).using Molly\nusing Makie\nusing LinearAlgebra\n\nstruct BondableAtom\n    mass::Float64\n    σ::Float64\n    ϵ::Float64\n    partners::Set{Int}\nend\n\nstruct BondableInteraction <: GeneralInteraction\n    nl_only::Bool\n    prob_formation::Float64\n    prob_break::Float64\n    dist_formation::Float64\n    b0::Float64\n    kb::Float64\nend\n\nfunction Molly.force!(forces, inter::BondableInteraction, s, i, j)\n    i == j && return\n    dr = vector(s.coords[i], s.coords[j], s.box_size)\n    r2 = sum(abs2, dr)\n    if j in s.atoms[i].partners\n        # Apply the force of a harmonic bond\n        c = inter.kb * (norm(dr) - inter.b0)\n        fdr = c * normalize(dr)\n        forces[i] += fdr\n        forces[j] -= fdr\n        # Break bonds randomly\n        if rand() < inter.prob_break\n            delete!(s.atoms[i].partners, j)\n            delete!(s.atoms[j].partners, i)\n        end\n    # Make bonds between close atoms randomly\n    elseif r2 < inter.b0 * inter.dist_formation && rand() < inter.prob_formation\n        push!(s.atoms[i].partners, j)\n        push!(s.atoms[j].partners, i)\n    end\n    return nothing\nend\n\nstruct BondLogger <: Logger\n    n_steps::Int\n    bonds::Vector{BitVector}\nend\n\nfunction Molly.log_property!(logger::BondLogger, s, step_n)\n    if step_n % logger.n_steps == 0\n        bonds = BitVector()\n        for i in 1:length(s.coords)\n            for j in 1:(i - 1)\n                push!(bonds, j in s.atoms[i].partners)\n            end\n        end\n        push!(logger.bonds, bonds)\n    end\nend\n\ntemperature = 0.01\ntimestep = 0.02\nbox_size = 10.0\nn_steps = 2_000\nn_atoms = 200\n\natoms = [BondableAtom(1.0, 0.1, 0.02, Set([])) for i in 1:n_atoms]\ncoords = [box_size .* rand(SVector{2}) for i in 1:n_atoms]\nvelocities = [velocity(1.0, temperature, dims=2) for i in 1:n_atoms]\ngeneral_inters = (SoftSphere(true), BondableInteraction(true, 0.1, 0.1, 1.1, 0.1, 2.0))\n\ns = Simulation(\n    simulator=VelocityVerlet(),\n    atoms=atoms,\n    general_inters=general_inters,\n    coords=coords,\n    velocities=velocities,\n    temperature=temperature,\n    box_size=box_size,\n    neighbour_finder=DistanceNeighbourFinder(trues(n_atoms, n_atoms), 10, 2.0),\n    thermostat=AndersenThermostat(5.0),\n    loggers=Dict(\"coords\" => CoordinateLogger(20, dims=2),\n                    \"bonds\" => BondLogger(20, [])),\n    timestep=timestep,\n    n_steps=n_steps\n)\n\nsimulate!(s)\n\nconnections = Tuple{Int, Int}[]\nfor i in 1:length(s.coords)\n    for j in 1:(i - 1)\n        push!(connections, (i, j))\n    end\nend\n\nvisualize(s.loggers[\"coords\"],\n            box_size,\n            \"sim_mutbond.gif\",\n            connections=connections,\n            connection_frames=s.loggers[\"bonds\"].bonds)(Image: Mutable bond simulation)"
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
    "text": "AndersenThermostat(coupling_const)\n\nRescale random velocities according to the Andersen thermostat.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Angletype",
    "page": "API",
    "title": "Molly.Angletype",
    "category": "type",
    "text": "Angletype(th0, cth)\n\nGromacs angle type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Atom",
    "page": "API",
    "title": "Molly.Atom",
    "category": "type",
    "text": "Atom(; <keyword arguments>)\n\nAn atom and its associated information. Properties unused in the simulation or in analysis can be left with their default values.\n\nArguments\n\nattype::AbstractString=\"\": the type of the atom.\nname::AbstractString=\"\": the name of the atom.\nresnum::Integer=0: the residue number if the atom is part of a polymer.\nresname::AbstractString=\"\": the residue name if the atom is part of a   polymer.\ncharge::T=0.0: the charge of the atom, used for electrostatic interactions.\nmass::T=0.0: the mass of the atom.\nσ::T=0.0: the Lennard-Jones finite distance at which the inter-particle   potential is zero.\nϵ::T=0.0: the Lennard-Jones depth of the potential well.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Atomtype",
    "page": "API",
    "title": "Molly.Atomtype",
    "category": "type",
    "text": "Atomtype(mass, charge, σ, ϵ)\n\nGromacs atom type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Bondtype",
    "page": "API",
    "title": "Molly.Bondtype",
    "category": "type",
    "text": "Bondtype(b0, kb)\n\nGromacs bond type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.CoordinateLogger",
    "page": "API",
    "title": "Molly.CoordinateLogger",
    "category": "type",
    "text": "CoordinateLogger(n_steps; dims=3)\n\nLog the coordinates throughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Coulomb",
    "page": "API",
    "title": "Molly.Coulomb",
    "category": "type",
    "text": "Coulomb(nl_only)\n\nThe Coulomb electrostatic interaction.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.DistanceNeighbourFinder",
    "page": "API",
    "title": "Molly.DistanceNeighbourFinder",
    "category": "type",
    "text": "DistanceNeighbourFinder(nb_matrix, n_steps, dist_cutoff)\nDistanceNeighbourFinder(nb_matrix, n_steps)\n\nFind close atoms by distance.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.GeneralInteraction",
    "page": "API",
    "title": "Molly.GeneralInteraction",
    "category": "type",
    "text": "A general interaction that will apply to all atom pairs. Custom general interactions should sub-type this type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Gravity",
    "page": "API",
    "title": "Molly.Gravity",
    "category": "type",
    "text": "Gravity(nl_only, G)\n\nThe gravitational interaction.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.HarmonicAngle",
    "page": "API",
    "title": "Molly.HarmonicAngle",
    "category": "type",
    "text": "HarmonicAngle(i, j, k, th0, cth)\n\nA bond angle between three atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.HarmonicBond",
    "page": "API",
    "title": "Molly.HarmonicBond",
    "category": "type",
    "text": "HarmonicBond(i, j, b0, kb)\n\nA harmonic bond between two atoms.\n\n\n\n\n\n"
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
    "text": "LennardJones(nl_only)\n\nThe Lennard-Jones 6-12 interaction.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Logger",
    "page": "API",
    "title": "Molly.Logger",
    "category": "type",
    "text": "A way to record a property, e.g. the temperature, throughout a simulation. Custom loggers should sub-type this type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.NeighbourFinder",
    "page": "API",
    "title": "Molly.NeighbourFinder",
    "category": "type",
    "text": "A way to find near atoms to save on simulation time. Custom neighbour finders should sub-type this type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.NoNeighbourFinder",
    "page": "API",
    "title": "Molly.NoNeighbourFinder",
    "category": "type",
    "text": "NoNeighbourFinder()\n\nPlaceholder neighbour finder that returns no neighbours. When using this neighbour finder, ensure that nl_only for the interactions is set to false.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.NoThermostat",
    "page": "API",
    "title": "Molly.NoThermostat",
    "category": "type",
    "text": "NoThermostat()\n\nPlaceholder thermostat that does nothing.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Simulation",
    "page": "API",
    "title": "Molly.Simulation",
    "category": "type",
    "text": "Simulation(; <keyword arguments>)\n\nThe data needed to define and run a molecular simulation. Properties unused in the simulation or in analysis can be left with their default values.\n\nArguments\n\nsimulator::Simulator: the type of simulation to run.\natoms::Vector{A}: the atoms, or atom equivalents, in the simulation. Can be   of any type.\nspecific_inter_lists::SI=(): the specific interactions in the simulation,   i.e. interactions between specific atoms such as bonds or angles. Typically   a Tuple.\ngeneral_inters::GI=(): the general interactions in the simulation, i.e.   interactions between all or most atoms such as electrostatics. Typically a   Tuple.\ncoords::C: the coordinates of the atoms in the simulation. Typically a   Vector of SVectors of any dimension and type T, where T is an   AbstractFloat type.\nvelocities::C=zero(coords): the velocities of the atoms in the simulation,   which should be the same type as the coordinates. The meaning of the   velocities depends on the simulator used, e.g. for the VelocityFreeVerlet   simulator they represent the previous step coordinates for the first step.\ntemperature::T=0.0: the temperature of the simulation.\nbox_size::T: the size of the cube in which the simulation takes place.\nneighbour_finder::NeighbourFinder=NoNeighbourFinder(): the neighbour finder   used to find close atoms and save on computation.\nthermostat::Thermostat=NoThermostat(): the thermostat which applies during   the simulation.\nloggers::Dict{String, <:Logger}=Dict(): the loggers that record properties   of interest during the simulation.\ntimestep::T=0.0: the timestep of the simulation.\nn_steps::Integer=0: the number of steps in the simulation.\nn_steps_made::Vector{Int}=[]: the number of steps already made during the   simulation. This is a Vector to allow the struct to be immutable.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Simulator",
    "page": "API",
    "title": "Molly.Simulator",
    "category": "type",
    "text": "A type of simulation to run, e.g. leap-frog integration or energy minimisation. Custom simulators should sub-type this type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.SoftSphere",
    "page": "API",
    "title": "Molly.SoftSphere",
    "category": "type",
    "text": "SoftSphere(nl_only)\n\nThe soft-sphere potential.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.SpecificInteraction",
    "page": "API",
    "title": "Molly.SpecificInteraction",
    "category": "type",
    "text": "A specific interaction between sets of specific atoms, e.g. a bond angle. Custom specific interactions should sub-type this type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.StructureWriter",
    "page": "API",
    "title": "Molly.StructureWriter",
    "category": "type",
    "text": "StructureWriter(n_steps, filepath)\n\nWrite 3D output structures to the PDB file format throughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.TemperatureLogger",
    "page": "API",
    "title": "Molly.TemperatureLogger",
    "category": "type",
    "text": "TemperatureLogger(n_steps)\nTemperatureLogger(T, n_steps)\n\nLog the temperature throughout a simulation.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Thermostat",
    "page": "API",
    "title": "Molly.Thermostat",
    "category": "type",
    "text": "A way to keep the temperature of a simulation constant. Custom thermostats should sub-type this type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Torsion",
    "page": "API",
    "title": "Molly.Torsion",
    "category": "type",
    "text": "Torsion(i, j, k, l, f1, f2, f3, f4)\n\nA dihedral torsion angle between four atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.Torsiontype",
    "page": "API",
    "title": "Molly.Torsiontype",
    "category": "type",
    "text": "Torsiontype(f1, f2, f3, f4)\n\nGromacs torsion type.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.VelocityFreeVerlet",
    "page": "API",
    "title": "Molly.VelocityFreeVerlet",
    "category": "type",
    "text": "VelocityFreeVerlet()\n\nThe velocity-free Verlet integrator, also known as the Störmer method. In this case the velocities given to the Simulator act as the previous step coordinates for the first step.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.VelocityVerlet",
    "page": "API",
    "title": "Molly.VelocityVerlet",
    "category": "type",
    "text": "VelocityVerlet()\n\nThe velocity Verlet integrator.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.accelerations-Tuple{Simulation,Any}",
    "page": "API",
    "title": "Molly.accelerations",
    "category": "method",
    "text": "accelerations(simulation, neighbours; parallel=true)\n\nCalculate the accelerations of all atoms using the general and specific interactions and Newton\'s second law.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.adjust_bounds-Tuple{Real,Real}",
    "page": "API",
    "title": "Molly.adjust_bounds",
    "category": "method",
    "text": "adjust_bounds(c, box_size)\n\nEnsure a coordinate is within the simulation box and return the coordinate.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.apply_thermostat!-Tuple{Simulation,NoThermostat}",
    "page": "API",
    "title": "Molly.apply_thermostat!",
    "category": "method",
    "text": "apply_thermostat!(simulation, thermostat)\n\nApply a thermostat to modify a simulation. Custom thermostats should implement this function.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.displacements-Tuple{Any,Real}",
    "page": "API",
    "title": "Molly.displacements",
    "category": "method",
    "text": "displacements(coords, box_size)\n\nGet the pairwise vector displacements of a set of coordinates, accounting for the periodic boundary conditions.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.distances-Tuple{Any,Real}",
    "page": "API",
    "title": "Molly.distances",
    "category": "method",
    "text": "distances(coords, box_size)\n\nGet the pairwise distances of a set of coordinates, accounting for the periodic boundary conditions.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.find_neighbours-Tuple{Simulation,Any,NoNeighbourFinder,Integer}",
    "page": "API",
    "title": "Molly.find_neighbours",
    "category": "method",
    "text": "find_neighbours(simulation, current_neighbours, neighbour_finder, step_n; parallel=true)\n\nObtain a list of close atoms in a system. Custom neighbour finders should implement this function.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.force!",
    "page": "API",
    "title": "Molly.force!",
    "category": "function",
    "text": "force!(forces, interaction, simulation, atom_i, atom_j)\n\nUpdate the force for an atom pair in response to a given interation type. Custom interaction types should implement this function.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.log_property!-Tuple{TemperatureLogger,Simulation,Integer}",
    "page": "API",
    "title": "Molly.log_property!",
    "category": "method",
    "text": "log_property!(logger, simulation, step_n)\n\nLog a property thoughout a simulation. Custom loggers should implement this function.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.maxwellboltzmann-Tuple{Type,Real,Real}",
    "page": "API",
    "title": "Molly.maxwellboltzmann",
    "category": "method",
    "text": "maxwellboltzmann(mass, temperature)\nmaxwellboltzmann(T, mass, temperature)\n\nDraw from the Maxwell-Boltzmann distribution.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.rdf-Tuple{Any,Real}",
    "page": "API",
    "title": "Molly.rdf",
    "category": "method",
    "text": "rdf(coords, box_size; npoints=200)\n\nGet the radial distribution function of a set of coordinates. This describes how density varies as a function of distance from each atom. Returns a list of distance bin centres and a list of the corresponding densities.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.readinputs-Tuple{Type,AbstractString,AbstractString}",
    "page": "API",
    "title": "Molly.readinputs",
    "category": "method",
    "text": "readinputs(topology_file, coordinate_file)\nreadinputs(T, topology_file, coordinate_file)\n\nRead a Gromacs topology flat file, i.e. all includes collapsed into one file, and a Gromacs coordinate file. Returns the atoms, specific interaction lists, general interaction lists, non-bonded matrix, coordinates and box size.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.simulate!-Tuple{Simulation,VelocityVerlet,Integer}",
    "page": "API",
    "title": "Molly.simulate!",
    "category": "method",
    "text": "simulate!(simulation; parallel=true)\nsimulate!(simulation, n_steps; parallel=true)\nsimulate!(simulation, simulator, n_steps; parallel=true)\n\nRun a simulation according to the rules of the given simulator. Custom simulators should implement this function.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.temperature-Tuple{Simulation}",
    "page": "API",
    "title": "Molly.temperature",
    "category": "method",
    "text": "temperature(simulation)\n\nCalculate the temperature of a system from the kinetic energy of the atoms.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.vector-Tuple{Any,Any,Real}",
    "page": "API",
    "title": "Molly.vector",
    "category": "method",
    "text": "vector(c1, c2, box_size)\n\nDisplacement between two coordinate values, accounting for the bounding box. The minimum image convention is used, so the displacement is to the closest version of the coordinates accounting for the periodic boundaries.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.vector1D-Tuple{Real,Real,Real}",
    "page": "API",
    "title": "Molly.vector1D",
    "category": "method",
    "text": "vector1D(c1, c2, box_size)\n\nDisplacement between two 1D coordinate values, accounting for the bounding box. The minimum image convention is used, so the displacement is to the closest version of the coordinate accounting for the periodic boundaries.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.velocity-Tuple{Type,Real,Real}",
    "page": "API",
    "title": "Molly.velocity",
    "category": "method",
    "text": "velocity(mass, temperature; dims=3)\nvelocity(T, mass, temperature; dims=3)\n\nGenerate a random velocity from the Maxwell-Boltzmann distribution.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly.visualize",
    "page": "API",
    "title": "Molly.visualize",
    "category": "function",
    "text": "visualize(coord_logger, box_size, out_filepath; <keyword arguments>)\n\nVisualize a simulation as an animation. This function is only available when Makie is imported. It can take a while to run, depending on the length and size of the simulation.\n\nArguments\n\nconnections=Tuple{Int, Int}[]: pairs of atoms indices to link with bonds.\nconnection_frames: the frames in which bonds are shown. Is a list of the   same length as the number of frames, where each item is a list of Bools of   the same length as connections. Defaults to always true.\ntrails::Integer=0: the number of preceding frames to show as transparent   trails.\nframerate::Integer=30: the frame rate of the animation.\ncolor=:purple: the color of the atoms. Can be a single color or a list of   colors of the same length as the number of atoms.\nconnection_color=:orange: the color of the bonds. Can be a single color or a   list of colors of the same length as connections.\nmarkersize=0.1: the size of the atom markers.\nlinewidth=2.0: the width of the bond lines.\ntransparency=true: whether transparency is active on the plot.\nkwargs...: other keyword arguments are passed to the plotting function.\n\n\n\n\n\n"
},

{
    "location": "api.html#Molly-API-1",
    "page": "API",
    "title": "Molly API",
    "category": "section",
    "text": "The API reference can be found here.Molly also re-exports StaticArrays.jl, making the likes of SVector available when you call using Molly.The visualize function is only available once you have called using Makie. Requires.jl is used to lazily load this code when Makie.jl is available, which reduces the dependencies of the package.Order   = [:module, :type, :constant, :function, :macro]Modules = [Molly]\nPrivate = false\nOrder   = [:module, :type, :constant, :function, :macro]"
},

]}
