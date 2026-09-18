# Free energy calculation

## Free energies with MBAR

### A brief introduction

One of the most relevant uses of molecular dynamics (MD) is the estimation of free energy (FE) changes along a given reaction coordinate. One may be interested in, for example, how favorable the binding of a ligand to a target protein is; or which conformer of a molecule is the most stable. These are the kind of questions that can be addressed with FE techniques.

Through the years, researchers have developed a collection of techniques to solve these problems. As early as 1954, Robert W Zwanzig introduced the [FE perturbation](https://doi.org/10.1063/1.1740409) method, leading to the Zwanzig equation:

```math
\Delta F_{A \rightarrow B} = F_B - F_A = k_B T \ln \left \langle \exp\left( - \frac{E_B - E_A}{k_B T} \right) \right \rangle _A
```

This states that, for a given system, the change of FE in going from state *A* to state *B* is equal to the FE difference between the two states and, more importantly, directly related to the total energy difference of states *A* and *B* through Botzmann statistics. In this equation, the angle brackets represent the expected value of the Boltzmann-weighted energy difference of the two states, but sampled only from conformations extracted from state *A*. This implies that, even when sampling only one state, we are able to infer information from the other, given that some niceness criteria is met, i.e. the energy difference between the two states is small enough. This reasoning about unsampled states by re-evaluating sampled states is known as reweighting.

A little over 20 years after Zwanzig introduced FE perturbation, Charles H Bennett expanded on this and developed the [Bennett Acceptance Ratio](https://doi.org/10.1016/0021-9991(76)90078-4) (BAR). Bennett built directly upon the statistical foundation of the Zwanzig equation, and recognized that both forward and reverse energy differences between two states contain complementary information; that is, while Zwanzig’s formulation reweights configurations from a single ensemble to estimate the free energy of another, BAR symmetrizes this process. It combines samples from both states and determines the free energy shift $\Delta F$ that makes each ensemble equally probable under some weighting derived from Boltzmann statistics. In this sense, BAR can be viewed as a generalization of Zwanzig’s exponential averaging, reducing to the Zwanzig equation when only one direction of sampling is available. However, it is because of this that BAR still suffers from the same issue as Zwanzig's reweighting method: the energy difference between states $A$ and $B$ must be small enough such that there is sufficient overlap between their configurational spaces, otherwise the necessary statistics for FE estimation will be very poor and intermediate steps between $A$ and $B$ are needed.

Thus, in 2008 Michael R Shirts and John D Chodera introduced the [Multistate Bennett Acceptance Ratio](https://doi.org/10.1063/1.2978177) (MBAR) method. MBAR expands on BAR by instead of just using two states *A* and *B*, considering a collection of *k* $\in$ *K* different thermodynamic states. The only thing that is expected from these states is that they must be sampled from equivalent thermodynamic ensembles, this is, all states should be NVT or NPT, etc.; but other than that, the specific Hamiltonian for each evaluated state can differ in an arbitrary manner. Then, a series of $n_k$ samples are drawn from each thermodynamic state, until a total of $N = \sum_{k}^{K} n_k$ samples are obtained. By evaluating each sample $n \in N$ with each Hamiltonian $\mathcal{H}_{k}; k \in K$, one obtains a matrix of reduced potentials $u_{nk} = \beta_{k} \left[ E_{k}(n)+p_{k}V(n) \right]$. MBAR then solves a set of self-consistent equations that yield the relative free energies $f_k$ of all states simultaneously, using every sample across all simulations to estimate each state’s free energy in a statistically optimal way. In this sense, MBAR generalizes BAR to an arbitrary number of thermodynamic states and provides the maximum-likelihood, minimum-variance estimator for free energies and ensemble averages, efficiently combining data from overlapping simulations into a unified framework. MBAR also allows the reweighting of any observable to a completely unsampled thermodynamic state. Because this themodynamic state is compared to a collection of sampled states, instead of just one, the conformational space between states is much more likely to overlap, therefore increasing the probability of the reweighting to be meaningful.

## How to run MBAR with Molly

### Defining the restraint interactions

In this example, we will use MBAR to calculate the Potential of Mean Force (PMF) along the central torsion of an alanine dipeptide molecule. In order to do that, we will have to run a series of independent biased simulations, where each will apply an umbrella potential to restrain the peptide torsion fixed around a given angle. We will first define the restraint interaction:

![Alanine dipeptide and the relevant torsion](images/dipeptide.png)

```julia
# restraints.jl file

using LinearAlgebra

struct DihedralRestraint{A, K}
    ϕ0::A # Radians, dimensionless number
    Δϕ::A # Radians, dimensionless number
    k::K  # Energy (e.g. kJ/mol)
end

# Store angles as plain Float64 radians, k should be energy
function DihedralRestraint(ϕ0::Unitful.AbstractQuantity, Δϕ::Unitful.AbstractQuantity, k)
    ϕ0r = ustrip(u"rad", ϕ0)
    Δϕr = ustrip(u"rad", Δϕ)
    return DihedralRestraint(ϕ0r, Δϕr, k)
end
```

Then, we have to define the potential energy and force functions that Molly will call when it encounters such an interaction. We will make use of the machinery present in Molly and define the restraint as a specific interaction, returning the force as a [`SpecificForce4Atoms`](@ref). The functional form used as the bias potential is a quadratic flat bottom angle restraint, using:

```math
\varphi^{\prime} = (\varphi - \varphi^0)\ \%\ 2\pi
```

```math
V\left( \varphi^{\prime} \right) = \begin{cases}
    \frac{1}{2} \cdot k \cdot \left( \varphi^{\prime} - \Delta\varphi \right)^2 & \mathrm{for} \ \lvert \varphi^{\prime} \rvert \ge \Delta\varphi \\
    0 & \mathrm{for} \ \lvert \varphi^{\prime} \rvert \lt \Delta\varphi
\end{cases}
```

Where $\varphi$ and $\varphi^0$ are the current and reference dihedral angles, respectively; $\Delta\varphi$ is the width of the flat bottom of the potential, and $k$ is the energy constant associated with the interaction. The analytic derivation of the force acting on each atom defining the dihedral, given the previous potential, is quite involved and beyond the scope of this tutorial. We do provide here, however, the Julia code used to define the potential energy and force:

```julia
# restraints.jl file

# Robust geometry with clamped inverses
function _dihedral_geom(ci, cj, ck, cl, boundary)
    b1 = vector(cj, ci, boundary)
    b2 = vector(cj, ck, boundary)
    b3 = vector(ck, cl, boundary)

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)

    b2n  = norm(b2)
    n1n2 = dot(n1, n1)
    n2n2 = dot(n2, n2)

    # Angle via atan2
    y = b2n * dot(b1, n2)
    x = dot(n1, n2)
    ϕ = atan(y, x)

    return ϕ, b1, b2, b3, n1, n2, b2n, n1n2, n2n2, x, y
end

_wrap_pi(x::Real) = (x + π) % (2π) - π # Take into account periodicity
_rad(x::Real) = Float64(x)
_rad(x) = Float64(ustrip(u"rad", x))

function Molly.potential_energy(inter::DihedralRestraint{FT, K},
                                ci, cj, ck, cl, boundary, args...) where {FT, K}
    b1 = vector(cj, ci, boundary)
    b2 = vector(cj, ck, boundary)
    b3 = vector(ck, cl, boundary)
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)

    b2n  = norm(b2)
    n1n2 = dot(n1, n1)
    n2n2 = dot(n2, n2)

    y = b2n * dot(b1, n2)
    x = dot(n1, n2)
    ϕ = atan(y, x)

    # Scale-aware tolerances
    Ls = max(ustrip(norm(b1) + norm(b2) + norm(b3)), 1.0) * oneunit(norm(b1))
    tol_b  = 1e-12 * Ls
    tol_xy = 1e-24 * (Ls^4)

    if !isfinite(ϕ) || b2n ≤ tol_b || abs(x) ≤ tol_xy || abs(y) ≤ tol_xy ||
                    sqrt(n1n2) ≤ tol_b^2 || sqrt(n2n2) ≤ tol_b^2
        return zero(inter.k)
    end

    ϕ0 = _rad(inter.ϕ0)
    Δ  = _rad(inter.Δϕ)
    d  = _wrap_pi(ϕ - ϕ0)
    ad = abs(d)

    if ad ≤ Δ
        return zero(inter.k)
    else
        diff = ad - Δ
        return FT(inter.k * (diff * diff) / 2)
    end
end

function Molly.force(inter::DihedralRestraint{FT, K},
                     ci, cj, ck, cl, boundary, args...) where {FT, K}
    b1 = vector(cj, ci, boundary)
    b2 = vector(cj, ck, boundary)
    b3 = vector(ck, cl, boundary)
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)

    b2n  = norm(b2)
    n1n2 = dot(n1, n1)
    n2n2 = dot(n2, n2)

    y = b2n * dot(b1, n2)
    x = dot(n1, n2)
    ϕ = atan(y, x)

    # Zero force with correct units (energy/length)
    F0 = FT(zero(inter.k) / oneunit(norm(b1)))
    Fz = SVector(F0, F0, F0)

    # Tolerances
    Ls = max(ustrip(norm(b1) + norm(b2) + norm(b3)), 1.0) * oneunit(norm(b1))
    tol_b  = 1e-12 * Ls
    tol_xy = 1e-24 * (Ls^4)

    if !isfinite(ϕ) || b2n ≤ tol_b || abs(x) ≤ tol_xy || abs(y) ≤ tol_xy ||
                    sqrt(n1n2) ≤ tol_b^2 || sqrt(n2n2) ≤ tol_b^2
        return SpecificForce4Atoms(Fz, Fz, Fz, Fz)
    end

    ϕ0 = _rad(inter.ϕ0)
    Δ  = _rad(inter.Δϕ)
    d  = _wrap_pi(ϕ - ϕ0)
    ad = abs(d)
    if ad ≤ Δ
        return SpecificForce4Atoms(Fz, Fz, Fz, Fz)
    end

    # dU/dϕ as energy
    dU_dϕ = inter.k * (ad - Δ) * (d ≥ 0 ? 1.0 : -1.0)

    # Safe inverses
    εL4 = 1e-32 * (Ls^4)
    εL2 = 1e-32 * (Ls^2)
    inv_n1  = 1 / max(n1n2, εL4)
    inv_n2  = 1 / max(n2n2, εL4)
    b22     = dot(b2, b2)
    inv_b22 = 1 / max(b22,  εL2)

    # Gradients (all ~ 1/L)
    g1 = (b2n * inv_n1) * n1
    g4 = (b2n * inv_n2) * n2
    s1 = dot(b1, b2) * inv_b22
    s3 = dot(b3, b2) * inv_b22
    g2 = -g1 + s1*g1 - s3*g4
    g3 = -g4 + s3*g4 - s1*g1

    F1 = -(dU_dϕ) * g1
    F2 = -(dU_dϕ) * g2
    F3 = -(dU_dϕ) * g3
    F4 = -(dU_dϕ) * g4
    return SpecificForce4Atoms(FT.(F1), FT.(F2), FT.(F3), FT.(F4))
end
```

### Setting up simulations

With this in hand, we are ready to set up the individual biased simulations. We will explore a full torsion around the dihedral, i.e. spanning 360 degrees. We will do so in 60 independent biased simulations, so 360 degrees / 60 simulations = 6 degrees increments per simulation. The system is quite well-behaved, so we can get away with using Float32 precision on GPU. We will use a time step of 1 fs to integrate the equations of motion, and will run the simulations in the NPT ensemble at 310 K and 1 bar of pressure.

```julia
# pulling.jl

using Molly
using CUDA

include("restraints.jl")

FT     = Float32               # Float precision
AT     = CuArray               # Array type, run simulations on CUDA GPU
N_WIN  = 60                    # Number of umbrella windows to generate
dR     = FT(6)u"deg"           # Increment in CV (torsion angle) in each consecutive window
ΔR     = FT(3)u"deg"           # Width of flat bottom potential
K_bias = FT(250)u"kJ * mol^-1" # Energy constant for restraint potential
Δt     = FT(1)u"fs"            # Simulation timestep
T0     = FT(310)u"K"           # Simulation temperature
P0     = FT(1)u"bar"           # Simulation pessure
```

One must take into account that the simulations will start from an initial (ideally equilibrated) configuration, and therefore an arbitrarirly imposed restraint may be too far away from the equilibrium distribution of the CV to be biased, causing numerical issues. Thus, we will begin our setup by gently pulling the system along the CV in a series of short, sequential simulations. How short? The answer to that question depends on the system to be simulated; the pulling simulations should be long enough so that the system has time to move towards and stabilize around the imposed biased equilibrium, but also sufficiently short as to not waste time in this initial sequential part, as once each umbrella window is equilibrated it can run in parallel with the rest. For this simple study case, equilibrating each window for 0.5 ns is enough. We can define:

```julia
# pulling.jl

tu      = unit(Δt)                   # Time units used for timestep
max_t   = uconvert(u, FT(0.5)u"ns")  # Simulation time in appropriate time units
N_STEPS = Int(floor(max_t / Δt))     # Number of simulation steps
```

We can now load the initial configuration into a [`System`](@ref):

```julia
# pulling.jl

data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir = joinpath(data_dir, "force_fields")

ff = MolecularForceField(
    joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
    units=true,
)

sys_0 = System(
    joinpath(data_dir, "..", "exercises", "dipeptide_equil.pdb"),
    ff;
    array_type=AT,
    float_type=FT,
    nonbonded_method=SetupCoulombReactionField(),
)

random_velocities!(sys_0, T0) # Initialize velocities from M-B distribution at target temperature
```

Now, before starting to produce the pulling simulations, we need to know what is the value of the CV (remember, the torsion angle) for an equilibrated system. We can make use of the functions defined in `restraints.jl`. Remember to take note of this value, it will be important later!

```julia
# pulling.jl

# Indices of the atoms defining the dihedral
i = 17
j = 15
k = 9
l = 7

coords_cpu = Molly.from_device(sys_0.coords)
eq_θ,      = _dihedral_geom(coords_cpu[i], coords_cpu[j], coords_cpu[k], coords_cpu[l], sys_0.boundary)
eq_θ       = _wrap_pi(eq_θ) # = -0.48948112328583804 rad
```

We are getting very close to running the pulling simulations. The only remaining things to define are the coupling algorithms to keep a constant temperature and pressure, which integrator to use, and also tell Molly every how many integration steps should we write the coordinates to a trajectory file:

```julia
# pulling.jl

τ_T        = FT(1)u"ps" # Thermostat coupling constant
# Apply thermostat every simulation step, required for Verlet-type integrators
thermostat = VelocityRescaleThermostat(T0, τ_T, n_steps = 1)

τ_P        = FT(1)u"ps" # Barostat coupling constant
# Apply barostat 10 times per τ_P, good balance of precision and computational overhead
frac       = uconvert(tu, 0.1 * τ_P)
n_P        = Int(floor(frac/Δt)) # The number of simulation steps
barostat   = CRescaleBarostat(P0, τ_P; n_steps = n_P)

# Create the integrator, remove COM motion every 100 steps
vverlet    = VelocityVerlet(Δt, (thermostat, barostat,), 100)

save_t     = uconvert(tu, FT(1)u"ps") # Save coordinates every picosecond
save_steps = Int(floor(save_t / Δt))  # The number of simulation steps
```

With all of this ready, we only need to sequentially run the pulling simulations. We use the coordinates and velocities at the end of simulation n to seed the beginning of simulation n + 1:

```julia
# pulling.jl

old_sys = deepcopy(sys_0) # Get a copy of the initial system
sils    = deepcopy(sys_0.specific_inter_lists) # Get the interactions lists of the unbiased system
for w in 1:N_WIN
    # Calculate where is the potential well located for a given window
    rest_θ = FT(eq_θ - (w-1) * uconvert(u"rad", dR))
    # Create the Dihedral restraint given our parameters
    dRest = DihedralRestraint(rest_θ*u"rad", uconvert(u"rad", ΔR) , K_bias)

    # Pack restraint into an independent interaction list
    rest_inter = InteractionList4Atoms(
        Molly.to_device([i], AT),
        Molly.to_device([j], AT),
        Molly.to_device([k], AT),
        Molly.to_device([l], AT),
        Molly.to_device([dRest], AT),
    )

    rest_inter = (sils..., rest_inter,) # Merge unbiased and biased into single tuple

    sys_w = System(
        deepcopy(old_sys); # Get the same layout as the unbiased system
        specific_inter_lists=rest_inter, # Overwrite interaction list with the one containing the bias
        loggers=(TrajectoryWriter(save_steps, "./pull_$(w).dcd"),),
    )

    simulate!(sys_w, vverlet, N_STEPS)

    # We also write the very last structure to a pdb file
    write_structure("./pull_$(w).pdb", sys_w)

    global old_sys = sys_w # Override old system with the newly simulated one
end
```

### Running umbrella simulations

Once we have run the pulling, we can write a small standalone script to produce the umbrella sampling simulations. These can be run in parallel as they are independent. The simulation setup must be exactly the same used to produce the pulling, except for the amount of time the simulations will be run for. In our case, each simulation was run for a total of 50 ns:

```julia
# individual_simulation.jl

using Molly
using CUDA

include("restraints.jl")

SIM_N = parse(Int, ARGS[1])      # Take the umbrella index as the first argument

FT     = Float32               # Float precision
AT     = CuArray               # Array type, run simulations on CUDA GPU
N_WIN  = 60                    # Number of umbrella windows to generate
dR     = FT(6)u"deg"           # Increment in CV (torsion angle) in each consecutive window
ΔR     = FT(3)u"deg"           # Width of flat bottom potential
K_bias = FT(250)u"kJ * mol^-1" # Energy constant for restraint potential
Δt     = FT(1)u"fs"            # Simulation timestep
T0     = FT(310)u"K"           # Simulation temperature
P0     = FT(1)u"bar"           # Simulation pessure

data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir = joinpath(data_dir, "force_fields")

ff = MolecularForceField(
    joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
    units=true,
)

sys = System(
    "pull_$(SIM_N).pdb", # Now we load the final structure for a given pull simulation
    ff;
    array_type=AT,
    float_type=FT,
    nonbonded_method=SetupCoulombReactionField(),
)

random_velocities!(sys, T0)

# Indices of atoms defining the dihedral
i = 17
j = 15
k = 9
l = 7

eq_θ = FT(-0.48948112328583804) # We get this from the equilibrium structure

tu         = unit(Δt)                 # Time units used for timestep
max_t      = uconvert(u, FT(50)u"ns") # Simulation time in appropriate time units
N_STEPS    = Int(floor(max_t / Δt))   # Number of simulation steps

τ_T        = FT(1)u"ps"               # Thermostat coupling constant
thermostat = VelocityRescaleThermostat(T0, τ_T, n_steps=1) # Apply thermostat every simulation step

τ_P        = FT(1)u"ps"               # Barostat coupling constant
frac       = uconvert(tu, 0.1 * τ_P)  # Apply barostat 10 times per τ_P, good balance of precision and computational overhead
n_P        = Int(floor(frac/Δt))      # To number of simulation steps
barostat   = CRescaleBarostat(P0, τ_P; n_steps=n_P)

vverlet    = VelocityVerlet(Δt, (thermostat, barostat,), 100) # Create the integrator, remove COM motion every 100 steps

save_t     = uconvert(tu, FT(1)u"ps")  # Save coordinates every picosecond
save_steps = Int(floor(save_t / Δt))   # To number of simulation steps

rest_θ = FT(eq_θ - (SIM_N-1) * uconvert(u"rad", dR)) # The equilibrium value for the bias
dRest  = DihedralRestraint(rest_θ*u"rad", uconvert(u"rad", ΔR) , K_bias) # Create interaction

# Pack into interaction list
rest_inter = InteractionList4Atoms(
    Molly.to_device([i], AT),
    Molly.to_device([j], AT),
    Molly.to_device([k], AT),
    Molly.to_device([l], AT),
    Molly.to_device([dRest], AT),
)

sils       = deepcopy(sys.specific_inter_lists) # Unbiased
rest_inter = (sils..., rest_inter,)             # Merge biased and unbiased

sys = System(
    deepcopy(sys); # Everything from the unbiased system
    specific_inter_lists=rest_inter, # Overwrite specific interactions
    loggers=(trj=TrajectoryWriter(save_steps, "./umbrella_$(SIM_N).dcd"),),
)

simulate!(sys, vverlet, N_STEPS)
```

### Calculating free energies with MBAR

Once all of the individual umbrella simulations are finished, we are ready to run MBAR on the results and estimate the free energy along our reaction coordinate. Remember from the first section of this tutorial that the MBAR equations are solved by evaluating every generated conformation with every used Hamiltonian. It follows, then, that we will first have to set up an array of [`System`](@ref) structs that represent each Hamiltonian. Moreover, we will be reading data from trajectories, so those Systems will actually be wrapped inside [`EnsembleSystem`](@ref) structs, which allow IO operations from trajectory files into data structures usable by Molly.

We start by defining variables that will be shared by all thermodynamic states. Notice how many things are shared with the parameters used to produce the simulations!

```julia
# MBAR.jl

using Molly
using CUDA

include("restraints.jl")

AT = CuArray
FT = Float32

# Bias parameters
dR     = FT(6)u"deg"           # Increment of CV in each umbrella window
ΔR     = FT(3)u"deg"           # Width for flat bottom potential
K_bias = FT(250)u"kJ * mol^-1" # Force used in the restraint

temp = FT(310)u"K"
pres = FT(1)u"bar"

data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir = joinpath(data_dir, "force_fields")

trajs_dir = "./" # Or wherever you have saved the umbrella simulations

ff = MolecularForceField(
    joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...;
    units=true,
)

sys_nobias = System(
    joinpath(data_dir, "..", "exercises", "dipeptide_equil.pdb"),
    ff;
    array_type=AT,
    float_type=FT,
    nonbonded_method=SetupCoulombReactionField(),
)

# Atom indices defining dihedral
i = 17
j = 15
k = 9
l = 7

eq_θ  = FT(-0.48948112328583804) # We get this from the equilibrium structure

N_TRJ       = 60 # The number of umbrella simulations produced
TRJ_SYSTEMS = Vector{EnsembleSystem}(undef, N_TRJ)

Threads.@threads for trj_n in 1:N_TRJ
    traj_path = joinpath(trajs_dir, "umbrella_$(trj_n).dcd")

    # Note that the restraint and specific interactions list must be created
    # in exactly the same way as for the umbrella simulations, we need exactly
    # the same Hamiltonian
    rest_θ = FT(eq_θ - (trj_n-1) * uconvert(u"rad", dR))
    dRest = DihedralRestraint(rest_θ*u"rad", uconvert(u"rad", ΔR) , K_bias)

    rest_inter = InteractionList4Atoms(
        Molly.to_device([i], AT),
        Molly.to_device([j], AT),
        Molly.to_device([k], AT),
        Molly.to_device([l], AT),
        Molly.to_device([dRest], AT),
    )

    sils = deepcopy(sys_nobias.specific_inter_lists)

    rest_sils = (sils..., rest_inter,)
    sys_rest = System(sys_nobias; specific_inter_lists=rest_sils)

    # Store in struct that allows reading trajectories
    sys_trj = EnsembleSystem(sys_rest, traj_path)

    TRJ_SYSTEMS[trj_n] = sys_trj
end
```

We now have a vector of structs that represent each system and its trajectory. The next step is to read the trajectories and sample the relevant magnitudes to solve MBAR and get our PMF. We will need the coordinates, the system boundaries (needed to calculate the volume for the $pV$ terms of the Hamiltonian, as we have run the simulations in the NPT ensemble) and, of course, the CV of interest. Note that MBAR assumes statistical independence of samples, so the selected conformations must be subsampled from decorrelated states. Molly does provide the functionality to estimate the statistical inefficiency of a given timeseries and subsequent subsampling.

```julia
# MBAR.jl

C  = Vector{<:Any}(undef, N_TRJ) # Vector to store coordinates
B  = Vector{<:Any}(undef, N_TRJ) # Vector to store boundaries
CV = Vector{<:Any}(undef, N_TRJ) # Vector to store the CV

# We discard the first 12500 frames (12.5 ns), assume system is still equilibrating there
FIRST_IDX = 12_500

Threads.@threads for nt in 1:N_TRJ
    trjsys = TRJ_SYSTEMS[nt]
    n_frames = Int(length(trjsys.trajectory))

    # Temp arrays to store potential energy, coordinates, boundaries and CV
    u, c, b, cv  = [], [], [], []

    # Iterate over trajectory frames
    for n in FIRST_IDX:n_frames
        current_sys = read_frame!(trjsys, n) # Read the current frame as a System
        pe = potential_energy(current_sys)
        coords = Molly.from_device(current_sys.coords)
        boundary = current_sys.boundary

        # Measure the CV at the current frame
        ϕ = _dihedral_geom(coords[i], coords[j], coords[k], coords[l], boundary)

        push!(u, pe)
        push!(c, coords)
        push!(b, boundary)
        push!(cv, ϕ*u"rad")
    end

    # Estimate the decorrelation time from the timeseries of the potential energy
    ineff = Molly.statistical_inefficiency(u; maxlag=n_frames-1)

    # Subsample arrays based on statistical inefficiency
    sub_coords = Molly.subsample(c,  ineff.stride; first=1)
    sub_bounds = Molly.subsample(b,  ineff.stride; first=1)
    sub_CV     = Molly.subsample(cv, ineff.stride; first=1)

    C[nt]  = sub_coords
    B[nt]  = sub_bounds
    CV[nt] = sub_CV
end
```

Next, we define a vector of thermodynamic states to represent each Hamiltonian used to produce the simulations, as well a single state that represents the system in the absence of bias potentials. We will use the [`ThermoState`](@ref) struct provided by Molly:

```julia
# MBAR.jl

# Assemble the thermodynamic systems for each umbrella window
energy_units = TRJ_SYSTEMS[1].system.energy_units
kBT          = uconvert(energy_units, Unitful.R * temp)
βi           = Float64(ustrip(1.0 / kBT))

states = ThermoState[ThermoState("win_$i", βi, pres, TRJ_SYSTEMS[i].system)
                     for i in eachindex(TRJ_SYSTEMS)]

target_state = ThermoState("target", βi, pres, sys_nobias)
```

We are finally in possession of everything needed to solve the MBAR equations and estimate the PMF along our CV! There are two paths we can take for this, the long path and the short path. For the sake of completeness, we describe the long path first. The first step for solving MBAR is assembling the reduced energy matrix. Molly provides a functionality just for that through the [`assemble_mbar_inputs`](@ref) method:

```julia
# MBAR.jl

mbar_gen = assemble_mbar_inputs(
    C, B, states; # Coordinates, boundaries and thermodynamic states
    target_state=target_state, # The target state, in our case the unbiased system
    energy_units=energy_units,
)

u        = mbar_gen.u        # Reduced energy matrix, K states by N sampled conformations
u_target = mbar_gen.u_target # The reduced energy of the N samples evaluated by the target state hamiltonian
N_counts = mbar_gen.N        # Number of sampled conformations
win_of   = mbar_gen.win_of   # Indexing helper that tells which k state was used to generate each n sample
shifts   = mbar_gen.shifts   # Numerical shifts, if used, for stability reasons when building the reduced energy matrix
```

This generates the necessary inputs to use the self-consistent iteration method to solve the MBAR equations. Of course, Molly provides the [`iterate_mbar`](@ref) method to do so:

```julia
# MBAR.jl

# Returns the relative free energy of each k state and log.(N_counts), needed for downstream computations
F_k, logN = iterate_mbar(u, win_of, N_counts)
```

With this, we can also produce a weight matrix and a vector of target weights, using the  [`mbar_weights`](@ref) method, that will let us reweight any arbitrary quantity from the sampled K thermodynamic states to any target state:

```julia
# MBAR.jl

# Returns the weights matrix and the target weights
W_s, w_target = mbar_weights(u, u_target, F_k, logN, N_counts; check=true, shifts=shifts)
```

And finally, we can estimate the PMF using the output of the previous step by calling the [`pmf_with_uncertainty`](@ref) method:

```julia
# MBAR.jl

pmf_result = pmf_with_uncertainty(u, u_target, F_k, N_counts, logN, CV; shifts=shifts, kBT=kBT)

centers   = pmf_result.centers        # The collective variable
PMF       = pmf_result.F              # PMF in kBT
PMF_enr   = pmf_result.F_energy       # PMF in energy units
sigma     = pmf_result.sigma_F        # Standard deviation in kBT
sigma_enr = pmf_result.sigma_F_energy # Standard deviation in energy units
```

But what about the short path? Well, we also provide an overload of the [`pmf_with_uncertainty`](@ref) method that allows to get the PMF in a single call by doing:

```julia
# MBAR.jl
pmf_result = pmf_with_uncertainty(
    C,            # Coordinates
    B,            # Boundaries
    states,       # Themodynamic states
    target_state, # Target state
    CV,           # Collective variable
)
```

Now one can put this into a graph, for example using a scatter for the free energy and making use of the calculated sigmas (see the previous code blocks) to shade the plot and give a feel for the uncertainties. The code is left as an exercise to the reader, but the results should look like something similar to this:

![PMF along the dipeptide torsion in kBT units](images/dihedral_pmf_kbt.png)
![PMF along the dipeptide torsion in energy units](images/dihedral_pmf_enr.png)

## Absolute Solvation/Binding Free Energies

### Absolute Solvation Free Energy

Absolute Solvation Free Energy (ASFE) calculations estimate the thermodynamic work required to transfer a single molecule from the gas phase into a solvent ($\Delta G_{\text{solv}}$). Rather than simulating the physical transfer process, ASFE uses an alchemical transformation pathway that decouples the solute’s electrostatic and van der Waals interactions with the solvent or in vacuum.

```
                    ΔG_solvation
    Solute (Vacuum)  ─────────►  Solute (Solvent)
        │                             │   
        │ ΔG_vac                      │ ΔG_solv
        ▼                             ▼   
    Null (Vacuum)    ─────────►   Null (Solvent)
                        ΔG=0
```
The absolute solvation free energy is calculated via thermodynamic cycle closure:

$$\Delta G_{\text{solv}} = \Delta G_{\text{lig}}^{\text{vac}} - \Delta G_{\text{lig}}^{\text{solv}}$$

### Absolute Binding Free Energy

Absolute Binding Free Energy (ABFE) calculations estimate the total binding affinity of a single ligand to a target protein ($\Delta G_{\text{bind}}^{\circ}$). Instead of simulating the slow physical association or dissociation trajectory, ABFE employs a thermodynamic cycle that alchemically annihilates the ligand in the protein-bound complex while also running a simulation in which the interactions are decoupled between ligand and solvent.

```
              ΔG_bind°
      Ligand ─────────►  Protein + Ligand
        │                    │
        │ ΔG_solv            │ ΔG_complex
        ▼                    ▼
       Null  ─────────►  Protein + Null
                ΔG=0         
```

The absolute binding free energy is computed by closing the thermodynamic cycle and applying positional/orientational restraints to keep the ligand in the binding pocket during decoupling:

$$\Delta G_{\text{bind}}^{\circ} = \Delta G^{\text{solv}}_{lig} - \Delta G^{\text{complex}}_{lig} + \Delta G_{\text{restraint}}$$

where $\Delta G_{\text{restraint}}$ explicitly accounts for the analytical cost of removing orientational and translational restraints in bulk solvent.

## Relative Binding Free Energies

Relative Binding Free Energy (RBFE) calculations estimate the difference in binding affinity between two chemically related ligands ($\Delta\Delta G_{bind}$) bound to a target protein. Rather than simulating the physical unbinding process directly, RBFE employs a thermodynamic cycle that transforms Ligand A into Ligand B in both the bound state (complexed with the protein in solvent) and the unbound state (ligand in solvent alone).          

```
                ΔG_bind(A)
      Ligand A  ─────────►  Complex A
        │                       │
        │ ΔG_solv(A→B)          │ ΔG_complex(A→B)
        ▼                       ▼
      Ligand B  ─────────►  Complex B
                ΔG_bind(B)
```

The relative binding free energy is calculated using the thermodynamic cycle closure:

$$\Delta\Delta G = \Delta G_{bind}(B) - \Delta G_{bind}(A) = \Delta G_{bound}(A \to B) - \Delta G_{solv}(A \to B)$$

## Hybrid System Setup

Molly allows for two setup options: single topology (parameter scaling) and dual topology (energy scaling). In a single topology for RBFE, shared atoms between ligands (Core atoms) retain their physical identity, with parameters (charges, vdW, bonds, angles) interpolated along an alchemical coupling parameter $\lambda \in [0, 1]$. Atoms present in only one ligand (Unique atoms) are smoothly created or decoupled using soft-core potentials for electrostatic and van der Waals interactions to prevent numerical singularities at $\lambda \to 0$ and $\lambda \to 1$. In a dual topology for RBFE, virtual atoms are added for the core atoms, these represent the core atoms in ligand B. Unique atoms for ligand B are added as atoms. For ABFE simulations, dual topology is most commonly used where the whole solute is annihilated/decoupled along an alchemical coupling parameter $\lambda \in [0, 1]$, but Molly could run ABFE simulations with single topology.

The mathematical difference between single vs dual topology is the effect of parameter scaling vs energy scaling on the total energy. In dual topology, energy scaling is used and the energy for LennardJones is calculated as:

$$E^{LJ}_{ij} = (1-\lambda)4\epsilon_{A}\left[ \left(\frac{\sigma_{A}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{A}}{r_{ij}}\right)^6\right] + \lambda4\epsilon_{B}\left[ \left(\frac{\sigma_{B}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{B}}{r_{ij}}\right)^6\right]$$
where
$$\epsilon_{X} = \sqrt{\epsilon^{X}_{i}\cdot\epsilon^{X}_{j}}$$ and 
$$\sigma_{X} = \frac{\sigma^{X}_{i}+\sigma^{X}_{j}}{2}$$

whereas in single topology the parameters are scaled:

$$E^{LJ}_{ij} = 4\epsilon_{AB}\left[ \left(\frac{\sigma_{AB}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{AB}}{r_{ij}}\right)^6\right]$$
where
$$\epsilon_{AB} = (1-\lambda)\sqrt{\epsilon^{A}_{i}\cdot\epsilon^{A}_{j}} + \lambda\sqrt{\epsilon^{B}_{i}\cdot\epsilon^{B}_{j}}$$
and
$$\sigma_{AB} = (1-\lambda)\frac{\sigma^{A}_{i}+\sigma^{A}_{j}}{2} + \lambda\frac{\sigma^{B}_{i}+\sigma^{B}_{j}}{2}$$

However, with a single topology, you could also choose between individual parameter scaling or state parameter scaling, which have different effects on interpolation. State parameter scaling is shown above where sigmas and epsilons are first mixed for each state and then interpolated using lambda. In individual parameter scaling, the parameters are first scaled before mixing.

State parameter scaling:
$$\sigma_{AB} = (1-\lambda)\frac{\sigma^{A}_{i}+\sigma^{A}_{j}}{2} + \lambda\frac{\sigma^{B}_{i}+\sigma^{B}_{j}}{2}$$

Individual parameter scaling:
$$\sigma_{i} = (1-\lambda)\sigma^{A}_{i}+\lambda\sigma^{B}_{i}$$;
$$\sigma_{j} = (1-\lambda)\sigma^{A}_{j}+\lambda\sigma^{B}_{j}$$;
$$\sigma_{AB} = \frac{\sigma_{i}+\sigma_{j}}{2}$$

The only exception in a single topology setup are the torsion potentials. These are always energy scaled rather than parameter scaled:

$$E_{torsion} = (1-\lambda)E^A_{torsion} + \lambda E^B_{torsion}$$

In Molly, a topology is chosen together with the lambda scheduler, see [`DefaultLambdaScheduler`](@ref):
```julia
# Default lambda scheduler, where electrostatics are turned off before sterics are turned off.
# The defaults are shown. With dual topology only the intraLJ option is used.
DefaultLambdaScheduler(
    dual=true,          # Dual topology (true) or single topology (false)
    LJindividual=false, # State (false) or individual (true) param scaling for LJ
    LJspecial=false,    # State (false) or individual (true) param scaling for LJ 1-4 inter.
    Cindividual=false,  # State (false) or individual (true) param scaling for Cou
    Cspecial=false,     # State (false) or individual (true) param scaling for Cou 1-4 inter.
    intraLJ=false,      # LJ between alchemical atoms is decoupled (false) or kept on (true)
                      )
```
the default is dual topology while in OpenFE single topology is used, OpenFE also uses other settings during set up and the differences between default and OpenFE are outlined below.

The other schedulers take the same options and differ in how the electrostatics and sterics are staged along $\lambda$: [`LinearLambdaScheduler`](@ref), [`GROMACSLambdaABFEScheduler`](@ref) and [`GROMACSLambdaRBFEScheduler`](@ref) (which use the two PME grids of GROMACS), [`NAMDLambdaScheduler`](@ref), [`QuartersLambdaScheduler`](@ref) and [`EleScaledLambdaScheduler`](@ref). [`OpenFEScheduler`](@ref) is described below.

### Matching OpenFE default settings

Molly has many more options for running RBFE calculations than OpenFE including different softcore potentials, variety of $\lambda$-scaled bonded energy functions (e.g. CMAP torsions, periodic and harmonic torsion, etc.), lambda schedulers, and the option for either single or dual topology. To match the OpenFE settings and architectural choices for setup, we have implemented the `OpenFEScheduler`. When using `OpenFEScheduler(), LJsoftcore="gapsys", Csoftcore="scaled"`, the system setup will be performed similar to the system setup in [OpenFE](https://github.com/OpenFreeEnergy/openfe/blob/main/src/openfe/protocols/openmm_rfe/_rfe_utils/relative.py), which uses a single topology and specific setup choices, including:

- Only using a softcore potential for the LennardJones potential, the CoulombEwald direct space is scaled by lambda directly without using a softcore.
- In CoulombEwald potential, the charges are individually scaled for the regular interactions: 

    $$q_{ij} = ((1-\lambda)q^A_i+\lambda q^B_i) \cdot ((1-\lambda)q^A_j+\lambda q^B_j)$$ 

    But state scaled for the 1-4 interactions: 

    $$q_{ij} = (1-\lambda)(q^A_iq^A_j) + \lambda(q^B_iq^B_j)$$

    For the LennardJones, the gapsys softcore is used and all parameters are state scaled.
- Atoms from the molecules (both core and unique atoms) are not contributing to the LJDispersionCorrection (OpenFE puts the epsilons of ligands to 0). BUT the alchemical atoms are counted towards the number of particles which is used to calculate averages.

So to run with OpenFE settings:
```julia
# OpenFE lambda scheduler with its defaults: single topology, state parameter scaling for LJ,
# LJ 1-4 and Coulomb 1-4 interactions, and individual parameter scaling for Coulomb.
scheduler = OpenFEScheduler(
                dual=false,          # Single topology
                LJindividual=false,  # State parameter scaling for LJ
                LJspecial=false,     # State parameter scaling for LJ 1-4 interactions
                Cindividual=true,    # Individual parameter scaling for Coulomb
                Cspecial=false,      # State parameter scaling for Coulomb 1-4 interactions
                intraLJ=false,       # LJ between alchemical atoms is decoupled
                            )

# For system setup, set LJsoftcore="gapsys", Csoftcore="scaled"
# More on system setup, see the example below
sys = RelativeFESystem(sysA, sysB, global_λ, mapping, core_mapAB;
                        scheduler=scheduler,
                        LJsoftcore="gapsys",
                        Csoftcore="scaled"
                      )
```

All the settings have been tested by comparing energies and forces between Molly and OpenFE, and the comparison is part of the test suite (`test/free_energy.jl`).

### Example set up for ABFE
Here we demonstrate how to set up a hybrid system to annihilate a benzene molecule in water and in vacuum.

First, we load benzene in water and benzene in vacuum with respective choices for electrostatic and steric interactions. The solvated system will use PME while the vacuum system uses short-range Coulomb with an infinite boundary and infinite non-bonded cutoff.

```julia
using Molly

# --- Variables ---
FT = Float64
AT = Array
data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir   = joinpath(data_dir, "force_fields")

# --- Force Field Setup ---
ff = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", "benzene.xml"])...; 
                            units=true, float_type=FT)

# --- Load Systems ---
sysb_solv = System(
        joinpath(data_dir, "benzene_solv_4.pdb"),
        ff;
        array_type=AT,
        float_type=FT,
        dist_cutoff=FT(1) * u"nm",
        dist_buffer=FT(0.2) * u"nm",
        nonbonded_method=SetupPME(),
        constraints=:hbonds,
        constraint_algorithm=SetupLINCS(),
        rigid_water=true,
        hydrogen_mass=3
    )

sysb_vac = System(
        joinpath(data_dir, "benzene_vac.pdb"),
        ff;
        array_type=AT,
        float_type=FT,
        boundary=CubicBoundary(FT(Inf) * u"nm"),
        dist_cutoff=FT(Inf) * u"nm",
        dist_buffer=FT(0) * u"nm",
        neighbor_finder_type=DistanceNeighborFinder,
        nonbonded_method=DistanceCutoff(FT(Inf) * u"nm"),
        constraints=:hbonds,
        constraint_algorithm=SetupLINCS(),
        rigid_water=true,
        hydrogen_mass=3
    )

# --- Mapping of solute indexes ---
mapping = collect(1:12)
```

Then we can use these systems to set up a hybrid system for each leg (solvated vs vacuum) with a dual topology `dual=true`, which absolute systems require. The benzene atoms are fully coupled at `global_λ = 0` and fully decoupled at `global_λ = 1`. The intramolecular electrostatics of benzene are annihilated together with its charges, as the PME mesh requires.

```julia
# --- Hybrid System Setup ---
global_λ = FT(0.0)
scheduler = DefaultLambdaScheduler(dual=true, intraLJ=true)

sys_solv = AbsoluteFESystem(sysb_solv, global_λ, mapping;
                             temp=298.0u"K",
                             units=true,
                             scheduler=scheduler,
                             loggers=(),
                             array_type=AT,
                             float_type=FT,
                             LJsoftcore="beutler",
                             Csoftcore="scaled"
                            )

sys_vac = AbsoluteFESystem(sysb_vac, global_λ, mapping;
                            temp=298.0u"K",
                            units=true,
                            scheduler=scheduler,
                            loggers=(),
                            array_type=AT,
                            float_type=FT,
                            LJsoftcore="beutler",
                            Csoftcore="scaled"
                          )
```

These systems can then be used in AWH, TSS or REMD simulations for free energy estimation, how to run free energy calculations with each of these methods is described below.

### Example set up for RBFE
Here we demonstrate how to set up a hybrid system to interpolate between two ligands for TYK2 (ejm31 and ejm50) using the OpenFE settings. See below how OpenFE settings were matched using Molly settings.
![The TYK2 ligands](images/ejm31_ejm50.png)

First, we have to load the individual systems and create a mapping to specify which atom indexes are the core and unique A atoms in system A, and the unique B atoms in system B. Also, a map on how the atom indexes are mapped between core A -> core B.

```julia
using Molly

# --- Variables ---
FT = Float64
AT = Array
data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
ff_dir     = joinpath(data_dir, "force_fields")

# --- Force Field Setup ---
ff_A = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", 
                                              "amber14/protein.ff14SB.xml"])...,
                                        joinpath(data_dir, "ejm31.xml"),
                                        ; units=true, float_type=FT)

ff_B = MolecularForceField(joinpath.(ff_dir, ["tip3p_standard.xml", 
                                              "amber14/protein.ff14SB.xml"])...,
                                        joinpath(data_dir, "ejm50.xml")
                                        ; units=true, float_type=FT)

# --- Load Systems ---
sysA = System(
        joinpath(data_dir,"tyk2_ejm31.pdb"),
        ff_A;
        nonbonded_method=SetupPME(approximate_erfc=false),
        center_coords=false,
        dist_cutoff=FT(0.9)u"nm",
    )

sysB = System(
        joinpath(data_dir, "tyk2_ejm50.pdb"),
        ff_B;
        nonbonded_method=SetupPME(approximate_erfc=false),
        center_coords=false,
        dist_cutoff=FT(0.9)u"nm",
    )

# --- Mapping of Unique and Core atoms ---
mapping = Dict("unique_A"=>[4701], "unique_B" => [4701,4703])
core = []
for i in 4671:4702
    if !(i in mapping["unique_A"])
        push!(core, i)
    end
end

mapping["core"] = core
core_mapAB = Dict()
for (i,a) in enumerate(sysA.atoms_data)
    if i in mapping["core"]
        for (j,b) in enumerate(sysB.atoms_data)
            if a.atom_name == b.atom_name && a.atom_name == b.atom_name
                core_mapAB[i] = j
            end
        end
    end
end
core_mapAB[4702] = 4702
```

Then we can use SystemA and SystemB to set up a hybrid system with a single topology. By specifying `dual=false` in the scheduler, a single topology is used rather than the default dual topology. (Below shows how you would set up using the OpenFE settings)

```julia
# --- Hybrid System Setup ---
global_λ = FT(0.0)
sys = RelativeFESystem(sysA, sysB, global_λ, mapping, core_mapAB; 
                        temp=298.0u"K", 
                        units=true,
                        scheduler=OpenFEScheduler(dual=false),
                        loggers=(),
                        array_type=AT,
                        float_type=FT,
                        LJsoftcore="gapsys",
                        Csoftcore="scaled"
                      )
```

This system can then be used in AWH, TSS or REMD simulations for free energy estimation, how to run free energy calculations with each of these methods is described below.

## Free energies with AWH

### Method overview

[The Accelerated Weight Histogram](https://doi.org/10.1103/PhysRevE.85.056708) (AWH) is an adaptive extended-ensemble method for estimating free energies over a discrete set of thermodynamic states where the thermodynamic-state index is treated as a dynamic variable. At each AWH iteration, the coordinates are propagated with the Hamiltonian of the active state, the current configuration is evaluated in all states, and a new active state is sampled from the conditional state probabilities.

For a target distribution $\rho_k$, current free energy estimate $f_k$, and reduced potential $u_k(x)$, the conditional weight of state $k$ is

```math
\omega_k(x) =
\frac{\exp\left(f_k + \log \rho_k - u_k(x)\right)}
{\sum_j \exp\left(f_j + \log \rho_j - u_j(x)\right)}.
```

The same weights are accumulated into a segment histogram. After `update_freq` samples, Molly updates the free energy estimate from the segment histogram and an effective reference count. During the initial stage, the reference count is controlled by `n_bias`, so smaller values give more aggressive updates. After the state space has been covered, the effective count grows with simulation time and the updates become smaller.

The examples in this section show two common uses. The alanine dipeptide example uses AWH to move between umbrella centers on a two-dimensional $\phi/\psi$ grid, then [deconvolves the umbrella bias](https://doi.org/10.1063/1.4890371) to obtain an unbiased PMF. The benzene hydration example uses AWH directly on a one-dimensional alchemical ladder, where the AWH free energy vector is the alchemical profile.

### Alanine dipeptide PMF

The full script is `scripts/awh_dipeptide.jl`. Run it from the package root:

```bash
julia scripts/awh_dipeptide.jl
```

The script uses CUDA arrays and selects GPU `0` by default. Set `MOLLY_CUDA_DEVICE` before launching Julia to choose a different device.

The system is the alanine dipeptide structure from the Molly exercises. The script minimizes it, assigns Maxwell-Boltzmann velocities, and equilibrates it at 310 K and 1 bar before constructing the AWH states.

```julia
using Molly
using CUDA
using GLMakie
using Random

CUDA.device!(parse(Int, get(ENV, "MOLLY_CUDA_DEVICE", "0")))

FT = Float32
AT = CuArray

DT = FT(4)u"fs"
TEMP = FT(310)u"K"
PRES = one(FT)u"bar"

thermostat = VelocityRescaleThermostat(TEMP, FT(0.1)u"ps"; n_steps = 1)
barostat = CRescaleBarostat(PRES, FT(4)u"ps"; n_steps = 100)
vverlet = VelocityVerlet(DT, (thermostat, barostat), 100)
```

The PMF is defined over the central $\phi$ and $\psi$ torsions. In this example each angle is divided into 20 periodic bins, giving 400 umbrella centers. Each AWH state is a copy of the equilibrated system with one flat-bottom torsion bias on $\phi$ and one on $\psi$.

```julia
PHI_INDS = [5, 7, 9, 15]
PSI_INDS = [7, 9, 15, 17]
PHI_CV = CalcTorsion(PHI_INDS, :pbc, true)
PSI_CV = CalcTorsion(PSI_INDS, :pbc, true)

N_PHI_STATES = 20
N_PSI_STATES = 20

PHI_MIN = FT(-π)
PHI_MAX = FT(π)
PSI_MIN = FT(-π)
PSI_MAX = FT(π)

FLAT_BOTTOM_WIDTH = ustrip(u"rad", FT(360 / N_PHI_STATES)u"°")
BIAS_K = FT(100.0)u"kJ * mol^-1"

PHI_TARGETS = collect(range(PHI_MIN, PHI_MAX; length=N_PHI_STATES + 1))[1:end-1]
PSI_TARGETS = collect(range(PSI_MIN, PSI_MAX; length=N_PSI_STATES + 1))[1:end-1]

thermo_states = ThermoState[]

for psi in PSI_TARGETS
    for phi in PHI_TARGETS
        bias_phi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, phi)
        bias_psi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, psi)

        sys_bias = System(
            deepcopy(sys);
            general_inters = (
                sys.general_inters...,
                BiasPotential(PHI_CV, bias_phi),
                BiasPotential(PSI_CV, bias_psi),
            ),
        )

        push!(thermo_states, ThermoState(sys_bias, vverlet))
    end
end
```

The AWH state index labels biased thermodynamic states, not the unbiased PMF bins. For this reason, the raw AWH free energies are not used directly as the Ramachandran free energy surface. Instead, the script attaches a [`PMFDeconvolution`](@ref) object to the simulation.

Here the automatic deconvolution path is safe because every thermodynamic state has exactly two [`BiasPotential`](@ref)s in `general_inters`, and they are appended in the same order as the PMF grid dimensions: first the $\phi$ bias, then the $\psi$ bias. [`PMFDeconvolution`](@ref) can therefore infer both the collective variables and the reduced bias-energy coupling matrix from the states themselves.

```julia
awh_state = AWHState(thermo_states; first_state = 1, n_bias = 100, reuse_neighbors = true)

pmf_deconv = PMFDeconvolution(
    awh_state;
    grid = ((PHI_MIN, PSI_MIN), (PHI_MAX, PSI_MAX), (N_PHI_STATES, N_PSI_STATES)),
)
```

!!! warning "Deconvolving explicit collective variables and bias potentials"

    Do not use the automatic path if:

    * The state contains extra `BiasPotential`s that are not PMF windows.
    * The bias order does not match the PMF grid axes.
    * More than one bias contributes to one PMF dimension.
    * The coordinate used for the plotted PMF is transformed differently from the coordinate seen by the bias.

    In those cases, provide the collective-variable function and the reduced coupling explicitly. For example, if an extra flat-bottom restraint were also stored in `general_inters`, the PMF deconvolution would need to ignore it:

```julia
GRID_CARTESIAN = CartesianIndices((N_PHI_STATES, N_PSI_STATES))

function torsion_value(coords, sys, cv)
    return FT(ustrip(calculate_cv(cv, coords, sys.atoms, sys.boundary, sys.velocities)))
end

function custom_pmf_cv(coords)
    active_sys = awh_state.active_sys
    return (
        torsion_value(coords, active_sys, PHI_CV),
        torsion_value(coords, active_sys, PSI_CV),
    )
end

function custom_pmf_coupling(cv_tuple, state_i)
    grid_i = GRID_CARTESIAN[state_i]
    bias_phi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, PHI_TARGETS[grid_i[1]])
    bias_psi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, PSI_TARGETS[grid_i[2]])
    bias_energy = potential_energy(bias_phi, FT(cv_tuple[1])) +
                  potential_energy(bias_psi, FT(cv_tuple[2]))
    return awh_state.λ_β[state_i] * FT(ustrip(bias_energy))
end

pmf_deconv = PMFDeconvolution(
    awh_state;
    grid = ((PHI_MIN, PSI_MIN), (PHI_MAX, PSI_MAX), (N_PHI_STATES, N_PSI_STATES)),
    cv = custom_pmf_cv,
    coupling = custom_pmf_coupling,
)
```

The simulation advances 50 MD steps per AWH iteration. The finite `well_tempered_factor` lets the target distribution adapt during the run, favoring lower-free-energy regions while still keeping all windows accessible. Higher values soften this behaviour.

```julia
N_MD_STEPS = 50
AWH_TIME = FT(25)u"ns"
TOTAL_STEPS = Int(floor(AWH_TIME / DT))

awh_sim = AWHSimulation(
    awh_state;
    num_md_steps = N_MD_STEPS,
    update_freq = 1,
    well_tempered_factor = FT(500.0),
    log_freq = 10,
    pmf = pmf_deconv,
)

simulate!(awh_sim, TOTAL_STEPS)
```

`AWHSimulation` stores its absolute MD step, so couplers and loggers retain their cadence
across AWH iterations and repeated `simulate!` calls. Pass `initial_step` when resuming from
a checkpoint whose coordinates and velocities already correspond to a nonzero step.

After the run, `pmf(awh_sim.pmf)` returns the deconvolved PMF in units of $k_B T$ by default. Infinite bins are converted to `NaN` before plotting so GLMakie can render regions with low statistical support in a separate color. The script plots the deconvolved PMF as a heatmap and adds a colorbar for the free energy scale.

```julia
pmf_result = pmf(awh_sim.pmf)
pmf_kbt = pmf_result.F
pmf_plot = map(x -> isfinite(x) ? x : FT(NaN), pmf_kbt)

fig_fe = Figure(size = (720, 720))

ax_fe = Axis(fig_fe[1,1],
    title = L"\textbf{Free Energy}",
    xlabel = L"\textbf{\phi / rad}",
    ylabel = L"\textbf{\psi / rad}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

hm_fe = heatmap!(
    ax_fe,
    PHI_TARGETS,
    PSI_TARGETS,
    pmf_plot;
    colormap = :inferno,
    nan_color = (:gray, 0.35),
)

Colorbar(
    fig_fe[1, 2],
    hm_fe;
    label = L"\textbf{F / k_{B}T}",
    labelsize = 20,
    ticklabelsize = 16,
)

ax_fe.aspect = DataAspect()
display(fig_fe)

save("awh_dipeptide_pmf.png", fig_fe)
```

![AWH deconvolved alanine dipeptide PMF](images/awh_dipeptide_pmf.png)

The convergence plot uses the maximum absolute free energy update recorded by AWH at each logged iteration. A decreasing `log10.(deltaF)` trace indicates that the adaptive bias updates are becoming smaller.

```julia
deltaF = awh_state.stats.max_delta_f_history
iter = awh_state.stats.step_indices

fig_df = Figure(size = (720, 720))

ax_df = Axis(
    fig_df[1,1],
    title = L"\textbf{Max. $\Delta$F}",
    xlabel = L"\textbf{Iteration}",
    ylabel = L"\textbf{log_{10}($\Delta$F)}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

lines!(
    ax_df,
    iter,
    log10.(deltaF);
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "PHI/PSI AWH",
)

axislegend(position = :rt, labelsize = 24)
display(fig_df)

save("awh_dipeptide_convergence.png", fig_df)
```

![AWH dipeptide convergence](images/awh_dipeptide_convergence.png)

The script also saves a normalized histogram of the visited AWH thermodynamic states. This is a simple diagnostic for how the sampler moved over the 400 biased $\phi/\psi$ windows.

```julia
state_bins = 0.5:1:(N_PHI_STATES * N_PSI_STATES + 0.5)
save_state_histogram(
    awh_visited_states(awh_state),
    state_bins,
    "awh_dipeptide_states.png",
)
```

![AWH dipeptide visited states](images/awh_dipeptide_states.png)

### Benzene hydration free energy

The full script can be found in `scripts/awh_solvation.jl`:

```bash
julia scripts/awh_solvation.jl
```

On one NVIDIA GeForce RTX 5090 the script takes roughly 2 h 45 min, about 2 h 5 min for the solvated leg and 35 min for the vacuum leg, including setup and equilibration.

[`AbsoluteFESystem`](@ref) decouples the solute from $\lambda=0$ (fully coupled) to $\lambda=1$ (fully decoupled), so the ladder of 20 states runs from the decoupled state 1 to the coupled state 20. The scheduler removes the charges from $\lambda=0$ to $\lambda=0.5$ with the sterics untouched, and then the sterics from $\lambda=0.5$ to $\lambda=1$. The intramolecular electrostatics of benzene are annihilated with its charges, while `intraLJ = true` keeps its intramolecular Lennard-Jones interactions. For the GROMACS convention with two PME grids, use `GROMACSLambdaABFEScheduler` instead.

```julia
FT = Float32
AT = CuArray

Δt = FT(4)u"fs"
T0 = FT(298.15)u"K"
P0 = FT(1)u"bar"

N_LAMBDA_STATES = 20
N_MD_STEPS = 50
AWH_TIME = FT(30)u"ns"
SOLVENT_EQUIL_TIME = FT(500)u"ps"
VACUUM_EQUIL_TIME = FT(100)u"ps"
CUTOFF = FT(1)u"nm"

lambda_schedule = FT.(range(1.0, stop = 0.0, length = N_LAMBDA_STATES))
scheduler = DefaultLambdaScheduler(dual = true, intraLJ = true)
```

The setup function builds either the solvated or vacuum leg. The solvated leg uses PME and a Langevin integrator coupled to an NPT barostat. The vacuum leg uses short-range Coulomb with an infinite boundary and infinite non-bonded cutoff so the benzene molecule is not split by a finite cutoff.

```julia
function setup_alchemical_awh(pdb_file, solute_indices; is_vacuum = false, 
                                rng = Random.default_rng())
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : CUTOFF
    nonbonded_method = is_vacuum ? DistanceCutoff(dist_cutoff) : SetupPME()
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing

    sys_base = System(
        pdb_file,
        ff;
        array_type = AT,
        float_type = FT,
        boundary = boundary,
        dist_cutoff = dist_cutoff,
        dist_buffer = FT(0) * u"nm",
        neighbor_finder_type = neighbor_finder_type,
        nonbonded_method = nonbonded_method,
        constraints = :hbonds,
        constraint_algorithm = SetupLINCS(),
        rigid_water = true,
        hydrogen_mass = 3,
    )

    integrator = if is_vacuum
        Langevin(; dt = Δt, temperature = T0, friction = FT(1)u"ps^-1",
                 coupling = nothing, remove_CM_motion = 100)
    else
        barostat = CRescaleBarostat(P0, FT(4)u"ps"; n_steps = 200)
        Langevin(dt = Δt, temperature = T0, friction = FT(1)u"ps^-1",
                 coupling = (barostat,), remove_CM_motion = 100)
    end
```

!!! warning "Integrating equations of motion with alchemical transformations"
    
    When running simulations in which some of the atoms are achemicaly decoupled from the system
    some care must be taken with the integrator of choice. Decoupling atoms can make them incredibly
    fast, which will appear as a huge contribution to the kinetic energy tensor. Therefore any temperature 
    calculation that depends on said tensor will be prone to errors, since one could argue
    that a fully decoupled set of atoms should not contribute to the overall temperature of the system.

    Therefore, and since Molly currently does not yet support temperature groups, the recommended way of
    integrating this kind of system is by using a [`Langevin`](@ref) scheme, instead of a [`VelocityVerlet`](@ref)
    integrator accompanied by a [`VelocityRescaleThermostat`](@ref) temperature coupling scheme; specially
    when using large integration timesteps.

    Note that the [`Langevin`](@ref) integrator is known to produce incorrect kinetics, but since
    free energies are an ensemble average property this issue can be safely ignored.

After minimization and equilibration, the script checks that the barostat has not shrunk the box below twice the cutoff, which the minimum image convention requires. It then creates one [`ThermoState`](@ref) per $\lambda$ value with [`AbsoluteFESystem`](@ref). This replaces the non-bonded interactions of the solute with $\lambda$-dependent versions, here the [Beutler soft core](https://doi.org/10.1016/0009-2614(94)00397-1) for Lennard-Jones and Coulomb scaled directly by $\lambda$, and rebuilds PME, the Ewald exclusions and the dispersion correction for the $\lambda$-dependent charges.

```julia
    if !is_vacuum
        edge = minimum(Molly.box_sides(sys_base.boundary))
        edge >= 2 * dist_cutoff || error("equilibrated box edge $edge 
                                           is smaller than twice the cutoff")
    end

    thermo_states = ThermoState[]

    for λ in lambda_schedule
        sys_w = AbsoluteFESystem(
            sys_base,
            FT(λ),
            solute_indices;
            scheduler = scheduler,
            LJsoftcore = "beutler",
            Csoftcore = "scaled",
            array_type = AT,
            float_type = FT,
        )

        push!(thermo_states, ThermoState(sys_w, deepcopy(integrator)))
    end
```

For an alchemical ladder, no PMF deconvolution is needed: the state coordinate already is the alchemical coordinate. The script disables the well-tempered target adaptation with `well_tempered_factor = Inf`, so AWH samples a fixed uniform target over the ladder.

```julia
    awh_state = AWHState(thermo_states; reuse_neighbors = true)
    awh_sim = AWHSimulation(
        awh_state;
        num_md_steps = N_MD_STEPS,
        update_freq = 1,
        well_tempered_factor = FT(Inf),
        log_freq = 10,
    )

    return awh_state, awh_sim
end
```

The two legs are run independently. The free energy profile is stored in `awh_state.f`, gauge-fixed so the first state is zero.

```julia
solute_idx = 1:12

awh_state_solv, awh_sim_solv = setup_alchemical_awh(
    joinpath(data_dir, "benzene_solv_4.pdb"),
    solute_idx;
    is_vacuum = false,
    rng = MersenneTwister(RNG_SEED),
)

awh_state_vac, awh_sim_vac = setup_alchemical_awh(
    joinpath(data_dir, "benzene_vac.pdb"),
    solute_idx;
    is_vacuum = true,
    rng = MersenneTwister(RNG_SEED + 1),
)

awh_steps = Int(floor(AWH_TIME / Δt))

simulate!(awh_sim_solv, awh_steps)
simulate!(awh_sim_vac, awh_steps)

f_solv = awh_state_solv.f
f_vac  = awh_state_vac.f
```

The script reverses both the $\lambda$ schedule and the free energy vectors before plotting, so $\lambda=0$ is on the left and $\lambda=1$ is on the right.

```julia
lambda_plot = reverse(lambda_schedule)
f_solv_plot = reverse(f_solv)
f_vac_plot = reverse(f_vac)

fig_fe = Figure(size = (720, 720))

ax_fe = Axis(fig_fe[1,1],
    title = L"\textbf{Alchemical Free Energy}",
    xlabel = L"\textbf{\lambda}",
    ylabel = L"\textbf{F / k_{B}T}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

lines!(
    ax_fe,
    lambda_plot, f_solv_plot;
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Solvated",
)

lines!(
    ax_fe,
    lambda_plot, f_vac_plot;
    color = :firebrick,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Vacuum",
)

axislegend(position = :rt, labelsize = 24)
display(fig_fe)

save("awh_solvation_profile.png", fig_fe)
```

![AWH alchemical free energy profile](images/awh_solvation_profile.png)

It also plots the maximum free energy update at each logged AWH iteration.

```julia
deltaF_solv = awh_state_solv.stats.max_delta_f_history
deltaF_vac  = awh_state_vac.stats.max_delta_f_history

iter_solv = awh_state_solv.stats.step_indices
iter_vac = awh_state_vac.stats.step_indices

fig_df = Figure(size = (720, 720))

ax_df = Axis(
    fig_df[1,1],
    title = L"\textbf{Max. $\Delta$F}",
    xlabel = L"\textbf{Iteration}",
    ylabel = L"\textbf{log_{10}($\Delta$F)}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

lines!(
    ax_df,
    iter_solv, log10.(deltaF_solv);
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Solvated",
)

lines!(
    ax_df,
    iter_vac, log10.(deltaF_vac);
    color = :firebrick,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Vacuum",
)

axislegend(position = :rt, labelsize = 24)
display(fig_df)

save("awh_solvation_convergence.png", fig_df)
```

![AWH alchemical convergence](images/awh_solvation_convergence.png)

The visited-state histogram is plotted for the solvent and vacuum legs on the same axes. The state index follows the $\lambda$ schedule, from the decoupled state 1 ($\lambda=1$) to the coupled state 20 ($\lambda=0$).

```julia
state_bins = 0.5:1:(N_LAMBDA_STATES + 0.5)
save_state_histogram(
    [awh_visited_states(awh_state_solv), awh_visited_states(awh_state_vac)],
    ["Solvated", "Vacuum"],
    state_bins,
    "awh_solvation_states.png",
)
```

![AWH alchemical visited states](images/awh_solvation_states.png)

Finally, the annihilation free energies are taken between the decoupled first state and the coupled last state, combined, and printed in both $k_B T$ and kJ mol^-1.

```julia
dG_solv = f_solv[1] - f_solv[end]
dG_vac  = f_vac[1]  - f_vac[end]

dG = dG_vac - dG_solv
beta = awh_state_solv.state_space.betas[1]

println("=========================================")
println("Annihilation in solvent (kBT): ", dG_solv)
println("Annihilation in vacuum (kBT):  ", dG_vac)
println("Hydration Free Energy (kBT):   ", dG)
println("=========================================")

println("=========================================")
println("Annihilation in solvent (kJ mol^-1): ", dG_solv / beta)
println("Annihilation in vacuum (kJ mol^-1):  ", dG_vac / beta)
println("Hydration Free Energy (kJ mol^-1):   ", dG / beta)
println("=========================================")
```

A run with these settings gives, in kJ mol^-1:

```text
=========================================
Annihilation in solvent (kJ mol^-1): -6.225
Annihilation in vacuum (kJ mol^-1):  -9.988
Hydration Free Energy (kJ mol^-1):   -3.763
=========================================
```

## Free energies with TSS

### Method overview

[Times Square Sampling](https://doi.org/10.1080/10618600.2023.2291108) (TSS) is an adaptive simulated-tempering method over a discrete set of thermodynamic states, referred to as "rungs" in the paper describing the technique. As in AWH, the active state determines which Hamiltonian propagates the molecular coordinates. However, TSS differs in how the free energies are estimated and how the global state space is managed, putting special attention on the efficient assignement of computational resources to the ladder of thermodynamics states.

Molly's TSS implementation uses local estimators on graph windows; a graph defines the topology of the thermodynamic states, for example a one-dimensional alchemical ladder or a two-dimensional torsion grid. Each window contains a subset of neighboring states. During a TSS cycle, each replica samples states inside one active window, updates the local estimator for that window, and then the window-level estimates are combined into a reported global free energy profile.

The reference distribution over rungs is called $\gamma$. It sets the asymptotic density used by the free energy estimator. With `adaptive_gamma = :covdet`, Molly updates $\gamma$ from a covariance-determinant estimate of local thermodynamic state density. This gives a larger target density to regions where the local state geometry indicates that more resolution is useful.

TSS can also forget early history. With `TSSHistoryForgetting(alpha, n_epochs)`, estimator history before approximately $\alpha \cdot t$ is dropped at TSS iteration $t$. Retained history is grouped into logarithmically spaced epochs. Those epochs are used both to remove old data and to compute jackknife uncertainty estimates without storing every individual sample.

### Alanine dipeptide PMF

The full script is `scripts/tss_dipeptide.jl`:

```bash
julia scripts/tss_dipeptide.jl
```

This example estimates the unbiased alanine dipeptide $\phi/\psi$ free energy surface with TSS. It starts from the alanine dipeptide structure, minimizes and equilibrates the system at 310 K and 1 bar, and then builds a 20 by 20 thermodynamic-state grid over the central $\phi$ and $\psi$ torsions.

```julia
PHI_INDS = [5, 7, 9, 15]
PSI_INDS = [7, 9, 15, 17]
PHI_CV = CalcTorsion(PHI_INDS, :pbc, true)
PSI_CV = CalcTorsion(PSI_INDS, :pbc, true)

N_PHI_STATES = 20
N_PSI_STATES = 20

PHI_MIN = FT(-π)
PHI_MAX = FT(π)
PSI_MIN = FT(-π)
PSI_MAX = FT(π)

FLAT_BOTTOM_WIDTH = ustrip(u"rad", FT(360 / N_PHI_STATES)u"°")
BIAS_K = FT(100.0)u"kJ * mol^-1"

PHI_TARGETS = collect(range(PHI_MIN, PHI_MAX; length=N_PHI_STATES + 1))[1:end-1]
PSI_TARGETS = collect(range(PSI_MIN, PSI_MAX; length=N_PSI_STATES + 1))[1:end-1]
```

The TSS thermodynamic states are biased copies of the equilibrated system, one for each $\phi/\psi$ umbrella center.

```julia
thermo_states = ThermoState[]

for psi in PSI_TARGETS
    for phi in PHI_TARGETS
        bias_phi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, phi)
        bias_psi = PeriodicFlatBottomBias(BIAS_K, FLAT_BOTTOM_WIDTH, psi)

        sys_bias = System(
            deepcopy(sys);
            general_inters = (
                sys.general_inters...,
                BiasPotential(PHI_CV, bias_phi),
                BiasPotential(PSI_CV, bias_psi),
            ),
        )

        push!(thermo_states, ThermoState(sys_bias, vverlet))
    end
end
```

The graph is a periodic two-dimensional grid. The window size is one fifth of the grid along each dimension, so each local estimator covers a $4 \times 4$ patch of neighboring umbrella centers. The TSS state enables history forgetting and covariance-determinant adaptive $\gamma$.

```julia
PHI_WIN_SIZE = Int(N_PHI_STATES // 5)
PSI_WIN_SIZE = Int(N_PSI_STATES // 5)

tss_graph = Molly.tss_grid_graph(
    (N_PHI_STATES, N_PSI_STATES);
    window_size = (PHI_WIN_SIZE, PSI_WIN_SIZE),
    periodic = (true, true),
)

tss_state = TSSState(
    thermo_states;
    graph = tss_graph,
    first_state = 1,
    first_window = 1,
    history_forgetting = TSSHistoryForgetting(alpha = FT(0.19), n_epochs = 16),
    adaptive_gamma = :covdet,
)
```

The TSS simulation uses four replicas. For more than one replica, the script constructs explicit [`ActiveThermoState`](@ref) objects with different starting states, assigns velocities, and passes them to [`TSSSimulation`](@ref).

```julia
N_MD_STEPS = 50
SELF_ADJ_STEPS = 5
N_REPLICAS = 4

tss_dipeptide_replica_rngs(seed::Integer) =
    [MersenneTwister(seed + replica_i - 1) for replica_i in 1:N_REPLICAS]

TSS_TIME = FT(25)u"ns"
TOTAL_STEPS = Int(floor(TSS_TIME / DT))
N_CYCLES = Int(floor(TOTAL_STEPS / (SELF_ADJ_STEPS * N_MD_STEPS)))

replica_first_states = [
    state_index(
        mod1(1 + (replica_i - 1) * max(1, N_PHI_STATES ÷ N_REPLICAS), N_PHI_STATES),
        mod1(1 + (replica_i - 1) * max(1, N_PSI_STATES ÷ N_REPLICAS), N_PSI_STATES),
    )
    for replica_i in 1:N_REPLICAS
]
initial_replica_rngs = tss_dipeptide_replica_rngs(RNG_SEED + 10)
replica_active_states = if N_REPLICAS == 1
    nothing
else
    states = [ActiveThermoState(tss_state.state_space, first_state)
              for first_state in replica_first_states]
    for (active_state, replica_rng) in zip(states, initial_replica_rngs)
        random_velocities!(active_state.active_sys, TEMP; rng = replica_rng)
    end
    states
end

pmf_deconv = PMFDeconvolution(
    tss_state;
    grid = ((PHI_MIN, PSI_MIN), (PHI_MAX, PSI_MAX), (N_PHI_STATES, N_PSI_STATES)),
)

tss_sim = TSSSimulation(
    tss_state;
    n_md_steps = N_MD_STEPS,
    n_cycles = N_CYCLES,
    self_adjustment_steps = SELF_ADJ_STEPS,
    n_replicas = N_REPLICAS,
    replica_active_states = replica_active_states,
    pmf = pmf_deconv,
    log_freq = 10,
)

simulate!(
    tss_sim;
    rng = rng,
    replica_rngs = tss_dipeptide_replica_rngs(RNG_SEED + 20),
    replica_parallel = :auto,
)
```

`TSSSimulation` likewise stores one absolute MD step shared by its physical replicas.
Each MD block receives that step, and repeated calls resume from `tss_sim.current_step`.
Use the `initial_step` constructor keyword when restoring an existing simulation state.

For PMF deconvolution, the script uses the automatic coupling path. This is safe for the same reason as in the AWH example above: each thermodynamic state contains exactly the two umbrella [`BiasPotential`](@ref)s that define the PMF dimensions, and they appear in the same order as the grid axes. The [`PMFDeconvolution`](@ref) object must be created before the run and passed to [`TSSSimulation`](@ref).

```julia
pmf_result = pmf(pmf_deconv)
pmf_kbt = pmf_result.F
pmf_plot = map(x -> isfinite(x) ? x : FT(NaN), pmf_kbt)

fig_fe = Figure(size = (720, 720))

ax_fe = Axis(fig_fe[1,1],
    title = L"\textbf{Free Energy}",
    xlabel = L"\textbf{\phi / rad}",
    ylabel = L"\textbf{\psi / rad}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

hm_fe = heatmap!(
    ax_fe,
    PHI_TARGETS,
    PSI_TARGETS,
    pmf_plot;
    colormap = :inferno,
    nan_color = (:gray, 0.35),
)

Colorbar(
    fig_fe[1, 2],
    hm_fe;
    label = L"\textbf{F / k_{B}T}",
    labelsize = 20,
    ticklabelsize = 16,
)

ax_fe.aspect = DataAspect()
display(fig_fe)
save("tss_dipeptide_pmf.png", fig_fe)
```

![TSS deconvolved alanine dipeptide PMF](images/tss_dipeptide_pmf.png)

The convergence plot uses `tss_state.stats.max_abs_delta_f`, the maximum absolute reported free energy update at each logged TSS iteration.

```julia
deltaF = tss_state.stats.max_abs_delta_f
iter = tss_state.stats.iterations

fig_df = Figure(size = (720, 720))

ax_df = Axis(
    fig_df[1,1],
    title = L"\textbf{Max. $\Delta$F}",
    xlabel = L"\textbf{Iteration}",
    ylabel = L"\textbf{log_{10}($\Delta$F)}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

lines!(
    ax_df,
    iter,
    log10.(deltaF);
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "PHI/PSI TSS",
)

axislegend(position = :rt, labelsize = 24)
display(fig_df)
save("tss_dipeptide_convergence.png", fig_df)
```

![TSS dipeptide convergence](images/tss_dipeptide_convergence.png)

The script also saves a normalized histogram of the visited TSS thermodynamic states. With multiple replicas, the histogram combines the state histories from all replicas.

```julia
state_bins = 0.5:1:(N_PHI_STATES * N_PSI_STATES + 0.5)
save_state_histogram(
    tss_visited_states(tss_state),
    state_bins,
    "tss_dipeptide_states.png",
)
```

![TSS dipeptide visited states](images/tss_dipeptide_states.png)

### Benzene hydration free energy

The full script is `scripts/tss_solvation.jl`:

```bash
julia scripts/tss_solvation.jl
```

On one NVIDIA GeForce RTX 5090 the script takes roughly 3 h 10 min, about 2 h 15 min for the solvated leg and 55 min for the vacuum leg, with the four replicas simulated one after the other on the GPU.

The ladder has 20 states from the decoupled state 1 ($\lambda=1$) to the coupled state 20 ($\lambda=0$), built with [`AbsoluteFESystem`](@ref) and the same scheduler as in the AWH example.

```julia
FT = Float32
AT = CuArray

Δt = FT(4)u"fs"
T0 = FT(298.15)u"K"
P0 = FT(1)u"bar"

N_LAMBDA_STATES = 20
TSS_WINDOW_SIZE = 4
N_MD_STEPS = 50
SELF_ADJUSTMENT_STEPS = 5
N_REPLICAS = 4
TSS_TIME = FT(15)u"ns"
SOLVENT_EQUIL_TIME = FT(500)u"ps"
VACUUM_EQUIL_TIME = FT(100)u"ps"
CUTOFF = FT(1)u"nm"

lambda_schedule = FT.(range(1.0, stop = 0.0, length = N_LAMBDA_STATES))
scheduler = DefaultLambdaScheduler(dual = true, intraLJ = true)

tss_solvation_replica_rngs(seed::Integer) =
    [MersenneTwister(seed + replica_i - 1) for replica_i in 1:N_REPLICAS]
```

The setup function constructs the solvated and vacuum legs. The non-bonded treatment is chosen before alchemical states are generated: PME for the solvated leg, and short-range Coulomb with an infinite cutoff for the vacuum leg.

```julia
function setup_alchemical_tss(pdb_file, solute_indices; is_vacuum = false, 
                                rng = Random.default_rng())
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : CUTOFF
    nonbonded_method = is_vacuum ? DistanceCutoff(dist_cutoff) : SetupPME()
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing

    sys_base = System(
        pdb_file,
        ff;
        array_type = AT,
        float_type = FT,
        boundary = boundary,
        dist_cutoff = dist_cutoff,
        dist_buffer = FT(0) * u"nm",
        neighbor_finder_type = neighbor_finder_type,
        nonbonded_method = nonbonded_method,
        constraints = :hbonds,
        constraint_algorithm = SetupLINCS(),
        rigid_water = true,
        hydrogen_mass = 3,
    )

    integrator = if is_vacuum
        Langevin(; dt = Δt, temperature = T0, friction = FT(1)u"ps^-1",
                 coupling = nothing, remove_CM_motion = 100)
    else
        barostat = CRescaleBarostat(P0, FT(4)u"ps"; n_steps = 200)
        Langevin(dt = Δt, temperature = T0, friction = FT(1)u"ps^-1",
                 coupling = (barostat,), remove_CM_motion = 100)
    end
```

After minimization and equilibration, the script checks that the barostat has not shrunk the box below twice the cutoff, which the minimum image convention requires. It then creates one [`ThermoState`](@ref) per $\lambda$ value with [`AbsoluteFESystem`](@ref). This replaces the non-bonded interactions of the solute with $\lambda$-dependent versions, here the [Beutler soft core](https://doi.org/10.1016/0009-2614(94)00397-1) for Lennard-Jones and Coulomb scaled directly by $\lambda$, and rebuilds PME, the Ewald exclusions and the dispersion correction for the $\lambda$-dependent charges.

```julia
    if !is_vacuum
        edge = minimum(Molly.box_sides(sys_base.boundary))
        edge >= 2 * dist_cutoff || error("equilibrated box edge $edge 
                                           is smaller than twice the cutoff")
    end

    thermo_states = ThermoState[]

    for λ in lambda_schedule
        sys_w = AbsoluteFESystem(
            sys_base,
            FT(λ),
            solute_indices;
            scheduler = scheduler,
            LJsoftcore = "beutler",
            Csoftcore = "scaled",
            array_type = AT,
            float_type = FT,
        )

        push!(thermo_states, ThermoState(sys_w, deepcopy(integrator)))
    end
```

The TSS graph is a one-dimensional non-periodic ladder. With 20 states and `TSS_WINDOW_SIZE = 4`, each local estimator spans four neighboring $\lambda$ states. The simulation uses four replicas and 15000 TSS cycles for a 15 ns run with this timestep and cycle definition.

```julia
    tss_graph = Molly.tss_grid_graph(
        (length(lambda_schedule),);
        window_size = (TSS_WINDOW_SIZE,),
        periodic = (false,),
    )

    tss_state = TSSState(
        thermo_states;
        graph = tss_graph,
        history_forgetting = TSSHistoryForgetting(alpha = FT(0.2), n_epochs = 16),
        adaptive_gamma = :covdet,
    )

    total_steps = Int(floor(TSS_TIME / Δt))
    n_cycles = Int(floor(total_steps / (SELF_ADJUSTMENT_STEPS * N_MD_STEPS)))
    first_states = N_REPLICAS == 1 ? [1] : round.(Int, range(1, length(lambda_schedule); 
                                                    length = N_REPLICAS))

    tss_sim = TSSSimulation(
        tss_state;
        n_md_steps = N_MD_STEPS,
        n_cycles = n_cycles,
        self_adjustment_steps = SELF_ADJUSTMENT_STEPS,
        n_replicas = N_REPLICAS,
        first_states = first_states,
        log_freq = 10,
    )

    return tss_state, tss_sim
end
```

The two legs are constructed and run independently. `replica_parallel = :auto` lets Molly choose a safe replica execution mode for the available hardware and array type.

```julia
solute_idx = 1:12

tss_state_solv, tss_sim_solv = setup_alchemical_tss(
    joinpath(data_dir, "benzene_solv_4.pdb"),
    solute_idx;
    is_vacuum = false,
    rng = MersenneTwister(RNG_SEED),
)

tss_state_vac, tss_sim_vac = setup_alchemical_tss(
    joinpath(data_dir, "benzene_vac.pdb"),
    solute_idx;
    is_vacuum = true,
    rng = MersenneTwister(RNG_SEED + 1),
)

simulate!(
    tss_sim_solv;
    rng = MersenneTwister(RNG_SEED + 2),
    replica_rngs = tss_solvation_replica_rngs(RNG_SEED + 20),
    replica_parallel = :auto,
)

simulate!(
    tss_sim_vac;
    rng = MersenneTwister(RNG_SEED + 3),
    replica_rngs = tss_solvation_replica_rngs(RNG_SEED + 40),
    replica_parallel = :auto,
)
```

After the run, `tss_free_energy_uncertainties` returns the reported free energies and jackknife standard errors over the retained epochs. The free energies are dimensionless, in units of $k_B T$.

```julia
jk_solv = tss_free_energy_uncertainties(tss_state_solv)
jk_vac  = tss_free_energy_uncertainties(tss_state_vac)

f_solv = jk_solv.free_energies
f_vac  = jk_vac.free_energies

se_solv = jk_solv.standard_errors
se_vac  = jk_vac.standard_errors

dG_solv = f_solv[1] - f_solv[end]
dG_vac  = f_vac[1]  - f_vac[end]

dG = dG_vac - dG_solv
dG_se = hypot(se_solv[end], se_vac[end])
```

The profile plot shows the solvent and vacuum annihilation free energies. The script reverses the $\lambda$ schedule and the estimated profiles before plotting, then draws bands spanning three jackknife standard errors.

```julia
lambda_plot = reverse(lambda_schedule)
f_solv_plot = reverse(f_solv)
f_vac_plot = reverse(f_vac)
se_solv_plot = reverse(se_solv)
se_vac_plot = reverse(se_vac)

fig_fe = Figure(size = (720, 720))

ax_fe = Axis(fig_fe[1,1],
    title = L"\textbf{Alchemical Free Energy}",
    xlabel = L"\textbf{\lambda}",
    ylabel = L"\textbf{F / k_{B}T}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

lines!(
    ax_fe,
    lambda_plot, f_solv_plot;
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Solvated",
)

band!(
    ax_fe,
    lambda_plot,
    f_solv_plot - 3 .* se_solv_plot,
    f_solv_plot + 3 .* se_solv_plot;
    color = :royalblue,
    alpha = 0.45,
)

lines!(
    ax_fe,
    lambda_plot, f_vac_plot;
    color = :firebrick,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Vacuum",
)

band!(
    ax_fe,
    lambda_plot,
    f_vac_plot - 3 .* se_vac_plot,
    f_vac_plot + 3 .* se_vac_plot;
    color = :firebrick,
    alpha = 0.45,
)

axislegend(position = :rt, labelsize = 24)
display(fig_fe)

save("tss_solvation_profile.png", fig_fe)
```

![TSS alchemical free energy profile](images/tss_solvation_profile.png)

The convergence plot shows `log10` of the maximum absolute reported TSS free energy update for each logged iteration.

```julia
deltaF_solv = tss_state_solv.stats.max_abs_delta_f
deltaF_vac  = tss_state_vac.stats.max_abs_delta_f

iter_solv = tss_state_solv.stats.iterations
iter_vac = tss_state_vac.stats.iterations

fig_df = Figure(size = (720, 720))

ax_df = Axis(
    fig_df[1,1],
    title = L"\textbf{Max. $\Delta$F}",
    xlabel = L"\textbf{Iteration}",
    ylabel = L"\textbf{log_{10}($\Delta$F)}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

lines!(
    ax_df,
    iter_solv, log10.(deltaF_solv);
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Solvated",
)

lines!(
    ax_df,
    iter_vac, log10.(deltaF_vac);
    color = :firebrick,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Vacuum",
)

axislegend(position = :rt, labelsize = 24)
ylims!(ax_df, -5, 0.5)
display(fig_df)

save("tss_solvation_convergence.png", fig_df)
```

![TSS alchemical convergence](images/tss_solvation_convergence.png)

The visited-state histogram combines the state histories from all replicas in each leg, then plots the solvent and vacuum distributions together.

```julia
state_bins = 0.5:1:(N_LAMBDA_STATES + 0.5)
save_state_histogram(
    [tss_visited_states(tss_state_solv), tss_visited_states(tss_state_vac)],
    ["Solvated", "Vacuum"],
    state_bins,
    "tss_solvation_states.png",
)
```

![TSS alchemical visited states](images/tss_solvation_states.png)

The final output reports the two annihilation free energies, the hydration free energy and its jackknife standard error.

```julia
beta = tss_state_solv.state_space.betas[1]

println("=========================================")
println("Annihilation in solvent (kBT): ", dG_solv)
println("Annihilation in vacuum (kBT):  ", dG_vac)
println("Hydration Free Energy (kBT):   ", dG)
println("Jackknife SE (kBT):            ", dG_se)
println("=========================================")

println("=========================================")
println("Annihilation in solvent (kJ mol^-1): ", dG_solv / beta)
println("Annihilation in vacuum (kJ mol^-1):  ", dG_vac / beta)
println("Hydration Free Energy (kJ mol^-1):   ", dG / beta)
println("Jackknife SE (kJ mol^-1):            ", dG_se / beta)
println("=========================================")
```

A run with these settings gives, in kJ mol^-1:

```text
=========================================
Annihilation in solvent (kJ mol^-1): -6.109
Annihilation in vacuum (kJ mol^-1):  -10.011
Hydration Free Energy (kJ mol^-1):   -3.901
Jackknife SE (kJ mol^-1):            0.176
=========================================
```

## Free energies with REMD

### Method overview

[Replica exchange molecular dynamics](https://doi.org/10.1016/S0009-2614(99)01123-9) (REMD) runs one replica in each state of a discrete ladder of thermodynamic states and periodically attempts to swap the configurations of neighbouring states. In [Hamiltonian replica exchange](https://doi.org/10.1063/1.1472510) (HREMD) the states share the temperature and differ in their Hamiltonian, here the alchemical coupling $\lambda_1, \lambda_2, \dots, \lambda_K$. Exchanges let a configuration that is trapped in one state escape through the others, which enhances sampling across the barriers along the alchemical path.

An exchange between states $i$ and $j$, occupied by configurations $x_i$ and $x_j$, is accepted with the Metropolis-Hastings criterion

$$P_{\text{acc}}(i \leftrightarrow j) = \min\left(1, \exp\left(-\Delta \Delta u\right)\right)$$

where the reduced energy difference is

$$\Delta \Delta u = \left[ u_j(x_i) + u_i(x_j) \right] - \left[ u_i(x_i) + u_j(x_j) \right]$$

In Molly, a [`ReplicaSystem`](@ref) is built from one [`ThermoState`](@ref) per state, and [`ReplicaExchangeMD`](@ref) sets the time step `dt` and the time between exchange attempts, `exchange_time`. [`simulate_remd!`](@ref) propagates every replica with the Hamiltonian of the state it is currently in and, after every `exchange_time`, attempts exchanges between neighbouring states, alternating between the pairs $(1, 2), (3, 4), \dots$ and $(2, 3), (4, 5), \dots$. `state_indices[k]` is the replica that is currently in state $k$, and accepted exchanges are recorded in `exchange_logger`. Repeated calls continue from `current_step`. With CPU arrays the states are propagated in parallel on threads. With GPU arrays each GPU is driven by its own worker process, and the systems are sent to the workers on the first call only; later calls send the coordinates, velocities and boxes.

### Free energy estimator

After the simulation, the configurations sampled in each state are evaluated in every state, and the Multistate Bennett Acceptance Ratio (MBAR) estimates the free energy of each state:

$$\hat{f}_k = -\log \sum_{i=1}^K \sum_{n=1}^{N_i} \frac{\exp\left[-u_k(x_{i,n})\right]}{\sum_{j=1}^K N_j \exp\left[\hat{f}_j - u_j(x_{i,n})\right]}$$

More detail on the MBAR method can be found above.

### Benzene hydration free energy

The full script is `scripts/hremd_solvation.jl`:

```bash
julia scripts/hremd_solvation.jl
```

On two NVIDIA GeForce RTX 5090 GPUs the script takes roughly 50 minutes, about 27 min for the solvated leg and 18 min for the vacuum leg, including setup, equilibration and MBAR.

The system, the ladder of 20 states from the decoupled state 1 ($\lambda=1$) to the coupled state 20 ($\lambda=0$) and the two-leg cycle are the same as in the AWH and TSS examples above; only the free energy method differs. [`simulate_remd!`](@ref) uses one worker process per GPU, each simulating a block of $\lambda$ states, so one worker is added for every GPU. Replica exchange draws its exchanges from the default random number generator, which is seeded here.

```julia
using Distributed

N_GPUS = 2
addprocs(N_GPUS; exeflags = "--project=$(Base.active_project())")
@everywhere using Molly, CUDA

gpu_devices = collect(CUDA.devices())[1:N_GPUS]

FT = Float32
AT = CuArray

Δt = FT(4)u"fs"
T0 = FT(298.15)u"K"
P0 = FT(1)u"bar"
RNG_SEED = 20240520
Random.seed!(RNG_SEED)

N_LAMBDA_STATES = 20
HREMD_TIME = FT(1.5)u"ns"
EXCHANGE_TIME = FT(2)u"ps"
SAMPLE_TIME = FT(4)u"ps"
EQUIL_FRAC = 0.2
SOLVENT_EQUIL_TIME = FT(500)u"ps"
VACUUM_EQUIL_TIME = FT(100)u"ps"
CUTOFF = FT(1)u"nm"

lambda_schedule = FT.(range(1.0, stop = 0.0, length = N_LAMBDA_STATES))
scheduler = DefaultLambdaScheduler(dual = true, intraLJ = true)
```

`HREMD_TIME` is the simulated time for each $\lambda$ state. One sample per state is stored every `SAMPLE_TIME`, and the first `EQUIL_FRAC` of the samples of each state are discarded before MBAR.

The setup function constructs the solvated and vacuum legs as in the AWH example: PME and an NPT barostat for the solvated leg, and short-range Coulomb with an infinite cutoff for the vacuum leg. Replica exchange starts every replica from the same configuration, so the function also runs the equilibration.

```julia
function build_thermo_states(pdb_file, solute_indices; is_vacuum = false,
                             rng = Random.default_rng())
    boundary = is_vacuum ? CubicBoundary(FT(Inf) * u"nm") : nothing
    dist_cutoff = is_vacuum ? FT(Inf) * u"nm" : CUTOFF
    nonbonded_method = is_vacuum ? DistanceCutoff(dist_cutoff) : SetupPME()
    neighbor_finder_type = is_vacuum ? DistanceNeighborFinder : nothing

    sys_base = System(
        pdb_file,
        ff;
        array_type = AT,
        float_type = FT,
        boundary = boundary,
        dist_cutoff = dist_cutoff,
        dist_buffer = FT(0) * u"nm",
        neighbor_finder_type = neighbor_finder_type,
        nonbonded_method = nonbonded_method,
        constraints = :hbonds,
        constraint_algorithm = SetupLINCS(),
        rigid_water = true,
        hydrogen_mass = 3,
    )

    if is_vacuum
        integrator = Langevin(; dt = Δt, temperature = T0, friction = FT(1)u"ps^-1",
                              coupling = nothing, remove_CM_motion = 100)
        int_eq     = Langevin(; dt = Δt / 2, temperature = T0, friction = FT(1)u"ps^-1",
                              coupling = nothing, remove_CM_motion = 100)
    else
        barostat   = CRescaleBarostat(P0, FT(4)u"ps"; n_steps = 200)
        integrator = Langevin(dt = Δt, temperature = T0, friction = FT(1)u"ps^-1",
                              coupling = (barostat,), remove_CM_motion = 100)
        int_eq     = Langevin(dt = Δt / 2, temperature = T0, friction = FT(1)u"ps^-1",
                              coupling = (barostat,), remove_CM_motion = 100)
    end
```

After minimization and equilibration, the function checks the box against the cutoff and creates one [`ThermoState`](@ref) per $\lambda$ value with [`AbsoluteFESystem`](@ref), as in the AWH example. The equilibrated system is returned as well, since all replicas start from it.

```julia
    minim = SteepestDescentMinimizer(step_size = FT(0.01)u"nm", max_steps = 1000)
    simulate!(sys_base, minim)
    random_velocities!(sys_base, T0; rng = rng)

    equil_time = is_vacuum ? VACUUM_EQUIL_TIME : SOLVENT_EQUIL_TIME
    equil_steps = Int(floor(equil_time / Δt))
    simulate!(sys_base, int_eq, 10_000; rng = rng)
    simulate!(sys_base, integrator, equil_steps; rng = rng)

    if !is_vacuum
        edge = minimum(Molly.box_sides(sys_base.boundary))
        edge >= 2 * dist_cutoff || error("equilibrated box edge $edge
                                          is smaller than twice the cutoff")
    end

    thermo_states = ThermoState[]

    for λ in lambda_schedule
        sys_w = AbsoluteFESystem(
            sys_base,
            FT(λ),
            solute_indices;
            scheduler = scheduler,
            LJsoftcore = "beutler",
            Csoftcore = "scaled",
            array_type = AT,
            float_type = FT,
        )

        push!(thermo_states, ThermoState(sys_w, deepcopy(integrator)))
    end

    return thermo_states, sys_base
end
```

Each leg starts every replica from the equilibrated configuration. [`simulate_remd!`](@ref) is called once per sampling interval and returns a new [`ReplicaSystem`](@ref). After each call, the configuration that is currently in each $\lambda$ state is stored for MBAR. Every call covers a whole and even number of exchange cycles, here two, so that both sets of neighbouring pairs are attempted equally often.

```julia
function run_hremd_leg(thermo_states, sys_base)
    K = length(thermo_states)
    repsys = ReplicaSystem(
        thermo_states,
        [copy(sys_base.coords) for _ in 1:K];
        replica_velocities = [copy(sys_base.velocities) for _ in 1:K],
        replica_boundaries = [sys_base.boundary for _ in 1:K],
        reuse_neighbors = true,
    )
    sim = ReplicaExchangeMD(dt = Δt, exchange_time = EXCHANGE_TIME)

    chunk_steps = Int(round(SAMPLE_TIME / Δt))
    n_chunks = Int(floor(HREMD_TIME / Δt)) ÷ chunk_steps
    coords_k = [Any[] for _ in 1:K]
    boundaries_k = [Any[] for _ in 1:K]

    for _ in 1:n_chunks
        repsys = simulate_remd!(repsys, sim, chunk_steps;
                                gpu_devices = gpu_devices, show_progress = false)

        for k in 1:K
            r = repsys.state_indices[k]
            push!(coords_k[k], Array(repsys.replica_coords[r]))
            push!(boundaries_k[k], repsys.replica_boundaries[r])
        end
    end

    return coords_k, boundaries_k, repsys.exchange_logger
end
```

Every stored configuration is then evaluated in every state with [`assemble_mbar_inputs`](@ref), and the MBAR equations are solved with [`iterate_mbar`](@ref). All states start from the same fully coupled configuration, so the first 20% of the samples of each state are discarded. The free energies are dimensionless, in units of $k_B T$.

```julia
function hremd_free_energies(thermo_states, coords_k, boundaries_k)
    K = length(thermo_states)
    coords_dev = [[Molly.to_device(c, AT) for c in v] for v in coords_k]
    mbar_gen = assemble_mbar_inputs(coords_dev, boundaries_k, thermo_states)

    # Index of every sample within its own state, to discard the start of each state
    sample_idx = zeros(Int, length(mbar_gen.win_of))
    seen = zeros(Int, K)
    for n in eachindex(mbar_gen.win_of)
        k = mbar_gen.win_of[n]
        seen[k] += 1
        sample_idx[n] = seen[k]
    end
    kept = [n for n in eachindex(sample_idx)
            if sample_idx[n] > floor(Int, EQUIL_FRAC * mbar_gen.N[mbar_gen.win_of[n]])]

    win_of = mbar_gen.win_of[kept]
    N_counts = [count(==(k), win_of) for k in 1:K]
    F_k, logN = iterate_mbar(mbar_gen.u[kept, :], win_of, N_counts)

    return F_k
end
```

The two legs are constructed and run one after the other, and the free energies are then estimated for both.

```julia
solute_idx = 1:12

thermo_solv, base_solv = build_thermo_states(
    joinpath(data_dir, "benzene_solv_4.pdb"),
    solute_idx;
    is_vacuum = false,
    rng = MersenneTwister(RNG_SEED),
)
coords_solv, boundaries_solv, log_solv = run_hremd_leg(thermo_solv, base_solv)

thermo_vac, base_vac = build_thermo_states(
    joinpath(data_dir, "benzene_vac.pdb"),
    solute_idx;
    is_vacuum = true,
    rng = MersenneTwister(RNG_SEED + 1),
)
coords_vac, boundaries_vac, log_vac = run_hremd_leg(thermo_vac, base_vac)

f_solv = hremd_free_energies(thermo_solv, coords_solv, boundaries_solv)
f_vac  = hremd_free_energies(thermo_vac, coords_vac, boundaries_vac)
```

As in the AWH example, the script reverses the $\lambda$ schedule and the free energy vectors before plotting, so $\lambda=0$ is on the left and $\lambda=1$ is on the right.

```julia
lambda_plot = reverse(lambda_schedule)
f_solv_plot = reverse(f_solv)
f_vac_plot = reverse(f_vac)

fig_fe = Figure(size = (720, 720))

ax_fe = Axis(fig_fe[1,1],
    title = L"\textbf{Alchemical Free Energy}",
    xlabel = L"\textbf{\lambda}",
    ylabel = L"\textbf{F / k_{B}T}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

lines!(
    ax_fe,
    lambda_plot, f_solv_plot;
    color = :royalblue,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Solvated",
)

lines!(
    ax_fe,
    lambda_plot, f_vac_plot;
    color = :firebrick,
    linewidth = 3,
    linecap = :round,
    joinstyle = :round,
    label = "Vacuum",
)

axislegend(position = :rt, labelsize = 24)
display(fig_fe)

save("hremd_solvation_profile.png", fig_fe)
```

![HREMD alchemical free energy profile](images/hremd_solvation_profile.png)

The exchange acceptance of every neighbouring pair of states shows how well the ladder mixes, and plays the role of the visited-state histograms of AWH and TSS. A pair that never exchanges splits the ladder in two and invalidates the MBAR estimate. Each pair is attempted every other exchange cycle.

```julia
function pair_acceptance(exchange_logger, K)
    n_cycles = Int(floor(HREMD_TIME / Δt)) ÷ Int(round(SAMPLE_TIME / Δt)) *
               (Int(round(SAMPLE_TIME / Δt)) ÷ Int(round(EXCHANGE_TIME / Δt)))
    accepted = zeros(Int, K - 1)
    for (n, m) in exchange_logger.indices
        accepted[min(n, m)] += 1
    end
    return accepted ./ (n_cycles ÷ 2)
end

acc_solv = pair_acceptance(log_solv, N_LAMBDA_STATES)
acc_vac  = pair_acceptance(log_vac, N_LAMBDA_STATES)

fig_acc = Figure(size = (720, 720))

ax_acc = Axis(
    fig_acc[1,1],
    title = L"\textbf{Exchange Acceptance}",
    xlabel = L"\textbf{State Pair}",
    ylabel = L"\textbf{Acceptance}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
)

scatterlines!(
    ax_acc,
    1:(N_LAMBDA_STATES - 1), acc_solv;
    color = :royalblue,
    linewidth = 3,
    markersize = 12,
    label = "Solvated",
)

scatterlines!(
    ax_acc,
    1:(N_LAMBDA_STATES - 1), acc_vac;
    color = :firebrick,
    linewidth = 3,
    markersize = 12,
    label = "Vacuum",
)

ylims!(ax_acc, 0, 1.05)
axislegend(position = :rb, labelsize = 24)
display(fig_acc)

save("hremd_solvation_acceptance.png", fig_acc)
```

![HREMD exchange acceptance](images/hremd_solvation_acceptance.png)

The replica exchange history follows every replica through the ladder of the solvated leg. The exchange logger stores `state_indices` after every exchange cycle, from which the state of each replica is recovered; a replica moves to a neighbouring state whenever an exchange is accepted. Replicas that travel the whole ladder carry configurations between the coupled and decoupled states, while a replica confined to a few states points to a bottleneck.

```julia
function replica_state_history(exchange_logger, K)
    snapshots = vcat([collect(1:K)], exchange_logger.replica_indices)
    return [[findfirst(==(r), s) for s in snapshots] for r in 1:K]
end

history_solv = replica_state_history(log_solv, N_LAMBDA_STATES)
times_ns = (0:(length(history_solv[1]) - 1)) .* ustrip(u"ns", EXCHANGE_TIME)
# All replicas are drawn, four of them highlighted to make individual walks visible
highlight = round.(Int, range(1, N_LAMBDA_STATES; length = 4))
highlight_colors = (:royalblue, :firebrick, :seagreen, :darkorange)

fig_hist = Figure(size = (720, 720))

ax_hist = Axis(
    fig_hist[1,1],
    title = L"\textbf{Replica Exchange History (Solvated)}",
    xlabel = L"\textbf{Time (ns)}",
    ylabel = L"\textbf{State}",
    xlabelsize = 20, ylabelsize = 20,
    titlesize = 24,
    xlabelfont = :bold, ylabelfont = :bold,
    xticklabelsize = 18, yticklabelsize = 18,
    yticks = 1:2:N_LAMBDA_STATES,
)

for r in 1:N_LAMBDA_STATES
    r in highlight && continue
    stairs!(
        ax_hist,
        times_ns, history_solv[r];
        step = :post,
        color = (:grey60, 0.5),
        linewidth = 0.6,
    )
end

for (i, r) in enumerate(highlight)
    stairs!(
        ax_hist,
        times_ns, history_solv[r];
        step = :post,
        color = highlight_colors[i],
        linewidth = 2,
        label = "Replica $r",
    )
end

axislegend(position = :rb, labelsize = 18, nbanks = 2)

display(fig_hist)

save("hremd_solvation_exchanges.png", fig_hist)
```

![HREMD replica exchange history](images/hremd_solvation_exchanges.png)

Finally, the annihilation free energies are taken between the decoupled first state and the coupled last state, combined, and printed in both $k_B T$ and kJ mol^-1, together with the lowest pair acceptance in solvent.

```julia
dG_solv = f_solv[1] - f_solv[end]
dG_vac  = f_vac[1]  - f_vac[end]

dG = dG_vac - dG_solv
beta = thermo_solv[1].beta

println("=========================================")
println("Annihilation in solvent (kBT): ", dG_solv)
println("Annihilation in vacuum (kBT):  ", dG_vac)
println("Hydration Free Energy (kBT):   ", dG)
println("=========================================")

println("=========================================")
println("Annihilation in solvent (kJ mol^-1): ", dG_solv / beta)
println("Annihilation in vacuum (kJ mol^-1):  ", dG_vac / beta)
println("Hydration Free Energy (kJ mol^-1):   ", dG / beta)
println("Lowest pair acceptance in solvent:   ", minimum(acc_solv))
println("=========================================")
```

A run with these settings gives, in kJ mol^-1:

```text
=========================================
Annihilation in solvent (kJ mol^-1): -5.826
Annihilation in vacuum (kJ mol^-1):  -10.012
Hydration Free Energy (kJ mol^-1):   -4.186
Lowest pair acceptance in solvent:   0.163
=========================================
```

A single run of this length is worth about $\pm 0.5$ kJ mol^-1: the first and second halves of the samples of this run differ by 0.9 kJ mol^-1, while the vacuum leg reproduces to 0.001 kJ mol^-1 between runs, so the scatter comes from the solvated leg alone. Its ladder mixes slowly where the cavity forms, the dip around states 3 and 4 in the acceptance plot above. Adding $\lambda$ states in that region, so that neighbouring states overlap more, and running for longer both reduce the scatter.
