# Differentiable simulation with Molly

!!! note
    There are still some rough edges when taking gradients through simulations. Please open an issue if you run into an error and remember to check your gradients against finite differencing.

In the last few years, the deep learning revolution has broadened to include the paradigm of [differentiable programming](https://en.wikipedia.org/wiki/Differentiable_programming).
The concept of using automatic differentiation (AD) to obtain exact gradients through physical simulations has many interesting applications, including parameterising force fields and training neural networks to describe atom potentials.

There are some projects that explore differentiable molecular simulations such as [Jax, M.D.](https://github.com/google/jax-md), [TorchMD](https://github.com/torchmd/torchmd) and [mdgrad](https://github.com/torchmd/mdgrad).
However Julia provides a strong suite of AD tools, with [Zygote.jl](https://github.com/FluxML/Zygote.jl) allowing source-to-source transformations for much of the language.
With Molly you can use the power of Zygote to obtain gradients through molecular simulations.
Reverse and forward mode AD can be used on the CPU and the GPU.
General and specific interactions work, along with neighbor lists, and the same abstractions for running simulations are used as in the main package.
Differentiable simulation does not currently work with units, user-defined types and some components of Molly.

## General interactions

First, we show how taking gradients through a simulation can be used to optimise an atom property in a [Lennard-Jones](https://en.wikipedia.org/wiki/Lennard-Jones_potential) gas.
In this type of simulation each atom has a σ value that determines how close it likes to get to other atoms.
We are going to find the σ value that results in a desired distance of each atom to its closest neighbor.
First we need a function to obtain the mean distance of each atom to its closest neighbor:
```julia
using Molly

function meanminseparation(final_coords, box_size)
    n_atoms = length(final_coords)
    sum_dists = 0.0
    for i in 1:n_atoms
        min_dist = 100.0
        for j in 1:n_atoms
            i == j && continue
            dist = sqrt(sum(abs2, vector(final_coords[i], final_coords[j], box_size)))
            min_dist = min(dist, min_dist)
        end
        sum_dists += min_dist
    end
    return sum_dists / n_atoms
end
```
Now we can set up and run the simulation in a similar way to that described in the [Molly documentation](@ref).
The difference is that we wrap the simulation in a `loss` function.
This returns a single value that we want to obtain gradients with respect to, in this case the value of the above function at the end of the simulation.
The `Zygote.ignore()` block allows us to ignore code for the purposes of obtaining gradients; you could add the [`visualize`](@ref) function there for example.
```julia
using Zygote
using Format

dist_true = 0.5
scale_σ_to_dist = 2 ^ (1 / 6)
σtrue = dist_true / scale_σ_to_dist

n_atoms = 50
n_steps = 500
atom_mass = 10.0
box_size = SVector(3.0, 3.0, 3.0)
temp = 0.01
neighbor_finder = DistanceVecNeighborFinder(
    nb_matrix=trues(n_atoms, n_atoms),
    n_steps=10,
    dist_cutoff=1.5,
)
general_inters = (LennardJones(nl_only=true),)
coords = place_atoms(n_atoms, box_size, 0.7)
velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]
simulator = VelocityVerlet(
    dt=0.02,
    coupling=RescaleThermostat(temp),
)

function loss(σ)
    atoms = [Atom(0, 0.0, atom_mass, σ, 0.2) for i in 1:n_atoms]
    loggers = Dict("coords" => CoordinateLogger(Float64, 10))

    s = System(
        atoms=atoms,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        box_size=box_size,
        neighbor_finder=neighbor_finder,
        loggers=loggers,
        force_unit=NoUnits,
        energy_unit=NoUnits,
        gpu_diff_safe=true,
    )

    mms_start = meanminseparation(s.coords, box_size)
    simulate!(s, simulator, n_steps)
    mms_end = meanminseparation(s.coords, box_size)
    loss_val = abs(mms_end - dist_true)

    Zygote.ignore() do
        printfmt("σ {:6.3f}  |  Mean min sep expected {:6.3f}  |  Mean min sep end {:6.3f}  |  Loss {:6.3f}  |  ",
                  σ, σ * (2 ^ (1 / 6)), mms_end, loss_val)
    end

    return loss_val
end
```
Now we can obtain the gradient of `loss` with respect to the atom property `σ`.
```julia
grad = gradient(loss, σtrue)[1]
```
We can use this gradient in a training loop to optimise `σ`, starting from an arbitrary value.
```julia
function train()
    σlearn = 0.60 / scale_σ_to_dist
    n_epochs = 20

    for epoch_n in 1:n_epochs
        printfmt("Epoch {:>2}  |  ", epoch_n)
        coords = place_atoms(n_atoms, box_size, 0.7)
        velocities = [velocity(atom_mass, temp) for i in 1:n_atoms]
        grad = gradient(loss, σlearn)[1]
        printfmt("Grad {:6.3f}\n", grad)
        σlearn -= grad * 1e-2
    end
end

train()
```
```
Epoch  1  |  σ  0.535  |  Mean min sep expected  0.600  |  Mean min sep end  0.589  |  Loss  0.089  |  Grad  0.912
Epoch  2  |  σ  0.525  |  Mean min sep expected  0.590  |  Mean min sep end  0.579  |  Loss  0.079  |  Grad  0.644
Epoch  3  |  σ  0.519  |  Mean min sep expected  0.583  |  Mean min sep end  0.571  |  Loss  0.071  |  Grad  1.081
Epoch  4  |  σ  0.508  |  Mean min sep expected  0.570  |  Mean min sep end  0.560  |  Loss  0.060  |  Grad  1.543
Epoch  5  |  σ  0.493  |  Mean min sep expected  0.553  |  Mean min sep end  0.546  |  Loss  0.046  |  Grad  0.939
Epoch  6  |  σ  0.483  |  Mean min sep expected  0.543  |  Mean min sep end  0.533  |  Loss  0.033  |  Grad  1.350
Epoch  7  |  σ  0.470  |  Mean min sep expected  0.527  |  Mean min sep end  0.519  |  Loss  0.019  |  Grad  0.818
Epoch  8  |  σ  0.462  |  Mean min sep expected  0.518  |  Mean min sep end  0.510  |  Loss  0.010  |  Grad  1.744
Epoch  9  |  σ  0.444  |  Mean min sep expected  0.499  |  Mean min sep end  0.492  |  Loss  0.008  |  Grad -0.967
Epoch 10  |  σ  0.454  |  Mean min sep expected  0.509  |  Mean min sep end  0.502  |  Loss  0.002  |  Grad  0.756
Epoch 11  |  σ  0.446  |  Mean min sep expected  0.501  |  Mean min sep end  0.494  |  Loss  0.006  |  Grad -1.228
Epoch 12  |  σ  0.459  |  Mean min sep expected  0.515  |  Mean min sep end  0.506  |  Loss  0.006  |  Grad  1.299
Epoch 13  |  σ  0.446  |  Mean min sep expected  0.500  |  Mean min sep end  0.493  |  Loss  0.007  |  Grad -0.884
Epoch 14  |  σ  0.454  |  Mean min sep expected  0.510  |  Mean min sep end  0.502  |  Loss  0.002  |  Grad  1.014
Epoch 15  |  σ  0.444  |  Mean min sep expected  0.499  |  Mean min sep end  0.492  |  Loss  0.008  |  Grad -1.010
Epoch 16  |  σ  0.454  |  Mean min sep expected  0.510  |  Mean min sep end  0.502  |  Loss  0.002  |  Grad  0.986
Epoch 17  |  σ  0.445  |  Mean min sep expected  0.499  |  Mean min sep end  0.492  |  Loss  0.008  |  Grad -1.138
Epoch 18  |  σ  0.456  |  Mean min sep expected  0.512  |  Mean min sep end  0.504  |  Loss  0.004  |  Grad  0.709
Epoch 19  |  σ  0.449  |  Mean min sep expected  0.504  |  Mean min sep end  0.496  |  Loss  0.004  |  Grad -0.803
Epoch 20  |  σ  0.457  |  Mean min sep expected  0.513  |  Mean min sep end  0.505  |  Loss  0.005  |  Grad  0.327
```
The final value we get is 0.457, close to the theoretical value of 0.445 if all atoms have a neighbor at the minimum pairwise energy distance.
The RDF looks as follows, with the purple line corresponding to the desired distance to the closest neighbor.
![LJ RDF](images/rdf_lj.png)

## Specific interactions

Next we look at obtaining gradients through simulations with specific interactions, e.g. bonds or angles between specified atoms.
We will simulate two triatomic molecules and search for a minimum energy bond angle that gives a desired distance between the atoms at the end of the simulation.
```julia
using Molly
using Zygote
using Format
using LinearAlgebra

dist_true = 1.0

n_steps = 150
atom_mass = 10.0
box_size = SVector(3.0, 3.0, 3.0)
temp = 0.001
coords = [
    SVector(0.8, 0.75, 1.5), SVector(1.5, 0.70, 1.5), SVector(2.3, 0.75, 1.5),
    SVector(0.8, 2.25, 1.5), SVector(1.5, 2.20, 1.5), SVector(2.3, 2.25, 1.5),
]
n_atoms = length(coords)
velocities = zero(coords)
simulator = VelocityVerlet(
    dt=0.05,
    coupling=BerendsenThermostat(temp, 0.5),
)

function loss(θ)
    atoms = [Atom(0, 0.0, atom_mass, 0.0, 0.0) for i in 1:n_atoms]
    loggers = Dict("coords" => CoordinateLogger(Float64, 2))
    specific_inter_lists = (
        InteractionList2Atoms(
            [1, 2, 4, 5],
            [2, 3, 5, 6],
            [HarmonicBond(b0=0.7, kb=100.0) for _ in 1:4],
        ),
        InteractionList3Atoms(
            [1, 4],
            [2, 5],
            [3, 6],
            [HarmonicAngle(th0=θ, cth=10.0), HarmonicAngle(th0=θ, cth=10.0)],
        ),
    )

    s = System(
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        coords=deepcopy(coords),
        velocities=deepcopy(velocities),
        box_size=box_size,
        loggers=loggers,
        force_unit=NoUnits,
        energy_unit=NoUnits,
        gpu_diff_safe=true,
    )

    simulate!(s, simulator, n_steps)

    d1 = norm(vector(s.coords[1], s.coords[3], box_size))
    d2 = norm(vector(s.coords[4], s.coords[6], box_size))
    dist_end = 0.5 * (d1 + d2)
    loss_val = abs(dist_end - dist_true)

    Zygote.ignore() do
        printfmt("θ {:5.1f}°  |  Final dist {:4.2f}  |  Loss {:5.3f}  |  ",
                 rad2deg(θ), dist_end, loss_val)
    end

    return loss_val
end

function train()
    θlearn = deg2rad(110.0)
    n_epochs = 20

    for epoch_n in 1:n_epochs
        printfmt("Epoch {:>2}  |  ", epoch_n)
        grad = gradient(loss, θlearn)[1]
        printfmt("Grad {:6.3f}\n", round(grad; digits=2))
        θlearn -= grad * 0.1
    end
end

train()
```
```
Epoch  1  |  θ 110.0°  |  Final dist 1.16  |  Loss 0.156  |  Grad  0.390
Epoch  2  |  θ 107.8°  |  Final dist 1.14  |  Loss 0.140  |  Grad  0.410
Epoch  3  |  θ 105.4°  |  Final dist 1.12  |  Loss 0.122  |  Grad  0.430
Epoch  4  |  θ 103.0°  |  Final dist 1.10  |  Loss 0.102  |  Grad  0.460
Epoch  5  |  θ 100.4°  |  Final dist 1.08  |  Loss 0.080  |  Grad  0.510
Epoch  6  |  θ  97.4°  |  Final dist 1.05  |  Loss 0.047  |  Grad  0.960
Epoch  7  |  θ  91.9°  |  Final dist 1.00  |  Loss 0.002  |  Grad -0.490
Epoch  8  |  θ  94.8°  |  Final dist 1.02  |  Loss 0.023  |  Grad  0.460
Epoch  9  |  θ  92.1°  |  Final dist 1.00  |  Loss 0.000  |  Grad -0.490
Epoch 10  |  θ  94.9°  |  Final dist 1.02  |  Loss 0.024  |  Grad  0.460
Epoch 11  |  θ  92.3°  |  Final dist 1.00  |  Loss 0.001  |  Grad  0.490
Epoch 12  |  θ  89.5°  |  Final dist 0.98  |  Loss 0.024  |  Grad -0.510
Epoch 13  |  θ  92.5°  |  Final dist 1.00  |  Loss 0.003  |  Grad  0.490
Epoch 14  |  θ  89.7°  |  Final dist 0.98  |  Loss 0.023  |  Grad -0.510
Epoch 15  |  θ  92.6°  |  Final dist 1.00  |  Loss 0.004  |  Grad  0.480
Epoch 16  |  θ  89.8°  |  Final dist 0.98  |  Loss 0.021  |  Grad -0.510
Epoch 17  |  θ  92.7°  |  Final dist 1.01  |  Loss 0.005  |  Grad  0.480
Epoch 18  |  θ  90.0°  |  Final dist 0.98  |  Loss 0.020  |  Grad -0.510
Epoch 19  |  θ  92.9°  |  Final dist 1.01  |  Loss 0.007  |  Grad  0.480
Epoch 20  |  θ  90.1°  |  Final dist 0.98  |  Loss 0.018  |  Grad -0.510
```
The final value we get is 90.1°, close to the theoretical value of 91.2° which is obtainable from trigonometry.
The final simulation looks like this:
![Angle simulation](images/sim_angle.gif)
In the presence of other forces this value would not be so trivially obtainable.
We can record the gradients for different values of `θ`:
```julia
θs = collect(0:3:180)[2:end]
grads = [gradient(loss, deg2rad(θ))[1] for θ in θs]
```
The plot of these shows that the gradient has the expected sign either side of the correct value.
![Angle gradient](images/grad_angle.png)

## Neural network potentials

Since gradients can be computed with Zygote, [Flux](https://fluxml.ai) models can also be incorporated into simulations.
Here we show a neural network in the force function, though they can also be used in other parts of the simulation.
This example also shows how gradients for multiple parameters can be obtained, in this case the parameters of the neural network.
The jump from single to multiple parameters is important because single parameters can be easily optimised using other approaches, whereas differentiable simulation is well-placed to optimise many parameters simultaneously.

We set up three pseudo-atoms and train a network to imitate the Julia logo by moving the bottom two atoms:
```julia
using Molly
using Zygote
using Flux
using Format
using LinearAlgebra

dist_true = 1.0f0

model = Chain(
    Dense(1, 5, relu),
    Dense(5, 1, tanh),
)
ps = params(model)

struct NNBond <: SpecificInteraction end

function Molly.force(b::NNBond, coords_i, coords_j, box_size)
    ab = vector(coords_i, coords_j, box_size)
    dist = norm(ab)
    f = model([dist])[1] * normalize(ab)
    return SpecificForce2Atoms(f, -f)
end

n_steps = 400
mass = 10.0f0
box_size = SVector(5.0f0, 5.0f0, 5.0f0)
temp = 0.001f0
coords = [SVector(2.3f0, 2.07f0, 0.0f0), SVector(2.5f0, 2.93f0, 0.0f0), SVector(2.7f0, 2.07f0, 0.0f0)]
n_atoms = length(coords)
velocities = zero(coords)
simulator = VelocityVerlet(dt=0.02f0, coupling=BerendsenThermostat(temp, 0.5f0))

function loss()
    atoms = [Atom(0, 0.0f0, mass, 0.0f0, 0.0f0) for i in 1:n_atoms]
    loggers = Dict("coords" => CoordinateLogger(Float32, 10))
    specific_inter_lists = (
        InteractionList2Atoms([1], [3], [NNBond()]),
    )

    s = System(
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        coords=deepcopy(coords),
        velocities=deepcopy(velocities),
        box_size=box_size,
        loggers=loggers,
        force_unit=NoUnits,
        energy_unit=NoUnits,
        gpu_diff_safe=true,
    )

    simulate!(s, simulator, n_steps)

    dist_end = (norm(vector(s.coords[1], s.coords[2], box_size)) +
                norm(vector(s.coords[2], s.coords[3], box_size)) +
                norm(vector(s.coords[3], s.coords[1], box_size))) / 3
    loss_val = abs(dist_end - dist_true)

    Zygote.ignore() do
        printfmt("Dist end {:6.3f}  |  Loss {:6.3f}\n", dist_end, loss_val)
        visualize(s.loggers["coords"], box_size, "sim.mp4")
    end

    return loss_val
end
```
Before training the result looks like this:
![Logo before](images/logo_before.gif)
```julia
function train()
    n_epochs = 20
    opt = ADAM(0.02, (0.9, 0.999))

    for epoch_n in 1:n_epochs
        printfmt("Epoch {:>2}  |  ", epoch_n)
        Flux.train!(loss, ps, ((),), opt)
    end
end

train()
```
```
Epoch  1  |  Dist end  0.611  |  Loss  0.389
Epoch  2  |  Dist end  0.842  |  Loss  0.158
Epoch  3  |  Dist end  0.870  |  Loss  0.130
Epoch  4  |  Dist end  0.905  |  Loss  0.095
Epoch  5  |  Dist end  0.946  |  Loss  0.054
Epoch  6  |  Dist end  0.992  |  Loss  0.008
Epoch  7  |  Dist end  1.041  |  Loss  0.041
Epoch  8  |  Dist end  1.066  |  Loss  0.066
Epoch  9  |  Dist end  1.074  |  Loss  0.074
Epoch 10  |  Dist end  1.071  |  Loss  0.071
Epoch 11  |  Dist end  1.060  |  Loss  0.060
Epoch 12  |  Dist end  1.042  |  Loss  0.042
Epoch 13  |  Dist end  1.020  |  Loss  0.020
Epoch 14  |  Dist end  0.994  |  Loss  0.006
Epoch 15  |  Dist end  0.979  |  Loss  0.021
Epoch 16  |  Dist end  0.972  |  Loss  0.028
Epoch 17  |  Dist end  0.971  |  Loss  0.029
Epoch 18  |  Dist end  0.976  |  Loss  0.024
Epoch 19  |  Dist end  0.985  |  Loss  0.015
Epoch 20  |  Dist end  0.999  |  Loss  0.001
```
After training it looks much better:
![Logo after](images/logo_after.gif)
You could replace the simple network here with a much more complicated model and it would theoretically be able to train, even if it might prove practically difficult (see discussion below).

## Molecular loss functions

Ultimately, you need some objective function in order to calculate the gradient for each parameter.
Here are some ideas for loss functions suitable for differentiable molecular simulations:
- The distance between atoms at the end of the simulation compared to some reference state. This loss is used in the examples given here, is physically reasonable, and has obvious bounds.
- The distance between atoms throughout the simulation.
- The radial distribution function of atoms.
- RMSD between atoms and a reference state - this would be suitable for macromolecules.
- dRMSD, the distance between a distance map and a reference distance map.
- The flexibility of a set of atoms over the simulation.
- Supramolecular geometry, for example assembly of molecules into straight fibres.
- The correlation of different velocities over the simulation.
- The energy of the system.
- The temperature of the system.
- Some measure of phase change or a critical point.
- A combination of the above, for example to obtain a force field relevant to both ordered and disordered proteins.
Some of these are currently not possible in Molly as the loggers are ignored for gradient purposes, but this will hopefully change in future.

## Tips and tricks

- Exploding gradients prove a problem when using the velocity Verlet integrator in the NVE ensemble. This is why the velocity rescaling and Berendsen thermostats were used in the above examples. It is likely that the development of suitable simulation strategies and thermostats will be necessary to unlock the potential of differentiable simulation.
- Do you *really* need a neural network to describe your potential? Think about learning a smaller number of physically-meaningful parameters before you put in a large neural network and expect it to learn. Whilst it is true that neural networks are universal function approximators, it does not follow that you will be able to train one by differentiating through  a long simulation. A 1000-step simulation with a 10-layer network at each step is analogous to training a 10,000 layer network (with shared weights).
- Forward mode AD holds much promise for differentiable simulation, provided the number of parameters is small, because the memory requirement is constant in the number of simulation steps. However, if the code runs slower than non-differentiable alternatives then the best approach may be to use finite differencing with the simulation as a black box.
