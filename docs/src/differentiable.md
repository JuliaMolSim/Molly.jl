# Differentiable simulation with Molly

!!! note
    This section can be read as a standalone introduction to differentiable simulation with Molly, but makes more sense in the context of the [Molly documentation](@ref).

!!! note
    The examples on this page have somewhat diverged from the main branch and will hopefully be updated soon.

In the last few years, the deep learning revolution has broadened to include the paradigm of [differentiable programming](https://en.wikipedia.org/wiki/Differentiable_programming).
The concept of using automatic differentiation to obtain exact gradients through physical simulations has many interesting applications, including parameterising forcefields and training neural networks to describe atom potentials.

There are some projects that explore differentiable molecular simulations, such as [Jax, M.D.](https://github.com/google/jax-md) and [DiffTaichi](https://github.com/yuanming-hu/difftaichi), or you can write your own algorithms in something like [PyTorch](https://pytorch.org).
However Julia provides a strong suite of autodiff tools, with [Zygote.jl](https://github.com/FluxML/Zygote.jl) allowing source-to-source transformations for much of the language.
The [differentiable](https://github.com/JuliaMolSim/Molly.jl/tree/differentiable) branch of Molly lets you use the power of Zygote to obtain gradients through molecular simulations.
It is not yet merged because it is experimental, untested, slow, liable to change, and only works for some parts of the main package.

With those caveats in mind, it provides a neat way to run differentiable simulations using the same abstractions as the main package.
In particular, you can define general and specific interactions, letting you move away from N-body simulations and describe molecular systems.
This is possible with something like a PyTorch tensor, but the force functions quickly get complicated to write.
It also lets you use neighbor lists and periodic boundary conditions, or add neural networks to your simulations.

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
n_steps = 250
mass = 10.0
box_size = 3.0
timestep = 0.05
temp = 3.0
simulator = VelocityVerlet()
neighbor_finder = DistanceNeighborFinder(ones(n_atoms, n_atoms), 10, 1.5)
thermostat = FrictionThermostat(0.95)
general_inters = (LennardJones(true),)
specific_inter_lists = ()
coords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]
velocities = [velocity(mass, temp) for i in 1:n_atoms]

function loss(σ)
    atoms = [Atom("", "", 0, "", 0.0, mass, σ, 0.2) for i in 1:n_atoms]
    loggers = Dict("coords" => CoordinateLogger(1))

    s = Simulation(
        simulator=simulator,
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        temperature=temp,
        box_size=box_size,
        neighbor_finder=neighbor_finder,
        thermostat=thermostat,
        loggers=loggers,
        timestep=timestep,
        n_steps=n_steps
    )

    mms_start = meanminseparation(coords, box_size)
    final_coords = simulate!(s)
    mms_end = meanminseparation(final_coords, box_size)
    loss_val = abs(mms_end - dist_true)

    Zygote.ignore() do
        printfmt("σ {:6.3f}  |  Mean min sep expected {:6.3f}  |  Mean min sep end {:6.3f}  |  Loss {:6.3f}  |  ",
                    σ, σ * (2 ^ (1 / 6)), mms_end, loss_val)
    end

    return loss_val
end
```
We use a simple friction thermostat that downscales velocities each step to avoid exploding gradients (see discussion below).
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
        coords = [box_size .* rand(SVector{3}) for i in 1:n_atoms]
        velocities = [velocity(mass, temp) for i in 1:n_atoms]
        grad = gradient(loss, σlearn)[1]
        printfmt("Grad {:6.3f}\n", grad)
        σlearn -= grad * 1e-2
    end
end

train()
```
```
Epoch  1  |  σ  0.535  |  Mean min sep expected  0.600  |  Mean min sep end  0.594  |  Loss  0.094  |  Grad -2.702
Epoch  2  |  σ  0.562  |  Mean min sep expected  0.630  |  Mean min sep end  0.617  |  Loss  0.117  |  Grad  0.871
Epoch  3  |  σ  0.553  |  Mean min sep expected  0.621  |  Mean min sep end  0.605  |  Loss  0.105  |  Grad  1.110
Epoch  4  |  σ  0.542  |  Mean min sep expected  0.608  |  Mean min sep end  0.591  |  Loss  0.091  |  Grad  0.970
Epoch  5  |  σ  0.532  |  Mean min sep expected  0.597  |  Mean min sep end  0.578  |  Loss  0.078  |  Grad  1.058
Epoch  6  |  σ  0.521  |  Mean min sep expected  0.585  |  Mean min sep end  0.567  |  Loss  0.067  |  Grad  1.157
Epoch  7  |  σ  0.510  |  Mean min sep expected  0.572  |  Mean min sep end  0.555  |  Loss  0.055  |  Grad  1.035
Epoch  8  |  σ  0.500  |  Mean min sep expected  0.561  |  Mean min sep end  0.543  |  Loss  0.043  |  Grad  1.052
Epoch  9  |  σ  0.489  |  Mean min sep expected  0.549  |  Mean min sep end  0.529  |  Loss  0.029  |  Grad  1.082
Epoch 10  |  σ  0.478  |  Mean min sep expected  0.537  |  Mean min sep end  0.517  |  Loss  0.017  |  Grad  1.109
Epoch 11  |  σ  0.467  |  Mean min sep expected  0.524  |  Mean min sep end  0.504  |  Loss  0.004  |  Grad  1.036
Epoch 12  |  σ  0.457  |  Mean min sep expected  0.513  |  Mean min sep end  0.493  |  Loss  0.007  |  Grad -1.018
Epoch 13  |  σ  0.467  |  Mean min sep expected  0.524  |  Mean min sep end  0.504  |  Loss  0.004  |  Grad  1.031
Epoch 14  |  σ  0.457  |  Mean min sep expected  0.513  |  Mean min sep end  0.493  |  Loss  0.007  |  Grad -1.054
Epoch 15  |  σ  0.467  |  Mean min sep expected  0.524  |  Mean min sep end  0.504  |  Loss  0.004  |  Grad  1.033
Epoch 16  |  σ  0.457  |  Mean min sep expected  0.513  |  Mean min sep end  0.493  |  Loss  0.007  |  Grad -1.058
Epoch 17  |  σ  0.467  |  Mean min sep expected  0.525  |  Mean min sep end  0.504  |  Loss  0.004  |  Grad  1.034
Epoch 18  |  σ  0.457  |  Mean min sep expected  0.513  |  Mean min sep end  0.493  |  Loss  0.007  |  Grad -1.061
Epoch 19  |  σ  0.468  |  Mean min sep expected  0.525  |  Mean min sep end  0.505  |  Loss  0.005  |  Grad  1.033
Epoch 20  |  σ  0.457  |  Mean min sep expected  0.513  |  Mean min sep end  0.494  |  Loss  0.006  |  Grad -1.044
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
mass = 10.0
box_size = 3.0
timestep = 0.05
temp = 0.0
integrator = VelocityVerlet()
neighbor_finder = NoNeighborFinder()
thermostat = FrictionThermostat(0.6)
general_inters = (LennardJones(false),)
coords = [
        SVector(0.8, 0.75, 1.5), SVector(1.5, 0.70, 1.5), SVector(2.3, 0.75, 1.5),
        SVector(0.8, 2.25, 1.5), SVector(1.5, 2.20, 1.5), SVector(2.3, 2.25, 1.5)]
n_atoms = length(coords)
n_dims = length(first(coords))
velocities = [velocity(mass, temp; dims=n_dims) for i in coords]

function loss(θ)
    atoms = [Atom("", "", 0, "", 0.0, mass, 0.0, 0.0) for i in 1:n_atoms]
    loggers = Dict("coords" => CoordinateLogger(2; dims=n_dims))
    specific_inter_lists = ([
            HarmonicBond(1, 2, 0.7, 100.0), HarmonicBond(2, 3, 0.7, 100.0),
            HarmonicBond(4, 5, 0.7, 100.0), HarmonicBond(5, 6, 0.7, 100.0)], [
            HarmonicAngle(1, 2, 3, θ, 10.0), HarmonicAngle(4, 5, 6, θ, 10.0)])

    s = Simulation(
        simulator=integrator,
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        temperature=temp,
        box_size=box_size,
        neighbor_finder=neighbor_finder,
        thermostat=thermostat,
        loggers=loggers,
        timestep=timestep,
        n_steps=n_steps
    )

    final_coords = simulate!(s)
    dist_end = 0.5 * (norm(vector(final_coords[1], final_coords[3], box_size)) +
                        norm(vector(final_coords[4], final_coords[6], box_size)))
    loss_val = abs(dist_end - dist_true) + 0.0 * sum(sum(final_coords))

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
        coords = [
            SVector(0.8, 0.75, 1.5), SVector(1.5, 0.74, 1.5), SVector(2.3, 0.75, 1.5),
            SVector(0.8, 2.25, 1.5), SVector(1.5, 2.24, 1.5), SVector(2.3, 2.25, 1.5)]
        velocities = [velocity(mass, temp; dims=n_dims) for i in 1:n_atoms]
        grad = gradient(loss, θlearn)[1]
        printfmt("Grad {:6.3f}\n", round(grad, digits=2))
        θlearn -= grad * 5e-3
    end
end

train()
```
```
Epoch  1  |  θ 110.0°  |  Final dist 1.15  |  Loss 0.147  |  Grad  3.080
Epoch  2  |  θ 109.1°  |  Final dist 1.14  |  Loss 0.141  |  Grad  3.400
Epoch  3  |  θ 108.1°  |  Final dist 1.13  |  Loss 0.134  |  Grad  3.570
Epoch  4  |  θ 107.1°  |  Final dist 1.13  |  Loss 0.126  |  Grad  3.750
Epoch  5  |  θ 106.0°  |  Final dist 1.12  |  Loss 0.119  |  Grad  3.950
Epoch  6  |  θ 104.9°  |  Final dist 1.11  |  Loss 0.110  |  Grad  4.150
Epoch  7  |  θ 103.7°  |  Final dist 1.10  |  Loss 0.101  |  Grad  4.370
Epoch  8  |  θ 102.5°  |  Final dist 1.09  |  Loss 0.092  |  Grad  4.600
Epoch  9  |  θ 101.2°  |  Final dist 1.08  |  Loss 0.082  |  Grad  4.840
Epoch 10  |  θ  99.8°  |  Final dist 1.07  |  Loss 0.071  |  Grad  5.090
Epoch 11  |  θ  98.3°  |  Final dist 1.06  |  Loss 0.059  |  Grad  5.340
Epoch 12  |  θ  96.8°  |  Final dist 1.05  |  Loss 0.047  |  Grad  5.590
Epoch 13  |  θ  95.2°  |  Final dist 1.03  |  Loss 0.034  |  Grad  5.840
Epoch 14  |  θ  93.5°  |  Final dist 1.02  |  Loss 0.020  |  Grad  6.070
Epoch 15  |  θ  91.8°  |  Final dist 1.01  |  Loss 0.005  |  Grad  6.290
Epoch 16  |  θ  90.0°  |  Final dist 0.99  |  Loss 0.010  |  Grad -6.480
Epoch 17  |  θ  91.8°  |  Final dist 1.01  |  Loss 0.005  |  Grad  6.350
Epoch 18  |  θ  90.0°  |  Final dist 0.99  |  Loss 0.010  |  Grad -6.480
Epoch 19  |  θ  91.9°  |  Final dist 1.01  |  Loss 0.006  |  Grad  6.340
Epoch 20  |  θ  90.0°  |  Final dist 0.99  |  Loss 0.009  |  Grad -6.480
```
The final value we get is 90.0°, close to the theoretical value of 91.2° which is obtainable from trigonometry.
The final simulation looks like this:
![Angle simulation](images/sim_angle.gif)
In the presence of other forces this value would not be so trivially obtainable.
We can record the gradients for different values of `θ`:
```julia
θs = collect(0:3:180)[2:end]
grads = Float64[]

for θ in θs
    coords = [
        SVector(0.8, 0.75, 1.5), SVector(1.5, 0.74, 1.5), SVector(2.3, 0.75, 1.5),
        SVector(0.8, 2.25, 1.5), SVector(1.5, 2.24, 1.5), SVector(2.3, 2.25, 1.5)]
    velocities = [velocity(mass, temp; dims=n_dims) for i in 1:n_atoms]
    push!(grads, gradient(loss, deg2rad(θ))[1])
end
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
    Dense(5, 1, tanh)
)
ps = params(model)

struct NNBond <: SpecificInteraction
    i::Int
    j::Int
end

# This method of defining the force is outdated compared to the master branch
function Molly.force(coords, b::NNBond, s::Simulation)
    ab = vector(coords[b.i], coords[b.j], s.box_size)
    dist = Float32(norm(ab))
    f = model([dist])[1] * normalize(ab)
    return [b.i, b.j], [f, -f]
end

n_steps = 400
mass = 10.0f0
box_size = 5.0f0
timestep = 0.02f0
temp = 0.0f0
integrator = VelocityVerlet()
neighbor_finder = NoNeighborFinder()
thermostat = FrictionThermostat(0.98f0)
general_inters = (LennardJones(false),) # Ignored due to atom parameters
coords = [SVector(2.3f0, 2.07f0), SVector(2.5f0, 2.93f0), SVector(2.7f0, 2.07f0)]
n_atoms = length(coords)
n_dims = length(first(coords))
velocities = zero(coords)

function loss()
    atoms = [Atom("", "", 0, "", 0.0f0, mass, 0.0f0, 0.0f0) for i in 1:n_atoms]
    loggers = Dict("coords" => CoordinateLogger(10; dims=n_dims))
    specific_inter_lists = ([NNBond(1, 3)],)

    s = Simulation(
        simulator=integrator,
        atoms=atoms,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        temperature=temp,
        box_size=box_size,
        neighbor_finder=neighbor_finder,
        thermostat=thermostat,
        loggers=loggers,
        timestep=timestep,
        n_steps=n_steps
    )

    final_coords = simulate!(s)
    dist_end = (norm(vector(final_coords[1], final_coords[2], box_size)) +
                norm(vector(final_coords[2], final_coords[3], box_size)) +
                norm(vector(final_coords[3], final_coords[1], box_size))) / 3
    loss_val = abs(dist_end - dist_true)

    Zygote.ignore() do
        printfmt("Dist end {:6.3f}  |  Loss {:6.3f}\n", dist_end, loss_val)
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
        coords = [SVector(2.3f0, 2.07f0), SVector(2.5f0, 2.93f0), SVector(2.7f0, 2.07f0)]
        velocities = zero(coords)
        printfmt("Epoch {:>2}  |  ", epoch_n)
        Flux.train!(loss, ps, ((),), opt)
    end
end

train()
```
```
Epoch  1  |  Dist end  0.675  |  Loss  0.325
Epoch  2  |  Dist end  0.715  |  Loss  0.285
Epoch  3  |  Dist end  0.756  |  Loss  0.244
Epoch  4  |  Dist end  0.793  |  Loss  0.207
Epoch  5  |  Dist end  0.832  |  Loss  0.168
Epoch  6  |  Dist end  0.874  |  Loss  0.126
Epoch  7  |  Dist end  0.918  |  Loss  0.082
Epoch  8  |  Dist end  0.963  |  Loss  0.037
Epoch  9  |  Dist end  1.008  |  Loss  0.008
Epoch 10  |  Dist end  1.036  |  Loss  0.036
Epoch 11  |  Dist end  1.052  |  Loss  0.052
Epoch 12  |  Dist end  1.060  |  Loss  0.060
Epoch 13  |  Dist end  1.060  |  Loss  0.060
Epoch 14  |  Dist end  1.054  |  Loss  0.054
Epoch 15  |  Dist end  1.044  |  Loss  0.044
Epoch 16  |  Dist end  1.029  |  Loss  0.029
Epoch 17  |  Dist end  1.011  |  Loss  0.011
Epoch 18  |  Dist end  0.990  |  Loss  0.010
Epoch 19  |  Dist end  0.977  |  Loss  0.023
Epoch 20  |  Dist end  0.971  |  Loss  0.029
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
- A combination of the above, for example to obtain a forcefield relevant to both ordered and disordered proteins.
Some of these are currently not possible in Molly as the loggers are ignored for gradient purposes, but this will hopefully change when Zygote gets mutation support.

## Tips and tricks

- Exploding gradients prove a problem when using the velocity Verlet integrator in the NVE ensemble. This is why a thermostat that downscales velocities was used in the above examples - presumably it decays the gradients to the level required for learning. It is likely that the development of suitable simulation strategies and thermostats will be necessary to unlock the potential of differentiable simulation.
- Do you *really* need a neural network to describe your potential? Think about learning a smaller number of physically-meaningful parameters before you put in a large neural network and expect it to learn. Whilst it is true that neural networks are universal function approximators, it does not follow that you will be able to train one by differentiating through  a long simulation. A 1000-step simulation with a 10-layer network at each step is analogous to training a 10,000 layer network (with shared weights).
- Forward mode autodiff holds much promise for differentiable simulation, provided the number of parameters is small, because the memory requirement is constant in the number of simulation steps.
