# Developer documentation

## Running tests

We use [ParallelTestRunner.jl](https://github.com/JuliaTesting/ParallelTestRunner.jl) to run the tests in parallel across multiple threads.
This allows different test groups to be run, for example `julia test/runtests.jl gradients` for the gradient tests or `julia test/runtests.jl \!gradients \!extra` for the non-gradient tests.
The tests will automatically include multithreading and/or GPU tests if multiple threads and/or a GPU are available.
Warnings appearing at the start of the test run due to unavailable backends is expected.
`test/runtests.jl` does not include all the tests, see the `test/extra` directory for more, though these extra tests do not need to be run for every change.
Various environmental variables can be set to modify the tests:
- `VISTESTS` determines whether to run the [GLMakie.jl](https://github.com/JuliaPlots/Makie.jl) plotting tests which will error on remote systems where a display is not available, default `VISTESTS=1`.
- `GPUTESTS` determines whether to run the GPU tests if a GPU is available, default `GPUTESTS=1`.
- `DEVICE` determines which GPU to run the GPU tests on, default `DEVICE=0`.
The CI run does not carry out all tests - for example the GPU tests are not run - and this is reflected in the code coverage.
Running the test file locally gives higher coverage.

## Periodic boundary conditions

Molly uses the minimum image convention when applying periodic boundary conditions, meaning that of all the periodic copies of an interacting atom, only the closest is considered.
This means that the cutoff distance should not be greater than half of any of the periodic box dimensions, otherwise interactions within the cutoff distance will be missed.

Molly generally keeps all atoms within the "main" periodic box, even when that means splitting molecules over the boundary.
This is different to some other software, where molecules are kept whole.
In practice this doesn't make too much difference since specific interactions like bonds use the nearest periodic image of an atom.
This could lead to issues if a different copy of the atom is the intended interacting atom, but for most molecular systems this is not a problem.
It does lead to some complexity in [`scale_coords!`](@ref), as molecules have to be made whole before scaling.
When writing out files, the default (`correction=:pbc`) is to move atoms such that molecules are whole.
The case where all atoms are in one periodic box can be accessed with `correction=:wrap`.
For more discussion, see the [OpenMM FAQs](https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions).

## Custom neighbor finders

To define your own neighbor finder, first define the `struct`:
```julia
struct MyNeighborFinder{E, S}
    eligible::E
    special::S
    n_steps::Int
    # Any other properties, e.g. a distance cutoff
end
```
Examples of three useful properties are given here: a matrix indicating atom pairs eligible for pairwise interactions, a matrix indicating atoms in a special arrangement such as 1-4 bonding, and a value determining how many time steps occur between each evaluation of the neighbor finder.
When a [`System`](@ref) is set up from a file with `neighbor_finder_type=MyNeighborFinder`, the `eligible` and `special` matrices are given as [`SparsePairMatrix`](@ref)s.
These can be indexed like any other `AbstractMatrix{Bool}` but take memory proportional to the number of pairs rather than to the square of the number of atoms, so avoid field types like `BitArray{2}` that would convert them to dense matrices.
Then, define the neighbor finding function that is called every step by the simulator:
```julia
function Molly.find_neighbors(sys,
                              nf::MyNeighborFinder,
                              current_neighbors=nothing,
                              step_n::Integer=0,
                              force_recompute::Bool=false;
                              n_threads::Integer=Threads.nthreads())
    if force_recompute || step_n % nf.n_steps == 0
        if isnothing(current_neighbors)
            neighbors = NeighborList()
        else
            neighbors = current_neighbors
        end
        empty!(neighbors)
        # Add to neighbors, for example
        push!(neighbors, (1, 2, false)) # atom i, atom j and whether they are in a special interaction
        return neighbors
    else
        return current_neighbors
    end
end
```
To use your custom neighbor finder, give it as the `neighbor_finder` argument when creating the [`System`](@ref).

## Exception types

Molly throws various standard exception types, especially `ArgumentError`s, but also has its own error types:

| Exception type                                     | Description                                                                                     |
| :------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| [`Molly.NaNSimulationError`](@ref)                 | A `NaN` is encountered during a simulation or setup                                             |
| [`Molly.ForceFieldXMLError`](@ref)                 | An error reading a force field XML file, [see more here](@ref "Simulating a protein")           |
| [`Molly.MissingResidueTemplateError`](@ref)        | A residue cannot be matched to a residue template, [see more here](@ref "Simulating a protein") |

## Fast math

On CUDA GPUs the pairwise force and energy kernels are compiled with fast math for `Float32` systems, where the loss in precision is not noticeable. `Float64` systems do not use fast math, since there is a noticeable loss in precision and that path is much slower anyway.

## Benchmarks

The `benchmark` directory contains some benchmarks for the package.

## ReaxFF implementation details

### Functional form

The ReaxFF implementation in Molly closely follows the 2008 ReaxFF functional form ([Chenoweth et al. 2008](https://doi.org/10.1021/jp709896w)), and has both CPU and GPU support. 
The main differences of this code compared with the original paper functional forms are summarized below.

In the following, ``c`` denotes `bo_cutoff`, and variables marked with a prime denote uncorrected bond orders.

**Uncorrected σ bond order (Eq. (2)).** The original SI writes ``BO'^\sigma = \exp[p_{\mathrm{bo1}}(r / r_o^\sigma)^{p_{\mathrm{bo2}}}]``.
This code applies a cutoff adjustment: ``BO'^\sigma = (1 + c)\exp[p_{\mathrm{bo1}}(r / r_o^\sigma)^{p_{\mathrm{bo2}}}] - c``, which likely comes from the original ReaxFF Fortran code (see [discussion](https://matsci.org/t/about-reaxff-implementation/46591/3)). 
The likely motivation for this is to ensure the total bond order goes to zero as a pair approaches the threshold of appearing on the bonded list.  

**Bond-order correction (Eq. (4a)).** The SI corrects the uncorrected σ bond order using ``BO^\sigma = BO'^\sigma \cdot f_1 \cdot f_4 \cdot f_5``, and corrects uncorrected π and ππ bond orders using ``f_1^2 \cdot f_4 \cdot f_5``. 
It then sums these corrected components to obtain the total corrected bond order.
This code corrects the total uncorrected bond order using ``BO = BO' \cdot f_1 \cdot f_4 \cdot f_5``, corrects uncorrected π and ππ bond orders using ``f_1^2 \cdot f_4 \cdot f_5``, and then obtains corrected σ bond order as ``BO^\sigma = BO - BO^\pi - BO^{\pi\pi}``.
This form is used in [Aktulga et al. (2012)](https://doi.org/10.1137/100808599) (Eq. (2.3)).

**Lone-pair count (Eq. (8)).** The first integer term is written as ``+\operatorname{int}(\Delta^e / 2)`` in the SI.
This code uses ``-\operatorname{trunc}(\Delta^e / 2)``, followed by the same exponential term afterwards.
The original printed positive sign appears to be inconsistent with the definition of ``\Delta^e`` in Eq. (7).

**Torsion ``V_2`` contribution (Eq. (16a)).** The exponential term in the SI contains ``p_{\mathrm{tor1}}(BO^\pi_{jk} - 1 + f_{11})^2``.
This code uses ``p_{\mathrm{tor1}}(2 - BO^\pi_{jk} - f_{11})^2``, which changes the offset inside the squared term.
This energy term is present in the original ReaxFF source code, see [Shchygol et al. (2019)](https://doi.org/10.1021/acs.jctc.9b00769) supporting information, section S2.2.

**Hydrogen-bond angular dependence (Eq. (18)).** The SI contains ``\sin^8(\theta/2)`` for the angular dependence of hydrogen bonds.
This code uses ``\sin^4(\theta/2)``.
This new form was used after the introduction of the ReaxFF water branch.

**Hydrogen-bond exponential signs (Eq. (18)).** The SI contains ``1 - \exp(+p_{\mathrm{hb2}} \cdot BO)`` and a radial exponential with ``+p_{\mathrm{hb3}}``.
This code uses ``1 - \exp(-p_{\mathrm{hb2}} \cdot BO)`` and ``-p_{\mathrm{hb3}}``.
This assumes the corresponding parameters in the force field file do not need negating after being read.

### Force-field compatibility

**Triple-bond stabilization:** This code selects C–O pairs by the type names `C` and `O` and ignores general parameter 38 (which extends the correction to other bond types).

**Inner-wall van der Waals:** This code implements the original shielded form without the inner-wall contribution. Atom parameters 30–32 (`rcore2`, `ecore2`, `acore2`) are ignored.

**Nonbonded taper:** This code assumes a zero lower taper radius: non-zero values of `lower_taper_radius` are ignored.

**Hydrogen-bond roles:** Hydrogen bonding uses the atom type `H`, with donor and acceptor types inferred from the hydrogen-bond parameter block. The atomic `p_hbond` flag is ignored.

**Polarization energy:** This code uses the conversion factor `332.06371 / 14.4 ≈ 23.05998`, consistent with its Coulomb and EEM constants, instead of the legacy value of `23.02`. This changes the polarization contribution by approximately 0.174%, and the corresponding reference value in the tests is rescaled accordingly. 

### Furman-Wales smoothness correction

In this code, `smoothness_correction=true` enables additional seventh-degree tapers for selected bond-order and bonded-interaction cutoffs.
These corrections are disabled by default; see [Furman and Wales smoothness correction](@ref).

The function `smoothness_taper(x, a, b)` returns an ascending taper value, as well as its derivative with respect to `x`.
Within the interval ``a < x < b``, the taper uses ``t = (x-a)/(b-a)`` and ``S(x) = 35t^4 - 84t^5 + 70t^6 - 20t^7``, with derivative ``S'(x) = 140t^3(1-t)^3/(b-a)``.
The taper is zero for ``x \leq a`` and one for ``x \geq b``, with its first three derivatives vanishing at both endpoints.
The corresponding descending taper is obtained using ``1-S(x)`` and negating the derivative.

Each uncorrected bond-order component (σ, π and ππ) is multiplied by its own taper over the interval ``[c, 4c]``, where ``c`` is `bo_cutoff`.
For the σ bond-order component, the existing ``(1+c)\exp(\ldots)-c`` change described above is also applied before tapering.
The tapered components feed into the usual bond-order correction and coordination calculations.
The additional truncation of corrected bond orders below ``10^{-10}``, present in standard ReaxFF, is not performed when the smoothness correction is enabled to avoid introducing another discontinuity.

The valence-angle, penalty and three-body conjugation energies are multiplied by the product of two tapers: one for each participating bond.
Torsion and four-body conjugation energies use the corresponding product of three taper terms.
Each taper is evaluated from a corrected total bond order over the interval `[valence_cutoff, 4 * valence_cutoff]`, while the energy terms use the existing `BO - valence_cutoff` terms.
The previous separate checks on the products of bond orders are bypassed when the smoothness correction is enabled, since they would otherwise introduce discontinuities.

Hydrogen-bond energies use both a donor hydrogen bond-order taper over the interval `[hbond_bo_cutoff, 4 * hbond_bo_cutoff]` and a decreasing hydrogen–acceptor distance taper over the interval ``[0.9R, R]``, where ``R`` is `hbond_dist_cutoff`.
The distance interval deliberately restricts this taper to the outer 10% of the cutoff, so that it retains the existing behaviour at shorter distances.
This differs from the description in the Furman paper, which states that a ``0.9R`` width is used.

The implementation also tapers the activation of the C–O triple-bond correction, which is not performed in the original paper.
Its energy is multiplied by an ascending taper over the corrected total bond order interval ``[0.9, 1.0]``, replacing the abrupt activation at ``BO = 1``. 
The taper is exactly one above ``BO = 1``, so the existing functional form is retained there.

All tapered terms also include the corresponding chain rule terms in their analytical force derivatives.

### Charge equilibration

Charge equilibration uses the electronegativity equalization method (EEM) ([Mortier et al. 1986](https://doi.org/10.1021/ja00275a013) and [Rappé and Goddard 1991](https://doi.org/10.1021/j100161a070)).

Charges are obtained by minimizing the charge-dependent electrostatic energy while preserving the total system charge as specified by the initial charge vector.
The solver uses the method of projected preconditioned conjugate gradients, closely following the algorithm detailed in [Shewchuk 1994](https://dl.acm.org/doi/10.5555/865018).
Previous converged charges are used as an initial guess for each new MD step.

EEM is implemented separately from ReaxFF, with an interaction-specific coupling supplying the Coulomb coefficients.
ReaxFF supplies its tapered, shielded Coulomb form.
Through `evaluate_before_pairwise`, EEM runs before pairwise and specific interactions and writes the equilibrated charges into `sys.atoms`.
EEM contributes the polarization self-energy, while ReaxFF supplies the Coulomb pair energy and forces.
