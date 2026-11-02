# ReaxFF force fields

Force-field filenames use the format `Author_Year_Elements.ff`. `X` place-holder atom types are omitted from the filename. Differences from the original supporting information are documented below.

## Chenoweth_2008_CHO.ff

Reference: K. Chenoweth, A. C. T. van Duin and W. A. Goddard III,
*ReaxFF Reactive Force Field for Molecular Dynamics Simulations of Hydrocarbon
Oxidation*, Journal of Physical Chemistry A 112, 1040–1053 (2008).
[DOI: 10.1021/jp709896w](https://doi.org/10.1021/jp709896w).

This is the corrected version. Compared with the original paper's supporting information:

- General parameter `kc2` is `70.0000` instead of `0.0000`. This correction was made
  on 3 November 2010 after consulting the ReaxFF authors, to address excessive stability 
  of C₂ molecules. See the
  [maintainer's explanation](https://matsci.org/t/lammps-users-observation-of-c2-molecules-resulting-from-lammps-reaxff-simulations/11391).
- The declared torsion count is `26` instead of `21`. The original supporting
  information contains 26 torsion entries despite declaring 21.


## Rahaman_2011_CHON.ff

Reference: O. Rahaman, A. C. T. van Duin, W. A. Goddard III and D. J. Doren,
*Development of a ReaxFF Reactive Force Field for Glycine and Application to
Solvent Effect and Tautomerization*, Journal of Physical Chemistry B 115,
249–261 (2011).
[DOI: 10.1021/jp108642r](https://doi.org/10.1021/jp108642r).

Compared with the original supporting information, the following angle entry was removed 
because it references atom type `5`, despite only four atom types being defined:

```text
1 2 5  0.0000 15.0000 3.0000 0.0000 0.0000 0.0000 1.0400
```

The declared angle count was consequently reduced from `42` to `41`. 


## Monti_2013_CHONSMgPNaCuCl.ff

Reference: S. Monti et al., *Exploring the conformational and reactive dynamics
of biomolecules in solution using an extended version of the glycine reactive
force field*, Physical Chemistry Chemical Physics 15, 15062–15077 (2013).
[DOI: 10.1039/C3CP51931G](https://doi.org/10.1039/C3CP51931G).

Also known as ProtReaxFF.

All parameter values, atom-type entries and section counts match the force-field
block on pages S5–S10 of the paper's supporting information.


## Zhang_2017_HO.ff

Reference: W. Zhang and A. C. T. van Duin, *Second-Generation ReaxFF Water Force
Field: Improvements in the Description of Water Density and OH-Anion Diffusion*,
Journal of Physical Chemistry B 121, 6021–6032 (2017).
[DOI: 10.1021/acs.jpcb.7b02548](https://doi.org/10.1021/acs.jpcb.7b02548).

All parameter values, atom-type entries and section counts match the force-field
block on pages S2–S3 of the paper's supporting information.


## Trnka_2018_CHON.ff

Reference: T. Trnka, I. Tvaroška and J. Koča, *Automated Training of ReaxFF
Reactive Force Fields for Energetics of Enzymatic Reactions*, Journal of Chemical
Theory and Computation 14, 291–302 (2018).
[DOI: 10.1021/acs.jctc.7b00870](https://doi.org/10.1021/acs.jctc.7b00870).

The file matches the original supporting-information force-field file `optimized.ff`.
