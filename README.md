# pyGPA

A python package collecting an assortment of Geometric Phase Analysis algorithms and tools to analyse regular (moiré) lattices. 

Geometric Phase Analysis enables the extraction of a _displacement field_ from a distorted lattice, by comparing to reference waves. As such, it is a spatial variant of lock-in amplification. This package implements several specializations of GPA. In particular, it extends the original GPA algorithm by optimizing over a range of reference vectors, (as previously described by\[[1]((https://doi.org/10.1016/j.optlaseng.2005.10.012)\] to achieve accurate characterization of larger distortions. This is especially useful when analysing (atomic) small angle moiré patterns.

From the displacement field, it is possible to extract local properties of the underlying lattice, but also to correct for the distortion, a process known in STM as the Lawler-Fujita algorithm \[[2](https://doi.org/10.1038/nature09169)\].

Once the displacement field is known, it is also possible to perform unit cell averaging directly from the distorted image.

## See also

https://github.com/TAdeJong/moire-lattice-generator

[T. Benschop, T.A. de Jong _et al._ Phys. Rev. Research *3*, 013153](https://doi.org/10.1103/PhysRevResearch.3.013153) (Code not actually used for that work, but builds upon the ideas)

https://github.com/TAdeJong/weighed_phase_unwrap

Kemao, Qian. "Two-dimensional windowed Fourier transform for fringe pattern analysis: principles, applications and implementations." [Optics and Lasers in Engineering 45.2 (2007): 304-317.](https://doi.org/10.1016/j.optlaseng.2005.10.012)

# Acknowledgement

This work was financially supported by the [Netherlands Organisation for Scientific Research (NWO/OCW)](https://www.nwo.nl/en/science-enw) as part of the [Frontiers of Nanoscience (NanoFront)](https://www.universiteitleiden.nl/en/research/research-projects/science/frontiers-of-nanoscience-nanofront) program.
