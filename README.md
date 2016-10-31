# FENLACE

The name is short for FEniCS based NLACE. FEniCS is a collection of free software with an extensive list of features for automated and efficient solution of differential equations (https://fenicsproject.org). NLACE is short for NonLinear Adjoint based Coefficients Estimator. It is developed by Assad A. Oberai, Paul Barbone and their students. More details about NLACE can be found by examing Prof. Oberai's (http://homepages.rpi.edu/~oberaa) and/or Prof. Barbone's related publication lists (http://people.bu.edu/barbone/biomech_imaging.html) . NLACE solves practical inverse problems / PDE-constrained optimization problems mainly in the biology-related fields, such as tissue BioMechanical Imaging and cell Traction Force Microscopy, where the former one inverts for tissue properties and the latter inverts for cell surface traction.

NLACE is written in FORTRAN and parallelized in openMP. In order to make use of the parallelism of MPI in FEniCS, we are trying to "re-write" NLACE in FEniCS with Python. The dolfin-adjoint package (http://www.dolfin-adjoint.org/) in FEniCS could further simplify our implementation and speed up running time.

This README only serves as a rough descritption and by no means complete. I am working on this out of my personal interest and any comments and/or collaborations from all NLACE developers are very welcome.

Required packages: Basically all essential packages in FEniCS plus dolfin-adjoint. Details of installing them can be easily found on their websites, though building them without root access can be a pain.


Brief code structure(for now):

lprep.py prepares the measured disp data in .h5 format for parallel reading. This is a work-around to get away from confusing dofmap in parallel environment. The solution comes from https://fenicsproject.org/qa/11448/assign-values-to-function-in-parallel.

lmain.py include the core code for inversion.

lstart.sh start the inversion. You may want to read this to adjust it to your needs.
