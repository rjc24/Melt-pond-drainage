# Melt pond drainage

A finite-volume code for simulating the time evolution of the sea-ice interior during melt-pond drainage. The ice is modelled as a mushy layer, meaning that the code can be easily adapted to other problems involving sea ice and mushy layers. The code is written exclusively in Python 3.x, using numpy 1.x and scipy 1.x. Details of the numerical method are presented in Campbell 2024 (not yet publicly available).

# Summary

The files provided in this repository are sufficient to calculate approximate solutions to the mushy-layer equations, given a set of initial conditions. A summary of the code in each file is presented below:

## Main code
- **Matrix_def.py**: Compiles the matrix **A** used to calculate solutions to the discretised heat/solute conservation equation at time t = t^{n+1}
- **Vector_def.py**: Compiles the vector **b** used to calculate solutions to the discretised heat/solute conservation equation at time t = t^{n+1}
- **elliptic_solve_sparse.py**: Solves Poisson equation to obtain pressure field _p_. From pressure _p_, horizontal and vertical velocity components _u_ and _w_ are calculated using Darcy's law.
- **NL_solve.py**: Performs Picard/Newton iteration to obtain approximate enthalpy _H_ and bulk salinity _C_ fields at time t = t^{n+1}. Repeatedly calls Matrix_def.py and Vector_def.py to calculate **A** and **b** using improved estimates for quantities at time t = t^{n+1}. Improved estimates are used in elliptic_solve_sparse.py to update estimates for pressure and velocity.
