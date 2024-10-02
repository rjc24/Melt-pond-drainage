# Melt pond drainage

A finite-volume code for simulating the time evolution of the sea-ice interior during melt-pond drainage. The ice is modelled as a mushy layer, meaning that the code can be easily adapted to other problems involving sea ice and mushy layers. The code is written exclusively in Python 3.x, using numpy 1.x and scipy 1.x. Details of the numerical method are presented in Campbell 2024 (not yet publicly available).

# Using the code
To use the code, simply follow these steps:
1. Place the files listed in the Main code section (see below) into a directory.
2. In the same directory, create two sub-directories with the names **data** and **arrays**. Within the **arrays** directory, create a further sub-directory titled **cont**. Finally, inside **/data/** and **/arrays/cont/**, create directories corresponding to each of the runs you wish to perform. These should be named following the convention **run_n**. Snapshots of the time-evolving data are saved to **data/run_n** and miscellaneous data are saved to **arrays/cont/run_n** (e.g. dimensionless parameters, grid resolution, boundary conditions, etc.).
3. Back in the main directory, run the file **Data_run.py** (or equivalent) using Python3.x. You will be prompted to enter the name of a directory for storing data, the name of the file containing initial conditions, various dimensionless parameters, the grid resolution, boundary condition settings and the simulation run time. The directory name should be of the format **run_n** and should match the name of directories previously created in **/data/** and **/arrays/cont/**. The file **Data_run.py** can be modified or replaced by a similar file to account for different initial conditions or to automate a large number of runs. This repository contains a small number of alternatives to **Data_run.py** with corresponding initial condition files as an example.

# Summary

The files provided in this repository are sufficient to calculate approximate solutions to the mushy-layer equations, given a set of initial conditions. A summary of the code in each file is presented below:

## Main code
- **Data_call.py**: Calculates approximate time-evolving solutions to the mushy-layer equations and saves the enthalpy, temperature, bulk salinity and pressure fields at user-specified snap shots. Called by **Data_run.py**. The initial conditions, dimensionless parameters, run time, etc. are specified in **Data_run.py**.
- **Timestep.py**: Calculates the solution at time t = t^{n+1} using adaptive time-stepping. Called by **Data_call.py**.
- **NL_solve.py**: Performs Picard/Newton iteration to solve systems of equations of the form **A****x** = **b** for enthalpy, bulk salinity and pressure at time t = t^{n+1}. Estimates for each quantity are used to iteratively produce newer, improved estimates. Called by **Timestep.py**.
- **Matrix_def.py**: Compiles the matrix **A** used to calculate solutions to the discretised heat/solute conservation equation at time t = t^{n+1}. Called by **NL_solve.py**.
- **Vector_def.py**: Compiles the vector **b** used to calculate solutions to the discretised heat/solute conservation equation at time t = t^{n+1}. Called by **NL_solve.py**.
- **elliptic_solve_sparse.py**: Solves Poisson equation to obtain pressure field. From pressure, horizontal and vertical velocity components are calculated using Darcy's law. Called by **NL_solve.py**.
- **Enthalpy_functions.py**: Contains various functions relating different quantities tracked in the mush interior. Called by various.
- **newton_functions.py**: Contains functions used to construct the Jacobian matrix for Newton's method. The Jacobian is currently based only on the diffusive terms of the heat/solute conservation equations. Improved perfomance could likely be obtained if the advective terms were accounted for as well. Called by **NL_solve.py**.
- **upwind.py**: Contains the function for calculating temperature and liquid concentration values at cell faces using a MUSCL-style limiter. Called by **NL_solve.py**.
- **t_int.py**: Linearly interpolates data in time to obtain fields at a user-specified snap shot. Called by **Data_call.py**.
