# Melt pond drainage

A finite-volume code for simulating the time evolution of the sea-ice interior during melt-pond drainage. The ice is modelled as a mushy layer, meaning that the code can be easily adapted to other problems involving sea ice and mushy layers. The code is written exclusively in Python 3.x, using numpy 1.x and scipy 1.x. Details of the numerical method are presented in Campbell 2024 (not yet publicly available).

# Using the code
To use the code, simply follow these steps:
1. Place the files from the **main_code** and **inits** directories into a single directory.
2. In the same directory, create two sub-directories with the names **data** and **arrays**. Within the **arrays** directory, create a further sub-directory titled **cont**. Finally, inside **/data/** and **/arrays/cont/**, create directories corresponding to each of the runs you wish to perform. These should be named following the convention **run_n**. Snapshots of the time-evolving data are saved to **data/run_n** and miscellaneous data are saved to **arrays/cont/run_n** (e.g. dimensionless parameters, grid resolution, boundary conditions, etc.).
3. Back in the main directory, run the file **Data_run.py** (or equivalent) using Python3.x. You will be prompted to enter the name of a directory for storing data, the name of the file containing initial conditions, various dimensionless parameters, the grid resolution, boundary condition settings and the simulation run time. The directory name should be of the format **run_n** and should match the name of directories previously created in **/data/** and **/arrays/cont/**. The file **Data_run.py** can be modified or replaced by a similar file to account for different initial conditions or to automate a large number of runs. This repository contains a small number of alternatives to **Data_run.py** with corresponding initial condition files. These are detailed in the summary section below.

# Summary

The files provided in this repository are sufficient to calculate approximate solutions to the mushy-layer equations, given a set of initial conditions. A summary of the code in each directory is presented below:

## main_code
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
- **interface.py**: Calculates approximate interface positions located at horizontal cell faces between liquid and mush cells. Improved interface positions can be calculated _a poteriori_ using quadratic extrapolation. Called by **Data_call.py**.
- **phys_params.py**: Contains parameters derived from physical quantities. Called by various.
- **num_params.py**: Contains parameters relating to the numerical implementation. Called by various.


## inits
- **Data_run.py**, **init_2D_1.py**: Files used for runs with a single straight channel.
- **Data_run_r.py**, **init_2D_r.py**: Files used for runs with a single tilted channel.
- **Data_run_t.py**, **init_2D_t.py**: Files used for runs with a single straight channel attended by tributary channels.
- **init_2D_c.py**: An initial condition file used to continue a run for a further period of time. This file can be used with any of the **Data_run** files listed above. The directory specified by the user should match the directory containing data from the run they wish to continue.

## misc
- **colormaps.py**: colour maps used for visualisation of the solutions
- **streamfunction.py**: Calculates the streamfunction from the solid fraction and pressure fields at a given time. This function is not used in the main code and is included for visualisation of the solutions.

## fixed_salinity
- **NL_solve.py**: An alternative version of NL_solve.py where the bulk salinity is fixed (i.e. does not evolve in time).
- **init_2D_1.py**: An alternative version of init_2D_1.py for fixed-salinity runs (differs in that the pond salinity is constant and equal to the bulk salinity of the ice). To generate fixed-salinity solutions, replace **NL_solve.py** and **init_2D_1.py** from the **main_code** directory with the files from the **fixed_salinity** directory. 

# Notes
Before using the code, the user should make themselves aware of the following:
- For numerical convenience, the salinity in the code is actually the _negative_ salinity. In our nondimensionalisation, the minimum salinity is _-Cr_ (the negative concentration ratio), meaning that in the code, the maximum salinity is _Cr_.
- If simulating the drainage of very fresh melt water (i.e. with salinity close to _-Cr_ or, in the code, _Cr_), the liquid will likely become constitutionally supercooled near mush-liquid interfaces. Due to the marginal equilibrium condition underpinning the enthalpy method used in the code, this results in the formation of solid fraction spikes which can cause the numerical method to become unstable. For this reason, the solute-conservation equation is solved with a small numerical diffusion term. The numerical diffusivity is based on the grid resolution and can be modified in the **num_params.py** file. Depending on the pond salinity, Peclet number and the boundary conditions being used, a larger numerical diffusivity may be required.
- The code currently only supports Dirichlet boundary conditions and _homogeneous_ Neumann boundary conditions on temperature and bulk salinity. If using _inhomogeneous_ Neumann conditions, the **Vector_def.py** file will have to be amended.
