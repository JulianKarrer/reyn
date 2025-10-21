<h1 align="center">REYN - FLUID SIMULATION</h1>
<h3 align="center">A CUDA-based implementation of Smoothed Particle Hydrodynamics for Fluid Simulation</h3>

<p align=center style="margin-top:50px;">
    <img src="./res/icon.png" width=300 height = 300/>
</p>

## FEATURES
- Zero-Copy GUI thanks to CUDA <-> OpenGL buffer sharing, with a framerate upwards-uncoupled from the simulation
- Interchangable kernel functions (Wendland C2, C6, Cubic B-Spline, Double Cosine, ...) with CRTP static polymorphism for maximum runtime performance and minimum implementation effort
- Automatic testing of kernel functions using `doctest`


## TODO
- [x] Test zero-copy OpenGL VBO to CUDA buffer interop
- [x] Add efficient visualization and GUI based on OpenGL interop, including ImGUI elements and single-batch billboard sphere rendering
- [x] Implement safer abstraction over device arrays (no manual `cudaMalloc` and `cudaFree`) that is interoperable `thrust` and OpenGL
- [ ] Implement acceleration datastructure, e.g. [Simon Green, 2012] using efficient prefix scan
- [ ] Add settings management with de-/serialization and expose solver and scene options to GUI
- [ ] Add scene management, sampling and de-/serialization
- [ ] Add Boundary handling [Akinci et al. 2012]
- [ ] Add adaptive time step size calculation using efficient reductions
- [ ] Implement standard iterative Equation-of-State based solvers (splitting, PCISPH)
- [ ] Implement Jacobi-style incompressible solvers (IISPH, DFSPH, ...)
