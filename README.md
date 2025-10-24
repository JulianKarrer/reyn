<h1 align="center">REYN - FLUID SIMULATION</h1>
<h3 align="center">A CUDA-based Implementation of Smoothed Particle Hydrodynamics for Fluid Simulation</h3>

<p align=center style="margin-top:50px;">
    <img src="./res/icon.png" width=300 height = 300/>
</p>

## FEATURES
- Zero-Copy GUI thanks to CUDA <-> OpenGL buffer sharing, with a framerate upwards-uncoupled from the simulation
- Interchangable kernel functions (Wendland C2, C6, Cubic B-Spline, Double Cosine, ...) with CRTP static polymorphism for maximum runtime performance and minimum implementation effort
- GPU-friendly spatial acceleration datastructure for fixed radius neighbourhood queries with work-efficient construction and constant query time
- Automatic testing of kernel functions and spatial acceleration datastructures using `doctest`


## TODO
- [x] Test zero-copy OpenGL VBO to CUDA buffer interop
- [x] Add efficient visualization and GUI based on OpenGL interop, including ImGUI elements and single-batch billboard sphere rendering
- [x] Implement safer abstraction over device arrays (no manual `cudaMalloc` and `cudaFree`) that is interoperable `thrust` and OpenGL
- [x] Implement acceleration datastructure, e.g. [Hoetzlein, 2014] using efficient prefix scan
- [ ] Reorder state according to space-filling curve to improve memory coherency
- [ ] Parameterize uniform grid with cell size different from search radius and benchmark for optimal grid size
- [ ] Add benchmarking for performance optimization
- [ ] Add settings management with de-/serialization and expose solver and scene options to GUI
- [ ] Add scene management, sampling and de-/serialization
- [ ] Add boundary handling [Akinci et al. 2012]
- [ ] Add adaptive time step size calculation using efficient reductions
- [ ] Implement standard iterative Equation-of-State based solvers (splitting, PCISPH)
- [ ] Implement Jacobi-style incompressible solvers (IISPH, DFSPH, ...)
- [ ] Optimize kernel launch configurations
- [ ] Use type aliases for safety (e.g. separating vectors and positions) and flexibility (change float precision, smaller index types for smaller scenes etc.)
