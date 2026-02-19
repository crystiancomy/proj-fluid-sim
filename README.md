# 2D Particle Fluid Simulation (Taichi)

A real-time GPU-accelerated 2D particle-based fluid-like simulation built with **Taichi (Python)**.

The simulation runs on Taichi’s cross-platform GPU backend, automatically targeting **CUDA, Vulkan, Metal, or DX12** depending on the system.

This project evolved from a naive O(n²) pairwise interaction model to a spatially partitioned system using a **uniform grid (spatial hashing)**, significantly improving scalability and performance.

---

## Demo

### v0.3 — Spatial Grid Optimized

![Fluid Simulation](demo.gif)

---

## Features

- Cross-platform GPU acceleration (CUDA / Vulkan / Metal / DX12 via Taichi)
- Semi-implicit Euler integration
- Gravity force
- Spring-based repulsion (collision response)
- Cohesion force (short-range attraction)
- Viscosity damping between neighbors
- Boundary constraints with energy loss
- Uniform grid spatial hashing (3×3 neighbor lookup)
- 5,000+ particles in real time

---

## Performance Evolution

### v0.2 — Naive Pairwise Interaction

All particles interact with each other:

O(n²)

Limited scalability (~2000 particles).

---

### v0.3 — Spatial Hashing (Uniform Grid)

Particles are inserted into grid cells based on position.
Each particle checks only neighbors inside its own cell and surrounding 8 cells.

Average complexity:

O(n · k)

Where **k** is the average number of particles per cell (typically small and bounded).

Enables stable real-time simulation with 5000+ particles.

---

## Technical Highlights

- GPU parallel kernels via Taichi
- Atomic grid insertion
- Data-oriented design (SoA layout)
- Reduced neighbor search complexity
- Clean separation between:
  - Force accumulation
  - Spatial partitioning
  - Integration step

---

## Next Steps

- Implement full density-based SPH model (in progress)
- Replace heuristic cohesion with pressure-based solver
- Improve viscosity formulation

---

## How to Run

```bash
pip install taichi
python FILE_NAME.py
```

---

## Note

The original O(n²) pairwise implementation will remain in the repository for performance comparison and architectural reference.
