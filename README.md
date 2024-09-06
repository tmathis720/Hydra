# HYDRA: A Modular Finite Volume Solver in Rust

**HYDRA** is a Rust-based project designed to provide a flexible, modular framework for solving partial differential equations (PDEs) using finite volume methods (FVM). It emphasizes clean architecture, test-driven development (TDD), and extensibility, inspired by PETSc and other scientific computing frameworks.

## Features

- Modular Design: HYDRA is structured to support easy extension and customization of solvers, mesh handling, and numerical methods.
- Mesh Handling: The framework supports parsing and handling 2D triangular meshes from Gmsh (.msh2 format), with automatic area and neighbor relationship calculations.
- Time-Stepping Methods: A flexible interface for integrating various time-stepping methods, including explicit and implicit schemes (e.g., Explicit Euler).
- Linear Solver Integration: Built-in linear solver for solving fluxes between elements using customizable flux calculation methods.
- Transport Solver: A module for calculating transport fluxes across mesh elements, useful for simulating physical phenomena like heat or mass transport.

## Getting Started

### Prerequisites

- Rust: You need to have Rust installed. To install Rust, visit rust-lang.org.
- Cargo: Cargo is the Rust package manager, installed with Rust. Ensure that Cargo is available by running cargo --version.

### Installation

1. Clone the repository:

''' bash

git clone https://github.com/your_username/hydra.git

'''

2. Change to the project directory:

''' bash

cd hydra

'''

3. Build the project:

''' bash

cargo build

'''

4. Run tests to verify the setup:

''' bash

cargo test

'''

### Usage

HYDRA can load 2D triangular meshes from a '.msh2' file and use the finite volume method to compute element fluxes and solve PDEs.

''' bash

cargo run --release

'''

For example, after loading a mesh:

''' rust

use mesh_mod::mesh_ops::Mesh;
use solvers_mod::linear::LinearSolver;
use time_stepping_mod::explicit_euler::ExplicitEuler;

fn main() {
    let mesh_file = "inputs/test.msh2";
    let mut mesh = Mesh::load_from_gmsh(mesh_file).unwrap();

    let mut solver = LinearSolver::new(mesh);
    let mut time_stepper = ExplicitEuler::new(0.01);

    for _ in 0..100 {
        time_stepper.step(&mut solver, 0.01);
    }
}

'''

### File Structure

- src/mesh_mod: Contains all mesh-related operations, including mesh loading, node and element handling, and neighbor relations.
- src/solvers_mod: Houses linear solver implementations for solving PDEs using flux calculations.
- src/transport_mod: Contains modules for transport flux calculations across mesh elements.
- src/time_stepping_mod: Modular time-stepping methods such as explicit Euler and (eventually) implicit schemes.
- src/numerical_mod: Numerical routines such as linear algebra operations and geometry functions.

### Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature).
5. Open a Pull Request.

### Testing

HYDRA uses cargo test for unit and integration tests. Make sure to run tests before submitting any changes:

''' bash

cargo test

'''

We follow a test-driven development (TDD) approach, and tests are placed alongside the respective modules.

### Future Roadmap

- Implement implicit time-stepping schemes.
- Add support for 3D meshes and more complex geometries.
- Incorporate boundary conditions (i.e., flow, pressure gradient, water level).
- Incorporate GPU acceleration for larger-scale problems.
- Enhance solver flexibility to handle nonlinear PDEs.

### License

HYDRA is licensed under the MIT License. See LICENSE for more details.
