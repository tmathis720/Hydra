# **Hydra Integration Guide**

---

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Hydra Architecture Overview](#2-hydra-architecture-overview)  
3. [Typical Workflow](#3-typical-workflow)  
   1. [Building or Loading a Mesh (Domain + I/O + Extrusion)](#31-building-or-loading-a-mesh-domain--io--extrusion)  
   2. [Assigning Boundary Conditions (Boundary Module)](#32-assigning-boundary-conditions-boundary-module)  
   3. [Defining Geometry Calculations (Geometry Module)](#33-defining-geometry-calculations-geometry-module)  
   4. [Setting Up Equations and Fields (Equation Module)](#34-setting-up-equations-and-fields-equation-module)  
   5. [Preparing Linear Algebra Components (Linalg Module)](#35-preparing-linear-algebra-components-linalg-module)  
   6. [Selecting/Configuring a Solver (Solver Module)](#36-selectingconfiguring-a-solver-solver-module)  
   7. [Choosing a Time Stepping Method (Time Stepping Module)](#37-choosing-a-time-stepping-method-time-stepping-module)  
   8. [Integrating in a Use Case (Use Cases Module)](#38-integrating-in-a-use-case-use-cases-module)  
   9. [Running the Simulation Loop](#39-running-the-simulation-loop)  
4. [Interface Adapters in Practice](#4-interface-adapters-in-practice)  
5. [Example End-to-End Snippet](#5-example-end-to-end-snippet)  
6. [Best Practices](#6-best-practices)  
7. [Conclusion](#7-conclusion)

---

## **1. Introduction**

Hydra is a **modular** framework for building numerical simulations, particularly PDE-based and HPC/CFD applications. Each Hydra **module** focuses on one aspect:

- **Domain**: Manages mesh entities, connectivity, and data “sections.”
- **Boundary**: Attaches and applies boundary conditions to domain entities.
- **Geometry**: Computes geometric properties (area, volume, normals, centroids).
- **Linear Algebra**: Vectors, matrices, and operations required for iterative solvers.
- **Solver**: Krylov subspace solvers (CG, GMRES) and preconditioners (Jacobi, ILU, etc.).
- **Time Stepping**: Evolve time-dependent problems via Euler, Runge-Kutta, or other schemes.
- **Equation**: Defines physical PDEs (momentum, energy, turbulence, etc.) that assemble fluxes.
- **Interface Adapters**: Bridges Hydra’s data with external or typical linear algebra structures or domain-building approaches.
- **Extrusion**: Converts a 2D mesh into a 3D mesh by layering (quad → hex, tri → prism).
- **Use Cases**: High-level workflows for matrix construction, PISO solver patterns, etc.
- **Input/Output**: Reading and writing files for meshes (Gmsh) and matrices (MatrixMarket), plus generation of standard shapes.

This integration guide shows how these pieces **fit together**.

---

## **2. Hydra Architecture Overview**

A typical Hydra-based simulation might:

1. **Obtain** or **build** a domain mesh (2D or 3D).  
2. **Specify** boundary conditions on faces or cells.  
3. **Compute** geometry data (centroids, volumes, etc.).  
4. **Define** one or more PDE equations (momentum, energy, turbulence).  
5. **Prepare** linear algebra objects for the system \(A x = b\).  
6. **Choose** an iterative solver (e.g., GMRES) and optional preconditioner.  
7. **Pick** a time-stepping approach if time-dependent.  
8. **Tie** them together in a main loop or **EquationManager** + **TimeStepper** approach.  
9. **Run** the simulation, writing outputs or building solutions.

**Cross-cutting** modules like **Interface Adapters** or **Use Cases** fill in the gaps (e.g., reading/writing external data, building standard domain shapes, or providing a high-level approach like the PISO loop).

---

## **3. Typical Workflow**

Below is a step-by-step outline of how one might combine Hydra modules to run a simple PDE simulation.

### **3.1 Building or Loading a Mesh (Domain + I/O + Extrusion)**

1. **Load a Mesh** from a file:
   ```rust
   use hydra::input_output::gmsh_parser::GmshParser;
   let mesh = GmshParser::from_gmsh_file("my_mesh.msh")?;
   ```
   Or **Generate** one programmatically (e.g., a 2D rectangle):
   ```rust
   use hydra::input_output::mesh_generation::MeshGenerator;
   let mesh_2d = MeshGenerator::generate_rectangle_2d(10.0, 5.0, 10, 5);
   ```
2. **Extrude** if your simulation is 3D but you only have a 2D mesh:
   ```rust
   use hydra::extrusion::interface_adapters::extrusion_service::ExtrusionService;
   let extruded_mesh = ExtrusionService::extrude_mesh(&mesh_2d, /*depth=*/ 5.0, /*layers=*/ 4)?;
   ```
   Now `extruded_mesh` is a Hydra `Mesh` with 3D cells.

3. **Manipulate** or reorder the domain if needed (domain `apply_reordering(...)`, etc.).

### **3.2 Assigning Boundary Conditions (Boundary Module)**

1. **Obtain** or create a `BoundaryConditionHandler`:
   ```rust
   use hydra::boundary::bc_handler::BoundaryConditionHandler;
   let mut bc_handler = BoundaryConditionHandler::new();
   ```
2. **Set** BCs for relevant faces in the mesh:
   ```rust
   use hydra::domain::MeshEntity;
   use hydra::boundary::bc_handler::BoundaryCondition;

   let face = MeshEntity::Face(10);
   bc_handler.set_bc(face, BoundaryCondition::Dirichlet(1.0));
   ```
3. This BC handler is passed to PDE equations or managers, so they can **apply** conditions to system matrices or flux calculations.

### **3.3 Defining Geometry Calculations (Geometry Module)**

1. Create a `Geometry` object to store or compute geometric info:
   ```rust
   use hydra::geometry::Geometry;
   let mut geometry = Geometry::new();
   ```
2. If you want to compute, say, the area or centroid of faces/cells:
   ```rust
   let face_area = geometry.compute_face_area(face_id, face_shape, &face_vertices);
   let cell_volume = geometry.compute_cell_volume(&mesh, &cell_entity);
   ```
3. The domain module’s `Mesh` can be used to retrieve vertex indices and pass them to `geometry` as needed. This ensures PDE flux calculations have correct geometry.

### **3.4 Setting Up Equations and Fields (Equation Module)**

1. Create a `Fields` object for your unknowns (e.g., “velocity,” “pressure,” “energy,” etc.):
   ```rust
   use hydra::equation::fields::Fields;
   let mut fields = Fields::new();
   // Insert scalar or vector fields:
   fields.set_scalar_field_value("pressure", cell_entity, 0.0);
   fields.set_vector_field_value("velocity", cell_entity, Vector3([0.0, 0.0, 0.0]));
   ```
2. Define or use a `PhysicalEquation`:
   ```rust
   use hydra::equation::momentum_equation::MomentumEquation;
   let momentum_eq = MomentumEquation::new();
   ```
3. An `EquationManager` can store multiple equations:
   ```rust
   use hydra::equation::manager::EquationManager;
   let mut eq_manager = EquationManager::new(... time_stepper..., mesh, bc_handler);
   eq_manager.add_equation(momentum_eq);
   // add other eqns, e.g., EnergyEquation
   ```

### **3.5 Preparing Linear Algebra Components (Linalg Module)**

If your PDE solve requires building or manipulating a system matrix:

1. Use `MatrixBuilder` or `VectorBuilder` to create your data:
   ```rust
   use hydra::linalg::matrix::matrix_builder::MatrixBuilder;
   let mat = MatrixBuilder::build_dense_matrix(100, 100);
   // or build_matrix::<SparseMatrix>...
   ```
2. Similarly, build an RHS vector:
   ```rust
   use hydra::linalg::vector::vector_builder::VectorBuilder;
   let rhs = VectorBuilder::build_dense_vector(100);
   ```
3. Fill them with PDE assembly results or from `fields/fluxes`.

### **3.6 Selecting/Configuring a Solver (Solver Module)**

1. Choose between **CG** for SPD or **GMRES** for general systems:
   ```rust
   use hydra::solver::{cg::ConjugateGradient, gmres::GMRES};
   let mut solver = ConjugateGradient::new(1000, 1e-8);
   // or
   let mut solver = GMRES::new(1000, 1e-6, 50);
   ```
2. Attach a **preconditioner**:
   ```rust
   use hydra::solver::preconditioner::{Jacobi, PreconditionerFactory};
   solver.set_preconditioner(Box::new(Jacobi::default()));
   // or
   solver.set_preconditioner(PreconditionerFactory::create_lu(&matrix));
   ```

### **3.7 Choosing a Time Stepping Method (Time Stepping Module)**

1. If the problem is time-dependent, define a `TimeStepper`:
   ```rust
   use hydra::time_stepping::methods::euler::ExplicitEuler;
   let mut stepper = ExplicitEuler::new(/*time_step=*/ 0.01, /*start_time=*/ 0.0, /*end_time=*/ 1.0);
   // or BackwardEuler, etc.
   ```
2. The `EquationManager` can implement `TimeDependentProblem`, letting the stepper call `compute_rhs(...)`.

### **3.8 Integrating in a Use Case (Use Cases Module)**

1. For building a matrix or RHS systematically, use `matrix_construction` or `rhs_construction`:
   ```rust
   use hydra::use_cases::matrix_construction::MatrixConstruction;
   let mut mat = MatrixConstruction::build_zero_matrix(10, 10);
   MatrixConstruction::initialize_matrix_with_value(&mut mat, 0.0);
   ```
2. If you use the **PISO** approach:
   - The `piso::PISOSolver` or `piso::nonlinear_loop::solve_nonlinear_system` can orchestrate the predictor–pressure–velocity correction steps.

### **3.9 Running the Simulation Loop**

**Pseudocode**:

```rust
// A. Initialize fields, set up domain/mesh, eq_manager, bc_handler, time_stepper, etc.
// B. For each time step:
for step in 0..num_steps {
    // 1. eq_manager.step(&mut fields) // if eq_manager is your TimeDependentProblem
    // or call stepper.step(...) with eq_manager as the problem
    // 2. Possibly output some fields or intermediate data
}
// C. Final data output or post-processing
```

If you’re using PISO in a “nonlinear_loop” style, the manager or solver calls the sub-steps for you.

---

## **4. Interface Adapters in Practice**

If you have to:

1. Convert a Hydra `Section<Scalar>` to a **dense** vector for an external library, use `SectionMatVecAdapter`.  
2. Build a domain programmatically with `DomainBuilder` from `interface_adapters/domain_adapter.rs`.  
3. Or solve a system read from a MatrixMarket file with `SystemSolver::solve_from_file_with_solver(...)`.

This layer ensures Hydra’s PDE approach can talk to standard HPC or external formats.

---

## **5. Example End-to-End Snippet**

Below is a **truncated** version showing how these steps might appear in code (error handling omitted for brevity):

```rust
use hydra::{
    domain::mesh::Mesh,
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
    geometry::Geometry,
    equation::{fields::Fields, manager::EquationManager, momentum_equation::MomentumEquation},
    solver::{cg::ConjugateGradient, preconditioner::Jacobi, ksp::KSP},
    time_stepping::{methods::euler::ExplicitEuler, TimeStepper},
    use_cases::matrix_construction::MatrixConstruction,
    input_output::gmsh_parser::GmshParser,
};

fn main() {
    // 1. Load a mesh from Gmsh
    let mesh: Mesh = GmshParser::from_gmsh_file("my2dmesh.msh").unwrap();

    // 2. Create boundary handler and assign BCs
    let mut bc_handler = BoundaryConditionHandler::new();
    bc_handler.set_bc(MeshEntity::Face(0), BoundaryCondition::Dirichlet(1.0));

    // 3. Optionally extrude if 3D needed, or keep 2D

    // 4. Prepare geometry if needed
    let mut geometry = Geometry::new(); // e.g., compute face areas, cell centroids

    // 5. Setup fields and an equation
    let mut fields = Fields::new();
    let eq = MomentumEquation::new();
    // Add initial velocity, pressure, etc. to fields

    // 6. Choose a solver (Conjugate Gradient + Jacobi preconditioner)
    let mut cg_solver = ConjugateGradient::new(1000, 1e-6);
    cg_solver.set_preconditioner(Box::new(Jacobi::default()));

    // 7. Time stepping method (explicit Euler)
    let mut time_stepper = ExplicitEuler::new(0.01, 0.0, 1.0);

    // 8. EquationManager for PDE integration
    let domain = std::sync::Arc::new(std::sync::RwLock::new(mesh));
    let bc_handler_arc = std::sync::Arc::new(std::sync::RwLock::new(bc_handler));
    let mut eq_manager = EquationManager::new(
        Box::new(time_stepper),
        domain,
        bc_handler_arc,
    );
    eq_manager.add_equation(eq);

    // 9. Main loop
    let mut time = 0.0;
    while time < 1.0 {
        eq_manager.step(&mut fields);  // internally uses eq_manager as TimeDependentProblem
        time += eq_manager.time_stepper.get_time_step();
    }

    // Done: fields now hold the final solution state
}
```

---

## **6. Best Practices**

1. **Modularization**: Keep PDE logic in `equation::` submodules. Keep mesh logic in `domain::mesh`. Connect them via `EquationManager`.  
2. **One Source** for BC**: Use `BoundaryConditionHandler` to store all BC info.  
3. **Performance**: 
   - Reorder mesh with domain’s reordering or use “Cuthill-McKee.”  
   - Consider advanced preconditioners (ILU, AMG).  
   - Use parallel methods from linalg or domain where possible.  
4. **Testing**: 
   - Start with small or generated meshes.  
   - Validate boundary conditions or PDE steps individually.  
5. **Keep It Consistent**: 
   - Use the same indexing for matrix, vectors, and `Section` data.  
   - In 3D, ensure geometry (normals, areas) is correct for the cells/faces.

---

## **7. Conclusion**

This **integration guide** demonstrates how to orchestrate all Hydra modules in a typical simulation pipeline:

- **Domain** + **I/O** + **Extrusion** → build or import a mesh (2D or 3D).  
- **Boundary** → set boundary conditions for PDE.  
- **Geometry** → compute shape-based properties.  
- **Equation** → define physical PDE fluxes.  
- **Linalg** + **Solver** → handle big linear solves.  
- **Time Stepping** → evolve the problem in time.  
- **Interface Adapters** → connect Hydra’s data with external formats or standard HPC approaches.  
- **Use Cases** → incorporate higher-level patterns like matrix construction or PISO.

By combining the modules in the right sequence, Hydra users can build robust, parallel PDE simulations that read or generate domains, apply boundary conditions, compute geometry, run advanced solvers, and manage solution updates over time. Each piece remains **modular**, letting you swap solvers, refine meshes, or add new PDEs without breaking the rest of the workflow.