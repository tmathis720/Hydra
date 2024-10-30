# Step-by-Step Guide for Implementing a Cell-Centered Discretization Scheme Using Clean Architecture and TDD

This guide provides a detailed roadmap for implementing a cell-centered discretization scheme in Hydra, following Clean Architecture principles and Test-Driven Development (TDD). We will leverage the existing codebase and modules (`Domain`, `Boundary`, `Geometry`, etc.) to ensure modularity, scalability, and testability.

---

## **Section 1: Architectural Foundation Using Clean Architecture Principles**

Clean Architecture organizes code into layers, each with specific responsibilities. We'll map these layers to our project modules to establish a solid architectural foundation.

### **1.1 Domain Layer**

**Purpose**: Encapsulate the core business logic and data structures, independent of external concerns.

#### **Step 1.1.1: Define Core Entities**

- **`MeshEntity`**: Represent mesh elements (`Vertex`, `Edge`, `Face`, `Cell`).

  ```rust
  // src/domain/mesh_entity.rs
  #[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Clone, Copy)]
  pub enum MeshEntity {
      Vertex(usize),
      Edge(usize),
      Face(usize),
      Cell(usize),
  }
  ```

- **`Arrow`**: Capture directed relationships between `MeshEntity` instances.

  ```rust
  // src/domain/mesh_entity.rs
  pub struct Arrow {
      pub from: MeshEntity,
      pub to: MeshEntity,
  }
  ```

#### **Step 1.1.2: Implement the `Sieve` Structure**

- **Purpose**: Manage relationships between entities.

  ```rust
  // src/domain/sieve.rs
  pub struct Sieve {
      pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
  }
  ```

- **Methods**:

  - `add_arrow(from, to)`: Establish relationships.
  - `cone(point)`, `closure(point)`: Traverse entity relationships.

#### **Step 1.1.3: Create the `Mesh` Struct**

- **Purpose**: Aggregate entities and their relationships.

  ```rust
  // src/domain/mesh/mod.rs
  pub struct Mesh {
      pub sieve: Arc<Sieve>,
      pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
      pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,
      // Boundary data channels
      pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,
      pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,
  }
  ```

#### **Step 1.1.4: Define Boundary Conditions**

- **`BoundaryCondition` Enum**: Represent different boundary condition types.

  ```rust
  // src/boundary/bc_handler.rs
  pub enum BoundaryCondition {
      Dirichlet(f64),
      Neumann(f64),
      Robin { alpha: f64, beta: f64 },
      DirichletFn(BoundaryConditionFn),
      NeumannFn(BoundaryConditionFn),
  }
  ```

- **`BoundaryConditionHandler`**: Manage boundary conditions applied to entities.

  ```rust
  // src/boundary/bc_handler.rs
  pub struct BoundaryConditionHandler {
      conditions: DashMap<MeshEntity, BoundaryCondition>,
  }
  ```

### **1.2 Use Cases Layer**

**Purpose**: Contain application-specific business rules.

#### **Step 1.2.1: Develop Discretization Modules**

- **Convective and Diffusive Flux Calculations**: Implement modules to compute fluxes using cell-centered values.

- **Gradient and Divergence Computations**: Create utility functions for geometric terms.

#### **Step 1.2.2: Implement Time-Stepping and Solver Orchestration**

- **Time-Stepping Controller**: Allow selection of time-stepping schemes at runtime.

- **Solver Integration**: Incorporate Krylov solvers with preconditioner options.

### **1.3 Interface Adapters Layer**

**Purpose**: Convert data from the format most convenient for use cases and entities to the format most convenient for frameworks and drivers.

#### **Step 1.3.1: Data Input/Output**

- **Mesh I/O Adapters**: Implement functions to read/write mesh and field data.

#### **Step 1.3.2: Boundary Condition Mapping**

- **Adapters**: Map boundary condition entities to actionable computational states.

### **1.4 Framework and Drivers Layer**

**Purpose**: Contain details like UI, databases, frameworks, and external interfaces.

#### **Step 1.4.1: Numerical Libraries Integration**

- **`faer` Library**: Utilize for matrix operations and sparse linear algebra.

#### **Step 1.4.2: Parallelization Support**

- **Threading and MPI Readiness**: Design data structures to enable parallelism.

---

## **Section 2: Implementation Steps for Cell-Centered Scheme**

### **2.1 Mesh and Geometry Representation**

#### **Step 2.1.1: Implement Mesh Data Structures**

- **Define `Mesh`**: Use the existing `Mesh` struct in `src/domain/mesh/mod.rs`.

- **Utility Functions**: Implement methods to compute volumes, areas, centroids.

  ```rust
  // src/domain/mesh/geometry.rs
  impl Mesh {
      pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
          // Compute centroid
      }
      pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
          // Compute face area
      }
  }
  ```

#### **Step 2.1.2: Leverage the `Geometry` Module**

- **Geometric Calculations**: Use `src/geometry` for shape representations.

### **2.2 Flux Calculation Modules**

#### **Step 2.2.1: Convective Fluxes**

- **Upwind Schemes**: Implement upwind discretization for convective terms.

  ```rust
  // src/solver/convective_flux.rs (new file)
  pub fn compute_convective_flux(...) {
      // Implement upwind scheme
  }
  ```

#### **Step 2.2.2: Diffusive Fluxes**

- **Central Differencing**: Use central differences for diffusive terms.

  ```rust
  // src/solver/diffusive_flux.rs (new file)
  pub fn compute_diffusive_flux(...) {
      // Implement central differencing
  }
  ```

### **2.3 Time-Stepping Framework**

#### **Step 2.3.1: Implement Time-Stepping Schemes**

- **Explicit Methods**: Implement Runge-Kutta methods in `src/time_stepping/methods/runge_kutta.rs`.

- **Implicit Methods**: Implement Crank-Nicolson in `src/time_stepping/methods/crank_nicolson.rs`.

#### **Step 2.3.2: Controller for Time-Stepping**

- **Selection Logic**: Allow runtime selection of schemes.

  ```rust
  // src/time_stepping/ts.rs
  pub enum TimeSteppingMethod {
      ExplicitEuler,
      RungeKutta,
      CrankNicolson,
  }

  pub struct TimeStepper {
      method: TimeSteppingMethod,
      // Additional fields
  }

  impl TimeStepper {
      pub fn step(&self, ...) {
          match self.method {
              TimeSteppingMethod::ExplicitEuler => { /* ... */ },
              TimeSteppingMethod::RungeKutta => { /* ... */ },
              TimeSteppingMethod::CrankNicolson => { /* ... */ },
          }
      }
  }
  ```

### **2.4 Boundary Condition Handling**

#### **Step 2.4.1: Integrate Boundary Conditions**

- **Apply BCs in Solvers**: Use `BoundaryConditionHandler` to apply conditions during assembly.

  ```rust
  // In solver assembly code
  let bc_handler = BoundaryConditionHandler::new();
  bc_handler.apply_bc(&mut matrix, &mut rhs, &boundary_entities, &entity_to_index, time);
  ```

---

## **Section 3: Test-Driven Development and Testing Strategy**

### **3.1 Unit Tests**

#### **Step 3.1.1: Test Core Components**

- **`MeshEntity` and `Arrow`**: Verify entity creation and relationships.

  ```rust
  // src/domain/mesh_entity.rs
  #[cfg(test)]
  mod tests {
      #[test]
      fn test_mesh_entity_creation() {
          // Test code
      }
  }
  ```

#### **Step 3.1.2: Test Flux Calculations**

- **Convective Flux Module**: Validate upwind schemes.

  ```rust
  // src/solver/convective_flux.rs
  #[cfg(test)]
  mod tests {
      #[test]
      fn test_convective_flux_upwind() {
          // Test code
      }
  }
  ```

### **3.2 Integration Tests**

#### **Step 3.2.1: Simulate Standard Test Cases**

- **Lid-Driven Cavity Flow**: Implement test to simulate this case.

  ```rust
  // src/tests/lid_driven_cavity.rs
  #[cfg(test)]
  mod tests {
      #[test]
      fn test_lid_driven_cavity_flow() {
          // Test code
      }
  }
  ```

### **3.3 Performance and Profiling Tests**

#### **Step 3.3.1: Benchmark Solver Performance**

- **Scalability Tests**: Measure performance with increasing mesh sizes.

  ```rust
  // src/tests/performance_tests.rs
  #[cfg(test)]
  mod tests {
      #[bench]
      fn bench_solver_performance(b: &mut Bencher) {
          // Benchmark code
      }
  }
  ```

---

# Detailed Implementation Steps

Below, we provide a detailed breakdown of each step, including code snippets and explanations to guide the implementation.

## **1. Architectural Foundation**

### **1.1 Domain Layer**

#### **1.1.1 Implementing Core Entities**

- **`MeshEntity` and `Arrow`**: These are fundamental for representing mesh elements and their relationships.

  ```rust
  // src/domain/mesh_entity.rs
  #[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Clone, Copy)]
  pub enum MeshEntity {
      Vertex(usize),
      Edge(usize),
      Face(usize),
      Cell(usize),
  }

  pub struct Arrow {
      pub from: MeshEntity,
      pub to: MeshEntity,
  }
  ```

- **Usage**: These entities are used throughout the mesh management and solver routines.

#### **1.1.2 Implementing `Sieve` for Relationship Management**

- **Purpose**: Efficiently manage entity relationships.

  ```rust
  // src/domain/sieve.rs
  pub struct Sieve {
      pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
  }
  ```

- **Key Methods**:

  - `add_arrow(from, to)`: Add a directed relationship.
  - `cone(entity)`: Get entities directly connected.
  - `closure(entity)`: Get all connected entities recursively.

#### **1.1.3 Creating the `Mesh` Struct**

- **Combines Entities and Relationships**.

  ```rust
  // src/domain/mesh/mod.rs
  pub struct Mesh {
      pub sieve: Arc<Sieve>,
      pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,
      pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,
      // Boundary data channels
      pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,
      pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,
  }
  ```

- **Methods**:

  - `add_entity(entity)`: Add entities to the mesh.
  - `add_relationship(from, to)`: Define relationships.

#### **1.1.4 Defining Boundary Conditions**

- **`BoundaryCondition` Enum**: Supports various boundary types.

  ```rust
  // src/boundary/bc_handler.rs
  pub enum BoundaryCondition {
      Dirichlet(f64),
      Neumann(f64),
      Robin { alpha: f64, beta: f64 },
      // Function-based conditions
      DirichletFn(BoundaryConditionFn),
      NeumannFn(BoundaryConditionFn),
  }
  ```

- **`BoundaryConditionHandler`**: Manages the application of BCs.

  ```rust
  // src/boundary/bc_handler.rs
  pub struct BoundaryConditionHandler {
      conditions: DashMap<MeshEntity, BoundaryCondition>,
  }
  ```

### **1.2 Use Cases Layer**

#### **1.2.1 Developing Discretization Modules**

- **Create Modules for Flux Calculations**:

  - **Convective Fluxes**: Implement upwind schemes.

    ```rust
    // src/solver/convective_flux.rs
    pub fn compute_convective_flux(mesh: &Mesh, field: &Section<f64>) {
        // Implementation
    }
    ```

  - **Diffusive Fluxes**: Implement central differencing.

    ```rust
    // src/solver/diffusive_flux.rs
    pub fn compute_diffusive_flux(mesh: &Mesh, field: &Section<f64>) {
        // Implementation
    }
    ```

#### **1.2.2 Implementing Time-Stepping and Solver Orchestration**

- **Time-Stepping Controller**:

  ```rust
  // src/time_stepping/ts.rs
  pub struct TimeStepper {
      method: TimeSteppingMethod,
  }

  impl TimeStepper {
      pub fn step(&self, ...) {
          // Call appropriate method
      }
  }
  ```

- **Solver Integration**:

  ```rust
  // src/solver/mod.rs
  pub fn solve_system(matrix: &Mat<f64>, rhs: &Mat<f64>, preconditioner: Option<&Preconditioner>) -> Vec<f64> {
      // Solver implementation
  }
  ```

### **1.3 Interface Adapters Layer**

#### **1.3.1 Data Input/Output**

- **Mesh I/O**:

  ```rust
  // src/input_output/gmsh_parser.rs
  pub fn read_mesh(file_path: &str) -> Mesh {
      // Parse Gmsh file
  }
  ```

#### **1.3.2 Boundary Condition Mapping**

- **Adapters to Convert Boundary Data**:

  ```rust
  // src/boundary/mod.rs
  pub fn apply_boundary_conditions(mesh: &Mesh, bc_handler: &BoundaryConditionHandler) {
      // Map and apply BCs
  }
  ```

### **1.4 Framework and Drivers Layer**

#### **1.4.1 Integrating Numerical Libraries**

- **Using `faer` for Linear Algebra**:

  ```rust
  // src/linalg/matrix/mat_impl.rs
  use faer::Mat;

  pub struct Matrix {
      data: Mat<f64>,
  }
  ```

#### **1.4.2 Preparing for Parallelization**

- **Design Data Structures for Concurrency**:

  - Use thread-safe structures like `DashMap` and `Arc<RwLock<...>>`.

  ```rust
  // Example in src/domain/sieve.rs
  pub struct Sieve {
      pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
  }
  ```

---

## **2. Implementation Steps for Cell-Centered Scheme**

### **2.1 Mesh and Geometry Representation**

#### **2.1.1 Implement Mesh Data Structures**

- **Leverage Existing `Mesh` and `MeshEntity`**.

#### **2.1.2 Implement Utility Functions**

- **Compute Geometric Quantities**:

  ```rust
  // src/domain/mesh/geometry.rs
  impl Mesh {
      pub fn compute_cell_volume(&self, cell: &MeshEntity) -> f64 {
          // Compute volume
      }
  }
  ```

### **2.2 Flux Calculation Modules**

#### **2.2.1 Implement Convective Fluxes**

- **Upwind Scheme Implementation**:

  ```rust
  // src/solver/convective_flux.rs
  pub fn compute_convective_flux(mesh: &Mesh, field: &Section<f64>, fluxes: &mut Section<f64>) {
      // Loop over faces and compute fluxes
  }
  ```

#### **2.2.2 Implement Diffusive Fluxes**

- **Central Differencing Implementation**:

  ```rust
  // src/solver/diffusive_flux.rs
  pub fn compute_diffusive_flux(mesh: &Mesh, field: &Section<f64>, fluxes: &mut Section<f64>) {
      // Compute gradients and fluxes
  }
  ```

### **2.3 Time-Stepping Framework**

#### **2.3.1 Implement Time-Stepping Methods**

- **Explicit Methods**:

  ```rust
  // src/time_stepping/methods/runge_kutta.rs
  pub fn runge_kutta_step(...) {
      // Implement RK method
  }
  ```

- **Implicit Methods**:

  ```rust
  // src/time_stepping/methods/crank_nicolson.rs
  pub fn crank_nicolson_step(...) {
      // Implement CN method
  }
  ```

#### **2.3.2 Implement Time-Stepping Controller**

- **Select and Invoke Methods**.

### **2.4 Boundary Condition Handling**

#### **2.4.1 Apply Boundary Conditions During Assembly**

- **In Flux Computations**: Adjust fluxes at boundaries according to BCs.

  ```rust
  // src/solver/fluxes.rs
  pub fn apply_boundary_conditions(...) {
      // Modify fluxes based on BCs
  }
  ```

---

## **3. Test-Driven Development and Testing Strategy**

### **3.1 Unit Tests**

#### **3.1.1 Testing Core Components**

- **`MeshEntity` Tests**:

  ```rust
  // src/domain/mesh_entity.rs
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_mesh_entity_creation() {
          let vertex = MeshEntity::Vertex(1);
          assert_eq!(vertex.id(), 1);
          assert_eq!(vertex.entity_type(), "Vertex");
      }
  }
  ```

#### **3.1.2 Testing Flux Calculations**

- **Convective Flux Tests**:

  ```rust
  // src/solver/convective_flux.rs
  #[cfg(test)]
  mod tests {
      #[test]
      fn test_upwind_flux_computation() {
          // Set up simple mesh and field
          // Compute fluxes
          // Assert correctness
      }
  }
  ```

### **3.2 Integration Tests**

#### **3.2.1 Simulating Lid-Driven Cavity Flow**

- **Test Setup**:

  ```rust
  // src/tests/lid_driven_cavity.rs
  #[cfg(test)]
  mod tests {
      #[test]
      fn test_lid_driven_cavity_flow() {
          // Initialize mesh
          // Set boundary conditions
          // Run simulation
          // Validate results against benchmark
      }
  }
  ```

### **3.3 Performance and Profiling Tests**

#### **3.3.1 Benchmarking Solver Performance**

- **Using Criterion or Similar**:

  ```rust
  // src/tests/performance_tests.rs
  #[cfg(test)]
  mod tests {
      use criterion::{criterion_group, criterion_main, Criterion};

      fn bench_solver(c: &mut Criterion) {
          c.bench_function("solver_performance", |b| {
              b.iter(|| {
                  // Setup and run solver
              });
          });
      }

      criterion_group!(benches, bench_solver);
      criterion_main!(benches);
  }
  ```

---

# Conclusion

By following this guide, you can systematically implement a cell-centered discretization scheme in Hydra, ensuring that your codebase remains modular, scalable, and testable. Leveraging Clean Architecture principles separates concerns across layers, while TDD ensures that each component is thoroughly tested, leading to a robust and maintainable codebase.

---

**Note**: This guide assumes familiarity with Rust programming, numerical methods for PDEs, and software architecture principles. Adjustments may be necessary based on specific project requirements and existing codebase nuances.