We are in the process of implementing the following roadmap for Hydra. 

1) There are several problems with the current source and we want to implement as much of the skeleton of the below roadmap in advance, and then develop and test it as we go. 
2) Please use the source code included after this roadmap as a starting point, and then analyze the problem, 
3) Develop a comprehensive solution in the form of complete source code with documentation and test cases, storing these outputs in your memory for this conversation.
4) Provide a narrative output of the enhancements to each file, appropriately summarizing the changes in terms of a detailed itemized list of changes with sufficient details to remind us of the complete context of the comprehensive solution we developed and stored in memory for Step 3 of these instructions.

The roadmaps is as follows:



# Roadmap for Enhancing the `Equation` Module in Hydra

---

## **Introduction**

This roadmap outlines a comprehensive plan to address the weaknesses identified in the `Equation` module of the Hydra framework. The goal is to enhance the module's utility for coding and implementing complex boundary-fitted geophysical hydrodynamic models of environmental-scale natural systems, such as lakes, reservoirs, coastal environments, and oceans. The roadmap provides detailed tasks and implementation strategies, leveraging the capabilities of other Hydra modules like `Domain`, `Boundary`, `Geometry`, `Solver`, `Linear Algebra`, and `Time Stepping`.

---

## **1. Complete Key Implementations**

### **1.1. Develop the Momentum Equation**

**Objective**: Fully implement the momentum equation to simulate fluid motion, including support for vector fields and appropriate numerical schemes.

#### **Tasks**

1. **Define the MomentumEquation Struct**:

   - **Create `momentum_equation.rs`**: Develop the file to house the momentum equation implementation.
   - **Struct Definition**:
     ```rust
     pub struct MomentumEquation {
         pub density: f64,
         pub viscosity: f64,
         // Additional physical parameters
     }
     ```

2. **Implement PhysicalEquation for MomentumEquation**:

   - **Assemble Method**:
     ```rust
     impl PhysicalEquation for MomentumEquation {
         fn assemble(
             &self,
             domain: &Mesh,
             fields: &Fields,
             fluxes: &mut Fluxes,
             boundary_handler: &BoundaryConditionHandler,
         ) {
             self.calculate_momentum_fluxes(domain, fields, fluxes, boundary_handler);
         }
     }
     ```

3. **Calculate Momentum Fluxes**:

   - **Implement `calculate_momentum_fluxes` Method**:
     - Compute convective and diffusive fluxes for momentum.
     - Use `velocity_field` and `pressure_field` from `Fields`.
     - Handle vector quantities appropriately.

   - **Integration with Geometry Module**:
     - Use `Geometry` for face normals, areas, and centroids.
     - Ensure correct orientation of vectors.

4. **Support Vector Fields in Fields Struct**:

   - **Modify `Fields` Struct**:
     ```rust
     pub struct Fields {
         pub velocity_field: Section<[f64; 3]>,
         pub pressure_field: Section<f64>,
         pub velocity_gradient: Section<[[f64; 3]; 3]>, // Gradient of velocity
         // Other fields...
     }
     ```
     - Store velocity gradients as tensors for accurate calculations.

5. **Implement Boundary Conditions for Momentum**:

   - **Extend `BoundaryCondition` Enum**:
     - Include types specific to momentum, such as wall functions, slip/no-slip conditions.

   - **Apply Boundary Conditions**:
     - Modify `calculate_momentum_fluxes` to apply appropriate boundary conditions at walls and inlets/outlets.

#### **Implementation Details**

- **Numerical Schemes**:
  - Use appropriate discretization schemes (e.g., upwind, central difference) for convective and diffusive terms.
  - Incorporate flux limiters to prevent numerical oscillations.

- **Parallel Computation**:
  - Leverage parallel iterators (e.g., Rayon) when looping over faces or cells to enhance performance.

---

### **1.2. Enhance Turbulence Models**

**Objective**: Expand turbulence modeling capabilities, ensuring compatibility with the momentum equation.

#### **Tasks**

1. **Implement Additional Turbulence Models**:

   - **Extend `turbulence_models.rs`**:
     - Implement models like k-ω, Reynolds Stress Models (RSM), Large Eddy Simulation (LES).

2. **Couple Turbulence Models with Momentum Equation**:

   - **Modify MomentumEquation**:
     - Include turbulence effects in momentum flux calculations.
     - Use turbulence quantities (e.g., turbulent viscosity) from `Fields`.

3. **Update Fields Struct for Turbulence Quantities**:

   - **Add Turbulence Fields**:
     ```rust
     pub struct Fields {
         // Existing fields...
         pub turbulent_viscosity: Section<f64>,
         pub k_field: Section<f64>, // Turbulent kinetic energy
         pub omega_field: Section<f64>, // Specific dissipation rate
         // Other turbulence-related fields...
     }
     ```

4. **Implement Turbulence Model Assembly**:

   - **Implement PhysicalEquation for Each Model**:
     - Ensure each turbulence model computes necessary quantities and updates `Fields` accordingly.

5. **Boundary Conditions for Turbulence Models**:

   - **Extend Boundary Handling**:
     - Implement boundary conditions specific to turbulence quantities (e.g., wall functions).

#### **Implementation Details**

- **Solver Integration**:
  - Since turbulence models often lead to additional equations, integrate with the `Solver` module to handle coupled systems.

- **Validation**:
  - Test turbulence models against benchmark problems to ensure accuracy.

---

## **2. Improve Flexibility and Extensibility**

### **2.1. Refactor Fields Struct**

**Objective**: Enhance the `Fields` struct to allow for flexibility in adding new variables and models.

#### **Tasks**

1. **Use a Map or HashMap for Fields**:

   - **Redefine Fields Struct**:
     ```rust
     use std::collections::HashMap;

     pub struct Fields {
         pub scalar_fields: HashMap<String, Section<f64>>,
         pub vector_fields: HashMap<String, Section<[f64; 3]>>,
         pub tensor_fields: HashMap<String, Section<[[f64; 3]; 3]>>,
     }
     ```

2. **Implement Getter and Setter Methods**:

   - **Encapsulate Field Access**:
     ```rust
     impl Fields {
         pub fn get_scalar_field(&self, name: &str) -> Option<&Section<f64>> { /* ... */ }
         pub fn set_scalar_field(&mut self, name: &str, field: Section<f64>) { /* ... */ }
         // Similar methods for vector and tensor fields
     }
     ```

3. **Update Equations to Use New Fields Structure**:

   - Modify all equations to access fields via the getter methods.

4. **Ensure Type Safety**:

   - Implement checks to ensure correct field types are accessed.

#### **Implementation Details**

- **Dynamic Field Addition**:
  - Allow users to define custom fields required for specific models.
- **Performance Considerations**:
  - Optimize field access to minimize overhead from using HashMaps.

---

### **2.2. Parameterize Equations**

**Objective**: Allow equations to accept parameters dynamically to enhance configurability.

#### **Tasks**

1. **Define Parameter Structs**:

   - Create structs to hold parameters for each equation.
     ```rust
     pub struct MomentumParameters {
         pub density: f64,
         pub viscosity: f64,
         // Additional parameters
     }
     ```

2. **Modify Equation Structs to Accept Parameters**:

   - Pass parameters during equation initialization.
     ```rust
     pub struct MomentumEquation {
         pub params: MomentumParameters,
         // Other fields
     }
     ```

3. **Implement Configuration Parsing**:

   - Use configuration files (e.g., JSON, YAML) or command-line arguments to set parameters.
   - Parse configurations at runtime and initialize equations accordingly.

4. **Update EquationManager**:

   - Allow adding equations with parameters.
     ```rust
     pub fn add_equation<E: PhysicalEquation + 'static>(&mut self, equation: E) { /* ... */ }
     ```

#### **Implementation Details**

- **Default Parameters**:
  - Provide default values for parameters to simplify usage.
- **Validation**:
  - Validate parameter values to ensure physical realism.

---

## **3. Strengthen Integration with Other Modules**

### **3.1. Integrate with the Solver Module**

**Objective**: Incorporate the `Solver` module to solve linear systems arising from implicit discretizations.

#### **Tasks**

1. **Modify Equations to Formulate Linear Systems**:

   - **Assemble System Matrices and Vectors**:
     - Each equation should assemble its contribution to the global system.

2. **Define a Global Assembler**:

   - **Create a `SystemAssembler` Struct**:
     - Collects contributions from all equations.
     - Assembles global matrices and right-hand-side vectors.

3. **Interface with the Solver Module**:

   - **Use the `Solver` Trait**:
     - Implement methods to solve the assembled system using solvers like CG or GMRES.

4. **Implement Implicit Time Integration**:

   - **Modify Time Steppers**:
     - Incorporate solver calls within time-stepping schemes for implicit methods (e.g., Backward Euler).

5. **Optimize Matrix Storage**:

   - Use sparse matrix representations from the `Linear Algebra` module to store large systems efficiently.

#### **Implementation Details**

- **Parallel Assembly**:
  - Use parallelism when assembling the global system to improve performance.
- **Solver Configuration**:
  - Allow users to select solvers and preconditioners based on problem characteristics.

---

### **3.2. Incorporate Time Stepping**

**Objective**: Integrate time-stepping methods to support transient simulations.

#### **Tasks**

1. **Modify Equations for Time Dependence**:

   - Include time derivatives in the equations where necessary.

2. **Implement Time-Dependent Boundary Conditions**:

   - Extend `BoundaryCondition` to handle time-dependent functions.

3. **Integrate with the `Time Stepping` Module**:

   - Use the `TimeStepper` trait to advance the solution in time.
   - Equations should provide methods to compute residuals or updates required by time-stepping schemes.

4. **Support Adaptive Time Stepping**:

   - Implement error estimation and step size control for adaptive methods.

#### **Implementation Details**

- **Data Management**:
  - Store previous time step values as needed for multi-step methods.
- **Stability Considerations**:
  - Ensure numerical schemes are stable for the chosen time step sizes.

---

### **3.3. Enhance Boundary Condition Handling**

**Objective**: Implement full support for all boundary condition types, including Robin and time-dependent conditions.

#### **Tasks**

1. **Extend `BoundaryCondition` Enum**:

   - Add variants for all supported boundary conditions.
     ```rust
     pub enum BoundaryCondition {
         Dirichlet(f64),
         Neumann(f64),
         Robin { alpha: f64, beta: f64 },
         TimeDependentDirichlet(Box<dyn Fn(f64) -> f64>),
         // Other types...
     }
     ```

2. **Update BoundaryConditionHandler**:

   - Handle time-dependent boundary conditions.
   - Provide methods to evaluate boundary conditions at a given time.

3. **Modify Equations to Apply Boundary Conditions Correctly**:

   - Ensure that boundary conditions are applied during flux calculations and assembly.

4. **Implement Boundary Condition Parsing**:

   - Allow users to specify boundary conditions in configuration files.

#### **Implementation Details**

- **Function Pointers and Closures**:
  - Use function pointers or closures to represent time-dependent boundary functions.

- **Performance**:
  - Optimize boundary condition evaluations to minimize overhead.

---

## **4. Enhance Numerical Methods**

### **4.1. Implement Advanced Reconstruction Methods**

**Objective**: Implement higher-order reconstruction methods for improved accuracy.

#### **Tasks**

1. **Add New Reconstruction Methods**:

   - Implement methods like MUSCL, ENO, and WENO schemes.
   - Define a `ReconstructionMethod` trait similar to `GradientMethod`.

2. **Modify Equations to Use Reconstruction Methods**:

   - Allow equations to select the reconstruction method for face value computation.

3. **Parameterize Reconstruction Choices**:

   - Enable users to choose reconstruction methods via configurations.

#### **Implementation Details**

- **Non-linear Limiters**:
  - Incorporate limiters appropriate for higher-order methods to prevent oscillations.

- **Computational Efficiency**:
  - Optimize implementations to balance accuracy and performance.

---

### **4.2. Improve Gradient Calculation**

**Objective**: Introduce adaptive gradient methods that can handle unstructured and complex meshes more robustly.

#### **Tasks**

1. **Implement Additional Gradient Methods**:

   - Include methods like Least Squares Gradient Reconstruction.
   - Extend `GradientCalculationMethod` enum.

2. **Error Estimation**:

   - Implement techniques to estimate gradient errors.
   - Use error estimates for adaptive mesh refinement (if applicable).

3. **Integration with Mesh Module**:

   - Ensure gradient calculations are compatible with unstructured meshes.

#### **Implementation Details**

- **Sparse Matrix Operations**:
  - Use sparse matrices for least squares computations to improve efficiency.

- **Parallelization**:
  - Parallelize gradient calculations over cells.

---

### **4.3. Expand Flux Limiter Options**

**Objective**: Provide more flux limiter choices and allow users to select limiters based on problem requirements.

#### **Tasks**

1. **Implement Additional Flux Limiters**:

   - Include limiters like Van Leer, Barth-Jespersen, and Sweby.

2. **Parameterize Flux Limiter Selection**:

   - Allow users to choose flux limiters via configuration files or runtime parameters.

3. **Modify Equations to Use Selected Flux Limiters**:

   - Pass the selected flux limiter to equations during assembly.

#### **Implementation Details**

- **Testing and Validation**:
  - Validate new limiters against benchmark problems.

- **Documentation**:
  - Provide guidance on selecting appropriate limiters for different scenarios.

---

## **5. Improve Error Handling and Robustness**

### **5.1. Implement Comprehensive Error Checking**

**Objective**: Ensure all functions validate inputs and handle exceptions gracefully.

#### **Tasks**

1. **Input Validation**:

   - Check for invalid inputs (e.g., zero volumes, null references).
   - Use `Result` and `Option` types effectively.

2. **Error Propagation**:

   - Propagate errors up the call stack with meaningful messages.

3. **Implement Logging**:

   - Use a logging library to record errors and warnings.

4. **Graceful Degradation**:

   - Handle non-critical errors without crashing the program.

#### **Implementation Details**

- **Custom Error Types**:
  - Define custom error types for different modules.

- **Testing**:
  - Write tests to ensure error handling works as expected.

---

### **5.2. Develop Comprehensive Test Suites**

**Objective**: Validate all components thoroughly to ensure correctness.

#### **Tasks**

1. **Unit Tests**:

   - Write unit tests for all functions and methods.

2. **Integration Tests**:

   - Test interactions between modules (e.g., equations with solver and time stepping).

3. **Benchmark Problems**:

   - Implement standard benchmark cases (e.g., lid-driven cavity, flow over a flat plate).

4. **Continuous Integration**:

   - Set up CI/CD pipelines to run tests automatically on code changes.

#### **Implementation Details**

- **Code Coverage**:
  - Aim for high code coverage with tests.

- **Performance Testing**:
  - Benchmark performance to identify bottlenecks.

---

## **6. Optimize for Performance and Scalability**

### **6.1. Implement Parallel Computing Support**

**Objective**: Utilize parallel processing to handle large meshes efficiently.

#### **Tasks**

1. **Parallelize Loops**:

   - Use Rayon or other parallel libraries to parallelize loops over cells and faces.

2. **Thread Safety**:

   - Ensure shared data structures are accessed safely (e.g., use synchronization primitives where necessary).

3. **Distributed Computing**:

   - Consider integrating MPI for distributed memory parallelism in large-scale simulations.

#### **Implementation Details**

- **Load Balancing**:
  - Ensure work is evenly distributed across threads or processes.

- **Scalability Testing**:
  - Test the code on large problems to evaluate scalability.

---

### **6.2. Optimize Memory Management**

**Objective**: Reduce memory usage, especially for large-scale simulations.

#### **Tasks**

1. **Efficient Data Structures**:

   - Use appropriate data structures (e.g., sparse matrices, compressed storage formats).

2. **Avoid Unnecessary Copies**:

   - Use references and borrowing to prevent unnecessary data copying.

3. **Memory Profiling**:

   - Profile memory usage to identify and fix leaks or excessive consumption.

#### **Implementation Details**

- **Cache Optimization**:
  - Organize data to improve cache locality.

- **Garbage Collection**:
  - In Rust, ensure proper ownership and lifetimes to prevent memory issues.

---

## **7. Documentation and User Guidance**

### **7.1. Develop Comprehensive Documentation**

**Objective**: Provide detailed documentation for all public interfaces and modules.

#### **Tasks**

1. **API Documentation**:

   - Use Rust doc comments to generate documentation.

2. **User Guides and Tutorials**:

   - Write guides explaining how to use the module, with examples.

3. **Reference Manual**:

   - Document all equations, numerical methods, and parameters.

#### **Implementation Details**

- **Automated Documentation Generation**:
  - Use tools like `rustdoc` to generate and host documentation.

- **Code Examples**:
  - Include code snippets demonstrating common tasks.

---

### **7.2. Enhance User Configurability**

**Objective**: Allow users to configure simulations via input files or scripting interfaces.

#### **Tasks**

1. **Implement Configuration File Parsing**:

   - Support formats like JSON, YAML, or TOML.

2. **Command-Line Interface**:

   - Provide options to set parameters via command-line arguments.

3. **Scripting Support**:

   - Consider integrating with scripting languages (e.g., Python bindings) for flexibility.

#### **Implementation Details**

- **Validation**:
  - Validate configurations and provide meaningful error messages.

- **Defaults and Overrides**:
  - Allow default settings with the option to override specific parameters.

---

## **Conclusion**

By following this roadmap, the `Equation` module can be significantly enhanced to meet the needs of complex geophysical hydrodynamic modeling. The detailed tasks and implementation strategies provided aim to improve the module's functionality, flexibility, performance, and usability. Integrating closely with other Hydra modules and adhering to best practices will result in a robust and efficient simulation framework capable of addressing real-world environmental modeling challenges.

---

**Next Steps**:

- **Prioritize Tasks**: Determine the order in which tasks should be tackled based on project needs and resource availability.
- **Assign Responsibilities**: Allocate tasks to team members with the appropriate expertise.
- **Set Milestones**: Establish timelines for completing each phase of the roadmap.
- **Monitor Progress**: Regularly review progress and adjust the plan as necessary.

By systematically addressing each area of improvement, the Hydra `Equation` module can evolve into a powerful tool for scientists and engineers working on environmental-scale hydrodynamic simulations.


---

Consistent with the instructions, here is the source code to use as a starting point.

`src/equation/mod.rs` :


```rust
use fields::{Fields, Fluxes};

use crate::{
    boundary::bc_handler::BoundaryConditionHandler,
    time_stepping::{TimeDependentProblem, TimeSteppingError},
    Matrix, Mesh, Vector,
};

pub mod equation;
pub mod reconstruction;
pub mod gradient;
pub mod flux_limiter;

pub mod fields;
pub mod manager;
pub mod energy_equation;
/* pub mod turbulence_models; */
pub mod momentum_equation;

pub trait PhysicalEquation<T> {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields<T>,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}

impl<T> TimeDependentProblem for Box<dyn PhysicalEquation<T>> {
    type State = Vec<f64>; // Replace with the actual state type.
    type Time = f64;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Implement based on PhysicalEquation requirements.
        unimplemented!()
    }

    fn initial_state(&self) -> Self::State {
        vec![0.0; 10] // Replace with actual initial state logic.
    }

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
        time
    }

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
        None // Replace with matrix logic if needed.
    }

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        Ok(()) // Replace with solver logic.
    }
}
```

---

`src/equation/equation.rs`

```rust
use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::BoundaryConditionHandler;

pub struct Equation {}

impl Equation {
    pub fn calculate_fluxes(
        &self,
        domain: &Mesh,
        velocity_field: &Section<[f64; 3]>,
        pressure_field: &Section<f64>,
        fluxes: &mut Section<[f64; 3]>,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64, // Accept current_time as a parameter
    ) {
        let _ = pressure_field;
        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);

                let velocity_dot_normal = velocity_field
                    .restrict(&face)
                    .map(|vel| vel.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>())
                    .unwrap_or(0.0);

                let flux = [velocity_dot_normal * area, 0.0, 0.0];
                fluxes.set_data(face.clone(), flux);

                // Boundary condition logic
                let mut matrix = faer::MatMut::default();
                let mut rhs = faer::MatMut::default();
                let boundary_entities = boundary_handler.get_boundary_faces();
                let entity_to_index = domain.get_entity_to_index();

                boundary_handler.apply_bc(
                    &mut matrix,
                    &mut rhs,
                    &boundary_entities,
                    &entity_to_index,
                    current_time, // Pass current_time
                );
            }
        }
    }
}

```

---

`src/equation/fields.rs`

```rust
use crate::{domain::Section, MeshEntity};

pub struct Fields<FieldType> {
    pub velocity_field: Section<[f64; 3]>,
    pub pressure_field: Section<f64>,
    pub velocity_gradient: Section<[[f64; 3]; 3]>,
    pub temperature_field: Section<f64>,
    pub temperature_gradient: Section<[f64; 3]>,
    pub k_field: Section<f64>,
    pub epsilon_field: Section<f64>,
    pub gradient: Section<FieldType>,
    pub field: Section<FieldType>, // Fixed placeholder `_`
}

impl<T> Fields<T> {
    pub fn new() -> Self {
        Self {
            velocity_field: Section::new(),
            pressure_field: Section::new(),
            velocity_gradient: Section::new(),
            temperature_field: Section::new(),
            temperature_gradient: Section::new(),
            k_field: Section::new(),
            epsilon_field: Section::new(),
            gradient: Section::new(),
            field: Section::new(), // Initialized correctly
        }
    }


    pub fn get_velocity(&self, entity: &MeshEntity) -> Option<[f64; 3]> {
        self.velocity_field.restrict(entity)
    }

    pub fn get_pressure(&self, entity: &MeshEntity) -> Option<f64> {
        self.pressure_field.restrict(entity)
    }

    pub fn get_velocity_gradient(&self, entity: &MeshEntity) -> Option<[[f64; 3]; 3]> {
        self.velocity_gradient.restrict(entity)
    }

    pub fn set_velocity(&mut self, entity: MeshEntity, value: [f64; 3]) {
        self.velocity_field.set_data(entity, value);
    }

    pub fn set_pressure(&mut self, entity: MeshEntity, value: f64) {
        self.pressure_field.set_data(entity, value);
    }

    pub fn set_velocity_gradient(&mut self, entity: MeshEntity, value: [[f64; 3]; 3]) {
        self.velocity_gradient.set_data(entity, value);
    }
}

pub struct Fluxes {
    pub momentum_fluxes: Section<[f64; 3]>,
    pub energy_fluxes: Section<f64>,
    pub turbulence_fluxes: Section<[f64; 2]>,
}

impl Fluxes {
    pub fn new() -> Self {
        Self {
            momentum_fluxes: Section::new(),
            energy_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        }
    }

    pub fn add_momentum_flux(&mut self, entity: MeshEntity, value: [f64; 3]) {
        if let Some(mut current) = self.momentum_fluxes.restrict(&entity) {
            for i in 0..3 {
                current[i] += value[i];
            }
            self.momentum_fluxes.set_data(entity, current);
        } else {
            self.momentum_fluxes.set_data(entity, value);
        }
    }

    pub fn add_energy_flux(&mut self, entity: MeshEntity, value: f64) {
        if let Some(mut current) = self.energy_fluxes.restrict(&entity) {
            current += value;
            self.energy_fluxes.set_data(entity, current);
        } else {
            self.energy_fluxes.set_data(entity, value);
        }
    }

    pub fn add_turbulence_flux(&mut self, entity: MeshEntity, value: [f64; 2]) {
        if let Some(mut current) = self.turbulence_fluxes.restrict(&entity) {
            for i in 0..2 {
                current[i] += value[i];
            }
            self.turbulence_fluxes.set_data(entity, current);
        } else {
            self.turbulence_fluxes.set_data(entity, value);
        }
    }
}
```

---

`src/equation/manager.rs`

```rust
use crate::{boundary::bc_handler::BoundaryConditionHandler, domain::mesh::Mesh};
use crate::time_stepping::TimeStepper;
use super::{Fields, Fluxes, PhysicalEquation};

pub struct EquationManager<FieldType, TStepper> {
    equations: Vec<Box<dyn PhysicalEquation<FieldType>>>, // Single `Box`
    time_stepper: TStepper,
}

impl<FieldType, TStepper> EquationManager<FieldType, TStepper>
where
    TStepper: TimeStepper<Box<dyn PhysicalEquation<FieldType>>>, // Reflect the correct bound
{
    pub fn new(time_stepper: TStepper) -> Self {
        Self {
            equations: Vec::new(),
            time_stepper,
        }
    }

    pub fn add_equation<E: PhysicalEquation<FieldType> + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    pub fn assemble_all(
        &self,
        domain: &Mesh,
        fields: &Fields<FieldType>,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        let current_time = self.time_stepper.current_time();
        for equation in &self.equations {
            equation.assemble(domain, fields, fluxes, boundary_handler, current_time);
        }
    }

    pub fn step(&mut self, fields: &mut Vec<f64>) {
        let current_time = self.time_stepper.current_time();
        self.time_stepper
            .step(&self.equations, self.time_stepper.get_time_step(), current_time, fields)
            .expect("Time-stepping failed");
    }
}
```

---

`src/domain/sieve.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A `Sieve` struct that manages the relationships (arrows) between `MeshEntity`  
/// elements, organized in an adjacency map.
///
/// The adjacency map tracks directed relations between entities in the mesh.  
/// It supports operations such as adding relationships, querying direct  
/// relations (cones), and computing closure and star sets for entities.
#[derive(Clone, Debug)]
pub struct Sieve {
    /// A thread-safe adjacency map where each key is a `MeshEntity`,  
    /// and the value is a set of `MeshEntity` objects related to the key.  
    pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
}

impl Sieve {
    /// Creates a new empty `Sieve` instance with an empty adjacency map.
    pub fn new() -> Self {
        Sieve {
            adjacency: DashMap::new(),
        }
    }

    /// Adds a directed relationship (arrow) between two `MeshEntity` elements.  
    /// The relationship is stored in the adjacency map from the `from` entity  
    /// to the `to` entity.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.adjacency
            .entry(from)
            .or_insert_with(DashMap::new)
            .insert(to, ());
    }

    /// Retrieves all entities directly related to the given entity (`point`).  
    /// This operation is referred to as retrieving the cone of the entity.  
    /// Returns `None` if there are no related entities.
    pub fn cone(&self, point: &MeshEntity) -> Option<Vec<MeshEntity>> {
        self.adjacency.get(point).map(|cone| {
            cone.iter().map(|entry| entry.key().clone()).collect()
        })
    }

    /// Computes the closure of a given `MeshEntity`.  
    /// The closure includes the entity itself and all entities it covers (cones) recursively.
    pub fn closure(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        let stack = DashMap::new();
        stack.insert(point.clone(), ());

        while !stack.is_empty() {
            let keys: Vec<MeshEntity> = stack.iter().map(|entry| entry.key().clone()).collect();
            for p in keys {
                if result.insert(p.clone(), ()).is_none() {
                    if let Some(cones) = self.cone(&p) {
                        for q in cones {
                            stack.insert(q, ());
                        }
                    }
                }
                stack.remove(&p);
            }
        }
        result
    }

    /// Computes the star of a given `MeshEntity`.  
    /// The star includes the entity itself and all entities that directly cover it (supports).
    pub fn star(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        result.insert(point.clone(), ());
        let supports = self.support(point);
        for support in supports {
            result.insert(support, ());
        }
        result
    }

    /// Retrieves all entities that support the given entity (`point`).  
    /// These are the entities that have an arrow pointing to `point`.
    pub fn support(&self, point: &MeshEntity) -> Vec<MeshEntity> {
        let mut supports = Vec::new();
        self.adjacency.iter().for_each(|entry| {
            let from = entry.key();
            if entry.value().contains_key(point) {
                supports.push(from.clone());
            }
        });
        supports
    }

    /// Computes the meet operation for two entities, `p` and `q`.  
    /// This is the intersection of their closures.
    pub fn meet(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let closure_p = self.closure(p);
        let closure_q = self.closure(q);
        let result = DashMap::new();

        closure_p.iter().for_each(|entry| {
            let key = entry.key();
            if closure_q.contains_key(key) {
                result.insert(key.clone(), ());
            }
        });

        result
    }

    /// Computes the join operation for two entities, `p` and `q`.  
    /// This is the union of their stars.
    pub fn join(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let star_p = self.star(p);
        let star_q = self.star(q);
        let result = DashMap::new();

        star_p.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });
        star_q.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });

        result
    }

    /// Applies a given function in parallel to all adjacency map entries.  
    /// This function is executed concurrently over each entity and its  
    /// corresponding set of related entities.
    pub fn par_for_each_adjacent<F>(&self, func: F)
    where
        F: Fn((&MeshEntity, Vec<MeshEntity>)) + Sync + Send,
    {
        // Collect entries from DashMap to avoid borrow conflicts
        let entries: Vec<_> = self.adjacency.iter().map(|entry| {
            let key = entry.key().clone();
            let values: Vec<MeshEntity> = entry.value().iter().map(|e| e.key().clone()).collect();
            (key, values)
        }).collect();

        // Execute in parallel over collected entries
        entries.par_iter().for_each(|entry| {
            func((&entry.0, entry.1.clone()));
        });
    }
}

```

---

`src/domain/section.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.  
/// It provides methods for setting, updating, and retrieving data, and supports  
/// parallel updates for performance improvements.  
///
/// Example usage:
///
///    let section = Section::new();  
///    let vertex = MeshEntity::Vertex(1);  
///    section.set_data(vertex, 42);  
///    assert_eq!(section.restrict(&vertex), Some(42));  
/// 
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.  
    pub data: DashMap<MeshEntity, T>,
}

impl<T> Section<T> {
    /// Creates a new `Section` with an empty data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn new() -> Self {
        Section {
            data: DashMap::new(),
        }
    }

    /// Sets the data associated with a given `MeshEntity`.  
    /// This method inserts the `entity` and its corresponding `value` into the data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    section.set_data(MeshEntity::Vertex(1), 10);  
    ///
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Restricts the data for a given `MeshEntity` by returning an immutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 42);  
    ///    assert_eq!(section.restrict(&vertex), Some(42));  
    ///
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Applies the given function in parallel to update all data values in the section.
    ///
    /// Example usage:
    ///
    ///    section.parallel_update(|v| *v += 1);  
    ///
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
        T: Send + Sync,
    {
        // Clone the keys to ensure safe access to each mutable entry in parallel.
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        // Apply the update function to each entry in parallel.
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Restricts the data for a given `MeshEntity` by returning a mutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 5);  
    ///    let mut value = section.restrict_mut(&vertex).unwrap();  
    ///    value = 10;  
    ///    section.set_data(vertex, value);  
    ///
    pub fn restrict_data_mut(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates the data for a specific `MeshEntity` by replacing the existing value  
    /// with the new value.  
    ///
    /// Example usage:
    ///
    ///    section.update_data(&MeshEntity::Vertex(1), 15);  
    ///
    pub fn update_data(&self, entity: &MeshEntity, new_value: T) {
        self.data.insert(*entity, new_value);
    }

    /// Clears all data from the section, removing all entity associations.  
    ///
    /// Example usage:
    ///
    ///    section.clear();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Retrieves all `MeshEntity` objects associated with the section.  
    ///
    /// Returns a vector containing all mesh entities currently stored in the section.  
    ///
    /// Example usage:
    ///
    ///    let entities = section.entities();  
    ///
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves all data stored in the section as immutable copies.  
    ///
    /// Returns a vector of data values.  
    ///
    /// Example usage:
    ///
    ///    let all_data = section.all_data();  
    ///
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Retrieves all data stored in the section with mutable access.  
    ///
    /// Returns a vector of data values that can be modified.  
    ///
    /// Example usage:
    ///
    ///    let all_data_mut = section.all_data_mut();  
    ///
    pub fn all_data_mut(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter_mut().map(|entry| entry.value().clone()).collect()
    }
}
```

---

`src/domain/mesh/entities.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

impl Mesh {
    /// Adds a new `MeshEntity` to the mesh.  
    /// The entity will be inserted into the thread-safe `entities` set.  
    /// 
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    mesh.add_entity(vertex);  
    /// 
    pub fn add_entity(&self, entity: MeshEntity) {
        self.entities.write().unwrap().insert(entity);
    }

    /// Establishes a relationship (arrow) between two mesh entities.  
    /// This creates an arrow from the `from` entity to the `to` entity  
    /// in the sieve structure.  
    ///
    /// Example usage:
    /// 
    ///    let mut mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(2);  
    ///    mesh.add_relationship(vertex, edge);  
    /// 
    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Adds an arrow from one mesh entity to another in the sieve structure.  
    /// This method is a simple delegate to the `Sieve`'s `add_arrow` method.
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(2);  
    ///    mesh.add_arrow(vertex, edge);  
    /// 
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Sets the 3D coordinates for a vertex and adds the vertex entity  
    /// to the mesh if it's not already present.  
    /// 
    /// This method inserts the vertex's coordinates into the  
    /// `vertex_coordinates` map and adds the vertex to the `entities` set.
    ///
    /// Example usage:
    /// 
    ///    let mut mesh = Mesh::new();  
    ///    mesh.set_vertex_coordinates(1, [1.0, 2.0, 3.0]);  
    ///    assert_eq!(mesh.get_vertex_coordinates(1), Some([1.0, 2.0, 3.0]));  
    ///
    pub fn set_vertex_coordinates(&mut self, vertex_id: usize, coords: [f64; 3]) {
        self.vertex_coordinates.insert(vertex_id, coords);
        self.add_entity(MeshEntity::Vertex(vertex_id));
    }

    /// Retrieves the 3D coordinates of a vertex by its identifier.  
    ///
    /// Returns `None` if the vertex does not exist in the `vertex_coordinates` map.
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let coords = mesh.get_vertex_coordinates(1);  
    ///    assert!(coords.is_none());  
    ///
    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        self.vertex_coordinates.get(&vertex_id).cloned()
    }

    /// Counts the number of entities of a specified type (e.g., Vertex, Edge, Face, Cell)  
    /// within the mesh.  
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let count = mesh.count_entities(&MeshEntity::Vertex(1));  
    ///    assert_eq!(count, 0);  
    ///
    pub fn count_entities(&self, entity_type: &MeshEntity) -> usize {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| match (e, entity_type) {
                (MeshEntity::Vertex(_), MeshEntity::Vertex(_)) => true,
                (MeshEntity::Cell(_), MeshEntity::Cell(_)) => true,
                (MeshEntity::Edge(_), MeshEntity::Edge(_)) => true,
                (MeshEntity::Face(_), MeshEntity::Face(_)) => true,
                _ => false,
            })
            .count()
    }

    /// Applies a given function to each entity in the mesh in parallel.  
    ///
    /// The function `func` is applied to all mesh entities concurrently  
    /// using Rayon’s parallel iterator.
    ///
    /// Example usage:
    /// 
    ///    mesh.par_for_each_entity(|entity| {  
    ///        println!("{:?}", entity);  
    ///    });  
    ///
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();
        entities.par_iter().for_each(func);
    }

    /// Retrieves all the `Cell` entities from the mesh.  
    ///
    /// This method returns a `Vec<MeshEntity>` containing all entities  
    /// classified as cells.
    ///
    /// Example usage:
    /// 
    ///    let cells = mesh.get_cells();  
    ///    assert!(cells.is_empty());  
    ///
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Retrieves all the `Face` entities from the mesh.  
    ///
    /// This method returns a `Vec<MeshEntity>` containing all entities  
    /// classified as faces.
    ///
    /// Example usage:
    /// 
    ///    let faces = mesh.get_faces();  
    ///    assert!(faces.is_empty());  
    ///
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    /// Retrieves the vertices of the given face.
    pub fn get_vertices_of_face(&self, face: &MeshEntity) -> Vec<MeshEntity> {
        self.sieve.cone(face).unwrap_or_default()
            .into_iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .collect()
    }

    /// Computes properties for each entity in the mesh in parallel,  
    /// returning a map of `MeshEntity` to the computed property.  
    ///
    /// The `compute_fn` is a user-provided function that takes a reference  
    /// to a `MeshEntity` and returns a computed value of type `PropertyType`.  
    ///
    /// Example usage:
    /// 
    ///    let properties = mesh.compute_properties(|entity| {  
    ///        entity.get_id()  
    ///    });  
    ///
    pub fn compute_properties<F, PropertyType>(&self, compute_fn: F) -> FxHashMap<MeshEntity, PropertyType>
    where
        F: Fn(&MeshEntity) -> PropertyType + Sync + Send,
        PropertyType: Send,
    {
        let entities = self.entities.read().unwrap();
        entities
            .par_iter()
            .map(|entity| (*entity, compute_fn(entity)))
            .collect()
    }

    /// Retrieves the ordered neighboring cells for each cell in the mesh.
    ///
    /// This method is designed for use in flux computations and gradient reconstruction,
    /// and returns the neighboring cells in a predetermined, consistent order.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which neighbors are retrieved.
    ///
    /// # Returns
    /// A vector of neighboring cells ordered for consistency in TVD calculations.
    pub fn get_ordered_neighbors(&self, cell: &MeshEntity) -> Vec<MeshEntity> {
        let mut neighbors = Vec::new();
        if let Some(faces) = self.get_faces_of_cell(cell) {
            for face in faces.iter() {
                let cells_sharing_face = self.get_cells_sharing_face(&face.key());
                for neighbor in cells_sharing_face.iter() {
                    if *neighbor.key() != *cell {
                        neighbors.push(*neighbor.key());
                    }
                }
            }
        }
        neighbors.sort_by(|a, b| a.get_id().cmp(&b.get_id())); // Ensures consistent ordering by ID
        neighbors
    }

    /// Maps each `MeshEntity` in the mesh to a unique index.
    pub fn get_entity_to_index(&self) -> DashMap<MeshEntity, usize> {
        let entity_to_index = DashMap::new();
        let entities = self.entities.read().unwrap();
        entities.iter().enumerate().for_each(|(index, entity)| {
            entity_to_index.insert(entity.clone(), index);
        });

        entity_to_index
    }
}
```

---

`src/domain/mesh/geometry.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;

impl Mesh {
    /// Retrieves all the faces of a given cell, filtering only face entities.
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<DashMap<MeshEntity, ()>> {
        self.sieve.cone(cell).map(|set| {
            let faces = DashMap::new();
            set.into_iter()
                .filter(|entity| matches!(entity, MeshEntity::Face(_)))
                .for_each(|face| {
                    faces.insert(face, ());
                });
            faces
        })
    }

    /// Retrieves all the cells that share the given face, filtering only cell entities that are present in the mesh.
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let cells = DashMap::new();
        let entities = self.entities.read().unwrap();
        self.sieve
            .support(face)
            .into_iter()
            .filter(|entity| matches!(entity, MeshEntity::Cell(_)) && entities.contains(entity))
            .for_each(|cell| {
                cells.insert(cell, ());
            });
        cells
    }

    /// Computes the Euclidean distance between two cells based on their centroids.
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.
    pub fn get_face_area(&self, face: &MeshEntity) -> Option<f64> {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let mut geometry = Geometry::new();
        let face_id = face.get_id();
        Some(geometry.compute_face_area(face_id, face_shape, &face_vertices))
    }

    /// Computes the centroid of a cell based on its vertices.
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let _cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        let mut geometry = Geometry::new();
        geometry.compute_cell_centroid(self, cell)
    }

    /// Retrieves all vertices connected to the given vertex by shared cells.
    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let neighbors = DashMap::new();
        let connected_cells = self.sieve.support(vertex);

        connected_cells.into_iter().for_each(|cell| {
            if let Some(cell_vertices) = self.sieve.cone(&cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(v.clone(), ());
                    }
                }
            }
        });
        neighbors.into_iter().map(|(vertex, _)| vertex).collect()
    }

    /// Returns an iterator over the IDs of all vertices in the mesh.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on the number of vertices it has.
    pub fn get_cell_shape(&self, cell: &MeshEntity) -> Result<CellShape, String> {
        let cell_vertices = self.get_cell_vertices(cell);
        match cell_vertices.len() {
            4 => Ok(CellShape::Tetrahedron),
            5 => Ok(CellShape::Pyramid),
            6 => Ok(CellShape::Prism),
            8 => Ok(CellShape::Hexahedron),
            _ => Err(format!(
                "Unsupported cell shape with {} vertices. Expected 4, 5, 6, or 8 vertices.",
                cell_vertices.len()
            )),
        }
    }

    /// Retrieves the vertices of a cell and their coordinates, sorted by vertex ID.
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_entities) = self.sieve.cone(cell) {
            for entity in connected_entities {
                if let MeshEntity::Vertex(vertex_id) = entity {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                }
            }
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }

    /// Retrieves the vertices of a face and their coordinates, sorted by vertex ID.
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                }
            }
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }

    /// Computes the normal vector of a face based on its vertices and shape.
    ///
    /// This function calculates the outward normal vector for a face by leveraging the
    /// `Geometry` module. It determines the face shape and uses the vertices to compute
    /// the vector. The orientation of the normal can optionally depend on a neighboring cell.
    ///
    /// # Arguments
    /// * `face` - The face entity for which the normal is computed.
    /// * `reference_cell` - Optional cell entity to determine the normal orientation.
    ///
    /// # Returns
    /// * `Option<[f64; 3]>` - The computed normal vector if successful, otherwise `None`.
    pub fn get_face_normal(
        &self,
        face: &MeshEntity,
        reference_cell: Option<&MeshEntity>,
    ) -> Option<[f64; 3]> {
        // Retrieve face vertices
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let geometry = Geometry::new();
        let normal = match face_shape {
            FaceShape::Triangle => geometry.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => geometry.compute_quadrilateral_normal(&face_vertices),
        };

        // If a reference cell is provided, adjust the normal's orientation
        if let Some(cell) = reference_cell {
            let cell_centroid = self.get_cell_centroid(cell);
            let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            // Compute the vector from the face centroid to the cell centroid
            let to_cell_vector = [
                cell_centroid[0] - face_centroid[0],
                cell_centroid[1] - face_centroid[1],
                cell_centroid[2] - face_centroid[2],
            ];

            // Ensure the normal points outward by checking the dot product
            let dot_product = normal[0] * to_cell_vector[0]
                + normal[1] * to_cell_vector[1]
                + normal[2] * to_cell_vector[2];

            if dot_product < 0.0 {
                // Reverse the normal direction to make it outward-pointing
                return Some([-normal[0], -normal[1], -normal[2]]);
            }
        }

        Some(normal)
    }
}
```

---

`src/geometry/mod.rs`

```rust
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};
use std::sync::Mutex;

// Module for handling geometric data and computations
// 2D Shape Modules
pub mod quadrilateral;
pub mod triangle;
// 3D Shape Modules
pub mod tetrahedron;
pub mod hexahedron;
pub mod prism;
pub mod pyramid;

/// The `Geometry` struct stores geometric data for a mesh, including vertex coordinates, 
/// cell centroids, and volumes. It also maintains a cache of computed properties such as 
/// volume and centroid for reuse, optimizing performance by avoiding redundant calculations.
pub struct Geometry {
    pub vertices: Vec<[f64; 3]>,        // 3D coordinates for each vertex
    pub cell_centroids: Vec<[f64; 3]>,  // Centroid positions for each cell
    pub cell_volumes: Vec<f64>,         // Volumes of each cell
    pub cache: Mutex<FxHashMap<usize, GeometryCache>>, // Cache for computed properties, with thread safety
}

/// The `GeometryCache` struct stores computed properties of geometric entities, 
/// including volume, centroid, and area, with an optional "dirty" flag for lazy evaluation.
#[derive(Default)]
pub struct GeometryCache {
    pub volume: Option<f64>,
    pub centroid: Option<[f64; 3]>,
    pub area: Option<f64>,
    pub normal: Option<[f64; 3]>,  // Stores a precomputed normal vector for a face
}

/// `CellShape` enumerates the different cell shapes in a mesh, including:
/// * Tetrahedron
/// * Hexahedron
/// * Prism
/// * Pyramid
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellShape {
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

/// `FaceShape` enumerates the different face shapes in a mesh, including:
/// * Triangle
/// * Quadrilateral
#[derive(Debug, Clone, Copy)]
pub enum FaceShape {
    Triangle,
    Quadrilateral,
}

impl Geometry {
    /// Initializes a new `Geometry` instance with empty data.
    pub fn new() -> Geometry {
        Geometry {
            vertices: Vec::new(),
            cell_centroids: Vec::new(),
            cell_volumes: Vec::new(),
            cache: Mutex::new(FxHashMap::default()),
        }
    }

    /// Adds or updates a vertex in the geometry. If the vertex already exists,
    /// it updates its coordinates; otherwise, it adds a new vertex.
    ///
    /// # Arguments
    /// * `vertex_index` - The index of the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]) {
        if vertex_index >= self.vertices.len() {
            self.vertices.resize(vertex_index + 1, [0.0, 0.0, 0.0]);
        }
        self.vertices[vertex_index] = coords;
        self.invalidate_cache();
    }

    /// Computes and returns the centroid of a specified cell using the cell's shape and vertices.
    /// Caches the result for reuse.
    pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> [f64; 3] {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.centroid) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let centroid = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_centroid(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_centroid(&cell_vertices),
            CellShape::Prism => self.compute_prism_centroid(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_centroid(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().centroid = Some(centroid);
        centroid
    }

    /// Computes the volume of a given cell using its shape and vertex coordinates.
    /// The computed volume is cached for efficiency.
    pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> f64 {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.volume) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let volume = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_volume(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_volume(&cell_vertices),
            CellShape::Prism => self.compute_prism_volume(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_volume(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().volume = Some(volume);
        volume
    }

    /// Calculates Euclidean distance between two points in 3D space.
    pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt()
    }

    /// Computes the area of a 2D face based on its shape, caching the result.
    pub fn compute_face_area(&mut self, face_id: usize, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64 {
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.area) {
            return cached;
        }

        let area = match face_shape {
            FaceShape::Triangle => self.compute_triangle_area(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_area(face_vertices),
        };

        self.cache.lock().unwrap().entry(face_id).or_default().area = Some(area);
        area
    }

    /// Computes the centroid of a 2D face based on its shape.
    ///
    /// # Arguments
    /// * `face_shape` - Enum defining the shape of the face (e.g., Triangle, Quadrilateral).
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the face.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the face centroid.
    pub fn compute_face_centroid(&self, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        match face_shape {
            FaceShape::Triangle => self.compute_triangle_centroid(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_centroid(face_vertices),
        }
    }

    /// Computes and caches the normal vector for a face based on its shape.
    ///
    /// This function determines the face shape and calls the appropriate 
    /// function to compute the normal vector.
    ///
    /// # Arguments
    /// * `mesh` - A reference to the mesh.
    /// * `face` - The face entity for which to compute the normal.
    /// * `cell` - The cell associated with the face, used to determine the orientation.
    ///
    /// # Returns
    /// * `Option<[f64; 3]>` - The computed normal vector, or `None` if it could not be computed.
    pub fn compute_face_normal(
        &mut self,
        mesh: &Mesh,
        face: &MeshEntity,
        _cell: &MeshEntity,
    ) -> Option<[f64; 3]> {
        let face_id = face.get_id();

        // Check if the normal is already cached
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.normal) {
            return Some(cached);
        }

        let face_vertices = mesh.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let normal = match face_shape {
            FaceShape::Triangle => self.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_normal(&face_vertices),
        };

        // Cache the normal vector for future use
        self.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal);

        Some(normal)
    }

    /// Invalidate the cache when geometry changes (e.g., vertex updates).
    fn invalidate_cache(&mut self) {
        self.cache.lock().unwrap().clear();
    }

    /// Computes the total volume of all cells.
    pub fn compute_total_volume(&self) -> f64 {
        self.cell_volumes.par_iter().sum()
    }

    /// Updates all cell volumes in parallel using mesh information.
    pub fn update_all_cell_volumes(&mut self, mesh: &Mesh) {
        let new_volumes: Vec<f64> = mesh
            .get_cells()
            .par_iter()
            .map(|cell| {
                let mut temp_geometry = Geometry::new();
                temp_geometry.compute_cell_volume(mesh, cell)
            })
            .collect();

        self.cell_volumes = new_volumes;
    }

    /// Computes the total centroid of all cells.
    pub fn compute_total_centroid(&self) -> [f64; 3] {
        let total_centroid: [f64; 3] = self.cell_centroids
            .par_iter()
            .cloned()
            .reduce(
                || [0.0, 0.0, 0.0],
                |acc, centroid| [
                    acc[0] + centroid[0],
                    acc[1] + centroid[1],
                    acc[2] + centroid[2],
                ],
            );

        let num_centroids = self.cell_centroids.len() as f64;
        [
            total_centroid[0] / num_centroids,
            total_centroid[1] / num_centroids,
            total_centroid[2] / num_centroids,
        ]
    }
}
```

---

`src/solver/ksp.rs`

```rust
//! Enhancements to the KSP module to introduce an interface adapter for flexible usage.
//!
//! This adds the `SolverManager` for high-level integration of solvers and preconditioners.

use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use std::sync::Arc;

#[derive(Debug)]
pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
}

/// KSP trait for Krylov solvers, encompassing solvers like CG and GMRES.
pub trait KSP {
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult;
}

/// Struct representing a high-level interface for managing solver configuration.
pub struct SolverManager {
    solver: Box<dyn KSP>,
    preconditioner: Option<Arc<dyn Preconditioner>>,
}

impl SolverManager {
    /// Creates a new `SolverManager` instance with a specified solver.
    ///
    /// # Arguments
    /// - `solver`: The Krylov solver to be used.
    ///
    /// # Returns
    /// A new `SolverManager` instance.
    pub fn new(solver: Box<dyn KSP>) -> Self {
        SolverManager {
            solver,
            preconditioner: None,
        }
    }

    /// Sets a preconditioner for the solver.
    ///
    /// # Arguments
    /// - `preconditioner`: The preconditioner to be used.
    pub fn set_preconditioner(&mut self, preconditioner: Arc<dyn Preconditioner>) {
        self.preconditioner = Some(preconditioner);
    }

    /// Solves a system `Ax = b` using the configured solver and optional preconditioner.
    ///
    /// # Arguments
    /// - `a`: The system matrix `A`.
    /// - `b`: The right-hand side vector `b`.
    /// - `x`: The solution vector `x`, which will be updated with the computed solution.
    ///
    /// # Returns
    /// A `SolverResult` containing convergence information and the final residual norm.
    pub fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        if let Some(preconditioner) = &self.preconditioner {
            preconditioner.apply(a, b, x);
        }
        self.solver.solve(a, b, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::preconditioner::Jacobi;
    use crate::solver::cg::ConjugateGradient;
    use faer::{mat, Mat};

    #[test]
    fn test_solver_manager_with_jacobi_preconditioner() {
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];
        let b = mat![
            [1.0],
            [2.0],
        ];
        let mut x = Mat::<f64>::zeros(2, 1);

        // Initialize CG solver and solver manager
        let cg_solver = ConjugateGradient::new(100, 1e-6);
        let mut solver_manager = SolverManager::new(Box::new(cg_solver));

        // Set Jacobi preconditioner
        let jacobi_preconditioner = Arc::new(Jacobi::default());
        solver_manager.set_preconditioner(jacobi_preconditioner);

        // Solve the system
        let result = solver_manager.solve(&a, &b, &mut x);

        // Validate results
        assert!(result.converged, "Solver did not converge");
        assert!(result.residual_norm <= 1e-6, "Residual norm too large");
        assert!(
            !crate::linalg::vector::traits::Vector::as_slice(&x).contains(&f64::NAN),
            "Solution contains NaN values"
        );
    }
}
```

---

`src/time_stepping/ts.rs`

```rust
use crate::linalg::Matrix;
use crate::linalg::Vector;

#[derive(Debug)]
pub enum TimeSteppingError {
    InvalidStep,
    SolverError(String),
}

pub trait TimeDependentProblem {
    type State: Vector + Clone;
    type Time: Copy + PartialOrd;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar;

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>>;

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}

pub trait TimeStepper<P>
where
    P: TimeDependentProblem + Sized,
{
    fn current_time(&self) -> P::Time;

    fn set_current_time(&mut self, time: P::Time);

    fn step(
        &mut self,
        problems: &[P], // Accept slice of `P`
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
    ) -> Result<P::Time, TimeSteppingError>;

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);

    fn set_time_step(&mut self, dt: P::Time);

    fn get_time_step(&self) -> P::Time;
}

```

---

`src/use_cases/matrix_construction.rs`

```rust
// src/use_cases/matrix_construction.rs

use crate::interface_adapters::matrix_adapter::MatrixAdapter;
use crate::linalg::matrix::{MatrixOperations, ExtendedMatrixOperations};
use faer::Mat;

/// Constructs and initializes a matrix for simulation.
/// Provides functions for creating, resizing, and initializing matrices.
pub struct MatrixConstruction;

impl MatrixConstruction {
    /// Constructs a dense matrix with specified dimensions and fills it with zeros.
    pub fn build_zero_matrix(rows: usize, cols: usize) -> Mat<f64> {
        MatrixAdapter::new_dense_matrix(rows, cols)
    }

    /// Initializes a matrix with a specific value.
    /// This can be useful for setting initial conditions for simulations.
    pub fn initialize_matrix_with_value<T: MatrixOperations>(matrix: &mut T, value: f64) {
        let (rows, cols) = matrix.size();
        for row in 0..rows {
            for col in 0..cols {
                MatrixAdapter::set_element(matrix, row, col, value);
            }
        }
    }

    /// Resizes a matrix to new dimensions, maintaining existing data if possible.
    pub fn resize_matrix<T: ExtendedMatrixOperations>(matrix: &mut T, new_rows: usize, new_cols: usize) {
        MatrixAdapter::resize_matrix(matrix, new_rows, new_cols);
    }
}

```

---

`src/use_cases/rhs_construction.rs`

```rust
// src/use_cases/rhs_construction.rs

use crate::interface_adapters::vector_adapter::VectorAdapter;
use crate::linalg::vector::Vector;
use faer::Mat;

/// Constructs and initializes the right-hand side (RHS) vector for a linear system.
pub struct RHSConstruction;

impl RHSConstruction {
    /// Builds a dense RHS vector of a specified length, initialized to zero.
    pub fn build_zero_rhs(size: usize) -> Mat<f64> {
        VectorAdapter::new_dense_vector(size)
    }

    /// Initializes the RHS vector with a specific value across all elements.
    pub fn initialize_rhs_with_value<T: Vector<Scalar = f64>>(vector: &mut T, value: f64) {
        for i in 0..vector.len() {
            VectorAdapter::set_element(vector, i, value);
        }
    }

    /// Resizes the RHS vector to a new length.
    pub fn resize_rhs(vector: &mut Mat<f64>, new_size: usize) {
        let mut new_vector = Mat::<f64>::zeros(new_size, 1);
        for i in 0..usize::min(vector.nrows(), new_size) {
            new_vector.write(i, 0, vector.read(i, 0));
        }
        *vector = new_vector; // Replace the old vector with the resized one
    }
}

```

---

`src/equation/reconstruction/reconstruct.rs`

```rust
// src/equation/reconstruction/reconstruct.rs

/// Reconstructs the solution at a face center by extrapolating from the cell value 
/// and its gradient. This approach is critical for finite volume methods as it
/// provides a face-centered scalar value, which is essential for flux calculations.
///
/// # Arguments
///
/// * `cell_value` - The scalar field value at the cell center, representing the primary
///                  field quantity (e.g., temperature, pressure).
/// * `gradient` - The gradient vector `[f64; 3]` representing the rate of change of the
///                scalar field within the cell in each spatial direction. This gradient
///                allows for a linear approximation of the field near the cell center.
/// * `cell_center` - Coordinates `[f64; 3]` of the cell center, where `cell_value` and
///                   `gradient` are defined.
/// * `face_center` - Coordinates `[f64; 3]` of the face center where the scalar field 
///                   value is to be reconstructed.
///
/// # Returns
///
/// The reconstructed scalar field value at the face center, determined by linearly 
/// extrapolating from the cell center using the gradient.
///
/// # Example
///
/// ```rust
/// use hydra::equation::reconstruction::reconstruct::reconstruct_face_value;
///
/// let cell_value = 1.0;
/// let gradient = [2.0, 0.0, 0.0];
/// let cell_center = [0.0, 0.0, 0.0];
/// let face_center = [0.5, 0.0, 0.0];
///
/// let reconstructed_value = reconstruct_face_value(cell_value, gradient, cell_center, face_center);
/// assert_eq!(reconstructed_value, 2.0);
/// ```
pub fn reconstruct_face_value(
    cell_value: f64,
    gradient: [f64; 3],
    cell_center: [f64; 3],
    face_center: [f64; 3],
) -> f64 {
    let delta = [
        face_center[0] - cell_center[0],
        face_center[1] - cell_center[1],
        face_center[2] - cell_center[2],
    ];
    cell_value + gradient[0] * delta[0] + gradient[1] * delta[1] + gradient[2] * delta[2]
}

```

---

`src/equation/gradient/mod.rs`

```rust
//! Module for gradient calculation in finite element and finite volume methods.
//!
//! This module provides a flexible framework for computing gradients using
//! different numerical methods. It defines the `Gradient` struct, which serves
//! as the main interface for gradient computation, and supports multiple
//! gradient calculation methods via the `GradientCalculationMethod` enum and
//! `GradientMethod` trait.

use crate::boundary::bc_handler::BoundaryConditionHandler;
use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::geometry::Geometry;
use std::error::Error;

pub mod gradient_calc;
pub mod tests;

use gradient_calc::FiniteVolumeGradient;

/// Enum representing the available gradient calculation methods.
pub enum GradientCalculationMethod {
    FiniteVolume,
    // Additional methods can be added here as needed
}

impl GradientCalculationMethod {
    /// Factory function to create a specific gradient calculation method based on the enum variant.
    pub fn create_method(&self) -> Box<dyn GradientMethod> {
        match self {
            GradientCalculationMethod::FiniteVolume => Box::new(FiniteVolumeGradient {}),
            // Extend here with other methods as needed
        }
    }
}

/// Trait defining the interface for gradient calculation methods.
///
/// Each gradient calculation method must implement this trait, which includes
/// the `calculate_gradient` function for computing the gradient at a given cell.
pub trait GradientMethod {
    /// Computes the gradient for a given cell.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh structure containing cells and faces.
    /// - `boundary_handler`: Reference to the boundary condition handler.
    /// - `geometry`: Geometry utilities for computing areas, volumes, etc.
    /// - `field`: Scalar field values for each cell.
    /// - `cell`: The current cell for which the gradient is computed.
    /// - `time`: Current simulation time.
    ///
    /// # Returns
    /// - `Ok([f64; 3])`: Computed gradient vector.
    /// - `Err(Box<dyn Error>)`: If any error occurs during computation.
    fn calculate_gradient(
        &self,
        mesh: &Mesh,
        boundary_handler: &BoundaryConditionHandler,
        geometry: &mut Geometry,
        field: &Section<f64>,
        cell: &MeshEntity,
        time: f64,
    ) -> Result<[f64; 3], Box<dyn Error>>;
}

/// Gradient calculator that accepts a gradient method for flexible computation.
///
/// This struct serves as the main interface for computing gradients across the mesh.
/// It delegates the actual gradient computation to the specified `GradientMethod`.
pub struct Gradient<'a> {
    mesh: &'a Mesh,
    boundary_handler: &'a BoundaryConditionHandler,
    geometry: Geometry,
    method: Box<dyn GradientMethod>,
}

impl<'a> Gradient<'a> {
    /// Constructs a new `Gradient` calculator with the specified calculation method.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh structure.
    /// - `boundary_handler`: Reference to the boundary condition handler.
    /// - `method`: The gradient calculation method to use.
    pub fn new(
        mesh: &'a Mesh,
        boundary_handler: &'a BoundaryConditionHandler,
        method: GradientCalculationMethod,
    ) -> Self {
        Self {
            mesh,
            boundary_handler,
            geometry: Geometry::new(),
            method: method.create_method(),
        }
    }

    /// Computes the gradient of a scalar field across each cell in the mesh.
    ///
    /// # Parameters
    /// - `field`: Scalar field values for each cell.
    /// - `gradient`: Mutable section to store the computed gradient vectors.
    /// - `time`: Current simulation time.
    ///
    /// # Returns
    /// - `Ok(())`: If gradients are successfully computed for all cells.
    /// - `Err(Box<dyn Error>)`: If any error occurs during computation.
    pub fn compute_gradient(
        &mut self,  // Changed to mutable reference
        field: &Section<f64>,
        gradient: &mut Section<[f64; 3]>,
        time: f64,
    ) -> Result<(), Box<dyn Error>> {
        for cell in self.mesh.get_cells() {
            let grad_phi = self.method.calculate_gradient(
                self.mesh,
                self.boundary_handler,
                &mut self.geometry,  // Now mutable
                field,
                &cell,
                time,
            )?;
            gradient.set_data(cell, grad_phi);
        }
        Ok(())
    }
}
```

---

`src/equation/flux_limiter/flux_limiters.rs`

```rust
/// Trait defining a generic Flux Limiter, which adjusts flux values
/// to prevent numerical oscillations, crucial for Total Variation Diminishing (TVD) schemes.
/// 
/// # Purpose
/// This trait provides a method `limit` to calculate a modified value
/// based on neighboring values, which helps in maintaining the stability
/// and accuracy of the finite volume method by applying flux limiters.
/// 
/// # Method
/// - `limit`: Takes left and right flux values and returns a constrained value
/// to mitigate oscillations at cell interfaces.
pub trait FluxLimiter {
    /// Applies the limiter to two neighboring values to prevent oscillations.
    ///
    /// # Parameters
    /// - `left_value`: The flux value on the left side of the interface.
    /// - `right_value`: The flux value on the right side of the interface.
    ///
    /// # Returns
    /// A modified value that limits oscillations, ensuring TVD compliance.
    fn limit(&self, left_value: f64, right_value: f64) -> f64;
}

/// Implementation of the Minmod flux limiter.
///
/// # Characteristics
/// The Minmod limiter is a simple, commonly used limiter that chooses the minimum
/// absolute value of the left and right values while preserving the sign. It is effective
/// for handling sharp gradients without introducing non-physical oscillations.
/// 
/// # Implementation Details
/// - If `left_value` and `right_value` have opposite signs or are zero, it returns 0.0
///   to avoid oscillations.
/// - Otherwise, it selects the smaller absolute value, retaining the original sign.
pub struct Minmod;

/// Implementation of the Superbee flux limiter.
///
/// # Characteristics
/// The Superbee limiter provides higher resolution compared to Minmod and is more aggressive,
/// capturing sharp gradients while preserving stability. This limiter is suitable
/// for problems where capturing steep gradients is essential.
/// 
/// # Implementation Details
/// - If `left_value` and `right_value` have opposite signs or are zero, it returns 0.0,
///   preventing oscillations.
/// - Otherwise, it calculates two options based on twice the left and right values,
///   clamping them within the original range, and selects the larger of the two.
pub struct Superbee;

impl FluxLimiter for Minmod {
    /// Applies the Minmod flux limiter to two neighboring values.
    ///
    /// # Parameters
    /// - `left_value`: Flux value from the left side of the cell interface.
    /// - `right_value`: Flux value from the right side of the cell interface.
    ///
    /// # Returns
    /// - `0.0` if the values have different signs (indicating an oscillation).
    /// - Otherwise, returns the value with the smaller magnitude, preserving the sign.
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            println!("Minmod: Different signs or zero - returning 0.0");
            0.0 // Different signs or zero: prevent oscillations by returning zero
        } else {
            // Take the minimum magnitude value, maintaining its original sign
            let result = if left_value.abs() < right_value.abs() {
                left_value
            } else {
                right_value
            };
            println!("Minmod: left_value = {}, right_value = {}, result = {}", left_value, right_value, result);
            result
        }
    }
}

impl FluxLimiter for Superbee {
    /// Applies the Superbee flux limiter to two neighboring values.
    ///
    /// # Parameters
    /// - `left_value`: Flux value from the left side of the cell interface.
    /// - `right_value`: Flux value from the right side of the cell interface.
    ///
    /// # Returns
    /// - `0.0` if the values have different signs, to prevent oscillations.
    /// - Otherwise, calculates two possible limited values and returns the maximum
    ///   to ensure higher resolution while maintaining stability.
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            println!("Superbee: Different signs or zero - returning 0.0");
            0.0 // Different signs: prevent oscillations by returning zero
        } else {
            // Calculate two limited values and return the maximum to capture sharp gradients
            let option1 = (2.0 * left_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let option2 = (2.0 * right_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let result = option1.max(option2);

            println!(
                "Superbee: left_value = {}, right_value = {}, option1 = {}, option2 = {}, result = {}",
                left_value, right_value, option1, option2, result
            );

            result
        }
    }
}
```

---

`src/linalg/matrix/traits.rs`

```rust
// src/linalg/matrix/traits.rs

use crate::linalg::Vector;

/// Trait defining essential matrix operations (abstract over dense, sparse)
/// Define that any type implementing Matrix must be Send and Sync
pub trait Matrix: Send + Sync {
    type Scalar: Copy + Send + Sync;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>); // y = A * x
    fn get(&self, i: usize, j: usize) -> Self::Scalar;
    fn trace(&self) -> Self::Scalar;
    fn frobenius_norm(&self) -> Self::Scalar;
    fn as_slice(&self) -> Box<[Self::Scalar]>;
    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]>;
}

/// Trait defining matrix operations for building and manipulation
pub trait MatrixOperations: Send + Sync {
    fn construct(rows: usize, cols: usize) -> Self
    where
        Self: Sized;
    fn set(&mut self, row: usize, col: usize, value: f64);
    fn get(&self, row: usize, col: usize) -> f64;
    fn size(&self) -> (usize, usize);
}

/// Extended matrix operations trait for resizing
pub trait ExtendedMatrixOperations: MatrixOperations {
    fn resize(&mut self, new_rows: usize, new_cols: usize)
    where
        Self: Sized;
}
```

---

`src/linalg/vector/traits.rs`

```rust
// src/vector/traits.rs


/// Trait defining a set of common operations for vectors.
/// It abstracts over different vector types, enabling flexible implementations
/// for standard dense vectors or more complex matrix structures.
///
/// # Requirements:
/// Implementations of `Vector` must be thread-safe (`Send` and `Sync`).
pub trait Vector: Send + Sync {
    /// The scalar type of the vector elements.
    type Scalar: Copy + Send + Sync;

    /// Returns the length (number of elements) of the vector.
    fn len(&self) -> usize;

    /// Retrieves the element at index `i`.
    ///
    /// # Panics
    /// Panics if the index `i` is out of bounds.
    fn get(&self, i: usize) -> Self::Scalar;

    /// Sets the element at index `i` to `value`.
    ///
    /// # Panics
    /// Panics if the index `i` is out of bounds.
    fn set(&mut self, i: usize, value: Self::Scalar);

    /// Provides a slice of the underlying data.
    fn as_slice(&self) -> &[f64];

    /// Provides a mutable slice of the underlying data.
    fn as_mut_slice(&mut self) -> &mut [Self::Scalar];

    /// Computes the dot product of `self` with another vector `other`.
    ///
    /// # Example
    /// 
    /// ```rust
    /// use hydra::linalg::vector::traits::Vector;
    /// let vec1: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let vec2: Vec<f64> = vec![4.0, 5.0, 6.0];
    /// let dot_product = vec1.dot(&vec2);
    /// assert_eq!(dot_product, 32.0);
    /// ```
    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar;

    /// Computes the Euclidean norm (L2 norm) of the vector.
    ///
    /// # Example
    /// ```rust
    /// use hydra::linalg::vector::traits::Vector;
    /// let vec: Vec<f64> = vec![3.0, 4.0];
    /// let norm = vec.norm();
    /// assert_eq!(norm, 5.0);
    /// ```
    fn norm(&self) -> Self::Scalar;

    /// Scales the vector by multiplying each element by the scalar `scalar`.
    fn scale(&mut self, scalar: Self::Scalar);

    /// Performs the operation `self = a * x + self`, also known as AXPY.
    fn axpy(&mut self, a: Self::Scalar, x: &dyn Vector<Scalar = Self::Scalar>);

    /// Adds another vector `other` to `self` element-wise.
    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Multiplies `self` by another vector `other` element-wise.
    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Divides `self` by another vector `other` element-wise.
    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Computes the cross product with another vector `other` (for 3D vectors only).
    ///
    /// # Errors
    /// Returns an error if the vectors are not 3-dimensional.
    fn cross(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) -> Result<(), &'static str>;

    /// Computes the sum of all elements in the vector.
    fn sum(&self) -> Self::Scalar;

    /// Returns the maximum element of the vector.
    fn max(&self) -> Self::Scalar;

    /// Returns the minimum element of the vector.
    fn min(&self) -> Self::Scalar;

    /// Returns the mean value of the vector.
    fn mean(&self) -> Self::Scalar;

    /// Returns the variance of the vector.
    fn variance(&self) -> Self::Scalar;
}
```

---

Here are the instructions again to help outline the problem we are trying to solve once more, now that you have ready the source code completely.

We are in the process of implementing the following roadmap for Hydra. 

1) There are several problems with the current source and we want to implement as much of the skeleton of the below roadmap in advance, and then develop and test it as we go. 
2) Please use the source code included after this roadmap as a starting point, and then analyze the problem, 
3) Develop a comprehensive solution in the form of complete source code with documentation and test cases, storing these outputs in your memory for this conversation.
4) Provide a narrative output of the enhancements to each file, appropriately summarizing the changes in terms of a detailed itemized list of changes with sufficient details to remind us of the complete context of the comprehensive solution we developed and stored in memory for Step 3 of these instructions.