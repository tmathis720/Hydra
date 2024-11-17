Based on the following summary of changes, as well as the current compiler errors we are facing, please provide complete corrected version of the source code provided below as a starting point.

**Summary of Changes to the `Equation` Module**

During our conversation, we focused on addressing several compiler errors related to the `Equation` module and its associated components. The primary goal was to resolve type mismatches, implement necessary traits, and ensure consistency across the module while maintaining the use of custom wrapper types like `Vector3` and `Scalar`. Below is a detailed summary of the changes made:

1. **Introduction of Custom Wrapper Structs (`Vector3`, `Scalar`, etc.):**
   - To enable operator overloading and implement mathematical operations essential for numerical computations, we introduced custom wrapper structs such as `Vector3`, `Vector2`, `Scalar`, and `Tensor3x3`.
   - These structs wrap around primitive types (e.g., `[f64; 3]`, `f64`) and allow us to implement traits like `AddAssign`, `Mul`, and custom methods required for vector and scalar operations.
   - This change overcomes Rust's orphan rule, which prevents us from implementing foreign traits on foreign types, thereby enabling us to perform element-wise operations and use standard mathematical expressions in the code.

2. **Updates to the `Section<T>` Struct in `src/domain/section.rs`:**
   - The `Section<T>` struct was updated to work with the custom wrapper types instead of primitive arrays or scalars.
   - Trait bounds for `T` were specified to include `AddAssign` and `Mul<f64, Output = T>`, ensuring that types used in `Section<T>` support the necessary mathematical operations.
   - Methods like `set_data`, `restrict`, and `update_with_derivative` were adjusted to handle the new types, allowing for consistent data manipulation and updates.

3. **Modifications in `Fields` and `Fluxes` Structs:**
   - In `src/equation/fields.rs`, the `Fields` struct now uses `Section<Scalar>`, `Section<Vector3>`, and `Section<Tensor3x3>` for scalar, vector, and tensor fields, respectively.
   - The `Fluxes` struct similarly uses these custom types for momentum, energy, and turbulence fluxes.
   - Methods for setting and retrieving field values were updated to accommodate the wrapper types, ensuring consistency and type safety across the module.

4. **Adjustments in Equation Implementations:**
   - In files like `src/equation/equation.rs`, `src/equation/momentum_equation.rs`, and `src/equation/energy_equation.rs`, calculations now use the custom wrapper types for consistency.
   - Mathematical operations within these files were updated to utilize the methods and traits implemented for the custom types.
   - Boundary condition applications were modified to match the new data types, ensuring that boundary handling is consistent with field computations.

5. **Gradient Calculation Updates:**
   - The gradient calculation in `src/equation/gradient/gradient_calc.rs` was modified to use `Section<Scalar>` and `Section<Vector3>`.
   - The `GradientMethod` trait and its implementations now work with the custom wrapper types, ensuring that gradient computations are consistent with the rest of the module.
   - Methods like `calculate_gradient` were adjusted to handle the new types and ensure correct mathematical operations.

6. **Handling of Trait Bounds and Generic Types:**
   - Trait bounds were carefully specified to ensure that types used in generic structs and functions satisfy all required traits.
   - For example, the `TimeStepper` and `TimeDependentProblem` traits now include bounds that guarantee necessary methods and operations are available on generic types.
   - This change helps prevent compiler errors related to missing trait implementations and ensures type safety.

7. **Improved Error Handling:**
   - Error handling was enhanced by returning `Result` types in methods that could fail, allowing for better management of exceptional cases.
   - This change increases the robustness of the module and makes it easier to debug and maintain.

**Architecture and Design Decisions Affecting Module Structure and Function**

1. **Use of Custom Wrapper Types:**
   - The decision to introduce custom wrapper structs was crucial for enabling operator overloading and element-wise mathematical operations.
   - By owning these types, we could implement necessary traits that are not possible with primitive array types due to Rust's orphan rule.
   - This design choice improves code readability and maintainability by allowing intuitive mathematical expressions throughout the module.

2. **Consistency Across the Module:**
   - Ensuring that all components of the `Equation` module use the same data types promotes consistency and reduces the likelihood of type mismatches.
   - The custom wrapper types serve as a unified way to represent vectors, scalars, and tensors, simplifying the interfaces between different parts of the module.

3. **Enhanced Trait Implementations:**
   - Implementing traits like `AddAssign` and `Mul` for our custom types allowed us to perform mathematical operations essential for finite element and finite volume methods.
   - This design decision facilitated the implementation of numerical methods without compromising code clarity or safety.

4. **Modular and Extensible Design:**
   - The `Equation` module was designed to be modular, with separate files and structs handling different physical equations (e.g., momentum, energy).
   - By abstracting common functionalities (e.g., gradient computation, boundary handling), we made the module more extensible for future additions or modifications.
   - This modularity also improves code organization, making it easier to navigate and understand.

5. **Use of Traits and Dynamic Dispatch:**
   - Traits like `PhysicalEquation` and `GradientMethod` were used with trait objects (`Box<dyn Trait>`) to allow for flexibility in choosing different implementations at runtime.
   - This design allows the `EquationManager` to manage multiple equations and gradient methods without being tightly coupled to specific implementations.
   - It enhances the module's ability to accommodate various physical models and numerical methods.

6. **Separation of Concerns:**
   - The module structure was organized to separate different concerns, such as field management, equation assembly, and time stepping.
   - This separation improves code maintainability and makes individual components easier to test and modify independently.
   - For example, changes to the gradient calculation method do not directly impact the way fields are stored or how time stepping is performed.

7. **Thread Safety and Concurrency Considerations:**
   - Data structures like `Section<T>` use `DashMap` to provide thread-safe access to data, which is important for parallel computations common in numerical simulations.
   - The code is designed to accommodate parallelism, which can enhance performance when working with large meshes or complex computations.

8. **Improved Error Propagation and Handling:**
   - Methods that could fail now return `Result` types with `Box<dyn Error>`, allowing errors to be propagated up the call stack.
   - This approach provides more informative error messages and makes debugging easier, enhancing the reliability of the module.

9. **Integration of Boundary Conditions:**
   - Boundary conditions are handled consistently across equations using the `BoundaryConditionHandler`, which manages different types of boundary conditions.
   - The design allows for flexible application of boundary conditions, including time-dependent and function-based conditions.

10. **Flexible Time Stepping Mechanisms:**
    - The `TimeStepper` trait was designed to support different time-stepping methods (e.g., explicit Euler, backward Euler).
    - This flexibility allows the simulation to choose the most appropriate time-stepping method for the problem at hand.

**Conclusion**

The changes made to the `Equation` module were aimed at resolving compiler errors and enhancing the overall architecture and design. By introducing custom wrapper types and carefully implementing necessary traits, we enabled mathematical operations essential for numerical methods in computational physics. The design decisions focused on creating a consistent, modular, and extensible module structure that can be maintained and expanded upon in the future. This approach ensures that the module is robust, flexible, and capable of accommodating complex simulations involving various physical equations.

---

**Note:** The decision to stick with custom wrapper types, despite the added complexity, was made to maintain control over the data structures and ensure that they meet the specific requirements of our simulation framework. This choice aligns with the need for fine-grained control over mathematical operations and type implementations in a specialized computational environment.

---

Here are the compiler errors we are encountering currently:

```bash
error[E0308]: mismatched types
  --> src\time_stepping\methods\euler.rs:46:41
   |
46 |         state.update_state(&derivative, dt);
   |               ------------              ^^ expected `f64`, found associated type
   |               |
   |               arguments to this method are incorrect
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
note: method defined here
  --> src\equation\fields.rs:6:8
   |
6  |     fn update_state(&mut self, derivative: &Self, dt: f64);
   |        ^^^^^^^^^^^^
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
24 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:22:9
   |
21 |     fn current_time(&self) -> P::Time {
   |                               ------- expected `<P as TimeDependentProblem>::Time` because of return type      
22 |         self.current_time
   |         ^^^^^^^^^^^^^^^^^ expected associated type, found `f64`
   |
   = note: expected associated type `<P as TimeDependentProblem>::Time`
                         found type `f64`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:26:29
   |
26 |         self.current_time = time;
   |         -----------------   ^^^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:36:26
   |
36 |         self.time_step = dt;
   |         --------------   ^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0277]: cannot add-assign `<P as TimeDependentProblem>::Time` to `f64`
  --> src\time_stepping\methods\backward_euler.rs:47:27
   |
47 |         self.current_time += dt;
   |                           ^^ no implementation for `f64 += <P as TimeDependentProblem>::Time`
   |
   = help: the trait `AddAssign<<P as TimeDependentProblem>::Time>` is not implemented for `f64`
help: consider extending the `where` clause, but there might be an alternative better way to express this requirement
   |
19 |     P: TimeDependentProblem, f64: AddAssign<<P as TimeDependentProblem>::Time>
   |                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

error[E0308]: mismatched types
   --> src\time_stepping\methods\backward_euler.rs:58:12
    |
58  |         Ok(self.time_step)
    |         -- ^^^^^^^^^^^^^^ expected associated type, found `f64`
    |         |
    |         arguments to this enum variant are incorrect
    |
    = note: expected associated type `<P as TimeDependentProblem>::Time`
                          found type `f64`
help: the type constructed contains `f64` due to the type of the argument passed
   --> src\time_stepping\methods\backward_euler.rs:58:9
    |
58  |         Ok(self.time_step)
    |         ^^^--------------^
    |            |
    |            this argument influences the type of `Ok`
note: tuple variant defined here
   --> C:\Users\tmath\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\result.rs:531:5
    |
531 |     Ok(#[stable(feature = "rust1", since = "1.0.0")] T),
    |     ^^
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
    |
19  |     P: TimeDependentProblem<Time = f64>,
    |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:62:29
   |
62 |         self.current_time = start_time;
   |         -----------------   ^^^^^^^^^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:66:26
   |
66 |         self.time_step = dt;
   |         --------------   ^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:70:9
   |
69 |     fn get_time_step(&self) -> P::Time {
   |                                ------- expected `<P as TimeDependentProblem>::Time` because of return type     
70 |         self.time_step
   |         ^^^^^^^^^^^^^^ expected associated type, found `f64`
   |
   = note: expected associated type `<P as TimeDependentProblem>::Time`
                         found type `f64`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0277]: `&Vector3` is not an iterator
  --> src\equation\equation.rs:24:45
   |
24 |                     .map(|vel| vel.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>())
   |                                             ^^^ `&Vector3` is not an iterator
   |
   = help: the trait `Iterator` is not implemented for `&Vector3`, which is required by `&Vector3: IntoIterator`    
   = note: required for `&Vector3` to implement `IntoIterator`

error[E0599]: `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
  --> src\equation\equation.rs:24:58
   |
24 |                     .map(|vel| vel.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>())
   |                                                          ^^^ `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
   |
  ::: C:\Users\tmath\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\iter\adapters\zip.rs:15:1
   |
15 | pub struct Zip<A, B> {
   | -------------------- doesn't satisfy `_: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `&Vector3: Iterator`
           which is required by `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           which is required by `&mut std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`

error[E0608]: cannot index into a value of type `Vector3`
  --> src\equation\gradient\gradient_calc.rs:69:63
   |
69 |                         grad_phi[i] += delta_phi * flux_vector[i];
   |                                                               ^^^

error[E0308]: mismatched types
   --> src\equation\gradient\gradient_calc.rs:73:64
    |
73  | ...   self.apply_boundary_condition(face, phi_c, flux_vector, time, &mut grad_phi, boundary_handler, geome... 
    |            ------------------------              ^^^^^^^^^^^ expected `[f64; 3]`, found `Vector3`
    |            |
    |            arguments to this method are incorrect
    |
note: method defined here
   --> src\equation\gradient\gradient_calc.rs:102:8
    |
102 |     fn apply_boundary_condition(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
106 |         flux_vector: [f64; 3],
    |         ---------------------

error[E0277]: `&Vector3` is not an iterator
  --> src\equation\momentum_equation.rs:93:65
   |
93 |                 let velocity_dot_normal = avg_velocity.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>(); 
   |                                                                 ^^^ `&Vector3` is not an iterator
   |
   = help: the trait `Iterator` is not implemented for `&Vector3`, which is required by `&Vector3: IntoIterator`    
   = note: required for `&Vector3` to implement `IntoIterator`

error[E0599]: `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
  --> src\equation\momentum_equation.rs:93:78
   |
93 |                 let velocity_dot_normal = avg_velocity.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>(); 
   |                                                                              ^^^ `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
   |
  ::: C:\Users\tmath\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\iter\adapters\zip.rs:15:1
   |
15 | pub struct Zip<A, B> {
   | -------------------- doesn't satisfy `_: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `&Vector3: Iterator`
           which is required by `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           which is required by `&mut std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`

error[E0502]: cannot borrow `*self` as immutable because it is also borrowed as mutable
  --> src\equation\manager.rs:54:19
   |
52 |         let time_stepper = &mut self.time_stepper;
   |                            ---------------------- mutable borrow occurs here
53 |         time_stepper
54 |             .step(self, time_step, current_time, fields)
   |              ---- ^^^^ immutable borrow occurs here
   |              |
   |              mutable borrow later used by call

Some errors have detailed explanations: E0277, E0308, E0502, E0599, E0608.
For more information about an error, try `rustc --explain E0277`.
error: could not compile `hydra` (lib) due to 16 previous errors
```

---

Here is the source code where the errors occur, or where additional context is required for the solution in other source codes.

`src/domain/section.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use std::ops::{AddAssign, Mul};

#[derive(Clone, Copy)]
pub struct Vector3(pub [f64; 3]);

impl AddAssign for Vector3 {
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vector3([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}

impl Vector3 {
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}

#[derive(Clone, Copy)]
pub struct Tensor3x3(pub [[f64; 3]; 3]);

impl AddAssign for Tensor3x3 {
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            for j in 0..3 {
                self.0[i][j] += other.0[i][j];
            }
        }
    }
}

impl Mul<f64> for Tensor3x3 {
    type Output = Tensor3x3;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.0[i][j] * rhs;
            }
        }
        Tensor3x3(result)
    }
}

#[derive(Clone, Copy)]
pub struct Scalar(pub f64);

impl AddAssign for Scalar {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Mul<f64> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: f64) -> Self::Output {
        Scalar(self.0 * rhs)
    }
}

#[derive(Clone, Copy)]
pub struct Vector2(pub [f64; 2]);

impl AddAssign for Vector2 {
    fn add_assign(&mut self, other: Self) {
        for i in 0..2 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector2 {
    type Output = Vector2;

    fn mul(self, rhs: f64) -> Self::Output {
        Vector2([self.0[0] * rhs, self.0[1] * rhs])
    }
}

/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.
#[derive(Clone)]
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.
    pub data: DashMap<MeshEntity, T>,
}

impl<T> Section<T>
where
    T: Clone + AddAssign + Mul<f64, Output = T> + Send + Sync,
{
    /// Creates a new `Section` with an empty data map.
    pub fn new() -> Self {
        Section {
            data: DashMap::new(),
        }
    }

    /// Sets the data associated with a given `MeshEntity`.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Restricts the data for a given `MeshEntity` by returning an immutable copy of the data
    /// associated with the `entity`, if it exists.
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T> {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Applies the given function in parallel to update all data values in the section.
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
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

    /// Updates the section data by adding the derivative multiplied by dt
    pub fn update_with_derivative(&self, derivative: &Section<T>, dt: f64) {
        for entry in derivative.data.iter() {
            let entity = entry.key();
            let deriv_value = entry.value().clone() * dt;
            if let Some(mut state_value) = self.data.get_mut(entity) {
                *state_value.value_mut() += deriv_value;
            } else {
                self.data.insert(*entity, deriv_value);
            }
        }
    }

    /// Retrieves all `MeshEntity` objects associated with the section.
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves all data stored in the section as immutable copies.
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    fn clone(&self) -> Self {
        Section {
            data: self.data.clone(),
        }
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

`src/boundary/bc_handler.rs`

```rust
use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use lazy_static::lazy_static;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use crate::boundary::mixed::MixedBC;
use crate::boundary::cauchy::CauchyBC;
use faer::MatMut;

pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// BoundaryCondition represents various types of boundary conditions
/// that can be applied to mesh entities.
#[derive(Clone)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    DirichletFn(BoundaryConditionFn),
    NeumannFn(BoundaryConditionFn),
}

/// The BoundaryConditionHandler struct is responsible for managing
/// boundary conditions associated with specific mesh entities.
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

lazy_static! {
    static ref GLOBAL_BC_HANDLER: Arc<RwLock<BoundaryConditionHandler>> =
        Arc::new(RwLock::new(BoundaryConditionHandler::new()));
}

impl BoundaryConditionHandler {
    /// Creates a new BoundaryConditionHandler with an empty map to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    pub fn global() -> Arc<RwLock<BoundaryConditionHandler>> {
        GLOBAL_BC_HANDLER.clone()
    }

    /// Sets a boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Retrieves the boundary condition applied to a specific mesh entity, if it exists.
    pub fn get_bc(&self, entity: &MeshEntity) -> Option<BoundaryCondition> {
        self.conditions.get(entity).map(|entry| entry.clone())
    }

    pub fn get_boundary_faces(&self) -> Vec<MeshEntity> {
        self.conditions.iter()
            .map(|entry| entry.key().clone()) // Extract the keys (MeshEntities) from the map
            .filter(|entity| matches!(entity, MeshEntity::Face(_))) // Filter for Face entities
            .collect()
    }

    /// Applies the boundary conditions to the system matrices and right-hand side vectors.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        boundary_entities: &[MeshEntity],
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        for entity in boundary_entities {
            if let Some(bc) = self.get_bc(entity) {
                let index = *entity_to_index.get(entity).unwrap();
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let robin_bc = RobinBC::new();
                        robin_bc.apply_robin(matrix, rhs, index, alpha, beta);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, value);
                    }
                    BoundaryCondition::Mixed { gamma, delta } => {
                        let mixed_bc = MixedBC::new();
                        mixed_bc.apply_mixed(matrix, rhs, index, gamma, delta);
                    }
                    BoundaryCondition::Cauchy { lambda, mu } => {
                        let cauchy_bc = CauchyBC::new();
                        cauchy_bc.apply_cauchy(matrix, rhs, index, lambda, mu);
                    }
                }
            }
        }
    }
}


/// The BoundaryConditionApply trait defines the `apply` method, which is used to apply 
/// a boundary condition to a given mesh entity.
pub trait BoundaryConditionApply {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    );
}

impl BoundaryConditionApply for BoundaryCondition {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        let index = *entity_to_index.get(entity).unwrap();
        match self {
            BoundaryCondition::Dirichlet(value) => {
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, *value);
            }
            BoundaryCondition::Neumann(flux) => {
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, *flux);
            }
            BoundaryCondition::Robin { alpha, beta } => {
                let robin_bc = RobinBC::new();
                robin_bc.apply_robin(matrix, rhs, index, *alpha, *beta);
            }
            BoundaryCondition::DirichletFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
            }
            BoundaryCondition::NeumannFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, value);
            }
            BoundaryCondition::Mixed { gamma, delta } => {
                let mixed_bc = MixedBC::new();
                mixed_bc.apply_mixed(matrix, rhs, index, *gamma, *delta);
            }
            BoundaryCondition::Cauchy { lambda, mu } => {
                let cauchy_bc = CauchyBC::new();
                cauchy_bc.apply_cauchy(matrix, rhs, index, *lambda, *mu);
            }
        }
    }
}
```

---

`src/equation/mod.rs`

```rust
use fields::{Fields, Fluxes};

use crate::{
    boundary::bc_handler::BoundaryConditionHandler,
    Mesh,
};

pub mod equation;
pub mod reconstruction;
pub mod gradient;
pub mod flux_limiter;

pub mod fields;
pub mod manager;
pub mod energy_equation;
pub mod momentum_equation;

pub trait PhysicalEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}

// Removed generic parameter `T` from `PhysicalEquation`
// Updated imports and structures to align with new `Fields` implementation
```

---

`src/equation/fields.rs`

```rust
use rustc_hash::FxHashMap;
use crate::{domain::Section, MeshEntity};
use super::super::domain::section::{Vector3, Tensor3x3, Scalar, Vector2};

pub trait UpdateState {
    fn update_state(&mut self, derivative: &Self, dt: f64);
}

#[derive(Clone)]
pub struct Fields {
    pub scalar_fields: FxHashMap<String, Section<Scalar>>,
    pub vector_fields: FxHashMap<String, Section<Vector3>>,
    pub tensor_fields: FxHashMap<String, Section<Tensor3x3>>,
}

impl Fields {
    pub fn new() -> Self {
        Self {
            scalar_fields: FxHashMap::default(),
            vector_fields: FxHashMap::default(),
            tensor_fields: FxHashMap::default(),
        }
    }

    pub fn get_scalar_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Scalar> {
        self.scalar_fields.get(name)?.restrict(entity)
    }

    pub fn set_scalar_field_value(&mut self, name: &str, entity: MeshEntity, value: Scalar) {
        if let Some(field) = self.scalar_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.scalar_fields.insert(name.to_string(), field);
        }
    }

    pub fn get_vector_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Vector3> {
        self.vector_fields.get(name)?.restrict(entity)
    }

    pub fn set_vector_field_value(&mut self, name: &str, entity: MeshEntity, value: Vector3) {
        if let Some(field) = self.vector_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.vector_fields.insert(name.to_string(), field);
        }
    }

    pub fn get_tensor_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Tensor3x3> {
        self.tensor_fields.get(name)?.restrict(entity)
    }

    pub fn set_tensor_field_value(&mut self, name: &str, entity: MeshEntity, value: Tensor3x3) {
        if let Some(field) = self.tensor_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.tensor_fields.insert(name.to_string(), field);
        }
    }

    pub fn add_turbulence_fields(&mut self) {
        self.scalar_fields.insert("turbulent_viscosity".to_string(), Section::new());
        self.scalar_fields.insert("k_field".to_string(), Section::new());
        self.scalar_fields.insert("omega_field".to_string(), Section::new());
    }

    /// Updates the derivative fields based on the computed fluxes
    pub fn update_from_fluxes(&mut self, _fluxes: &Fluxes) {
        // Implement logic to update derivative fields from fluxes
        // This is domain-specific and should be implemented accordingly
    }
}

impl UpdateState for Fields {
    fn update_state(&mut self, derivative: &Fields, dt: f64) {
        // Update scalar fields
        for (key, section) in &derivative.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get_mut(key) {
                state_section.update_with_derivative(section, dt);
            }
        }

        // Update vector fields
        for (key, section) in &derivative.vector_fields {
            if let Some(state_section) = self.vector_fields.get_mut(key) {
                state_section.update_with_derivative(section, dt);
            }
        }

        // Update tensor fields if needed
        for (key, section) in &derivative.tensor_fields {
            if let Some(state_section) = self.tensor_fields.get_mut(key) {
                state_section.update_with_derivative(section, dt);
            }
        }
    }
}

pub struct Fluxes {
    pub momentum_fluxes: Section<Vector3>,
    pub energy_fluxes: Section<Scalar>,
    pub turbulence_fluxes: Section<Vector2>,
}

impl Fluxes {
    pub fn new() -> Self {
        Self {
            momentum_fluxes: Section::new(),
            energy_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        }
    }

    pub fn add_momentum_flux(&mut self, entity: MeshEntity, value: Vector3) {
        if let Some(mut current) = self.momentum_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.momentum_fluxes.set_data(entity, value);
        }
    }

    pub fn add_energy_flux(&mut self, entity: MeshEntity, value: Scalar) {
        if let Some(mut current) = self.energy_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.energy_fluxes.set_data(entity, value);
        }
    }

    pub fn add_turbulence_flux(&mut self, entity: MeshEntity, value: Vector2) {
        if let Some(mut current) = self.turbulence_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.turbulence_fluxes.set_data(entity, value);
        }
    }
}
```

---

`src/equation/manager.rs`

```rust
use crate::{
    boundary::bc_handler::BoundaryConditionHandler,
    domain::mesh::Mesh,
    time_stepping::{TimeDependentProblem, TimeStepper, TimeSteppingError},
    Matrix,
};
use super::{Fields, Fluxes, PhysicalEquation};
use std::sync::{Arc, RwLock};

pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
    time_stepper: Box<dyn TimeStepper<Self>>,
    domain: Arc<RwLock<Mesh>>,
    boundary_handler: Arc<RwLock<BoundaryConditionHandler>>,
}

impl EquationManager {
    pub fn new(
        time_stepper: Box<dyn TimeStepper<Self>>,
        domain: Arc<RwLock<Mesh>>,
        boundary_handler: Arc<RwLock<BoundaryConditionHandler>>,
    ) -> Self {
        Self {
            equations: Vec::new(),
            time_stepper,
            domain,
            boundary_handler,
        }
    }

    pub fn add_equation<E: PhysicalEquation + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    pub fn assemble_all(
        &self,
        fields: &Fields,
        fluxes: &mut Fluxes,
    ) {
        let current_time = self.time_stepper.current_time();
        let domain = self.domain.read().unwrap();
        let boundary_handler = self.boundary_handler.read().unwrap();
        for equation in &self.equations {
            equation.assemble(&domain, fields, fluxes, &boundary_handler, current_time);
        }
    }

    pub fn step(&mut self, fields: &mut Fields) {
        let current_time = self.time_stepper.current_time();
        let time_step = self.time_stepper.get_time_step();
        // Avoid borrowing self.time_stepper both mutably and immutably
        let time_stepper = &mut self.time_stepper;
        time_stepper
            .step(self, time_step, current_time, fields)
            .expect("Time-stepping failed");
    }
}

impl TimeDependentProblem for EquationManager {
    type State = Fields;
    type Time = f64;

    fn compute_rhs(
        &self,
        _time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Create a new Fluxes object to store the computed fluxes
        let mut fluxes = Fluxes::new();

        // Assemble all equations to compute the fluxes
        let _domain = self.domain.read().unwrap();
        let _boundary_handler = self.boundary_handler.read().unwrap();
        self.assemble_all(
            state,
            &mut fluxes,
        );

        // Compute the derivative (RHS) based on the fluxes
        derivative.update_from_fluxes(&fluxes);

        Ok(())
    }

    fn initial_state(&self) -> Self::State {
        // Initialize fields with appropriate initial conditions
        Fields::new()
    }

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
        // Return assembled system matrix if needed
        None
    }

    fn solve_linear_system(
        &self,
        _matrix: &mut dyn Matrix<Scalar = f64>,
        _state: &mut Self::State,
        _rhs: &Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Implement solver logic to solve the linear system
        Ok(())
    }
}
```

---

`src/equation/equation.rs`

```rust
use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::BoundaryConditionHandler;
use crate::domain::section::{Vector3, Scalar};

pub struct Equation {}

impl Equation {
    pub fn calculate_fluxes(
        &self,
        domain: &Mesh,
        velocity_field: &Section<Vector3>,
        pressure_field: &Section<Scalar>,
        fluxes: &mut Section<Vector3>,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64, // Accept current_time as a parameter
    ) {
        let _ = pressure_field;
        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);

                let velocity_dot_normal = velocity_field
                    .restrict(&face)
                    .map(|vel| vel.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>())
                    .unwrap_or(0.0);

                let flux = Vector3([velocity_dot_normal * area, 0.0, 0.0]);
                fluxes.set_data(face.clone(), flux);

                // Boundary condition logic
                let mut matrix = faer::Mat::<f64>::zeros(1, 1).as_mut();
                let mut rhs = faer::Mat::<f64>::zeros(1, 1).as_mut();
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

`src/equation/momentum_equation.rs`

```rust
use crate::{
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
    geometry::Geometry, Mesh,
};
use super::{
    fields::{Fields, Fluxes},
    PhysicalEquation,
};
use crate::domain::section::{Vector3, Scalar};

pub struct MomentumParameters {
    pub density: f64,
    pub viscosity: f64,
}

pub struct MomentumEquation {
    pub params: MomentumParameters,
}

impl PhysicalEquation for MomentumEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        self.calculate_momentum_fluxes(domain, fields, fluxes, boundary_handler, current_time);
    }
}

impl MomentumEquation {
    pub fn calculate_momentum_fluxes(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        let _ = current_time;
        let mut _geometry = Geometry::new();

        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);

                // Get the cells adjacent to the face
                let cells = domain.get_cells_sharing_face(&face);

                // Initialize variables
                let mut velocity_a = Vector3([0.0; 3]);
                let mut pressure_a = Scalar(0.0);
                let mut velocity_b = Vector3([0.0; 3]);
                let mut pressure_b = Scalar(0.0);

                let mut has_cell_b = false;

                // Iterate over adjacent cells
                let mut iter = cells.iter();
                if let Some(cell_entry) = iter.next() {
                    let cell_a = cell_entry.key().clone();
                    if let Some(vel) = fields.get_vector_field_value("velocity_field", &cell_a) {
                        velocity_a = vel;
                    }
                    if let Some(pres) = fields.get_scalar_field_value("pressure_field", &cell_a) {
                        pressure_a = pres;
                    }
                }
                if let Some(cell_entry) = iter.next() {
                    let cell_b = cell_entry.key().clone();
                    has_cell_b = true;
                    if let Some(vel) = fields.get_vector_field_value("velocity_field", &cell_b) {
                        velocity_b = vel;
                    }
                    if let Some(pres) = fields.get_scalar_field_value("pressure_field", &cell_b) {
                        pressure_b = pres;
                    }
                }

                // Compute convective flux
                let avg_velocity = if has_cell_b {
                    Vector3([
                        0.5 * (velocity_a.0[0] + velocity_b.0[0]),
                        0.5 * (velocity_a.0[1] + velocity_b.0[1]),
                        0.5 * (velocity_a.0[2] + velocity_b.0[2]),
                    ])
                } else {
                    velocity_a
                };

                let velocity_dot_normal = avg_velocity.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();

                let convective_flux = self.params.density * velocity_dot_normal * area;

                // Compute pressure flux
                let pressure_flux = if has_cell_b {
                    0.5 * (pressure_a.0 + pressure_b.0) * area
                } else {
                    pressure_a.0 * area
                };

                // Compute diffusive flux (simplified for demonstration)
                // In practice, this would involve gradients of velocity
                let diffusive_flux = self.params.viscosity * area;

                // Total flux vector (assuming 3D for demonstration)
                let total_flux = Vector3([
                    convective_flux - pressure_flux + diffusive_flux,
                    0.0,
                    0.0,
                ]);

                // Update fluxes
                fluxes.add_momentum_flux(face.clone(), total_flux);

                // Apply boundary conditions
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    match bc {
                        BoundaryCondition::Dirichlet(_value) => {
                            // Apply Dirichlet condition
                            // Adjust fluxes or impose values as necessary
                        }
                        BoundaryCondition::Neumann(_value) => {
                            // Apply Neumann condition
                            // Modify fluxes accordingly
                        }
                        _ => (),
                    }
                }
            }
        }
    }
}
```

---

`src/equation/gradient/gradient_calc.rs`

```rust
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::domain::{mesh::Mesh, MeshEntity, Section};
use crate::geometry::{FaceShape, Geometry};
use crate::equation::gradient::GradientMethod;
use crate::domain::section::{Scalar};
use std::error::Error;

/// Struct for the finite volume gradient calculation method.
///
/// This struct implements the `GradientMethod` trait for finite volume
/// computations of gradient.
pub struct FiniteVolumeGradient;

impl GradientMethod for FiniteVolumeGradient {
    /// Computes the gradient of a scalar field across each cell in the mesh.
    ///
    /// # Parameters
    /// - `mesh`: Reference to the mesh structure containing cell and face connectivity.
    /// - `boundary_handler`: Reference to a handler that manages boundary conditions.
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
        field: &Section<Scalar>,
        cell: &MeshEntity,
        time: f64,
    ) -> Result<[f64; 3], Box<dyn Error>> {
        let phi_c = field.restrict(cell).ok_or("Field value not found for cell")?.0;
        let mut grad_phi = [0.0; 3];
        let cell_vertices = mesh.get_cell_vertices(cell);

        if cell_vertices.is_empty() {
            return Err(format!("Cell {:?} has 0 vertices; cannot compute volume or gradient.", cell).into());
        }

        let volume = geometry.compute_cell_volume(mesh, cell);
        if volume == 0.0 {
            return Err("Cell volume is zero; cannot compute gradient.".into());
        }

        if let Some(faces) = mesh.get_faces_of_cell(cell) {
            for face_entry in faces.iter() {
                let face = face_entry.key();
                let face_vertices = mesh.get_face_vertices(face);
                let face_shape = self.determine_face_shape(face_vertices.len())?;
                let area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
                let normal = geometry.compute_face_normal(mesh, face, cell)
                    .ok_or("Face normal not found")?;
                let flux_vector = normal * area;
                let neighbor_cells = mesh.get_cells_sharing_face(face);

                let nb_cell = neighbor_cells.iter()
                    .find(|neighbor| *neighbor.key() != *cell)
                    .map(|entry| entry.key().clone());

                if let Some(nb_cell) = nb_cell {
                    let phi_nb = field.restrict(&nb_cell).ok_or("Field value not found for neighbor cell")?.0;
                    let delta_phi = phi_nb - phi_c;
                    for i in 0..3 {
                        grad_phi[i] += delta_phi * flux_vector[i];
                    }
                } else {
                    // Pass boundary_handler directly to the function
                    self.apply_boundary_condition(face, phi_c, flux_vector, time, &mut grad_phi, boundary_handler, geometry, mesh)?;
                }
            }

            for i in 0..3 {
                grad_phi[i] /= volume;
            }
        }

        Ok(grad_phi)
    }
}

impl FiniteVolumeGradient {
    /// Applies boundary conditions for a face without a neighboring cell.
    ///
    /// # Parameters
    /// - `face`: The face entity for which boundary conditions are applied.
    /// - `phi_c`: Scalar field value at the current cell.
    /// - `flux_vector`: Scaled normal vector representing face flux direction.
    /// - `time`: Simulation time, required for time-dependent boundary functions.
    /// - `grad_phi`: Accumulator array to which boundary contributions will be added.
    /// - `boundary_handler`: Boundary condition handler.
    /// - `geometry`: Geometry utility for calculations.
    /// - `mesh`: Mesh structure to access cell and face data.
    ///
    /// # Returns
    /// - `Ok(())`: Boundary condition successfully applied.
    /// - `Err(Box<dyn Error>)`: If the boundary condition type is unsupported.
    fn apply_boundary_condition(
        &self,
        face: &MeshEntity,
        phi_c: f64,
        flux_vector: [f64; 3],
        time: f64,
        grad_phi: &mut [f64; 3],
        boundary_handler: &BoundaryConditionHandler,
        geometry: &mut Geometry,
        mesh: &Mesh,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(bc) = boundary_handler.get_bc(face) {
            match bc {
                BoundaryCondition::Dirichlet(value) => {
                    self.apply_dirichlet_boundary(value, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Neumann(flux) => {
                    self.apply_neumann_boundary(flux, flux_vector, grad_phi);
                }
                BoundaryCondition::Robin { alpha: _, beta: _ } => {
                    return Err("Robin boundary condition not implemented for gradient computation".into());
                }
                BoundaryCondition::DirichletFn(fn_bc) => {
                    let coords = geometry.compute_face_centroid(FaceShape::Triangle, &mesh.get_face_vertices(face));
                    let phi_nb = fn_bc(time, &coords);
                    self.apply_dirichlet_boundary(phi_nb, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::NeumannFn(fn_bc) => {
                    let coords = geometry.compute_face_centroid(FaceShape::Triangle, &mesh.get_face_vertices(face));
                    let flux = fn_bc(time, &coords);
                    self.apply_neumann_boundary(flux, flux_vector, grad_phi);
                }
                BoundaryCondition::Mixed { gamma, delta } => {
                    self.apply_mixed_boundary(gamma, delta, phi_c, flux_vector, grad_phi);
                }
                BoundaryCondition::Cauchy { lambda, mu } => {
                    self.apply_cauchy_boundary(lambda, mu, flux_vector, grad_phi);
                }
            }
        }
        Ok(())
    }
    
    /// Applies a Dirichlet boundary condition by adding flux contribution.
    fn apply_dirichlet_boundary(&self, value: f64, phi_c: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        let delta_phi = value - phi_c;
        for i in 0..3 {
            grad_phi[i] += delta_phi * flux_vector[i];
        }
    }
    
    /// Applies a Neumann boundary condition by adding constant flux.
    fn apply_neumann_boundary(&self, flux: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        for i in 0..3 {
            grad_phi[i] += flux * flux_vector[i];
        }
    }
    
    /// Applies a Mixed boundary condition by combining field value and flux.
    fn apply_mixed_boundary(&self, gamma: f64, delta: f64, phi_c: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        let mixed_contrib = gamma * phi_c + delta;
        for i in 0..3 {
            grad_phi[i] += mixed_contrib * flux_vector[i];
        }
    }
    
    /// Applies a Cauchy boundary condition by adding lambda to flux and mu to field.
    fn apply_cauchy_boundary(&self, lambda: f64, mu: f64, flux_vector: [f64; 3], grad_phi: &mut [f64; 3]) {
        for i in 0..3 {
            grad_phi[i] += lambda * flux_vector[i] + mu;
        }
    }
    
    /// Determines face shape based on vertex count.
    fn determine_face_shape(&self, vertex_count: usize) -> Result<FaceShape, Box<dyn Error>> {
        match vertex_count {
            3 => Ok(FaceShape::Triangle),
            4 => Ok(FaceShape::Quadrilateral),
            _ => Err(format!(
                "Unsupported face shape with {} vertices for gradient computation",
                vertex_count
            )
            .into()),
        }
    }
}
```

---

`src/time_stepping/ts.rs`

```rust
use crate::linalg::Matrix;

#[derive(Debug)]
pub enum TimeSteppingError {
    InvalidStep,
    SolverError(String),
}

pub trait TimeDependentProblem {
    type State: Clone;
    type Time: Copy + PartialOrd + std::ops::Add<Output = Self::Time>;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

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
        problem: &P,
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

`src/time_stepping/mod.rs`

```rust
pub mod ts;
pub mod methods;
pub mod adaptivity;

pub use ts::{TimeStepper, TimeSteppingError, TimeDependentProblem};
pub use methods::backward_euler::BackwardEuler;
pub use methods::euler::ExplicitEuler;
```

---

`src/time_stepping/methods/backward_euler.rs`

```rust
use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

pub struct BackwardEuler {
    current_time: f64,
    time_step: f64,
}

impl BackwardEuler {
    pub fn new(start_time: f64, time_step: f64) -> Self {
        Self {
            current_time: start_time,
            time_step,
        }
    }
}

impl<P> TimeStepper<P> for BackwardEuler
where
    P: TimeDependentProblem,
{
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        self.time_step = dt;

        let mut matrix = problem
            .get_matrix()
            .ok_or(TimeSteppingError::SolverError("Matrix is required for Backward Euler.".into()))?;
        let mut rhs = state.clone();

        problem.compute_rhs(current_time, state, &mut rhs)?;
        problem.solve_linear_system(matrix.as_mut(), state, &rhs)?;

        // Update the current time
        self.current_time += dt;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
    ) -> Result<P::Time, TimeSteppingError> {
        // Adaptive step logic (placeholder)
        Ok(self.time_step)
    }

    fn set_time_interval(&mut self, start_time: P::Time, _end_time: P::Time) {
        self.current_time = start_time;
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }
    
    fn get_time_step(&self) -> P::Time {
        self.time_step
    }
}
```

---

`src/time_stepping/methods/euler.rs`

```rust
use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
use crate::equation::fields::UpdateState;

pub struct ExplicitEuler<P: TimeDependentProblem> {
    current_time: P::Time,
    time_step: P::Time,
    start_time: P::Time,
    end_time: P::Time,
}

impl<P: TimeDependentProblem> ExplicitEuler<P> {
    pub fn new(time_step: P::Time, start_time: P::Time, end_time: P::Time) -> Self {
        Self {
            current_time: start_time,
            time_step,
            start_time,
            end_time,
        }
    }
}

impl<P> TimeStepper<P> for ExplicitEuler<P>
where
    P: TimeDependentProblem,
    P::State: UpdateState,
{
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut derivative = problem.initial_state(); // Initialize derivative
        problem.compute_rhs(current_time, state, &mut derivative)?;

        // Update the state: state = state + dt * derivative
        state.update_state(&derivative, dt);

        self.current_time = current_time + dt;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
    ) -> Result<P::Time, TimeSteppingError> {
        // For simplicity, not implemented
        unimplemented!()
    }

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time) {
        self.start_time = start_time;
        self.end_time = end_time;
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }

    fn get_time_step(&self) -> P::Time {
        self.time_step
    }
}
```

---

Here again are the compiler errors associated with some of the above source code:

```bash
error[E0308]: mismatched types
  --> src\time_stepping\methods\euler.rs:46:41
   |
46 |         state.update_state(&derivative, dt);
   |               ------------              ^^ expected `f64`, found associated type
   |               |
   |               arguments to this method are incorrect
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
note: method defined here
  --> src\equation\fields.rs:6:8
   |
6  |     fn update_state(&mut self, derivative: &Self, dt: f64);
   |        ^^^^^^^^^^^^
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
24 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:22:9
   |
21 |     fn current_time(&self) -> P::Time {
   |                               ------- expected `<P as TimeDependentProblem>::Time` because of return type      
22 |         self.current_time
   |         ^^^^^^^^^^^^^^^^^ expected associated type, found `f64`
   |
   = note: expected associated type `<P as TimeDependentProblem>::Time`
                         found type `f64`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:26:29
   |
26 |         self.current_time = time;
   |         -----------------   ^^^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:36:26
   |
36 |         self.time_step = dt;
   |         --------------   ^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0277]: cannot add-assign `<P as TimeDependentProblem>::Time` to `f64`
  --> src\time_stepping\methods\backward_euler.rs:47:27
   |
47 |         self.current_time += dt;
   |                           ^^ no implementation for `f64 += <P as TimeDependentProblem>::Time`
   |
   = help: the trait `AddAssign<<P as TimeDependentProblem>::Time>` is not implemented for `f64`
help: consider extending the `where` clause, but there might be an alternative better way to express this requirement
   |
19 |     P: TimeDependentProblem, f64: AddAssign<<P as TimeDependentProblem>::Time>
   |                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

error[E0308]: mismatched types
   --> src\time_stepping\methods\backward_euler.rs:58:12
    |
58  |         Ok(self.time_step)
    |         -- ^^^^^^^^^^^^^^ expected associated type, found `f64`
    |         |
    |         arguments to this enum variant are incorrect
    |
    = note: expected associated type `<P as TimeDependentProblem>::Time`
                          found type `f64`
help: the type constructed contains `f64` due to the type of the argument passed
   --> src\time_stepping\methods\backward_euler.rs:58:9
    |
58  |         Ok(self.time_step)
    |         ^^^--------------^
    |            |
    |            this argument influences the type of `Ok`
note: tuple variant defined here
   --> C:\Users\tmath\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\result.rs:531:5
    |
531 |     Ok(#[stable(feature = "rust1", since = "1.0.0")] T),
    |     ^^
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
    |
19  |     P: TimeDependentProblem<Time = f64>,
    |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:62:29
   |
62 |         self.current_time = start_time;
   |         -----------------   ^^^^^^^^^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:66:26
   |
66 |         self.time_step = dt;
   |         --------------   ^^ expected `f64`, found associated type
   |         |
   |         expected due to the type of this binding
   |
   = note:         expected type `f64`
           found associated type `<P as TimeDependentProblem>::Time`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0308]: mismatched types
  --> src\time_stepping\methods\backward_euler.rs:70:9
   |
69 |     fn get_time_step(&self) -> P::Time {
   |                                ------- expected `<P as TimeDependentProblem>::Time` because of return type     
70 |         self.time_step
   |         ^^^^^^^^^^^^^^ expected associated type, found `f64`
   |
   = note: expected associated type `<P as TimeDependentProblem>::Time`
                         found type `f64`
help: consider constraining the associated type `<P as TimeDependentProblem>::Time` to `f64`
   |
19 |     P: TimeDependentProblem<Time = f64>,
   |                            ++++++++++++

error[E0277]: `&Vector3` is not an iterator
  --> src\equation\equation.rs:24:45
   |
24 |                     .map(|vel| vel.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>())
   |                                             ^^^ `&Vector3` is not an iterator
   |
   = help: the trait `Iterator` is not implemented for `&Vector3`, which is required by `&Vector3: IntoIterator`    
   = note: required for `&Vector3` to implement `IntoIterator`

error[E0599]: `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
  --> src\equation\equation.rs:24:58
   |
24 |                     .map(|vel| vel.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>())
   |                                                          ^^^ `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
   |
  ::: C:\Users\tmath\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\iter\adapters\zip.rs:15:1
   |
15 | pub struct Zip<A, B> {
   | -------------------- doesn't satisfy `_: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `&Vector3: Iterator`
           which is required by `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           which is required by `&mut std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`

error[E0608]: cannot index into a value of type `Vector3`
  --> src\equation\gradient\gradient_calc.rs:69:63
   |
69 |                         grad_phi[i] += delta_phi * flux_vector[i];
   |                                                               ^^^

error[E0308]: mismatched types
   --> src\equation\gradient\gradient_calc.rs:73:64
    |
73  | ...   self.apply_boundary_condition(face, phi_c, flux_vector, time, &mut grad_phi, boundary_handler, geome... 
    |            ------------------------              ^^^^^^^^^^^ expected `[f64; 3]`, found `Vector3`
    |            |
    |            arguments to this method are incorrect
    |
note: method defined here
   --> src\equation\gradient\gradient_calc.rs:102:8
    |
102 |     fn apply_boundary_condition(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^
...
106 |         flux_vector: [f64; 3],
    |         ---------------------

error[E0277]: `&Vector3` is not an iterator
  --> src\equation\momentum_equation.rs:93:65
   |
93 |                 let velocity_dot_normal = avg_velocity.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>(); 
   |                                                                 ^^^ `&Vector3` is not an iterator
   |
   = help: the trait `Iterator` is not implemented for `&Vector3`, which is required by `&Vector3: IntoIterator`    
   = note: required for `&Vector3` to implement `IntoIterator`

error[E0599]: `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
  --> src\equation\momentum_equation.rs:93:78
   |
93 |                 let velocity_dot_normal = avg_velocity.0.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>(); 
   |                                                                              ^^^ `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>` is not an iterator
   |
  ::: C:\Users\tmath\.rustup\toolchains\stable-x86_64-pc-windows-msvc\lib/rustlib/src/rust\library\core\src\iter\adapters\zip.rs:15:1
   |
15 | pub struct Zip<A, B> {
   | -------------------- doesn't satisfy `_: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `&Vector3: Iterator`
           which is required by `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           `std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`
           which is required by `&mut std::iter::Zip<std::slice::Iter<'_, f64>, &Vector3>: Iterator`

error[E0502]: cannot borrow `*self` as immutable because it is also borrowed as mutable
  --> src\equation\manager.rs:54:19
   |
52 |         let time_stepper = &mut self.time_stepper;
   |                            ---------------------- mutable borrow occurs here
53 |         time_stepper
54 |             .step(self, time_step, current_time, fields)
   |              ---- ^^^^ immutable borrow occurs here
   |              |
   |              mutable borrow later used by call

Some errors have detailed explanations: E0277, E0308, E0502, E0599, E0608.
For more information about an error, try `rustc --explain E0277`.
error: could not compile `hydra` (lib) due to 16 previous errors
```

