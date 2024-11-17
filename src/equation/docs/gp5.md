Please help address the following compiler errors related to the recent changes. The offending source code is provided below these error output. Please generate complete corrected source code in response for each of the files provided.

Here is the error output:

```bash
error[E0407]: method `time_to_scalar` is not a member of trait `TimeDependentProblem`
  --> src\equation\manager.rs:66:5
   |
66 | /     fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
67 | |         // Convert time to scalar if needed (implementation depends on Vector trait)
68 | |         unimplemented!()
69 | |     }
   | |_____^ not a member of trait `TimeDependentProblem`

error[E0107]: struct takes 0 generic arguments but 1 generic argument was supplied
  --> src\equation\energy_equation.rs:17:18
   |
17 |         fields: &Fields<T>,
   |                  ^^^^^^--- help: remove the unnecessary generics
   |                  |
   |                  expected 0 generic arguments
   |
note: struct defined here, with 0 generic parameters
  --> src\equation\fields.rs:6:12
   |
6  | pub struct Fields {
   |            ^^^^^^

error[E0117]: only traits defined in the current crate can be implemented for arbitrary types
   --> src\domain\section.rs:199:1
    |
199 | impl AddAssign for [f64; 3] {
    | ^^^^^---------^^^^^--------
    | |    |             |
    | |    |             this is not defined in the current crate because arrays are always foreign
    | |    this is not defined in the current crate because this is a foreign trait
    | impl doesn't use only types from inside the current crate
    |
    = note: define and implement a trait or new type instead

error[E0117]: only traits defined in the current crate can be implemented for primitive types
   --> src\domain\section.rs:215:1
    |
215 | impl AddAssign for f64 {
    | ^^^^^---------^^^^^---
    | |    |             |
    | |    |             `f64` is not defined in the current crate
    | |    `f64` is not defined in the current crate
    | impl doesn't use only types from inside the current crate
    |
    = note: define and implement a trait or new type instead

error[E0117]: only traits defined in the current crate can be implemented for arbitrary types
   --> src\domain\section.rs:207:1
    |
207 | impl Mul<f64> for [f64; 3] {
    | ^^^^^--------^^^^^--------
    | |    |            |
    | |    |            this is not defined in the current crate because arrays are always foreign
    | |    `f64` is not defined in the current crate
    | impl doesn't use only types from inside the current crate
    |
    = note: define and implement a trait or new type instead

error[E0117]: only traits defined in the current crate can be implemented for primitive types
   --> src\domain\section.rs:221:1
    |
221 | impl Mul<f64> for f64 {
    | ^^^^^--------^^^^^---
    | |    |            |
    | |    |            `f64` is not defined in the current crate
    | |    `f64` is not defined in the current crate
    | impl doesn't use only types from inside the current crate
    |
    = note: define and implement a trait or new type instead

error[E0053]: method `step` has an incompatible type for trait
  --> src\time_stepping\methods\backward_euler.rs:28:19
   |
17 | impl<P: TimeDependentProblem<Time = f64>> TimeStepper<P> for BackwardEuler {
   |      - expected this type parameter
...
28 |         problems: &[P],
   |                   ^^^^ expected type parameter `P`, found `[P]`
   |
note: type in trait
  --> src\time_stepping\ts.rs:42:18
   |
42 |         problem: &P,
   |                  ^^
   = note: expected signature `fn(&mut BackwardEuler, &P, _, _, &mut _) -> Result<_, _>`
              found signature `fn(&mut BackwardEuler, &[P], _, _, &mut _) -> Result<_, _>`
help: change the parameter type to match the trait
   |
28 |         problems: &P,
   |                   ~~

error[E0107]: trait takes 0 generic arguments but 1 generic argument was supplied
  --> src\equation\energy_equation.rs:13:9
   |
13 | impl<T> PhysicalEquation<T> for EnergyEquation {
   |         ^^^^^^^^^^^^^^^^--- help: remove the unnecessary generics
   |         |
   |         expected 0 generic arguments
   |
note: trait defined here, with 0 generic parameters
  --> src\equation\mod.rs:18:11
   |
18 | pub trait PhysicalEquation {
   |           ^^^^^^^^^^^^^^^^

error[E0277]: the trait bound `Fields: Vector` is not satisfied
  --> src\equation\manager.rs:66:51
   |
66 |     fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
   |                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `Vector` is not implemented for `Fields`
   |
   = help: the following other types implement trait `Vector`:
             Mat<f64>
             Vec<f64>

error[E0207]: the type parameter `T` is not constrained by the impl trait, self type, or predicates
  --> src\equation\energy_equation.rs:13:6
   |
13 | impl<T> PhysicalEquation<T> for EnergyEquation {
   |      ^ unconstrained type parameter

error[E0599]: no method named `update_state` found for mutable reference `&mut <P as TimeDependentProblem>::State` in the current scope
  --> src\time_stepping\methods\euler.rs:44:15
   |
44 |         state.update_state(&derivative, dt);
   |               ^^^^^^^^^^^^ method not found in `&mut <P as TimeDependentProblem>::State`

error[E0277]: the trait bound `Section<f64>: Clone` is not satisfied
 --> src\equation\fields.rs:7:5
  |
5 | #[derive(Clone)]
  |          ----- in this derive macro expansion
6 | pub struct Fields {
7 |     pub scalar_fields: FxHashMap<String, Section<f64>>,
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `Clone` is not implemented for `Section<f64>`, which is required by `HashMap<String, Section<f64>, FxBuildHasher>: Clone`
  |
  = note: required for `HashMap<String, Section<f64>, FxBuildHasher>` to implement `Clone`
  = note: this error originates in the derive macro `Clone` (in Nightly builds, run with -Z macro-backtrace for more info)
help: consider annotating `Section<f64>` with `#[derive(Clone)]`
 --> src\domain\section.rs:18:1
  |
18+ #[derive(Clone)]
19| pub struct Section<T> {
  |

error[E0277]: the trait bound `Section<[f64; 3]>: Clone` is not satisfied
 --> src\equation\fields.rs:8:5
  |
5 | #[derive(Clone)]
  |          ----- in this derive macro expansion
...
8 |     pub vector_fields: FxHashMap<String, Section<[f64; 3]>>,
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `Clone` is not implemented for `Section<[f64; 3]>`, which is required by `HashMap<String, Section<[f64; 3]>, FxBuildHasher>: Clone`
  |
  = note: required for `HashMap<String, Section<[f64; 3]>, FxBuildHasher>` to implement `Clone`
  = note: this error originates in the derive macro `Clone` (in Nightly builds, run with -Z macro-backtrace for more info)
help: consider annotating `Section<[f64; 3]>` with `#[derive(Clone)]`
 --> src\domain\section.rs:18:1
  |
18+ #[derive(Clone)]
19| pub struct Section<T> {
  |

error[E0277]: the trait bound `Section<[[f64; 3]; 3]>: Clone` is not satisfied
 --> src\equation\fields.rs:9:5
  |
5 | #[derive(Clone)]
  |          ----- in this derive macro expansion
...
9 |     pub tensor_fields: FxHashMap<String, Section<[[f64; 3]; 3]>>,
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `Clone` is not implemented for `Section<[[f64; 3]; 3]>`, which is required by `HashMap<String, Section<[[f64; 3]; 3]>, FxBuildHasher>: Clone`
  |
  = note: required for `HashMap<String, Section<[[f64; 3]; 3]>, FxBuildHasher>` to implement `Clone`
  = note: this error originates in the derive macro `Clone` (in Nightly builds, run with -Z macro-backtrace for more info)
help: consider annotating `Section<[[f64; 3]; 3]>` with `#[derive(Clone)]`
 --> src\domain\section.rs:18:1
  |
18+ #[derive(Clone)]
19| pub struct Section<T> {
  |

error[E0599]: the method `restrict` exists for reference `&Section<[[f64; 3]; 3]>`, but its trait bounds were not satisfied
  --> src\equation\fields.rs:50:39
   |
50 |         self.tensor_fields.get(name)?.restrict(entity)
   |                                       ^^^^^^^^ method cannot be called on `&Section<[[f64; 3]; 3]>` due to unsatisfied trait bounds
   |
note: the following trait bounds were not satisfied:
      `[[f64; 3]; 3]: AddAssign`
      `[[f64; 3]; 3]: Mul<f64>`
  --> src\domain\section.rs:25:16
   |
23 | impl<T> Section<T>
   |         ----------
24 | where
25 |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
   |                ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unsatisfied trait bound introduced here    
   |                |
   |                unsatisfied trait bound introduced here

error[E0599]: the method `set_data` exists for mutable reference `&mut Section<[[f64; 3]; 3]>`, but its trait bounds were not satisfied
  --> src\equation\fields.rs:55:19
   |
55 |             field.set_data(entity, value);
   |                   ^^^^^^^^ method cannot be called on `&mut Section<[[f64; 3]; 3]>` due to unsatisfied trait bounds
   |
note: the following trait bounds were not satisfied:
      `[[f64; 3]; 3]: AddAssign`
      `[[f64; 3]; 3]: Mul<f64>`
  --> src\domain\section.rs:25:16
   |
23 | impl<T> Section<T>
   |         ----------
24 | where
25 |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
   |                ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unsatisfied trait bound introduced here    
   |                |
   |                unsatisfied trait bound introduced here

error[E0277]: cannot add-assign `[[f64; 3]; 3]` to `[[f64; 3]; 3]`
  --> src\equation\fields.rs:57:29
   |
57 |             let mut field = Section::new();
   |                             ^^^^^^^^^^^^^^ no implementation for `[[f64; 3]; 3] += [[f64; 3]; 3]`
   |
   = help: the trait `AddAssign` is not implemented for `[[f64; 3]; 3]`
   = help: the trait `AddAssign` is implemented for `[f64; 3]`
note: required by a bound in `Section::<T>::new`
  --> src\domain\section.rs:25:16
   |
25 |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
   |                ^^^^^^^^^^^^^^^^^^^ required by this bound in `Section::<T>::new`
...
34 |     pub fn new() -> Self {
   |            --- required by a bound in this associated function

error[E0277]: cannot multiply `[[f64; 3]; 3]` by `f64`
  --> src\equation\fields.rs:57:29
   |
57 |             let mut field = Section::new();
   |                             ^^^^^^^^^^^^^^ no implementation for `[[f64; 3]; 3] * f64`
   |
   = help: the trait `Mul<f64>` is not implemented for `[[f64; 3]; 3]`
   = help: the trait `Mul<f64>` is implemented for `[f64; 3]`
   = help: for that trait implementation, expected `f64`, found `[f64; 3]`
note: required by a bound in `Section::<T>::new`
  --> src\domain\section.rs:25:38
   |
25 |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
   |                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ required by this bound in `Section::<T>::new`
...
34 |     pub fn new() -> Self {
   |            --- required by a bound in this associated function

error[E0277]: cannot add-assign `[[f64; 3]; 3]` to `[[f64; 3]; 3]`
  --> src\equation\fields.rs:58:19
   |
58 |             field.set_data(entity, value);
   |                   ^^^^^^^^ no implementation for `[[f64; 3]; 3] += [[f64; 3]; 3]`
   |
   = help: the trait `AddAssign` is not implemented for `[[f64; 3]; 3]`
   = help: the trait `AddAssign` is implemented for `[f64; 3]`
note: required by a bound in `Section::<T>::set_data`
  --> src\domain\section.rs:25:16
   |
25 |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
   |                ^^^^^^^^^^^^^^^^^^^ required by this bound in `Section::<T>::set_data`
...
54 |     pub fn set_data(&self, entity: MeshEntity, value: T) {
   |            -------- required by a bound in this associated function

error[E0277]: cannot multiply `[[f64; 3]; 3]` by `f64`
  --> src\equation\fields.rs:58:19
   |
58 |             field.set_data(entity, value);
   |                   ^^^^^^^^ no implementation for `[[f64; 3]; 3] * f64`
   |
   = help: the trait `Mul<f64>` is not implemented for `[[f64; 3]; 3]`
   = help: the trait `Mul<f64>` is implemented for `[f64; 3]`
   = help: for that trait implementation, expected `f64`, found `[f64; 3]`
note: required by a bound in `Section::<T>::set_data`
  --> src\domain\section.rs:25:38
   |
25 |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
   |                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ required by this bound in `Section::<T>::set_data`
...
54 |     pub fn set_data(&self, entity: MeshEntity, value: T) {
   |            -------- required by a bound in this associated function

error[E0277]: cannot add-assign `[f64; 2]` to `[f64; 2]`
   --> src\equation\fields.rs:108:32
    |
108 |             turbulence_fluxes: Section::new(),
    |                                ^^^^^^^^^^^^^^ no implementation for `[f64; 2] += [f64; 2]`
    |
    = help: the trait `AddAssign` is not implemented for `[f64; 2]`
    = help: the trait `AddAssign` is implemented for `[f64; 3]`
note: required by a bound in `Section::<T>::new`
   --> src\domain\section.rs:25:16
    |
25  |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
    |                ^^^^^^^^^^^^^^^^^^^ required by this bound in `Section::<T>::new`
...
34  |     pub fn new() -> Self {
    |            --- required by a bound in this associated function

error[E0277]: cannot multiply `[f64; 2]` by `f64`
   --> src\equation\fields.rs:108:32
    |
108 |             turbulence_fluxes: Section::new(),
    |                                ^^^^^^^^^^^^^^ no implementation for `[f64; 2] * f64`
    |
    = help: the trait `Mul<f64>` is not implemented for `[f64; 2]`
    = help: the trait `Mul<f64>` is implemented for `[f64; 3]`
note: required by a bound in `Section::<T>::new`
   --> src\domain\section.rs:25:38
    |
25  |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
    |                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ required by this bound in `Section::<T>::new`
...
34  |     pub fn new() -> Self {
    |            --- required by a bound in this associated function

error[E0599]: the method `restrict` exists for struct `Section<[f64; 2]>`, but its trait bounds were not satisfied  
   --> src\equation\fields.rs:133:59
    |
133 |         if let Some(mut current) = self.turbulence_fluxes.restrict(&entity) {
    |                                                           ^^^^^^^^ method cannot be called on `Section<[f64; 2]>` due to unsatisfied trait bounds
    |
   ::: src\domain\section.rs:18:1
    |
18  | pub struct Section<T> {
    | --------------------- method `restrict` not found for this struct
    |
note: the following trait bounds were not satisfied:
      `[f64; 2]: AddAssign`
      `[f64; 2]: Mul<f64>`
   --> src\domain\section.rs:25:16
    |
23  | impl<T> Section<T>
    |         ----------
24  | where
25  |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
    |                ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unsatisfied trait bound introduced here   
    |                |
    |                unsatisfied trait bound introduced here

error[E0599]: the method `set_data` exists for struct `Section<[f64; 2]>`, but its trait bounds were not satisfied 
   --> src\equation\fields.rs:137:36
    |
137 |             self.turbulence_fluxes.set_data(entity, current);
    |                                    ^^^^^^^^ method cannot be called on `Section<[f64; 2]>` due to unsatisfied trait bounds
    |
   ::: src\domain\section.rs:18:1
    |
18  | pub struct Section<T> {
    | --------------------- method `set_data` not found for this struct
    |
note: the following trait bounds were not satisfied:
      `[f64; 2]: AddAssign`
      `[f64; 2]: Mul<f64>`
   --> src\domain\section.rs:25:16
    |
23  | impl<T> Section<T>
    |         ----------
24  | where
25  |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
    |                ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unsatisfied trait bound introduced here   
    |                |
    |                unsatisfied trait bound introduced here

error[E0599]: the method `set_data` exists for struct `Section<[f64; 2]>`, but its trait bounds were not satisfied  
   --> src\equation\fields.rs:139:36
    |
139 |             self.turbulence_fluxes.set_data(entity, value);
    |                                    ^^^^^^^^ method cannot be called on `Section<[f64; 2]>` due to unsatisfied trait bounds
    |
   ::: src\domain\section.rs:18:1
    |
18  | pub struct Section<T> {
    | --------------------- method `set_data` not found for this struct
    |
note: the following trait bounds were not satisfied:
      `[f64; 2]: AddAssign`
      `[f64; 2]: Mul<f64>`
   --> src\domain\section.rs:25:16
    |
23  | impl<T> Section<T>
    |         ----------
24  | where
25  |     T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
    |                ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unsatisfied trait bound introduced here   
    |                |
    |                unsatisfied trait bound introduced here

error[E0277]: the trait bound `Fields: Vector` is not satisfied
  --> src\equation\manager.rs:66:83
   |
66 |       fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
   |  ___________________________________________________________________________________^
67 | |         // Convert time to scalar if needed (implementation depends on Vector trait)
68 | |         unimplemented!()
69 | |     }
   | |_____^ the trait `Vector` is not implemented for `Fields`
   |
   = help: the following other types implement trait `Vector`:
             Mat<f64>
             Vec<f64>

error[E0609]: no field `temperature_field` on type `&Fields`
  --> src\equation\energy_equation.rs:24:21
   |
24 |             &fields.temperature_field,
   |                     ^^^^^^^^^^^^^^^^^ unknown field
   |
   = note: available fields are: `scalar_fields`, `vector_fields`, `tensor_fields`

error[E0609]: no field `temperature_gradient` on type `&Fields`
  --> src\equation\energy_equation.rs:25:21
   |
25 |             &fields.temperature_gradient,
   |                     ^^^^^^^^^^^^^^^^^^^^ unknown field
   |
   = note: available fields are: `scalar_fields`, `vector_fields`, `tensor_fields`

error[E0609]: no field `velocity_field` on type `&Fields`
  --> src\equation\energy_equation.rs:26:21
   |
26 |             &fields.velocity_field,
   |                     ^^^^^^^^^^^^^^ unknown field
   |
   = note: available fields are: `scalar_fields`, `vector_fields`, `tensor_fields`

error[E0502]: cannot borrow `*self.time_stepper` as mutable because it is also borrowed as immutable
  --> src\equation\manager.rs:41:9
   |
41 | /         self.time_stepper
42 | |             .step(self, self.time_stepper.get_time_step(), current_time, fields)
   | |______________----_----_________________________________________________________^ mutable borrow occurs here  
   |                |    |
   |                |    immutable borrow occurs here
   |                immutable borrow later used by call

Some errors have detailed explanations: E0053, E0107, E0117, E0207, E0277, E0407, E0502, E0599, E0609.
For more information about an error, try `rustc --explain E0053`.
```

---

`src/domain/section.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use std::ops::{AddAssign, Mul};


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

impl<T> Section<T>
where
    T: Clone + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
{
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

    fn clone(&self) -> Self {
        Section {
            data: self.data.clone(),
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

    /// Updates the section data by adding the derivative multiplied by dt
    pub fn update_with_derivative(&mut self, derivative: &Section<T>, dt: f64) {
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
}


impl AddAssign for [f64; 3] {
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            self[i] += other[i];
        }
    }
}

impl Mul<f64> for [f64; 3] {
    type Output = [f64; 3];

    fn mul(self, rhs: f64) -> Self::Output {
        [self[0] * rhs, self[1] * rhs, self[2] * rhs]
    }
}

impl AddAssign for f64 {
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }
}

impl Mul<f64> for f64 {
    type Output = f64;

    fn mul(self, rhs: f64) -> Self::Output {
        self * rhs
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

`src/time_stepping/methods/euler.rs`

```rust
use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};

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

impl<P: TimeDependentProblem<Time = f64>> TimeStepper<P> for BackwardEuler {
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    fn step(
        &mut self,
        problems: &[P],
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        self.time_step = dt;

        // Iterate through each problem and apply Backward Euler
        for problem in problems {
            let mut matrix = problem
                .get_matrix()
                .ok_or(TimeSteppingError::SolverError("Matrix is required for Backward Euler.".into()))?;
            let mut rhs = state.clone();

            problem.compute_rhs(current_time, state, &mut rhs)?;
            problem.solve_linear_system(matrix.as_mut(), state, &rhs)?;
        }

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
    
    fn get_time_step(&self) -> <P as TimeDependentProblem>::Time {
        todo!()
    }
}

```

---

`src/equation/fields.rs`

```rust
use rustc_hash::FxHashMap;
use crate::{domain::Section, MeshEntity};


#[derive(Clone)]
pub struct Fields {
    pub scalar_fields: FxHashMap<String, Section<f64>>,
    pub vector_fields: FxHashMap<String, Section<[f64; 3]>>,
    pub tensor_fields: FxHashMap<String, Section<[[f64; 3]; 3]>>,
}

impl Fields {
    pub fn new() -> Self {
        Self {
            scalar_fields: FxHashMap::default(),
            vector_fields: FxHashMap::default(),
            tensor_fields: FxHashMap::default(),
        }
    }

    pub fn get_scalar_field_value(&self, name: &str, entity: &MeshEntity) -> Option<f64> {
        self.scalar_fields.get(name)?.restrict(entity)
    }

    pub fn set_scalar_field_value(&mut self, name: &str, entity: MeshEntity, value: f64) {
        if let Some(field) = self.scalar_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.scalar_fields.insert(name.to_string(), field);
        }
    }

    pub fn get_vector_field_value(&self, name: &str, entity: &MeshEntity) -> Option<[f64; 3]> {
        self.vector_fields.get(name)?.restrict(entity)
    }

    pub fn set_vector_field_value(&mut self, name: &str, entity: MeshEntity, value: [f64; 3]) {
        if let Some(field) = self.vector_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.vector_fields.insert(name.to_string(), field);
        }
    }

    pub fn get_tensor_field_value(&self, name: &str, entity: &MeshEntity) -> Option<[[f64; 3]; 3]> {
        self.tensor_fields.get(name)?.restrict(entity)
    }

    pub fn set_tensor_field_value(&mut self, name: &str, entity: MeshEntity, value: [[f64; 3]; 3]) {
        if let Some(field) = self.tensor_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let mut field = Section::new();
            field.set_data(entity, value);
            self.tensor_fields.insert(name.to_string(), field);
        }
    }

    pub fn add_turbulence_fields(&mut self) {
        self.scalar_fields.insert("turbulent_viscosity".to_string(), Section::new());
        self.scalar_fields.insert("k_field".to_string(), Section::new());
        self.scalar_fields.insert("omega_field".to_string(), Section::new());
    }

    /// Updates the state by adding the derivative multiplied by dt
    pub fn update_state(&mut self, derivative: &Fields, dt: f64) {
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
    }

    /// Updates the derivative fields based on the computed fluxes
    pub fn update_from_fluxes(&mut self, fluxes: &Fluxes) {
        let _ = fluxes;
        // Implement logic to update derivative fields from fluxes
        // For example, update momentum fields based on momentum_fluxes
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
use crate::{
    boundary::bc_handler::BoundaryConditionHandler,
    domain::mesh::Mesh,
    time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError},
    Matrix, Vector,
};
use super::{Fields, Fluxes, PhysicalEquation};

pub struct EquationManager {
    equations: Vec<Box<dyn PhysicalEquation>>,
    time_stepper: Box<dyn TimeStepper<Self>>,
}

impl EquationManager {
    pub fn new(time_stepper: Box<dyn TimeStepper<Self>>) -> Self {
        Self {
            equations: Vec::new(),
            time_stepper,
        }
    }

    pub fn add_equation<E: PhysicalEquation + 'static>(&mut self, equation: E) {
        self.equations.push(Box::new(equation));
    }

    pub fn assemble_all(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        let current_time = self.time_stepper.current_time();
        for equation in &self.equations {
            equation.assemble(domain, fields, fluxes, boundary_handler, current_time);
        }
    }

    pub fn step(&mut self, fields: &mut Fields) {
        let current_time = self.time_stepper.current_time();
        self.time_stepper
            .step(self, self.time_stepper.get_time_step(), current_time, fields)
            .expect("Time-stepping failed");
    }
}

impl TimeDependentProblem for EquationManager {
    type State = Fields;
    type Time = f64;

    fn compute_rhs(
        &self,
        _time: Self::Time,
        _state: &Self::State,
        _derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError> {
        // Implement RHS computation based on the assembled fluxes
        unimplemented!()
    }

    fn initial_state(&self) -> Self::State {
        // Initialize fields with appropriate initial conditions
        Fields::new()
    }

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar {
        // Convert time to scalar if needed (implementation depends on Vector trait)
        unimplemented!()
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

`src/equation/momentum_equation.rs`

```rust
use crate::{
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
    domain::mesh::Mesh,
    geometry::Geometry,
};
use super::{
    fields::{Fields, Fluxes},
    PhysicalEquation,
};

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
                let mut velocity_a = [0.0; 3];
                let mut pressure_a = 0.0;
                let mut velocity_b = [0.0; 3];
                let mut pressure_b = 0.0;

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
                    [
                        0.5 * (velocity_a[0] + velocity_b[0]),
                        0.5 * (velocity_a[1] + velocity_b[1]),
                        0.5 * (velocity_a[2] + velocity_b[2]),
                    ]
                } else {
                    velocity_a
                };

                let velocity_dot_normal = avg_velocity.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();

                let convective_flux = self.params.density * velocity_dot_normal * area;

                // Compute pressure flux
                let pressure_flux = if has_cell_b {
                    0.5 * (pressure_a + pressure_b) * area
                } else {
                    pressure_a * area
                };

                // Compute diffusive flux (simplified for demonstration)
                // In practice, this would involve gradients of velocity
                let diffusive_flux = self.params.viscosity * area;

                // Total flux vector (assuming 3D for demonstration)
                let total_flux = [
                    convective_flux - pressure_flux + diffusive_flux,
                    0.0,
                    0.0,
                ];

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




#[cfg(test)]
mod tests {
    use faer::Mat;

    use super::*;
    use crate::boundary::bc_handler::{BoundaryConditionHandler, BoundaryCondition};
    use crate::domain::{mesh::Mesh, MeshEntity};
    use crate::equation::fields::{Fields, Fluxes};
    use crate::equation::PhysicalEquation;

    fn create_mock_mesh() -> Mesh {
        let mut mesh = Mesh::new();

        // Define mock entities
        let face = MeshEntity::Face(1);
        let cell = MeshEntity::Cell(1);
        let vertex = MeshEntity::Vertex(1);

        // Add entities to the mesh
        mesh.add_entity(face.clone());
        mesh.add_entity(cell.clone());
        mesh.add_entity(vertex.clone());

        // Add relationships
        mesh.add_relationship(cell.clone(), face.clone());
        mesh.add_relationship(face.clone(), vertex.clone());

        mesh
    }

    fn create_mock_fields() -> Fields<f64> {
        let mut fields = Fields::new();

        let entity = MeshEntity::Cell(1);
        fields.set_velocity(entity.clone(), [1.0, 0.0, 0.0]);
        fields.set_pressure(entity.clone(), 1.0);

        fields
    }

    fn create_mock_fluxes() -> Fluxes {
        Fluxes::new()
    }

    fn create_mock_boundary_handler() -> BoundaryConditionHandler {
        let mut handler = BoundaryConditionHandler::new();

        let face = MeshEntity::Face(1);
        handler.set_bc(face, BoundaryCondition::Dirichlet(0.0));

        handler
    }

    #[test]
    fn test_momentum_equation_struct() {
        let density = 1.225;
        let viscosity = 1.81e-5;

        let momentum_eq = MomentumEquation { density, viscosity };

        assert_eq!(momentum_eq.density, density);
        assert_eq!(momentum_eq.viscosity, viscosity);
    }

    #[test]
    fn test_assemble_method() {
        let density = 1.225;
        let viscosity = 1.81e-5;

        let momentum_eq = MomentumEquation { density, viscosity };
        let domain = create_mock_mesh();
        let fields = create_mock_fields();
        let mut fluxes = create_mock_fluxes();
        let boundary_handler = create_mock_boundary_handler();
        let current_time = 0.0;

        momentum_eq.assemble(&domain, &fields, &mut fluxes, &boundary_handler, current_time);

        // Check that the fluxes were computed
        let face = MeshEntity::Face(1);
        assert!(fluxes.momentum_fluxes.restrict(&face).is_some());
    }

    #[test]
    fn test_calculate_momentum_fluxes() {
        let density = 1.225;
        let viscosity = 1.81e-5;

        let momentum_eq = MomentumEquation { density, viscosity };
        let domain = create_mock_mesh();
        let fields = create_mock_fields();
        let mut fluxes = create_mock_fluxes();
        let boundary_handler = create_mock_boundary_handler();
        let current_time = 0.0;

        momentum_eq.calculate_momentum_fluxes(
            &domain,
            &fields,
            &mut fluxes,
            &boundary_handler,
            current_time,
        );

        let face = MeshEntity::Face(1);
        assert!(fluxes.momentum_fluxes.restrict(&face).is_some());
    }

    #[test]
    fn test_apply_boundary_conditions() {
        let domain = create_mock_mesh();
        let fluxes = Fluxes::new();
        let mut matrix = Mat::zeros(2, 2); // Example matrix, adjust dimensions as needed
        let mut rhs = Mat::zeros(2, 1); // Example RHS vector
        let boundary_handler = create_mock_boundary_handler();
        let boundary_entities = boundary_handler.get_boundary_faces();
        let entity_to_index = domain.get_entity_to_index();
        let current_time = 0.0;

        // Use apply_bc with correct arguments
        boundary_handler.apply_bc(
            &mut matrix.as_mut(),
            &mut rhs.as_mut(),
            &boundary_entities,
            &entity_to_index,
            current_time,
        );

        // Add assertions based on expected behavior
        assert!(matrix.as_ref().nrows() > 0, "Matrix should have rows after applying BCs.");
        assert!(rhs.as_ref().ncols() > 0, "RHS vector should have entries after applying BCs.");
    }

    #[test]
    fn test_geometry_integration() {
        let domain = create_mock_mesh();

        for face in domain.get_faces() {
            let normal = domain.get_face_normal(&face, None).unwrap_or([0.0, 0.0, 0.0]);
            let area = domain.get_face_area(&face).unwrap_or(0.0);

            assert!(normal.iter().any(|&n| n != 0.0)); // Check that normal is non-zero
            assert!(area > 0.0); // Check that area is positive
        }
    }
}

```

---

`src/equation/energy_equation.rs`

```rust
use crate::equation::reconstruction::reconstruct::reconstruct_face_value;
use crate::equation::PhysicalEquation;
use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::geometry::{Geometry, FaceShape};

use super::fields::{Fields, Fluxes};

pub struct EnergyEquation {
    pub thermal_conductivity: f64, // Coefficient for thermal conduction
}

impl<T> PhysicalEquation<T> for EnergyEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields<T>,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        self.calculate_energy_fluxes(
            domain,
            &fields.temperature_field,
            &fields.temperature_gradient,
            &fields.velocity_field,
            &mut fluxes.energy_fluxes,
            boundary_handler,
        );
    }
}

impl EnergyEquation {
    pub fn new(thermal_conductivity: f64) -> Self {
        EnergyEquation { thermal_conductivity }
    }

    pub fn calculate_energy_fluxes(
        &self,
        domain: &Mesh,
        temperature_field: &Section<f64>,
        temperature_gradient: &Section<[f64; 3]>,
        velocity_field: &Section<[f64; 3]>,
        energy_fluxes: &mut Section<f64>,
        boundary_handler: &BoundaryConditionHandler,
    ) {
        let mut geometry = Geometry::new();
    
        for face in domain.get_faces() {
            let face_vertices = domain.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
            };
            let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);
    
            let cells = domain.get_cells_sharing_face(&face);
            let cell_a = cells
                .iter()
                .next()
                .map(|entry| entry.key().clone())
                .expect("Face should have at least one associated cell.");
    
            let temp_a = temperature_field
                .restrict(&cell_a)
                .expect("Temperature not found for cell");
            let grad_temp_a = temperature_gradient
                .restrict(&cell_a)
                .expect("Temperature gradient not found for cell");
    
            let mut face_temperature = reconstruct_face_value(
                temp_a,
                grad_temp_a,
                geometry.compute_cell_centroid(domain, &cell_a),
                face_center,
            );
    
            let velocity = velocity_field
                .restrict(&face)
                .expect("Velocity not found at face");
            let face_normal = geometry
                .compute_face_normal(domain, &face, &cell_a)
                .expect("Normal not found for face");
    
            let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
    
            let total_flux;
    
            if cells.len() == 1 {
                // Boundary face
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    println!("Applying boundary condition on face {:?}", face);
                    match bc {
                        BoundaryCondition::Dirichlet(value) => {
                            println!(
                                "Dirichlet condition, setting face temperature to {}",
                                value
                            );
                            face_temperature = value; // Enforce Dirichlet on face temperature
            
                            // Recompute conductive flux based on temperature difference
                            let cell_centroid = geometry.compute_cell_centroid(domain, &cell_a);
                            let distance =
                                Geometry::compute_distance(&cell_centroid, &face_center);
            
                            let temp_gradient_normal = (face_temperature - temp_a) / distance;
                            let face_normal_length = face_normal
                                .iter()
                                .map(|n| n * n)
                                .sum::<f64>()
                                .sqrt();
            
                            let conductive_flux = -self.thermal_conductivity
                                * temp_gradient_normal
                                * face_normal_length;
            
                            // Compute convective flux
                            let vel_dot_n = velocity
                                .iter()
                                .zip(&face_normal)
                                .map(|(v, n)| v * n)
                                .sum::<f64>();
                            let rho = 1.0;
                            let convective_flux = rho * face_temperature * vel_dot_n;
            
                            total_flux = (conductive_flux + convective_flux) * face_area;
                        }
                        BoundaryCondition::Neumann(flux) => {
                            println!("Neumann condition, setting total flux to {}", flux);
                            total_flux = flux * face_area; // Enforce Neumann directly over the face area
                        }
                        BoundaryCondition::Robin { alpha, beta } => {
                            println!(
                                "Robin condition, alpha: {}, beta: {}",
                                alpha, beta
                            );
                            // For Robin boundary condition, the flux is defined by:
                            // q = alpha * (face_temperature - beta)
                            // where:
                            // - alpha is the heat transfer coefficient
                            // - beta is the ambient temperature
                            // We compute the conductive flux accordingly.
            
                            // Compute conductive flux based on Robin condition
                            let conductive_flux = -alpha * (face_temperature - beta);
            
                            // Compute convective flux as before
                            let vel_dot_n = velocity
                                .iter()
                                .zip(&face_normal)
                                .map(|(v, n)| v * n)
                                .sum::<f64>();
                            let rho = 1.0;
                            let convective_flux = rho * face_temperature * vel_dot_n;
            
                            total_flux = (conductive_flux + convective_flux) * face_area;
                        }
                        _ => {
                            // Default behavior if no specific boundary condition is matched
                            // Proceed with normal calculation
                            total_flux = self.compute_flux(
                                temp_a,
                                face_temperature,
                                &grad_temp_a,
                                &face_normal,
                                &velocity,
                                face_area,
                            );
                        }
                    }
                } else {
                    // No boundary condition specified; proceed with normal calculation
                    total_flux = self.compute_flux(
                        temp_a,
                        face_temperature,
                        &grad_temp_a,
                        &face_normal,
                        &velocity,
                        face_area,
                    );
                }
            } else {
                // Internal face
                total_flux = self.compute_flux(
                    temp_a,
                    face_temperature,
                    &grad_temp_a,
                    &face_normal,
                    &velocity,
                    face_area,
                );
            }
            
    
            println!("Storing total flux: {} for face {:?}", total_flux, face);
            energy_fluxes.set_data(face, total_flux);
        }
    }
    
    // Helper function to compute flux
    fn compute_flux(
        &self,
        temp_a: f64,
        face_temperature: f64,
        grad_temp_a: &[f64; 3],
        face_normal: &[f64; 3],
        velocity: &[f64; 3],
        face_area: f64,
    ) -> f64 {
        let _ = temp_a;
        let conductive_flux = -self.thermal_conductivity
            * (grad_temp_a[0] * face_normal[0]
                + grad_temp_a[1] * face_normal[1]
                + grad_temp_a[2] * face_normal[2]);
    
        let rho = 1.0;
        let convective_flux = rho
            * face_temperature
            * (velocity[0] * face_normal[0]
                + velocity[1] * face_normal[1]
                + velocity[2] * face_normal[2]);
    
        (conductive_flux + convective_flux) * face_area
    }
            
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{mesh::Mesh, Section, mesh_entity::MeshEntity};
    use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};

    /// Creates a simple mesh with cells, faces, and vertices, to be used across tests.
    fn create_simple_mesh_with_faces() -> Mesh {
        let mut mesh = Mesh::new();
    
        // Define vertices
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);
    
        // Add vertices to mesh and set coordinates
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);
    
        // Create a face and associate it with vertices
        let face = MeshEntity::Face(1);
        mesh.add_entity(face);
        mesh.add_relationship(face.clone(), vertex1);
        mesh.add_relationship(face.clone(), vertex2);
        mesh.add_relationship(face.clone(), vertex3);
        mesh.add_relationship(face.clone(), vertex4);
    
        // Create cells and connect each cell to the face and vertices
        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        mesh.add_entity(cell1);
        mesh.add_entity(cell2);
        mesh.add_relationship(cell1, face.clone());
        mesh.add_relationship(cell2, face.clone());
    
        // Ensure both cells are connected to all vertices (to fully define geometry)
        for &vertex in &[vertex1, vertex2, vertex3, vertex4] {
            mesh.add_relationship(cell1, vertex);
            mesh.add_relationship(cell2, vertex);
        }
    
        mesh
    }
    
    fn create_simple_mesh_with_boundary_face() -> Mesh {
        let mut mesh = Mesh::new();
    
        // Define vertices with distinct coordinates
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);
    
        // Add vertices to mesh and set coordinates
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);
    
        // Create a face and associate it with vertices (e.g., vertices 1, 2, and 3)
        let face = MeshEntity::Face(1);
        mesh.add_entity(face);
        mesh.add_relationship(face.clone(), vertex1);
        mesh.add_relationship(face.clone(), vertex2);
        mesh.add_relationship(face.clone(), vertex3);
    
        // Create a cell and connect it to the face and vertices
        let cell1 = MeshEntity::Cell(1);
        mesh.add_entity(cell1);
        mesh.add_relationship(cell1, face.clone());
        // Connect cell to all vertices
        for &vertex in &[vertex1, vertex2, vertex3, vertex4] {
            mesh.add_relationship(cell1, vertex);
        }
    
        mesh
    }
    

    #[test]
    fn test_energy_equation_initialization() {
        let thermal_conductivity = 0.5;
        let energy_eq = EnergyEquation::new(thermal_conductivity);
        assert_eq!(energy_eq.thermal_conductivity, 0.5);
    }

    #[test]
    fn test_flux_calculation_no_boundary_conditions() {
        let mesh = create_simple_mesh_with_faces();
        let boundary_handler = BoundaryConditionHandler::new();

        let temperature_field = Section::new();
        let temperature_gradient = Section::new();
        let velocity_field = Section::new();
        let mut energy_fluxes = Section::new();

        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let face = MeshEntity::Face(1);

        // Assign temperature and gradient values for both cells associated with the face
        temperature_field.set_data(cell1, 300.0);
        temperature_field.set_data(cell2, 310.0); // Temperature for the second cell
        temperature_gradient.set_data(cell1, [10.0, 0.0, 0.0]);
        temperature_gradient.set_data(cell2, [10.0, 0.0, 0.0]);
        velocity_field.set_data(face, [2.0, 0.0, 0.0]);

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.calculate_energy_fluxes(
            &mesh,
            &temperature_field,
            &temperature_gradient,
            &velocity_field,
            &mut energy_fluxes,
            &boundary_handler,
        );

        assert!(energy_fluxes.restrict(&face).is_some(), "Flux should be calculated for the face.");
    }


    #[test]
    fn test_flux_calculation_with_dirichlet_boundary_condition() {
        let mesh = create_simple_mesh_with_boundary_face();
        let boundary_handler = BoundaryConditionHandler::new();

        let temperature_field = Section::new();
        let temperature_gradient = Section::new();
        let velocity_field = Section::new();
        let mut energy_fluxes = Section::new();

        let cell1 = MeshEntity::Cell(1);
        let face = MeshEntity::Face(1);

        // Set temperature and gradient data for the cell associated with the face
        temperature_field.set_data(cell1, 300.0);
        temperature_gradient.set_data(cell1, [10.0, 0.0, 0.0]);
        velocity_field.set_data(face, [2.0, 0.0, 0.0]);

        // Apply a Dirichlet boundary condition on the face with a fixed temperature value
        boundary_handler.set_bc(face, BoundaryCondition::Dirichlet(100.0));

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.calculate_energy_fluxes(
            &mesh,
            &temperature_field,
            &temperature_gradient,
            &velocity_field,
            &mut energy_fluxes,
            &boundary_handler,
        );

        // Retrieve the calculated flux
        let calculated_flux = energy_fluxes.restrict(&face).expect("Flux not calculated.");

        // Manually compute the expected flux using the Dirichlet temperature
        let mut geometry = Geometry::new();
        let face_vertices = mesh.get_face_vertices(&face);
        let face_shape = FaceShape::Triangle;
        let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);
        let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
        let face_normal = geometry.compute_face_normal(&mesh, &face, &cell1).unwrap();

        // Use the boundary temperature for face_temperature
        let face_temperature = 100.0;
        let temp_a = 300.0;

        // Compute the distance between cell centroid and face centroid
        let cell_centroid = geometry.compute_cell_centroid(&mesh, &cell1);
        let distance = Geometry::compute_distance(&cell_centroid, &face_center);

        // Ensure distance is not zero
        assert!(
            distance > 0.0,
            "Distance between cell centroid and face centroid should be greater than zero."
        );

        // Compute the temperature gradient normal to the face
        let temp_gradient_normal = (face_temperature - temp_a) / distance;

        // Compute the magnitude of the face normal vector
        let face_normal_length = face_normal.iter().map(|n| n * n).sum::<f64>().sqrt();

        // Compute conductive flux based on the temperature difference
        let conductive_flux = -energy_eq.thermal_conductivity * temp_gradient_normal * face_normal_length;

        // Compute convective flux
        let velocity = velocity_field.restrict(&face).unwrap();
        let vel_dot_n = velocity.iter().zip(&face_normal).map(|(v, n)| v * n).sum::<f64>();
        let rho = 1.0;
        let convective_flux = rho * face_temperature * vel_dot_n;

        // Total expected flux
        let expected_flux = (conductive_flux + convective_flux) * face_area;

        // Check that calculated_flux matches expected_flux within tolerance
        assert!(
            (calculated_flux - expected_flux).abs() < 1e-6,
            "Calculated flux {} does not match expected flux {}.",
            calculated_flux,
            expected_flux
        );
    }




    #[test]
    fn test_flux_calculation_with_neumann_boundary_condition() {
        let mesh = create_simple_mesh_with_faces();
        let boundary_handler = BoundaryConditionHandler::new();

        let temperature_field = Section::new();
        let temperature_gradient = Section::new();
        let velocity_field = Section::new();
        let mut energy_fluxes = Section::new();

        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let face = MeshEntity::Face(1);

        // Set temperature and gradient data for both cells associated with the face
        temperature_field.set_data(cell1, 300.0);
        temperature_field.set_data(cell2, 310.0);
        temperature_gradient.set_data(cell1, [10.0, 0.0, 0.0]);
        temperature_gradient.set_data(cell2, [10.0, 0.0, 0.0]);
        velocity_field.set_data(face, [2.0, 0.0, 0.0]);

        // Apply a Neumann boundary condition with a flux increment of 50.0
        boundary_handler.set_bc(face, BoundaryCondition::Neumann(50.0));

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.calculate_energy_fluxes(
            &mesh,
            &temperature_field,
            &temperature_gradient,
            &velocity_field,
            &mut energy_fluxes,
            &boundary_handler,
        );

        // Verify that the Neumann boundary condition adjusted the flux
        let calculated_flux = energy_fluxes.restrict(&face).expect("Flux not calculated.");
        assert!(calculated_flux > 0.0, "Flux should be adjusted by Neumann boundary condition.");
    }

    #[test]
    fn test_flux_calculation_with_robin_boundary_condition() {
        let mesh = create_simple_mesh_with_faces();
        let boundary_handler = BoundaryConditionHandler::new();

        let temperature_field = Section::new();
        let temperature_gradient = Section::new();
        let velocity_field = Section::new();
        let mut energy_fluxes = Section::new();

        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let face = MeshEntity::Face(1);

        // Set temperature and gradient data for both cells associated with the face
        temperature_field.set_data(cell1, 300.0);
        temperature_field.set_data(cell2, 310.0);
        temperature_gradient.set_data(cell1, [10.0, 0.0, 0.0]);
        temperature_gradient.set_data(cell2, [10.0, 0.0, 0.0]);
        velocity_field.set_data(face, [2.0, 0.0, 0.0]);

        // Apply a Robin boundary condition with parameters alpha and beta
        boundary_handler.set_bc(face, BoundaryCondition::Robin { alpha: 0.3, beta: 0.7 });

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.calculate_energy_fluxes(
            &mesh,
            &temperature_field,
            &temperature_gradient,
            &velocity_field,
            &mut energy_fluxes,
            &boundary_handler,
        );

        // Verify that the Robin boundary condition affected the flux
        let calculated_flux = energy_fluxes.restrict(&face).expect("Flux not calculated.");
        assert!(calculated_flux != 0.0, "Flux should be affected by Robin boundary conditions.");
    }

    #[test]
    fn test_assemble_function_integration() {
        let mesh = create_simple_mesh_with_faces();
        let boundary_handler = BoundaryConditionHandler::new();

        let fields = Fields {
            temperature_field: Section::new(),
            temperature_gradient: Section::new(),
            velocity_field: Section::new(),
            field: Section::new(),
            gradient: Section::new(),
            k_field: Section::new(),
            epsilon_field: Section::new(),
            pressure_field: todo!(),
            velocity_gradient: todo!(),
        };
        
        let mut fluxes = Fluxes {
            energy_fluxes: Section::new(),
            momentum_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        };

        let cell1 = MeshEntity::Cell(1);
        let cell2 = MeshEntity::Cell(2);
        let face = MeshEntity::Face(1);

        // Set values for temperature, gradient, and velocity for cells and face
        fields.temperature_field.set_data(cell1, 300.0);
        fields.temperature_field.set_data(cell2, 310.0);
        fields.temperature_gradient.set_data(cell1, [10.0, 0.0, 0.0]);
        fields.temperature_gradient.set_data(cell2, [10.0, 0.0, 0.0]);
        fields.velocity_field.set_data(face, [2.0, 0.0, 0.0]);

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, current_time);

        // Verify that energy fluxes were computed and stored for the face
        assert!(fluxes.energy_fluxes.restrict(&face).is_some(), "Energy fluxes should be computed and stored.");
    }

}
```