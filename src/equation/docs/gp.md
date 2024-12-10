Determine the best approach to address the unit test failures reported below, using the source code provided to ensure that a complete answer is generated in response.

```bash
successes:
    equation::energy_equation::tests::test_energy_equation_scaling_with_thermal_conductivity

failures:

---- equation::energy_equation::tests::test_internal_face_flux_computation stdout ----
thread 'equation::energy_equation::tests::test_internal_face_flux_computation' panicked at src\equation\energy_equation.rs:286:17:
Energy flux for internal face Face(7) was not computed.
stack backtrace:
   0: std::panicking::begin_panic_handler
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\std\src\panicking.rs:652     
   1: core::panicking::panic_fmt
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\core\src\panicking.rs:72     
   2: hydra::equation::energy_equation::tests::test_internal_face_flux_computation
             at .\src\equation\energy_equation.rs:286
   3: hydra::equation::energy_equation::tests::test_internal_face_flux_computation::closure$0        
             at .\src\equation\energy_equation.rs:271
   4: core::ops::function::FnOnce::call_once<hydra::equation::energy_equation::tests::test_internal_face_flux_computation::closure_env$0,tuple$<> >
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23\library\core\src\ops\function.rs:250 
   5: core::ops::function::FnOnce::call_once
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\core\src\ops\function.rs:250 
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.

---- equation::energy_equation::tests::test_energy_equation_with_dirichlet_bc stdout ----
thread 'equation::energy_equation::tests::test_energy_equation_with_dirichlet_bc' panicked at src\equation\energy_equation.rs:230:9:
Energy flux for boundary face was not computed.
stack backtrace:
   0: std::panicking::begin_panic_handler
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\std\src\panicking.rs:652     
   1: core::panicking::panic_fmt
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\core\src\panicking.rs:72     
   2: hydra::equation::energy_equation::tests::test_energy_equation_with_dirichlet_bc
             at .\src\equation\energy_equation.rs:230
   3: hydra::equation::energy_equation::tests::test_energy_equation_with_dirichlet_bc::closure$0     
             at .\src\equation\energy_equation.rs:213
   4: core::ops::function::FnOnce::call_once<hydra::equation::energy_equation::tests::test_energy_equation_with_dirichlet_bc::closure_env$0,tuple$<> >
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23\library\core\src\ops\function.rs:250 
   5: core::ops::function::FnOnce::call_once
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\core\src\ops\function.rs:250 
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.

---- equation::energy_equation::tests::test_energy_equation_with_neumann_bc stdout ----
thread 'equation::energy_equation::tests::test_energy_equation_with_neumann_bc' panicked at src\equation\energy_equation.rs:258:9:
Energy flux for boundary face was not computed.
stack backtrace:
   0: std::panicking::begin_panic_handler
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\std\src\panicking.rs:652     
   1: core::panicking::panic_fmt
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\core\src\panicking.rs:72     
   2: hydra::equation::energy_equation::tests::test_energy_equation_with_neumann_bc
             at .\src\equation\energy_equation.rs:258
   3: hydra::equation::energy_equation::tests::test_energy_equation_with_neumann_bc::closure$0       
             at .\src\equation\energy_equation.rs:241
   4: core::ops::function::FnOnce::call_once<hydra::equation::energy_equation::tests::test_energy_equation_with_neumann_bc::closure_env$0,tuple$<> >
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23\library\core\src\ops\function.rs:250 
   5: core::ops::function::FnOnce::call_once
             at /rustc/3f5fd8dd41153bc5fdca9427e9e05be2c767ba23/library\core\src\ops\function.rs:250 
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.


failures:
    equation::energy_equation::tests::test_energy_equation_with_dirichlet_bc
    equation::energy_equation::tests::test_energy_equation_with_neumann_bc
    equation::energy_equation::tests::test_internal_face_flux_computation

test result: FAILED. 1 passed; 3 failed; 0 ignored; 0 measured; 324 filtered out; finished in 0.05s
```

---

`src/equation/energy_equation.rs`

```rust
use crate::equation::PhysicalEquation;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::geometry::{Geometry, FaceShape};
use crate::domain::section::{Scalar, Vector3};
use crate::Mesh;

use super::fields::{Fields, Fluxes};
use super::reconstruction::reconstruct::reconstruct_face_value;

/// Represents the energy equation governing heat transfer in the domain.
/// Includes functionality for computing fluxes due to conduction and convection,
/// and handles various boundary conditions.
pub struct EnergyEquation {
    /// Coefficient for thermal conduction, representing the material's conductivity.
    pub thermal_conductivity: f64,
}

impl PhysicalEquation for EnergyEquation {
    /// Assembles the energy equation by computing energy fluxes for each face in the domain.
    ///
    /// # Parameters
    /// - `domain`: The mesh defining the simulation domain.
    /// - `fields`: The current field data, such as temperature and velocity.
    /// - `fluxes`: The fluxes to be computed and updated.
    /// - `boundary_handler`: Handler for boundary conditions.
    /// - `current_time`: The current simulation time.
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        self.calculate_energy_fluxes(
            domain,
            fields,
            fluxes,
            boundary_handler,
            current_time,
        );
    }
}

impl EnergyEquation {
    /// Creates a new energy equation with a specified thermal conductivity.
    pub fn new(thermal_conductivity: f64) -> Self {
        EnergyEquation { thermal_conductivity }
    }

    /// Calculates energy fluxes across all faces in the domain.
    ///
    /// This method computes conductive and convective fluxes, taking into account
    /// boundary conditions and internal cell interactions.
    fn calculate_energy_fluxes(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        _current_time: f64,
    ) {
        let mut geometry = Geometry::new();

        for face in domain.get_faces() {
            let face_vertices = domain.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue,
            };
            let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

            let cells = domain.get_cells_sharing_face(&face);
            let cell_a = cells
                .iter()
                .next()
                .map(|entry| entry.key().clone())
                .expect("Face should have at least one associated cell.");
            let temp_a = fields.get_scalar_field_value("temperature", &cell_a)
                .expect("Temperature not found for cell");
            let grad_temp_a = fields.get_vector_field_value("temperature_gradient", &cell_a)
                .expect("Temperature gradient not found for cell");

            let face_temperature = reconstruct_face_value(
                temp_a.0,
                grad_temp_a.0,
                geometry.compute_cell_centroid(domain, &cell_a),
                face_center,
            );

            let velocity = fields.get_vector_field_value("velocity", &face)
                .expect("Velocity not found at face");
            let face_normal = geometry.compute_face_normal(domain, &face, &cell_a)
                .expect("Normal not found for face");
            let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);

            let total_flux;

            if cells.len() == 1 {
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    match bc {
                        BoundaryCondition::Dirichlet(value) => {
                            let adjusted_face_temp = Scalar(value);
                            let temp_gradient_normal =
                                (adjusted_face_temp.0 - temp_a.0) /
                                Geometry::compute_distance(
                                    &geometry.compute_cell_centroid(domain, &cell_a),
                                    &face_center,
                                );
                            let conductive_flux = -self.thermal_conductivity * temp_gradient_normal * face_normal.magnitude();
                            let convective_flux = face_temperature * velocity.dot(&face_normal);
                            total_flux = Scalar((conductive_flux + convective_flux) * face_area);
                        }
                        BoundaryCondition::Neumann(flux) => {
                            total_flux = Scalar(flux * face_area);
                        }
                        _ => {
                            total_flux = self.compute_flux_combined(
                                temp_a, Scalar(face_temperature), &grad_temp_a, &face_normal, &velocity, face_area,
                            );
                        }
                    }
                } else {
                    total_flux = self.compute_flux_combined(
                        temp_a, Scalar(face_temperature), &grad_temp_a, &face_normal, &velocity, face_area,
                    );
                }
            } else {
                total_flux = self.compute_flux_combined(
                    temp_a, Scalar(face_temperature), &grad_temp_a, &face_normal, &velocity, face_area,
                );
            }

            fluxes.add_energy_flux(face, total_flux);
        }
    }

    /// Computes the combined flux due to conduction and convection.
    fn compute_flux_combined(
        &self,
        _temp_a: Scalar,
        face_temperature: Scalar,
        grad_temp_a: &Vector3,
        face_normal: &Vector3,
        velocity: &Vector3,
        face_area: f64,
    ) -> Scalar {
        let conductive_flux = -self.thermal_conductivity * grad_temp_a.dot(face_normal);
        let rho = 1.0; // Assume constant density
        let convective_flux = rho * face_temperature.0 * velocity.dot(face_normal);
        Scalar((conductive_flux + convective_flux) * face_area)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::bc_handler::{BoundaryConditionHandler, BoundaryCondition};
    use crate::domain::section::{Scalar, Vector3};
    use crate::interface_adapters::domain_adapter::DomainBuilder;
    use crate::equation::fields::{Fields, Fluxes};
    use crate::MeshEntity;

    /// Helper function to set up a basic 3D mesh for testing.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();

        // Add vertices
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [1.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 1.0, 0.0])
            .add_vertex(5, [0.0, 0.0, 1.0])
            .add_vertex(6, [1.0, 0.0, 1.0])
            .add_vertex(7, [1.0, 1.0, 1.0])
            .add_vertex(8, [0.0, 1.0, 1.0]);

        // Add a hexahedron cell
        builder.add_cell(vec![1, 2, 3, 4, 5, 6, 7, 8]);

        builder.build()
    }

    /// Helper function to populate field data for the mesh.
    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();

        // Set temperature field
        for cell in mesh.get_cells() {
            fields.set_scalar_field_value("temperature", cell, Scalar(300.0));
            fields.set_vector_field_value(
                "temperature_gradient",
                cell,
                Vector3([10.0, 5.0, -2.0]),
            );
        }

        // Set velocity field for faces
        for face in mesh.get_faces() {
            fields.set_vector_field_value(
                "velocity",
                face,
                Vector3([1.0, 0.0, 0.0]),
            );
        }

        fields
    }

    #[test]
    fn test_energy_equation_with_dirichlet_bc() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new();

        // Set Dirichlet boundary condition on a face
        let boundary_face = MeshEntity::Face(1);
        boundary_handler.set_bc(boundary_face, BoundaryCondition::Dirichlet(400.0));

        let energy_eq = EnergyEquation::new(0.5);

        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Check that fluxes were computed for the boundary face
        let computed_flux = fluxes.energy_fluxes.restrict(&boundary_face);

        assert!(
            computed_flux.is_some(),
            "Energy flux for boundary face was not computed."
        );
        println!(
            "Computed energy flux for boundary face: {:?}",
            computed_flux.unwrap().0
        );
    }

    #[test]
    fn test_energy_equation_with_neumann_bc() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new();

        // Set Neumann boundary condition on a face
        let boundary_face = MeshEntity::Face(2);
        boundary_handler.set_bc(boundary_face, BoundaryCondition::Neumann(5.0));

        let energy_eq = EnergyEquation::new(0.5);

        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Check that fluxes were computed for the boundary face
        let computed_flux = fluxes.energy_fluxes.restrict(&boundary_face);

        assert!(
            computed_flux.is_some(),
            "Energy flux for boundary face was not computed."
        );
        assert!(
            (computed_flux.unwrap().0 - 5.0 * 1.0).abs() < 1e-6,
            "Neumann flux mismatch. Expected {}, got {}",
            5.0 * 1.0,
            computed_flux.unwrap().0
        );
    }

    #[test]
    fn test_internal_face_flux_computation() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new(); // No boundary conditions

        let energy_eq = EnergyEquation::new(0.5);

        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Check that fluxes were computed for internal faces
        for face in mesh.get_faces() {
            if boundary_handler.get_bc(&face).is_none() {
                let computed_flux = fluxes.energy_fluxes.restrict(&face);

                assert!(
                    computed_flux.is_some(),
                    "Energy flux for internal face {:?} was not computed.",
                    face
                );
                println!(
                    "Computed energy flux for internal face {:?}: {:?}",
                    face,
                    computed_flux.unwrap().0
                );
            }
        }
    }

    #[test]
    fn test_energy_equation_scaling_with_thermal_conductivity() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new(); // No boundary conditions
    
        let energy_eq_high_conductivity = EnergyEquation::new(1.0);
        let energy_eq_low_conductivity = EnergyEquation::new(0.1);
    
        // Compute fluxes with high thermal conductivity
        energy_eq_high_conductivity.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);
        let flux_high: Vec<Scalar> = fluxes.energy_fluxes.all_data(); // Access energy fluxes only
    
        // Clear and compute fluxes with low thermal conductivity
        fluxes.energy_fluxes.clear(); // Clear only the energy fluxes
        energy_eq_low_conductivity.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);
        let flux_low: Vec<Scalar> = fluxes.energy_fluxes.all_data(); // Access energy fluxes only
    
        for (high, low) in flux_high.iter().zip(flux_low.iter()) {
            assert!(
                (high.0 / low.0 - 10.0).abs() < 1e-6,
                "Scaling mismatch between high and low conductivity fluxes."
            );
        }
    }
    
}
```

---

`src/equation/equation.rs`

```rust
use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::domain::section::{Vector3, Scalar};

/// Represents a generic equation framework for computing fluxes
/// in a simulation domain. This implementation is designed to handle
/// flux calculations based on velocity fields, boundary conditions, and
/// mesh geometry.
pub struct Equation {}

impl Equation {
    /// Calculates the fluxes for the given domain and stores them in the `fluxes` section.
    ///
    /// # Parameters
    /// - `domain`: The mesh representing the simulation domain.
    /// - `velocity_field`: Section containing velocity vectors for faces.
    /// - `_pressure_field`: Section containing scalar pressure values (not currently used).
    /// - `fluxes`: Section where computed fluxes will be stored.
    /// - `boundary_handler`: Handler for boundary conditions.
    /// - `current_time`: Current time in the simulation for time-dependent boundary conditions.
    pub fn calculate_fluxes(
        &self,
        domain: &Mesh,
        velocity_field: &Section<Vector3>,
        _pressure_field: &Section<Scalar>,
        fluxes: &mut Section<Vector3>,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        // Generate a mapping from entities to indices for matrix assembly.
        let entity_to_index = domain.get_entity_to_index();
        let boundary_entities = boundary_handler.get_boundary_faces();

        // Map boundary entities to indices.
        for (i, entity) in boundary_entities.iter().enumerate() {
            entity_to_index.insert(entity.clone(), i);
        }

        // Process each face in the domain for flux calculations.
        for face in domain.get_faces() {
            println!("Processing face: {:?}", face);

            // Retrieve face normal and area.
            let normal = match domain.get_face_normal(&face, None) {
                Some(normal) => normal,
                None => {
                    println!("Face {:?} has no normal! Skipping.", face);
                    continue; // Skip faces with no normal.
                }
            };

            let area = match domain.get_face_area(&face) {
                Some(area) => area,
                None => {
                    println!("Face {:?} has no area! Skipping.", face);
                    continue; // Skip faces with no area.
                }
            };

            // Compute the flux based on the velocity field.
            if let Some(velocity) = velocity_field.restrict(&face) {
                let velocity_dot_normal: f64 = velocity.iter()
                    .zip(normal.iter())
                    .map(|(v, n)| v * n)
                    .sum();

                // Compute the base flux using the velocity and face properties.
                let base_flux = Vector3([
                    velocity_dot_normal * area,
                    velocity_dot_normal * normal[1] * area,
                    velocity_dot_normal * normal[2] * area,
                ]);

                fluxes.set_data(face.clone(), base_flux);
            } else {
                println!("Face {:?} missing velocity data! Skipping.", face);
                continue;
            }

            // Apply boundary conditions to modify the flux.
            if let Some(bc) = boundary_handler.get_bc(&face) {
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        // Dirichlet condition: Set a fixed flux value.
                        fluxes.set_data(face.clone(), Vector3([value, 0.0, 0.0]));
                    }
                    BoundaryCondition::Neumann(flux_value) => {
                        // Neumann condition: Modify the existing flux.
                        let existing_flux = fluxes.restrict(&face).unwrap_or(Vector3([0.0, 0.0, 0.0]));
                        let updated_flux = Vector3([
                            existing_flux[0] + flux_value,
                            existing_flux[1],
                            existing_flux[2],
                        ]);
                        fluxes.set_data(face.clone(), updated_flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        // Robin condition: Scale the flux with `alpha` and add `beta`.
                        let existing_flux = fluxes.restrict(&face).unwrap_or(Vector3([0.0, 0.0, 0.0]));
                        let updated_flux = Vector3([
                            existing_flux[0] * alpha + beta,
                            existing_flux[1] * alpha,
                            existing_flux[2] * alpha,
                        ]);
                        fluxes.set_data(face.clone(), updated_flux);
                    }
                    _ => {
                        // Handle unsupported boundary conditions.
                        println!("Unsupported boundary condition for face {:?}: {:?}", face, bc);
                    }
                }
            } else {
                println!("No boundary condition for face {:?}", face);
                continue; // Skip faces with no boundary conditions.
            }
        }

        // Prepare for matrix assembly for boundary condition enforcement.
        let num_boundary_entities = boundary_entities.len();
        let mut matrix_storage = faer::Mat::<f64>::zeros(num_boundary_entities, num_boundary_entities);
        let mut rhs_storage = faer::Mat::<f64>::zeros(num_boundary_entities, 1);
        let mut matrix = matrix_storage.as_mut();
        let mut rhs = rhs_storage.as_mut();

        // Apply boundary conditions to construct the system of equations.
        boundary_handler.apply_bc(
            &mut matrix,
            &mut rhs,
            &boundary_entities,
            &entity_to_index,
            current_time,
        );
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface_adapters::domain_adapter::DomainBuilder;
    use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
    use crate::domain::section::{Vector3, Scalar};
    use crate::domain::Section;
    use crate::MeshEntity;

    #[test]
    fn test_calculate_fluxes_with_domain_builder() {
        let mut domain_builder = DomainBuilder::new();
        domain_builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_cell(vec![1, 2, 3]);
    
        let mesh = domain_builder.build();
    
        for face in mesh.get_faces() {
            assert!(mesh.get_face_area(&face).is_some(), "Face {:?} missing area!", face);
        }
    
        let velocity_field = Section::<Vector3>::new();
        velocity_field.set_data(MeshEntity::Face(1), Vector3([1.0, 0.0, 0.0]));
    
        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();
    
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(MeshEntity::Face(1), BoundaryCondition::Dirichlet(5.0));
    
        let equation = Equation {};
        equation.calculate_fluxes(
            &mesh,
            &velocity_field,
            &pressure_field,
            &mut fluxes,
            &boundary_handler,
            0.0,
        );
    
        assert!(fluxes.restrict(&MeshEntity::Face(1)).is_some());
        assert_eq!(
            fluxes.restrict(&MeshEntity::Face(1)).unwrap(),
            Vector3([5.0, 0.0, 0.0])
        );
    }

    #[test]
    fn test_boundary_conditions_integration() {
        let mut domain_builder = DomainBuilder::new();
        domain_builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_cell(vec![1, 2, 3]);
        
        let mesh = domain_builder.build();
        
        for face in mesh.get_faces() {
            assert!(mesh.get_face_normal(&face, None).is_some(), "Face {:?} missing normal!", face);
        }
        
        let velocity_field = Section::<Vector3>::new();
        // Set velocity data for all faces
        velocity_field.set_data(MeshEntity::Face(1), Vector3([1.0, 0.0, 0.0]));
        velocity_field.set_data(MeshEntity::Face(2), Vector3([0.5, 0.0, 0.0]));
        velocity_field.set_data(MeshEntity::Face(3), Vector3([0.0, 1.0, 0.0]));
        
        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();
        
        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(MeshEntity::Face(1), BoundaryCondition::Dirichlet(5.0));
        boundary_handler.set_bc(MeshEntity::Face(2), BoundaryCondition::Neumann(1.0));
        
        let equation = Equation {};
        equation.calculate_fluxes(
            &mesh,
            &velocity_field,
            &pressure_field,
            &mut fluxes,
            &boundary_handler,
            0.0,
        );
        
        // Expected flux values need to be recalculated
        let flux_face_2 = fluxes.restrict(&MeshEntity::Face(2)).unwrap();
        println!("Computed Flux for Face(2): {:?}", flux_face_2);
        
        assert_eq!(
            flux_face_2,
            Vector3([0.5, 0.35355339059327373, 0.0]) // Replace with correct values
        );
    }

    #[test]
    fn test_robin_boundary_condition() {
        let mut domain_builder = DomainBuilder::new();
        domain_builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_cell(vec![1, 2, 3]);
        
        let mesh = domain_builder.build();

        for face in mesh.get_faces() {
            assert!(mesh.get_face_normal(&face, None).is_some(), "Face {:?} missing normal!", face);
        }

        let velocity_field = Section::<Vector3>::new();
        velocity_field.set_data(MeshEntity::Face(1), Vector3([1.0, 0.0, 0.0]));
        velocity_field.set_data(MeshEntity::Face(2), Vector3([0.5, 0.0, 0.0]));
        velocity_field.set_data(MeshEntity::Face(3), Vector3([0.0, 1.0, 0.0]));

        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();

        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(
            MeshEntity::Face(3),
            BoundaryCondition::Robin { alpha: 0.8, beta: 2.0 },
        );

        let equation = Equation {};
        equation.calculate_fluxes(
            &mesh,
            &velocity_field,
            &pressure_field,
            &mut fluxes,
            &boundary_handler,
            0.0,
        );

        // Compute expected flux for the Robin condition manually
        let velocity = velocity_field.restrict(&MeshEntity::Face(3)).unwrap();
        let face_normal = mesh.get_face_normal(&MeshEntity::Face(3), None).unwrap();
        let face_area = mesh.get_face_area(&MeshEntity::Face(3)).unwrap();

        let velocity_dot_normal: f64 = velocity.iter().zip(face_normal.iter()).map(|(v, n)| v * n).sum();
        let base_flux = Vector3([
            velocity_dot_normal * face_area,
            velocity_dot_normal * face_normal[1] * face_area,
            velocity_dot_normal * face_normal[2] * face_area,
        ]);

        let expected_flux = Vector3([
            base_flux[0] * 0.8 + 2.0,
            base_flux[1] * 0.8,
            base_flux[2] * 0.8,
        ]);

        assert_eq!(
            fluxes.restrict(&MeshEntity::Face(3)).unwrap(),
            expected_flux,
            "Robin boundary condition flux mismatch!"
        );
    }

    #[test]
    fn test_internal_face_no_boundary_condition() {
        let mut domain_builder = DomainBuilder::new();
        domain_builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_cell(vec![1, 2, 3])
            .add_vertex(4, [1.0, 1.0, 0.0])
            .add_cell(vec![2, 3, 4]);

        let mesh = domain_builder.build();

        let velocity_field = Section::<Vector3>::new();
        velocity_field.set_data(MeshEntity::Face(2), Vector3([0.5, 0.0, 0.0])); // Internal face

        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();

        let boundary_handler = BoundaryConditionHandler::new(); // No boundary conditions applied

        let equation = Equation {};
        equation.calculate_fluxes(
            &mesh,
            &velocity_field,
            &pressure_field,
            &mut fluxes,
            &boundary_handler,
            0.0,
        );

        // Compute expected flux for internal face
        let velocity = velocity_field.restrict(&MeshEntity::Face(2)).unwrap();
        let face_normal = mesh.get_face_normal(&MeshEntity::Face(2), None).unwrap();
        let face_area = mesh.get_face_area(&MeshEntity::Face(2)).unwrap();

        let velocity_dot_normal: f64 = velocity.iter().zip(face_normal.iter()).map(|(v, n)| v * n).sum();
        let expected_flux = Vector3([
            velocity_dot_normal * face_area,
            velocity_dot_normal * face_normal[1] * face_area,
            velocity_dot_normal * face_normal[2] * face_area,
        ]);

        assert_eq!(
            fluxes.restrict(&MeshEntity::Face(2)).unwrap(),
            expected_flux,
            "Internal face flux computation mismatch!"
        );
    }

    #[test]
    fn test_face_missing_normal_or_area() {
        let mut domain_builder = DomainBuilder::new();
        domain_builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_cell(vec![1, 2]);

        let mesh = domain_builder.build();

        let velocity_field = Section::<Vector3>::new();
        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();

        let boundary_handler = BoundaryConditionHandler::new();

        let equation = Equation {};
        equation.calculate_fluxes(
            &mesh,
            &velocity_field,
            &pressure_field,
            &mut fluxes,
            &boundary_handler,
            0.0,
        );

        // No assertions here, just verify that faces without normal or area are skipped.
        // Check logs for "Face {:?} has no normal! Skipping." or "Face {:?} has no area! Skipping."
        println!("Check logs to ensure missing normal or area faces were skipped");
    }

}
```

---

`src/equation/fields.rs`

```rust
use rustc_hash::FxHashMap;
use crate::{domain::Section, MeshEntity};
use super::super::domain::section::{Vector3, Tensor3x3, Scalar, Vector2};

/// Trait `UpdateState` defines methods for updating and comparing the state of objects.
pub trait UpdateState {
    /// Updates the state of an object using a derivative and a time step (`dt`).
    fn update_state(&mut self, derivative: &Self, dt: f64);

    /// Computes the difference between the current state and another state.
    fn difference(&self, other: &Self) -> Self;

    /// Computes the norm (magnitude) of the state for convergence checks.
    fn norm(&self) -> f64;
}

/// Represents the fields (scalar, vector, and tensor) stored for a simulation domain.
#[derive(Clone, Debug)]
pub struct Fields {
    pub scalar_fields: FxHashMap<String, Section<Scalar>>, // Scalar fields (e.g., temperature, energy).
    pub vector_fields: FxHashMap<String, Section<Vector3>>, // Vector fields (e.g., velocity, momentum).
    pub tensor_fields: FxHashMap<String, Section<Tensor3x3>>, // Tensor fields (e.g., stress tensors).
}

impl Fields {
    /// Creates a new, empty instance of `Fields`.
    pub fn new() -> Self {
        Self {
            scalar_fields: FxHashMap::default(),
            vector_fields: FxHashMap::default(),
            tensor_fields: FxHashMap::default(),
        }
    }

    /// Retrieves the scalar field value for a specific entity.
    pub fn get_scalar_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Scalar> {
        self.scalar_fields.get(name)?.restrict(entity)
    }

    /// Sets the scalar field value for a specific entity, creating the field if it doesn't exist.
    pub fn set_scalar_field_value(&mut self, name: &str, entity: MeshEntity, value: Scalar) {
        if let Some(field) = self.scalar_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.scalar_fields.insert(name.to_string(), field);
        }
    }

    /// Retrieves the vector field value for a specific entity.
    pub fn get_vector_field_value(&self, name: &str, entity: &MeshEntity) -> Option<Vector3> {
        self.vector_fields.get(name)?.restrict(entity)
    }

    /// Sets the vector field value for a specific entity, creating the field if it doesn't exist.
    pub fn set_vector_field_value(&mut self, name: &str, entity: MeshEntity, value: Vector3) {
        if let Some(field) = self.vector_fields.get_mut(name) {
            field.set_data(entity, value);
        } else {
            let field = Section::new();
            field.set_data(entity, value);
            self.vector_fields.insert(name.to_string(), field);
        }
    }

    /// Updates the fields using the given fluxes.
    /// This operation typically happens after fluxes are computed across the mesh.
    pub fn update_from_fluxes(&mut self, fluxes: &Fluxes) {
        for entry in fluxes.energy_fluxes.data.iter() {
            let field = self
                .scalar_fields
                .entry("energy".to_string())
                .or_insert_with(Section::new);
            println!(
                "Updating scalar field 'energy' for entity {:?} with value {:?}",
                entry.key(),
                entry.value()
            );
            field.set_data(*entry.key(), *entry.value());
        }

        for entry in fluxes.momentum_fluxes.data.iter() {
            let field = self
                .vector_fields
                .entry("momentum".to_string())
                .or_insert_with(Section::new);
            println!(
                "Updating vector field 'momentum' for entity {:?} with value {:?}",
                entry.key(),
                entry.value()
            );
            field.set_data(*entry.key(), *entry.value());
        }
    }
}

impl UpdateState for Fields {
    /// Updates the current state of scalar and vector fields by applying
    /// the derivative (rate of change) multiplied by a time step (`dt`).
    ///
    /// This function iterates over each field in the `derivative` structure, adding
    /// the scaled derivative values to the corresponding fields in `self`. If a field
    /// does not exist in the current state, it creates a new field and applies the update.
    ///
    /// # Arguments
    /// * `derivative` - A `Fields` instance representing the rate of change for each field.
    /// * `dt` - The time step, used to scale the derivative before updating.
    fn update_state(&mut self, derivative: &Fields, dt: f64) {
        // Update scalar fields
        for (key, section) in &derivative.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get_mut(key) {
                // Update existing scalar field with the derivative
                state_section.update_with_derivative(section, dt);
            } else {
                // Create a new scalar field if it does not exist
                let new_section = Section::new();
                new_section.update_with_derivative(section, dt);
                self.scalar_fields.insert(key.clone(), new_section);
            }
        }

        // Update vector fields
        for (key, section) in &derivative.vector_fields {
            if let Some(state_section) = self.vector_fields.get_mut(key) {
                // Update existing vector field with the derivative
                state_section.update_with_derivative(section, dt);
            } else {
                // Create a new vector field if it does not exist
                let new_section = Section::new();
                new_section.update_with_derivative(section, dt);
                self.vector_fields.insert(key.clone(), new_section);
            }
        }
    }

    /// Computes the difference between the current state and another `Fields` instance.
    ///
    /// The difference is calculated for each field by subtracting the values in the `other`
    /// instance from the corresponding fields in `self`. If a field is not present in the
    /// current state, it is skipped.
    ///
    /// # Arguments
    /// * `other` - A `Fields` instance to compare against.
    ///
    /// # Returns
    /// A new `Fields` instance containing the difference for each field.
    fn difference(&self, other: &Self) -> Self {
        let mut result = self.clone();

        // Compute the difference for scalar fields
        for (key, section) in &other.scalar_fields {
            if let Some(state_section) = self.scalar_fields.get(key) {
                result.scalar_fields.insert(key.clone(), state_section.clone() - section.clone());
            }
        }

        // Compute the difference for vector fields
        for (key, section) in &other.vector_fields {
            if let Some(state_section) = self.vector_fields.get(key) {
                result.vector_fields.insert(key.clone(), state_section.clone() - section.clone());
            }
        }

        result
    }

    /// Computes the norm (magnitude) of the current state.
    ///
    /// The norm is calculated by summing the squares of all scalar field values and
    /// then taking the square root of the total. This provides a scalar metric
    /// representing the overall magnitude of the state, useful for convergence checks.
    ///
    /// # Returns
    /// A `f64` value representing the norm of the state.
    fn norm(&self) -> f64 {
        self.scalar_fields
            .values()
            .flat_map(|section| {
                // Iterate over each scalar field value, computing its square
                section
                    .data
                    .iter()
                    .map(|entry| entry.value().0 * entry.value().0)
            })
            .sum::<f64>() // Sum all squared values
            .sqrt() // Take the square root of the sum
    }
}


/// Represents flux data for various field quantities.
pub struct Fluxes {
    pub momentum_fluxes: Section<Vector3>, // Fluxes for momentum.
    pub energy_fluxes: Section<Scalar>,   // Fluxes for energy.
    pub turbulence_fluxes: Section<Vector2>, // Fluxes for turbulence models.
}

impl Fluxes {
    /// Creates a new instance of `Fluxes` with empty sections.
    pub fn new() -> Self {
        Self {
            momentum_fluxes: Section::new(),
            energy_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        }
    }

    /// Adds a momentum flux for the given entity.
    pub fn add_momentum_flux(&mut self, entity: MeshEntity, value: Vector3) {
        if let Some(mut current) = self.momentum_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.momentum_fluxes.set_data(entity, value);
        }
    }

    /// Adds an energy flux for the given entity.
    pub fn add_energy_flux(&mut self, entity: MeshEntity, value: Scalar) {
        if let Some(mut current) = self.energy_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.energy_fluxes.set_data(entity, value);
        }
    }

    /// Adds a turbulence flux for the given entity.
    pub fn add_turbulence_flux(&mut self, entity: MeshEntity, value: Vector2) {
        if let Some(mut current) = self.turbulence_fluxes.data.get_mut(&entity) {
            *current.value_mut() += value;
        } else {
            self.turbulence_fluxes.set_data(entity, value);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{section::Vector3, MeshEntity};

    fn create_test_entity(id: usize) -> MeshEntity {
        MeshEntity::Cell(id)
    }

    #[test]
    fn test_fields_initialization() {
        let fields = Fields::new();
        assert!(fields.scalar_fields.is_empty());
        assert!(fields.vector_fields.is_empty());
    }

    #[test]
    fn test_set_and_get_scalar_field_value() {
        let mut fields = Fields::new();
        let entity = create_test_entity(1);

        fields.set_scalar_field_value("test", entity, Scalar(1.0));
        let value = fields.get_scalar_field_value("test", &entity);

        assert_eq!(value.unwrap(), Scalar(1.0));
    }

    #[test]
    fn test_set_and_get_vector_field_value() {
        let mut fields = Fields::new();
        let entity = create_test_entity(1);

        fields.set_vector_field_value("velocity", entity, Vector3([1.0, 0.0, 0.0]));
        let value = fields.get_vector_field_value("velocity", &entity);

        assert_eq!(value.unwrap(), Vector3([1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_fluxes_add_and_update_fields() {
        let mut fields = Fields::new();
        let mut fluxes = Fluxes::new();
        let entity = create_test_entity(1);

        fluxes.add_energy_flux(entity, Scalar(2.0));
        fluxes.add_momentum_flux(entity, Vector3([0.5, 0.5, 0.5]));

        fields.update_from_fluxes(&fluxes);

        assert_eq!(
            fields.get_scalar_field_value("energy", &entity).unwrap(),
            Scalar(2.0)
        );
        assert_eq!(
            fields.get_vector_field_value("momentum", &entity).unwrap(),
            Vector3([0.5, 0.5, 0.5])
        );
    }

    #[test]
    fn test_update_state() {
        let mut fields = Fields::new();
        let mut derivative = Fields::new();
        let entity = create_test_entity(1);

        derivative.set_scalar_field_value("test", entity, Scalar(1.0));
        fields.update_state(&derivative, 2.0);

        assert_eq!(fields.get_scalar_field_value("test", &entity).unwrap(), Scalar(2.0));
    }

    #[test]
    fn test_difference_and_norm() {
        let mut fields = Fields::new();
        let mut other = Fields::new();
        let entity = create_test_entity(1);

        fields.set_scalar_field_value("test", entity, Scalar(2.0));
        other.set_scalar_field_value("test", entity, Scalar(1.0));

        let diff = fields.difference(&other);
        assert_eq!(diff.get_scalar_field_value("test", &entity).unwrap(), Scalar(1.0));

        let norm = fields.norm();
        assert!((norm - 2.0).abs() < 1e-6);
    }
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
use crate::domain::section::{Scalar, Vector3};
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
        field: &Section<Scalar>,
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
        field: &Section<Scalar>,
        gradient: &mut Section<Vector3>,
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
            gradient.set_data(cell, Vector3(grad_phi));
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

`src/domain/sieve.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::domain::mesh_entity::MeshEntity;

/// A `Sieve` struct that manages the relationships (arrows) between `MeshEntity` elements.
/// 
/// The `Sieve` uses an adjacency map to represent directed relations between entities in the mesh.
/// This structure enables querying relationships like cones, closures, and stars, making it a
/// versatile tool for managing mesh topology.
#[derive(Clone, Debug)]
pub struct Sieve {
    /// Thread-safe adjacency map.
    /// - **Key**: A `MeshEntity` representing the source entity in the relationship.
    /// - **Value**: A `DashMap` of `MeshEntity` objects that are related to the key entity.
    pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
}

impl Sieve {
    /// Creates a new `Sieve` instance with an empty adjacency map.
    ///
    /// # Returns
    /// - A new `Sieve` with no relationships.
    pub fn new() -> Self {
        Sieve {
            adjacency: DashMap::new(),
        }
    }

    /// Adds a directed relationship (arrow) between two `MeshEntity` elements.
    ///
    /// # Parameters
    /// - `from`: The source entity.
    /// - `to`: The target entity related to the source.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.adjacency
            .entry(from)
            .or_insert_with(DashMap::new)
            .insert(to, ());
    }

    /// Retrieves all entities directly related to the given entity (`point`).
    ///
    /// This operation is referred to as retrieving the **cone** of the entity.
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which the cone is retrieved.
    ///
    /// # Returns
    /// - A `Vec<MeshEntity>` containing entities in the cone, or `None` if there are no related entities.
    pub fn cone(&self, point: &MeshEntity) -> Option<Vec<MeshEntity>> {
        self.adjacency.get(point).map(|cone| {
            cone.iter().map(|entry| entry.key().clone()).collect()
        })
    }

    /// Computes the closure of a given `MeshEntity`.
    ///
    /// The closure includes:
    /// - The entity itself.
    /// - All entities it covers (cones) recursively.
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which the closure is computed.
    ///
    /// # Returns
    /// - A `DashMap` containing all entities in the closure.
    pub fn closure(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        let stack = DashMap::new();
        stack.insert(point.clone(), ());

        // Traverse all related entities using a stack
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
    ///
    /// The star includes:
    /// - The entity itself.
    /// - All entities that directly cover it (supports).
    /// - All entities that the entity directly points to (cone).
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which the star is computed.
    ///
    /// # Returns
    /// - A `DashMap` containing all entities in the star.
    pub fn star(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        result.insert(point.clone(), ());
        
        // Include supports (entities pointing to `point`)
        let supports = self.support(point);
        for support in supports {
            result.insert(support, ());
        }
        
        // Include cone (entities that `point` points to)
        if let Some(cones) = self.cone(point) {
            for cone_entity in cones {
                result.insert(cone_entity, ());
            }
        }
        
        result
    }

    /// Retrieves all entities that support the given entity (`point`).
    ///
    /// These are entities that have an arrow pointing to `point`.
    ///
    /// # Parameters
    /// - `point`: The `MeshEntity` for which supports are retrieved.
    ///
    /// # Returns
    /// - A `Vec<MeshEntity>` containing all supporting entities.
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
    ///
    /// The meet is defined as the intersection of their closures.
    ///
    /// # Parameters
    /// - `p`: The first `MeshEntity`.
    /// - `q`: The second `MeshEntity`.
    ///
    /// # Returns
    /// - A `DashMap` containing all entities in the intersection of the two closures.
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
    ///
    /// The join is defined as the union of their stars.
    ///
    /// # Parameters
    /// - `p`: The first `MeshEntity`.
    /// - `q`: The second `MeshEntity`.
    ///
    /// # Returns
    /// - A `DashMap` containing all entities in the union of the two stars.
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
    ///
    /// # Parameters
    /// - `func`: A closure that operates on each key-value pair in the adjacency map.
    ///   The function is called with a tuple containing:
    ///   - A reference to a `MeshEntity` key.
    ///   - A `Vec<MeshEntity>` of entities related to the key.
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

    /// Converts the internal adjacency map into a standard `HashMap`.
    ///
    /// The resulting map contains each `MeshEntity` as a key and its related entities as a `Vec<MeshEntity>`.
    ///
    /// # Returns
    /// - An `FxHashMap` containing the adjacency relationships.
    pub fn to_adjacency_map(&self) -> FxHashMap<MeshEntity, Vec<MeshEntity>> {
        let mut adjacency_map: FxHashMap<MeshEntity, Vec<MeshEntity>> = FxHashMap::default();

        // Convert the thread-safe DashMap to FxHashMap
        for entry in self.adjacency.iter() {
            let key = *entry.key();
            let values: Vec<MeshEntity> = entry.value().iter().map(|v| *v.key()).collect();
            adjacency_map.insert(key, values);
        }

        adjacency_map
    }
}
```

---

`src/domain/section.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;
use std::ops::{AddAssign, Mul};
use std::ops::{Add, Sub, Neg, Div};

/// Represents a 3D vector with three floating-point components.
///
/// The `Vector3` struct is a simple abstraction for 3D vectors, providing methods and operator
/// overloads for basic arithmetic operations, indexing, and iteration. This implementation is
/// designed for use in computational geometry, physics simulations, or similar fields requiring
/// manipulation of 3D data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector3(pub [f64; 3]);

impl AddAssign for Vector3 {
    /// Implements the `+=` operator for `Vector3`.
    ///
    /// Adds another `Vector3` to this vector component-wise. The operation is performed
    /// in place, modifying the original vector.
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;

    /// Implements scalar multiplication for `Vector3`.
    ///
    /// Multiplies each component of the vector by the scalar value `rhs`. The resulting
    /// `Vector3` is a new vector, leaving the original vector unchanged.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector3([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}

impl Vector3 {
    /// Returns an iterator over the components of the vector.
    ///
    /// The iterator allows read-only access to the vector's components in order.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}

impl std::ops::Index<usize> for Vector3 {
    type Output = f64;

    /// Implements indexing for `Vector3`.
    ///
    /// Allows direct read-only access to a specific component of the vector by its index.
    /// Valid indices are `0`, `1`, and `2` corresponding to the `x`, `y`, and `z` components.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Sub for Vector3 {
    type Output = Vector3;

    /// Implements subtraction for `Vector3`.
    ///
    /// Computes the difference between two vectors component-wise, returning a new vector.
    fn sub(self, rhs: Self) -> Self::Output {
        Vector3([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl Neg for Vector3 {
    type Output = Vector3;

    /// Implements negation for `Vector3`.
    ///
    /// Negates each component of the vector, returning a new vector with opposite direction.
    fn neg(self) -> Self::Output {
        Vector3([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl std::ops::IndexMut<usize> for Vector3 {
    /// Implements mutable indexing for `Vector3`.
    ///
    /// Allows direct modification of a specific component of the vector by its index.
    /// Valid indices are `0`, `1`, and `2` corresponding to the `x`, `y`, and `z` components.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Vector3 {
    type Item = f64;
    type IntoIter = std::array::IntoIter<f64, 3>;

    /// Converts the vector into an iterator of its components.
    ///
    /// Consumes the `Vector3` and produces an iterator that yields the `x`, `y`, and `z`
    /// components in order.
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Vector3 {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    /// Converts a reference to the vector into an iterator of its components.
    ///
    /// Produces an iterator that yields immutable references to the `x`, `y`, and `z`
    /// components in order.
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Add for Vector3 {
    type Output = Vector3;

    /// Implements addition for `Vector3`.
    ///
    /// Adds two vectors component-wise, returning a new vector with the result.
    fn add(self, rhs: Self) -> Self::Output {
        Vector3([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl<'a> Add for &'a Vector3 {
    type Output = Vector3;

    /// Implements addition for references to `Vector3`.
    ///
    /// Adds two vectors component-wise, returning a new `Vector3` without consuming the operands.
    /// This implementation allows adding borrowed `Vector3` instances, which can improve
    /// performance by avoiding unnecessary cloning or copying.
    fn add(self, rhs: Self) -> Self::Output {
        Vector3([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Vector3 {
    /// Computes the magnitude (norm) of the vector.
    pub fn magnitude(&self) -> f64 {
        self.0.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// Computes the dot product of two vectors.
    pub fn dot(&self, other: &Vector3) -> f64 {
        self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
    }
}

impl Mul<Vector3> for f64 {
    type Output = Vector3;

    /// Implements scalar multiplication for `Vector3`.
    ///
    /// Multiplies a scalar `f64` by a `Vector3`, scaling each component of the vector by the scalar.
    /// This implementation consumes the `Vector3` operand and produces a new scaled vector.
    fn mul(self, rhs: Vector3) -> Self::Output {
        Vector3([
            self * rhs.0[0],
            self * rhs.0[1],
            self * rhs.0[2],
        ])
    }
}

impl Mul<&Vector3> for f64 {
    type Output = Vector3;

    /// Implements scalar multiplication for a reference to `Vector3`.
    ///
    /// Multiplies a scalar `f64` by a borrowed `Vector3`, scaling each component of the vector
    /// without consuming the `Vector3`.
    fn mul(self, rhs: &Vector3) -> Self::Output {
        Vector3([
            self * rhs.0[0],
            self * rhs.0[1],
            self * rhs.0[2],
        ])
    }
}

/// Represents a 3x3 tensor with floating-point components.
///
/// The `Tensor3x3` struct is a simple abstraction for rank-2 tensors in 3D space.
/// It supports component-wise arithmetic operations, including addition and scalar multiplication.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tensor3x3(pub [[f64; 3]; 3]);

impl AddAssign for Tensor3x3 {
    /// Implements the `+=` operator for `Tensor3x3`.
    ///
    /// Adds another `Tensor3x3` to this tensor component-wise. The operation is performed
    /// in place, modifying the original tensor.
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

    /// Implements scalar multiplication for `Tensor3x3`.
    ///
    /// Multiplies each component of the tensor by a scalar `rhs`. The resulting tensor
    /// is a new `Tensor3x3`, leaving the original tensor unchanged.
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

/// Represents a scalar value.
///
/// The `Scalar` struct is a wrapper for a floating-point value, used in mathematical
/// and physical computations where type safety or domain-specific semantics are desired.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scalar(pub f64);

impl AddAssign for Scalar {
    /// Implements the `+=` operator for `Scalar`.
    ///
    /// Adds another scalar to this scalar. The operation is performed in place.
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Mul<f64> for Scalar {
    type Output = Scalar;

    /// Implements scalar multiplication for `Scalar`.
    ///
    /// Multiplies the scalar value by another scalar `rhs`. The resulting value
    /// is wrapped in a new `Scalar`.
    fn mul(self, rhs: f64) -> Self::Output {
        Scalar(self.0 * rhs)
    }
}

/// Represents a 2D vector with two floating-point components.
///
/// The `Vector2` struct is a simple abstraction for 2D vectors, providing methods and
/// operator overloads for basic arithmetic operations. It is suitable for computations
/// in 2D geometry, physics, or graphics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector2(pub [f64; 2]);

impl AddAssign for Vector2 {
    /// Implements the `+=` operator for `Vector2`.
    ///
    /// Adds another `Vector2` to this vector component-wise. The operation is performed
    /// in place, modifying the original vector.
    fn add_assign(&mut self, other: Self) {
        for i in 0..2 {
            self.0[i] += other.0[i];
        }
    }
}

impl Mul<f64> for Vector2 {
    type Output = Vector2;

    /// Implements scalar multiplication for `Vector2`.
    ///
    /// Multiplies each component of the vector by a scalar `rhs`. The resulting vector
    /// is a new `Vector2`, leaving the original vector unchanged.
    fn mul(self, rhs: f64) -> Self::Output {
        Vector2([self.0[0] * rhs, self.0[1] * rhs])
    }
}


/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.
///
/// The `Section` structure is designed to store data (of generic type `T`) linked to entities
/// in a computational mesh (`MeshEntity`). It provides methods for efficient data management,
/// parallel updates, and mathematical operations. This abstraction is particularly useful
/// in simulations and finite element/volume computations where values like scalars or vectors
/// are associated with mesh components.
#[derive(Clone, Debug)]
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.
    ///
    /// The `DashMap` ensures thread-safe operations and allows concurrent reads and writes
    /// on the data without explicit locking, making it ideal for parallel computations.
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

    /// Associates a given `MeshEntity` with a value of type `T`.
    ///
    /// If the `MeshEntity` already exists in the section, its value is overwritten.
    ///
    /// # Parameters
    /// - `entity`: The `MeshEntity` to associate with the value.
    /// - `value`: The value of type `T` to store.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves a copy of the data associated with the specified `MeshEntity`, if it exists.
    ///
    /// # Parameters
    /// - `entity`: The `MeshEntity` whose data is being requested.
    ///
    /// # Returns
    /// An `Option<T>` containing the associated value if it exists, or `None` if the entity
    /// is not in the section.
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T> {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates all data values in the section in parallel using the provided function.
    ///
    /// # Parameters
    /// - `update_fn`: A function that takes a mutable reference to a value of type `T`
    ///   and updates it. This function must be thread-safe (`Sync` + `Send`) as updates
    ///   are applied concurrently.
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
    {
        // Collect all keys to avoid holding references during parallel iteration.
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        // Update values in parallel.
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Updates the section by adding the derivative multiplied by a time step `dt`.
    ///
    /// This method performs an in-place update of the section's values, adding the product
    /// of a derivative (from another section) and a scalar time step `dt`. If an entity
    /// exists in the derivative but not in the current section, it is added.
    ///
    /// # Parameters
    /// - `derivative`: A `Section` containing the derivative values.
    /// - `dt`: A scalar value representing the time step.
    pub fn update_with_derivative(&self, derivative: &Section<T>, dt: f64) {
        for entry in derivative.data.iter() {
            let entity = entry.key();
            let deriv_value = entry.value().clone() * dt;

            // Update existing value or insert a new one.
            if let Some(mut state_value) = self.data.get_mut(entity) {
                *state_value.value_mut() += deriv_value;
            } else {
                self.data.insert(*entity, deriv_value);
            }
        }
    }

    /// Returns a list of all `MeshEntity` objects associated with this section.
    ///
    /// # Returns
    /// A `Vec<MeshEntity>` containing all the keys from the section's data map.
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Returns all data stored in the section as a vector of immutable copies.
    ///
    /// # Returns
    /// A `Vec<T>` containing all the values stored in the section.
    /// Requires `T` to implement `Clone`.
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Clears all data from the section.
    ///
    /// This method removes all entries from the section, leaving it empty.
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Scales all data values in the section by the specified factor.
    ///
    /// This method multiplies each value in the section by the given scalar factor.
    /// The updates are applied in parallel for efficiency.
    ///
    /// # Parameters
    /// - `factor`: The scalar value by which to scale all entries.
    pub fn scale(&self, factor: f64) {
        self.parallel_update(|value| {
            *value = value.clone() * factor;
        });
    }
}

// Add for Section<Scalar>
impl Add for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements addition for `Section<Scalar>`.
    ///
    /// This operator performs a component-wise addition of two `Section<Scalar>` instances.
    /// If a key exists in both sections, their corresponding values are added. If a key exists
    /// in only one section, its value is copied to the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Scalar>` operand (consumed).
    /// - `rhs`: The second `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` containing the sum of the two sections.
    fn add(self, rhs: Self) -> Self::Output {
        let result = self.clone(); // Clone the first section to use as a base
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair from the second section
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 += value.0; // Add values if the key exists in both sections
            } else {
                result.set_data(*key, *value); // Insert the value if the key only exists in `rhs`
            }
        }
        result
    }
}

// Sub for Section<Scalar>
impl Sub for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements subtraction for `Section<Scalar>`.
    ///
    /// This operator performs a component-wise subtraction of two `Section<Scalar>` instances.
    /// If a key exists in both sections, their corresponding values are subtracted. If a key exists
    /// in only one section, its value is added or negated in the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Scalar>` operand (consumed).
    /// - `rhs`: The second `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` containing the difference of the two sections.
    fn sub(self, rhs: Self) -> Self::Output {
        let result = self.clone(); // Clone the first section to use as a base
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair from the second section
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 -= value.0; // Subtract values if the key exists in both sections
            } else {
                result.set_data(*key, Scalar(-value.0)); // Negate and insert the value if the key only exists in `rhs`
            }
        }
        result
    }
}

// Neg for Section<Scalar>
impl Neg for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements negation for `Section<Scalar>`.
    ///
    /// This operator negates each value in the `Section<Scalar>` component-wise.
    ///
    /// # Parameters
    /// - `self`: The `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` with all values negated.
    fn neg(self) -> Self::Output {
        let result = self.clone(); // Clone the section to preserve original data
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 = -value.0; // Negate the scalar value
        }
        result
    }
}

// Div for Section<Scalar>
impl Div<f64> for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements scalar division for `Section<Scalar>`.
    ///
    /// Divides each value in the `Section<Scalar>` by a scalar `rhs` component-wise.
    ///
    /// # Parameters
    /// - `self`: The `Section<Scalar>` operand (consumed).
    /// - `rhs`: A scalar `f64` divisor.
    ///
    /// # Returns
    /// A new `Section<Scalar>` with all values scaled by `1/rhs`.
    fn div(self, rhs: f64) -> Self::Output {
        let result = self.clone(); // Clone the section to preserve original data
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 /= rhs; // Divide the scalar value by `rhs`
        }
        result
    }
}

// Sub for Section<Vector3>
impl Sub for Section<Vector3> {
    type Output = Section<Vector3>;

    /// Implements subtraction for `Section<Vector3>`.
    ///
    /// This operator performs a component-wise subtraction of two `Section<Vector3>` instances.
    /// If a key exists in both sections, their corresponding vectors are subtracted. If a key exists
    /// in only one section, its value is added or negated in the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Vector3>` operand (consumed).
    /// - `rhs`: The second `Section<Vector3>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Vector3>` containing the difference of the two sections.
    fn sub(self, rhs: Self) -> Self::Output {
        let result = Section::new(); // Create a new section to hold the result

        // Process all keys from the `rhs` section
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair();
            if let Some(current) = self.data.get(key) {
                result.set_data(*key, *current.value() - *value); // Subtract if the key exists in both sections
            } else {
                result.set_data(*key, -*value); // Negate and add if the key only exists in `rhs`
            }
        }

        // Process all keys from the `self` section that are not in `rhs`
        for entry in self.data.iter() {
            let (key, value) = entry.pair();
            if !rhs.data.contains_key(key) {
                result.set_data(*key, *value); // Add the value if the key only exists in `self`
            }
        }

        result
    }
}
```

---

`src/interface_adapters/domain_adapter.rs`

```rust

use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};
use crate::domain::mesh::geometry_validation::GeometryValidation;
use crate::domain::mesh::reordering::cuthill_mckee;
use crate::Geometry;

/// `DomainBuilder` is a utility for constructing a mesh domain by incrementally adding vertices,
/// edges, faces, and cells. It provides methods to build the mesh and apply reordering for
/// performance optimization.
pub struct DomainBuilder {
    mesh: Mesh,
}

impl DomainBuilder {
    /// Creates a new `DomainBuilder` with an empty mesh.
    pub fn new() -> Self {
        Self {
            mesh: Mesh::new(),
        }
    }

    /// Adds a vertex to the domain with a specified ID and coordinates.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    pub fn add_vertex(&mut self, id: usize, coords: [f64; 3]) -> &mut Self {
        self.mesh.set_vertex_coordinates(id, coords);
        self.mesh
            .entities
            .write()
            .unwrap()
            .insert(MeshEntity::Vertex(id));
        self
    }

    /// Adds an edge connecting two vertices.
    ///
    /// # Arguments
    ///
    /// * `vertex1` - The ID of the first vertex.
    /// * `vertex2` - The ID of the second vertex.
    pub fn add_edge(&mut self, vertex1: usize, vertex2: usize) -> &mut Self {
        let edge_id = self.mesh.entities.read().unwrap().len() + 1; // Ensure unique IDs
        let edge = MeshEntity::Edge(edge_id);

        // Add relationships in the sieve
        self.mesh
            .add_arrow(MeshEntity::Vertex(vertex1), edge);
        self.mesh
            .add_arrow(MeshEntity::Vertex(vertex2), edge);
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex1));
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex2));

        // Add the edge to the entities set
        self.mesh.entities.write().unwrap().insert(edge);
        self
    }

    /// Adds a cell with the given vertices and automatically creates faces.
    ///
    /// # Arguments
    ///
    /// * `vertex_ids` - A vector of vertex IDs that define the cell.
    pub fn add_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);
    
        let num_faces = self.mesh.count_entities(&MeshEntity::Face(0));
        let mut face_id_counter = num_faces + 1;
    
        let num_vertices = vertex_ids.len();
        for i in 0..num_vertices {
            let v1 = vertex_ids[i];
            let v2 = vertex_ids[(i + 1) % num_vertices];
            let face = MeshEntity::Face(face_id_counter);
            face_id_counter += 1;
    
            let vertex1 = MeshEntity::Vertex(v1);
            let vertex2 = MeshEntity::Vertex(v2);
    
            // Add relationships in the sieve
            self.mesh.add_arrow(face.clone(), vertex1.clone());
            self.mesh.add_arrow(face.clone(), vertex2.clone());
            
            // Optionally, add reverse relationships if needed
            self.mesh.add_arrow(vertex1, face.clone());
            self.mesh.add_arrow(vertex2, face.clone());
    
            self.mesh.entities.write().unwrap().insert(face);
    
            // Compute and store the face area in GeometryCache
            if let Some(coords1) = self.mesh.get_vertex_coordinates(v1) {
                if let Some(coords2) = self.mesh.get_vertex_coordinates(v2) {
                    let area = Geometry::compute_distance(&coords1, &coords2);
                    let geometry = Geometry::new();
                    geometry.cache.lock().unwrap().entry(face.get_id()).or_default().area = Some(area);
                }
            }
    
            self.mesh.add_arrow(cell, face);
        }
    
        for &vertex_id in &vertex_ids {
            let vertex = MeshEntity::Vertex(vertex_id);
            self.mesh.entities.write().unwrap().insert(vertex);
        }
        self.mesh.entities.write().unwrap().insert(cell);
        self
    }

    pub fn add_tetrahedron_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        assert_eq!(vertex_ids.len(), 4, "Tetrahedron must have 4 vertices");
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        let face_id_start = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;

        let face_vertices = vec![
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[2]], // Face 1
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[3]], // Face 2
            vec![vertex_ids[1], vertex_ids[2], vertex_ids[3]], // Face 3
            vec![vertex_ids[2], vertex_ids[0], vertex_ids[3]], // Face 4
        ];

        for (i, fv) in face_vertices.iter().enumerate() {
            let face_id = face_id_start + i;
            let face = MeshEntity::Face(face_id);

            // Add arrows from face to vertices
            for &vid in fv {
                let vertex = MeshEntity::Vertex(vid);
                self.mesh.add_arrow(face.clone(), vertex.clone());
                self.mesh.add_arrow(vertex.clone(), face.clone());
            }

            self.mesh.entities.write().unwrap().insert(face.clone());

            // Add arrows from cell to face
            self.mesh.add_arrow(cell.clone(), face.clone());
            self.mesh.add_arrow(face.clone(), cell.clone());
        }

        // Add arrows from cell to vertices
        for &vid in &vertex_ids {
            let vertex = MeshEntity::Vertex(vid);
            self.mesh.add_arrow(cell.clone(), vertex.clone());
            self.mesh.add_arrow(vertex.clone(), cell.clone());
        }

        self.mesh.entities.write().unwrap().insert(cell);

        self
    }

    /// Applies reordering to improve solver performance using the Cuthill-McKee algorithm.
    pub fn apply_reordering(&mut self) {
        let entities: Vec<_> = self
            .mesh
            .entities
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();
        let adjacency: FxHashMap<_, _> = self.mesh.sieve.to_adjacency_map();
        let reordered = cuthill_mckee(&entities, &adjacency);

        // Apply the reordering
        self.mesh
            .apply_reordering(&reordered.iter().map(|e| e.get_id()).collect::<Vec<_>>());
    }

    /// Performs geometry validation to ensure mesh integrity.
    pub fn validate_geometry(&self) {
        assert!(
            GeometryValidation::test_vertex_coordinates(&self.mesh).is_ok(),
            "Geometry validation failed: Duplicate or invalid vertex coordinates."
        );
    }

    /// Finalizes and returns the built `Mesh`.
    pub fn build(self) -> Mesh {
        self.mesh
    }
}

/// Represents a domain entity with optional boundary conditions and material properties.
pub struct DomainEntity {
    pub entity: MeshEntity,
    pub boundary_conditions: Option<String>,
    pub material_properties: Option<String>,
}

impl DomainEntity {
    /// Creates a new `DomainEntity`.
    ///
    /// # Arguments
    ///
    /// * `entity` - The mesh entity to associate with this domain entity.
    pub fn new(entity: MeshEntity) -> Self {
        Self {
            entity,
            boundary_conditions: None,
            material_properties: None,
        }
    }

    /// Sets boundary conditions for the entity.
    ///
    /// # Arguments
    ///
    /// * `bc` - A string describing the boundary condition.
    pub fn set_boundary_conditions(mut self, bc: &str) -> Self {
        self.boundary_conditions = Some(bc.to_string());
        self
    }

    /// Sets material properties for the entity.
    ///
    /// # Arguments
    ///
    /// * `properties` - A string describing the material properties.
    pub fn set_material_properties(mut self, properties: &str) -> Self {
        self.material_properties = Some(properties.to_string());
        self
    }
}
```

---

`src/domain/mesh/mod.rs`

```rust
pub mod entities;
pub mod geometry;
pub mod reordering;
pub mod boundary;
pub mod hierarchical;
pub mod topology;
pub mod geometry_validation;
pub mod boundary_validation;

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crossbeam::channel::{Sender, Receiver};
use lazy_static::lazy_static;

// Delegate methods to corresponding modules

/// Represents the mesh structure, which is composed of a sieve for entity management,  
/// a set of mesh entities, vertex coordinates, and channels for boundary data.  
/// 
/// The `Mesh` struct is the central component for managing mesh entities and  
/// their relationships. It stores entities such as vertices, edges, faces,  
/// and cells, along with their geometric data and boundary-related information.  
#[derive(Clone, Debug)]
pub struct Mesh {
    /// The sieve structure used for organizing the mesh entities' relationships.  
    pub sieve: Arc<Sieve>,  
    /// A thread-safe, read-write lock for managing mesh entities.  
    /// This set contains all `MeshEntity` objects in the mesh.  
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,  
    /// A map from vertex indices to their 3D coordinates.  
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,  
    /// An optional channel sender for transmitting boundary data related to mesh entities.  
    pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,  
    /// An optional channel receiver for receiving boundary data related to mesh entities.  
    pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,  
}

lazy_static! {
    static ref GLOBAL_MESH: Arc<RwLock<Mesh>> = Arc::new(RwLock::new(Mesh::new()));
}

impl Mesh {
    /// Creates a new instance of the `Mesh` struct with initialized components.  
    /// 
    /// This method sets up the sieve, entity set, vertex coordinate map,  
    /// and a channel for boundary data communication between mesh components.  
    ///
    /// The `Sender` and `Receiver` are unbounded channels used to pass boundary  
    /// data between mesh modules asynchronously.
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam::channel::unbounded();
        Mesh {
            sieve: Arc::new(Sieve::new()),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender),
            boundary_data_receiver: Some(receiver),
        }
    }

    pub fn global() -> Arc<RwLock<Mesh>> {
        GLOBAL_MESH.clone()
    }
}

#[cfg(test)]
pub mod tests;
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
    ///
    /// This method inserts the entity into the mesh's thread-safe `entities` set,
    /// ensuring it becomes part of the mesh's domain. The `entities` set tracks all
    /// vertices, edges, faces, and cells in the mesh.
    pub fn add_entity(&self, entity: MeshEntity) {
        self.entities.write().unwrap().insert(entity);
    }

    /// Establishes a directed relationship (arrow) between two mesh entities.
    ///
    /// This relationship is added to the sieve structure, representing a connection
    /// from the `from` entity to the `to` entity. Relationships are useful for
    /// defining adjacency and connectivity in the mesh.
    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Adds a directed arrow between two mesh entities.
    ///
    /// This is a direct delegate to the `Sieve`'s `add_arrow` method, simplifying
    /// the addition of directed relationships in the mesh's connectivity structure.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Sets the 3D coordinates of a vertex and adds the vertex to the mesh.
    ///
    /// This method ensures the vertex is registered in the `vertex_coordinates` map
    /// with its corresponding coordinates, and also adds the vertex to the mesh's
    /// `entities` set if not already present.
    ///
    /// # Arguments
    /// - `vertex_id`: Unique identifier of the vertex.
    /// - `coords`: 3D coordinates of the vertex.
    pub fn set_vertex_coordinates(&mut self, vertex_id: usize, coords: [f64; 3]) {
        self.vertex_coordinates.insert(vertex_id, coords);
        self.add_entity(MeshEntity::Vertex(vertex_id));
    }

    /// Retrieves the 3D coordinates of a vertex by its identifier.
    ///
    /// If the vertex does not exist in the `vertex_coordinates` map, this method
    /// returns `None`.
    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        self.vertex_coordinates.get(&vertex_id).cloned()
    }

    /// Counts the number of entities of a specific type in the mesh.
    ///
    /// This method iterates through all entities in the mesh and counts those
    /// matching the type specified in `entity_type`.
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

    /// Applies a user-defined function to each entity in the mesh in parallel.
    ///
    /// The function is executed concurrently using Rayon’s parallel iterator,
    /// ensuring efficient processing of large meshes.
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();
        entities.par_iter().for_each(func);
    }

    /// Retrieves all `Cell` entities from the mesh.
    ///
    /// This method filters the mesh's `entities` set and collects all elements
    /// classified as cells, returning them as a vector.
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Retrieves all `Face` entities from the mesh.
    ///
    /// Similar to `get_cells`, this method filters the mesh's `entities` set and
    /// collects all elements classified as faces.
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    /// Retrieves the vertices of a given face entity.
    ///
    /// This method queries the sieve structure to find all vertices directly
    /// connected to the specified face entity.
    pub fn get_vertices_of_face(&self, face: &MeshEntity) -> Vec<MeshEntity> {
        self.sieve.cone(face).unwrap_or_default()
            .into_iter()
            .filter(|e| matches!(e, MeshEntity::Vertex(_)))
            .collect()
    }

    /// Computes properties for each entity in the mesh in parallel.
    ///
    /// The user provides a function `compute_fn` that maps a `MeshEntity` to a
    /// property of type `PropertyType`. This function is applied to all entities
    /// in the mesh concurrently, and the results are returned in a map.
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

    /// Retrieves the ordered neighboring cells for a given cell.
    ///
    /// This method is useful for numerical methods that require consistent ordering
    /// of neighbors, such as flux calculations or gradient reconstruction. Neighbors
    /// are sorted by their unique identifiers to ensure deterministic results.
    ///
    /// # Returns
    /// A vector of neighboring cells sorted by ID.
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
    ///
    /// This method creates a mapping from each entity to a unique index, which can
    /// be useful for tasks like matrix assembly or entity-based data storage.
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
use crate::domain;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;
use crate::domain::section::Vector3;

impl Mesh {
    /// Retrieves all the faces of a given cell.
    ///
    /// This function collects entities connected to the provided cell and filters
    /// them to include only face entities. The result is returned as a `DashMap`.
    ///
    /// # Arguments
    /// * `cell` - A `MeshEntity` representing the cell whose faces are being retrieved.
    ///
    /// # Returns
    /// * `Option<DashMap<MeshEntity, ()>>` - A map of face entities connected to the cell.
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

    /// Retrieves all the cells that share a given face.
    ///
    /// This function identifies all cell entities that share a specified face,
    /// filtering only valid cell entities present in the mesh.
    ///
    /// # Arguments
    /// * `face` - A `MeshEntity` representing the face.
    ///
    /// # Returns
    /// * `DashMap<MeshEntity, ()>` - A map of cell entities sharing the face.
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
    ///
    /// # Arguments
    /// * `cell_i` - The first cell entity.
    /// * `cell_j` - The second cell entity.
    ///
    /// # Returns
    /// * `f64` - The computed distance between the centroids of the two cells.
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.
    ///
    /// # Arguments
    /// * `face` - The face entity for which to compute the area.
    ///
    /// # Returns
    /// * `Option<f64>` - The area of the face, or `None` if the face shape is unsupported.
    pub fn get_face_area(&self, face: &MeshEntity) -> Option<f64> {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let mut geometry = Geometry::new();
        let face_id = face.get_id();
        Some(geometry.compute_face_area(face_id, face_shape, &face_vertices))
    }

    /// Computes the centroid of a cell based on its vertices.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which to compute the centroid.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the cell's centroid.
    ///
    /// # Panics
    /// This function panics if the cell has an unsupported number of vertices.
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

    /// Retrieves all vertices connected to a given vertex via shared cells.
    ///
    /// # Arguments
    /// * `vertex` - The vertex entity for which to find neighboring vertices.
    ///
    /// # Returns
    /// * `Vec<MeshEntity>` - A list of vertex entities neighboring the given vertex.
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

    /// Returns an iterator over all vertex IDs in the mesh.
    ///
    /// # Returns
    /// * `impl Iterator<Item = &usize>` - An iterator over vertex IDs.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on its vertex count.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which to determine the shape.
    ///
    /// # Returns
    /// * `Result<CellShape, String>` - The determined cell shape or an error message if unsupported.
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

    /// Retrieves the vertices of a cell, sorted by vertex ID.
    ///
    /// # Arguments
    /// * `cell` - The cell entity whose vertices are being retrieved.
    ///
    /// # Returns
    /// * `Vec<[f64; 3]>` - The 3D coordinates of the cell's vertices, sorted by ID.
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

    /// Retrieves the vertices of a face, sorted by vertex ID.
    ///
    /// # Arguments
    /// * `face` - The face entity whose vertices are being retrieved.
    ///
    /// # Returns
    /// * `Vec<[f64; 3]>` - The 3D coordinates of the face's vertices, sorted by ID.
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

    /// Computes the outward normal vector for a face based on its shape and vertices.
    ///
    /// Optionally adjusts the normal's orientation based on a reference cell's centroid.
    ///
    /// # Arguments
    /// * `face` - The face entity for which to compute the normal.
    /// * `reference_cell` - An optional reference cell entity to adjust the orientation.
    ///
    /// # Returns
    /// * `Option<Vector3>` - The computed normal vector, or `None` if the face shape is unsupported.
    pub fn get_face_normal(
        &self,
        face: &MeshEntity,
        reference_cell: Option<&MeshEntity>,
    ) -> Option<Vector3> {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let geometry = Geometry::new();
        let normal = match face_shape {
            FaceShape::Edge => geometry.compute_edge_normal(&face_vertices),
            FaceShape::Triangle => geometry.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => geometry.compute_quadrilateral_normal(&face_vertices),
        };

        // Adjust normal orientation if a reference cell is provided
        if let Some(cell) = reference_cell {
            let cell_centroid = self.get_cell_centroid(cell);
            let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            let to_cell_vector = [
                cell_centroid[0] - face_centroid[0],
                cell_centroid[1] - face_centroid[1],
                cell_centroid[2] - face_centroid[2],
            ];

            let dot_product = normal[0] * to_cell_vector[0]
                + normal[1] * to_cell_vector[1]
                + normal[2] * to_cell_vector[2];

            if dot_product < 0.0 {
                // Reverse the normal if it points inward
                return Some(domain::section::Vector3([-normal[0], -normal[1], -normal[2]]));
            }
        }

        Some(domain::section::Vector3(normal))
    }
}
```

---

`src/boundary/bc_handler.rs`

```rust
//! Boundary Condition Module
//!
//! This module provides functionality for defining and applying various types of boundary
//! conditions (Dirichlet, Neumann, Robin, etc.) to mesh entities in a computational fluid
//! dynamics (CFD) simulation.
//!
//! # Overview
//! - `BoundaryCondition`: Enum representing supported boundary condition types.
//! - `BoundaryConditionHandler`: Manages boundary conditions for mesh entities.
//! - `BoundaryConditionApply`: Trait for applying boundary conditions to system matrices.
//!
//! # Computational Context
//! Boundary conditions play a crucial role in solving partial differential equations (PDEs)
//! in CFD. This module ensures compatibility with Hydra's unstructured grid and time-stepping
//! framework.

use dashmap::DashMap;
use std::sync::{Arc, RwLock};
use lazy_static::lazy_static;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use crate::boundary::mixed::MixedBC;
use crate::boundary::cauchy::CauchyBC;
use crate::boundary::solid_wall::SolidWallBC;
use faer::MatMut;

pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// Wrapper for function-based boundary conditions, allowing metadata for equality and debug.
#[derive(Clone)]
pub struct FunctionWrapper {
    pub description: String, // Metadata to identify the function
    pub function: BoundaryConditionFn,
}

impl PartialEq for FunctionWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.description == other.description
    }
}

impl std::fmt::Debug for FunctionWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FunctionWrapper {{ description: {} }}", self.description)
    }
}

/// Enum defining various boundary condition types.
///
/// # Variants
/// - `Dirichlet(f64)`: Specifies a fixed value at the boundary.
/// - `Neumann(f64)`: Specifies a fixed flux at the boundary.
/// - `Robin { alpha, beta }`: Combines Dirichlet and Neumann conditions.
/// - `Mixed { gamma, delta }`: A hybrid boundary condition.
/// - `Cauchy { lambda, mu }`: Used in fluid-structure interaction problems.
/// - `DirichletFn(FunctionWrapper)`: Functional Dirichlet condition with metadata.
/// - `NeumannFn(FunctionWrapper)`: Functional Neumann condition with metadata.
/// - `Periodic { pairs }`: Specifies a Periodic boundary condition between pairs.
///
/// # Notes
/// Functional boundary conditions allow time-dependent or spatially varying constraints.
#[derive(Clone, PartialEq, Debug)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    SolidWallInviscid,
    SolidWallViscous { normal_velocity: f64 },
    DirichletFn(FunctionWrapper),
    NeumannFn(FunctionWrapper),
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

    /// Applies boundary conditions to system matrices and RHS vectors.
    ///
    /// # Parameters
    /// - `matrix`: Mutable reference to the system matrix.
    /// - `rhs`: Mutable reference to the right-hand side vector.
    /// - `boundary_entities`: List of mesh entities to which boundary conditions are applied.
    /// - `entity_to_index`: Maps mesh entities to matrix indices.
    /// - `time`: Current simulation time for time-dependent conditions.
    ///
    /// # Computational Notes
    /// Modifies matrix coefficients and RHS values based on the type of boundary condition.
    /// Ensures consistency with finite volume and finite element methods as per Hydra's framework.
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
                    BoundaryCondition::SolidWallInviscid | BoundaryCondition::SolidWallViscous { .. } => {
                        let solid_wall_bc = SolidWallBC::new();
                        solid_wall_bc.apply_bc(matrix, rhs, entity_to_index);
                    }
                    BoundaryCondition::DirichletFn(wrapper) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = (wrapper.function)(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::NeumannFn(wrapper) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = (wrapper.function)(time, &coords);
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
            BoundaryCondition::SolidWallInviscid | BoundaryCondition::SolidWallViscous { .. } => {
                let solid_wall_bc = SolidWallBC::new();
                solid_wall_bc.apply_bc(matrix, rhs, entity_to_index);
            }
            BoundaryCondition::DirichletFn(wrapper) => {
                let coords = [0.0, 0.0, 0.0];
                let value = (wrapper.function)(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
            }
            BoundaryCondition::NeumannFn(wrapper) => {
                let coords = [0.0, 0.0, 0.0];
                let value = (wrapper.function)(time, &coords);
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

Correct the code provided below which is used for generating simple domains:

`src/interface_adapters/domain_adapter.rs`

```rust

use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};
use crate::domain::mesh::geometry_validation::GeometryValidation;
use crate::domain::mesh::reordering::cuthill_mckee;
use crate::Geometry;

/// `DomainBuilder` is a utility for constructing a mesh domain by incrementally adding vertices,
/// edges, faces, and cells. It provides methods to build the mesh and apply reordering for
/// performance optimization.
pub struct DomainBuilder {
    mesh: Mesh,
}

impl DomainBuilder {
    /// Creates a new `DomainBuilder` with an empty mesh.
    pub fn new() -> Self {
        Self {
            mesh: Mesh::new(),
        }
    }

    /// Adds a vertex to the domain with a specified ID and coordinates.
    ///
    /// # Arguments
    ///
    /// * `id` - A unique identifier for the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    pub fn add_vertex(&mut self, id: usize, coords: [f64; 3]) -> &mut Self {
        self.mesh.set_vertex_coordinates(id, coords);
        self.mesh
            .entities
            .write()
            .unwrap()
            .insert(MeshEntity::Vertex(id));
        self
    }

    /// Adds an edge connecting two vertices.
    ///
    /// # Arguments
    ///
    /// * `vertex1` - The ID of the first vertex.
    /// * `vertex2` - The ID of the second vertex.
    pub fn add_edge(&mut self, vertex1: usize, vertex2: usize) -> &mut Self {
        let edge_id = self.mesh.entities.read().unwrap().len() + 1; // Ensure unique IDs
        let edge = MeshEntity::Edge(edge_id);

        // Add relationships in the sieve
        self.mesh
            .add_arrow(MeshEntity::Vertex(vertex1), edge);
        self.mesh
            .add_arrow(MeshEntity::Vertex(vertex2), edge);
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex1));
        self.mesh.add_arrow(edge, MeshEntity::Vertex(vertex2));

        // Add the edge to the entities set
        self.mesh.entities.write().unwrap().insert(edge);
        self
    }

    /// Adds a cell with the given vertices and automatically creates faces.
    ///
    /// # Arguments
    ///
    /// * `vertex_ids` - A vector of vertex IDs that define the cell.
    pub fn add_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);
    
        let num_faces = self.mesh.count_entities(&MeshEntity::Face(0));
        let mut face_id_counter = num_faces + 1;
    
        let num_vertices = vertex_ids.len();
        for i in 0..num_vertices {
            let v1 = vertex_ids[i];
            let v2 = vertex_ids[(i + 1) % num_vertices];
            let face = MeshEntity::Face(face_id_counter);
            face_id_counter += 1;
    
            let vertex1 = MeshEntity::Vertex(v1);
            let vertex2 = MeshEntity::Vertex(v2);
    
            // Add relationships in the sieve
            self.mesh.add_arrow(face.clone(), vertex1.clone());
            self.mesh.add_arrow(face.clone(), vertex2.clone());
            
            // Optionally, add reverse relationships if needed
            self.mesh.add_arrow(vertex1, face.clone());
            self.mesh.add_arrow(vertex2, face.clone());
    
            self.mesh.entities.write().unwrap().insert(face);
    
            // Compute and store the face area in GeometryCache
            if let Some(coords1) = self.mesh.get_vertex_coordinates(v1) {
                if let Some(coords2) = self.mesh.get_vertex_coordinates(v2) {
                    let area = Geometry::compute_distance(&coords1, &coords2);
                    let geometry = Geometry::new();
                    geometry.cache.lock().unwrap().entry(face.get_id()).or_default().area = Some(area);
                }
            }
    
            self.mesh.add_arrow(cell, face);
        }
    
        for &vertex_id in &vertex_ids {
            let vertex = MeshEntity::Vertex(vertex_id);
            self.mesh.entities.write().unwrap().insert(vertex);
        }
        self.mesh.entities.write().unwrap().insert(cell);
        self
    }

    pub fn add_tetrahedron_cell(&mut self, vertex_ids: Vec<usize>) -> &mut Self {
        assert_eq!(vertex_ids.len(), 4, "Tetrahedron must have 4 vertices");
        let cell_id = self.mesh.entities.read().unwrap().len() + 1;
        let cell = MeshEntity::Cell(cell_id);

        let face_id_start = self.mesh.count_entities(&MeshEntity::Face(0)) + 1;

        let face_vertices = vec![
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[2]], // Face 1
            vec![vertex_ids[0], vertex_ids[1], vertex_ids[3]], // Face 2
            vec![vertex_ids[1], vertex_ids[2], vertex_ids[3]], // Face 3
            vec![vertex_ids[2], vertex_ids[0], vertex_ids[3]], // Face 4
        ];

        for (i, fv) in face_vertices.iter().enumerate() {
            let face_id = face_id_start + i;
            let face = MeshEntity::Face(face_id);

            // Add arrows from face to vertices
            for &vid in fv {
                let vertex = MeshEntity::Vertex(vid);
                self.mesh.add_arrow(face.clone(), vertex.clone());
                self.mesh.add_arrow(vertex.clone(), face.clone());
            }

            self.mesh.entities.write().unwrap().insert(face.clone());

            // Add arrows from cell to face
            self.mesh.add_arrow(cell.clone(), face.clone());
            self.mesh.add_arrow(face.clone(), cell.clone());
        }

        // Add arrows from cell to vertices
        for &vid in &vertex_ids {
            let vertex = MeshEntity::Vertex(vid);
            self.mesh.add_arrow(cell.clone(), vertex.clone());
            self.mesh.add_arrow(vertex.clone(), cell.clone());
        }

        self.mesh.entities.write().unwrap().insert(cell);

        self
    }

    /// Applies reordering to improve solver performance using the Cuthill-McKee algorithm.
    pub fn apply_reordering(&mut self) {
        let entities: Vec<_> = self
            .mesh
            .entities
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect();
        let adjacency: FxHashMap<_, _> = self.mesh.sieve.to_adjacency_map();
        let reordered = cuthill_mckee(&entities, &adjacency);

        // Apply the reordering
        self.mesh
            .apply_reordering(&reordered.iter().map(|e| e.get_id()).collect::<Vec<_>>());
    }

    /// Performs geometry validation to ensure mesh integrity.
    pub fn validate_geometry(&self) {
        assert!(
            GeometryValidation::test_vertex_coordinates(&self.mesh).is_ok(),
            "Geometry validation failed: Duplicate or invalid vertex coordinates."
        );
    }

    /// Finalizes and returns the built `Mesh`.
    pub fn build(self) -> Mesh {
        self.mesh
    }
}

/// Represents a domain entity with optional boundary conditions and material properties.
pub struct DomainEntity {
    pub entity: MeshEntity,
    pub boundary_conditions: Option<String>,
    pub material_properties: Option<String>,
}

impl DomainEntity {
    /// Creates a new `DomainEntity`.
    ///
    /// # Arguments
    ///
    /// * `entity` - The mesh entity to associate with this domain entity.
    pub fn new(entity: MeshEntity) -> Self {
        Self {
            entity,
            boundary_conditions: None,
            material_properties: None,
        }
    }

    /// Sets boundary conditions for the entity.
    ///
    /// # Arguments
    ///
    /// * `bc` - A string describing the boundary condition.
    pub fn set_boundary_conditions(mut self, bc: &str) -> Self {
        self.boundary_conditions = Some(bc.to_string());
        self
    }

    /// Sets material properties for the entity.
    ///
    /// # Arguments
    ///
    /// * `properties` - A string describing the material properties.
    pub fn set_material_properties(mut self, properties: &str) -> Self {
        self.material_properties = Some(properties.to_string());
        self
    }
}
```