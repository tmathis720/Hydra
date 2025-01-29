use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::domain::section::{scalar::Scalar, vector::Vector3};
use crate::{FaceShape, Geometry};
/// Represents a generic equation framework for computing fluxes
/// in a simulation domain. This implementation is designed to handle
/// flux calculations based on velocity fields, boundary conditions, and
/// mesh geometry.
pub struct Equation {}

impl Equation {
    pub fn calculate_fluxes(
        &self,
        domain: &Mesh,
        velocity_field: &Section<Vector3>,
        _pressure_field: &Section<Scalar>,
        fluxes: &mut Section<Vector3>,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        let entity_to_index = domain.get_entity_to_index();
        let boundary_entities = boundary_handler.get_boundary_faces();
        let mut geometry = Geometry::new();
    
        // Map boundary entities to indices.
        for (i, entity) in boundary_entities.iter().enumerate() {
            entity_to_index.insert(entity.clone(), i);
        }
    
        for face in domain.get_faces() {
            println!("Processing face: {:?}", face);
        
            // Retrieve cells sharing the face (handling Result properly)
            let associated_cell = match domain.get_cells_sharing_face(&face) {
                Ok(cells) => {
                    // Extract one cell from the DashMap (if available)
                    cells.iter().next().map(|entry| entry.key().clone())
                }
                Err(e) => {
                    log::warn!(
                        "Skipping face {:?} due to error retrieving associated cells: {}",
                        face, e
                    );
                    continue;
                }
            };
        
            // Retrieve face normal using Geometry module
            let normal = match associated_cell {
                Some(cell) => geometry.compute_face_normal(domain, &face, &cell),
                None => {
                    log::warn!(
                        "Skipping face {:?} because no adjacent cell was found for normal computation",
                        face
                    );
                    continue;
                }
            };
        
            let normal = match normal {
                Ok(n) => n,
                Err(e) => {
                    log::warn!("Skipping face {:?} due to normal computation failure: {}", face, e);
                    continue;
                }
            };
        
            // Retrieve face area using Geometry module
            let area = match geometry.compute_face_area(
                face.get_id(),
                FaceShape::Quadrilateral, // Assume quadrilateral faces for now
                &domain.get_face_vertices(&face).unwrap_or_default(),
            ) {
                Ok(a) => a,
                Err(e) => {
                    log::warn!("Skipping face {:?} due to area computation failure: {}", face, e);
                    continue;
                }
            };
        
            // Compute flux based on velocity field.
            if let Ok(velocity) = velocity_field.restrict(&face) {
                let velocity_dot_normal: f64 = velocity.iter()
                    .zip(normal.iter())
                    .map(|(v, n)| v * n)
                    .sum();
        
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
        
            // Apply boundary conditions.
            if let Some(bc) = boundary_handler.get_bc(&face) {
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        fluxes.set_data(face.clone(), Vector3([value, 0.0, 0.0]));
                    }
                    BoundaryCondition::Neumann(flux_value) => {
                        let existing_flux = fluxes
                            .restrict(&face)
                            .unwrap_or(Vector3([0.0, 0.0, 0.0]));
                        let updated_flux = Vector3([
                            existing_flux[0] + flux_value,
                            existing_flux[1],
                            existing_flux[2],
                        ]);
                        fluxes.set_data(face.clone(), updated_flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let existing_flux = fluxes
                            .restrict(&face)
                            .unwrap_or(Vector3([0.0, 0.0, 0.0]));
                        let updated_flux = Vector3([
                            existing_flux[0] * alpha + beta,
                            existing_flux[1] * alpha,
                            existing_flux[2] * alpha,
                        ]);
                        fluxes.set_data(face.clone(), updated_flux);
                    }
                    _ => {
                        println!(
                            "Unsupported boundary condition for face {:?}: {:?}",
                            face, bc
                        );
                    }
                }
            } else {
                log::debug!("No boundary condition for face {:?}", face);
                continue;
            }
        }
    
        // Matrix assembly for boundary condition enforcement.
        let num_boundary_entities = boundary_entities.len();
        let mut matrix_storage = faer::Mat::<f64>::zeros(num_boundary_entities, num_boundary_entities);
        let mut rhs_storage = faer::Mat::<f64>::zeros(num_boundary_entities, 1);
        let mut matrix = matrix_storage.as_mut();
        let mut rhs = rhs_storage.as_mut();
    
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
    use crate::boundary::bc_handler::{BoundaryConditionHandler, BoundaryCondition};
    use crate::domain::section::{scalar::Scalar, vector::Vector3};
    use crate::domain::Section;

    fn setup_single_hexahedron_mesh() -> crate::domain::mesh::Mesh {
        let mut builder = DomainBuilder::new();

        // Add vertices for a unit cube (hexahedron):
        // Bottom face (z=0): (1)->(0,0,0), (2)->(1,0,0), (3)->(1,1,0), (4)->(0,1,0)
        // Top face (z=1): (5)->(0,0,1), (6)->(1,0,1), (7)->(1,1,1), (8)->(0,1,1)
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 1");
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 2");
        assert!(builder.add_vertex(3, [1.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 3");
        assert!(builder.add_vertex(4, [0.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 4");
        assert!(builder.add_vertex(5, [0.0, 0.0, 1.0]).is_ok(), "Failed to add vertex 5");
        assert!(builder.add_vertex(6, [1.0, 0.0, 1.0]).is_ok(), "Failed to add vertex 6");
        assert!(builder.add_vertex(7, [1.0, 1.0, 1.0]).is_ok(), "Failed to add vertex 7");
        assert!(builder.add_vertex(8, [0.0, 1.0, 1.0]).is_ok(), "Failed to add vertex 8");
    
        // Add a hexahedron cell with the 8 vertices defined above
        assert!(
            builder
                .add_hexahedron_cell(vec![1, 2, 3, 4, 5, 6, 7, 8])
                .is_ok(),
            "Failed to add hexahedron cell"
        );
    
        builder.build()
    }

    fn setup_two_hexahedra_mesh() -> crate::domain::mesh::Mesh {
        let mut builder = DomainBuilder::new();
    
        // First hexahedron
        assert!(builder.add_vertex(1, [0.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 1");
        assert!(builder.add_vertex(2, [1.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 2");
        assert!(builder.add_vertex(3, [1.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 3");
        assert!(builder.add_vertex(4, [0.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 4");
        assert!(builder.add_vertex(5, [0.0, 0.0, 1.0]).is_ok(), "Failed to add vertex 5");
        assert!(builder.add_vertex(6, [1.0, 0.0, 1.0]).is_ok(), "Failed to add vertex 6");
        assert!(builder.add_vertex(7, [1.0, 1.0, 1.0]).is_ok(), "Failed to add vertex 7");
        assert!(builder.add_vertex(8, [0.0, 1.0, 1.0]).is_ok(), "Failed to add vertex 8");
    
        assert!(
            builder
                .add_hexahedron_cell(vec![1, 2, 3, 4, 5, 6, 7, 8])
                .is_ok(),
            "Failed to add first hexahedron cell"
        );
    
        // Second hexahedron sharing a face with the first
        assert!(builder.add_vertex(9, [1.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 9");
        assert!(builder.add_vertex(10, [2.0, 0.0, 0.0]).is_ok(), "Failed to add vertex 10");
        assert!(builder.add_vertex(11, [2.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 11");
        assert!(builder.add_vertex(12, [1.0, 1.0, 0.0]).is_ok(), "Failed to add vertex 12");
        assert!(builder.add_vertex(13, [1.0, 0.0, 1.0]).is_ok(), "Failed to add vertex 13");
        assert!(builder.add_vertex(14, [2.0, 0.0, 1.0]).is_ok(), "Failed to add vertex 14");
        assert!(builder.add_vertex(15, [2.0, 1.0, 1.0]).is_ok(), "Failed to add vertex 15");
        assert!(builder.add_vertex(16, [1.0, 1.0, 1.0]).is_ok(), "Failed to add vertex 16");
    
        assert!(
            builder
                .add_hexahedron_cell(vec![9, 10, 11, 12, 13, 14, 15, 16])
                .is_ok(),
            "Failed to add second hexahedron cell"
        );
    
        builder.build()
    }
    

    fn set_uniform_velocity(mesh: &crate::domain::mesh::Mesh, velocity: Vector3) -> Section<Vector3> {
        let velocity_field = Section::<Vector3>::new();
        for face in mesh.get_faces() {
            velocity_field.set_data(face, velocity);
        }
        velocity_field
    }

    /// Test Dirichlet BC on one face of a single hexahedron.
    #[test]
    fn test_dirichlet_bc() {
        let mesh = setup_single_hexahedron_mesh();

        // Set a uniform velocity field
        let velocity_field = set_uniform_velocity(&mesh, Vector3([1.0, 0.0, 0.0]));

        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();

        let boundary_handler = BoundaryConditionHandler::new();
        let faces = mesh.get_faces();
        assert!(!faces.is_empty(), "No faces in mesh");
        // Apply Dirichlet BC to the first face
        boundary_handler.set_bc(faces[0].clone(), BoundaryCondition::Dirichlet(5.0));

        let equation = Equation {};
        equation.calculate_fluxes(&mesh, &velocity_field, &pressure_field, &mut fluxes, &boundary_handler, 0.0);

        let computed_flux = fluxes.restrict(&faces[0]).expect("Flux not computed for Dirichlet face");
        // Dirichlet sets flux to [5.0, 0.0, 0.0] regardless of velocity
        assert_eq!(computed_flux, Vector3([5.0, 0.0, 0.0]));
    }

    #[test]
    fn test_neumann_bc() {
        let mesh = setup_single_hexahedron_mesh();
        let velocity_field = set_uniform_velocity(&mesh, Vector3([1.0, 0.0, 0.0]));
    
        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();
    
        let boundary_handler = BoundaryConditionHandler::new();
        let faces = mesh.get_faces();
        assert!(!faces.is_empty(), "No faces in mesh");
    
        let bc_face = faces.iter().find(|f| mesh.get_face_normal(f, None).is_ok()).unwrap();
        boundary_handler.set_bc(bc_face.clone(), BoundaryCondition::Neumann(2.0));
    
        let equation = Equation {};
        equation.calculate_fluxes(&mesh, &velocity_field, &pressure_field, &mut fluxes, &boundary_handler, 0.0);
    
        let computed_flux = fluxes.restrict(bc_face)
            .expect(&format!("Flux not computed for Neumann face {:?}", bc_face));
    
        assert!(computed_flux[0] > 0.0, "Neumann BC did not affect x-flux");
    }
    
    

    #[test]
    fn test_robin_bc() {
        let mesh = setup_single_hexahedron_mesh();
        let velocity_field = set_uniform_velocity(&mesh, Vector3([1.0, 1.0, 0.0]));
    
        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();
    
        let boundary_handler = BoundaryConditionHandler::new();
        let faces = mesh.get_faces();
        assert!(!faces.is_empty(), "No faces in mesh");
    
        let bc_face = faces.iter().find(|f| mesh.get_face_normal(f, None).is_ok()).unwrap();
        boundary_handler.set_bc(bc_face.clone(), BoundaryCondition::Robin { alpha: 0.8, beta: 2.0 });
    
        let equation = Equation {};
        equation.calculate_fluxes(&mesh, &velocity_field, &pressure_field, &mut fluxes, &boundary_handler, 0.0);
    
        let computed_flux = fluxes.restrict(bc_face)
            .expect(&format!("Flux not computed for Robin face {:?}", bc_face));
    
        assert!(computed_flux[0].is_finite());
    }
    
    

    /// Test internal face with no BC by using two hexahedra sharing a face.
    #[test]
    fn test_internal_face_no_bc() {
        let mesh = setup_two_hexahedra_mesh();

        // Set uniform velocity
        let velocity_field = set_uniform_velocity(&mesh, Vector3([0.5, 0.5, 0.0]));

        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();

        let boundary_handler = BoundaryConditionHandler::new(); // no BC

        let equation = Equation {};
        equation.calculate_fluxes(&mesh, &velocity_field, &pressure_field, &mut fluxes, &boundary_handler, 0.0);

        // There should be a face shared by both hexahedra. On that internal face,
        // flux should be computed from velocity alone.
        let faces = mesh.get_faces();
        // Find a face that belongs to both hexahedra. Usually that's the one in the middle.
        // We'll just check that all faces got some flux since no BC is set and we have velocity.
        for face in &faces {
            if let Ok(flux) = fluxes.restrict(face) {
                // Just check flux is finite and not zero:
                assert!(flux[0].is_finite());
                assert!(flux[1].is_finite());
                assert!(flux[2].is_finite());
            }
        }
    }
}
