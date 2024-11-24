use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::domain::section::{Vector3, Scalar};

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
    
        // Map boundary entities to indices
        for (i, entity) in boundary_entities.iter().enumerate() {
            entity_to_index.insert(entity.clone(), i);
        }
    
        for face in domain.get_faces() {
            println!("Processing face: {:?}", face);
    
            let normal = match domain.get_face_normal(&face, None) {
                Some(normal) => normal,
                None => {
                    println!("Face {:?} has no normal! Skipping.", face);
                    continue; // Skip faces with no normal
                }
            };
    
            let area = match domain.get_face_area(&face) {
                Some(area) => area,
                None => {
                    println!("Face {:?} has no area! Skipping.", face);
                    continue; // Skip faces with no area
                }
            };
    
            if let Some(velocity) = velocity_field.restrict(&face) {
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
    
            if let Some(bc) = boundary_handler.get_bc(&face) {
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        fluxes.set_data(face.clone(), Vector3([value, 0.0, 0.0]));
                    }
                    BoundaryCondition::Neumann(flux_value) => {
                        let existing_flux = fluxes.restrict(&face).unwrap_or(Vector3([0.0, 0.0, 0.0]));
                        let updated_flux = Vector3([
                            existing_flux[0] + flux_value,
                            existing_flux[1],
                            existing_flux[2],
                        ]);
                        fluxes.set_data(face.clone(), updated_flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let existing_flux = fluxes.restrict(&face).unwrap_or(Vector3([0.0, 0.0, 0.0]));
                        let updated_flux = Vector3([
                            existing_flux[0] * alpha + beta,
                            existing_flux[1] * alpha,
                            existing_flux[2] * alpha,
                        ]);
                        fluxes.set_data(face.clone(), updated_flux);
                    }
                    _ => println!("Unsupported boundary condition for face {:?}: {:?}", face, bc),
                }
            } else {
                println!("No boundary condition for face {:?}", face);
                continue; // Skip faces with no boundary conditions
            }
        }
    
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
