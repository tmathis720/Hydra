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
        // Prepare entity-to-index mapping for all boundary entities
        let entity_to_index = domain.get_entity_to_index();
        let boundary_entities = boundary_handler.get_boundary_faces();

        for (i, entity) in boundary_entities.iter().enumerate() {
            entity_to_index.insert(entity.clone(), i);
        }

        // Iterate over all faces in the domain
        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);

                // Calculate velocity dot normal
                let velocity_dot_normal = velocity_field
                    .restrict(&face)
                    .map(|vel| vel.iter().zip(normal.iter()).map(|(v, n)| v * n).sum::<f64>())
                    .unwrap_or(0.0);

                // Calculate flux vector
                let flux = Vector3([
                    velocity_dot_normal * area,
                    velocity_dot_normal * normal[1] * area,
                    velocity_dot_normal * normal[2] * area,
                ]);
                fluxes.set_data(face.clone(), flux);

                // Handle boundary conditions for the face
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
                        BoundaryCondition::Mixed { gamma, delta } => {
                            let existing_flux = fluxes
                                .restrict(&face)
                                .unwrap_or(Vector3([0.0, 0.0, 0.0]));
                            let updated_flux = Vector3([
                                existing_flux[0] * gamma + delta,
                                existing_flux[1] * gamma,
                                existing_flux[2] * gamma,
                            ]);
                            fluxes.set_data(face.clone(), updated_flux);
                        }
                        BoundaryCondition::Cauchy { lambda, mu } => {
                            let existing_flux = fluxes
                                .restrict(&face)
                                .unwrap_or(Vector3([0.0, 0.0, 0.0]));
                            let updated_flux = Vector3([
                                lambda * existing_flux[0] + mu,
                                lambda * existing_flux[1],
                                lambda * existing_flux[2],
                            ]);
                            fluxes.set_data(face.clone(), updated_flux);
                        }
                        BoundaryCondition::DirichletFn(_) | BoundaryCondition::NeumannFn(_) => {
                            // Placeholder for functional boundary conditions
                        }
                    }
                }
            }
        }

        // Apply global boundary conditions
        let mut matrix_storage = faer::Mat::<f64>::zeros(1, 1);
        let mut rhs_storage = faer::Mat::<f64>::zeros(1, 1);
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

        let velocity_field = Section::<Vector3>::new();
        let pressure_field = Section::<Scalar>::new();
        let mut fluxes = Section::<Vector3>::new();

        let boundary_handler = BoundaryConditionHandler::new();
        boundary_handler.set_bc(MeshEntity::Face(1), BoundaryCondition::Dirichlet(5.0));
        boundary_handler.set_bc(MeshEntity::Face(2), BoundaryCondition::Neumann(1.0));

        // Ensure entity-to-index mapping is initialized
        let entity_to_index = mesh.get_entity_to_index();
        let boundary_faces = boundary_handler.get_boundary_faces();
        for (i, entity) in boundary_faces.iter().enumerate() {
            entity_to_index.insert(entity.clone(), i);
        }

        let equation = Equation {};
        equation.calculate_fluxes(
            &mesh,
            &velocity_field,
            &pressure_field,
            &mut fluxes,
            &boundary_handler,
            0.0,
        );

        assert_eq!(
            fluxes.restrict(&MeshEntity::Face(1)).unwrap(),
            Vector3([5.0, 0.0, 0.0])
        );

        let face2_flux = fluxes.restrict(&MeshEntity::Face(2)).unwrap();
        assert_eq!(face2_flux, Vector3([1.0, 0.0, 0.0])); // Neumann adds flux value
    }

}
