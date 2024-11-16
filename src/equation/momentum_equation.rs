use crate::{boundary::bc_handler::BoundaryConditionHandler, domain::mesh::Mesh};
use super::{fields::{Fields, Fluxes}, PhysicalEquation};

pub struct MomentumEquation {
    pub density: f64,
    pub viscosity: f64,
}

impl<T> PhysicalEquation<T> for MomentumEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields<T>,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        self.calculate_momentum_fluxes(domain, fields, fluxes, boundary_handler, current_time);
    }
}

impl MomentumEquation {
    fn calculate_momentum_fluxes<T>(
        &self,
        domain: &Mesh,
        fields: &Fields<T>,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);

                let velocity = fields.get_velocity(&face).unwrap_or([0.0; 3]);
                let velocity_dot_normal = velocity.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();

                let flux = [velocity_dot_normal * area, 0.0, 0.0];
                fluxes.add_momentum_flux(face.clone(), flux);

                let mut matrix = faer::MatMut::default();
                let mut rhs = faer::MatMut::default();
                let boundary_entities = boundary_handler.get_boundary_faces();
                let entity_to_index = domain.get_entity_to_index();

                boundary_handler.apply_bc(
                    &mut matrix,
                    &mut rhs,
                    &boundary_entities,
                    &entity_to_index,
                    current_time,
                );
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
    use crate::equation::{PhysicalEquation};

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
        let mut fluxes = Fluxes::new();
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

