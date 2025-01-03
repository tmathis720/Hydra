use crate::equation::PhysicalEquation;
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::geometry::{Geometry, FaceShape};
use crate::domain::section::{scalar::Scalar, vector::Vector3};
use crate::Mesh;

use super::fields::{Fields, Fluxes};
use super::reconstruction::{LinearReconstruction, ReconstructionMethod};

/// Struct representing the energy equation, modeling heat transfer processes
/// in the simulation domain.
/// Provides methods for calculating conductive and convective fluxes, and
/// handles various boundary conditions.
pub struct EnergyEquation {
    /// Thermal conductivity coefficient (W/m·K), a material property.
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
    /// Constructs a new `EnergyEquation` with the specified thermal conductivity.
    ///
    /// # Parameters
    /// - `thermal_conductivity`: The material's thermal conductivity (W/m·K).
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
                _ => continue, // Unsupported face shape
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

            let reconstruction: Box<dyn ReconstructionMethod> = Box::new(LinearReconstruction);
            let face_temperature = reconstruction.reconstruct(
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

            // Compute total flux considering boundary conditions or cell-cell interactions
            let total_flux;

            if cells.len() == 1 {
                // Boundary face handling
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
                // Interface face handling
                total_flux = self.compute_flux_combined(
                    temp_a, Scalar(face_temperature), &grad_temp_a, &face_normal, &velocity, face_area,
                );
            }

            fluxes.add_energy_flux(face, total_flux);
        }
    }

    /// Computes the combined flux from conduction and convection mechanisms.
    ///
    /// # Parameters
    /// - `temp_a`: Temperature at cell center A.
    /// - `face_temperature`: Reconstructed temperature at the face.
    /// - `grad_temp_a`: Temperature gradient at cell A.
    /// - `face_normal`: Normal vector of the face.
    /// - `velocity`: Velocity vector at the face.
    /// - `face_area`: Area of the face.
    ///
    /// # Returns
    /// Combined energy flux through the face.
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
    use crate::domain::section::{scalar::Scalar, vector::Vector3};
    use crate::interface_adapters::domain_adapter::DomainBuilder;
    use crate::equation::fields::{Fields, Fluxes};
    use crate::MeshEntity;

    /// Helper function to set up a basic 3D mesh for testing using a hexahedron cell.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();

        // Add vertices for a unit cube (hexahedron):
        // Bottom face (z=0): (1)->(0,0,0), (2)->(1,0,0), (3)->(1,1,0), (4)->(0,1,0)
        // Top face (z=1): (5)->(0,0,1), (6)->(1,0,1), (7)->(1,1,1), (8)->(0,1,1)
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [1.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 1.0, 0.0])
            .add_vertex(5, [0.0, 0.0, 1.0])
            .add_vertex(6, [1.0, 0.0, 1.0])
            .add_vertex(7, [1.0, 1.0, 1.0])
            .add_vertex(8, [0.0, 1.0, 1.0]);

        // Add a hexahedron cell with the 8 vertices defined above
        builder.add_hexahedron_cell(vec![1, 2, 3, 4, 5, 6, 7, 8]);

        builder.build()
    }

    /// Helper function to populate field data for the mesh.
    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();

        // Set temperature field for cells
        for cell in mesh.get_cells() {
            fields.set_scalar_field_value("temperature", cell, Scalar(300.0));
            fields.set_vector_field_value("temperature_gradient", cell, Vector3([10.0, 5.0, -2.0]));
        }

        // Set velocity field for faces
        for face in mesh.get_faces() {
            fields.set_vector_field_value("velocity", face, Vector3([1.0, 0.0, 0.0]));
        }

        fields
    }

    #[test]
    fn test_energy_equation_with_dirichlet_bc() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new();

        // We know the hexahedron has 6 faces. Let's set a Dirichlet BC on Face(1).
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

        // Set Neumann boundary condition on another face, for example Face(2).
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

        // With Neumann(5.0), the flux should be flux * area. For a unit cube face area = 1.0.
        let flux_val = computed_flux.unwrap().0;
        assert!(
            (flux_val - 5.0).abs() < 1e-6,
            "Neumann flux mismatch. Expected ~5.0, got {}",
            flux_val
        );
    }

    #[test]
    fn test_internal_face_flux_computation() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new(); // No BCs

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Check that fluxes were computed for internal faces
        // In a single hexahedron, strictly speaking, there are no internal faces shared by two cells.
        // This test might be more meaningful if we had more than one cell, but we'll assume that the
        // code checks "internal" as those without BCs. Here, all faces belong to the single cell and are boundary faces.
        // For demonstration, let's just ensure that all faces have computed fluxes unless a BC is missing.
        for face in mesh.get_faces() {
            if boundary_handler.get_bc(&face).is_none() {
                let computed_flux = fluxes.energy_fluxes.restrict(&face);
                assert!(
                    computed_flux.is_some(),
                    "Energy flux for internal (non-BC) face {:?} was not computed.",
                    face
                );
            }
        }
    }

    #[test]
    fn test_energy_equation_scaling_with_thermal_conductivity() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);

        // Make sure we have a non-zero normal component to generate a non-zero flux.
        // For instance, adjust the velocity so it has a component normal to one of the faces:
        for face in mesh.get_faces() {
            // Add a small normal component (e.g., in the y-direction) to ensure non-zero dot product
            fields.set_vector_field_value("velocity", face, Vector3([1.0, 1.0, 0.0]));
        }

        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new(); // No boundary conditions
        
        let energy_eq_high_conductivity = EnergyEquation::new(1.0);
        let energy_eq_low_conductivity = EnergyEquation::new(0.1);
        
        // Compute fluxes with high thermal conductivity
        energy_eq_high_conductivity.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);
        let flux_high: Vec<Scalar> = fluxes.energy_fluxes.all_data();
        
        // Clear and compute fluxes with low thermal conductivity
        fluxes.energy_fluxes.clear();
        energy_eq_low_conductivity.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);
        let flux_low: Vec<Scalar> = fluxes.energy_fluxes.all_data();
        
        for (high, low) in flux_high.iter().zip(flux_low.iter()) {
            // Ensure both fluxes are non-zero to avoid division by zero
            if low.0.abs() < 1e-14 {
                // If low flux is effectively zero, print a warning and continue.
                // In a real scenario, you might want to adjust your setup to ensure non-zero flux.
                eprintln!("Warning: low conductivity flux is near zero, cannot test scaling ratio.");
                continue;
            }
            
            let ratio = high.0 / low.0;
            assert!(
                (ratio - 10.0).abs() < 1e-6,
                "Scaling mismatch: expected ratio ~10, got {} (high={}, low={})",
                ratio, high.0, low.0
            );
        }
    }

}
