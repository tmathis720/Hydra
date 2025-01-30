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
            // Retrieve face vertices safely
            let face_vertices = match domain.get_face_vertices(&face) {
                Ok(vertices) => vertices,
                Err(err) => {
                    log::error!("Failed to get face vertices for {:?}: {}", face, err);
                    continue;
                }
            };

            // Determine face shape
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => {
                    log::warn!(
                        "Skipping face {:?} with unsupported shape ({} vertices)",
                        face,
                        face_vertices.len()
                    );
                    continue;
                }
            };

            // Compute face centroid
            let face_center = match geometry.compute_face_centroid(face_shape, &face_vertices) {
                Ok(center) => center,
                Err(err) => {
                    log::error!(
                        "Failed to compute centroid for face {:?}: {}",
                        face,
                        err
                    );
                    continue;
                }
            };

            // Retrieve cells adjacent to the face
            let cells = match domain.get_cells_sharing_face(&face) {
                Ok(cells) if !cells.is_empty() => cells,
                Ok(_) => {
                    log::warn!("Skipping face {:?} with no adjacent cells", face);
                    continue;
                }
                Err(err) => {
                    log::error!("Failed to get cells sharing face {:?}: {}", face, err);
                    continue;
                }
            };

            // Extract first cell data
            let cell_a = match cells.iter().next() {
                Some(entry) => entry.key().clone(),
                None => {
                    log::warn!("Skipping face {:?} with missing associated cell", face);
                    continue;
                }
            };

            // Retrieve temperature and gradient at the cell
            let temp_a = match fields.get_scalar_field_value("temperature", &cell_a) {
                Some(temp) => temp,
                None => {
                    log::error!("Temperature field missing for cell {:?}", cell_a);
                    continue;
                }
            };

            let grad_temp_a = match fields.get_vector_field_value("temperature_gradient", &cell_a) {
                Some(grad) => grad,
                None => {
                    log::error!("Temperature gradient missing for cell {:?}", cell_a);
                    continue;
                }
            };

            // Compute cell centroid
            let cell_centroid = match geometry.compute_cell_centroid(domain, &cell_a) {
                Ok(centroid) => centroid,
                Err(err) => {
                    log::error!(
                        "Failed to compute centroid for cell {:?}: {}",
                        cell_a,
                        err
                    );
                    continue;
                }
            };


            // Reconstruct temperature at the face
            let reconstruction: Box<dyn ReconstructionMethod> = Box::new(LinearReconstruction);

            let face_temperature = reconstruction.reconstruct(temp_a.0, grad_temp_a.0, cell_centroid, face_center);



            // Retrieve velocity field at face
            let velocity = match fields.get_vector_field_value("velocity", &face) {
                Some(v) => v,
                None => {
                    log::error!("Velocity field missing for face {:?}", face);
                    continue;
                }
            };

            // Compute face normal
            let face_normal = match geometry.compute_face_normal(domain, &face, &cell_a) {
                Ok(normal) => normal,
                Err(err) => {
                    log::error!(
                        "Failed to compute normal for face {:?}: {}",
                        face,
                        err
                    );
                    continue;
                }
            };

            // Compute face area
            let face_area = match geometry.compute_face_area(face.get_id(), face_shape, &face_vertices) {
                Ok(area) => area,
                Err(err) => {
                    log::error!(
                        "Failed to compute area for face {:?}: {}",
                        face,
                        err
                    );
                    continue;
                }
            };

            // Compute total flux considering boundary conditions or cell-cell interactions
            let total_flux;

            if cells.len() == 1 {
                // Boundary face handling
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    match bc {
                        BoundaryCondition::Dirichlet(value) => {
                            let adjusted_face_temp = Scalar(value);
                            let temp_gradient_normal =
                                (adjusted_face_temp.0 - temp_a.0)
                                    / match Geometry::compute_distance(&cell_centroid, &face_center) {
                                        Ok(distance) => distance,
                                        Err(err) => {
                                            log::error!(
                                                "Failed to compute distance between cell {:?} and face {:?}: {}",
                                                cell_a,
                                                face,
                                                err
                                            );
                                            continue;
                                        }
                                    };

                            let conductive_flux =
                                -self.thermal_conductivity * temp_gradient_normal * face_normal.magnitude();
                            let convective_flux = face_temperature * velocity.dot(&face_normal);
                            total_flux = Scalar((conductive_flux + convective_flux) * face_area);
                        }
                        BoundaryCondition::Neumann(flux) => {
                            total_flux = Scalar(flux * face_area);
                        }
                        _ => {
                            total_flux = self.compute_flux_combined(
                                temp_a,
                                Scalar(face_temperature),
                                &grad_temp_a,
                                &face_normal,
                                &velocity,
                                face_area,
                            );
                        }
                    }
                } else {
                    total_flux = self.compute_flux_combined(
                        temp_a,
                        Scalar(face_temperature),
                        &grad_temp_a,
                        &face_normal,
                        &velocity,
                        face_area,
                    );
                }
            } else {
                // Interface face handling
                total_flux = self.compute_flux_combined(
                    temp_a,
                    Scalar(face_temperature),
                    &grad_temp_a,
                    &face_normal,
                    &velocity,
                    face_area,
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
    use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
    use crate::domain::section::{scalar::Scalar, vector::Vector3};
    use crate::equation::fields::{Fields, Fluxes};
    use crate::interface_adapters::domain_adapter::DomainBuilder;

    /// Helper function to set up a basic 3D mesh for testing using a single hexahedron cell.
    fn setup_simple_mesh() -> Mesh {
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
    

    /// Helper function to populate field data for the mesh.
    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();

        // Set temperature field for each cell
        for cell in mesh.get_cells() {
            fields.set_scalar_field_value("temperature", cell, Scalar(300.0));
            fields.set_vector_field_value("temperature_gradient", cell, Vector3([10.0, 5.0, -2.0]));
        }

        // Set velocity field for each face
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

        // The single hexahedron should have 6 faces. Let's confirm and pick the first face for Dirichlet BC.
        let all_faces: Vec<_> = mesh.get_faces().into_iter().collect();
        assert_eq!(
            all_faces.len(),
            6,
            "Expected 6 faces for a single-hex mesh, found {}",
            all_faces.len()
        );

        let dirichlet_face = all_faces[0];
        boundary_handler.set_bc(dirichlet_face, BoundaryCondition::Dirichlet(400.0))
            .expect("Failed to set Dirichlet BC.");

        // Assemble energy equation
        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Check that fluxes were computed for the face with the Dirichlet BC
        let computed_flux = fluxes.energy_fluxes.restrict(&dirichlet_face);
        assert!(
            computed_flux.is_ok(),
            "Energy flux for Dirichlet boundary face was not computed."
        );
        println!(
            "Computed energy flux for Dirichlet face {:?}: {:?}",
            dirichlet_face,
            computed_flux.unwrap().0
        );
    }

    #[test]
    fn test_energy_equation_with_neumann_bc() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new();

        // Pick the second face for the Neumann BC
        let all_faces: Vec<_> = mesh.get_faces().into_iter().collect();
        assert!(all_faces.len() >= 2, "Not enough faces to set a Neumann BC.");
        let neumann_face = all_faces[1];

        boundary_handler.set_bc(neumann_face, BoundaryCondition::Neumann(5.0))
            .expect("Failed to set Neumann BC.");

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Check that fluxes were computed for the face with the Neumann BC
        let computed_flux = fluxes.energy_fluxes.restrict(&neumann_face);
        assert!(
            computed_flux.is_ok(),
            "Energy flux for Neumann boundary face was not computed."
        );

        // For a unit cube, face area = 1.0. With Neumann(5.0), flux = 5.0 * area = 5.0
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

        // Assemble energy equation
        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Check that fluxes were computed on faces without boundary conditions.
        // In a single-hexahedron domain, all faces are physically boundary faces,
        // but we only treat them as boundary if we assigned a BC. Otherwise, we treat them as "internal" for testing.
        let all_faces: Vec<_> = mesh.get_faces().into_iter().collect();
        for face in &all_faces {
            if boundary_handler.get_bc(face).is_none() {
                let computed_flux = fluxes.energy_fluxes.restrict(face);
                assert!(
                    computed_flux.is_ok(),
                    "Energy flux for internal-like (non-BC) face {:?} was not computed.",
                    face
                );
            }
        }
    }

    #[test]
    fn test_energy_equation_scaling_with_thermal_conductivity() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);

        // Add a small normal component to velocity so conduction/convection flux is non-zero
        for face in mesh.get_faces() {
            fields.set_vector_field_value("velocity", face, Vector3([1.0, 1.0, 0.0]));
        }

        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new(); // No BCs

        // Compute fluxes once with high thermal conductivity and once with low
        let energy_eq_high_conductivity = EnergyEquation::new(1.0);
        let energy_eq_low_conductivity = EnergyEquation::new(0.1);

        // High conductivity
        energy_eq_high_conductivity.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);
        let flux_high = fluxes.energy_fluxes.all_data();

        // Clear and do low conductivity
        fluxes.energy_fluxes.clear();
        energy_eq_low_conductivity.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);
        let flux_low = fluxes.energy_fluxes.all_data();

        // Compare ratio of fluxes
        for (high, low) in flux_high.iter().zip(flux_low.iter()) {
            // If the low flux is effectively zero, skip ratio check
            if low.0.abs() < 1e-14 {
                eprintln!("Warning: low conductivity flux is near zero. Cannot check ratio.");
                continue;
            }

            let ratio = high.0 / low.0;
            assert!(
                (ratio - 10.0).abs() < 1e-6,
                "Scaling mismatch: expected ratio ~10, got {} (high={}, low={})",
                ratio,
                high.0,
                low.0
            );
        }
    }
}
