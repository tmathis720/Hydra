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
