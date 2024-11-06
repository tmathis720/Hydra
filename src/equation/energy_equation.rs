use crate::equation::reconstruction::reconstruct::reconstruct_face_value;
use crate::equation::PhysicalEquation;
use crate::domain::{mesh::Mesh, Section};
use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
use crate::geometry::{Geometry, FaceShape};

use super::fields::{Fields, Fluxes};

pub struct EnergyEquation {
    pub thermal_conductivity: f64, // Coefficient for thermal conduction
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

        // Iterate over all faces in the mesh
        for face in domain.get_faces() {
            // Obtain cell data to reconstruct face-centered values
            let face_vertices = domain.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
            };
            let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

            // Retrieve the cells sharing the face
            let cells = domain.get_cells_sharing_face(&face);
            let cell_a = cells.iter().next().map(|entry| entry.key().clone()).expect("Face should have at least one associated cell.");

            // Temperature and gradient at cell center
            let temp_a = temperature_field.restrict(&cell_a).expect("Temperature not found for cell");
            let grad_temp_a = temperature_gradient.restrict(&cell_a).expect("Temperature gradient not found for cell");

            // Reconstruct temperature at the face center
            let face_temperature = reconstruct_face_value(
                temp_a,
                grad_temp_a,
                geometry.compute_cell_centroid(domain, &cell_a),
                face_center,
            );

            // Calculate face-centered velocity
            let velocity = velocity_field.restrict(&face).expect("Velocity not found at face");

            // Calculate conductive flux: -κ * ∇T · n
            let face_normal = geometry.compute_face_normal(domain, &face, &cell_a).expect("Normal not found for face");
            let conductive_flux = -self.thermal_conductivity * (
                grad_temp_a[0] * face_normal[0] +
                grad_temp_a[1] * face_normal[1] +
                grad_temp_a[2] * face_normal[2]
            );

            // Calculate convective flux: ρ * u * T * (u · n)
            let rho = 1.0; // Assuming a constant density
            let convective_flux = rho * face_temperature * (
                velocity[0] * face_normal[0] +
                velocity[1] * face_normal[1] +
                velocity[2] * face_normal[2]
            );

            // Total flux on the face
            let face_area = geometry.compute_face_area(face.get_id(), face_shape, &face_vertices);
            let mut total_flux = (conductive_flux + convective_flux) * face_area;

            // Apply boundary conditions if the face is at a boundary
            if cells.len() == 1 { // Indicates a boundary face
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    total_flux = match bc {
                        BoundaryCondition::Dirichlet(value) => value,
                        BoundaryCondition::Neumann(flux) => total_flux + flux,
                        BoundaryCondition::Robin { alpha, beta } => {
                            alpha * face_temperature + beta * total_flux
                        }
                        BoundaryCondition::Mixed { gamma, delta } => {
                            gamma * total_flux + delta
                        }
                        BoundaryCondition::Cauchy { lambda, mu } => {
                            lambda * total_flux + mu * face_temperature
                        }
                        _ => total_flux, // Default to total_flux if no special condition
                    };
                }
            }

            // Store the computed flux in the energy flux section
            energy_fluxes.set_data(face, total_flux);
        }
    }
}


impl PhysicalEquation for EnergyEquation {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
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

        let calculated_flux = energy_fluxes.restrict(&face).expect("Flux not calculated.");
        assert_eq!(calculated_flux, 100.0, "Flux should match Dirichlet boundary value.");
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

        let calculated_flux = energy_fluxes.restrict(&face).expect("Flux not calculated.");
        assert!(calculated_flux > 0.0, "Flux should be adjusted by Neumann boundary.");
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
        };
        
        let mut fluxes = Fluxes {
            energy_fluxes: Section::new(),
            momentum_fluxes: Section::new(),
            turbulence_fluxes: Section::new(),
        };

        let cell = MeshEntity::Cell(1);
        let face = MeshEntity::Face(1);

        fields.temperature_field.set_data(cell, 300.0);
        fields.temperature_gradient.set_data(cell, [10.0, 0.0, 0.0]);
        fields.velocity_field.set_data(face, [2.0, 0.0, 0.0]);

        let energy_eq = EnergyEquation::new(0.5);
        energy_eq.assemble(&mesh, &fields, &mut fluxes, &boundary_handler);

        assert!(fluxes.energy_fluxes.restrict(&face).is_some(), "Energy fluxes should be computed and stored.");
    }
}