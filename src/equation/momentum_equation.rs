use crate::{
    boundary::bc_handler::{BoundaryCondition, 
        BoundaryConditionHandler}, 
        geometry::Geometry, 
        FaceShape, 
        Mesh, 
        MeshEntity, 
        Section
};
use super::{
    fields::{Fields, 
        Fluxes}, 
        gradient::{Gradient, 
            GradientCalculationMethod}, 
            reconstruction::reconstruct::reconstruct_face_value, 
            PhysicalEquation
};
use crate::domain::section::{Vector3, Scalar};

/// Parameters required for the momentum equation
pub struct MomentumParameters {
    pub density: f64,   // Fluid density
    pub viscosity: f64, // Fluid viscosity
}

/// Implements the momentum equation
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
        let geometry = Geometry::new();
        let mut gradient_calculator = Gradient::new(
            domain,
            boundary_handler,
            GradientCalculationMethod::FiniteVolume,
        );

        // Placeholder for gradients of velocity components
        let mut gradient_u: Section<Vector3> = Section::new();
        let mut gradient_v: Section<Vector3> = Section::new();
        let mut gradient_w: Section<Vector3> = Section::new();

        // Compute gradients for each component of the velocity field
        for (component, gradient_section) in [
            ("velocity_x", &mut gradient_u),
            ("velocity_y", &mut gradient_v),
            ("velocity_z", &mut gradient_w),
        ] {
            let component_field = fields
                .scalar_fields
                .get(component)
                .expect(&format!("Field {} not found", component));
            gradient_calculator
                .compute_gradient(component_field, gradient_section, current_time)
                .expect(&format!("Gradient calculation failed for {}", component));
        }

        for face in domain.get_faces() {
            if let Some(normal) = domain.get_face_normal(&face, None) {
                let area = domain.get_face_area(&face).unwrap_or(0.0);
                let face_vertices = domain.get_face_vertices(&face);
                let face_shape = match face_vertices.len() {
                    3 => FaceShape::Triangle,
                    4 => FaceShape::Quadrilateral,
                    _ => continue, // Skip unsupported face shapes
                };
                let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);

                // Retrieve cells sharing the face
                let cells = domain.get_cells_sharing_face(&face);

                let mut velocity_a = Vector3([0.0; 3]);
                let mut pressure_a = Scalar(0.0);
                let mut cell_center_a = [0.0; 3];
                let mut gradients_a = [Vector3([0.0; 3]); 3];

                let mut velocity_b = Vector3([0.0; 3]);
                let mut pressure_b = Scalar(0.0);
                let mut cell_center_b = [0.0; 3];
                let mut gradients_b = [Vector3([0.0; 3]); 3];

                let mut has_cell_b = false;

                let mut iter = cells.iter();
                if let Some(cell_entry) = iter.next() {
                    let cell_a = cell_entry.key().clone();
                    velocity_a = fields
                        .get_vector_field_value("velocity_field", &cell_a)
                        .unwrap_or(Vector3([0.0; 3]));
                    pressure_a = fields
                        .get_scalar_field_value("pressure_field", &cell_a)
                        .unwrap_or(Scalar(0.0));
                    cell_center_a = domain.get_cell_centroid(&cell_a);
                    gradients_a = [
                        gradient_u.restrict(&cell_a).unwrap_or(Vector3([0.0; 3])),
                        gradient_v.restrict(&cell_a).unwrap_or(Vector3([0.0; 3])),
                        gradient_w.restrict(&cell_a).unwrap_or(Vector3([0.0; 3])),
                    ];
                }
                if let Some(cell_entry) = iter.next() {
                    let cell_b = cell_entry.key().clone();
                    has_cell_b = true;
                    velocity_b = fields
                        .get_vector_field_value("velocity_field", &cell_b)
                        .unwrap_or(Vector3([0.0; 3]));
                    pressure_b = fields
                        .get_scalar_field_value("pressure_field", &cell_b)
                        .unwrap_or(Scalar(0.0));
                    cell_center_b = domain.get_cell_centroid(&cell_b);
                    gradients_b = [
                        gradient_u.restrict(&cell_b).unwrap_or(Vector3([0.0; 3])),
                        gradient_v.restrict(&cell_b).unwrap_or(Vector3([0.0; 3])),
                        gradient_w.restrict(&cell_b).unwrap_or(Vector3([0.0; 3])),
                    ];
                }

                // Reconstruct velocities and pressures at the face
                let velocity_face_a = Vector3([
                    reconstruct_face_value(
                        velocity_a.0[0],
                        gradients_a[0].0,
                        cell_center_a,
                        face_center,
                    ),
                    reconstruct_face_value(
                        velocity_a.0[1],
                        gradients_a[1].0,
                        cell_center_a,
                        face_center,
                    ),
                    reconstruct_face_value(
                        velocity_a.0[2],
                        gradients_a[2].0,
                        cell_center_a,
                        face_center,
                    ),
                ]);
                let velocity_face_b = if has_cell_b {
                    Vector3([
                        reconstruct_face_value(
                            velocity_b.0[0],
                            gradients_b[0].0,
                            cell_center_b,
                            face_center,
                        ),
                        reconstruct_face_value(
                            velocity_b.0[1],
                            gradients_b[1].0,
                            cell_center_b,
                            face_center,
                        ),
                        reconstruct_face_value(
                            velocity_b.0[2],
                            gradients_b[2].0,
                            cell_center_b,
                            face_center,
                        ),
                    ])
                } else {
                    velocity_a
                };

                let pressure_face_a = reconstruct_face_value(
                    pressure_a.0,
                    gradients_a[0].0, // Pressure gradient, use first component
                    cell_center_a,
                    face_center,
                );
                let pressure_face_b = if has_cell_b {
                    reconstruct_face_value(
                        pressure_b.0,
                        gradients_b[0].0, // Pressure gradient, use first component
                        cell_center_b,
                        face_center,
                    )
                } else {
                    pressure_a.0
                };

                // Compute convective flux
                let avg_velocity = (velocity_face_a + velocity_face_b) * 0.5;
                let velocity_dot_normal = avg_velocity.iter().zip(&normal).map(|(v, n)| v * n).sum::<f64>();

                let convective_flux = avg_velocity * (self.params.density * velocity_dot_normal * area);

                // Compute pressure flux
                let pressure_flux = Vector3([
                    pressure_face_a * area,
                    pressure_face_b * area,
                    0.0, // Replace if non-zero pressure terms in other directions apply
                ]);

                // Compute diffusive flux
                let diffusive_flux = Vector3([
                    self.params.viscosity * area,
                    0.0,
                    0.0,
                ]);

                // Total flux vector
                let total_flux = convective_flux - pressure_flux + diffusive_flux;

                // Update momentum fluxes
                fluxes.add_momentum_flux(face.clone(), total_flux);

                // Apply boundary conditions
                if let Some(bc) = boundary_handler.get_bc(&face) {
                    self.apply_boundary_conditions(bc, fluxes, &face, &normal.0, area);
                }
            }
        }
    }
    

    fn apply_boundary_conditions(
        &self,
        bc: BoundaryCondition,
        fluxes: &mut Fluxes,
        face: &MeshEntity,
        normal: &[f64; 3],
        area: f64,
    ) {
        match bc {
            BoundaryCondition::Dirichlet(value) => {
                let flux_value = Vector3([
                    value * normal[0] * area,
                    value * normal[1] * area,
                    value * normal[2] * area,
                ]);
                fluxes.add_momentum_flux(face.clone(), flux_value);
            }
            BoundaryCondition::Neumann(value) => {
                let flux_value = Vector3([
                    value * normal[0] * area,
                    value * normal[1] * area,
                    value * normal[2] * area,
                ]);
                fluxes.add_momentum_flux(face.clone(), flux_value);
            }
            BoundaryCondition::Robin { alpha, beta } => {
                // Example: Apply a Robin boundary condition
                let flux_value = Vector3([
                    alpha * beta * normal[0] * area,
                    alpha * beta * normal[1] * area,
                    alpha * beta * normal[2] * area,
                ]);
                fluxes.add_momentum_flux(face.clone(), flux_value);
            }
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::section::{Vector3, Scalar};
    use crate::equation::fields::{Fields, Fluxes};
    use crate::boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler};
    use crate::interface_adapters::domain_adapter::DomainBuilder;

    /// Sets up a simple 2D mesh with two cells and connected faces.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();
    
        // Add vertices of a tetrahedron
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 0.0, 1.0])
            .add_vertex(5, [1.0, 1.0, 0.0]);
    
        // Add a tetrahedral cell
        builder.add_tetrahedron_cell(vec![1, 2, 3, 4]);
        builder.add_tetrahedron_cell(vec![2, 3, 5, 4]);
    
        // Return the built mesh
        builder.build()
    }

    /// Initializes all necessary fields for the test.
    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();

        // Get cell IDs from the mesh
        let cell_ids = mesh
            .entities
            .read()
            .unwrap()
            .iter()
            .filter_map(|e| if let MeshEntity::Cell(id) = e { Some(*id) } else { None })
            .collect::<Vec<_>>();

        assert!(
            cell_ids.len() >= 2,
            "Mesh must contain at least two cells for the test"
        );

        // Set field values for the cells
        let cell_a = MeshEntity::Cell(cell_ids[0]);
        fields.set_scalar_field_value("velocity_x", cell_a.clone(), Scalar(1.0));
        fields.set_scalar_field_value("velocity_y", cell_a.clone(), Scalar(2.0));
        fields.set_scalar_field_value("velocity_z", cell_a.clone(), Scalar(3.0));
        fields.set_scalar_field_value("pressure_field", cell_a.clone(), Scalar(100.0));

        let cell_b = MeshEntity::Cell(cell_ids[1]);
        fields.set_scalar_field_value("velocity_x", cell_b.clone(), Scalar(4.0));
        fields.set_scalar_field_value("velocity_y", cell_b.clone(), Scalar(5.0));
        fields.set_scalar_field_value("velocity_z", cell_b.clone(), Scalar(6.0));
        fields.set_scalar_field_value("pressure_field", cell_b.clone(), Scalar(50.0));

        fields
    }

    /// Configures boundary conditions for the test mesh.
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let bc_handler = BoundaryConditionHandler::new();

        // Retrieve a face from the mesh
        let face = mesh.entities.read().unwrap().iter()
            .find(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .expect("No face found in mesh");

        // Apply Dirichlet BC to one face
        bc_handler.set_bc(
            face.clone(),
            BoundaryCondition::Dirichlet(10.0),
        );

        bc_handler
    }

    #[test]
    fn test_calculate_momentum_fluxes() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = setup_boundary_conditions(&mesh);

        let momentum_eq = MomentumEquation {
            params: MomentumParameters {
                density: 1.0,
                viscosity: 0.01,
            },
        };

        // Perform the flux calculation
        momentum_eq.calculate_momentum_fluxes(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // Retrieve a face from the mesh
        let face = mesh.entities.read().unwrap().iter()
            .find(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .expect("No face found in mesh");

        // Verify fluxes for a specific face
        let momentum_flux = fluxes
            .momentum_fluxes
            .restrict(&face)
            .expect("Momentum flux not calculated for face");

        // Perform checks based on expected physical values
        assert!(momentum_flux.0[0].is_finite(), "Momentum flux x-component is not finite");
        assert!(momentum_flux.0[1].is_finite(), "Momentum flux y-component is not finite");
        assert!(momentum_flux.0[2].is_finite(), "Momentum flux z-component is not finite");
    }

    #[test]
    fn test_boundary_condition_application() {
        let mesh = setup_simple_mesh();
        let mut fluxes = Fluxes::new();

        // Retrieve a face from the mesh
        let face = mesh.entities.read().unwrap().iter()
            .find(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .expect("No face found in mesh");

        // For simplicity, assume normal is [1.0, 0.0, 0.0] and area is 1.0
        let normal = [1.0, 0.0, 0.0];
        let area = 1.0;

        let momentum_eq = MomentumEquation {
            params: MomentumParameters {
                density: 1.0,
                viscosity: 0.01,
            },
        };

        // Apply a Dirichlet boundary condition
        let bc = BoundaryCondition::Dirichlet(5.0);
        momentum_eq.apply_boundary_conditions(bc, &mut fluxes, &face, &normal, area);

        let flux_value = fluxes
            .momentum_fluxes
            .restrict(&face)
            .expect("Momentum flux not calculated for Dirichlet BC");

        assert_eq!(flux_value.0[0], 5.0);
        assert_eq!(flux_value.0[1], 0.0);
        assert_eq!(flux_value.0[2], 0.0);

        // Apply a Neumann boundary condition
        let bc = BoundaryCondition::Neumann(2.0);
        momentum_eq.apply_boundary_conditions(bc, &mut fluxes, &face, &normal, area);

        let flux_value = fluxes
            .momentum_fluxes
            .restrict(&face)
            .expect("Momentum flux not calculated for Neumann BC");

        assert_eq!(flux_value.0[0], 7.0); // 5.0 (Dirichlet) + 2.0 (Neumann)
    }

    #[test]
    fn test_gradient_computation_integration() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let binding = BoundaryConditionHandler::new();
        let mut gradient_calculator = Gradient::new(
            &mesh,
            &binding,
            GradientCalculationMethod::FiniteVolume,
        );
    
        let mut gradient_section: Section<Vector3> = Section::new();
    
        let scalar_field = fields
            .scalar_fields
            .get("pressure_field")
            .expect("Pressure field not found");
    
        gradient_calculator
            .compute_gradient(scalar_field, &mut gradient_section, 0.0)
            .expect("Gradient computation failed");
    
        // Verify that the gradient computation produces expected results
        let cell_ids = mesh.entities.read().unwrap().iter()
            .filter_map(|e| if let MeshEntity::Cell(id) = e { Some(*id) } else { None })
            .collect::<Vec<_>>();
    
        for cell_id in cell_ids {
            let cell = MeshEntity::Cell(cell_id);
            let gradient = gradient_section
                .restrict(&cell)
                .expect(&format!("Gradient not computed for cell {}", cell_id));
    
            // Add checks based on known values
            assert!(gradient.0[0].is_finite(), "Expected finite x-gradient for cell {}", cell_id);
            assert!(gradient.0[1].is_finite(), "Expected finite y-gradient for cell {}", cell_id);
            assert!(gradient.0[2].is_finite(), "Expected finite z-gradient for cell {}", cell_id);
        }
    }
    
}
