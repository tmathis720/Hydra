use crate::{
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
    geometry::{Geometry, FaceShape},
    domain::section::{Scalar, Vector3},
    Mesh, MeshEntity, Section,
};
use super::{
    fields::{Fields, Fluxes},
    gradient::{Gradient, GradientCalculationMethod},
    reconstruction::reconstruct::reconstruct_face_value,
    PhysicalEquation,
};

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
    /// Main driver for computing momentum fluxes across all faces.
    pub fn calculate_momentum_fluxes(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        let geometry = Geometry::new();

        let (gradient_u, gradient_v, gradient_w) =
            self.compute_velocity_gradients(domain, fields, boundary_handler, current_time);

        for face in domain.get_faces() {
            let face_vertices = domain.get_face_vertices(&face);
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue, // Skip unsupported face shapes
            };

            let normal = match domain.get_face_normal(&face, None) {
                Some(n) => n,
                None => continue,
            };

            let area = match domain.get_face_area(&face) {
                Some(a) => a,
                None => continue,
            };

            let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);
            let cells = domain.get_cells_sharing_face(&face);

            // Extract cell data
            let (_cell_a, velocity_a, pressure_a, center_a, grads_a) =
                self.extract_cell_data(domain, fields, &gradient_u, &gradient_v, &gradient_w, &cells, 0);
            let (has_cell_b, velocity_b, pressure_b, center_b, grads_b) =
                if cells.len() > 1 {
                    let (_cell_b, vb, pb, cb, gb) =
                        self.extract_cell_data(domain, fields, &gradient_u, &gradient_v, &gradient_w, &cells, 1);
                    (true, vb, pb, cb, gb)
                } else {
                    (false, velocity_a, pressure_a, center_a, grads_a)
                };

            // Reconstruct fields at face
            let velocity_face_a = self.reconstruct_face_velocity(velocity_a, &grads_a, center_a, face_center);
            let velocity_face_b = if has_cell_b {
                self.reconstruct_face_velocity(velocity_b, &grads_b, center_b, face_center)
            } else {
                velocity_face_a
            };

            let pressure_face_a = self.reconstruct_face_pressure(pressure_a, grads_a[0], center_a, face_center);
            let pressure_face_b = if has_cell_b {
                self.reconstruct_face_pressure(pressure_b, grads_b[0], center_b, face_center)
            } else {
                pressure_face_a
            };

            // Compute fluxes
            let convective_flux = self.compute_convective_flux(velocity_face_a, velocity_face_b, &normal, area);
            let pressure_flux = self.compute_pressure_flux(pressure_face_a, pressure_face_b, &normal, area);
            let diffusive_flux = self.compute_diffusive_flux(
                &grads_a,
                if has_cell_b { Some(&grads_b) } else { None },
                &normal,
                area,
            );

            let total_flux = convective_flux - pressure_flux + diffusive_flux;
            fluxes.add_momentum_flux(face.clone(), total_flux);

            // Apply boundary conditions if any
            if let Some(bc) = boundary_handler.get_bc(&face) {
                self.apply_boundary_conditions(bc, fluxes, &face, &normal.0, area);
            }
        }
    }

    /// Computes velocity gradients using a specified gradient method.
    fn compute_velocity_gradients(
        &self,
        domain: &Mesh,
        fields: &Fields,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) -> (Section<Vector3>, Section<Vector3>, Section<Vector3>) {
        let mut gradient_calculator = Gradient::new(
            domain,
            boundary_handler,
            GradientCalculationMethod::FiniteVolume,
        );

        let mut gradient_u: Section<Vector3> = Section::new();
        let mut gradient_v: Section<Vector3> = Section::new();
        let mut gradient_w: Section<Vector3> = Section::new();

        for (component, gradient_section) in [
            ("velocity_x", &mut gradient_u),
            ("velocity_y", &mut gradient_v),
            ("velocity_z", &mut gradient_w),
        ] {
            let component_field = fields.scalar_fields.get(component)
                .expect(&format!("Field {} not found", component));
            gradient_calculator
                .compute_gradient(component_field, gradient_section, current_time)
                .expect(&format!("Gradient calculation failed for {}", component));
        }

        (gradient_u, gradient_v, gradient_w)
    }

    /// Extracts velocity, pressure, and gradient data for a given cell from the mesh and fields.
    // Previously expected `&Section<()>` but now we accept `&DashMap<MeshEntity, ()>`:
    fn extract_cell_data(
        &self,
        domain: &Mesh,
        fields: &Fields,
        gradient_u: &Section<Vector3>,
        gradient_v: &Section<Vector3>,
        gradient_w: &Section<Vector3>,
        cells: &dashmap::DashMap<MeshEntity, ()>,  // Changed here
        cell_index: usize,
    ) -> (MeshEntity, Vector3, Scalar, [f64; 3], [Vector3; 3]) {
        // We can iterate over `cells` directly because it's a DashMap:
        let mut cell_entries: Vec<(MeshEntity, ())> = cells.iter().map(|entry| (*entry.key(), *entry.value())).collect();
        cell_entries.sort_by_key(|(entity, _)| entity.get_id()); 
        // Sorting ensures a consistent ordering if needed, 
        // but this may not be necessary depending on the logic.

        let (cell, _) = cell_entries
            .get(cell_index)
            .expect("Cell index out of range");

        let velocity_field = fields.get_vector_field_value("velocity_field", cell)
            .unwrap_or(Vector3([0.0; 3]));
        let pressure_field = fields.get_scalar_field_value("pressure_field", cell)
            .unwrap_or(Scalar(0.0));
        let cell_center = domain.get_cell_centroid(cell);

        let grads = [
            gradient_u.restrict(cell).unwrap_or(Vector3([0.0; 3])),
            gradient_v.restrict(cell).unwrap_or(Vector3([0.0; 3])),
            gradient_w.restrict(cell).unwrap_or(Vector3([0.0; 3])),
        ];

        (*cell, velocity_field, pressure_field, cell_center, grads)
    }


    /// Reconstructs the velocity components at the face using cell-centered data and gradients.
    fn reconstruct_face_velocity(
        &self,
        velocity: Vector3,
        grads: &[Vector3; 3],
        cell_center: [f64; 3],
        face_center: [f64; 3],
    ) -> Vector3 {
        Vector3([
            reconstruct_face_value(velocity.0[0], grads[0].0, cell_center, face_center),
            reconstruct_face_value(velocity.0[1], grads[1].0, cell_center, face_center),
            reconstruct_face_value(velocity.0[2], grads[2].0, cell_center, face_center),
        ])
    }

    /// Reconstructs the pressure at the face.
    fn reconstruct_face_pressure(
        &self,
        pressure: Scalar,
        pressure_grad: Vector3,
        cell_center: [f64; 3],
        face_center: [f64; 3],
    ) -> f64 {
        reconstruct_face_value(pressure.0, pressure_grad.0, cell_center, face_center)
    }

    /// Computes the convective flux using the average face velocity and face normal.
    fn compute_convective_flux(
        &self,
        velocity_face_a: Vector3,
        velocity_face_b: Vector3,
        normal: &Vector3,
        area: f64,
    ) -> Vector3 {
        let avg_velocity = (velocity_face_a + velocity_face_b) * 0.5;
        let velocity_dot_normal: f64 = avg_velocity.iter().zip(normal.iter()).map(|(v, n)| v * n).sum();
        avg_velocity * (self.params.density * velocity_dot_normal * area)
    }

    /// Computes the pressure flux across the face.
    /// 
    /// # Parameters
    /// - `pressure_face_a`: The reconstructed pressure at the face from cell A's perspective.
    /// - `pressure_face_b`: The reconstructed pressure at the face from cell B's perspective.
    /// - `normal`: The face normal vector.
    /// - `area`: The face area.
    ///
    /// # Returns
    /// A `Vector3` representing the pressure flux vector across the face.
    fn compute_pressure_flux(
        &self,
        pressure_face_a: f64,
        pressure_face_b: f64,
        normal: &Vector3,
        area: f64,
    ) -> Vector3 {
        // Average the face pressures from both sides
        let pressure = 0.5 * (pressure_face_a + pressure_face_b);

        // The pressure flux is the averaged pressure times the normal times the area
        Vector3([
            pressure * normal[0] * area,
            pressure * normal[1] * area,
            pressure * normal[2] * area,
        ])
    }


    /// Computes the diffusive flux due to viscosity using the velocity gradients.
    /// 
    /// # Parameters
    /// - `grads_a`: Array of velocity component gradients (u, v, w) for the first cell.
    /// - `grads_b`: Optional array of velocity component gradients for the adjacent cell (if it exists).
    /// - `normal`: The face normal vector.
    /// - `area`: The area of the face.
    ///
    /// # Returns
    /// A `Vector3` representing the diffusive flux vector across the face.
    fn compute_diffusive_flux(
        &self,
        grads_a: &[Vector3; 3],
        grads_b: Option<&[Vector3; 3]>,
        normal: &Vector3,
        area: f64,
    ) -> Vector3 {
        // Compute the average gradient if we have two cells
        let avg_grad_u;
        let avg_grad_v;
        let avg_grad_w;
        
        if let Some(grads_b) = grads_b {
            avg_grad_u = Vector3([
                0.5 * (grads_a[0].0[0] + grads_b[0].0[0]),
                0.5 * (grads_a[0].0[1] + grads_b[0].0[1]),
                0.5 * (grads_a[0].0[2] + grads_b[0].0[2]),
            ]);

            avg_grad_v = Vector3([
                0.5 * (grads_a[1].0[0] + grads_b[1].0[0]),
                0.5 * (grads_a[1].0[1] + grads_b[1].0[1]),
                0.5 * (grads_a[1].0[2] + grads_b[1].0[2]),
            ]);

            avg_grad_w = Vector3([
                0.5 * (grads_a[2].0[0] + grads_b[2].0[0]),
                0.5 * (grads_a[2].0[1] + grads_b[2].0[1]),
                0.5 * (grads_a[2].0[2] + grads_b[2].0[2]),
            ]);
        } else {
            // Only one cell, so just use that cell's gradients
            avg_grad_u = grads_a[0];
            avg_grad_v = grads_a[1];
            avg_grad_w = grads_a[2];
        }

        // Compute normal derivatives of each velocity component
        let du_dn = avg_grad_u.dot(normal);
        let dv_dn = avg_grad_v.dot(normal);
        let dw_dn = avg_grad_w.dot(normal);

        // Diffusive flux = μ * area * [du/dn, dv/dn, dw/dn]
        Vector3([
            self.params.viscosity * du_dn * area,
            self.params.viscosity * dv_dn * area,
            self.params.viscosity * dw_dn * area,
        ])
    }


    /// Applies boundary conditions by adjusting flux values.
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
                // Example Robin BC: flux = alpha * beta * n * area
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

    fn setup_simple_mesh() -> Mesh {
        let mut builder = DomainBuilder::new();

        // Add vertices to form a simple tetrahedron-based mesh
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 0.0, 1.0])
            .add_vertex(5, [1.0, 1.0, 0.0]);

        builder.add_tetrahedron_cell(vec![1, 2, 3, 4]);
        builder.add_tetrahedron_cell(vec![2, 3, 5, 4]);

        builder.build()
    }

    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();

        let cell_ids = mesh.entities.read().unwrap().iter()
            .filter_map(|e| if let MeshEntity::Cell(id) = e { Some(*id) } else { None })
            .collect::<Vec<_>>();

        let cell_a = MeshEntity::Cell(cell_ids[0]);
        fields.set_scalar_field_value("velocity_x", cell_a.clone(), Scalar(1.0));
        fields.set_scalar_field_value("velocity_y", cell_a.clone(), Scalar(2.0));
        fields.set_scalar_field_value("velocity_z", cell_a.clone(), Scalar(3.0));
        fields.set_scalar_field_value("pressure_field", cell_a.clone(), Scalar(100.0));
        fields.set_vector_field_value("velocity_field", cell_a.clone(), Vector3([1.0, 2.0, 3.0]));

        let cell_b = MeshEntity::Cell(cell_ids[1]);
        fields.set_scalar_field_value("velocity_x", cell_b.clone(), Scalar(4.0));
        fields.set_scalar_field_value("velocity_y", cell_b.clone(), Scalar(5.0));
        fields.set_scalar_field_value("velocity_z", cell_b.clone(), Scalar(6.0));
        fields.set_scalar_field_value("pressure_field", cell_b.clone(), Scalar(50.0));
        fields.set_vector_field_value("velocity_field", cell_b.clone(), Vector3([4.0, 5.0, 6.0]));

        fields
    }

    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let bc_handler = BoundaryConditionHandler::new();
        let face = mesh.entities.read().unwrap().iter()
            .find(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .expect("No face found in mesh");

        bc_handler.set_bc(face, BoundaryCondition::Dirichlet(10.0));
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

        momentum_eq.calculate_momentum_fluxes(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        let face = mesh.entities.read().unwrap().iter()
            .find(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .expect("No face found");

        let momentum_flux = fluxes.momentum_fluxes.restrict(&face)
            .expect("Momentum flux not computed for face");

        assert!(momentum_flux.0[0].is_finite(), "Invalid flux x-component");
        assert!(momentum_flux.0[1].is_finite(), "Invalid flux y-component");
        assert!(momentum_flux.0[2].is_finite(), "Invalid flux z-component");
    }

    #[test]
    fn test_boundary_condition_application() {
        let mesh = setup_simple_mesh();
        let mut fluxes = Fluxes::new();
        let face = mesh.entities.read().unwrap().iter()
            .find(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .expect("No face found in mesh");

        let momentum_eq = MomentumEquation {
            params: MomentumParameters {
                density: 1.0,
                viscosity: 0.01,
            },
        };

        let normal = [1.0, 0.0, 0.0];
        let area = 1.0;

        // Dirichlet BC
        momentum_eq.apply_boundary_conditions(BoundaryCondition::Dirichlet(5.0), &mut fluxes, &face, &normal, area);
        let flux_value = fluxes.momentum_fluxes.restrict(&face).unwrap();
        assert_eq!(flux_value.0[0], 5.0);
        assert_eq!(flux_value.0[1], 0.0);
        assert_eq!(flux_value.0[2], 0.0);

        // Neumann BC
        momentum_eq.apply_boundary_conditions(BoundaryCondition::Neumann(2.0), &mut fluxes, &face, &normal, area);
        let flux_value = fluxes.momentum_fluxes.restrict(&face).unwrap();
        assert_eq!(flux_value.0[0], 7.0); // 5.0 (from Dirichlet) + 2.0 (from Neumann)
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

        let scalar_field = fields.scalar_fields.get("pressure_field").expect("Pressure field missing");
        let mut gradient_section: Section<Vector3> = Section::new();

        gradient_calculator
            .compute_gradient(scalar_field, &mut gradient_section, 0.0)
            .expect("Gradient computation failed");

        let cell_ids = mesh.entities.read().unwrap().iter()
            .filter_map(|e| if let MeshEntity::Cell(id) = e { Some(*id) } else { None })
            .collect::<Vec<_>>();

        for cell_id in cell_ids {
            let cell = MeshEntity::Cell(cell_id);
            let grad = gradient_section.restrict(&cell).expect("Gradient not computed for a cell");
            assert!(grad.0[0].is_finite(), "Non-finite x-gradient");
            assert!(grad.0[1].is_finite(), "Non-finite y-gradient");
            assert!(grad.0[2].is_finite(), "Non-finite z-gradient");
        }
    }
}

#[cfg(test)]
mod diffusive_flux_tests {
    use super::*;
    use crate::domain::section::Vector3;

    #[test]
    fn test_compute_diffusive_flux_single_cell() {
        let momentum_eq = MomentumEquation {
            params: MomentumParameters {
                density: 1.0,
                viscosity: 0.1,
            },
        };

        // Suppose we have gradients for a single cell:
        // du/dx = 2.0, du/dy = 0.0, du/dz = 0.0
        // dv/dx = 0.0, dv/dy = 3.0, dv/dz = 0.0
        // dw/dx = 0.0, dw/dy = 0.0, dw/dz = 4.0
        let grads_a = [
            Vector3([2.0, 0.0, 0.0]), // ∇u
            Vector3([0.0, 3.0, 0.0]), // ∇v
            Vector3([0.0, 0.0, 4.0]), // ∇w
        ];

        // Face normal and area
        let normal = Vector3([1.0, 0.0, 0.0]); // normal pointing in x-direction
        let area = 2.0;

        // No second cell
        let diff_flux = momentum_eq.compute_diffusive_flux(&grads_a, None, &normal, area);

        // du/dn = du/dx since normal is x-direction -> du/dn = 2.0
        // dv/dn = dv/dx = 0.0
        // dw/dn = dw/dx = 0.0
        // Diffusive flux = μ * area * [du/dn, dv/dn, dw/dn]
        // = 0.1 * 2.0 * [2.0, 0.0, 0.0] = [0.4, 0.0, 0.0]

        assert!((diff_flux.0[0] - 0.4).abs() < 1e-12);
        assert!((diff_flux.0[1] - 0.0).abs() < 1e-12);
        assert!((diff_flux.0[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_diffusive_flux_two_cells() {
        let momentum_eq = MomentumEquation {
            params: MomentumParameters {
                density: 1.0,
                viscosity: 0.5,
            },
        };

        // Cell A gradients
        let grads_a = [
            Vector3([2.0, 1.0, 0.0]), // ∇u (A)
            Vector3([0.0, 3.0, 0.0]), // ∇v (A)
            Vector3([0.0, 0.0, 4.0]), // ∇w (A)
        ];

        // Cell B gradients
        let grads_b = [
            Vector3([4.0, -1.0, 2.0]), // ∇u (B)
            Vector3([1.0, 2.0, 0.0]),  // ∇v (B)
            Vector3([2.0, 0.0, 1.0]),  // ∇w (B)
        ];

        let normal = Vector3([0.0, 1.0, 0.0]); // normal in y-direction
        let area = 1.0;

        let diff_flux = momentum_eq.compute_diffusive_flux(&grads_a, Some(&grads_b), &normal, area);

        // Average gradients:
        // ∇u = [(2.0 + 4.0)/2, (1.0 + -1.0)/2, (0.0 + 2.0)/2] = [3.0, 0.0, 1.0]
        // ∇v = [(0.0 + 1.0)/2, (3.0 + 2.0)/2, (0.0 + 0.0)/2] = [0.5, 2.5, 0.0]
        // ∇w = [(0.0 + 2.0)/2, (0.0 + 0.0)/2, (4.0 + 1.0)/2] = [1.0, 0.0, 2.5]

        // du/dn = ∇u · n = (3.0, 0.0, 1.0) · (0.0, 1.0, 0.0) = 0.0
        // dv/dn = ∇v · n = (0.5, 2.5, 0.0) · (0.0, 1.0, 0.0) = 2.5
        // dw/dn = ∇w · n = (1.0, 0.0, 2.5) · (0.0, 1.0, 0.0) = 0.0

        // Diffusive flux = μ * area * [du/dn, dv/dn, dw/dn]
        // μ = 0.5, area = 1.0
        // = 0.5 * [0.0, 2.5, 0.0] = [0.0, 1.25, 0.0]

        assert!((diff_flux.0[0] - 0.0).abs() < 1e-12);
        assert!((diff_flux.0[1] - 1.25).abs() < 1e-12);
        assert!((diff_flux.0[2] - 0.0).abs() < 1e-12);
    }
}

#[cfg(test)]
mod pressure_flux_tests {
    use super::*;
    use crate::domain::section::Vector3;

    #[test]
    fn test_compute_pressure_flux() {
        let momentum_eq = MomentumEquation {
            params: MomentumParameters {
                density: 1.0,    // Density not used directly in pressure flux
                viscosity: 0.01, // Viscosity not used in pressure flux
            },
        };

        // Suppose we have a face with pressures pA and pB from two neighboring cells:
        let pressure_face_a = 100.0;
        let pressure_face_b = 80.0;

        // Face normal pointing along the x-axis
        let normal = Vector3([1.0, 0.0, 0.0]);
        let area = 2.0;

        // Compute the pressure flux
        let flux = momentum_eq.compute_pressure_flux(pressure_face_a, pressure_face_b, &normal, area);

        // The average pressure = (100.0 + 80.0) / 2 = 90.0
        // Pressure flux = p_avg * normal * area = 90.0 * [1.0, 0.0, 0.0] * 2.0 = [180.0, 0.0, 0.0]

        assert!((flux.0[0] - 180.0).abs() < 1e-12, "Incorrect x-component of pressure flux");
        assert!((flux.0[1] - 0.0).abs() < 1e-12, "y-component should be zero");
        assert!((flux.0[2] - 0.0).abs() < 1e-12, "z-component should be zero");
    }

    #[test]
    fn test_compute_pressure_flux_non_aligned_normal() {
        let momentum_eq = MomentumEquation {
            params: MomentumParameters {
                density: 1.0,
                viscosity: 0.01,
            },
        };

        // Pressures on each side
        let pressure_face_a = 50.0;
        let pressure_face_b = 70.0;

        // Average pressure = 60.0

        // Now consider a normal not aligned with coordinate axes:
        // For example, a normal in the direction (1,1,0) but not normalized.
        let normal = Vector3([1.0, 1.0, 0.0]);
        let area = 1.0;

        let flux = momentum_eq.compute_pressure_flux(pressure_face_a, pressure_face_b, &normal, area);

        // flux = p_avg * normal * area = 60.0 * [1.0, 1.0, 0.0] * 1.0
        // = [60.0, 60.0, 0.0]

        assert!((flux.0[0] - 60.0).abs() < 1e-12, "Incorrect x-component of pressure flux");
        assert!((flux.0[1] - 60.0).abs() < 1e-12, "Incorrect y-component of pressure flux");
        assert!((flux.0[2] - 0.0).abs() < 1e-12, "z-component should be zero");
    }
}
