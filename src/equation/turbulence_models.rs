use crate::{
    boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler}, domain::{mesh::Mesh, section::{vector, Scalar, Vector3}, Section}, equation::{
        fields::{Fields, Fluxes},
        PhysicalEquation,
    }, geometry::{FaceShape, Geometry}, MeshEntity
};

use crate::equation::gradient::{Gradient, GradientCalculationMethod};

use super::reconstruction::{LinearReconstruction, ReconstructionMethod};

/// A trait defining the required interface for turbulence models.
///
/// This trait provides a skeletal structure. Concrete turbulence models can implement:
/// - Which scalar/vector fields they need (e.g., TKE, dissipation, salinity, temperature).
/// - How to compute eddy viscosities, diffusivities, and other turbulent properties.
/// - How to calculate and add turbulence fluxes to `fluxes.turbulence_fluxes`.
pub trait TurbulenceModel {
    fn average_gradient(&self, g_a: Vector3, g_b: Vector3, has_cell_b: bool) -> Vector3;
    fn extract_cell_scalars(&self, domain: &Mesh, fields: &Fields, grad_1: &Section<Vector3>, grad_2: &Section<Vector3>, cells: &[(MeshEntity, ())], cell_index: usize) -> (f64, f64, Vector3, Vector3, [f64; 3]);
    /// Computes turbulence fluxes and updates `turbulence_fluxes`.
    ///
    /// # Parameters
    /// - `domain`: The computational mesh.
    /// - `fields`: Current field data.
    /// - `fluxes`: Fluxes to be updated.
    /// - `boundary_handler`: For applying boundary conditions.
    /// - `current_time`: Current simulation time.
    fn compute_turbulence_fluxes(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    );
}

/// A generic ocean turbulence model (GOTM) skeleton.
///
/// This structure outlines the steps needed to implement a general ocean turbulence model.
/// In a full implementation, you might specify which fields represent turbulence variables,
/// how to compute eddy viscosities/diffusivities, and how to handle boundary conditions.
pub struct GOTMModel {
    // Placeholder for model parameters and constants
    pub eddy_viscosity: f64,
    pub eddy_diffusivity: f64,
    // Additional parameters can be added here
}

impl GOTMModel {
    /// Creates a new GOTMModel with default parameters (placeholder).
    pub fn new() -> Self {
        GOTMModel {
            eddy_viscosity: 1e-5,   // Just a placeholder value
            eddy_diffusivity: 1e-6, // Just a placeholder value
        }
    }

    /// Example function: compute a scalar gradient needed by the turbulence model.
    ///
    /// In a real model, you'd specify which scalar fields you need (e.g., "k", "epsilon",
    /// "temperature", "salinity"), and compute their gradients.
    fn compute_required_gradients(
        &self,
        domain: &Mesh,
        boundary_handler: &BoundaryConditionHandler,
        fields: &Fields,
        current_time: f64,
    ) -> (Section<Vector3>, Section<Vector3>) {
        let mut gradient_calculator = Gradient::new(domain, boundary_handler, GradientCalculationMethod::FiniteVolume);

        // As an example, let's say the turbulence model needs gradients of two scalar fields:
        // "turb_scalar_1" and "turb_scalar_2". In GOTM, these could be e.g. TKE and Dissipation.
        let turb_scalar_1 = fields.scalar_fields.get("turb_scalar_1")
            .expect("turb_scalar_1 field not found");
        let turb_scalar_2 = fields.scalar_fields.get("turb_scalar_2")
            .expect("turb_scalar_2 field not found");

        let mut grad_1: Section<Vector3> = Section::new();
        let mut grad_2: Section<Vector3> = Section::new();

        gradient_calculator.compute_gradient(turb_scalar_1, &mut grad_1, current_time)
            .expect("Gradient computation failed for turb_scalar_1");
        gradient_calculator.compute_gradient(turb_scalar_2, &mut grad_2, current_time)
            .expect("Gradient computation failed for turb_scalar_2");

        (grad_1, grad_2)
    }

    /// Example function: compute a diffusive flux for turbulence-related scalars.
    /// In a real model, you'd use actual formulas for eddy diffusivity, etc.
    fn compute_diffusive_flux(
        &self,
        normal: &Vector3,
        area: f64,
        grad_val: Vector3,
        eddy_diff: f64,
    ) -> f64 {
        let dphi_dn = grad_val.dot(normal);
        // Simple Fickian diffusion: flux = -eddy_diff * dphi/dn * area
        -eddy_diff * dphi_dn * area
    }

    /// Apply boundary conditions to turbulence fluxes if needed.
    /// Here we just provide a placeholder that can be extended.
    fn apply_turbulence_bc(
        &self,
        _bc: &BoundaryCondition,
        _flux: &mut vector::Vector2,
        _normal: &Vector3,
        _area: f64,
    ) {
        // In a real model, you'd modify flux based on BC type.
    }
}

impl TurbulenceModel for GOTMModel {
    fn compute_turbulence_fluxes(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        let (grad_1, grad_2) = self.compute_required_gradients(domain, boundary_handler, fields, current_time);
        let geometry = Geometry::new();

        // Loop over faces and compute fluxes
        for face in domain.get_faces() {
            let face_vertices = domain.get_face_vertices(&face);
            let face_vertices = face_vertices.unwrap();
            let face_shape = match face_vertices.len() {
                3 => FaceShape::Triangle,
                4 => FaceShape::Quadrilateral,
                _ => continue,
            };

            let normal = match domain.get_face_normal(&face, None) {
                Ok(n) => n,
                Err(_) => continue,
            };

            let area = match domain.get_face_area(&face) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let face_center = geometry.compute_face_centroid(face_shape, &face_vertices);
            let cells = domain.get_cells_sharing_face(&face).unwrap();

            if cells.is_empty() {
                continue;
            }

            // For demonstration, assume we have a scalar "turb_scalar_1" and "turb_scalar_2"
            // from each cell. We'll reconstruct them at the face and compute their fluxes.
            let cell_entries: Vec<_> = cells.iter().map(|e| (*e.key(), *e.value())).collect();
            let mut sorted_cells = cell_entries.clone();
            sorted_cells.sort_by_key(|(ent, _)| ent.get_id());

            let (scalar_1_a, scalar_2_a, grad_1_a, grad_2_a, center_a) =
                self.extract_cell_scalars(domain, fields, &grad_1, &grad_2, &sorted_cells, 0);

            let has_cell_b = sorted_cells.len() > 1;
            let (scalar_1_b, scalar_2_b, grad_1_b, grad_2_b, center_b) = if has_cell_b {
                let (s1_b, s2_b, g1_b, g2_b, c_b) =
                    self.extract_cell_scalars(domain, fields, &grad_1, &grad_2, &sorted_cells, 1);
                (s1_b, s2_b, g1_b, g2_b, c_b)
            } else {
                (scalar_1_a, scalar_2_a, grad_1_a, grad_2_a, center_a)
            };

            // Reconstruct scalars at face
            let reconstruction: Box<dyn ReconstructionMethod> = Box::new(LinearReconstruction);
            let scalar_1_face_a = reconstruction.reconstruct(scalar_1_a, grad_1_a.0, center_a, face_center);
            let scalar_2_face_a = reconstruction.reconstruct(scalar_2_a, grad_2_a.0, center_a, face_center);

            let scalar_1_face_b = if has_cell_b {
                reconstruction.reconstruct(scalar_1_b, grad_1_b.0, center_b, face_center)
            } else {
                scalar_1_face_a
            };
            let scalar_2_face_b = if has_cell_b {
                reconstruction.reconstruct(scalar_2_b, grad_2_b.0, center_b, face_center)
            } else {
                scalar_2_face_a
            };

            // Average the scalars at the face
            let _scalar_1_face = 0.5 * (scalar_1_face_a + scalar_1_face_b);
            let _scalar_2_face = 0.5 * (scalar_2_face_a + scalar_2_face_b);

            // Average gradients
            let avg_grad_1 = self.average_gradient(grad_1_a, grad_1_b, has_cell_b);
            let avg_grad_2 = self.average_gradient(grad_2_a, grad_2_b, has_cell_b);

            // Compute diffusive fluxes for these scalars using eddy_diffusivity (just a placeholder)
            let flux_scalar_1 = self.compute_diffusive_flux(&normal, area, avg_grad_1, self.eddy_diffusivity);
            let flux_scalar_2 = self.compute_diffusive_flux(&normal, area, avg_grad_2, self.eddy_diffusivity);

            let mut turb_flux = vector::Vector2([flux_scalar_1, flux_scalar_2]);

            // Apply BC if present
            if let Some(bc) = boundary_handler.get_bc(&face) {
                self.apply_turbulence_bc(&bc, &mut turb_flux, &normal, area);
            }

            // Add the computed flux to turbulence_fluxes
            if let Some(mut current) = fluxes.turbulence_fluxes.data.get_mut(&face) {
                current.value_mut().0[0] += turb_flux.0[0];
                current.value_mut().0[1] += turb_flux.0[1];
            } else {
                fluxes.turbulence_fluxes.set_data(face, turb_flux);
            }
        }
    }

    fn extract_cell_scalars(
        &self,
        domain: &Mesh,
        fields: &Fields,
        grad_1: &Section<Vector3>,
        grad_2: &Section<Vector3>,
        cells: &[(MeshEntity, ())],
        cell_index: usize,
    ) -> (f64, f64, Vector3, Vector3, [f64; 3]) {
        let (cell, _) = cells.get(cell_index).expect("Cell index out of range");

        // Retrieve scalar field values from fields. Assume "turb_scalar_1" and "turb_scalar_2".
        let scalar_1 = fields.scalar_fields.get("turb_scalar_1")
            .and_then(|f| f.restrict(cell).ok())
            .unwrap_or(Scalar(0.0)).0;
        let scalar_2 = fields.scalar_fields.get("turb_scalar_2")
            .and_then(|f| f.restrict(cell).ok())
            .unwrap_or(Scalar(0.0)).0;

        let g1 = grad_1.restrict(cell).unwrap_or(Vector3([0.0; 3]));
        let g2 = grad_2.restrict(cell).unwrap_or(Vector3([0.0; 3]));

        let center = domain.get_cell_centroid(cell).unwrap();

        (scalar_1, scalar_2, g1, g2, center)
    }

    fn average_gradient(&self, g_a: Vector3, g_b: Vector3, has_cell_b: bool) -> Vector3 {
        if has_cell_b {
            Vector3([
                0.5 * (g_a[0] + g_b[0]),
                0.5 * (g_a[1] + g_b[1]),
                0.5 * (g_a[2] + g_b[2]),
            ])
        } else {
            g_a
        }
    }
}

/// Implementing `PhysicalEquation` for GOTMModel allows it to be integrated into the solver.
impl PhysicalEquation for GOTMModel {
    fn assemble(
        &self,
        domain: &Mesh,
        fields: &Fields,
        fluxes: &mut Fluxes,
        boundary_handler: &BoundaryConditionHandler,
        current_time: f64,
    ) {
        self.compute_turbulence_fluxes(domain, fields, fluxes, boundary_handler, current_time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        equation::fields::{Fields, Fluxes},
        boundary::bc_handler::BoundaryConditionHandler,
        interface_adapters::domain_adapter::DomainBuilder,
        domain::section::Scalar,
    };

    fn setup_simple_mesh() -> crate::domain::mesh::Mesh {
        let mut builder = DomainBuilder::new();

        // A minimal mesh
        builder
            .add_vertex(1, [0.0, 0.0, 0.0])
            .add_vertex(2, [1.0, 0.0, 0.0])
            .add_vertex(3, [0.0, 1.0, 0.0])
            .add_vertex(4, [0.0, 0.0, 1.0]);

        builder.add_tetrahedron_cell(vec![1, 2, 3, 4]);

        builder.build()
    }

    fn setup_fields(mesh: &crate::domain::mesh::Mesh) -> Fields {
        let mut fields = Fields::new();

        let cell = mesh.get_cells()[0];

        // Define two turbulence-related scalars: turb_scalar_1, turb_scalar_2
        fields.set_scalar_field_value("turb_scalar_1", cell, Scalar(1.0));
        fields.set_scalar_field_value("turb_scalar_2", cell, Scalar(2.0));

        fields
    }

    #[test]
    fn test_gotm_model_assemble() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let boundary_handler = BoundaryConditionHandler::new();

        let gotm_model = GOTMModel::new();
        gotm_model.assemble(&mesh, &fields, &mut fluxes, &boundary_handler, 0.0);

        // With a single tetrahedron cell, we have 4 faces. Some fluxes should have been computed.
        let faces = mesh.get_faces();
        let mut count = 0;
        for face in faces {
            if let Ok(flux) = fluxes.turbulence_fluxes.restrict(&face) {
                // Just verify the flux is finite
                assert!(flux.0[0].is_finite(), "Flux for turb_scalar_1 is not finite");
                assert!(flux.0[1].is_finite(), "Flux for turb_scalar_2 is not finite");
                count += 1;
            }
        }

        // Expect at least one face to have turbulence fluxes computed
        assert!(count > 0, "At least one face should have a turbulence flux");
    }
}
