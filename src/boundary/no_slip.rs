use crate::domain::Element;
use crate::domain::Face;

pub struct NoSlipBoundary;

impl NoSlipBoundary {
    /// Apply no-slip boundary condition (velocity should be zero)
    pub fn apply_boundary(&self, element: &mut Element, _dt: f64) {
        element.momentum = 0.0; // No-slip boundary means zero velocity
    }

    // Function to determine if a node is on a boundary (can be expanded)
    pub fn is_no_slip_face(&self, face: &Face) -> bool {
        face.nodes.0 == 0 || face.nodes.1 == 0 // Adjust logic as needed
    }
}