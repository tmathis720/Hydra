use crate::domain::Element;
use nalgebra::Vector3;

/// Struct representing a solver for eddy viscosity-based diffusion.
pub struct EddyViscositySolver {
    pub nu_t: f64, // Eddy viscosity coefficient (turbulent viscosity)
}

impl EddyViscositySolver {
    /// Applies eddy viscosity-based diffusion between two elements.
    ///
    /// This method diffuses momentum between two elements based on the velocity difference
    /// and eddy viscosity. The flux is computed using a simple eddy viscosity model.
    ///
    /// # Arguments
    /// * `element_left` - The left element.
    /// * `element_right` - The right element.
    /// * `dt` - Time step size.
    pub fn apply_diffusion(&self, element_left: &mut Element, element_right: &mut Element, dt: f64) {
        // Compute the velocity difference
        let velocity_diff = element_right.velocity - element_left.velocity;

        // Compute the diffusive flux (proportional to velocity gradient and viscosity)
        let flux = self.nu_t * velocity_diff;

        // Apply the diffusive flux to the momentum of both elements
        element_left.momentum += flux * dt;
        element_right.momentum -= flux * dt;
    }

    /// Compute the velocity gradient between two elements.
    ///
    /// # Arguments
    /// * `element_left` - The left element.
    /// * `element_right` - The right element.
    ///
    /// # Returns
    /// A `Vector3<f64>` representing the velocity difference.
    fn _compute_velocity_gradient(&self, element_left: &Element, element_right: &Element) -> Vector3<f64> {
        element_right.velocity - element_left.velocity
    }

    /// Update the velocity of an element based on its momentum and mass.
    ///
    /// This ensures that after applying diffusion, the velocities are updated accordingly.
    ///
    /// # Arguments
    /// * `element` - The element whose velocity needs to be updated.
    pub fn update_velocity(&self, element: &mut Element) {
        if element.mass > 0.0 {
            element.velocity = element.momentum / element.mass;
        } else {
            element.velocity = Vector3::zeros();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Element;
    use nalgebra::Vector3;

    #[test]
    fn test_apply_diffusion() {
        // Create two elements with different velocities
        let mut element_left = Element {
            velocity: Vector3::new(2.0, 0.0, 0.0),
            momentum: Vector3::new(2.0, 0.0, 0.0),
            mass: 1.0,
            ..Default::default()
        };

        let mut element_right = Element {
            velocity: Vector3::new(4.0, 0.0, 0.0),
            momentum: Vector3::new(4.0, 0.0, 0.0),
            mass: 1.0,
            ..Default::default()
        };

        let solver = EddyViscositySolver { nu_t: 0.5 };
        let dt = 0.1;

        // Apply diffusion
        solver.apply_diffusion(&mut element_left, &mut element_right, dt);

        // Check that the momentum of both elements has changed
        assert!(element_left.momentum.x < 2.0, "Left element momentum should decrease");
        assert!(element_right.momentum.x > 4.0, "Right element momentum should increase");
    }

    #[test]
    fn test_velocity_update() {
        let mut element = Element {
            momentum: Vector3::new(6.0, 3.0, 0.0),
            mass: 2.0,
            ..Default::default()
        };

        let solver = EddyViscositySolver { nu_t: 0.5 };

        // Update the velocity based on momentum and mass
        solver.update_velocity(&mut element);

        // Check if the velocity was correctly updated
        assert_eq!(element.velocity.x, 3.0);
        assert_eq!(element.velocity.y, 1.5);
        assert_eq!(element.velocity.z, 0.0);
    }
}
