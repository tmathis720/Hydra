use crate::domain::Element;
use nalgebra::Vector3;

/// Struct representing a solver for eddy viscosity-based diffusion.
pub struct EddyViscositySolver {
    pub nu_t: f64,     // Eddy viscosity coefficient
    pub cs: f64,       // Smagorinsky constant (default: 0.1 - 0.2 typically)
    pub delta: f64,    // Grid scale (element size in the horizontal direction)
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
    pub fn apply_horizontal_diffusion(
        &self, 
        element_left: &mut Element, 
        element_right: &mut Element, 
        dt: f64
    ) {
        // Compute velocity gradient in the horizontal plane
        let velocity_gradient = self._compute_velocity_gradient(element_left, element_right);
        
        // Calculate strain-rate tensor (only in the horizontal components)
        let strain_rate = 0.5 * (velocity_gradient.x.abs() + velocity_gradient.y.abs());

        // Calculate the eddy viscosity based on the Smagorinsky model
        let nu_t = (self.cs * self.delta).powi(2) * strain_rate;

        // Compute the diffusive flux in the horizontal direction
        let flux = nu_t * velocity_gradient;

        // Apply the diffusive flux to the momentum of both elements (only in the horizontal direction)
        element_left.momentum.x += flux.x * dt;
        element_right.momentum.x -= flux.x * dt;
        element_left.momentum.y += flux.y * dt;
        element_right.momentum.y -= flux.y * dt;
    }

    /// Compute the velocity gradient between two elements.
    ///
    /// # Arguments
    /// * `element_left` - The left element.
    /// * `element_right` - The right element.
    ///
    /// # Returns
    /// A `Vector3<f64>` representing the velocity difference.
    /// Computes velocity gradient in the horizontal plane (ignores the z component).
    fn _compute_velocity_gradient(&self, element_left: &Element, element_right: &Element) -> Vector3<f64> {
        Vector3::new(
            element_right.velocity.x - element_left.velocity.x,
            element_right.velocity.y - element_left.velocity.y,
            0.0, // Ignoring z-direction for horizontal diffusion
        )
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
    fn test_horizontal_diffusion() {
        let mut element_left = Element {
            velocity: Vector3::new(1.0, 0.5, 0.0),
            mass: 2.0,
            ..Default::default()
        };
        let mut element_right = Element {
            velocity: Vector3::new(10.0, 0.5, 0.0),
            mass: 2.0,
            ..Default::default()
        };
    
        let solver = EddyViscositySolver { nu_t: 0.5, cs: 0.1, delta: 1.0 };
    
        // Apply diffusion with a timestep of 1.0
        solver.apply_horizontal_diffusion(&mut element_left, &mut element_right, 10.0);
    
        // Check that the velocity has diffused only in the horizontal direction
        println!("element_left momentum: {:?}", element_left.momentum);  // Debugging
        assert!(element_left.momentum.x > 1.0, "Expected element_left momentum.x to be greater than 1.0");
        assert_eq!(element_left.momentum.z, 0.0); // z-component should remain zero
    }

    #[test]
    fn test_velocity_update() {
        let mut element = Element {
            momentum: Vector3::new(6.0, 3.0, 0.0),
            mass: 2.0,
            ..Default::default()
        };

        let solver = EddyViscositySolver { nu_t: 0.5, cs: 0.1, delta: 1.0  };

        // Update the velocity based on momentum and mass
        solver.update_velocity(&mut element);

        // Check if the velocity was correctly updated
        assert_eq!(element.velocity.x, 3.0);
        assert_eq!(element.velocity.y, 1.5);
        assert_eq!(element.velocity.z, 0.0);
    }
}
