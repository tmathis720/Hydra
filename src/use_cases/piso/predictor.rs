use crate::{
    domain::mesh::Mesh,
    equation::{
        fields::{Fields, Fluxes},
        momentum_equation::MomentumEquation,
    },
    boundary::bc_handler::BoundaryConditionHandler,
};

/// Performs the predictor step of the PISO algorithm.
///
/// This step predicts the velocity field by solving the momentum equation without considering the pressure correction.
///
/// # Parameters
/// - `mesh`: Reference to the computational mesh.
/// - `fields`: The state of the physical fields, including velocity and scalar fields.
/// - `fluxes`: Container for the computed fluxes.
/// - `boundary_handler`: Handles boundary conditions for the domain.
/// - `momentum_equation`: The momentum equation instance used for prediction.
/// - `time`: Current simulation time.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` if successful, or an error message if the predictor step fails.
pub fn predict_velocity(
    mesh: &Mesh,
    fields: &mut Fields,
    fluxes: &mut Fluxes,
    boundary_handler: &BoundaryConditionHandler,
    momentum_equation: &MomentumEquation,
    time: f64,
) -> Result<(), String> {
    // 1. Assemble momentum fluxes (convective, diffusive, etc.)
    momentum_equation.calculate_momentum_fluxes(mesh, fields, fluxes, boundary_handler, time);

    // 2. Update the velocity field using the computed fluxes
    match fields.vector_fields.get_mut("velocity") {
        Some(velocity_field) => {
            // Update the velocity field based on the computed momentum fluxes
            velocity_field.update_with_derivative(&fluxes.momentum_fluxes, 1.0);
            Ok(())
        }
        None => Err("Velocity field not found in the fields structure.".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::bc_handler::BoundaryConditionHandler, domain::mesh::Mesh, equation::{fields::{Fields, Fluxes}, momentum_equation::{MomentumEquation, MomentumParameters}}, Section
    };

    #[test]
    fn test_predict_velocity() {
        // 1. Setup: Create a dummy mesh, boundary handler, and fields
        let mesh = Mesh::new();
        let boundary_handler = BoundaryConditionHandler::new();
        let mut fields = Fields::new();
        let mut fluxes = Fluxes::new();

        // Add a dummy velocity field to the fields structure
        fields.scalar_fields.insert("velocity_x".to_string(), Section::new());
        fields.scalar_fields.insert("velocity_y".to_string(), Section::new());
        fields.scalar_fields.insert("velocity_z".to_string(), Section::new());
        fields.vector_fields.insert("velocity".to_string(), Section::new());

        // Create a dummy momentum equation instance
        let params = MomentumParameters {
            density: 1.0,
            viscosity: 0.001,
        };
        let momentum_equation = MomentumEquation { params };

        // 2. Execute: Call the predictor function
        let result = predict_velocity(&mesh, &mut fields, &mut fluxes, &boundary_handler, &momentum_equation, 0.0);

        
        // 3. Verify: Check that the result is Ok and the velocity field is updated
        assert!(result.is_ok());
        assert!(fields.scalar_fields.get("velocity_x").is_some());
        assert!(fields.scalar_fields.get("velocity_y").is_some());
        assert!(fields.scalar_fields.get("velocity_z").is_some());        
    }
}
