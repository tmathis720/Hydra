use crate::{
    boundary::bc_handler::BoundaryConditionHandler, domain::{mesh::Mesh, section::{Scalar, Vector3}}, equation::{
        fields::Fields,
        gradient::{Gradient, GradientCalculationMethod},
    }, Section
};

/// Corrects the velocity field using the pressure correction to ensure divergence-free flow.
///
/// # Parameters
/// - `mesh`: The computational mesh.
/// - `fields`: The current state of the physical fields, including velocity and pressure.
/// - `pressure_correction`: The pressure correction computed from the pressure Poisson equation.
/// - `boundary_handler`: Handles boundary conditions for the domain.
///
/// # Returns
/// - `Result<(), String>`: Returns `Ok(())` if successful, or an error message on failure.
pub fn correct_velocity(
    mesh: &Mesh,
    fields: &mut Fields,
    pressure_correction: &Section<Scalar>,
    boundary_handler: &BoundaryConditionHandler,
) -> Result<(), String> {
    // 1. Retrieve the velocity field
    let velocity_field = match fields.vector_fields.get_mut("velocity") {
        Some(field) => field,
        None => return Err("Velocity field not found in the fields structure.".to_string()),
    };

    // 2. Initialize gradient calculator
    let gradient_method = GradientCalculationMethod::FiniteVolume;
    let mut gradient_calculator = Gradient::new(mesh, boundary_handler, gradient_method);

    // 3. Compute the gradient of the pressure correction
    let mut pressure_gradient = Section::<Vector3>::new();
    gradient_calculator.compute_gradient(pressure_correction, &mut pressure_gradient, 0.0)?;

    // 4. Correct the velocity field using the pressure gradient
    velocity_field.update_with_derivative(&pressure_gradient, -1.0); // Apply the correction

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::mesh::Mesh,
        boundary::bc_handler::BoundaryConditionHandler,
        equation::fields::Fields,
        domain::section::{Section, Scalar, Vector3},
    };

    #[test]
    fn test_correct_velocity() {
        // 1. Create a mock mesh and boundary handler
        let mesh = Mesh::new();
        let boundary_handler = BoundaryConditionHandler::new();

        // 2. Initialize fields with dummy velocity and pressure data
        let mut fields = Fields::new();
        let mut velocity_field = Section::<Vector3>::new();
        velocity_field.set_data(1.into(), Vector3([1.0, 0.0, 0.0]));
        fields.vector_fields.insert("velocity".to_string(), velocity_field);

        let mut pressure_correction = Section::<Scalar>::new();
        pressure_correction.set_data(1.into(), Scalar(0.5));

        // 3. Run the velocity correction
        let result = correct_velocity(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        // 4. Assert successful correction
        assert!(result.is_ok());
        assert!(fields.vector_fields.get("velocity").is_some());
    }
}
