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
    println!("Starting velocity correction.");

    // 1. Retrieve the velocity field
    let velocity_field = match fields.vector_fields.get_mut("velocity") {
        Some(field) => {
            println!("Velocity field found.");
            field
        }
        None => {
            println!("Error: Velocity field not found.");
            return Err("Velocity field not found in the fields structure.".to_string());
        }
    };

    // 2. Initialize gradient calculator
    println!("Initializing gradient calculator using the Finite Volume method.");
    let gradient_method = GradientCalculationMethod::FiniteVolume;
    let mut gradient_calculator = Gradient::new(mesh, boundary_handler, gradient_method);

    // 3. Compute the gradient of the pressure correction
    println!("Computing gradient of the pressure correction.");
    let mut pressure_gradient = Section::<Vector3>::new();
    gradient_calculator
        .compute_gradient(pressure_correction, &mut pressure_gradient, 0.0)
        .map_err(|e| {
            println!("Error during gradient computation: {:?}", e);
            format!("Gradient computation failed: {:?}", e)
        })?;

    // Debugging: Print pressure correction gradient values
    println!("Pressure gradient computed. Values:");
    for entry in pressure_gradient.data.iter() {
        let (entity, gradient) = entry.pair();
        println!("Entity {:?}: Gradient = {:?}", entity, gradient.0);
    }

    // 4. Correct the velocity field using the pressure gradient
    println!("Correcting the velocity field using the computed pressure gradient.");
    velocity_field.update_with_derivative(&pressure_gradient, -1.0); // Apply the correction
    println!("Velocity field updated successfully.");

    Ok(())
}
