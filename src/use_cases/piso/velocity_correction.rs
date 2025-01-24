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

#[cfg(test)]
mod correct_velocity_tests {
    use super::*;
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        domain::mesh::Mesh, domain::section::{Scalar, Vector3},
        equation::fields::Fields, interface_adapters::domain_adapter::DomainBuilder,
    };

    /// Sets up a simple tetrahedral mesh for testing.
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

    /// Sets up fields, including a velocity field and a pressure field, with all cells initialized.
    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();

        // Assign velocity and pressure values to all cells
        for cell in mesh.get_cells() {
            fields.set_vector_field_value("velocity", cell.clone(), Vector3([1.0, 1.0, 1.0]));
            fields.set_scalar_field_value("pressure", cell.clone(), Scalar(100.0));
        }

        fields
    }

    /// Sets up a mock pressure correction field for all cells.
    fn setup_pressure_correction(mesh: &Mesh) -> Section<Scalar> {
        let pressure_correction = Section::new();
        for cell in mesh.get_cells() {
            pressure_correction.set_data(cell.clone(), Scalar(10.0)); // Default correction
        }
        pressure_correction
    }

    /// Sets up boundary conditions for the mesh, ensuring consistent application.
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let boundary_handler = BoundaryConditionHandler::new();

        // Assign Dirichlet boundary conditions to all faces
        for face in mesh.get_faces() {
            boundary_handler.set_bc(face.clone(), BoundaryCondition::Dirichlet(0.0));
        }

        boundary_handler
    }

    #[test]
    fn test_correct_velocity_success() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let pressure_correction = setup_pressure_correction(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        let result = correct_velocity(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        assert!(result.is_ok(), "Velocity correction should succeed.");
        let velocity_field = fields.vector_fields.get("velocity").expect("Velocity field missing");

        for entry in velocity_field.data.iter() {
            let (entity, velocity) = entry.pair();
            println!("Entity {:?}: Updated velocity = {:?}", entity, velocity.0);
        }
    }

    #[test]
    fn test_correct_velocity_missing_velocity_field() {
        let mesh = setup_simple_mesh();
        let mut fields = Fields::new(); // Missing velocity field
        let pressure_correction = setup_pressure_correction(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        let result = correct_velocity(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        assert!(result.is_err(), "Should fail when velocity field is missing.");
        assert_eq!(
            result.unwrap_err(),
            "Velocity field not found in the fields structure.",
            "Unexpected error message."
        );
    }

    #[test]
    fn test_correct_velocity_invalid_pressure_correction() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let pressure_correction = Section::<Scalar>::new(); // Empty pressure correction
        let boundary_handler = setup_boundary_conditions(&mesh);

        let result = correct_velocity(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        assert!(result.is_err(), "Should fail with empty pressure correction.");
        assert!(
            result.unwrap_err().contains("Gradient computation failed"),
            "Unexpected error message for gradient failure."
        );
    }

    #[test]
    fn test_correct_velocity_boundary_conditions() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let pressure_correction = setup_pressure_correction(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);

        let result = correct_velocity(&mesh, &mut fields, &pressure_correction, &boundary_handler);

        assert!(result.is_ok(), "Boundary conditions should be applied correctly.");
        let velocity_field = fields.vector_fields.get("velocity").expect("Velocity field missing");

        for entry in velocity_field.data.iter() {
            let (entity, velocity) = entry.pair();
            println!("Entity {:?}: Velocity with BC applied = {:?}", entity, velocity.0);
        }
    }
}
