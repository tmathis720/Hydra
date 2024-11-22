#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};
    use crate::{boundary::bc_handler::BoundaryConditionHandler, domain::{mesh::Mesh, section::{Scalar, Vector3}}, equation::{equation::Equation, fields::{Fields, Fluxes}, manager::EquationManager, PhysicalEquation}, equation::manager::NoOpStepper, MeshEntity, Section};

    // Helper to create a default Mesh and BoundaryConditionHandler
    fn setup_environment() -> (Arc<RwLock<Mesh>>, Arc<RwLock<BoundaryConditionHandler>>) {
        let domain = Arc::new(RwLock::new(Mesh::new()));
        let boundary_handler = Arc::new(RwLock::new(BoundaryConditionHandler::new()));
        (domain, boundary_handler)
    }

    #[test]
    fn test_physical_equation_assemble() {
        let (domain, boundary_handler) = setup_environment();
        let fields = Fields::new();
        let mut fluxes = Fluxes::new();

        struct MockEquation;
        impl PhysicalEquation for MockEquation {
            fn assemble(
                &self,
                _domain: &Mesh,
                _fields: &Fields,
                _fluxes: &mut Fluxes,
                _boundary_handler: &BoundaryConditionHandler,
                _current_time: f64,
            ) {
                // Mock behavior: no operation, just validate callable
            }
        }

        let equation = MockEquation {};
        equation.assemble(
            &domain.read().unwrap(),
            &fields,
            &mut fluxes,
            &boundary_handler.read().unwrap(),
            0.0,
        );

        // Assert that the fluxes remain unmodified (no behavior in mock)
        assert!(fluxes.momentum_fluxes.data.is_empty());
    }

    #[test]
    fn test_equation_manager_add_and_assemble() {
        let (domain, boundary_handler) = setup_environment();
        let time_stepper = Box::new(NoOpStepper);
        let mut manager = EquationManager::new(time_stepper, domain.clone(), boundary_handler.clone());

        struct TestEquation;
        impl PhysicalEquation for TestEquation {
            fn assemble(
                &self,
                _domain: &Mesh,
                _fields: &Fields,
                fluxes: &mut Fluxes,
                _boundary_handler: &BoundaryConditionHandler,
                _current_time: f64,
            ) {
                fluxes.add_energy_flux(MeshEntity::Face(0), Scalar(1.0)); // Mock flux addition
            }
        }

        manager.add_equation(TestEquation {});

        let fields = Fields::new();
        let mut fluxes = Fluxes::new();
        manager.assemble_all(&fields, &mut fluxes);

        // Compare with owned value
        let energy_flux = fluxes.energy_fluxes.data.get(&MeshEntity::Face(0)).unwrap();
        assert!(*energy_flux == Scalar(1.0));
    }

    #[test]
    fn test_equation_manager_step() {
        let (domain, boundary_handler) = setup_environment();
        let time_stepper = Box::new(NoOpStepper);
        let mut manager = EquationManager::new(time_stepper, domain.clone(), boundary_handler.clone());

        let mut fields = Fields::new();
        fields.set_scalar_field_value("test", MeshEntity::Cell(0), Scalar(2.0));

        manager.step(&mut fields);

        // As NoOpStepper does nothing, fields should remain unchanged
        let scalar_value = fields.get_scalar_field_value("test", &MeshEntity::Cell(0)).unwrap();
        assert!(scalar_value == Scalar(2.0));
    }

    #[test]
    fn test_fluxes_manipulation() {
        let mut fluxes = Fluxes::new();

        // Add momentum flux
        fluxes.add_momentum_flux(MeshEntity::Face(0), Vector3([1.0, 0.0, 0.0]));
        fluxes.add_energy_flux(MeshEntity::Face(0), Scalar(2.0));

        // Compare momentum flux
        let momentum_flux = fluxes.momentum_fluxes.data.get(&MeshEntity::Face(0)).unwrap();
        assert!(*momentum_flux.value() == Vector3([1.0, 0.0, 0.0]));

        // Compare energy flux
        let energy_flux = fluxes.energy_fluxes.data.get(&MeshEntity::Face(0)).unwrap();
        assert!(*energy_flux == Scalar(2.0));
    }

    #[test]
    fn test_calculate_fluxes() {
        let mesh = Mesh::new();
        let velocity_field = Section::new(); // Mock empty field
        let pressure_field = Section::new(); // Mock empty field
        let mut fluxes = Section::<Vector3>::new();
        let boundary_handler = BoundaryConditionHandler::new();

        let equation = Equation {};
        equation.calculate_fluxes(
            &mesh,
            &velocity_field,
            &pressure_field,
            &mut fluxes,
            &boundary_handler,
            0.0,
        );

        // Assert something about fluxes when behavior in `calculate_fluxes` is implemented
    }
}
