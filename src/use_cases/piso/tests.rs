#[cfg(test)]
mod tests {
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler},
        domain::{mesh::Mesh, section::{Scalar, Vector3}},
        equation::fields::{Fields, Fluxes},
        solver::ksp::{create_solver, SolverType},
        time_stepping::{ts::FixedTimeStepper, TimeDependentProblem, TimeSteppingError},
        use_cases::piso::{predictor, pressure_correction, velocity_correction, PISOConfig, PISOSolver},
        Matrix, MeshEntity, Section,
    };

    /// Setup a simple tetrahedral mesh for testing.
    fn setup_simple_mesh() -> Mesh {
        let mut builder = crate::interface_adapters::domain_adapter::DomainBuilder::new();
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

    /// Setup fields with scalar and vector data for a mesh.
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

    /// Setup boundary conditions for a mesh.
    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let bc_handler = BoundaryConditionHandler::new();
        let face = mesh.entities.read().unwrap().iter()
            .find(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .expect("No face found in mesh");

        bc_handler.set_bc(face, BoundaryCondition::Dirichlet(10.0));
        bc_handler
    }

    /// Setup the PISO configuration for tests.
    fn setup_piso_config() -> PISOConfig {
        PISOConfig {
            max_iterations: 10,
            tolerance: 1e-5,
            relaxation_factor: 0.7,
        }
    }

    /// Tests the PISO solver's constructor.
    #[test]
    fn test_piso_constructor() {
        let mesh = setup_simple_mesh();
        let solver = create_solver(SolverType::ConjugateGradient, 100, 1e-6, 0);
        let time_stepper = FixedTimeStepper::<MockProblem>::new(0.0.into(), 1.0.into(), 0.1.into(), solver);
        let config = setup_piso_config();

        struct MockProblem;

        impl TimeDependentProblem for MockProblem {
            type State = Fields;
            type Time = f64;

            fn compute_rhs(&self, _time: Self::Time, _state: &Self::State, _derivative: &mut Self::State) -> Result<(), TimeSteppingError> {
                Ok(())
            }

            fn initial_state(&self) -> Self::State {
                Fields::new()
            }

            fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
                None
            }

            fn solve_linear_system(&self, _matrix: &mut dyn Matrix<Scalar = f64>, _state: &mut Self::State, _rhs: &Self::State) -> Result<(), TimeSteppingError> {
                Ok(())
            }
        }

        let piso_solver = PISOSolver::new(mesh, Box::new(time_stepper), config);

        assert!(piso_solver.config.max_iterations > 0);
        assert_eq!(piso_solver.config.relaxation_factor, 0.7);
    }

    /// Tests the predictor step.
    #[test]
    fn test_predictor_step() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);
        let mut fluxes = Fluxes::new();
        let momentum_equation = crate::equation::momentum_equation::MomentumEquation::with_parameters(1.0, 0.001);

        let result = predictor::predict_velocity(
            &mesh,
            &mut fields,
            &mut fluxes,
            &boundary_handler,
            &momentum_equation,
            0.0,
        );

        assert!(result.is_ok(), "Predictor step failed: {:?}", result.err());
    }

    /// Tests the pressure correction step.
    #[test]
    fn test_pressure_correction_step() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);
        let fluxes = Fluxes::new();
        let mut solver = create_solver(SolverType::GMRES, 100, 1e-6, 50);

        let result = pressure_correction::solve_pressure_poisson(
            &mesh,
            &mut fields,
            &fluxes,
            &boundary_handler,
            &mut *solver,
        );

        assert!(result.is_ok(), "Pressure correction failed: {:?}", result.err());
        let residual = result.unwrap().residual;
        assert!(residual < 1e-5, "Residual too high: {}", residual);
    }

    /// Tests the velocity correction step.
    #[test]
    fn test_velocity_correction_step() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let boundary_handler = setup_boundary_conditions(&mesh);
        let pressure_correction = Section::<Scalar>::new();

        let result = velocity_correction::correct_velocity(
            &mesh,
            &mut fields,
            &pressure_correction,
            &boundary_handler,
        );

        assert!(result.is_ok(), "Velocity correction failed: {:?}", result.err());
    }

    /// Integration test for the full PISO solver.
    #[test]
    fn test_piso_solver_integration() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let _boundary_handler = setup_boundary_conditions(&mesh);
        let config = setup_piso_config();
        let solver = create_solver(SolverType::ConjugateGradient, 100, 1e-6, 0);
        let time_stepper = FixedTimeStepper::new(0.0.into(), 1.0.into(), 0.1.into(), solver);
        let mut piso_solver = PISOSolver::new(mesh, Box::new(time_stepper), config);

        struct MockProblem;

        impl TimeDependentProblem for MockProblem {
            type State = Fields;
            type Time = f64;

            fn compute_rhs(&self, _time: Self::Time, _state: &Self::State, _derivative: &mut Self::State) -> Result<(), TimeSteppingError> {
                Ok(())
            }

            fn initial_state(&self) -> Self::State {
                Fields::new()
            }

            fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
                None
            }

            fn solve_linear_system(&self, _matrix: &mut dyn Matrix<Scalar = f64>, _state: &mut Self::State, _rhs: &Self::State) -> Result<(), TimeSteppingError> {
                Ok(())
            }
        }

        let problem = MockProblem;

        let result = piso_solver.solve(&problem, &mut fields);
        assert!(result.is_ok(), "PISO solver failed: {:?}", result.err());
    }
}
