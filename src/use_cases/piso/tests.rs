#[cfg(test)]
mod piso_tests {
    use crate::{
        boundary::bc_handler::{BoundaryCondition, BoundaryConditionHandler}, domain::{mesh::Mesh, section::{Scalar, Vector3}}, equation::{fields::{Fields, Fluxes}, momentum_equation::MomentumEquation}, interface_adapters::domain_adapter::DomainBuilder, linalg::Matrix, solver::ksp::{create_solver, SolverManager, SolverType}, time_stepping::{ts::FixedTimeStepper, TimeDependentProblem, TimeSteppingError}, use_cases::piso::{
            nonlinear_loop::{solve_nonlinear_system, NonlinearLoopConfig}, predictor::predict_velocity, pressure_correction::solve_pressure_poisson, velocity_correction::correct_velocity, PISOConfig, PISOSolver
        }
    };

    /// Dummy `ExampleProblem` implementing `TimeDependentProblem`
    struct ExampleProblem;

    impl TimeDependentProblem for ExampleProblem {
        type State = Fields;
        type Time = f64;

        fn compute_rhs(
            &self,
            _time: Self::Time,
            _state: &Self::State,
            _derivative: &mut Self::State,
        ) -> Result<(), TimeSteppingError> {
            Ok(())
        }

        fn initial_state(&self) -> Self::State {
            Fields::new()
        }

        fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
            None
        }

        fn solve_linear_system(
            &self,
            _matrix: &mut dyn Matrix<Scalar = f64>,
            _state: &mut Self::State,
            _rhs: &Self::State,
        ) -> Result<(), TimeSteppingError> {
            Ok(())
        }
    }

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

    fn setup_fields(mesh: &Mesh) -> Fields {
        let mut fields = Fields::new();

        let cell_ids = mesh.get_cells().iter().map(|cell| cell.clone()).collect::<Vec<_>>();

        let cell_a = cell_ids[0].clone();
        fields.set_scalar_field_value("velocity_x", cell_a.clone(), Scalar(1.0));
        fields.set_scalar_field_value("velocity_y", cell_a.clone(), Scalar(2.0));
        fields.set_scalar_field_value("velocity_z", cell_a.clone(), Scalar(3.0));
        fields.set_scalar_field_value("pressure", cell_a.clone(), Scalar(100.0));
        fields.set_vector_field_value("velocity", cell_a.clone(), Vector3([1.0, 2.0, 3.0]));

        let cell_b = cell_ids[1].clone();
        fields.set_scalar_field_value("velocity_x", cell_b.clone(), Scalar(4.0));
        fields.set_scalar_field_value("velocity_y", cell_b.clone(), Scalar(5.0));
        fields.set_scalar_field_value("velocity_z", cell_b.clone(), Scalar(6.0));
        fields.set_scalar_field_value("pressure", cell_b.clone(), Scalar(50.0));
        fields.set_vector_field_value("velocity", cell_b.clone(), Vector3([4.0, 5.0, 6.0]));

        fields
    }

    fn setup_boundary_conditions(mesh: &Mesh) -> BoundaryConditionHandler {
        let bc_handler = BoundaryConditionHandler::new();
        let face = mesh.get_faces().iter().next().cloned().expect("No face found in mesh");
        bc_handler.set_bc(face, BoundaryCondition::Dirichlet(10.0));
        bc_handler
    }

    fn setup_solver() -> SolverManager {
        let solver = create_solver(SolverType::ConjugateGradient, 100, 1e-6, 0);
        SolverManager::new(solver)
    }

    #[test]
    fn test_piso_predictor() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let mut fluxes = Fluxes::new();
        let bc_handler = setup_boundary_conditions(&mesh);

        let momentum_eq = MomentumEquation::with_parameters(1.0, 0.001);

        // Execute predictor step
        let result = predict_velocity(&mesh, &mut fields, &mut fluxes, &bc_handler, &momentum_eq, 0.0);
        assert!(result.is_ok(), "Predictor step failed.");
    }

    #[test]
    fn test_pressure_correction() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let fluxes = Fluxes::new();
        let bc_handler = setup_boundary_conditions(&mesh);

        let mut solver_manager = setup_solver();

        // Execute pressure correction
        let result = solve_pressure_poisson(
            &mesh,
            &mut fields,
            &fluxes,
            &bc_handler,
            solver_manager.solver.as_mut(),
        );

        assert!(result.is_ok(), "Pressure correction step failed.");
        let pressure_correction = result.unwrap();
        assert!(pressure_correction.residual < 1e-6, "Pressure correction did not converge.");
    }

    #[test]
    fn test_velocity_correction() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let bc_handler = setup_boundary_conditions(&mesh);

        let pressure_correction = fields.scalar_fields.get("pressure").unwrap().clone();

        // Execute velocity correction
        let result = correct_velocity(&mesh, &mut fields, &pressure_correction, &bc_handler);
        assert!(result.is_ok(), "Velocity correction step failed.");
    }

    #[test]
    fn test_nonlinear_loop_convergence() {
        let mesh = setup_simple_mesh();
        let mut fields = setup_fields(&mesh);
        let bc_handler = setup_boundary_conditions(&mesh);

        let mut solver_manager = setup_solver();
        let config = NonlinearLoopConfig {
            max_iterations: 50,
            tolerance: 1e-6,
        };

        // Execute nonlinear loop
        let result = solve_nonlinear_system(
            &mesh,
            &mut fields,
            &bc_handler,
            solver_manager.solver.as_mut(),
            &config,
        );

        assert!(result.is_ok(), "Nonlinear loop did not converge.");
    }

    #[test]
    fn test_piso_solver_full_cycle() {
        let mesh = setup_simple_mesh();
        let fields = setup_fields(&mesh);
        let _bc_handler = setup_boundary_conditions(&mesh);

        let solver_manager = setup_solver();

        let time_stepper = FixedTimeStepper::new(0.0, 1.0, 0.1, solver_manager.solver);
        let mut piso_solver = PISOSolver::new(
            mesh.clone(),
            Box::new(time_stepper),
            PISOConfig {
                max_iterations: 10,
                tolerance: 1e-6,
                relaxation_factor: 0.7,
            },
        );

        // Execute a full PISO cycle
        let mut state = fields.clone();
        let problem = ExampleProblem {};
        let result = piso_solver.solve(&problem, &mut state);
        assert!(result.is_ok(), "PISO solver failed.");
    }
}
