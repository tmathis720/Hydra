use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
use crate::equation::fields::UpdateState;

/// Implements the Crank-Nicolson time-stepping method.
///
/// The Crank-Nicolson method is a second-order accurate, implicit time-stepping scheme.
/// It calculates the solution by averaging the contributions of the explicit (Forward Euler)
/// and implicit (Backward Euler) methods. This makes it both accurate and stable for stiff problems.
///
/// # Key Features
/// - Second-order accuracy in time.
/// - Suitable for stiff systems due to implicitness.
/// - Solves a linear system of equations at each step.
///
/// # Parameters
/// - `P`: The time-dependent problem to be solved, implementing `TimeDependentProblem`.
pub struct CrankNicolson<P: TimeDependentProblem> {
    /// The current simulation time.
    current_time: P::Time,
    /// The time step size.
    time_step: P::Time,
    /// The start time of the simulation.
    start_time: P::Time,
    /// The end time of the simulation.
    end_time: P::Time,
}

impl<P: TimeDependentProblem> CrankNicolson<P> {
    /// Creates a new instance of the Crank-Nicolson time-stepper.
    ///
    /// # Parameters
    /// - `time_step`: The size of each time step.
    /// - `start_time`: The initial simulation time.
    /// - `end_time`: The final simulation time.
    pub fn new(time_step: P::Time, start_time: P::Time, end_time: P::Time) -> Self {
        Self {
            current_time: start_time,
            time_step,
            start_time,
            end_time,
        }
    }
}

impl<P> TimeStepper<P> for CrankNicolson<P>
where
    P: TimeDependentProblem,
    P::State: UpdateState,
    P::Time: From<f64> + Into<f64> + Copy,
{
    /// Returns the current simulation time.
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    /// Updates the current simulation time.
    ///
    /// # Parameters
    /// - `time`: The new simulation time.
    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    /// Advances the solution by one time step using the Crank-Nicolson method.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem to solve.
    /// - `dt`: The time step size.
    /// - `current_time`: The current simulation time.
    /// - `state`: The current state of the system, updated in place.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `TimeSteppingError` if the problem fails to compute the right-hand side (RHS)
    ///   or solve the linear system.
    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let dt_f64: f64 = dt.into();
        self.time_step = dt;

        // Compute the explicit contribution (RHS at the current state)
        let mut rhs_explicit = state.clone();
        problem.compute_rhs(current_time, state, &mut rhs_explicit)?;

        // Compute the weighted RHS for the implicit system: RHS = state + 0.5 * dt * (explicit + implicit contributions)
        let mut rhs_combined = state.clone();
        rhs_combined.update_state(&rhs_explicit, 0.5 * dt_f64);

        let mut matrix = problem
            .get_matrix()
            .ok_or(TimeSteppingError::SolverError(
                "Matrix is required for Crank-Nicolson.".into(),
            ))?;

        // Solve the implicit system: A * state = RHS
        problem.solve_linear_system(matrix.as_mut(), state, &rhs_combined)?;

        // Update the current time
        self.current_time = current_time + dt;

        Ok(())
    }

    /// Adaptive time-stepping is not implemented for Crank-Nicolson.
    ///
    /// # Returns
    /// - `Err(TimeSteppingError::InvalidStep)` indicating the feature is unavailable.
    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
        _tol: f64,
    ) -> Result<P::Time, TimeSteppingError> {
        Err(TimeSteppingError::InvalidStep)
    }

    /// Sets the time interval for the simulation.
    ///
    /// # Parameters
    /// - `start_time`: The start time of the simulation.
    /// - `end_time`: The end time of the simulation.
    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time) {
        self.start_time = start_time;
        self.end_time = end_time;
    }

    /// Sets the time step size.
    ///
    /// # Parameters
    /// - `dt`: The new time step size.
    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }

    /// Returns the current time step size.
    ///
    /// # Returns
    /// - The current time step size.
    fn get_time_step(&self) -> P::Time {
        self.time_step
    }

    /// Placeholder method for accessing the solver.
    ///
    /// # Returns
    /// - A mutable reference to the solver.
    ///
    /// # Note
    /// This method is currently unimplemented.
    fn get_solver(&mut self) -> &mut dyn crate::solver::KSP {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::matrix::MatrixOperations;
    use crate::time_stepping::methods::crank_nicolson::CrankNicolson;
    use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::section::{Scalar, Section};
    use crate::equation::fields::Fields;

    /// Creates a mock `MeshEntity` for testing purposes.
    ///
    /// # Parameters
    /// - `id`: Identifier for the `MeshEntity`.
    ///
    /// # Returns
    /// - A new `MeshEntity` representing a vertex.
    fn create_test_mesh_entity(id: usize) -> MeshEntity {
        MeshEntity::Vertex(id)
    }

    /// Mock state implementation for testing.
    #[derive(Clone, Debug)]
    struct MockState {
        fields: Fields,
    }

    impl MockState {
        /// Creates a new `MockState` with a scalar field initialized to a given value.
        ///
        /// # Parameters
        /// - `value`: Initial value for the scalar field.
        ///
        /// # Returns
        /// - A new instance of `MockState`.
        fn new(value: f64) -> Self {
            let mut fields = Fields::new();
            let section = Section::new();
            let entity = create_test_mesh_entity(1);
            section.set_data(entity, Scalar(value));
            fields.scalar_fields.insert("value".to_string(), section);
            MockState { fields }
        }
    }

    impl crate::equation::fields::UpdateState for MockState {
        fn compute_residual(&self, rhs: &Self) -> f64 {
            self.fields
                .scalar_fields
                .iter()
                .map(|(key, section)| {
                    let rhs_section = rhs.fields.scalar_fields.get(key).unwrap();
                    section
                        .all_data()
                        .iter()
                        .zip(rhs_section.all_data().iter())
                        .map(|(s, r)| (s.0 - r.0).abs())
                        .sum::<f64>()
                })
                .sum()
        }

        fn update_state(&mut self, derivative: &Self, dt: f64) {
            for (key, section) in &derivative.fields.scalar_fields {
                self.fields.scalar_fields.entry(key.clone()).and_modify(|val| {
                    val.update_with_derivative(section, dt);
                });
            }
        }

        fn difference(&self, other: &Self) -> Self {
            let mut diff_fields = Fields::new();
            for (key, section) in &self.fields.scalar_fields {
                if let Some(other_section) = other.fields.scalar_fields.get(key) {
                    diff_fields.scalar_fields.insert(
                        key.clone(),
                        section.clone() - other_section.clone(),
                    );
                }
            }
            MockState { fields: diff_fields }
        }

        fn norm(&self) -> f64 {
            self.fields
                .scalar_fields
                .values()
                .map(|section| section.all_data().iter().map(|s| s.0.abs()).sum::<f64>())
                .sum()
        }
    }

    /// Mock problem implementation for testing the Crank-Nicolson method.
    struct MockProblem;

    impl TimeDependentProblem for MockProblem {
        type State = MockState;
        type Time = f64;

        fn compute_rhs(
            &self,
            _time: Self::Time,
            state: &Self::State,
            derivative: &mut Self::State,
        ) -> Result<(), TimeSteppingError> {
            for (key, section) in &state.fields.scalar_fields {
                let value = section.all_data().first().unwrap_or(&Scalar(0.0)).0;
                let rhs_section = Section::new();
                let entity = create_test_mesh_entity(1);
                rhs_section.set_data(entity, Scalar(-value)); // Example: u' = -u
                derivative.fields.scalar_fields.insert(key.clone(), rhs_section);
            }
            Ok(())
        }

        fn initial_state(&self) -> Self::State {
            MockState::new(1.0)
        }

        fn get_matrix(&self) -> Option<Box<dyn crate::linalg::matrix::Matrix<Scalar = f64>>> {
            Some(Box::new(crate::linalg::matrix::dense_matrix::DenseMatrix::construct(1, 1)))
                }

        fn solve_linear_system(
            &self,
            _matrix: &mut dyn crate::Matrix<Scalar = f64>,
            state: &mut Self::State,
            rhs: &Self::State,
        ) -> Result<(), TimeSteppingError> {
            for (key, rhs_section) in &rhs.fields.scalar_fields {
                let rhs_value = rhs_section.all_data().first().unwrap_or(&Scalar(0.0)).0;
                let result_section = Section::new();
                let entity = create_test_mesh_entity(1);
                let dt = 0.1;
                result_section.set_data(entity, Scalar(rhs_value / (1.0 + 0.5 * dt)));
                state.fields.scalar_fields.insert(key.clone(), result_section);
            }
            Ok(())
        }
    }

    #[test]
    fn test_crank_nicolson_step() {
        let mut solver = CrankNicolson::new(0.1, 0.0, 1.0);
        let problem = MockProblem;
        let mut state = problem.initial_state();

        assert!(solver
            .step(&problem, 0.1, 0.0, &mut state)
            .is_ok());

        let value = state
            .fields
            .scalar_fields
            .get("value")
            .unwrap()
            .all_data()
            .first()
            .unwrap()
            .0;

        let expected_value = 1.0 / (1.0 + 0.05); // Crank-Nicolson formula for u' = -u
        assert!(
            (value - expected_value).abs() < 1e-1,
            "State value should be updated correctly. Expected: {}, Found: {}",
            expected_value,
            value
        );
    }

    #[test]
    fn test_time_step_set_and_get() {
        let mut solver = CrankNicolson::<MockProblem>::new(0.1, 0.0, 1.0);
        solver.set_time_step(0.2);
        let time_step: f64 = solver.get_time_step().into();
        assert!(
            (time_step - 0.2).abs() < 1e-5,
            "Time step should be set and retrieved correctly. Expected: 0.2, Found: {}",
            time_step
        );
    }

    #[test]
    fn test_time_interval() {
        let mut solver = CrankNicolson::<MockProblem>::new(0.1, 0.0, 1.0);
        solver.set_time_interval(0.5.into(), 1.0.into());
        let start_time: f64 = solver.start_time.into();
        let end_time: f64 = solver.end_time.into();
        assert!(
            (start_time - 0.5).abs() < 1e-5,
            "Start time should be set correctly. Expected: 0.5, Found: {}",
            start_time
        );
        assert!(
            (end_time - 1.0).abs() < 1e-5,
            "End time should be set correctly. Expected: 1.0, Found: {}",
            end_time
        );
    }
}
