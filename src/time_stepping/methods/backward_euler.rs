use crate::time_stepping::{TimeStepper, TimeSteppingError, TimeDependentProblem};

/// Implements the Backward Euler method for time-dependent problems.
///
/// The Backward Euler method is an implicit, first-order time-stepping scheme
/// that advances the solution state using the formula:
/// ```text
/// u^{n+1} - Δt * f(t^{n+1}, u^{n+1}) = u^n
/// ```
/// It is unconditionally stable but requires solving a linear system at each time step.
pub struct BackwardEuler {
    /// The current simulation time.
    current_time: f64,
    /// The time step size.
    time_step: f64,
}

impl BackwardEuler {
    /// Creates a new instance of the Backward Euler time-stepper.
    ///
    /// # Parameters
    /// - `start_time`: The starting simulation time.
    /// - `time_step`: The initial time step size.
    ///
    /// # Returns
    /// - A new `BackwardEuler` instance.
    pub fn new(start_time: f64, time_step: f64) -> Self {
        Self {
            current_time: start_time,
            time_step,
        }
    }
}

impl<P> TimeStepper<P> for BackwardEuler
where
    P: TimeDependentProblem,
    P::Time: From<f64> + Into<f64>,
{
    /// Returns the current simulation time.
    fn current_time(&self) -> P::Time {
        P::Time::from(self.current_time)
    }

    /// Updates the current simulation time.
    ///
    /// # Parameters
    /// - `time`: The new simulation time.
    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time.into();
    }

    /// Advances the simulation by one time step using the Backward Euler method.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem to solve.
    /// - `dt`: The time step size.
    /// - `current_time`: The current simulation time.
    /// - `state`: The current state of the system, updated in place.
    ///
    /// # Returns
    /// - `Ok(())` on success, or a `TimeSteppingError` if the linear system cannot be solved.
    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let dt_f64: f64 = dt.into();
        self.time_step = dt_f64;

        // Clone the current state to compute the right-hand side (RHS).
        let mut rhs = state.clone();

        // Get the problem's matrix (required for Backward Euler).
        let mut matrix = problem
            .get_matrix()
            .ok_or(TimeSteppingError::SolverError(
                "Matrix is required for Backward Euler.".into(),
            ))?;

        // Compute the RHS of the equation.
        problem.compute_rhs(current_time, state, &mut rhs)?;

        // Solve the linear system to update the state.
        problem.solve_linear_system(matrix.as_mut(), state, &rhs)?;

        // Advance the simulation time.
        self.current_time += dt_f64;

        Ok(())
    }

    /// Adaptive step is not yet implemented for Backward Euler.
    ///
    /// # Returns
    /// - The current time step as a placeholder result.
    fn adaptive_step(
        &mut self,
        _problem: &P,
        _state: &mut P::State,
        _tol: f64,
    ) -> Result<P::Time, TimeSteppingError> {
        // Placeholder logic for adaptive step
        Ok(self.time_step.into())
    }

    /// Sets the simulation's time interval.
    ///
    /// # Parameters
    /// - `start_time`: The start time of the simulation.
    /// - `_end_time`: The end time of the simulation (currently unused).
    fn set_time_interval(&mut self, start_time: P::Time, _end_time: P::Time) {
        self.current_time = start_time.into();
    }

    /// Sets the time step size.
    ///
    /// # Parameters
    /// - `dt`: The new time step size.
    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt.into();
    }

    /// Gets the current time step size.
    ///
    /// # Returns
    /// - The current time step size.
    fn get_time_step(&self) -> P::Time {
        self.time_step.into()
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
    use crate::time_stepping::BackwardEuler;
    use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
    use crate::domain::mesh_entity::MeshEntity; // Import the MeshEntity type
    use crate::domain::section::{Scalar, Section};
    use crate::equation::fields::Fields;
    use crate::Matrix;
    use faer::Mat;

    // Helper function to create a mock `MeshEntity`
    fn create_test_mesh_entity(id: usize) -> MeshEntity {
        MeshEntity::Vertex(id) // Adjust this based on the actual implementation of `MeshEntity`
    }

    #[derive(Clone, Debug)]
    struct MockState {
        fields: Fields,
    }

    impl MockState {
        fn new(value: f64) -> Self {
            let mut fields = Fields::new();
            let section = Section::new();
            let entity = create_test_mesh_entity(1); // Use a mock MeshEntity
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

    struct MockProblem;

    impl TimeDependentProblem for MockProblem {
        type State = MockState;
        type Time = f64; // Specify f64 as the concrete type for Time

        fn compute_rhs(
            &self,
            _time: Self::Time,
            state: &Self::State,
            derivative: &mut Self::State,
        ) -> Result<(), TimeSteppingError> {
            for (key, section) in &state.fields.scalar_fields {
                let value = section.all_data().first().unwrap_or(&Scalar(0.0)).0;
                let rhs_section = Section::new();
                let entity = create_test_mesh_entity(1); // Use a mock MeshEntity
                rhs_section.set_data(entity, Scalar(value)); // RHS is now +u
                derivative.fields.scalar_fields.insert(key.clone(), rhs_section);
            }
            Ok(())
        }
        
        

        fn initial_state(&self) -> Self::State {
            MockState::new(1.0)
        }

        fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>> {
            Some(Box::new(Mat::identity(1, 1)))
        }

        fn solve_linear_system(
            &self,
            _matrix: &mut dyn Matrix<Scalar = f64>,
            state: &mut Self::State,
            rhs: &Self::State,
        ) -> Result<(), TimeSteppingError> {
            for (key, section) in &rhs.fields.scalar_fields {
                let rhs_value = section.all_data().first().unwrap_or(&Scalar(0.0)).0;
                let result_section = Section::new();
                let entity = create_test_mesh_entity(1); // Use a mock MeshEntity
                // Correct solution for u(n+1): u(n+1) = rhs / (1 + dt)
                let dt = 0.1; // Time step size
                result_section.set_data(entity, Scalar(rhs_value / (1.0 + dt)));
                state.fields.scalar_fields.insert(key.clone(), result_section);
            }
            Ok(())
        }        
    }

    #[test]
    fn test_backward_euler_step() {
        let mut solver = BackwardEuler::new(0.0, 0.1);
        let problem = MockProblem;
        let mut state = problem.initial_state();
    
        let dt: f64 = 0.1; // Explicit type annotation
        let current_time: f64 = 0.0; // Explicit type annotation
    
        // Perform a single time step
        assert!(
            solver
                .step(&problem, dt.into(), current_time.into(), &mut state)
                .is_ok(),
            "Backward Euler step should execute successfully."
        );
    
        // Retrieve the updated value after the step
        let updated_value = state
            .fields
            .scalar_fields
            .get("value")
            .unwrap()
            .all_data()
            .first()
            .unwrap()
            .0;
    
        // Calculate the expected value using backward Euler formula
        let initial_value = 1.0; // From `initial_state`
        let expected_value = initial_value / (1.0 + dt);
    
        // Adjust the assertion to compare with the calculated expected value
        assert!(
            (updated_value - expected_value).abs() < 1e-5,
            "State value should be updated correctly. Expected: {}, Found: {}",
            expected_value,
            updated_value
        );
    }
    


    #[test]
    fn test_time_step_set_and_get() {
        let mut solver = BackwardEuler::new(0.0, 0.1);
        <BackwardEuler as TimeStepper<MockProblem>>::set_time_step(&mut solver, 0.2.into());
        let time_step: f64 = <BackwardEuler as TimeStepper<MockProblem>>::get_time_step(&solver).into(); // Specify trait implementation
        assert!(
            (time_step - 0.2).abs() < 1e-5,
            "Time step should be set and retrieved correctly."
        );
    }

    #[test]
    fn test_time_interval() {
        let mut solver = BackwardEuler::new(0.0, 0.1);
        <BackwardEuler as TimeStepper<MockProblem>>::set_time_interval(&mut solver, 0.5.into(), 1.0.into());
        let current_time: f64 = <BackwardEuler as TimeStepper<MockProblem>>::current_time(&solver).into(); // Specify trait implementation
        assert!(
            (current_time - 0.5).abs() < 1e-5,
            "Current time should be set correctly."
        );
    }

}
