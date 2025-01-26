use crate::solver::ksp::SolverManager;
use crate::solver::{GMRES, KSP};
use crate::time_stepping::adaptivity::error_estimate::estimate_error;
use crate::time_stepping::adaptivity::step_size_control::adjust_step_size;
use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
use crate::equation::fields::UpdateState;

/// Implements the Explicit Euler method for solving time-dependent problems.
///
/// The Explicit Euler method is a first-order time-stepping scheme that advances
/// the solution state using the formula:
/// ```text
/// u^{n+1} = u^n + Δt * f(t^n, u^n)
/// ```
/// This method is simple but conditionally stable, requiring careful step size selection.
///
/// # Type Parameters
/// - `P`: The problem type, which must implement `TimeDependentProblem`.
pub struct ExplicitEuler<P: TimeDependentProblem> {
    /// The current simulation time.
    current_time: P::Time,
    /// The time step size.
    time_step: P::Time,
    /// The start time of the simulation.
    start_time: P::Time,
    /// The end time of the simulation.
    end_time: P::Time,
    /// Manages the solver for linear systems (if needed).
    solver_manager: SolverManager,
}

impl<P: TimeDependentProblem> ExplicitEuler<P> {
    /// Creates a new instance of the Explicit Euler time-stepper.
    ///
    /// # Parameters
    /// - `time_step`: The time step size.
    /// - `start_time`: The start time of the simulation.
    /// - `end_time`: The end time of the simulation.
    ///
    /// # Returns
    /// - A new `ExplicitEuler` instance.
    pub fn new(time_step: P::Time, start_time: P::Time, end_time: P::Time) -> Self {
        Self {
            current_time: start_time,
            time_step,
            start_time,
            end_time,
            solver_manager: SolverManager::new(Box::new(GMRES::new(1000, 1e-6, 100))),
        }
    }
}

impl<P> TimeStepper<P> for ExplicitEuler<P>
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

    /// Advances the simulation by one time step using the Explicit Euler method.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem to solve.
    /// - `dt`: The time step size.
    /// - `current_time`: The current simulation time.
    /// - `state`: The state of the system, updated in place.
    ///
    /// # Returns
    /// - `Ok(())` on success, or a `TimeSteppingError` on failure.
    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        // Compute the derivative (RHS of the problem).
        let mut derivative = problem.initial_state();
        problem.compute_rhs(current_time, state, &mut derivative)?;

        // Update the state: state = state + dt * derivative
        let dt_f64: f64 = dt.into();
        state.update_state(&derivative, dt_f64);

        self.current_time = current_time + dt;

        Ok(())
    }

    /// Attempts an adaptive time step by estimating and controlling error.
    ///
    /// The adaptive scheme uses a higher-order estimate to compute error and adjust the step size.
    ///
    /// # Parameters
    /// - `problem`: The time-dependent problem to solve.
    /// - `state`: The current state of the system.
    /// - `tol`: The error tolerance for the step.
    ///
    /// # Returns
    /// - The adjusted step size on success, or a `TimeSteppingError`.
    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
        tol: f64,
    ) -> Result<P::Time, TimeSteppingError> {
        let mut error = f64::INFINITY;
        let mut dt = self.time_step.into();
        while error > tol {
            // Compute a high-order step
            let mut temp_state = state.clone();
            let mid_dt = P::Time::from(0.5 * dt);
            self.step(problem, mid_dt, self.current_time, &mut temp_state)?;

            // Compute the full step for comparison
            let mut high_order_state = temp_state.clone();
            self.step(problem, mid_dt, self.current_time + mid_dt, &mut high_order_state)?;

            // Estimate error and adjust the time step
            error = estimate_error(problem, state, P::Time::from(dt))?;
            dt = adjust_step_size(dt, error, tol, 0.9, 2.0); // Safety factor: 0.9, Max growth: 2.0
        }
        self.set_time_step(P::Time::from(dt));
        Ok(P::Time::from(dt))
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

    /// Sets the time step size for the simulation.
    ///
    /// # Parameters
    /// - `dt`: The new time step size.
    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }

    /// Gets the current time step size.
    ///
    /// # Returns
    /// - The current time step size.
    fn get_time_step(&self) -> P::Time {
        self.time_step
    }

    /// Provides access to the solver used in the time-stepping scheme.
    ///
    /// # Returns
    /// - A mutable reference to the underlying solver.
    fn get_solver(&mut self) -> &mut dyn KSP {
        &mut *self.solver_manager.solver
    }
}

#[cfg(test)]
mod tests {
    use crate::time_stepping::ExplicitEuler;
    use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::section::{Scalar, Section};
    use crate::equation::fields::Fields;

    fn create_test_mesh_entity(id: usize) -> MeshEntity {
        MeshEntity::Vertex(id)
    }

    #[derive(Clone, Debug)]
    struct MockState {
        fields: Fields,
    }

    impl MockState {
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
                    if let Ok(diff_section) = section.clone() - other_section.clone() {
                        diff_fields.scalar_fields.insert(key.clone(), diff_section);
                    }
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
                rhs_section.set_data(entity, Scalar(-value));
                derivative.fields.scalar_fields.insert(key.clone(), rhs_section);
            }
            Ok(())
        }

        fn initial_state(&self) -> Self::State {
            MockState::new(1.0)
        }
        
        fn get_matrix(&self) -> Option<Box<dyn crate::Matrix<Scalar = f64>>> {
            todo!()
        }
        
        fn solve_linear_system(
            &self,
            _matrix: &mut dyn crate::Matrix<Scalar = f64>,
            _state: &mut Self::State,
            _rhs: &Self::State,
        ) -> Result<(), TimeSteppingError> {
            todo!()
        }
    }

    #[test]
    fn test_explicit_euler_step() {
        let mut solver = ExplicitEuler::new(0.1, 0.0, 1.0);
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

        let expected_value = 1.0 - 0.1 * 1.0; // Forward Euler formula: u' = -u
        assert!(
            (value - expected_value).abs() < 1e-5,
            "State value should be updated correctly. Expected: {}, Found: {}",
            expected_value,
            value
        );
    }

    #[test]
    fn test_time_step_set_and_get() {
        // Explicitly specify MockProblem as the TimeDependentProblem type
        let mut solver = ExplicitEuler::<MockProblem>::new(0.1, 0.0, 1.0);
        solver.set_time_step(0.2);
        let time_step: f64 = solver.get_time_step().into(); // Explicit conversion into f64
        assert!(
            (time_step - 0.2).abs() < 1e-5,
            "Time step should be set and retrieved correctly. Expected: 0.2, Found: {}",
            time_step
        );
    }

    #[test]
    fn test_time_interval() {
        // Explicitly specify MockProblem as the TimeDependentProblem type
        let mut solver = ExplicitEuler::<MockProblem>::new(0.1, 0.0, 1.0);
        solver.set_time_interval(0.5.into(), 1.0.into());
        let start_time: f64 = solver.start_time.into(); // Explicit conversion into f64
        let end_time: f64 = solver.end_time.into(); // Explicit conversion into f64
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
