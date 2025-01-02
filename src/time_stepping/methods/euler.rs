use crate::solver::ksp::SolverManager;
use crate::solver::KSP;
use crate::time_stepping::adaptivity::error_estimate::estimate_error;
use crate::time_stepping::adaptivity::step_size_control::adjust_step_size;
use crate::time_stepping::{TimeStepper, TimeDependentProblem, TimeSteppingError};
use crate::equation::fields::UpdateState;

pub struct ExplicitEuler<P: TimeDependentProblem> {
    current_time: P::Time,
    time_step: P::Time,
    start_time: P::Time,
    end_time: P::Time,
    solver_manager: SolverManager, // Added solver manager
}

impl<P: TimeDependentProblem> ExplicitEuler<P> {
    pub fn new(time_step: P::Time, start_time: P::Time, end_time: P::Time) -> Self {
        Self {
            current_time: start_time,
            time_step,
            start_time,
            end_time,
            solver_manager: todo!(),
        }
    }
}

impl<P> TimeStepper<P> for ExplicitEuler<P>
where
    P: TimeDependentProblem,
    P::State: UpdateState,
    P::Time: From<f64> + Into<f64> + Copy,
{
    fn current_time(&self) -> P::Time {
        self.current_time
    }

    fn set_current_time(&mut self, time: P::Time) {
        self.current_time = time;
    }

    fn step(
        &mut self,
        problem: &P,
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError> {
        let mut derivative = problem.initial_state(); // Initialize derivative
        problem.compute_rhs(current_time, state, &mut derivative)?;

        // Update the state: state = state + dt * derivative
        let dt_f64: f64 = dt.into();
        state.update_state(&derivative, dt_f64);

        self.current_time = current_time + dt;

        Ok(())
    }

    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
        tol: f64,
    ) -> Result<P::Time, TimeSteppingError> {
        let mut error = f64::INFINITY;
        let mut dt = self.time_step.into();
        while error > tol {
            // Compute high-order step
            let mut temp_state = state.clone();
            let mid_dt = P::Time::from(0.5 * dt);
            self.step(problem, mid_dt, self.current_time, &mut temp_state)?;

            // Compute full step for comparison
            let mut high_order_state = temp_state.clone();
            self.step(problem, mid_dt, self.current_time + mid_dt, &mut high_order_state)?;

            error = estimate_error(problem, state, P::Time::from(dt))?;
            dt = adjust_step_size(dt, error, tol, 0.9, 2.0);
        }
        self.set_time_step(P::Time::from(dt));
        Ok(P::Time::from(dt))
    }

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time) {
        self.start_time = start_time;
        self.end_time = end_time;
    }

    fn set_time_step(&mut self, dt: P::Time) {
        self.time_step = dt;
    }

    fn get_time_step(&self) -> P::Time {
        self.time_step
    }
    
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
