use crate::solvers_mod::LinearSolver;

/// Base trait for time-stepping schemes
pub trait TimeStepper {
    /// Advance the system by a single time step
    fn step(&mut self, solver: &mut LinearSolver, dt: f64);

    /// Run the time-stepping process for a given number of steps
    fn run(&mut self, solver: &mut LinearSolver, dt: f64, num_steps: usize) {
        for _ in 0..num_steps {
            self.step(solver, dt);
        }
    }
}
