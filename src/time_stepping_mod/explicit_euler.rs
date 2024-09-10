use crate::solvers_mod::LinearSolver;  // Correcting the import here
use crate::time_stepping_mod::base::TimeStepper;

pub struct ExplicitEuler {
    pub dt: f64, // time step size
}

impl ExplicitEuler {
    // Constructor to create a new ExplicitEuler with a time step size
    pub fn new(dt: f64) -> Self {
        ExplicitEuler { dt }
    }
}

impl TimeStepper for ExplicitEuler {
    fn step(&mut self, solver: &mut LinearSolver, _dt: f64) {
        let fluxes = solver.compute_fluxes();
        solver.update_states(fluxes, self.dt);  // Use the ExplicitEuler's time step here
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_mod::mesh_ops::Mesh;
    use crate::solvers_mod::linear::LinearSolver;

/*     #[test]
    fn test_explicit_euler_step() {
        // Setup a simple mesh
        let mut mesh = Mesh::new();
        let mut tol = 0.01;
        let mut max_iter = 100;
        mesh.add_node(1, 0.0, 0.0, 0.0);
        mesh.add_node(2, 1.0, 0.0, 0.0);
        mesh.add_node(3, 1.0, 1.0, 0.0);
        mesh.add_element(1, [1, 2, 3], vec![1]);
        mesh.elements[0].state = 1.0;  // Initialize state

        // Create solver and time stepper
        let mut solver = LinearSolver::new(mesh, tol, max_iter);
        let mut euler = ExplicitEuler::new(0.01);  // Initialize with a time step of 0.01

        // Step the solver forward in time
        euler.step(&mut solver, 0.01);  // Use the time step in the Euler method

        // Check if the state has been updated
        assert!(solver.mesh.elements[0].state > 1.0, "State should have increased");
    } */
}
