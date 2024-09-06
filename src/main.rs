mod mesh_mod;
mod solvers_mod;
mod numerical_mod;
mod time_stepping_mod;
mod transport_mod;

use mesh_mod::mesh_ops::Mesh;
use crate::solvers_mod::linear::LinearSolver;
use crate::time_stepping_mod::explicit_euler::ExplicitEuler;
use crate::time_stepping_mod::base::TimeStepper; // Import the TimeStepper trait

fn main() {
    // Mesh loading and operations
    let mesh_file = "C:/rust_projects/HYDRA/inputs/test.msh2";
    let mut mesh = Mesh::load_from_gmsh(mesh_file).unwrap();
    let mut tol = 0.01;
    let mut max_iter = 100;
    println!("Mesh loaded: {} nodes, {} elements", mesh.nodes.len(), mesh.elements.len());

    // Initialize the linear solver with the mesh
    let mut solver = LinearSolver::new(mesh, tol, max_iter);

    // Create an Explicit Euler time stepper with a time step size of 0.01
    let mut time_stepper = ExplicitEuler::new(0.01);

    // Set the number of time steps (example: 10 steps)
    let num_steps = 10;

    // Time-stepping loop
    for step in 0..num_steps {
        // Perform a single time step
        time_stepper.step(&mut solver, 0.01);

        // Output the current state of the elements after each time step
        for (i, element) in solver.mesh.elements.iter().enumerate() {
            println!("Step {}: Element {} state: {}", step, i, element.state);
        }
    }
}
