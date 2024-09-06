// File: tests/solver_integration.rs

use hydra::mesh_mod::mesh_ops::Mesh;
use hydra::solvers_mod::linear::LinearSolver;

#[test]
fn test_solver_with_time_stepper() {
    let mut mesh = Mesh::new();
    mesh.add_node(1, 0.0, 0.0, 0.0);
    mesh.add_node(2, 1.0, 0.0, 0.0);
    mesh.add_element(1, [1, 2], vec![1]);
    mesh.elements[0].state = 1.0;

    let mut solver = LinearSolver::new(mesh);
    let mut euler = ExplicitEuler::new(0.01);

    let new_states = solver.compute_fluxes();
    euler.step(&mut solver, new_states);

    assert!(solver.mesh.elements[0].state > 1.0);
}
