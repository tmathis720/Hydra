mod domain;
mod numerical;
mod solver;
mod timestep;
mod transport;

use crate::domain::element::Element;

fn main() {
    let num_cells = 10;
    let length = 10.0;
    let velocity = 1.0;  // Constant velocity for advection
    let dt = 0.1;  // Time step
    let total_time = 1.0;  // Total simulation time

    // Initialize the domain
    let mut cells = Element::initialize_domain(num_cells, length, velocity);

    // Compute the initial total mass
    let initial_mass = Element::total_mass(&cells);
    let initial_momentum = Element::total_momentum(&cells);
    println!("Initial total mass: {}", initial_mass);
    println!("Initial total momentum: {}", initial_momentum);

    // Expected outcomes
    let expected_mass = 10.0; // Mass should remain constant
    let expected_momentum = initial_momentum + expected_momentum_increase(num_cells, velocity, dt, total_time);

    // Run the time loop
    timestep::euler::time_loop(&mut cells, velocity, dt, total_time, initial_mass, initial_momentum);

    // Compute the final total mass
    let final_mass = Element::total_mass(&cells);
    let final_momentum = Element::total_momentum(&cells);
    println!("Final total mass: {}", final_mass);
    println!("Final total momentum: {}", final_momentum);

    // Check if mass is conserved
    assert!(
        (expected_mass - final_mass).abs() < 1e-6,
        "Mass is NOT conserved! Expected: {}, Got: {}",
        expected_mass,
        final_mass
    );
    println!("Mass is conserved!");

    // Check if momentum matches the expected outcome
    assert!(
        (expected_momentum - final_momentum).abs() < 1e-6,
        "Momentum is NOT correct! Expected: {}, Got: {}",
        expected_momentum,
        final_momentum
    );
    println!("Momentum is correct!");
}

// Function to calculate expected momentum increase
fn expected_momentum_increase(num_cells: usize, velocity: f64, dt: f64, total_time: f64) -> f64 {
    let mass_per_cell = 1.0;  // Each cell has a mass of 1.0 initially
    let num_steps = (total_time / dt) as usize;

    // Total momentum is mass * velocity * number of steps
    num_steps as f64 * mass_per_cell * velocity * num_cells as f64
}