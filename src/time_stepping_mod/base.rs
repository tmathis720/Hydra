use crate::mesh_mod::cell_ops;
use crate::solvers_mod::LinearSolver;
use crate::{transport_mod::flux_ops::FluxCalculator, mesh_mod::cell_ops::Cell};

// Time loop function
pub fn time_loop(
    cells: &mut [Cell],
    velocity: f64,
    dt: f64,
    total_time: f64,
    initial_mass: f64,
    initial_momentum: f64,
) {
    let num_steps = (total_time / dt) as usize;  // Time steps

    for step in 0..num_steps {
        // Compute fluxes for all cells
        FluxCalculator::compute_fluxes_cells(cells, velocity);

        // Update mass in each cell
        for cell in cells.iter_mut() {  // Mutably borrow each cell, not move
            cell.update_mass(dt);
            cell.update_momentum(dt);
        }

        // Enforce exact mass conservation after updating
        cell_ops::enforce_mass_conservation(cells, initial_mass);

        // Enforce exact momentum conservation after udpating
        cell_ops::enforce_momentum_conservation(cells, initial_momentum + (step as f64 + 1.0) * initial_mass);

        // Optionally print the current mass for debugging
        let current_mass = Cell::total_mass(cells);
        let current_momentum = Cell::total_momentum(cells);
        print_debugging_info(step, current_mass, initial_mass, current_momentum, initial_momentum);
    }
}

fn print_debugging_info(
    step: usize, 
    mass: f64, 
    total_mass_initial: f64, 
    momentum: f64, 
    total_momentum_initial: f64,
) {
    println!(
        "Step: {}, Current mass: {}, Mass difference (relative): {}%, Current momentum: {}, Momentum difference (relative): {}%",
        step,
        mass,
        ((mass - total_mass_initial) / total_mass_initial) * 100.0,
        momentum,
        ((momentum - total_momentum_initial) / total_momentum_initial) * 100.0
    );
}

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
