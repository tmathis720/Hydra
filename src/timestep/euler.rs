use crate::domain::element::Element;
use crate::transport::flux;

// Time loop function
pub fn time_loop(
    elements: &mut [Element],
    velocity: f64,
    dt: f64,
    total_time: f64,
    initial_mass: f64,
    initial_momentum: f64,
) {
    let num_steps = (total_time / dt) as usize;  // Time steps

    for step in 0..num_steps {
        // Compute fluxes for all elements
        flux::compute_fluxes(elements, velocity);

        // Update mass in each element
        for element in elements.iter_mut() {  // Mutably borrow each element, not move
            element.update_mass(dt);
            element.update_momentum(dt);
        }

        // Enforce exact mass conservation after updating
        Element::enforce_mass_conservation(elements, initial_mass);

        // Enforce exact momentum conservation after udpating
        Element::enforce_momentum_conservation(elements, initial_momentum + (step as f64 + 1.0) * initial_mass);

        // Optionally print the current mass for debugging
        let current_mass = Element::total_mass(elements);
        let current_momentum = Element::total_momentum(elements);
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