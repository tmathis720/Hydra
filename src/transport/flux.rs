use crate::domain::element::Element;

// This assumes a simple advection model (constant velocity)
pub fn compute_fluxes(elements: &mut [Element], velocity: f64) {
    let num_cells = elements.len();

    for i in 0..num_cells {
        // Reflective boundary at the left
        if i == 0 {
            // Reflect mass flux without adding or removing mass
            elements[i].flux_left = -elements[i].flux_right;  // Reflect mass and momentum perfectly
        } else {
            elements[i].flux_left = velocity * elements[i - 1].density;  // Mass flux between elements
        }

        // Reflective boundary at the right
        if i == num_cells - 1 {
            // Reflect mass flux perfectly at the boundary
            elements[i].flux_right = -elements[i].flux_left;  // Reflect mass and momentum perfectly
        } else {
            elements[i].flux_right = velocity * elements[i].density;  // Mass flux between elements
        }

        // Momentum flux = mass flux * velocity (momentum = mass * velocity)
        if i > 0 {
            let mass_flux = velocity * elements[i - 1].density;
            let momentum_flux = mass_flux * velocity;
            elements[i].momentum_x += momentum_flux * elements[i].volume;
        }
    }
}