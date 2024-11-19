pub fn adjust_step_size(
    current_dt: f64,
    error: f64,
    tol: f64,
    safety_factor: f64,
    growth_factor: f64,
) -> f64 {
    let ratio = (tol / error).powf(0.5); // Using a 2nd-order method assumption
    let new_dt = safety_factor * current_dt * ratio;
    new_dt.clamp(current_dt / growth_factor, current_dt * growth_factor)
}
