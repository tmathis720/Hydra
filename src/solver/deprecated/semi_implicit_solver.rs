pub struct SemiImplicitSolver;

impl SemiImplicitSolver {
    pub fn semi_implicit_update(&self, flux: f64, current_value: f64, dt: f64) -> f64 {
        let explicit_term = flux * dt;
        let implicit_term = current_value / (1.0 + dt);
        implicit_term + explicit_term
    }
}
