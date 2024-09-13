pub struct CrankNicolsonSolver;

impl CrankNicolsonSolver {
    pub fn crank_nicolson_update(&self, flux: f64, current_value: f64, dt: f64) -> f64 {
        let explicit_term = 0.5 * flux * dt;
        let implicit_term = current_value / (1.0 + 0.5 * dt);
        implicit_term + explicit_term
    }
}
