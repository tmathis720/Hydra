pub struct FluxLimiter;

impl FluxLimiter {
    pub fn superbee_limiter(r: f64) -> f64 {
        r.max(0.0).min(2.0).max(r.min(1.0))
    }

    pub fn apply_limiter(&self, flux: f64, left_flux: f64, right_flux: f64) -> f64 {
        let r = right_flux / left_flux;
        let phi = FluxLimiter::superbee_limiter(r);
        phi * flux
    }
}
