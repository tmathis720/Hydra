use log::debug;

/// Trait defining a generic Flux Limiter, which adjusts flux values
/// to prevent numerical oscillations, crucial for Total Variation Diminishing (TVD) schemes.
/// 
/// # Purpose
/// This trait provides a method `limit` to calculate a modified value
/// based on neighboring values, which helps in maintaining the stability
/// and accuracy of the finite volume method by applying flux limiters.
pub trait FluxLimiter: Send + Sync {
    /// Applies the limiter to two neighboring values to prevent oscillations.
    ///
    /// # Parameters
    /// - `left_value`: The flux value on the left side of the interface.
    /// - `right_value`: The flux value on the right side of the interface.
    ///
    /// # Returns
    /// A modified value that limits oscillations, ensuring TVD compliance.
    fn limit(&self, left_value: f64, right_value: f64) -> f64;
}

/// Implementation of the Minmod flux limiter.
pub struct Minmod;

/// Implementation of the Superbee flux limiter.
pub struct Superbee;

/// Implementation of the Van Leer flux limiter.
pub struct VanLeer;

/// Implementation of the Van Albada flux limiter.
pub struct VanAlbada;

/// Implementation of the Koren flux limiter.
pub struct Koren;

/// Implementation of the Beam-Warming flux limiter.
pub struct BeamWarming;

impl FluxLimiter for Minmod {
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            debug!("Minmod: Opposite signs or zero - limiting to 0.0");
            0.0
        } else {
            let result = left_value.min(right_value);
            debug!("Minmod: left={}, right={}, result={}", left_value, right_value, result);
            result
        }
    }
}

impl FluxLimiter for Superbee {
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            debug!("Superbee: Opposite signs or zero - limiting to 0.0");
            0.0
        } else {
            let option1 = (2.0 * left_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let option2 = (2.0 * right_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let result = option1.max(option2);

            debug!(
                "Superbee: left={}, right={}, option1={}, option2={}, result={}",
                left_value, right_value, option1, option2, result
            );

            result
        }
    }
}

impl FluxLimiter for VanLeer {
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            debug!("VanLeer: Opposite signs or zero - limiting to 0.0");
            0.0
        } else {
            let r = left_value / right_value;
            let result = (r.abs() + r) / (1.0 + r.abs());

            debug!(
                "VanLeer: left={}, right={}, r={}, result={}",
                left_value, right_value, r, result
            );

            result
        }
    }
}

impl FluxLimiter for VanAlbada {
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            debug!("VanAlbada: Opposite signs or zero - limiting to 0.0");
            0.0
        } else {
            let numerator = left_value * right_value * (left_value + right_value);
            let denominator = left_value.powi(2) + right_value.powi(2) + f64::EPSILON;
            let result = numerator / denominator;

            debug!(
                "VanAlbada: left={}, right={}, numerator={}, denominator={}, result={}",
                left_value, right_value, numerator, denominator, result
            );

            result
        }
    }
}

impl FluxLimiter for Koren {
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            debug!("Koren: Opposite signs or zero - limiting to 0.0");
            0.0
        } else {
            let r = left_value / right_value;
            let result = (2.0 * r).min(1.0).min((1.0 / 3.0) + (2.0 * r / 3.0));

            debug!(
                "Koren: left={}, right={}, r={}, result={}",
                left_value, right_value, r, result
            );

            result
        }
    }
}

impl FluxLimiter for BeamWarming {
    fn limit(&self, left_value: f64, _right_value: f64) -> f64 {
        debug!("BeamWarming: No limitation applied, returning left_value={}", left_value);
        left_value
    }
}
