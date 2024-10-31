/// Trait defining a generic Flux Limiter, which adjusts flux values
/// to prevent numerical oscillations, crucial for Total Variation Diminishing (TVD) schemes.
/// 
/// # Purpose
/// This trait provides a method `limit` to calculate a modified value
/// based on neighboring values, which helps in maintaining the stability
/// and accuracy of the finite volume method by applying flux limiters.
/// 
/// # Method
/// - `limit`: Takes left and right flux values and returns a constrained value
/// to mitigate oscillations at cell interfaces.
pub trait FluxLimiter {
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
///
/// # Characteristics
/// The Minmod limiter is a simple, commonly used limiter that chooses the minimum
/// absolute value of the left and right values while preserving the sign. It is effective
/// for handling sharp gradients without introducing non-physical oscillations.
/// 
/// # Implementation Details
/// - If `left_value` and `right_value` have opposite signs or are zero, it returns 0.0
///   to avoid oscillations.
/// - Otherwise, it selects the smaller absolute value, retaining the original sign.
pub struct Minmod;

/// Implementation of the Superbee flux limiter.
///
/// # Characteristics
/// The Superbee limiter provides higher resolution compared to Minmod and is more aggressive,
/// capturing sharp gradients while preserving stability. This limiter is suitable
/// for problems where capturing steep gradients is essential.
/// 
/// # Implementation Details
/// - If `left_value` and `right_value` have opposite signs or are zero, it returns 0.0,
///   preventing oscillations.
/// - Otherwise, it calculates two options based on twice the left and right values,
///   clamping them within the original range, and selects the larger of the two.
pub struct Superbee;

impl FluxLimiter for Minmod {
    /// Applies the Minmod flux limiter to two neighboring values.
    ///
    /// # Parameters
    /// - `left_value`: Flux value from the left side of the cell interface.
    /// - `right_value`: Flux value from the right side of the cell interface.
    ///
    /// # Returns
    /// - `0.0` if the values have different signs (indicating an oscillation).
    /// - Otherwise, returns the value with the smaller magnitude, preserving the sign.
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            println!("Minmod: Different signs or zero - returning 0.0");
            0.0 // Different signs or zero: prevent oscillations by returning zero
        } else {
            // Take the minimum magnitude value, maintaining its original sign
            let result = if left_value.abs() < right_value.abs() {
                left_value
            } else {
                right_value
            };
            println!("Minmod: left_value = {}, right_value = {}, result = {}", left_value, right_value, result);
            result
        }
    }
}

impl FluxLimiter for Superbee {
    /// Applies the Superbee flux limiter to two neighboring values.
    ///
    /// # Parameters
    /// - `left_value`: Flux value from the left side of the cell interface.
    /// - `right_value`: Flux value from the right side of the cell interface.
    ///
    /// # Returns
    /// - `0.0` if the values have different signs, to prevent oscillations.
    /// - Otherwise, calculates two possible limited values and returns the maximum
    ///   to ensure higher resolution while maintaining stability.
    fn limit(&self, left_value: f64, right_value: f64) -> f64 {
        if left_value * right_value <= 0.0 {
            println!("Superbee: Different signs or zero - returning 0.0");
            0.0 // Different signs: prevent oscillations by returning zero
        } else {
            // Calculate two limited values and return the maximum to capture sharp gradients
            let option1 = (2.0 * left_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let option2 = (2.0 * right_value).clamp(left_value.min(right_value), left_value.max(right_value));
            let result = option1.max(option2);

            println!(
                "Superbee: left_value = {}, right_value = {}, option1 = {}, option2 = {}, result = {}",
                left_value, right_value, option1, option2, result
            );

            result
        }
    }
}
