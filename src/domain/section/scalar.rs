use std::ops::{AddAssign, Mul};

/// Represents a scalar value.
///
/// The `Scalar` struct is a wrapper for a floating-point value, used in mathematical
/// and physical computations where type safety or domain-specific semantics are desired.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scalar(pub f64);

impl AddAssign for Scalar {
    /// Implements the `+=` operator for `Scalar`.
    ///
    /// Adds another scalar to this scalar. The operation is performed in place.
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Mul<f64> for Scalar {
    type Output = Scalar;

    /// Implements scalar multiplication for `Scalar`.
    ///
    /// Multiplies the scalar value by another scalar `rhs`. The resulting value
    /// is wrapped in a new `Scalar`.
    fn mul(self, rhs: f64) -> Self::Output {
        Scalar(self.0 * rhs)
    }
}

