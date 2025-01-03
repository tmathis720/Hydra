use std::ops::{AddAssign, Mul};

/// Represents a 3x3 tensor with floating-point components.
///
/// The `Tensor3x3` struct is a simple abstraction for rank-2 tensors in 3D space.
/// It supports component-wise arithmetic operations, including addition and scalar multiplication.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tensor3x3(pub [[f64; 3]; 3]);

impl AddAssign for Tensor3x3 {
    /// Implements the `+=` operator for `Tensor3x3`.
    ///
    /// Adds another `Tensor3x3` to this tensor component-wise. The operation is performed
    /// in place, modifying the original tensor.
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            for j in 0..3 {
                self.0[i][j] += other.0[i][j];
            }
        }
    }
}

impl Mul<f64> for Tensor3x3 {
    type Output = Tensor3x3;

    /// Implements scalar multiplication for `Tensor3x3`.
    ///
    /// Multiplies each component of the tensor by a scalar `rhs`. The resulting tensor
    /// is a new `Tensor3x3`, leaving the original tensor unchanged.
    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.0[i][j] * rhs;
            }
        }
        Tensor3x3(result)
    }
}
