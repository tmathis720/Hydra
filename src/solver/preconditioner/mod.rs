pub mod jacobi;
pub mod lu;

pub use jacobi::Jacobi;
pub use lu::LU;

use crate::linalg::{Matrix, Vector};

// Preconditioner trait
pub trait Preconditioner {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>);
}

