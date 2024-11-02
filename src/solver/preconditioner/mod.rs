pub mod jacobi;
pub mod lu;
pub mod ilu;
pub mod cholesky;

pub use jacobi::Jacobi;
pub use lu::LU;
pub use ilu::ILU;
pub use cholesky::CholeskyPreconditioner;

use crate::linalg::{Matrix, Vector};

// Preconditioner trait
pub trait Preconditioner {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>);
}
