pub mod jacobi;
pub mod lu;
pub mod ilu;
pub mod cholesky;

pub use jacobi::Jacobi;
pub use lu::LU;
pub use ilu::ILU;
pub use cholesky::CholeskyPreconditioner;

use crate::linalg::{Matrix, Vector};
use faer::mat::Mat;
use std::sync::Arc;

/// Preconditioner trait with `Arc` for easier integration and thread safety.
pub trait Preconditioner: Send + Sync {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>);
}

/// `PreconditionerFactory` provides static methods to create common preconditioners.
/// This design promotes flexible creation and integration of preconditioners in a modular way.
pub struct PreconditionerFactory;

impl PreconditionerFactory {
    /// Creates a `Jacobi` preconditioner wrapped in `Arc`.
    pub fn create_jacobi() -> Arc<dyn Preconditioner> {
        Arc::new(Jacobi::default())
    }

    /// Creates an `ILU` preconditioner wrapped in `Arc` from a provided matrix.
    ///
    /// # Arguments
    /// - `matrix`: The matrix to use for constructing the ILU preconditioner.
    ///
    /// # Returns
    /// `Arc<dyn Preconditioner>` instance of the ILU preconditioner.
    pub fn create_ilu(matrix: &Mat<f64>) -> Arc<dyn Preconditioner> {
        Arc::new(ILU::new(matrix))
    }

    /// Creates a `CholeskyPreconditioner` wrapped in `Arc` from a provided matrix.
    ///
    /// # Arguments
    /// - `matrix`: The symmetric positive definite matrix to use for Cholesky decomposition.
    ///
    /// # Returns
    /// Result containing `Arc<dyn Preconditioner>` or an error if decomposition fails.
    pub fn create_cholesky(matrix: &Mat<f64>) -> Result<Arc<dyn Preconditioner>, Box<dyn std::error::Error>> {
        let preconditioner = CholeskyPreconditioner::new(matrix)?;
        Ok(Arc::new(preconditioner))
    }

    /// Creates an `LU` preconditioner wrapped in `Arc` from a provided matrix.
    ///
    /// # Arguments
    /// - `matrix`: The matrix to use for LU decomposition.
    ///
    /// # Returns
    /// `Arc<dyn Preconditioner>` instance of the LU preconditioner.
    pub fn create_lu(matrix: &Mat<f64>) -> Arc<dyn Preconditioner> {
        Arc::new(LU::new(matrix))
    }
}
