pub mod base;
pub mod linear;
pub mod weno;
pub mod ppm;

use log::{error, info};
use thiserror::Error;

pub use base::ReconstructionMethod;
pub use linear::LinearReconstruction;
pub use weno::WENOReconstruction;
pub use ppm::PPMReconstruction;

/// Enumeration for selecting different reconstruction methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructionType {
    Linear,
    WENO,
    PPM,
}

/// Error type for invalid reconstruction method selection.
#[derive(Error, Debug)]
pub enum ReconstructionError {
    #[error("Invalid reconstruction method specified: {0}")]
    InvalidMethod(String),
}

/// Factory function to create a reconstruction method dynamically.
///
/// # Parameters
/// - `method`: The reconstruction method to be used.
///
/// # Returns
/// - An instance of a struct implementing `ReconstructionMethod`.
/// - Returns `Err(ReconstructionError::InvalidMethod)` if an invalid method is specified.
///
/// # Example
/// ```rust
/// let recon = create_reconstruction(ReconstructionType::Linear).unwrap();
/// let face_value = recon.reconstruct(1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
/// ```
pub fn create_reconstruction(
    method: ReconstructionType,
) -> Result<Box<dyn ReconstructionMethod + Send + Sync>, ReconstructionError> {
    match method {
        ReconstructionType::Linear => {
            info!("Using Linear Reconstruction method.");
            Ok(Box::new(LinearReconstruction))
        }
        ReconstructionType::WENO => {
            info!("Using WENO Reconstruction method.");
            Ok(Box::new(WENOReconstruction))
        }
        ReconstructionType::PPM => {
            info!("Using PPM Reconstruction method.");
            Ok(Box::new(PPMReconstruction))
        }
        _ => {
            error!("Invalid reconstruction method: {:?}", method);
            Err(ReconstructionError::InvalidMethod(format!("{:?}", method)))
        }
    }
}
