pub mod flux_limiters;

use log::info;
use std::fmt;

/// Importing available flux limiter implementations
use flux_limiters::{BeamWarming, FluxLimiter, Koren, Minmod, Superbee, VanAlbada, VanLeer};

/// Enum to represent different flux limiter types
#[derive(Debug, Clone, Copy)]
pub enum FluxLimiterType {
    Minmod,
    Superbee,
    VanLeer,
    VanAlbada,
    Koren,
    BeamWarming,
}

/// Error type for handling flux limiter selection failures
#[derive(Debug)]
pub enum FluxLimiterError {
    InvalidLimiterType(String),
}

impl fmt::Display for FluxLimiterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FluxLimiterError::InvalidLimiterType(name) => {
                write!(f, "Invalid flux limiter type: {}", name)
            }
        }
    }
}

impl std::error::Error for FluxLimiterError {}

/// Factory function to create the appropriate flux limiter instance.
///
/// # Arguments
/// - `limiter_type`: The selected limiter type.
///
/// # Returns
/// - A boxed `FluxLimiter` instance or an error if an invalid type is provided.
pub fn create_flux_limiter(limiter_type: FluxLimiterType) -> Result<Box<dyn FluxLimiter + Send + Sync>, FluxLimiterError> {
    match limiter_type {
        FluxLimiterType::Minmod => {
            info!("Using Minmod flux limiter.");
            Ok(Box::new(Minmod))
        }
        FluxLimiterType::Superbee => {
            info!("Using Superbee flux limiter.");
            Ok(Box::new(Superbee))
        }
        FluxLimiterType::VanLeer => {
            info!("Using VanLeer flux limiter.");
            Ok(Box::new(VanLeer))
        }
        FluxLimiterType::VanAlbada => {
            info!("Using VanAlbada flux limiter.");
            Ok(Box::new(VanAlbada))
        }
        FluxLimiterType::Koren => {
            info!("Using Koren flux limiter.");
            Ok(Box::new(Koren))
        }
        FluxLimiterType::BeamWarming => {
            info!("Using BeamWarming flux limiter.");
            Ok(Box::new(BeamWarming))
        }
    }
}

/// Unit tests for FluxLimiter factory function
#[cfg(test)]
mod tests;
