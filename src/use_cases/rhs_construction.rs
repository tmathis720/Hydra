// src/use_cases/rhs_construction.rs

use crate::interface_adapters::vector_adapter::VectorAdapter;
use crate::linalg::vector::Vector;
use faer::Mat;

/// Constructs and initializes the right-hand side (RHS) vector for a linear system.
pub struct RHSConstruction;

impl RHSConstruction {
    /// Builds a dense RHS vector of a specified length, initialized to zero.
    pub fn build_zero_rhs(size: usize) -> Mat<f64> {
        VectorAdapter::new_dense_vector(size)
    }

    /// Initializes the RHS vector with a specific value across all elements.
    pub fn initialize_rhs_with_value<T: Vector<Scalar = f64>>(vector: &mut T, value: f64) {
        for i in 0..vector.len() {
            VectorAdapter::set_element(vector, i, value);
        }
    }

    /// Resizes the RHS vector to a new length.
    pub fn resize_rhs(vector: &mut Mat<f64>, new_size: usize) {
        let mut new_vector = Mat::<f64>::zeros(new_size, 1);
        for i in 0..usize::min(vector.nrows(), new_size) {
            new_vector.write(i, 0, vector.read(i, 0));
        }
        *vector = new_vector; // Replace the old vector with the resized one
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_zero_rhs() {
        let size = 4;
        let rhs = RHSConstruction::build_zero_rhs(size);

        assert_eq!(rhs.nrows(), size, "RHS vector should have the specified size.");
        assert_eq!(rhs.ncols(), 1, "RHS vector should have a single column.");
        for i in 0..size {
            assert_eq!(rhs.read(i, 0), 0.0, "RHS vector should be initialized to zero.");
        }
    }

    #[test]
    fn test_initialize_rhs_with_value() {
        let mut rhs = RHSConstruction::build_zero_rhs(5);
        let init_value = 3.5;
        RHSConstruction::initialize_rhs_with_value(&mut rhs, init_value);

        for i in 0..rhs.nrows() {
            assert_eq!(
                rhs.read(i, 0),
                init_value,
                "Each RHS element should be initialized to the specified value."
            );
        }
    }

    #[test]
    fn test_resize_rhs() {
        let mut rhs = RHSConstruction::build_zero_rhs(3);
        rhs.write(0, 0, 1.0);
        rhs.write(1, 0, 2.0);
        rhs.write(2, 0, 3.0);

        RHSConstruction::resize_rhs(&mut rhs, 5);

        assert_eq!(rhs.nrows(), 5, "RHS vector should have the new specified size after resizing.");
        assert_eq!(rhs.read(0, 0), 1.0, "Original data should be preserved.");
        assert_eq!(rhs.read(1, 0), 2.0, "Original data should be preserved.");
        assert_eq!(rhs.read(2, 0), 3.0, "Original data should be preserved.");
        assert_eq!(rhs.read(3, 0), 0.0, "New elements should be initialized to zero.");
        assert_eq!(rhs.read(4, 0), 0.0, "New elements should be initialized to zero.");
    }
}
