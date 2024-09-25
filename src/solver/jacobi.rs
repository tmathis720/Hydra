use nalgebra::{DMatrix, DVector};

// Jacobi Preconditioner
pub struct Jacobi<'a> {
    a: &'a DMatrix<f64>,
}

impl<'a> Jacobi<'a> {
    pub fn new(a: &'a DMatrix<f64>) -> Self {
        Jacobi { a }
    }

    // Method to apply the preconditioner
    pub fn apply_preconditioner(&self, r: &DVector<f64>) -> DVector<f64> {
        let mut z = DVector::zeros(r.len());
        for i in 0..self.a.nrows() {
            z[i] = r[i] / self.a[(i, i)];
        }
        z
    }

    // Static function that does not require capturing
    pub fn apply_preconditioner_static(a: &DMatrix<f64>, r: &DVector<f64>) -> DVector<f64> {
        let mut z = DVector::zeros(r.len());
        for i in 0..a.nrows() {
            if a[(i, i)] == 0.0 {
                panic!("Division by zero in Jacobi preconditioner: diagonal element is zero.");
            }
            z[i] = r[i] / a[(i, i)];
        }
        z
    }
}

// Test Module
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    #[test]
    fn test_apply_preconditioner() {
        // Setup a 3x3 diagonal matrix for Jacobi preconditioning
        let a = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 9.0, 16.0]));
        let r = DVector::from_vec(vec![8.0, 18.0, 32.0]);
        let jacobi = Jacobi::new(&a);

        // Apply the Jacobi preconditioner
        let z = jacobi.apply_preconditioner(&r);

        // Assert that the preconditioner works correctly
        assert_eq!(z, DVector::from_vec(vec![2.0, 2.0, 2.0])); // r[i] / a[i,i]
    }

    #[test]
    fn test_apply_preconditioner_static() {
        // Setup a 3x3 diagonal matrix for Jacobi preconditioning
        let a = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 5.0, 10.0]));
        let r = DVector::from_vec(vec![4.0, 10.0, 30.0]);

        // Apply the static Jacobi preconditioner
        let z = Jacobi::apply_preconditioner_static(&a, &r);

        // Assert that the preconditioner works correctly
        assert_eq!(z, DVector::from_vec(vec![2.0, 2.0, 3.0])); // r[i] / a[i,i]
    }

    #[test]
    fn test_jacobi_preconditioner_with_zero() {
        // Test with zero in the diagonal to check for proper error handling (dividing by zero)
        let a = DMatrix::from_diagonal(&DVector::from_vec(vec![0.0, 5.0, 10.0]));
        let r = DVector::from_vec(vec![4.0, 10.0, 30.0]);

        // Apply the Jacobi preconditioner, expect a panic or error handling
        let result = std::panic::catch_unwind(|| Jacobi::apply_preconditioner_static(&a, &r));

        // Ensure that the method panics when dividing by zero
        assert!(result.is_err());
    }
}