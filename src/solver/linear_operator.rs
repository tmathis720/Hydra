// src/solver/linear_operator.rs

use faer_core::Matrix;

pub trait LinearOperator {
    fn matvec(&self, x: &[f64], y: &mut [f64]);
    fn matmul(&self, other: &Self) -> Self;
}

impl LinearOperator for Matrix<f64> {
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let result = self * Matrix::from_column_slice(x.len(), 1, x);
        for (i, &val) in result.iter().enumerate() {
            y[i] = val;
        }
    }

    fn matmul(&self, other: &Self) -> Self {
        self * other
    }
}