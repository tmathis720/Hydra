use std::collections::HashMap;
use crate::linalg::Vector;
use crate::linalg::matrix::traits::{Matrix, MatrixOperations, ExtendedMatrixOperations};

/// A simple sparse matrix implementation using a HashMap to store non-zero elements.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    nrows: usize,
    ncols: usize,
    data: HashMap<(usize, usize), f64>, // Maps (row, col) to value
}

impl SparseMatrix {
    /// Creates a new sparse matrix with the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            nrows: rows,
            ncols: cols,
            data: HashMap::new(),
        }
    }

    /// Adds a value to the matrix at the specified row and column.
    ///
    /// If an entry already exists at the given position, the value is added to the existing value.
    /// If no entry exists, a new one is created.
    ///
    /// # Parameters
    /// - `row`: The row index of the entry.
    /// - `col`: The column index of the entry.
    /// - `value`: The value to add.
    pub fn add_entry(&mut self, row: usize, col: usize, value: f64) {
        let entry = self.data.entry((row, col)).or_insert(0.0);
        *entry += value;

        // Remove the entry if it becomes approximately zero to maintain sparsity
        if entry.abs() < f64::EPSILON {
            self.data.remove(&(row, col));
        }
    }

    /// Applies the matrix to a vector.
    ///
    /// Performs the operation `y = A * x`, where `A` is the sparse matrix, `x` is the input vector,
    /// and `y` is the result vector.
    ///
    /// # Parameters
    /// - `x`: The input vector (assumed to have size equal to the number of columns).
    /// - `y`: The result vector, which will be overwritten with the result (size equal to the number of rows).
    pub fn apply_to_vector(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>) {
        // Ensure that y is properly initialized
        for i in 0..self.nrows {
            y.set(i, 0.0); // Initialize to zero
        }

        // Iterate over all non-zero entries in the matrix
        for (&(row, col), &value) in self.data.iter() {
            let x_value = x.get(col); // Get the corresponding value in the input vector
            let y_value = y.get(row); // Get the current value in the result vector
            y.set(row, y_value + value * x_value); // Accumulate the result
        }
    }
}

/// Implement the `Matrix` trait for `SparseMatrix`.
impl Matrix for SparseMatrix {
    type Scalar = f64;

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn mat_vec(&self, x: &dyn Vector<Scalar = f64>, y: &mut dyn Vector<Scalar = f64>) {
        assert_eq!(x.len(), self.ncols, "Vector size mismatch");
        assert_eq!(y.len(), self.nrows, "Vector size mismatch");

        // Initialize y to zero
        for i in 0..y.len() {
            y.set(i, 0.0);
        }

        // Perform sparse matrix-vector multiplication
        for (&(row, col), &value) in self.data.iter() {
            let x_val = x.get(col);
            let y_val = y.get(row);
            y.set(row, y_val + value * x_val);
        }
    }

    fn get(&self, i: usize, j: usize) -> Self::Scalar {
        *self.data.get(&(i, j)).unwrap_or(&0.0)
    }

    fn trace(&self) -> Self::Scalar {
        let mut trace_sum = 0.0;
        for row in 0..self.nrows {
            if let Some(value) = self.data.get(&(row, row)) {
                trace_sum += value;
            }
        }
        trace_sum
    }

    fn frobenius_norm(&self) -> Self::Scalar {
        self.data.values().map(|&val| val * val).sum::<f64>().sqrt()
    }

    fn as_slice(&self) -> Box<[Self::Scalar]> {
        panic!("SparseMatrix does not support contiguous slices.");
    }

    fn as_slice_mut(&mut self) -> Box<[Self::Scalar]> {
        panic!("SparseMatrix does not support mutable slices.");
    }
}

/// Implement the `MatrixOperations` trait for `SparseMatrix`.
impl MatrixOperations for SparseMatrix {
    fn construct(rows: usize, cols: usize) -> Self {
        SparseMatrix::new(rows, cols)
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        if value.abs() < f64::EPSILON {
            self.data.remove(&(row, col)); // Remove near-zero values to maintain sparsity
        } else {
            self.data.insert((row, col), value);
        }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        // Access the underlying data storage explicitly to avoid ambiguity
        self.data.get(&(row, col)).copied().unwrap_or(0.0)
    }

    fn size(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
}

/// Implement the `ExtendedMatrixOperations` trait for `SparseMatrix`.
impl ExtendedMatrixOperations for SparseMatrix {
    fn resize(&mut self, new_rows: usize, new_cols: usize) {
        let mut new_data = HashMap::new();
        for (&(row, col), &value) in self.data.iter() {
            if row < new_rows && col < new_cols {
                new_data.insert((row, col), value);
            }
        }
        self.nrows = new_rows;
        self.ncols = new_cols;
        self.data = new_data;
    }
}
