use nalgebra::{DVector, DMatrix};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*; // For parallelism

// Jacobi Preconditioner
pub struct JacobiPreconditioner<'a> {
    a: &'a DMatrix<f64>,
}

impl<'a> JacobiPreconditioner<'a> {
    pub fn new(a: &'a DMatrix<f64>) -> Self {
        JacobiPreconditioner { a }
    }

    // Static method that doesn't require capturing and uses parallelism
    fn apply_preconditioner_static(a: &DMatrix<f64>, r: &DVector<f64>) -> DVector<f64> {
        let mut z = DVector::zeros(r.len());

        // Iterate over each element of the matrix and apply Jacobi preconditioning
        for i in 0..a.nrows() {
            let ai = a[(i, i)];
            if ai != 0.0 {
                z[i] = r[i] / ai;
            }
        }

        z
    }

    fn apply_preconditioner(z: &mut DVector<f64>, a: &DMatrix<f64>) {
        z.as_mut_slice()
         .par_iter_mut()
         .enumerate()
         .for_each(|(i, zi)| {
             let ai = a[(i, i)];
             if ai != 0.0 {
                 *zi /= ai;
             }
         });
    }

    pub fn apply_parallel(a: &DMatrix<f64>, r: &DVector<f64>, z: &mut DVector<f64>) {
        z.as_mut_slice()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, zi)| {
                let ai = a[(i, i)];
                if ai != 0.0 {
                    *zi = r[i] / ai;
                }
            });
    }
}
