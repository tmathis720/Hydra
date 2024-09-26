use rayon::prelude::*;
use faer_core::{Mat, MatMut};
use faer_core::dyn_stack::ReborrowMut;

pub struct JacobiPreconditioner<'a> {
    a: &'a Mat<f64>,
}

impl<'a> JacobiPreconditioner<'a> {
    pub fn new(a: &'a Mat<f64>) -> Self {
        JacobiPreconditioner { a }
    }

    pub fn apply_parallel(&self, r: &Mat<f64>, z: &mut MatMut<f64>) {
        (0..z.nrows()).into_par_iter().for_each(|i| {
            let ai = self.a.read(i, i);  // Safely read from matrix `a`
            if ai != 0.0 {
                let ri = r.read(i, 0);  // Safely read from `r`
                let mut z_row = z.rb_mut().row(i);  // Mutably borrow the specific row in `z`
                z_row.write(0, ri / ai);  // Mutably write to the row in `z`
            }
        });
    }

    pub fn apply_sequential(&self, r: &Mat<f64>, z: &mut MatMut<f64>) {
        for i in 0..z.nrows() {
            let ai = self.a[(i, i)];
            if ai != 0.0 {
                let ri = r[(i, 0)];
                z.write(i, 0, ri / ai);
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use faer_core::mat;

    #[test]
    fn test_jacobi_preconditioner() {
        // Create a test diagonal matrix 'a'
        let a = mat![
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0]
        ];

        // Create a right-hand side matrix (vector form)
        let r = mat![
            [8.0],
            [9.0],
            [4.0]
        ];

        // Initialize an empty result matrix 'z'
        let mut z = Mat::<f64>::zeros(3, 1);

        // Create a mutable view for 'z'
        let mut z_mut = z.as_mut();

        // Create the Jacobi preconditioner with matrix 'a'
        let jacobi = JacobiPreconditioner::new(&a);

        // Apply the preconditioner in parallel
        jacobi.apply_parallel(&r, &mut z_mut);

        // Check the result values
        assert_eq!(z.read(0, 0), 2.0);  // 8 / 4
        assert_eq!(z.read(1, 0), 3.0);  // 9 / 3
        assert_eq!(z.read(2, 0), 2.0);  // 4 / 2
    }
}
