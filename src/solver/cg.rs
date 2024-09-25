use nalgebra::{DMatrix, DVector};

// Define the struct for the CG solver
pub struct ConjugateGradient<'a> {
    a: &'a DMatrix<f64>,
    b: &'a DVector<f64>,
    x: DVector<f64>,
    preconditioner: Option<Box<dyn Fn(&DVector<f64>) -> DVector<f64> + 'a>>,  // Allow 'a lifetime
}

impl<'a> ConjugateGradient<'a> {
    pub fn new(
        a: &'a DMatrix<f64>, 
        b: &'a DVector<f64>, 
        preconditioner: Option<Box<dyn Fn(&DVector<f64>) -> DVector<f64> + 'a>>,
    ) -> Self {
        let x = DVector::zeros(b.len());
        ConjugateGradient { a, b, x, preconditioner }
    }

    pub fn solve(&mut self) -> Result<DVector<f64>, &'static str> {
        let mut r = self.b - self.a * &self.x; // Initial residual
        let z = if let Some(precondition) = &self.preconditioner {
            precondition(&r)  // Apply preconditioner if available
        } else {
            r.clone()
        };

        let mut p = z.clone();
        let mut rz_old = r.dot(&z);

        // Add a threshold for stagnation detection
        let tolerance = 1e-6;

        for iter in 0..1000 {
            let ap = self.a * &p;
            let denom = p.dot(&ap);
            
            // Check for small denominator indicating possible singular matrix
            if denom.abs() < tolerance {
                return Err("CG solver detected a singular or near-singular matrix (denominator close to zero)");
            }

            let alpha = rz_old / denom;
            self.x += alpha * &p;
            r -= alpha * &ap;

            // Check if the residual is small enough for convergence
            if r.norm() < tolerance {
                return Ok(self.x.clone());
            }

            let z = if let Some(precondition) = &self.preconditioner {
                precondition(&r)
            } else {
                r.clone()
            };

            let rz_new = r.dot(&z);

            // Check for stagnation: if the residual hasn't changed significantly, return an error
            if (rz_new - rz_old).abs() < tolerance {
                return Err("CG solver detected stagnation; possible non-convergence due to singular matrix.");
            }

            let beta = rz_new / rz_old;
            p = z + beta * &p;
            rz_old = rz_new;
        }

        Err("CG did not converge within the maximum number of iterations")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use crate::solver::jacobi::Jacobi;

    #[test]
    fn test_conjugate_gradient_no_preconditioner() {
        // A simple 3x3 SPD system (symmetric positive definite)
        let a = DMatrix::from_vec(3, 3, vec![
            4.0, 1.0, 0.0, 
            1.0, 3.0, 0.0, 
            0.0, 0.0, 2.0,
        ]);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let mut cg_solver = ConjugateGradient::new(&a, &b, None);
        let solution = cg_solver.solve().expect("CG should converge");

        // Verify solution
        let expected_solution = DVector::from_vec(vec![0.090909, 0.636364, 1.5]);
        assert!((solution - expected_solution).norm() < 1e-6);
    }

    #[test]
    fn test_conjugate_gradient_with_jacobi_preconditioner() {
        // A simple 3x3 SPD system (symmetric positive definite)
        let a = DMatrix::from_vec(3, 3, vec![
            4.0, 1.0, 0.0, 
            1.0, 3.0, 0.0, 
            0.0, 0.0, 2.0,
        ]);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        // Jacobi preconditioner
        let jacobi_preconditioner = Jacobi::new(&a);
        
        // No 'static lifetime issue anymore
        let preconditioner: Option<Box<dyn Fn(&DVector<f64>) -> DVector<f64>>> = Some(Box::new(move |r: &DVector<f64>| {
            jacobi_preconditioner.apply_preconditioner(r)
        }));

        let mut cg_solver = ConjugateGradient::new(&a, &b, preconditioner);
        let solution = cg_solver.solve().expect("CG should converge with Jacobi preconditioner");

        // Verify solution
        let expected_solution = DVector::from_vec(vec![0.090909, 0.636364, 1.5]);
        assert!((solution - expected_solution).norm() < 1e-6);
    }

    /* #[test]
    fn test_conjugate_gradient_non_convergence() {
        // An ill-conditioned system that will not converge (singular matrix)
        let a = DMatrix::from_vec(3, 3, vec![
            1.0, 2.0, 3.0, 
            2.0, 4.0, 6.0, 
            3.0, 6.0, 9.0,  // Singular matrix (rank 1)
        ]);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let mut cg_solver = ConjugateGradient::new(&a, &b, None);
        let result = cg_solver.solve();

        // Ensure that the solver doesn't converge
        assert!(result.is_err(), "CG should not converge for singular systems");
    } */
}