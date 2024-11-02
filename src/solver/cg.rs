use crate::solver::ksp::{KSP, SolverResult};
use crate::solver::preconditioner::Preconditioner;
use crate::linalg::{Matrix, Vector};
use rayon::prelude::*;

pub struct ConjugateGradient {
    pub max_iter: usize,
    pub tol: f64,
    pub preconditioner: Option<Box<dyn Preconditioner>>,
}

impl ConjugateGradient {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        ConjugateGradient {
            max_iter,
            tol,
            preconditioner: None,
        }
    }

    pub fn set_preconditioner(&mut self, preconditioner: Box<dyn Preconditioner>) {
        self.preconditioner = Some(preconditioner);
    }
}

impl KSP for ConjugateGradient {
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        let n = b.len();
        let mut r = vec![0.0; n]; // Residual vector
        let mut z = vec![0.0; n]; // Preconditioned residual
        let mut p = vec![0.0; n]; // Search direction
        let mut q = vec![0.0; n]; // A * p
        let mut temp_vec = vec![0.0; n]; // Temporary vector

        // Compute initial residual r = b - A * x
        a.mat_vec(x, &mut temp_vec as &mut dyn Vector<Scalar = f64>); // temp_vec = A * x
        r.par_iter_mut()
            .zip(b.as_slice())
            .zip(temp_vec.as_slice())
            .for_each(|((r_i, &b_i), &temp_i)| {
                *r_i = b_i - temp_i;
            });

        // Calculate the initial residual norm and check for convergence
        let mut residual_norm = euclidean_norm(&r as &dyn Vector<Scalar = f64>);
        if residual_norm <= self.tol {
            return SolverResult {
                converged: true,
                iterations: 0,
                residual_norm,
            };
        }

        // Apply preconditioner if available
        if let Some(preconditioner) = &self.preconditioner {
            preconditioner.apply(
                a,
                &r as &dyn Vector<Scalar = f64>,
                &mut z as &mut dyn Vector<Scalar = f64>,
            );
        } else {
            z.copy_from_slice(&r);
        }

        // Initialize search direction vector p with preconditioned residual
        p.copy_from_slice(&z);

        // Initialize rho as the dot product of r and z
        let mut rho = dot_product(&r as &dyn Vector<Scalar = f64>, &z as &dyn Vector<Scalar = f64>);
        let mut iterations = 0;

        // Start the main CG iteration loop
        while iterations < self.max_iter && residual_norm > self.tol {
            // Compute A * p and store in q
            a.mat_vec(
                &p as &dyn Vector<Scalar = f64>,
                &mut q as &mut dyn Vector<Scalar = f64>,
            );

            // Compute alpha = rho / (p^T * q)
            let pq = dot_product(&p as &dyn Vector<Scalar = f64>, &q as &dyn Vector<Scalar = f64>);
            if pq.abs() < 1e-12 {
                // Handle potential division by zero
                return SolverResult {
                    converged: false,
                    iterations,
                    residual_norm,
                };
            }
            let alpha = rho / pq;

            // Update solution vector x = x + alpha * p
            let x_slice = x.as_mut_slice();
            x_slice.par_iter_mut()
                .zip(&p)
                .for_each(|(x_i, &p_i)| {
                    *x_i += alpha * p_i;
                });

            // Update residual vector r = r - alpha * q
            r.par_iter_mut()
                .zip(&q)
                .for_each(|(r_i, &q_i)| {
                    *r_i -= alpha * q_i;
                });

            // Recalculate residual norm and check for convergence
            residual_norm = euclidean_norm(&r as &dyn Vector<Scalar = f64>);
            if residual_norm <= self.tol {
                break;
            }

            // Apply preconditioner to new residual
            if let Some(preconditioner) = &self.preconditioner {
                preconditioner.apply(
                    a,
                    &r as &dyn Vector<Scalar = f64>,
                    &mut z as &mut dyn Vector<Scalar = f64>,
                );
            } else {
                z.copy_from_slice(&r);
            }

            // Update rho and calculate beta for the next iteration
            let rho_new = dot_product(&r as &dyn Vector<Scalar = f64>, &z as &dyn Vector<Scalar = f64>);
            let beta = rho_new / rho;
            rho = rho_new;

            // Update search direction vector p = z + beta * p
            p.par_iter_mut()
                .zip(&z)
                .for_each(|(p_i, &z_i)| {
                    *p_i = z_i + beta * *p_i;
                });

            iterations += 1;
        }

        SolverResult {
            converged: residual_norm <= self.tol,
            iterations,
            residual_norm,
        }
    }
}

// Parallelized Helper Functions

/// Computes the dot product of two vectors `u` and `v` in parallel, using `rayon`.
///
/// # Arguments
/// - `u`: First vector.
/// - `v`: Second vector.
///
/// # Returns
/// - The dot product of `u` and `v`.
fn dot_product(u: &dyn Vector<Scalar = f64>, v: &dyn Vector<Scalar = f64>) -> f64 {
    u.as_slice()
        .par_iter()
        .zip(v.as_slice().par_iter())
        .map(|(&ui, &vi)| ui * vi)
        .sum()
}

/// Calculates the Euclidean norm of a vector `u` in parallel, returning `sqrt(sum(u_i^2))`.
///
/// # Arguments
/// - `u`: Vector for which to calculate the norm.
///
/// # Returns
/// - Euclidean (L2) norm of `u`.
fn euclidean_norm(u: &dyn Vector<Scalar = f64>) -> f64 {
    u.as_slice().par_iter().map(|&ui| ui * ui).sum::<f64>().sqrt()
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ksp::KSP;
    use crate::solver::preconditioner::Jacobi;
    use faer::mat;
    use faer::Mat;

    /// Test CG solver without a preconditioner on a small SPD matrix.
    /// Ensures convergence within tolerance and verifies the computed solution.
    #[test]
    fn test_cg_solver_no_preconditioner() {
        // Define a small SPD matrix A
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];

        // Right-hand side vector b
        let b = mat![
            [1.0],
            [2.0],
        ];

        // Initial guess x0
        let mut x = Mat::<f64>::zeros(2, 1);

        // Expected solution for reference
        let expected_x = mat![
            [0.09090909],
            [0.63636364],
        ];

        let mut cg = ConjugateGradient::new(100, 1e-6);
        let result = cg.solve(&a, &b, &mut x);

        println!("CG Solver Result: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // Verify computed solution against expected values
        for i in 0..x.nrows() {
            assert!(
                (x.read(i, 0) - expected_x.read(i, 0)).abs() < 1e-5,
                "x[{}] = {}, expected {}",
                i,
                x.read(i, 0),
                expected_x.read(i, 0)
            );
        }
    }

    /// Test CG solver with a Jacobi preconditioner on a small SPD matrix.
    /// Ensures that the solver with preconditioning achieves convergence and accuracy.
    #[test]
    fn test_cg_solver_with_jacobi_preconditioner() {
        // Define a small SPD matrix A
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];

        // Right-hand side vector b
        let b = mat![
            [1.0],
            [2.0],
        ];

        // Initial guess x0
        let mut x = Mat::<f64>::zeros(2, 1);

        // Expected solution for reference
        let expected_x = mat![
            [0.09090909],
            [0.63636364],
        ];

        let mut cg = ConjugateGradient::new(100, 1e-6);

        // Set Jacobi preconditioner
        let preconditioner = Box::new(Jacobi::default());
        cg.set_preconditioner(preconditioner);

        let result = cg.solve(&a, &b, &mut x);

        println!("CG Solver Result with Jacobi Preconditioner: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // Verify computed solution against expected values
        for i in 0..x.nrows() {
            assert!(
                (x.read(i, 0) - expected_x.read(i, 0)).abs() < 1e-5,
                "x[{}] = {}, expected {}",
                i,
                x.read(i, 0),
                expected_x.read(i, 0)
            );
        }
    }
}
