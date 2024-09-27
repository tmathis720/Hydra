use crate::solver::ksp::{KSP, SolverResult};
use crate::solver::preconditioner::Preconditioner;
use crate::solver::{Matrix, Vector};

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
        for i in 0..n {
            r[i] = b.get(i) - temp_vec[i];
        }

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

        p.copy_from_slice(&z);

        let mut rho = dot_product(
            &r as &dyn Vector<Scalar = f64>,
            &z as &dyn Vector<Scalar = f64>,
        );

        let mut iterations = 0;

        while iterations < self.max_iter && residual_norm > self.tol {
            a.mat_vec(
                &p as &dyn Vector<Scalar = f64>,
                &mut q as &mut dyn Vector<Scalar = f64>,
            );

            let pq = dot_product(
                &p as &dyn Vector<Scalar = f64>,
                &q as &dyn Vector<Scalar = f64>,
            );

            if pq.abs() < 1e-12 {
                // Handle potential division by zero
                return SolverResult {
                    converged: false,
                    iterations,
                    residual_norm,
                };
            }

            let alpha = rho / pq;

            // Update x = x + alpha * p
            for i in 0..n {
                x.set(i, x.get(i) + alpha * p[i]);
            }

            // Update r = r - alpha * q
            for i in 0..n {
                r[i] -= alpha * q[i];
            }

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

            let rho_new = dot_product(
                &r as &dyn Vector<Scalar = f64>,
                &z as &dyn Vector<Scalar = f64>,
            );

            let beta = rho_new / rho;
            rho = rho_new;

            // Update p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            iterations += 1;
        }

        SolverResult {
            converged: residual_norm <= self.tol,
            iterations,
            residual_norm,
        }
    }
}

// Helper functions
fn dot_product(u: &dyn Vector<Scalar = f64>, v: &dyn Vector<Scalar = f64>) -> f64 {
    u.as_slice()
        .iter()
        .zip(v.as_slice().iter())
        .map(|(&ui, &vi)| ui * vi)
        .sum()
}

fn euclidean_norm(u: &dyn Vector<Scalar = f64>) -> f64 {
    u.as_slice().iter().map(|&ui| ui * ui).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ksp::KSP;
    use crate::solver::preconditioner::Jacobi;
    use crate::solver::{Matrix, Vector};
    use faer::mat;
    use faer::Mat;

    #[test]
    fn test_cg_solver_no_preconditioner() {
        // SPD matrix A
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];

        // RHS vector b
        let b = mat![
            [1.0],
            [2.0],
        ];

        // Initial guess x0
        let mut x = Mat::<f64>::zeros(2, 1);

        // Expected solution x = [0.09090909, 0.63636364]
        let expected_x = mat![
            [0.09090909],
            [0.63636364],
        ];

        let mut cg = ConjugateGradient::new(100, 1e-6);

        let result = cg.solve(&a, &b, &mut x);

        println!("CG Solver Result: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // Verify the result
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

    #[test]
    fn test_cg_solver_with_jacobi_preconditioner() {
        // SPD matrix A
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];

        // RHS vector b
        let b = mat![
            [1.0],
            [2.0],
        ];

        // Initial guess x0
        let mut x = Mat::<f64>::zeros(2, 1);

        // Expected solution x = [0.09090909, 0.63636364]
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

        // Verify the result
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
