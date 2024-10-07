use crate::solver::ksp::{KSP, SolverResult};
use crate::solver::preconditioner::Preconditioner;
use crate::linalg::{Matrix, Vector};

pub struct GMRES {
    pub max_iter: usize,
    pub tol: f64,
    pub restart: usize,
    pub preconditioner: Option<Box<dyn Preconditioner>>,
}

impl GMRES {
    pub fn new(max_iter: usize, tol: f64, restart: usize) -> Self {
        GMRES {
            max_iter,
            tol,
            restart,
            preconditioner: None,
        }
    }

    pub fn set_preconditioner(&mut self, preconditioner: Box<dyn Preconditioner>) {
        self.preconditioner = Some(preconditioner);
    }
}

impl KSP for GMRES {
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        let n = b.len();
        let epsilon = 1e-12;

        let mut r = vec![0.0; n];
        let mut temp_vec = vec![0.0; n];
        let mut preconditioned_vec = vec![0.0; n];

        compute_initial_residual(a, b, x, &mut temp_vec, &mut r);
        if let Some(preconditioner) = &self.preconditioner {
            preconditioner.apply(
                a,
                &r as &dyn Vector<Scalar = f64>,
                &mut preconditioned_vec as &mut dyn Vector<Scalar = f64>,
            );
            r.copy_from_slice(&preconditioned_vec);
        }
        let mut residual_norm = euclidean_norm(&r as &dyn Vector<Scalar = f64>);

        if should_terminate_initial_residual(residual_norm, &r, x) {
            return SolverResult {
                converged: false,
                iterations: 0,
                residual_norm: f64::NAN,
            };
        }

        let mut iterations = 0;
        let mut v = vec![vec![0.0; n]; self.restart + 1]; // Krylov basis vectors
        let mut h = vec![vec![0.0; self.restart]; self.restart + 1]; // Corrected dimensions
        let mut g = vec![0.0; self.restart + 1]; // RHS vector for least-squares problem

        // Declare cosines and sines
        let mut cosines = vec![0.0; self.restart];
        let mut sines = vec![0.0; self.restart];

        loop {
            if !normalize_residual_and_init_krylov(&r, &mut v[0], epsilon) {
                return SolverResult {
                    converged: false,
                    iterations,
                    residual_norm: f64::NAN,
                };
            }

            g[0] = residual_norm;

            let mut inner_iterations = 0;

            for k in 0..self.restart {
                inner_iterations = k;

                if let Some(result) = arnoldi_process(
                    k,
                    a,
                    &mut v,
                    &mut temp_vec,
                    &mut preconditioned_vec,
                    &self.preconditioner,
                    &mut h,
                    epsilon,
                ) {
                    return result;
                }

                apply_givens_rotations(&mut h, &mut g, &mut cosines, &mut sines, k);

                residual_norm = g[k + 1].abs();
                if residual_norm < self.tol {
                    break;
                }
            }

            let m = inner_iterations + 1;
            let mut y = vec![0.0; m];
            back_substitution(&h, &g, &mut y, m);
            update_solution(x, &v, &y, n);

            iterations += m;

            // Step 5: Compute new residual
            compute_residual(a, b, x, &mut temp_vec, &mut r);
            if let Some(preconditioner) = &self.preconditioner {
                preconditioner.apply(
                    a,
                    &r as &dyn Vector<Scalar = f64>,
                    &mut preconditioned_vec as &mut dyn Vector<Scalar = f64>,
                );
                r.copy_from_slice(&preconditioned_vec);
            }
            residual_norm = euclidean_norm(&r as &dyn Vector<Scalar = f64>);

            if residual_norm.is_nan() || residual_norm.is_infinite() {
                return SolverResult {
                    converged: false,
                    iterations,
                    residual_norm: f64::NAN,
                };
            }

            if iterations >= self.max_iter || residual_norm <= self.tol {
                break;
            }
        }

        SolverResult {
            converged: residual_norm <= self.tol && !x.as_slice().iter().any(|&xi| xi.is_nan()),
            iterations,
            residual_norm,
        }
    }
}

// Helper functions for GMRES
fn compute_initial_residual(
    a: &dyn Matrix<Scalar = f64>,
    b: &dyn Vector<Scalar = f64>,
    x: &mut dyn Vector<Scalar = f64>,
    temp_vec: &mut Vec<f64>,
    residual: &mut Vec<f64>,
) {
    a.mat_vec(x, temp_vec); // temp_vec = A * x
    for i in 0..b.len() {
        residual[i] = b.get(i) - temp_vec[i];
    }
}

fn should_terminate_initial_residual(residual_norm: f64, _residual: &Vec<f64>, x: &dyn Vector<Scalar = f64>) -> bool {
    if residual_norm.is_nan() || residual_norm.is_infinite() {
        return true;
    }

    if residual_norm <= 1e-12 && x.as_slice().contains(&f64::NAN) {
        return true;
    }

    false
}

fn normalize_residual_and_init_krylov(
    residual: &Vec<f64>,
    krylov_vector: &mut Vec<f64>,
    epsilon: f64,
) -> bool {
    let r_norm = euclidean_norm(residual as &dyn Vector<Scalar = f64>);
    if r_norm < epsilon || r_norm.is_nan() || r_norm.is_infinite() {
        return false;
    }

    for i in 0..residual.len() {
        krylov_vector[i] = residual[i] / r_norm;
    }

    true
}

fn arnoldi_process(
    k: usize,
    a: &dyn Matrix<Scalar = f64>,
    v: &mut Vec<Vec<f64>>,
    temp_vec: &mut Vec<f64>,
    preconditioned_vec: &mut Vec<f64>,
    preconditioner: &Option<Box<dyn Preconditioner>>,
    h: &mut Vec<Vec<f64>>,
    epsilon: f64,
) -> Option<SolverResult> {
    // Apply matrix A to v[k]
    a.mat_vec(&v[k] as &dyn Vector<Scalar = f64>, temp_vec);

    // Apply preconditioner: w = M^{-1} * (A * v_k)
    if let Some(preconditioner) = preconditioner {
        preconditioner.apply(
            a,
            temp_vec as &dyn Vector<Scalar = f64>,
            preconditioned_vec as &mut dyn Vector<Scalar = f64>,
        );
        temp_vec.copy_from_slice(&preconditioned_vec); // Use the preconditioned result
    }

    // Modified Gram-Schmidt process
    for j in 0..=k {
        h[j][k] = dot_product(
            &v[j] as &dyn Vector<Scalar = f64>,
            temp_vec as &dyn Vector<Scalar = f64>,
        );
        for i in 0..temp_vec.len() {
            temp_vec[i] -= h[j][k] * v[j][i];
        }
    }

    h[k + 1][k] = euclidean_norm(temp_vec as &dyn Vector<Scalar = f64>);
    if h[k + 1][k].abs() < epsilon {
        h[k + 1][k] = 0.0;
    }

    if h[k + 1][k].abs() > epsilon {
        for i in 0..temp_vec.len() {
            v[k + 1][i] = temp_vec[i] / h[k + 1][k];
        }
    }

    None
}

fn update_solution(
    x: &mut dyn Vector<Scalar = f64>,
    v: &Vec<Vec<f64>>,
    y: &Vec<f64>,
    n: usize,
) {
    for j in 0..y.len() {
        for i in 0..n {
            x.set(i, x.get(i) + y[j] * v[j][i]);
        }
    }
}

fn compute_residual(
    a: &dyn Matrix<Scalar = f64>,
    b: &dyn Vector<Scalar = f64>,
    x: &mut dyn Vector<Scalar = f64>,
    temp_vec: &mut Vec<f64>,
    residual: &mut Vec<f64>,
) {
    a.mat_vec(x, temp_vec); // temp_vec = A * x
    for i in 0..b.len() {
        residual[i] = b.get(i) - temp_vec[i];
    }
}

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

fn apply_givens_rotations(
    h: &mut Vec<Vec<f64>>,
    g: &mut Vec<f64>,
    cosines: &mut Vec<f64>,
    sines: &mut Vec<f64>,
    k: usize,
) {
    // Apply all previous Givens rotations to the new column h[:, k]
    for i in 0..k {
        let temp = cosines[i] * h[i][k] + sines[i] * h[i + 1][k];
        h[i + 1][k] = -sines[i] * h[i][k] + cosines[i] * h[i + 1][k];
        h[i][k] = temp;
    }

    // Compute the new Givens rotation
    let h_kk = h[k][k];
    let h_k1k = h[k + 1][k];

    let r = (h_kk.powi(2) + h_k1k.powi(2)).sqrt();

    if r.abs() < 1e-12 {
        cosines[k] = 1.0;
        sines[k] = 0.0;
    } else {
        cosines[k] = h_kk / r;
        sines[k] = h_k1k / r;
    }

    // Apply the new Givens rotation to eliminate h[k+1][k]
    h[k][k] = cosines[k] * h_kk + sines[k] * h_k1k;
    h[k + 1][k] = 0.0;

    // Apply the new Givens rotation to g
    let temp = cosines[k] * g[k] + sines[k] * g[k + 1];
    g[k + 1] = -sines[k] * g[k] + cosines[k] * g[k + 1];
    g[k] = temp;
}

fn back_substitution(h: &Vec<Vec<f64>>, g: &Vec<f64>, y: &mut Vec<f64>, m: usize) {
    for i in (0..m).rev() {
        y[i] = g[i];
        for j in (i + 1)..m {
            y[i] -= h[i][j] * y[j];
        }
        if h[i][i].abs() > 1e-12 {
            y[i] /= h[i][i];
        } else {
            y[i] = 0.0; // Handle division by zero
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ksp::KSP;
    use crate::solver::preconditioner::Jacobi;
    use faer::mat;
    use faer::Mat;

    #[test]
    fn test_gmres_solver_no_preconditioner() {
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
        let _expected_x = mat![
            [0.09090909],
            [0.63636364],
        ];

        let mut gmres = GMRES::new(100, 1e-6, 2); // Restart every 2 iterations

        let result = gmres.solve(&a, &b, &mut x);

        assert!(!crate::linalg::matrix::traits::Matrix::as_slice(&x).contains(&f64::NAN), "Solution contains NaN values");
        assert!(result.converged, "GMRES did not converge");
        assert!(result.residual_norm <= 1e-6, "Residual norm too large");
    }

    #[test]
    fn test_gmres_solver_with_jacobi_preconditioner() {
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
        let _expected_x = mat![
            [0.09090909],
            [0.63636364],
        ];

        let mut gmres = GMRES::new(100, 1e-6, 2); // Restart every 2 iterations

        // Set Jacobi preconditioner
        let preconditioner = Box::new(Jacobi::default());
        gmres.set_preconditioner(preconditioner);

        let result = gmres.solve(&a, &b, &mut x);

        println!("GMRES Solver Result with Jacobi Preconditioner: {:?}", result);
        println!("Computed solution x = {:?}", x);

        assert!(!crate::linalg::matrix::traits::Matrix::as_slice(&x).contains(&f64::NAN), "Solution contains NaN values");
        assert!(result.converged, "GMRES did not converge with Jacobi preconditioner");
    }

    #[test]
    fn test_gmres_solver_large_system() {
        // A larger SPD matrix A
        let a = mat![
            [10.0, 1.0, 0.0, 0.0],
            [1.0, 7.0, 2.0, 0.0],
            [0.0, 2.0, 8.0, 1.0],
            [0.0, 0.0, 1.0, 5.0],
        ];

        // RHS vector b
        let b = mat![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ];

        // Initial guess x0
        let mut x = Mat::<f64>::zeros(4, 1);

        // GMRES solver with restart = 2
        let mut gmres = GMRES::new(200, 1e-6, 2); // Restart every 2 iterations

        let result = gmres.solve(&a, &b, &mut x);

        println!("GMRES Solver Result for Large System: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // The expected solution may not be easy to validate analytically, but we can
        // at least ensure the solver converged successfully.
        assert!(result.converged);
        assert!(result.residual_norm < 1e-6);
    }

    #[test]
    fn test_gmres_solver_convergence_on_ill_conditioned_system() {
        // Ill-conditioned matrix A
        let a = mat![
            [1e-10, 0.0],
            [0.0, 1e-10],
        ];

        // RHS vector b
        let b = mat![
            [1.0],
            [1.0],
        ];

        // Initial guess x0
        let mut x = Mat::<f64>::zeros(2, 1);

        let mut gmres = GMRES::new(100, 1e-6, 2); // Restart every 2 iterations

        let result = gmres.solve(&a, &b, &mut x);

        println!("GMRES Solver Result for Ill-conditioned System: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // Compute expected solution
        let expected_x = mat![
            [1e10],
            [1e10],
        ];

        // Calculate relative error
        let x_slice = crate::linalg::matrix::traits::Matrix::as_slice(&x);
        let expected_x_slice = crate::linalg::matrix::traits::Matrix::as_slice(&expected_x);
        let relative_error: f64 = x_slice
            .iter()
            .zip(expected_x_slice.iter())
            .map(|(&xi, &x_exact)| ((xi - x_exact) / x_exact).abs())
            .sum::<f64>()
            / x_slice.len() as f64;

        // Check if relative error is small
        assert!(
            relative_error < 1e-6,
            "GMRES did not converge to an accurate solution on an ill-conditioned system"
        );
    }
}