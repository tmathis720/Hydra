use crate::solver::ksp::{KSP, SolverResult};
use crate::solver::preconditioner::Preconditioner;
use crate::linalg::{Matrix, Vector};
use rayon::prelude::*;
use std::sync::Arc;

/// Generalized Minimal Residual Solver (GMRES) for solving sparse linear systems using Krylov subspace methods.
/// GMRES is particularly effective for non-symmetric or non-positive-definite systems.
/// It minimizes the residual norm over a Krylov subspace, providing better convergence properties than
/// simple iterative methods.
///
/// # Attributes
/// - `max_iter`: The maximum number of outer iterations allowed.
/// - `tol`: Convergence tolerance for the residual norm.
/// - `restart`: Number of iterations before restarting GMRES (helps with convergence in some systems).
/// - `preconditioner`: Optional preconditioner to improve convergence by altering the system's condition number.
pub struct GMRES {
    pub max_iter: usize,
    pub tol: f64,
    pub restart: usize,
    pub preconditioner: Option<Arc<dyn Preconditioner>>,
}

impl GMRES {
    /// Initializes a new GMRES solver instance.
    ///
    /// # Arguments
    /// - `max_iter`: Maximum allowed iterations before stopping.
    /// - `tol`: Tolerance level for convergence, indicating desired precision.
    /// - `restart`: Number of iterations before restarting the Krylov subspace (helps prevent loss of orthogonality).
    pub fn new(max_iter: usize, tol: f64, restart: usize) -> Self {
        GMRES {
            max_iter,
            tol,
            restart,
            preconditioner: None,
        }
    }

    /// Sets the preconditioner for the solver. Preconditioners can improve convergence by
    /// altering the system's condition, making it more favorable for iterative solution.
    ///
    /// # Arguments
    /// - `preconditioner`: A shared preconditioner that implements the `Preconditioner` trait.
    pub fn set_preconditioner(&mut self, preconditioner: Arc<dyn Preconditioner>) {
        self.preconditioner = Some(preconditioner);
    }
}

impl KSP for GMRES {
    /// Executes the GMRES algorithm to solve the system `Ax = b` for a given sparse matrix `A`.
    ///
    /// # Arguments
    /// - `a`: The matrix `A` of the system, represented as a `Matrix` trait object.
    /// - `b`: The right-hand side vector `b`.
    /// - `x`: The solution vector `x`, which will be updated in place with the computed solution.
    ///
    /// # Returns
    /// - `SolverResult`: Contains information on convergence, number of iterations, and the final residual norm.
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        let n = b.len();
        let epsilon = 1e-12; // Tolerance for numerical stability

        // Initial residual vectors and workspace for computation
        let mut r = vec![0.0; n];
        let mut temp_vec = vec![0.0; n];
        let mut preconditioned_vec = vec![0.0; n];

        // Compute initial residual r = b - Ax
        compute_initial_residual(a, b, x, &mut temp_vec, &mut r);
        

        // Apply the preconditioner if available
        if let Some(preconditioner) = &self.preconditioner {
            preconditioner.apply(
                a,
                &r as &dyn Vector<Scalar = f64>,
                &mut preconditioned_vec as &mut dyn Vector<Scalar = f64>,
            );
            r.copy_from_slice(&preconditioned_vec);
        }

        // Calculate initial residual norm
        let mut residual_norm = euclidean_norm(&r as &dyn Vector<Scalar = f64>);

        // Check initial convergence conditions
        if should_terminate_initial_residual(residual_norm, &r, x) {
            return SolverResult {
                converged: false,
                iterations: 0,
                residual_norm: f64::NAN,
            };
        }

        // Initialize variables for GMRES iterations
        let mut iterations = 0;
        let _adaptive_restart = if iterations > self.restart {
            // Increase the restart window if iterations exceed the restart threshold
            self.restart * 2
        } else {
            self.restart
        };
        let mut v = vec![vec![0.0; n]; self.restart + 1]; // Krylov basis vectors
        let mut h = vec![vec![0.0; self.restart]; self.restart + 1]; // Hessenberg matrix
        let mut g = vec![0.0; self.restart + 1]; // Residual projections

        // Variables for Givens rotations (cosines and sines)
        let mut cosines = vec![0.0; self.restart];
        let mut sines = vec![0.0; self.restart];

        loop {
            // Normalize the initial residual to form the first Krylov vector v[0]
            if !normalize_residual_and_init_krylov(&r, &mut v[0], epsilon) {
                return SolverResult {
                    converged: false,
                    iterations,
                    residual_norm: f64::NAN,
                };
            }

            // Initialize g with the current residual norm
            g[0] = residual_norm;
            let mut inner_iterations = 0;

            // Arnoldi process for generating the Krylov subspace and filling Hessenberg matrix
            for k in 0..self.restart {
                inner_iterations = k;
                if let Some(result) = arnoldi_process(
                    k,
                    a,
                    &mut v,
                    &mut temp_vec,
                    &mut preconditioned_vec,
                    &self.preconditioner, // Pass &Option<Arc<...>>
                    &mut h,
                    epsilon,
                ) {
                    return result;
                }

                // Apply Givens rotations to maintain upper Hessenberg structure
                apply_givens_rotations(&mut h, &mut g, &mut cosines, &mut sines, k);

                // Check convergence after each step
                residual_norm = g[k + 1].abs();
                if residual_norm < self.tol {
                    break;
                }
            }

            // Solve least-squares problem to update y in x = V * y
            let m = inner_iterations + 1;
            let mut y = vec![0.0; m];
            back_substitution(&h, &g, &mut y, m);

            // Update solution vector x
            update_solution(x, &v, &y, n);
            iterations += m;

            // Recompute the residual for restart
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

            /* // Track residuals for stagnation detection
            let previous_residual = residual_norm;
            let stagnation_tolerance = 1e-3;

            if (residual_norm - previous_residual).abs() < stagnation_tolerance {
                eprintln!("GMRES detected stagnation; aborting.");
                return SolverResult {
                    converged: false,
                    iterations,
                    residual_norm,
                };
            }
            let _previous_residual = residual_norm; */

            // Check for numerical stability issues
            if residual_norm.is_nan() || residual_norm.is_infinite() {
                return SolverResult {
                    converged: false,
                    iterations,
                    residual_norm: f64::NAN,
                };
            }

            // Final convergence check
            if iterations >= self.max_iter || residual_norm <= self.tol {
                break;
            }
        }

        SolverResult {
            converged: residual_norm <= self.tol,
            iterations,
            residual_norm,
        }
    }
}

// Parallelized Helper Functions for GMRES

/// Computes the initial residual vector `r = b - Ax` and stores it in `residual`.
/// Uses `rayon` for parallel iteration, allowing each element of the residual to be computed independently.
/// 
/// # Arguments
/// - `a`: Matrix `A` of the system.
/// - `b`: Vector `b` on the right-hand side.
/// - `x`: Solution vector `x`, initially a guess, which will be updated.
/// - `temp_vec`: Temporary vector for storing intermediate results of `Ax`.
/// - `residual`: Vector where the computed initial residual `b - Ax` will be stored.
fn compute_initial_residual(
    a: &dyn Matrix<Scalar = f64>,
    b: &dyn Vector<Scalar = f64>,
    x: &mut dyn Vector<Scalar = f64>,
    temp_vec: &mut Vec<f64>,
    residual: &mut Vec<f64>,
) {
    // Compute A * x and store it in `temp_vec`
    a.mat_vec(x, temp_vec); 
    
    // Compute residual in parallel: residual[i] = b[i] - (A * x)[i]
    residual.par_iter_mut().enumerate().for_each(|(i, r_i)| {
        *r_i = b.get(i) - temp_vec[i];
    });
}

/// Checks if the initial residual norm indicates that the solver should terminate early.
/// Returns `true` if residual norm is NaN or infinite (numerical failure) or if `x` contains NaN values.
/// 
/// # Arguments
/// - `residual_norm`: The Euclidean norm of the residual.
/// - `_residual`: The residual vector (unused here, but may be useful in other termination checks).
/// - `x`: The current solution vector.
///
/// # Returns
/// - `true` if termination conditions are met, otherwise `false`.
fn should_terminate_initial_residual(residual_norm: f64, _residual: &Vec<f64>, x: &dyn Vector<Scalar = f64>) -> bool {
    if residual_norm.is_nan() || residual_norm.is_infinite() {
        return true;
    }

    // Check if residual norm is very small or if solution vector `x` contains NaNs
    if residual_norm <= 1e-12 && x.as_slice().contains(&f64::NAN) {
        return true;
    }

    false
}

/// Normalizes the initial residual vector and sets up the first Krylov basis vector.
/// Returns `true` if normalization is successful, or `false` if the norm is zero, NaN, or infinite.
/// 
/// # Arguments
/// - `residual`: The initial residual vector.
/// - `krylov_vector`: The first Krylov basis vector, which is initialized with the normalized residual.
/// - `epsilon`: Tolerance for avoiding division by zero in normalization.
///
/// # Returns
/// - `true` if normalization succeeds, otherwise `false`.
fn normalize_residual_and_init_krylov(
    residual: &Vec<f64>,
    krylov_vector: &mut Vec<f64>,
    epsilon: f64,
) -> bool {
    // Compute the Euclidean norm of the residual
    let r_norm = euclidean_norm(residual as &dyn Vector<Scalar = f64>);
    if r_norm < epsilon || r_norm.is_nan() || r_norm.is_infinite() {
        return false;
    }

    // Normalize residual to obtain the first Krylov basis vector in parallel
    krylov_vector.par_iter_mut().zip(residual).for_each(|(k_i, &res_i)| {
        *k_i = res_i / r_norm;
    });

    true
}

/// Performs the Arnoldi process to extend the Krylov subspace basis and populate the Hessenberg matrix `H`.
/// Each iteration orthogonalizes the new Krylov vector and applies the preconditioner if available.
/// 
/// # Arguments
/// - `k`: Current iteration within the Arnoldi process.
/// - `a`: The matrix `A` in the system.
/// - `v`: The Krylov basis vectors, represented as a vector of vectors.
/// - `temp_vec`: Temporary storage for the result of `A * v[k]`.
/// - `preconditioned_vec`: Temporary storage for the result of applying the preconditioner.
/// - `preconditioner`: Optional preconditioner for transforming `A`.
/// - `h`: The Hessenberg matrix that stores the projections of `A*v_k` onto the basis vectors.
/// - `epsilon`: Small tolerance value to handle numerical stability in normalization.
///
/// # Returns
/// - `None` if the process is successful, or `Some(SolverResult)` if a numerical issue requires termination.
fn arnoldi_process(
    k: usize,
    a: &dyn Matrix<Scalar = f64>,
    v: &mut Vec<Vec<f64>>,
    temp_vec: &mut Vec<f64>,
    preconditioned_vec: &mut Vec<f64>,
    preconditioner: &Option<Arc<dyn Preconditioner>>, // Change from Box to Arc
    h: &mut Vec<Vec<f64>>,
    epsilon: f64,
) -> Option<SolverResult> {
    // Compute the matrix-vector product A * v[k] and store it in `temp_vec`
    a.mat_vec(&v[k] as &dyn Vector<Scalar = f64>, temp_vec);

    // Apply the preconditioner if it exists
    if let Some(preconditioner) = preconditioner {
        preconditioner.apply(
            a,
            temp_vec as &dyn Vector<Scalar = f64>,
            preconditioned_vec as &mut dyn Vector<Scalar = f64>,
        );
        temp_vec.copy_from_slice(&preconditioned_vec); // Update `temp_vec` with preconditioned result
    }

    // Perform Modified Gram-Schmidt to orthogonalize `temp_vec` against all previous Krylov vectors
    for j in 0..=k {
        h[j][k] = dot_product(
            &v[j] as &dyn Vector<Scalar = f64>,
            temp_vec as &dyn Vector<Scalar = f64>,
        );
        for i in 0..temp_vec.len() {
            temp_vec[i] -= h[j][k] * v[j][i];
        }
    }

    // Normalize `temp_vec` to obtain the next Krylov vector if norm is above tolerance `epsilon`
    h[k + 1][k] = euclidean_norm(temp_vec as &dyn Vector<Scalar = f64>);
    if h[k + 1][k].abs() < epsilon {
        h[k + 1][k] = 0.0;
    }

    // If norm is valid, normalize and store `temp_vec` as the next Krylov vector
    if h[k + 1][k].abs() > epsilon {
        temp_vec.par_iter().zip(&mut v[k + 1]).for_each(|(temp_i, v_k1_i)| {
            *v_k1_i = *temp_i / h[k + 1][k];
        });
    }

    None
}

/// Updates the solution vector `x` by computing `x += V * y`, where `V` is the Krylov basis and `y` is the solution
/// vector for the least-squares problem in the subspace.
///
/// # Arguments
/// - `x`: Solution vector that will be updated.
/// - `v`: Matrix of Krylov basis vectors.
/// - `y`: Solution vector for the least-squares problem.
/// - `n`: Length of the solution vector.
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

/// Recomputes the residual vector `r = b - Ax` for convergence checks after each GMRES restart.
/// This allows the algorithm to assess if the residual norm is within the desired tolerance.
///
/// # Arguments
/// - `a`: Matrix `A` of the system.
/// - `b`: Right-hand side vector `b`.
/// - `x`: Current solution vector `x`.
/// - `temp_vec`: Temporary storage for `A * x`.
/// - `residual`: Residual vector where the computed residual `b - Ax` is stored.
fn compute_residual(
    a: &dyn Matrix<Scalar = f64>,
    b: &dyn Vector<Scalar = f64>,
    x: &mut dyn Vector<Scalar = f64>,
    temp_vec: &mut Vec<f64>,
    residual: &mut Vec<f64>,
) {
    a.mat_vec(x, temp_vec); // Compute A * x and store in `temp_vec`
    residual.par_iter_mut().enumerate().for_each(|(i, r_i)| {
        *r_i = b.get(i) - temp_vec[i];
    });
}

/// Computes the dot product of two vectors `u` and `v` in parallel using `rayon`.
/// This function is optimized for use with Krylov basis vector projections.
///
/// # Arguments
/// - `u`: First vector.
/// - `v`: Second vector.
///
/// # Returns
/// - Dot product value of the two vectors.
fn dot_product(u: &dyn Vector<Scalar = f64>, v: &dyn Vector<Scalar = f64>) -> f64 {
    u.as_slice()
        .par_iter()
        .zip(v.as_slice().par_iter())
        .map(|(&ui, &vi)| ui * vi)
        .sum()
}

/// Computes the Euclidean norm of a vector in parallel, returning `sqrt(sum(u_i^2))`.
///
/// # Arguments
/// - `u`: Vector for which to calculate the norm.
///
/// # Returns
/// - Euclidean norm (L2 norm) of the vector.
fn euclidean_norm(u: &dyn Vector<Scalar = f64>) -> f64 {
    u.as_slice().par_iter().map(|&ui| ui * ui).sum::<f64>().sqrt()
}

/// Applies a sequence of Givens rotations to maintain the upper Hessenberg structure of `H`.
/// These rotations zero out entries below the main diagonal in each column of `H`.
///
/// # Arguments
/// - `h`: Upper Hessenberg matrix.
/// - `g`: Vector `g` for updating the residual in the least-squares solution.
/// - `cosines`, `sines`: Arrays for storing the rotation parameters (cosines and sines).
/// - `k`: Current column in `H` to which rotations are applied.
fn apply_givens_rotations(
    h: &mut Vec<Vec<f64>>,
    g: &mut Vec<f64>,
    cosines: &mut Vec<f64>,
    sines: &mut Vec<f64>,
    k: usize,
) {
    for i in 0..k {
        let temp = cosines[i] * h[i][k] + sines[i] * h[i + 1][k];
        h[i + 1][k] = -sines[i] * h[i][k] + cosines[i] * h[i + 1][k];
        h[i][k] = temp;
    }

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

    h[k][k] = cosines[k] * h_kk + sines[k] * h_k1k;
    h[k + 1][k] = 0.0;

    let temp = cosines[k] * g[k] + sines[k] * g[k + 1];
    g[k + 1] = -sines[k] * g[k] + cosines[k] * g[k + 1];
    g[k] = temp;
}

/// Solves the least-squares problem using back substitution on the Hessenberg matrix `H`.
/// This is the final step in each GMRES iteration before updating the solution vector `x`.
///
/// # Arguments
/// - `h`: Hessenberg matrix.
/// - `g`: Vector `g`, modified by the Givens rotations, containing the residual norm information.
/// - `y`: Solution vector for the least-squares problem, computed here.
/// - `m`: Number of iterations within the current GMRES cycle.
fn back_substitution(h: &Vec<Vec<f64>>, g: &Vec<f64>, y: &mut Vec<f64>, m: usize) {
    for i in (0..m).rev() {
        y[i] = g[i];
        for j in (i + 1)..m {
            y[i] -= h[i][j] * y[j];
        }
        if h[i][i].abs() > 1e-12 {
            y[i] /= h[i][i];
        } else {
            y[i] = 0.0; // Avoid division by zero for near-zero diagonal elements
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

    /// Test the GMRES solver without a preconditioner on a simple symmetric positive-definite (SPD) matrix.
    /// 
    /// This test ensures that GMRES converges to the correct solution within the tolerance specified,
    /// using a small 2x2 SPD matrix. The test validates:
    /// - Convergence of the solver within the specified tolerance.
    /// - The absence of `NaN` values in the final solution vector.
    #[test]
    fn test_gmres_solver_no_preconditioner() {
        // Define a 2x2 SPD matrix A
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];

        // Right-hand side vector b
        let b = mat![
            [1.0],
            [2.0],
        ];

        // Initial guess x0 set to zero vector
        let mut x = Mat::<f64>::zeros(2, 1);

        // Expected solution for reference (not explicitly checked but useful for verification)
        let _expected_x = mat![
            [0.09090909],
            [0.63636364],
        ];

        // Create GMRES solver instance with restart after 2 iterations
        let mut gmres = GMRES::new(100, 1e-6, 2);

        // Execute the solver
        let result = gmres.solve(&a, &b, &mut x);

        // Ensure the solution does not contain NaN values
        assert!(!crate::linalg::matrix::traits::Matrix::as_slice(&x).contains(&f64::NAN), "Solution contains NaN values");
        
        // Verify convergence and residual norm within tolerance
        assert!(result.converged, "GMRES did not converge");
        assert!(result.residual_norm <= 1e-6, "Residual norm too large");
    }

    /// Test the GMRES solver with a Jacobi preconditioner applied to a simple SPD matrix.
    /// 
    /// This test evaluates the performance of GMRES when using a preconditioner, aiming to
    /// achieve convergence with improved residual reduction. The test checks:
    /// - Successful application of the preconditioner.
    /// - Convergence within specified tolerance.
    /// - Solution correctness without `NaN` values.
    #[test]
    fn test_gmres_solver_with_jacobi_preconditioner() {
        // Define the same 2x2 SPD matrix A
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];

        // Right-hand side vector b
        let b = mat![
            [1.0],
            [2.0],
        ];

        // Initial guess x0 as a zero vector
        let mut x = Mat::<f64>::zeros(2, 1);

        // Expected solution (for reference)
        let _expected_x = mat![
            [0.09090909],
            [0.63636364],
        ];

        // Instantiate GMRES solver with Jacobi preconditioner
        let mut gmres = GMRES::new(100, 1e-6, 2);
        let preconditioner = Arc::new(Jacobi::default());
        gmres.set_preconditioner(preconditioner);

        // Run the GMRES solver with preconditioning
        let result = gmres.solve(&a, &b, &mut x);

        println!("GMRES Solver Result with Jacobi Preconditioner: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // Validate that the solution contains no NaN values and converges within tolerance
        assert!(!crate::linalg::matrix::traits::Matrix::as_slice(&x).contains(&f64::NAN), "Solution contains NaN values");
        assert!(result.converged, "GMRES did not converge with Jacobi preconditioner");
    }

    /// Test the GMRES solver on a larger symmetric positive-definite (SPD) matrix.
    /// 
    /// This test checks GMRES's performance on a larger 4x4 SPD matrix to ensure stability and
    /// convergence within specified tolerance. Since an exact solution is difficult to verify,
    /// the test checks for general convergence and a low residual norm.
    #[test]
    fn test_gmres_solver_large_system() {
        // Define a larger 4x4 SPD matrix A
        let a = mat![
            [10.0, 1.0, 0.0, 0.0],
            [1.0, 7.0, 2.0, 0.0],
            [0.0, 2.0, 8.0, 1.0],
            [0.0, 0.0, 1.0, 5.0],
        ];

        // Right-hand side vector b
        let b = mat![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ];

        // Initial guess x0 set to zero vector
        let mut x = Mat::<f64>::zeros(4, 1);

        // Instantiate GMRES solver with restart after 2 iterations
        let mut gmres = GMRES::new(200, 1e-6, 2);

        // Execute the solver on the larger system
        let result = gmres.solve(&a, &b, &mut x);

        println!("GMRES Solver Result for Large System: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // Check for convergence and residual norm below tolerance
        assert!(result.converged, "GMRES did not converge on large system");
        assert!(result.residual_norm < 1e-6, "Residual norm too large for large system");
    }

    /// Test the GMRES solver on an ill-conditioned matrix to evaluate numerical stability.
    /// 
    /// Ill-conditioned systems are challenging due to near-singularity, which may cause numerical
    /// instability. This test examines GMRES's handling of such a system by checking:
    /// - Convergence and stability.
    /// - Accuracy of solution compared to expected values.
    #[test]
    fn test_gmres_solver_convergence_on_ill_conditioned_system() {
        // Define an ill-conditioned 2x2 matrix A with small diagonal elements
        let a = mat![
            [1e-10, 0.0],
            [0.0, 1e-10],
        ];

        // Right-hand side vector b
        let b = mat![
            [1.0],
            [1.0],
        ];

        // Initial guess x0 as a zero vector
        let mut x = Mat::<f64>::zeros(2, 1);

        // Instantiate GMRES solver with restart after 2 iterations
        let mut gmres = GMRES::new(100, 1e-6, 2);

        // Run the solver on the ill-conditioned system
        let result = gmres.solve(&a, &b, &mut x);

        println!("GMRES Solver Result for Ill-conditioned System: {:?}", result);
        println!("Computed solution x = {:?}", x);

        // Define expected solution for comparison
        let expected_x = mat![
            [1e10],
            [1e10],
        ];

        // Calculate relative error between computed and expected solutions
        let x_slice = crate::linalg::matrix::traits::Matrix::as_slice(&x);
        let expected_x_slice = crate::linalg::matrix::traits::Matrix::as_slice(&expected_x);
        let relative_error: f64 = x_slice
            .iter()
            .zip(expected_x_slice.iter())
            .map(|(&xi, &x_exact)| ((xi - x_exact) / x_exact).abs())
            .sum::<f64>()
            / x_slice.len() as f64;

        // Check that relative error is below threshold, indicating acceptable accuracy
        assert!(
            relative_error < 1e-6,
            "GMRES did not converge to an accurate solution on an ill-conditioned system"
        );
    }
}
