Generate a detailed users guide for the `Solver` module for Hydra. I am going to provide the code for all of the parts of the `Solver` module below, and you can analyze and build the detailed outline based on this version of the source code.

`src/solver/mod.rs`

```rust
//! Main module for the solver interface in Hydra.
//!
//! This module houses the Krylov solvers and preconditioners,
//! facilitating flexible solver selection.
//! 
pub mod ksp;
pub mod cg;
pub mod preconditioner;
pub mod gmres;

pub use ksp::KSP;
pub use cg::ConjugateGradient;
pub use gmres::GMRES;

#[cfg(test)]
mod tests;
```

---

`src/solver/ksp.rs`

```rust

//! Enhancements to the KSP module to introduce an interface adapter for flexible usage.
//!
//! This adds the `SolverManager` for high-level integration of solvers and preconditioners.

use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use std::sync::Arc;

#[derive(Debug)]
pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
}

/// KSP trait for Krylov solvers, encompassing solvers like CG and GMRES.
pub trait KSP {
    fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult;
}

/// Struct representing a high-level interface for managing solver configuration.
pub struct SolverManager {
    solver: Box<dyn KSP>,
    preconditioner: Option<Arc<dyn Preconditioner>>,
}

impl SolverManager {
    /// Creates a new `SolverManager` instance with a specified solver.
    ///
    /// # Arguments
    /// - `solver`: The Krylov solver to be used.
    ///
    /// # Returns
    /// A new `SolverManager` instance.
    pub fn new(solver: Box<dyn KSP>) -> Self {
        SolverManager {
            solver,
            preconditioner: None,
        }
    }

    /// Sets a preconditioner for the solver.
    ///
    /// # Arguments
    /// - `preconditioner`: The preconditioner to be used.
    pub fn set_preconditioner(&mut self, preconditioner: Arc<dyn Preconditioner>) {
        self.preconditioner = Some(preconditioner);
    }

    /// Solves a system `Ax = b` using the configured solver and optional preconditioner.
    ///
    /// # Arguments
    /// - `a`: The system matrix `A`.
    /// - `b`: The right-hand side vector `b`.
    /// - `x`: The solution vector `x`, which will be updated with the computed solution.
    ///
    /// # Returns
    /// A `SolverResult` containing convergence information and the final residual norm.
    pub fn solve(
        &mut self,
        a: &dyn Matrix<Scalar = f64>,
        b: &dyn Vector<Scalar = f64>,
        x: &mut dyn Vector<Scalar = f64>,
    ) -> SolverResult {
        if let Some(preconditioner) = &self.preconditioner {
            preconditioner.apply(a, b, x);
        }
        self.solver.solve(a, b, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::preconditioner::Jacobi;
    use crate::solver::cg::ConjugateGradient;
    use faer::{mat, Mat};

    #[test]
    fn test_solver_manager_with_jacobi_preconditioner() {
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0],
        ];
        let b = mat![
            [1.0],
            [2.0],
        ];
        let mut x = Mat::<f64>::zeros(2, 1);

        // Initialize CG solver and solver manager
        let cg_solver = ConjugateGradient::new(100, 1e-6);
        let mut solver_manager = SolverManager::new(Box::new(cg_solver));

        // Set Jacobi preconditioner
        let jacobi_preconditioner = Arc::new(Jacobi::default());
        solver_manager.set_preconditioner(jacobi_preconditioner);

        // Solve the system
        let result = solver_manager.solve(&a, &b, &mut x);

        // Validate results
        assert!(result.converged, "Solver did not converge");
        assert!(result.residual_norm <= 1e-6, "Residual norm too large");
        assert!(
            !crate::linalg::vector::traits::Vector::as_slice(&x).contains(&f64::NAN),
            "Solution contains NaN values"
        );
    }
}
```

---

`src/solver/cg.rs`

```rust
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
```

---

`src/solver/gmres.rs`

```rust
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
```

---

`src/solver/tests.rs`

```rust
// src/solver/tests.rs

use std::sync::Arc;

use crate::solver::{ConjugateGradient, GMRES, KSP};
use crate::solver::preconditioner::{Jacobi, LU, ILU};
use faer::mat;
use faer::Mat;

use super::preconditioner::Preconditioner;

const TOLERANCE: f64 = 1e-6;

/// Test the Conjugate Gradient (CG) solver without a preconditioner.
#[test]
fn test_cg_solver_no_preconditioner() {
    let a = mat![
        [4.0, 1.0],
        [1.0, 3.0]
    ];
    let b = mat![
        [1.0],
        [2.0]
    ];
    let mut x = Mat::<f64>::zeros(2, 1);
    let expected_x = mat![
        [0.09090909],
        [0.63636364]
    ];

    let mut cg = ConjugateGradient::new(100, TOLERANCE);
    let result = cg.solve(&a, &b, &mut x);

    assert!(result.converged, "Conjugate Gradient did not converge.");
    assert!(result.residual_norm <= TOLERANCE, "Residual norm too large.");
    for i in 0..x.nrows() {
        assert!(
            (x.read(i, 0) - expected_x.read(i, 0)).abs() < TOLERANCE,
            "x[{}] = {}, expected {}",
            i,
            x.read(i, 0),
            expected_x.read(i, 0)
        );
    }
}

/// Test CG solver with Jacobi preconditioner.
#[test]
fn test_cg_solver_with_jacobi_preconditioner() {
    let a = mat![
        [4.0, 1.0],
        [1.0, 3.0]
    ];
    let b = mat![
        [1.0],
        [2.0]
    ];
    let mut x = Mat::<f64>::zeros(2, 1);
    let _expected_x = mat![
        [0.09090909],
        [0.63636364]
    ];

    let mut cg = ConjugateGradient::new(100, TOLERANCE);
    cg.set_preconditioner(Box::new(Jacobi::default()));
    let result = cg.solve(&a, &b, &mut x);

    assert!(result.converged, "CG with Jacobi preconditioner did not converge.");
    assert!(result.residual_norm <= TOLERANCE, "Residual norm too large.");
}

/// Test GMRES solver without preconditioner on symmetric positive-definite (SPD) matrix.
#[test]
fn test_gmres_solver_no_preconditioner() {
    let a = mat![
        [4.0, 1.0],
        [1.0, 3.0]
    ];
    let b = mat![
        [1.0],
        [2.0]
    ];
    let mut x = Mat::<f64>::zeros(2, 1);

    let mut gmres = GMRES::new(100, TOLERANCE, 2);
    let result = gmres.solve(&a, &b, &mut x);

    assert!(result.converged, "GMRES did not converge.");
    assert!(result.residual_norm <= TOLERANCE, "Residual norm too large.");
}

/// Test GMRES solver with LU preconditioner on a small matrix.
#[test]
fn test_gmres_solver_with_lu_preconditioner() {
    let a = mat![
        [4.0, 1.0],
        [1.0, 3.0]
    ];
    let b = mat![
        [1.0],
        [2.0]
    ];
    let mut x = Mat::<f64>::zeros(2, 1);

    let mut gmres = GMRES::new(100, TOLERANCE, 2);
    gmres.set_preconditioner(Arc::new(LU::new(&a)));
    let result = gmres.solve(&a, &b, &mut x);

    assert!(result.converged, "GMRES with LU preconditioner did not converge.");
    assert!(result.residual_norm <= TOLERANCE, "Residual norm too large.");
}

/// Test for ILU preconditioner application.
#[test]
fn test_ilu_preconditioner() {
    let matrix = mat![
        [10.0, 1.0, 0.0],
        [1.0, 7.0, 2.0],
        [0.0, 2.0, 8.0]
    ];
    let r = vec![11.0, 10.0, 10.0]; // Adjusted RHS vector
    let expected_z = vec![1.0, 1.0, 1.0];
    let mut z = vec![0.0; 3];

    let ilu_preconditioner = ILU::new(&matrix);
    ilu_preconditioner.apply(&matrix, &r, &mut z);

    for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
        println!("Index {}: computed = {}, expected = {}", i, computed, expected);
        assert!(
            (computed - expected).abs() < TOLERANCE,
            "ILU preconditioner produced unexpected result at index {}: computed {}, expected {}",
            i,
            computed,
            expected
        );
    }
}

```

---

`src/solver/preconditioner/mod.rs`

```rust
pub mod jacobi;
pub mod lu;
pub mod ilu;
pub mod cholesky;

pub use jacobi::Jacobi;
pub use lu::LU;
pub use ilu::ILU;
pub use cholesky::CholeskyPreconditioner;

use crate::linalg::{Matrix, Vector};
use faer::mat::Mat;
use std::sync::Arc;

/// Preconditioner trait with `Arc` for easier integration and thread safety.
pub trait Preconditioner: Send + Sync {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>);
}

/// `PreconditionerFactory` provides static methods to create common preconditioners.
/// This design promotes flexible creation and integration of preconditioners in a modular way.
pub struct PreconditionerFactory;

impl PreconditionerFactory {
    /// Creates a `Jacobi` preconditioner wrapped in `Arc`.
    pub fn create_jacobi() -> Arc<dyn Preconditioner> {
        Arc::new(Jacobi::default())
    }

    /// Creates an `ILU` preconditioner wrapped in `Arc` from a provided matrix.
    ///
    /// # Arguments
    /// - `matrix`: The matrix to use for constructing the ILU preconditioner.
    ///
    /// # Returns
    /// `Arc<dyn Preconditioner>` instance of the ILU preconditioner.
    pub fn create_ilu(matrix: &Mat<f64>) -> Arc<dyn Preconditioner> {
        Arc::new(ILU::new(matrix))
    }

    /// Creates a `CholeskyPreconditioner` wrapped in `Arc` from a provided matrix.
    ///
    /// # Arguments
    /// - `matrix`: The symmetric positive definite matrix to use for Cholesky decomposition.
    ///
    /// # Returns
    /// Result containing `Arc<dyn Preconditioner>` or an error if decomposition fails.
    pub fn create_cholesky(matrix: &Mat<f64>) -> Result<Arc<dyn Preconditioner>, Box<dyn std::error::Error>> {
        let preconditioner = CholeskyPreconditioner::new(matrix)?;
        Ok(Arc::new(preconditioner))
    }

    /// Creates an `LU` preconditioner wrapped in `Arc` from a provided matrix.
    ///
    /// # Arguments
    /// - `matrix`: The matrix to use for LU decomposition.
    ///
    /// # Returns
    /// `Arc<dyn Preconditioner>` instance of the LU preconditioner.
    pub fn create_lu(matrix: &Mat<f64>) -> Arc<dyn Preconditioner> {
        Arc::new(LU::new(matrix))
    }
}
```

---

`src/solver/preconditioner/cholesky.rs`

```rust
// src/solver/preconditioner/cholesky.rs
use faer::{mat::Mat, solvers::{Cholesky, SpSolver}, Side}; // Import Side for cholesky method argument
use crate::{linalg::Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use std::error::Error;

/// `CholeskyPreconditioner` holds the Cholesky decomposition result to
/// precondition a system for CG methods on symmetric positive definite matrices.
pub struct CholeskyPreconditioner {
    /// The Cholesky decomposition result
    l_factor: Cholesky<f64>,
}

impl CholeskyPreconditioner {
    /// Creates a new `CholeskyPreconditioner` by computing the Cholesky decomposition
    /// of a symmetric positive definite matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A reference to the matrix to be decomposed. Must be symmetric and positive definite.
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` containing the preconditioner if successful, or an error if the decomposition fails.
    pub fn new(matrix: &Mat<f64>) -> Result<Self, Box<dyn Error>> {
        // Specify the side for Cholesky decomposition as Lower
        let l_factor = matrix.cholesky(Side::Lower).map_err(|_| "Cholesky decomposition failed")?;
        Ok(Self { l_factor })
    }

    /// Applies the preconditioner to a given right-hand side vector to produce a preconditioned solution.
    ///
    /// # Arguments
    ///
    /// * `rhs` - A reference to the right-hand side matrix (vector) for the system.
    ///
    /// # Returns
    ///
    /// * `Ok(Mat<f64>)` containing the preconditioned solution, or an error if solving fails.
    /// Applies the preconditioner to a given vector `rhs` and returns the solution.
    ///
    /// # Arguments
    /// * `rhs` - The right-hand side vector as a `Mat<f64>`.
    pub fn apply(&self, rhs: &Mat<f64>) -> Result<Mat<f64>, Box<dyn Error>> {
        Ok(self.l_factor.solve(rhs))
    }
}

impl Preconditioner for CholeskyPreconditioner {
    fn apply(&self, _a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let rhs_mat = Mat::from_fn(r.len(), 1, |i, _| r.get(i));
        if let Ok(solution) = self.apply(&rhs_mat) {
            for i in 0..z.len() {
                z.set(i, solution[(i, 0)]);
            }
        }
    }
}

/// Example function to apply Cholesky preconditioning in the CG algorithm.
/// This function returns the preconditioned solution.
///
/// # Arguments
///
/// * `matrix` - The system matrix.
/// * `rhs` - The right-hand side vector.
///
/// # Returns
///
/// * `Result<Mat<f64>, Box<dyn Error>>` containing the solution vector or an error if any stage fails.
pub fn apply_cholesky_preconditioner(matrix: &Mat<f64>, rhs: &Mat<f64>) -> Result<Mat<f64>, Box<dyn Error>> {
    // Initialize the preconditioner
    let preconditioner = CholeskyPreconditioner::new(matrix)?;
    
    // Apply the preconditioner to obtain the preconditioned right-hand side
    preconditioner.apply(rhs)
}

// src/solver/preconditioner/cholesky.rs

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat::Mat;
    use approx::assert_relative_eq; // For floating-point comparisons

    #[test]
    fn test_cholesky_preconditioner_creation() {
        // Define a symmetric positive definite matrix (2x2 for simplicity)
        let mut matrix = Mat::<f64>::zeros(2, 2);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = 1.0;
        matrix[(1, 0)] = 1.0;
        matrix[(1, 1)] = 3.0;

        // Attempt to create the Cholesky preconditioner
        let preconditioner = CholeskyPreconditioner::new(&matrix);
        assert!(preconditioner.is_ok(), "Failed to create Cholesky preconditioner");
    }

    #[test]
    fn test_cholesky_preconditioner_application() {
        // Define a symmetric positive definite matrix (2x2 for simplicity)
        let mut matrix = Mat::<f64>::zeros(2, 2);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = 1.0;
        matrix[(1, 0)] = 1.0;
        matrix[(1, 1)] = 3.0;

        // Define a right-hand side vector
        let mut rhs = Mat::<f64>::zeros(2, 1);
        rhs[(0, 0)] = 1.0;
        rhs[(1, 0)] = 2.0;

        // Initialize the preconditioner
        let preconditioner = CholeskyPreconditioner::new(&matrix).expect("Preconditioner creation failed");

        // Apply the preconditioner
        let result = preconditioner.apply(&rhs).expect("Preconditioner application failed");

        // Define expected output (corrected values)
        let mut expected = Mat::<f64>::zeros(2, 1);
        expected[(0, 0)] = 0.0909091; // Corrected expected value
        expected[(1, 0)] = 0.6363636;

        // Verify results using approximate equality
        assert_relative_eq!(result[(0, 0)], expected[(0, 0)], epsilon = 1e-6);
        assert_relative_eq!(result[(1, 0)], expected[(1, 0)], epsilon = 1e-6);
    }

    #[test]
    fn test_apply_cholesky_preconditioner_function() {
        // Define a symmetric positive definite matrix (2x2 for simplicity)
        let mut matrix = Mat::<f64>::zeros(2, 2);
        matrix[(0, 0)] = 4.0;
        matrix[(0, 1)] = 1.0;
        matrix[(1, 0)] = 1.0;
        matrix[(1, 1)] = 3.0;

        // Define a right-hand side vector
        let mut rhs = Mat::<f64>::zeros(2, 1);
        rhs[(0, 0)] = 1.0;
        rhs[(1, 0)] = 2.0;

        // Use the apply_cholesky_preconditioner function
        let result = apply_cholesky_preconditioner(&matrix, &rhs).expect("Function application failed");

        // Define expected output (corrected values)
        let mut expected = Mat::<f64>::zeros(2, 1);
        expected[(0, 0)] = 0.0909091; // Corrected expected value
        expected[(1, 0)] = 0.6363636;

        // Verify results using approximate equality
        assert_relative_eq!(result[(0, 0)], expected[(0, 0)], epsilon = 1e-6);
        assert_relative_eq!(result[(1, 0)], expected[(1, 0)], epsilon = 1e-6);
    }
}
```

---

`src/solver/preconditioner/ilu.rs`

```rust
//! ILU Preconditioner approximation using a custom sparse ILU decomposition.
//!
//! This ILU preconditioner approximates the inverse of a matrix using a sparse LU
//! factorization method. It is especially effective for preconditioning iterative
//! solvers, improving convergence by preserving the original sparsity pattern.

use faer::mat::Mat;
use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;

/// ILU struct representing the incomplete LU factorization of a matrix.
pub struct ILU {
    l: Mat<f64>,
    u: Mat<f64>,
}

impl ILU {
    /// Constructs an ILU preconditioner for a given sparse matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A sparse matrix represented as a dense matrix `Mat<f64>`, to be factorized.
    ///
    /// # Returns
    ///
    /// Returns an `ILU` instance with L and U matrices approximating A.
    pub fn new(matrix: &Mat<f64>) -> Self {
        let n = matrix.nrows();
        let mut l = Mat::identity(n,n);
        let mut u = matrix.clone();

        // Perform ILU decomposition while preserving sparsity
        for i in 0..n {
            for j in (i+1)..n {
                if matrix.read(j, i) != 0.0 {
                    // Calculate L[j, i] as the scaling factor
                    let factor = u.read(j, i) / u.read(i, i);
                    l.write(j, i, factor);

                    // Update U row j based on the factor, preserving sparsity
                    for k in i..n {
                        let new_value = u.read(j, k) - factor * u.read(i, k);
                        if new_value.abs() > 1e-10 { // Keep sparsity by discarding small values
                            u.write(j, k, new_value);
                        } else {
                            u.write(j, k, 0.0); // Set small values to zero
                        }
                    }
                }
            }
        }

        ILU { l, u }
    }

    /// Applies the ILU preconditioner to solve `L * U * x = r`.
    ///
    /// This method uses forward and backward substitution to apply
    /// the preconditioned solution.
    fn apply_ilu(&self, rhs: &[f64], solution: &mut [f64]) {
        let n = rhs.len();
        let mut y = vec![0.0; n];

        // Forward substitution: solve L * y = rhs
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= self.l.read(i, j) * y[j];
            }
            y[i] = sum / self.l.read(i, i);
        }

        // Backward substitution: solve U * x = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i+1)..n {
                sum -= self.u.read(i, j) * solution[j];
            }
            solution[i] = sum / self.u.read(i, i);
        }
    }
}

impl Preconditioner for ILU {
    /// Applies the ILU preconditioner to the vector `r`, storing the result in `z`.
    fn apply(&self, _a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let mut intermediate = vec![0.0; r.len()];
        self.apply_ilu(r.as_slice(), &mut intermediate);

        // Copy the intermediate result into the solution vector `z`
        for i in 0..z.len() {
            z.set(i, intermediate[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    /// Tests that the ILU preconditioner produces results close to the expected solution for a simple case.
    #[test]
    fn test_ilu_preconditioner_simple() {
        let matrix = mat![
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0]
        ];
        let r = vec![5.0, 5.0, 3.0];
        let expected_z = vec![1.0, 1.0, 1.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }

    /// Tests that the ILU preconditioner behaves correctly with an identity matrix.
    #[test]
    fn test_ilu_preconditioner_identity() {
        let matrix = mat![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        assert_eq!(z, r, "For identity matrix, output should match input vector exactly.");
    }

    /// Tests the ILU preconditioner on a larger, sparse matrix to verify it maintains sparsity.
    #[test]
    fn test_ilu_preconditioner_sparse() {
        let matrix = mat![
            [10.0, 0.0, 2.0, 0.0],
            [3.0, 9.0, 0.0, 0.0],
            [0.0, 7.0, 8.0, 0.0],
            [0.0, 0.0, 6.0, 5.0]
        ];
        let r = vec![12.0, 12.0, 15.0, 11.0];
        let mut z = vec![0.0; 4];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        let expected_z = vec![1.0, 1.0, 1.0, 1.0];
        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }

    /// Tests behavior of the ILU preconditioner with a singular matrix.
    #[test]
    fn test_ilu_preconditioner_singular() {
        let matrix = mat![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ];
        let r = vec![6.0, 12.0, 18.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        // Since this matrix is singular, we expect NaNs or other indicators of failure
        assert!(
            z.iter().any(|&val| val.is_nan() || val.abs() > 1e6),
            "For singular matrix, solution should indicate failure (NaNs or very large values)."
        );
    }

    /// Tests the ILU preconditioner on a non-trivial system with non-zero off-diagonal elements.
    #[test]
    fn test_ilu_preconditioner_non_trivial() {
        let matrix = mat![
            [2.0, 3.0, 1.0],
            [6.0, 1.0, 4.0],
            [0.0, 2.0, 8.0]
        ];
        let r = vec![3.0, 10.0, 8.0];
        let expected_z = vec![1.0, 0.0, 1.0];
        let mut z = vec![0.0; 3];

        let ilu_preconditioner = ILU::new(&matrix);
        ilu_preconditioner.apply(&matrix, &r, &mut z);

        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }
}
```

---

`src/solver/preconditioner/jacobi.rs`

```rust
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;

// Example of a Jacobi preconditioner using Arc<Mutex<T>> for safe parallelism
#[derive(Default)]
pub struct Jacobi;

impl Preconditioner for Jacobi {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let z = Arc::new(Mutex::new(z));  // Wrap z in Arc<Mutex<T>> for thread-safe access
        let a_rows: Vec<usize> = (0..a.nrows()).collect();

        // Use par_iter to process each row in parallel
        a_rows.into_par_iter().for_each(|i| {
            let ai = a.get(i, i);
            if ai != 0.0 {
                let ri = r.get(i);

                // Lock the mutex to get mutable access to z
                let mut z_guard = z.lock().unwrap();
                z_guard.set(i, ri / ai);  // Set the value in z (z[i] = r[i] / a[i][i])
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;  // Import the Jacobi preconditioner from the parent module
    use faer::{mat, Mat};

    #[test]
    fn test_jacobi_preconditioner_simple() {
        // Create a simple diagonal matrix 'a'
        let a = mat![
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0]
        ];

        // Create a right-hand side vector 'r'
        let r = mat![
            [8.0],
            [9.0],
            [4.0]
        ];

        // Expected result after applying the Jacobi preconditioner
        let expected_z = mat![
            [2.0],  // 8 / 4
            [3.0],  // 9 / 3
            [2.0],  // 4 / 2
        ];

        // Initialize an empty result vector 'z'
        let mut z = Mat::<f64>::zeros(3, 1);

        // Create a Jacobi preconditioner and apply it
        let jacobi = Jacobi;
        jacobi.apply(&a, &r, &mut z);

        // Verify the result
        for i in 0..z.nrows() {
            assert_eq!(z.read(i, 0), expected_z.read(i, 0));
        }
    }

    #[test]
    fn test_jacobi_preconditioner_with_zero_diagonal() {
        // Create a diagonal matrix 'a' with a zero diagonal entry
        let a = mat![
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  // Zero on the diagonal
            [0.0, 0.0, 2.0]
        ];

        // Create a right-hand side vector 'r'
        let r = mat![
            [8.0],
            [9.0],
            [4.0]
        ];

        // Expected result: The second row should not be updated due to zero diagonal
        let expected_z = mat![
            [2.0],  // 8 / 4
            [0.0],  // Division by zero, should leave z[i] = 0.0
            [2.0],  // 4 / 2
        ];

        // Initialize an empty result vector 'z'
        let mut z = Mat::<f64>::zeros(3, 1);

        // Create a Jacobi preconditioner and apply it
        let jacobi = Jacobi;
        jacobi.apply(&a, &r, &mut z);

        // Verify the result, with zero handling
        for i in 0..z.nrows() {
            assert_eq!(z.read(i, 0), expected_z.read(i, 0));
        }
    }

    #[test]
    fn test_jacobi_preconditioner_large_matrix() {
        // Create a larger diagonal matrix 'a'
        let n = 100;
        let mut a = Mat::<f64>::zeros(n, n);
        let mut r = Mat::<f64>::zeros(n, 1);
        let mut expected_z = Mat::<f64>::zeros(n, 1);

        // Fill 'a' and 'r' with values
        for i in 0..n {
            a.write(i, i, (i + 1) as f64);  // Diagonal matrix with increasing values
            r.write(i, 0, (i + 1) as f64 * 2.0);  // Right-hand side vector
            expected_z.write(i, 0, 2.0);  // Expected result (since r[i] = 2 * a[i])
        }

        // Initialize an empty result vector 'z'
        let mut z = Mat::<f64>::zeros(n, 1);

        // Create a Jacobi preconditioner and apply it
        let jacobi = Jacobi;
        jacobi.apply(&a, &r, &mut z);

        // Verify the result
        for i in 0..z.nrows() {
            assert_eq!(z.read(i, 0), expected_z.read(i, 0));
        }
    }
}
```

---

`src/solver/preconditioner/lu.rs`

```rust
//! LU Preconditioner implementation using Faer's LU decomposition.
//!
//! This module provides an implementation of an LU preconditioner that leverages
//! Faer's high-performance LU decomposition routines. The preconditioner is designed
//! to solve linear systems efficiently by preconditioning them using partial pivot LU decomposition.
//!
//! ## Usage
//! The `LU` struct can be instantiated with any square matrix, which it decomposes
//! using partial pivoting. It then provides an efficient method to apply the preconditioner
//! to a given right-hand side vector, solving for the preconditioned solution.

use faer::{solvers::PartialPivLu, mat::Mat};
use faer::solvers::SpSolver;
use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;

/// LU preconditioner struct that holds the LU decomposition of a matrix.
///
/// The `LU` preconditioner uses partial pivot LU decomposition to enable efficient
/// solution of linear systems. It stores the decomposition internally and provides
/// methods for solving systems via the preconditioner.
pub struct LU {
    lu_decomp: PartialPivLu<f64>,
}

impl LU {
    /// Constructs a new LU preconditioner by performing an LU decomposition on the input matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A reference to a square matrix for which the LU decomposition will be computed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use faer::mat;
    /// use hydra::solver::preconditioner::LU;  // Corrected import
    ///
    /// let matrix = mat![
    ///     [4.0, 3.0],
    ///     [6.0, 3.0]
    /// ];
    ///
    /// let lu_preconditioner = LU::new(&matrix);
    /// ```
    pub fn new(matrix: &Mat<f64>) -> Self {
        let lu_decomp = PartialPivLu::new(matrix.as_ref());
        LU { lu_decomp }
    }

    /// Applies the LU preconditioner to the right-hand side vector `rhs`, storing the solution in `solution`.
    ///
    /// This function initializes a column matrix from `rhs`, then uses the stored LU decomposition
    /// to solve the system `LU * x = rhs`. The solution is then copied into the `solution` array.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side vector for which the solution is to be computed.
    /// * `solution` - The mutable vector where the solution will be stored.
    ///
    /// # Panics
    ///
    /// This function will panic if the dimensions of `rhs` and `solution` do not match.
    fn apply(&self, rhs: &[f64], solution: &mut [f64]) {
        let mut sol_matrix = Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);

        // Solve using LU decomposition and specify `as_slice` method for Vector trait
        self.lu_decomp.solve_in_place(sol_matrix.as_mut());
        solution.copy_from_slice(&<dyn Vector<Scalar = f64>>::as_slice(&sol_matrix));
    }
}

impl Preconditioner for LU {
    /// Applies the LU preconditioner to a given vector `r` and stores the result in `z`.
    ///
    /// This implementation creates an intermediate vector, applies the LU preconditioner,
    /// and then populates `z` with the solution.
    ///
    /// # Arguments
    ///
    /// * `_a` - The matrix, not used directly in this preconditioner.
    /// * `r` - The vector to which the preconditioner is applied.
    /// * `z` - The vector where the preconditioned result is stored.
    ///
    /// # Example
    ///
    /// ```rust
    /// use faer::mat;
    /// use hydra::solver::preconditioner::Preconditioner;
    /// use hydra::solver::preconditioner::LU;
    ///
    /// let a = mat![
    ///     [4.0, 3.0],
    ///     [6.0, 3.0]
    /// ];
    /// let lu = LU::new(&a);
    /// let r = vec![5.0, 3.0];
    /// let mut z = vec![0.0, 0.0];
    /// lu.apply(&a, &r, &mut z); // Updated to pass `a` as first argument
    /// ```
    fn apply(&self, _a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let mut intermediate = vec![0.0; r.len()];
        self.apply(r.as_slice(), &mut intermediate);
        for i in 0..z.len() {
            z.set(i, intermediate[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    const TOLERANCE: f64 = 1e-4;

    /// Tests that the LU preconditioner with an identity matrix
    /// returns the input vector unchanged.
    #[test]
    fn test_lu_preconditioner_identity() {
        let identity = mat![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let r = vec![1.0, 2.0, 3.0];
        let mut z = vec![0.0; 3];

        let lu_preconditioner = LU::new(&identity);
        lu_preconditioner.apply(&r, &mut z);

        println!("Input vector r: {:?}", r);
        println!("Output solution vector z: {:?}", z);
        assert_eq!(z, r, "Solution vector z should match the input vector r for the identity matrix.");
    }

    /// Tests that the LU preconditioner works on a simple 2x2 matrix.
    #[test]
    fn test_lu_preconditioner_simple() {
        let matrix = mat![
            [4.0, 3.0],
            [6.0, 3.0]
        ];
        let r = vec![10.0, 12.0];
        let mut z = vec![0.0; 2];

        let lu_preconditioner = LU::new(&matrix);
        lu_preconditioner.apply(&r, &mut z);

        let expected_z = vec![1.0, 2.0];
        println!("Expected solution vector: {:?}", expected_z);
        println!("Computed solution vector z: {:?}", z);
        for (computed, expected) in z.iter().zip(expected_z.iter()) {
            assert!(
                (computed - expected).abs() < TOLERANCE,
                "Computed solution {:?} does not match expected {:?} within tolerance",
                z,
                expected_z
            );
        }
    }

    /// Tests that the LU preconditioner behaves correctly on a non-trivial 3x3 system.
    #[test]
    fn test_lu_preconditioner_non_trivial() {
        let matrix = mat![
            [3.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0]
        ];
        let r = vec![5.0, 8.0, 8.0];
        let mut z = vec![0.0; 3];

        let lu_preconditioner = LU::new(&matrix);
        println!("Performing LU decomposition...");
        
        // Manually solving for reference, using exact calculation for expected_z
        // Expected solution: [1.0, 2.0, 3.0]
        let expected_z = vec![1.0, 2.0, 3.0];

        // Apply LU preconditioner and print decomposed values for analysis
        lu_preconditioner.apply(&r, &mut z);

        println!("Input matrix:\n{:?}", matrix);
        println!("Input RHS vector r: {:?}", r);
        println!("Expected solution vector: {:?}", expected_z);
        println!("Computed solution vector z: {:?}", z);
        
        // Detailed per-element comparison with expected values
        for (i, (&computed, &expected)) in z.iter().zip(expected_z.iter()).enumerate() {
            println!("Index {}: computed = {}, expected = {}", i, computed, expected);
            assert!(
                (computed - expected).abs() < 1e-3,
                "At index {}, computed value {} does not match expected {} within tolerance.",
                i, computed, expected
            );
        }
    }

    /// Tests the behavior of the LU preconditioner with a singular matrix (checking for NaN in result).
    #[test]
    fn test_lu_preconditioner_singular() {
        let singular_matrix = mat![
            [1.0, 2.0],
            [2.0, 4.0]
        ];
        let r = vec![3.0, 6.0];
        let mut z = vec![0.0; 2];

        println!("Testing with singular matrix:\n{:?}", singular_matrix);
        println!("RHS vector r: {:?}", r);

        // Attempt to apply LU decomposition on a singular matrix
        let lu_preconditioner = LU::new(&singular_matrix);
        lu_preconditioner.apply(&r, &mut z);

        println!("Resulting solution vector z: {:?}", z);
        assert!(
            z.iter().any(|&value| value.is_nan()),
            "Expected NaN in solution vector for singular matrix, but got {:?}",
            z
        );
    }
}
```

---

Generate a detailed users guide for the `Solver` module for Hydra. I am going to provide the code for all of the parts of the `Solver` module below, and you can analyze and build the detailed outline based on this version of the source code.