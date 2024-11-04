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

