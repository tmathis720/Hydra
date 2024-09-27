use faer::{Mat, MatMut};
use crate::solver::cg::ConjugateGradient;
use crate::solver::preconditioner::{Jacobi, LU};
use crate::solver::{Matrix, Vector};

pub struct SolverResult {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
}

// KSP trait, representing Krylov solvers (CG, GMRES, etc.)
pub trait KSP {
    fn solve(&mut self, a: &dyn Matrix<Scalar = f64>, b: &dyn Vector<Scalar = f64>, x: &mut dyn Vector<Scalar = f64>) -> SolverResult;
}

