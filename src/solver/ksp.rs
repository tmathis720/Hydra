use nalgebra::{DMatrix, DVector};
use faer_core::{Mat, Matrix};
use crate::solver::convert_to_faer;
use crate::solver::cg::ConjugateGradient;
use crate::solver::preconditioner::{JacobiPreconditioner, LUPreconditioner};

// SolverMethod Enum
pub enum SolverMethod {
    ConjugateGradient,
    None,
}

// Preconditioner Enum
pub enum Preconditioner {
    None,
    Jacobi,
    LU,
}

// KSP Struct
pub struct KSP {
    method: SolverMethod,
    preconditioner: Preconditioner,
}

impl KSP {
    pub fn new(method: SolverMethod, preconditioner: Preconditioner) -> Self {
        KSP { method, preconditioner }
    }

    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, &'static str> {
        // Preconditioner handling
        let preconditioner_fn: Option<Box<dyn Fn(&DVector<f64>) -> DVector<f64>>> = match self.preconditioner {
            Preconditioner::None => None,
            Preconditioner::Jacobi => {
                let a_cloned = a.clone();
                Some(Box::new(move |r: &DVector<f64>| {
                    let r_as_matrix = DMatrix::from_column_slice(r.len(), 1, r.as_slice());
                    JacobiPreconditioner::apply_preconditioner_static(&a_cloned, &r_as_matrix)
                }))
            },
            Preconditioner::LU => {
                let faer_matrix = convert_to_faer(a);
                let lu_precond = LUPreconditioner::new(&faer_matrix);
                Some(Box::new(move |r: &DVector<f64>| {
                    let mut solution = DVector::zeros(r.len());
                    let r_as_matrix = DMatrix::from_column_slice(r.len(), 1, r.as_slice());
                    lu_precond.apply(&r_as_matrix, &mut solution);
                    solution
                }))
            },
        };

        // Solver handling
        match self.method {
            SolverMethod::ConjugateGradient => {
                let mut cg_solver = ConjugateGradient::new(a);
                cg_solver.solve(b.as_slice(), &mut vec![0.0; b.len()]);
                Ok(DVector::from_vec(vec![0.0; b.len()])) // Dummy return
            },
            SolverMethod::None => Err("No solver method specified"),
        }
    }
}
