use crate::solver::cg::ConjugateGradient;
use crate::solver::jacobi::Jacobi;
use nalgebra::{DMatrix, DVector};

// Define a trait for general solvers
pub trait Solver {
    fn solve(&mut self) -> Result<DVector<f64>, &'static str>;
}

// Enum for different solvers, allowing dynamic selection
pub enum SolverMethod {
    ConjugateGradient,
    GMRES,  // For future solvers
}

// Enum for different preconditioners
pub enum Preconditioner {
    None,
    Jacobi,
    ILU,  // For future preconditioners
}

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
                // Clone `a` to avoid lifetime issues and ensure type consistency
                let a: DMatrix<f64> = a.clone();  
                Some(Box::new(move |r: &DVector<f64>| -> DVector<f64> {
                    // Explicit type annotation for closure result
                    Jacobi::apply_preconditioner_static(&a, r)
                }))
            },
            Preconditioner::ILU => None, // To be implemented
        };
    
        // Solver handling
        match self.method {
            SolverMethod::ConjugateGradient => {
                let mut cg_solver = ConjugateGradient::new(a, b, preconditioner_fn);
                cg_solver.solve()
            },
            SolverMethod::GMRES => Err("GMRES is not implemented yet"),  // Future expansion
        }
    }
}
