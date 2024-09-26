use faer_core::{Mat, MatMut};
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

    pub fn solve(&self, a: &Mat<f64>, b: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
        // Handle the preconditioner
        let preconditioner_fn: Option<Box<dyn Fn(&Mat<f64>, &mut MatMut<f64>)>> = match self.preconditioner {
            Preconditioner::None => None,
            Preconditioner::Jacobi => {
                let a_cloned = a.clone();
                Some(Box::new(move |r: &Mat<f64>, z: &mut MatMut<f64>| {
                    JacobiPreconditioner::new(&a_cloned).apply_parallel(r, z);
                }))
            },
            Preconditioner::LU => {
                let lu_precond = LUPreconditioner::new(a);
                Some(Box::new(move |r: &Mat<f64>, z: &mut MatMut<f64>| {
                    let mut z_mut = z.as_mut();  // Obtain a mutable view for `z`
                    let mut z_own = Mat::<f64>::zeros(z_mut.nrows(), z_mut.ncols());
                    for i in 0..z_mut.nrows() {
                        for j in 0..z_mut.ncols() {
                            z_own.write(i, j, z_mut.read(i, j));  // Manually copy elements
                        }
                    }
                    lu_precond.apply(r, &mut z_own);
                }))
            },
        };

        match self.method {
            SolverMethod::ConjugateGradient => {
                let mut cg_solver = ConjugateGradient::new(a);
                let mut solution_vec = Mat::<f64>::zeros(b.nrows(), 1);
                let mut x = Mat::<f64>::zeros(b.nrows(), 1);

                if let Some(precondition) = &preconditioner_fn {
                    let mut preconditioned_rhs = Mat::<f64>::zeros(b.nrows(), 1);
                    precondition(b, &mut preconditioned_rhs.as_mut());

                    cg_solver.solve(&preconditioned_rhs, &mut x);
                } else {
                    cg_solver.solve(b, &mut x);
                }

                for i in 0..x.nrows() {
                    solution_vec[(i, 0)] = x[(i, 0)];
                }

                Ok(solution_vec)
            },
            SolverMethod::None => Err("No solver method specified"),
        }
    }
}
