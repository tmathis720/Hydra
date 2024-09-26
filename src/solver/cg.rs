// src/solver/cg.rs

use crate::solver::linear_operator::LinearOperator;
use nalgebra::DMatrix;

pub struct ConjugateGradient<'a, M: LinearOperator> {
    matrix: &'a M,
}

impl<'a, M: LinearOperator> ConjugateGradient<'a, M> {
    pub fn new(matrix: &'a M) -> Self {
        ConjugateGradient { matrix }
    }

    pub fn solve(&self, rhs: &[f64], x: &mut [f64]) {
        let n = rhs.len();
        let mut r = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut ap = vec![0.0; n];

        // Initial residual
        self.matrix.matvec(x, &mut r);
        for i in 0..n {
            r[i] = rhs[i] - r[i];
            p[i] = r[i];
        }

        // CG iteration
        for _ in 0..n {
            self.matrix.matvec(&p, &mut ap);

            let alpha = dot(&r, &r) / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let beta = dot(&r, &r) / alpha;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }

            if norm(&r) < 1e-10 {
                break;
            }
        }
    }
}

// Simple utility functions for dot product and norm
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(ai, bi)| ai * bi).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}
