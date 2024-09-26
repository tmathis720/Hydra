use faer_core::Mat;
use crate::solver::linear_operator::LinearOperator;

pub struct ConjugateGradient<'a, M: LinearOperator> {
    matrix: &'a M,
}

impl<'a, M: LinearOperator> ConjugateGradient<'a, M> {
    pub fn new(matrix: &'a M) -> Self {
        ConjugateGradient { matrix }
    }

    pub fn solve(&self, rhs: &Mat<f64>, x: &mut Mat<f64>) {
        let n = rhs.nrows();
        let mut r = Mat::<f64>::zeros(n, 1);
        let mut p = Mat::<f64>::zeros(n, 1);
        let mut ap = Mat::<f64>::zeros(n, 1);

        self.matrix.matvec(x, &mut r);
        for i in 0..n {
            r[(i, 0)] = rhs[(i, 0)] - r[(i, 0)];
            p[(i, 0)] = r[(i, 0)];
        }

        for _ in 0..n {
            self.matrix.matvec(&p, &mut ap);

            let alpha = dot(&r, &r) / dot(&p, &ap);
            for i in 0..n {
                x[(i, 0)] += alpha * p[(i, 0)];
                r[(i, 0)] -= alpha * ap[(i, 0)];
            }

            let beta = dot(&r, &r) / alpha;
            for i in 0..n {
                p[(i, 0)] = r[(i, 0)] + beta * p[(i, 0)];
            }

            if norm(&r) < 1e-10 {
                break;
            }
        }
    }
}

fn dot(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut result = 0.0;
    for i in 0..a.nrows() {
        result += a[(i, 0)] * b[(i, 0)];
    }
    result
}

fn norm(v: &Mat<f64>) -> f64 {
    dot(v, v).sqrt()
}
