use faer_core::Mat;
use faer_core::mul::matmul;

pub trait LinearOperator {
    fn matvec(&self, x: &Mat<f64>, y: &mut Mat<f64>);
    fn matmul(&self, other: &Self) -> Self;
}

impl LinearOperator for Mat<f64> {
    fn matvec(&self, x: &Mat<f64>, y: &mut Mat<f64>) {
        for i in 0..self.nrows() {
            let mut sum = 0.0;
            for j in 0..self.ncols() {
                sum += self[(i, j)] * x[(j, 0)];
            }
            y[(i, 0)] = sum;
        }
    }

    fn matmul(&self, other: &Self) -> Self {
        let mut result = Mat::<f64>::zeros(self.nrows(), other.ncols());
        matmul(result.as_mut(), self.as_ref(), other.as_ref(), None, 1.0, faer_core::Parallelism::None);
        result
    }
}
