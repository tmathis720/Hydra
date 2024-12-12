use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use faer::mat::Mat;
use std::f64;

pub struct AMG {
    levels: Vec<AMGLevel>,
    nu_pre: usize,
    nu_post: usize,
}

struct AMGLevel {
    interpolation: Mat<f64>,
    restriction: Mat<f64>,
    coarse_matrix: Mat<f64>,
    diag_inv: Vec<f64>,
}

impl AMG {
    pub fn new(a: &Mat<f64>, max_levels: usize, coarsening_threshold: f64) -> Self {
        let mut levels = Vec::new();
        let mut current_matrix = a.clone();
        for _ in 0..max_levels {
            let n = current_matrix.nrows();
            if n <= 10 {
                break;
            }

            let (interpolation, restriction) =
                AMG::generate_operators(&current_matrix, coarsening_threshold);

            if interpolation.ncols() == 0 || current_matrix.nrows() == 0 {
                break; 
            }

            let coarse_matrix = &restriction * &current_matrix * &interpolation;

            levels.push(AMGLevel {
                interpolation,
                restriction,
                coarse_matrix: coarse_matrix.clone(),
                diag_inv: AMG::extract_diagonal_inverse(&coarse_matrix),
            });

            current_matrix = coarse_matrix;
        }

        AMG {
            levels,
            nu_pre: 1,
            nu_post: 1,
        }
    }

    fn generate_operators(a: &Mat<f64>, _threshold: f64) -> (Mat<f64>, Mat<f64>) {
        let n = a.nrows();
        let coarse_n = (n + 1) / 2; 
        let mut interpolation = Mat::<f64>::zeros(n, coarse_n);
        let mut restriction = Mat::<f64>::zeros(coarse_n, n);
        
        for fine_index in 0..n {
            let coarse_index = fine_index / 2;
            if coarse_index < coarse_n {
                let weight = if fine_index % 2 == 0 { 0.75 } else { 0.25 };
                interpolation.write(fine_index, coarse_index, weight);
                restriction.write(coarse_index, fine_index, weight);
            }
        }
        (interpolation, restriction)
    }

    fn extract_diagonal_inverse(m: &Mat<f64>) -> Vec<f64> {
        let n = m.nrows();
        let mut diag_inv = vec![0.0; n];
        for i in 0..n {
            let d = m.read(i, i);
            if d.abs() < 1e-14 {
                diag_inv[i] = 0.0;
            } else {
                diag_inv[i] = 1.0 / d;
            }
        }
        diag_inv
    }

    fn smooth_jacobi(a: &dyn Matrix<Scalar = f64>, diag_inv: &[f64], r: &[f64], z: &mut [f64], iterations: usize) {
        let n = r.len();
        let mut z_vec = z.to_vec(); // work with a local copy
        let mut temp = vec![0.0; n];
        for _ in 0..iterations {
            // a.mat_vec(z, temp)
            // Convert z_vec (Vec<f64>) and temp (Vec<f64>) to &dyn Vector
            a.mat_vec(&z_vec, &mut temp);

            for i in 0..n {
                temp[i] = r[i] - temp[i];
            }
            for i in 0..n {
                z_vec[i] += diag_inv[i] * temp[i];
            }
        }
        z.copy_from_slice(&z_vec);
    }

    fn apply_recursive(&self, level: usize, a: &dyn Matrix<Scalar = f64>, r: &[f64], z: &mut [f64]) {
        if level == self.levels.len() {
            // coarsest level: direct solve
            AMG::solve_direct(a, r, z);
        } else {
            // Pre-smoothing
            let diag_inv = &self.levels[level].diag_inv;
            AMG::smooth_jacobi(a, diag_inv, r, z, self.nu_pre);

            // Restrict residual: need r - A z
            let mut az = vec![0.0; a.nrows()];
            a.mat_vec(&z.to_vec(), &mut az); // convert z to vec here
            for i in 0..az.len() {
                az[i] = r[i] - az[i];
            }

            let restriction = &self.levels[level].restriction;
            let mut coarse_residual = vec![0.0; restriction.nrows()];
            restriction.mat_vec(&az, &mut coarse_residual);

            // Solve on coarse
            let mut coarse_solution = vec![0.0; coarse_residual.len()];
            self.apply_recursive(
                level + 1,
                &self.levels[level].coarse_matrix,
                &coarse_residual,
                &mut coarse_solution,
            );

            // Interpolate correction
            let interpolation = &self.levels[level].interpolation;
            let mut fine_correction = vec![0.0; interpolation.nrows()];
            interpolation.mat_vec(&coarse_solution, &mut fine_correction);

            // Update solution
            for i in 0..z.len() {
                z[i] += fine_correction[i];
            }

            // Post-smoothing
            AMG::smooth_jacobi(a, diag_inv, r, z, self.nu_post);
        }
    }

    fn solve_direct(a: &dyn Matrix<Scalar = f64>, r: &[f64], z: &mut [f64]) {
        let n = r.len();
        let mut temp_z = vec![0.0; n];
        // simple fixed jacobi as direct solve
        for _ in 0..10 {
            for i in 0..n {
                let diag = a.get(i, i);
                let mut sum = 0.0;
                for j in 0..n {
                    if j != i {
                        sum += a.get(i, j) * temp_z[j];
                    }
                }
                if diag.abs() > 1e-14 {
                    temp_z[i] = (r[i] - sum) / diag;
                } else {
                    temp_z[i] = 0.0;
                }
            }
        }
        z.copy_from_slice(&temp_z);
    }
}

impl Preconditioner for AMG {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let residual = r.as_slice().to_vec();
        let mut solution = vec![0.0; residual.len()];

        if self.levels.is_empty() {
            // no coarsening: just jacobi smoothing
            let diag_inv = AMG::extract_diagonal_inverse(&a.to_mat());
            AMG::smooth_jacobi(a, &diag_inv, &residual, &mut solution, 10);
        } else {
            self.apply_recursive(0, a, &residual, &mut solution);
        }

        for i in 0..z.len() {
            z.set(i, solution[i]);
        }
    }
}

// Convert a &dyn Matrix into a Mat<f64>
trait ToMat {
    fn to_mat(&self) -> Mat<f64>;
}

impl<'a> ToMat for dyn Matrix<Scalar = f64> + 'a {
    fn to_mat(&self) -> Mat<f64> {
        let rows = self.nrows();
        let cols = self.ncols();
        let mut m = Mat::<f64>::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.write(i, j, self.get(i, j));
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_amg_preconditioner_simple() {
        let matrix = mat![
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0]
        ];
        let r = vec![5.0, 5.0, 3.0];
        let mut z = vec![0.0; 3];

        let max_levels = 2;
        let coarsening_threshold = 0.1;
        let amg_preconditioner = AMG::new(&matrix, max_levels, coarsening_threshold);

        amg_preconditioner.apply(&matrix, &r, &mut z);

        let mut residual = vec![0.0; 3];
        matrix.mat_vec(&z, &mut residual);
        for i in 0..3 {
            residual[i] = r[i] - residual[i];
        }
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(residual_norm < 1.0, "Residual norm too high: {}", residual_norm);
    }

    #[test]
    fn test_amg_preconditioner_odd_size() {
        let matrix = mat![
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 4.0]
        ];
        let r = vec![5.0, 5.0, 3.0, 1.0];
        let mut z = vec![0.0; 4];

        let max_levels = 2;
        let coarsening_threshold = 0.1;
        let amg_preconditioner = AMG::new(&matrix, max_levels, coarsening_threshold);

        amg_preconditioner.apply(&matrix, &r, &mut z);

        let mut residual = vec![0.0; 4];
        matrix.mat_vec(&z, &mut residual);
        for i in 0..4 {
            residual[i] = r[i] - residual[i];
        }
        let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(residual_norm < 1.0, "Residual norm too high: {}", residual_norm);
    }
}
