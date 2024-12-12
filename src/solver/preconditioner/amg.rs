use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use faer::mat::Mat;

/// AMG preconditioner for sparse systems.
pub struct AMG {
    levels: Vec<AMGLevel>,
}

struct AMGLevel {
    interpolation: Mat<f64>, // Interpolation matrix
    restriction: Mat<f64>,   // Restriction matrix
    coarse_matrix: Mat<f64>, // Coarse-level matrix
}

impl AMG {
    /// Constructs an AMG preconditioner for a given sparse matrix.
    ///
    /// # Arguments
    /// - `a`: The system matrix.
    ///
    /// # Returns
    /// An `AMG` instance containing the hierarchy of levels.
    pub fn new(a: &Mat<f64>, max_levels: usize, coarsening_threshold: f64) -> Self {
        let mut levels = Vec::new();
        let mut current_matrix = a.clone();
    
        for _ in 0..max_levels {
            let (interpolation, restriction) = AMG::generate_operators(&current_matrix, coarsening_threshold);
    
            // Compute coarse matrix
            if current_matrix.nrows() == 0 || interpolation.ncols() == 0 {
                break; // Prevent invalid matrix multiplication
            }
            let coarse_matrix = restriction.clone() * &current_matrix * &interpolation;
    
            levels.push(AMGLevel {
                interpolation,
                restriction,
                coarse_matrix: coarse_matrix.clone(),
            });
    
            current_matrix = coarse_matrix; // Move to next coarse level
            if current_matrix.nrows() <= 10 {
                break; // Stop if the coarse matrix is too small
            }
        }
    
        AMG { levels }
    }
    

    /// Generate interpolation and restriction matrices for coarsening.
    fn generate_operators(a: &Mat<f64>, _threshold: f64) -> (Mat<f64>, Mat<f64>) {
        let n = a.nrows();
        let coarse_n = (n + 1) / 2; // Round up to ensure at least one row/column
    
        let mut interpolation = Mat::<f64>::zeros(n, coarse_n); // Fine-to-coarse mapping
        let mut restriction = Mat::<f64>::zeros(coarse_n, n);   // Coarse-to-fine mapping
    
        for fine_index in 0..n {
            let coarse_index = fine_index / 2; // Aggregate every 2 fine indices into one coarse
            if coarse_index < coarse_n {
                let weight = if fine_index % 2 == 0 { 0.75 } else { 0.25 }; // Smooth weight distribution
                interpolation.write(fine_index, coarse_index, weight);
                restriction.write(coarse_index, fine_index, weight);
            }
        }
    
        (interpolation, restriction)
    }

    /// Apply AMG preconditioner recursively across levels.
    fn apply_recursive(&self, level: usize, r: &[f64], z: &mut [f64]) {
        if level == self.levels.len() - 1 {
            // Coarsest level: Solve directly
            let coarse_matrix = &self.levels[level].coarse_matrix;
            AMG::solve_direct(coarse_matrix, r, z);
        } else {
            // Restrict to the coarse level
            let restriction = &self.levels[level].restriction;
            let mut coarse_residual = vec![0.0; restriction.nrows()];
            restriction.mat_vec(&Vec::from(r), &mut coarse_residual);
    
            // Solve on the coarse level
            let mut coarse_solution = vec![0.0; coarse_residual.len()];
            self.apply_recursive(level + 1, &coarse_residual, &mut coarse_solution);
    
            // Interpolate back to the fine level
            let interpolation = &self.levels[level].interpolation;
            let mut fine_solution = vec![0.0; interpolation.nrows()];
            interpolation.mat_vec(&coarse_solution, &mut fine_solution);
    
            for i in 0..fine_solution.len() {
                if i < z.len() {
                    z[i] += fine_solution[i];
                }
            }
        }
    }
    

    /// Solve directly on the coarsest level using Jacobi iteration.
    fn solve_direct(coarse_matrix: &Mat<f64>, r: &[f64], z: &mut [f64]) {
        let mut temp_z = vec![0.0; r.len()];
        for _ in 0..10 { // Fixed iterations for Jacobi
            for i in 0..r.len() {
                let diag = coarse_matrix.read(i, i);
                let sum: f64 = (0..coarse_matrix.ncols())
                    .filter(|&j| j != i)
                    .map(|j| coarse_matrix.read(i, j) * temp_z[j])
                    .sum();
                temp_z[i] = if diag.abs() > 1e-12 {
                    (r[i] - sum) / diag
                } else {
                    0.0
                };
            }
        }
        z.copy_from_slice(&temp_z);
    }
    
}

impl Preconditioner for AMG {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let _ = a;
        let residual = r.as_slice().to_vec();
        let mut solution = vec![0.0; residual.len()];

        // Apply AMG recursively
        self.apply_recursive(0, &residual, &mut solution);

        // Copy the solution back to `z`
        for i in 0..z.len() {
            z.set(i, solution[i]);
        }
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
    
        println!("Final residual norm: {}", residual_norm);
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
    
        println!("Final residual norm: {}", residual_norm);
        assert!(residual_norm < 1.0, "Residual norm too high: {}", residual_norm);
    }
    

}
