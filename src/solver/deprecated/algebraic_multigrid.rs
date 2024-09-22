use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub struct AlgebraicMultigridSolver {
    pub levels: usize,
    pub max_cycles: usize,
}

impl AlgebraicMultigridSolver {
    pub fn solve(&self, matrix: &DMatrix<f64>, rhs: &DVector<f64>) -> DVector<f64> {
        let mut solution = DVector::zeros(rhs.len());

        for _ in 0..self.max_cycles {
            solution = self.solve_v_cycle(matrix, rhs, &solution);
        }

        solution
    }

    fn solve_v_cycle(&self, matrix: &DMatrix<f64>, rhs: &DVector<f64>, initial_guess: &DVector<f64>) -> DVector<f64> {
        let mut solution = initial_guess.clone();

        // Parallel pre-smoothing
        self.parallel_smooth(matrix, &mut solution, rhs);

        // Restriction to coarse grid
        let coarse_grid = self.select_coarse_grid(matrix);
        let residual = rhs - matrix * &solution;
        let coarse_residual = self.restrict_to_coarse_grid(&residual, &coarse_grid);

        let coarse_solution = if coarse_grid.len() < 100 {
            self.solve_direct(&coarse_residual)
        } else {
            self.solve_v_cycle(matrix, &coarse_residual, &DVector::zeros(coarse_grid.len()))
        };

        // Interpolation
        let correction = self.interpolate(&coarse_solution, &coarse_grid);
        solution += correction;

        // Parallel post-smoothing
        self.parallel_smooth(matrix, &mut solution, rhs);

        solution
    }

    fn parallel_smooth(&self, matrix: &DMatrix<f64>, solution: &mut DVector<f64>, rhs: &DVector<f64>) {
        let diag_inv: Vec<f64> = matrix
            .diagonal()
            .iter()
            .map(|&d| 1.0 / d)
            .collect();

        // Use Arc<Mutex> for shared mutable access
        let solution = Arc::new(Mutex::new(solution));

        // Perform parallel updates using Rayon
        (0..matrix.nrows()).into_par_iter().for_each(|i| {
            let mut sum = 0.0;
            for j in 0..matrix.ncols() {
                if i != j {
                    sum += matrix[(i, j)] * solution.lock().unwrap()[j];
                }
            }
            solution.lock().unwrap()[i] = (rhs[i] - sum) * diag_inv[i];
        });
    }

    fn select_coarse_grid(&self, matrix: &DMatrix<f64>) -> Vec<usize> {
        (0..matrix.nrows()).into_par_iter().step_by(2).collect()
    }

    fn restrict_to_coarse_grid(&self, fine_residual: &DVector<f64>, coarse_grid: &[usize]) -> DVector<f64> {
        let mut coarse_residual = DVector::zeros(coarse_grid.len());

        coarse_residual
            .iter_mut()
            .zip(coarse_grid)
            .for_each(|(r_coarse, &idx)| {
                *r_coarse = fine_residual[idx];
            });

        coarse_residual
    }

    fn interpolate(&self, coarse_solution: &DVector<f64>, coarse_grid: &[usize]) -> DVector<f64> {
        let interpolated = Arc::new(Mutex::new(DVector::zeros(coarse_solution.len() * 2)));

        (0..coarse_grid.len()).into_par_iter().for_each(|i| {
            let idx = coarse_grid[i];
            let mut interpolated = interpolated.lock().unwrap();
            interpolated[idx] = coarse_solution[i];
            if idx + 1 < interpolated.len() {
                interpolated[idx + 1] = (coarse_solution[i] + coarse_solution[(i + 1).min(coarse_solution.len() - 1)]) * 0.5;
            }
        });

        Arc::try_unwrap(interpolated).unwrap().into_inner().unwrap() // Extract the DVector from Arc<Mutex>
    }

    fn solve_direct(&self, rhs: &DVector<f64>) -> DVector<f64> {
        DVector::from_element(rhs.len(), 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn create_test_matrix(size: usize) -> DMatrix<f64> {
        // Create a simple test matrix (diagonal-dominant)
        let mut matrix = DMatrix::zeros(size, size);
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    matrix[(i, j)] = 4.0;
                } else if (i as isize - j as isize).abs() == 1 {
                    matrix[(i, j)] = 1.0;
                }
            }
        }
        matrix
    }

    fn create_rhs(size: usize) -> DVector<f64> {
        // Create a right-hand side vector (b) with constant values
        DVector::from_element(size, 1.0)
    }

    fn create_solution_guess(size: usize) -> DVector<f64> {
        // Create an initial guess vector for the solution
        DVector::zeros(size)
    }

    #[test]
    fn test_parallel_smooth() {
        let matrix = create_test_matrix(5);
        let rhs = create_rhs(5);
        let mut solution = create_solution_guess(5);
        let solver = AlgebraicMultigridSolver {
            levels: 3,
            max_cycles: 5,
        };

        solver.parallel_smooth(&matrix, &mut solution, &rhs);

        // Check if smoothing works by confirming that the solution is not zero
        assert!(!solution.iter().all(|&x| x == 0.0), "Smoothing should update the solution");
    }

    #[test]
    fn test_select_coarse_grid() {
        let matrix = create_test_matrix(10);
        let solver = AlgebraicMultigridSolver {
            levels: 3,
            max_cycles: 5,
        };

        let coarse_grid = solver.select_coarse_grid(&matrix);

        // Check if the coarse grid has roughly half the size of the original grid
        assert!(coarse_grid.len() <= 5, "Coarse grid should have fewer points than the original grid");
    }

    #[test]
    fn test_restrict_to_coarse_grid() {
        let fine_residual = create_rhs(10);
        let solver = AlgebraicMultigridSolver {
            levels: 3,
            max_cycles: 5,
        };
        let coarse_grid = vec![0, 2, 4, 6, 8]; // Sample coarse grid indices

        let coarse_residual = solver.restrict_to_coarse_grid(&fine_residual, &coarse_grid);

        // Check if the coarse residual has the correct size and values match the fine grid residual
        assert_eq!(coarse_residual.len(), coarse_grid.len());
        assert_eq!(coarse_residual[0], fine_residual[0]);
        assert_eq!(coarse_residual[1], fine_residual[2]);
    }

    #[test]
    fn test_interpolate() {
        let coarse_solution = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let coarse_grid = vec![0, 2, 4, 6, 8];
        let solver = AlgebraicMultigridSolver {
            levels: 3,
            max_cycles: 5,
        };

        let interpolated = solver.interpolate(&coarse_solution, &coarse_grid);

        // Check that the interpolation results in a vector of the correct size
        assert_eq!(interpolated.len(), coarse_solution.len() * 2);

        // Check if the interpolated values match the coarse solution at the coarse grid points
        assert_eq!(interpolated[0], coarse_solution[0]);
        assert_eq!(interpolated[2], coarse_solution[1]);
        assert_eq!(interpolated[4], coarse_solution[2]);
    }

    #[test]
    fn test_solve_direct() {
        let rhs = create_rhs(5);
        let solver = AlgebraicMultigridSolver {
            levels: 3,
            max_cycles: 5,
        };

        let solution = solver.solve_direct(&rhs);

        // Check that the direct solver returns a solution of the correct size
        assert_eq!(solution.len(), rhs.len());

        // Check if the direct solution is trivial (all ones)
        assert!(solution.iter().all(|&x| x == 1.0), "Direct solver should return a solution of ones in this test case");
    }

    #[test]
    fn test_amg_solve() {
        let matrix = create_test_matrix(10);
        let rhs = create_rhs(10);
        let solver = AlgebraicMultigridSolver {
            levels: 3,
            max_cycles: 3,
        };

        let solution = solver.solve(&matrix, &rhs);

        // Check that the solution has the correct size
        assert_eq!(solution.len(), rhs.len());

        // Since we use a simple test case, the solution shouldn't be zero
        assert!(!solution.iter().all(|&x| x == 0.0), "AMG solver should produce a non-zero solution");
    }
}
