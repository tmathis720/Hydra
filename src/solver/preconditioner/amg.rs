use crate::linalg::{Matrix, Vector};
use crate::solver::preconditioner::Preconditioner;
use faer::mat::Mat;
use rayon::prelude::*; // Added for parallel operations
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

pub struct AMGTemporaryBuffers {
    pub level_buffers: Vec<LevelBuffer>,
}

pub struct LevelBuffer {
    pub az: Vec<f64>,               // Residual at fine level
    pub coarse_residual: Vec<f64>, // Residual restricted to coarse grid
    pub coarse_solution: Vec<f64>, // Solution at coarse level
    pub fine_correction: Vec<f64>, // Correction interpolated to fine grid
}

impl AMGTemporaryBuffers {
    pub fn new(level_sizes: &[(usize, usize)]) -> Self {
        let level_buffers = level_sizes
            .iter()
            .map(|&(fine_size, coarse_size)| LevelBuffer {
                az: vec![0.0; fine_size],
                coarse_residual: vec![0.0; coarse_size],
                coarse_solution: vec![0.0; coarse_size],
                fine_correction: vec![0.0; fine_size],
            })
            .collect();
        AMGTemporaryBuffers { level_buffers }
    }
}



impl AMG {
    pub fn new(a: &Mat<f64>, max_levels: usize, base_threshold: f64) -> Self {
        let mut levels = Vec::new();

        // Current (fine) matrix, which we'll successively coarsen
        let mut current_matrix = a.clone();
        let mut current_diag = Self::extract_diagonal_inverse(&current_matrix);

        for level_idx in 0..max_levels {
            // If too small, we stop
            let n = current_matrix.nrows();
            if n <= 10 {
                break;
            }

            // Build real interpolation and restriction for going from
            // current_matrix (fine) to the next coarse
            let adaptive_threshold = compute_adaptive_threshold(&current_matrix, base_threshold);
            let (mut interpolation, restriction) = AMG::generate_operators(
                &current_matrix,
                adaptive_threshold,
                false, // or true if you want double-pairwise
            );

            smooth_interpolation(&mut interpolation, &current_matrix, 0.5);
            minimize_energy(&mut interpolation, &current_matrix);

            // Build the new coarse matrix
            let coarse_matrix = &restriction * &current_matrix * &interpolation;
            let coarse_diag = Self::extract_diagonal_inverse(&coarse_matrix);

            // Store the operators for the fine matrix in "levels[level_idx]"
            // so that 'levels[level_idx].coarse_matrix' is the fine matrix,
            // and the next level is "coarse_matrix".
            levels.push(AMGLevel {
                interpolation,
                restriction,
                coarse_matrix: current_matrix.clone(), // the fine matrix for this level
                diag_inv: current_diag,                // diagonal of the fine matrix
            });

            // Next iteration: "current_matrix" becomes the new coarse
            current_matrix = coarse_matrix.clone();
            current_diag = coarse_diag;

            println!(
                "Level {} constructed: coarse matrix size = {}x{}",
                level_idx,
                current_matrix.nrows(),
                current_matrix.ncols()
            );
        }

        // Finally, push the **last** level, which is your final "coarse" matrix
        // with identity R/I (or you can store them if you plan further coarsening).
        let diag_inv_final = Self::extract_diagonal_inverse(&current_matrix);
        levels.push(AMGLevel {
            interpolation: Mat::identity(current_matrix.nrows(), current_matrix.nrows()),
            restriction: Mat::identity(current_matrix.nrows(), current_matrix.nrows()),
            coarse_matrix: current_matrix,
            diag_inv: diag_inv_final,
        });

        AMG {
            levels,
            nu_pre: 1,
            nu_post: 1,
        }
    }

    /// Enhanced `generate_operators` that supports double-pairwise and strength-of-connection-based coarsening.
    /// If `double_pairwise` is `true`, perform two-level pairwise aggregation;
    /// otherwise, use a single pass of greedy strength-based aggregation.
    fn generate_operators(
        a: &Mat<f64>,
        threshold: f64,
        double_pairwise: bool,
    ) -> (Mat<f64>, Mat<f64>) {
        // Step 1: Compute strength matrix
        let strength_matrix = compute_strength_matrix(a, threshold);

        // Step 2: Aggregate based on the chosen strategy
        let aggregates = if double_pairwise {
            double_pairwise_aggregation(&strength_matrix)
        } else {
            greedy_aggregation(&strength_matrix)
        };

        // Step 3: Construct prolongation (P) and restriction (R=P^T) matrices
        let prolongation = construct_prolongation(a, &aggregates);
        let restriction = prolongation.transpose().to_owned();

        (prolongation, restriction)
    }

    fn extract_diagonal_inverse(m: &Mat<f64>) -> Vec<f64> {
        assert_eq!(m.nrows(), m.ncols(), "Matrix must be square for diag_inv extraction");
        let n = m.nrows();
        let mut diag_inv = vec![0.0; n];
        for i in 0..n {
            let d = m.read(i, i);
            diag_inv[i] = if d.abs() < 1e-14 { 0.0 } else { 1.0 / d };
        }
        diag_inv
    }

    // Original Jacobi smoother (single-threaded)
    // We'll keep it for reference. We now introduce a parallel version below.
    fn _smooth_jacobi(a: &dyn Matrix<Scalar = f64>, diag_inv: &[f64], r: &[f64], z: &mut [f64], iterations: usize) {
        let n = r.len();
        let mut z_vec = z.to_vec();
        let mut temp = vec![0.0; n];
        for _ in 0..iterations {
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

    // Parallel Jacobi smoother using rayon
    fn smooth_jacobi_parallel(a: &dyn Matrix<Scalar = f64>, diag_inv: &[f64], r: &[f64], z: &mut [f64], iterations: usize) {
        let n = r.len();
        assert_eq!(diag_inv.len(), n, "diag_inv length mismatch: expected {}, got {}", n, diag_inv.len());
        assert_eq!(z.len(), n, "z length mismatch: expected {}, got {}", n, z.len());
        assert_eq!(a.nrows(), n, "Matrix row count mismatch: expected {}, got {}", n, a.nrows());
        
        let mut z_vec = z.to_vec();
        let mut temp = vec![0.0; n];
        for _ in 0..iterations {
            parallel_mat_vec(a, &z_vec, &mut temp);
            temp.par_iter_mut().enumerate().for_each(|(i, val)| {
                *val = r[i] - *val;
            });
            z_vec.par_iter_mut().enumerate().for_each(|(i, val)| *val += diag_inv[i] * temp[i]);
        }
        z.copy_from_slice(&z_vec);
    }

    fn apply_recursive(
        &self,
        level: usize,
        a: &dyn Matrix<Scalar = f64>,
        r: &[f64],
        z: &mut [f64],
        buffers: &mut AMGTemporaryBuffers,
    ) {
        // If this is the last level, do a direct solve
        if level + 1 == self.levels.len() {
            AMG::solve_direct(a, r, z);
            return;
        }
    
        let diag_inv = &self.levels[level].diag_inv;
        let restriction = &self.levels[level].restriction;
        let interpolation = &self.levels[level].interpolation;
        let _coarse_matrix = &self.levels[level + 1].coarse_matrix;
    
        let current_buffers = &mut buffers.level_buffers[level];
    
        // Pre-smoothing
        AMG::smooth_jacobi_parallel(a, diag_inv, r, z, self.nu_pre);
    
        // Compute current residual (r - Az)
        parallel_mat_vec(a, z, &mut current_buffers.az);
        current_buffers
            .az
            .iter_mut()
            .zip(r.iter())
            .for_each(|(az_val, r_val)| *az_val = r_val - *az_val);
    
        // Restrict residual to coarse grid
        parallel_mat_vec(
            restriction,
            &current_buffers.az,
            &mut current_buffers.coarse_residual,
        );
    
        // Recursive solve on coarse grid
        let level_sizes = self
            .levels
            .iter()
            .map(|level| (level.coarse_matrix.nrows(), level.coarse_matrix.ncols()))
            .collect::<Vec<_>>();
        let mut buffers = AMGTemporaryBuffers::new(&level_sizes);
        
        self.apply_recursive(0, a, r, z, &mut buffers);
        
    
        // Interpolate correction back to fine grid
        parallel_mat_vec(
            interpolation,
            &current_buffers.coarse_solution,
            &mut current_buffers.fine_correction,
        );
    
        // Update solution
        z.iter_mut()
            .zip(current_buffers.fine_correction.iter())
            .for_each(|(z_val, correction)| *z_val += correction);
    
        // Post-smoothing
        AMG::smooth_jacobi_parallel(a, diag_inv, r, z, self.nu_post);
    }
    
    
    
    
    

    fn solve_direct(a: &dyn Matrix<Scalar = f64>, r: &[f64], z: &mut [f64]) {
        let n = r.len();
        assert_eq!(n, z.len(), "Mismatch in sizes of solution and residual vectors");
    
        // Initialize solution vector to zero
        z.fill(0.0);
    
        // Weighted Jacobi smoother (with relaxation parameter omega)
        let omega = 0.8; // Relaxation parameter
        let max_iterations = 20; // Increased iterations for better smoothing
        let tol = 1e-8; // Adjusted tolerance for higher precision
    
        for _ in 0..max_iterations {
            let mut max_residual: f64 = 0.0;
    
            // Compute the updated solution and residuals in parallel
            let new_values_and_residuals: Vec<(f64, f64)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let diag = a.get(i, i);
                    if diag.abs() < 1e-14 {
                        return (0.0, 0.0); // Handle near-zero diagonal elements
                    }
    
                    // Compute the sum of off-diagonal terms
                    let sum: f64 = (0..n)
                        .filter(|&j| j != i)
                        .map(|j| a.get(i, j) * z[j])
                        .sum();
    
                    // Compute the new solution value
                    let new_value = z[i] + omega * (r[i] - sum - diag * z[i]) / diag;
    
                    // Calculate the residual for convergence check
                    let residual = (r[i] - (diag * new_value + sum)).abs();
                    (new_value, residual)
                })
                .collect();
            
            let new_values: Vec<f64> = new_values_and_residuals.iter().map(|&(new_value, _)| new_value).collect();
            let residuals: Vec<f64> = new_values_and_residuals.iter().map(|&(_, residual)| residual).collect();
    
            // Update solution and compute the maximum residual
            new_values
                .iter()
                .enumerate()
                .for_each(|(i, &new_value)| {
                    z[i] = new_value;
                    max_residual = max_residual.max(residuals[i]);
                });
    
            // Check for convergence
            if max_residual < tol {
                break;
            }
        }
    }
    
}

impl Preconditioner for AMG {
    fn apply(&self, a: &dyn Matrix<Scalar = f64>, r: &dyn Vector<Scalar = f64>, z: &mut dyn Vector<Scalar = f64>) {
        let residual = r.as_slice().to_vec();
        let mut solution = vec![0.0; residual.len()];

        if self.levels.is_empty() {
            // No coarsening: just Jacobi smoothing
            let diag_inv = AMG::extract_diagonal_inverse(&a.to_mat());
            AMG::smooth_jacobi_parallel(a, &diag_inv, &residual, &mut solution, 10);
        } else {
            // Pass the first coarse_matrix to align dimensions
            assert_eq!(
                self.levels[0].coarse_matrix.nrows(),
                a.nrows(),
                "Input matrix size mismatch at level 0"
            );
            let level_sizes = self
                .levels
                .iter()
                .map(|level| (level.coarse_matrix.nrows(), level.coarse_matrix.ncols()))
                .collect::<Vec<_>>();
            let mut buffers = AMGTemporaryBuffers::new(&level_sizes);
            self.apply_recursive(0, &self.levels[0].coarse_matrix, &residual, &mut solution, &mut buffers);
        }

        for i in 0..z.len() {
            z.set(i, solution[i]);
        }
    }
}


// ------------------- Additional Functions for Improvements -------------------

/// Compute anisotropy for each row of the matrix.
/// Anisotropy is defined as the ratio max_off_diag/diag.
fn compute_anisotropy(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    let mut anisotropy = vec![0.0; n];
    for i in 0..n {
        let diag = a.read(i, i);
        let mut max_off_diag: f64 = 0.0;
        for j in 0..n {
            if i != j {
                max_off_diag = max_off_diag.max(a.read(i, j).abs());
            }
        }
        if diag.abs() > 1e-14 {
            anisotropy[i] = max_off_diag / diag.abs();
        } else {
            anisotropy[i] = 0.0;
        }
    }
    anisotropy
}

/// Compute an adaptive threshold based on global anisotropy indicators.
fn compute_adaptive_threshold(a: &Mat<f64>, base_threshold: f64) -> f64 {
    let anis = compute_anisotropy(a);
    let avg_anis = if anis.is_empty() {
        1.0
    } else {
        anis.iter().sum::<f64>() / (anis.len() as f64)
    };
    // Adaptive threshold: Increase threshold if high anisotropy
    base_threshold * (1.0 + avg_anis.max(0.5))
}

/// Smooth the interpolation matrix to improve prolongation accuracy.
fn smooth_interpolation(interpolation: &mut Mat<f64>, matrix: &Mat<f64>, weight: f64) {
    let rows = interpolation.nrows();
    let cols = interpolation.ncols();
    for i in 0..rows {
        for j in 0..cols {
            let value = interpolation.read(i, j);
            let smoothed_value = value - weight * matrix.read(i, j);
            interpolation.write(i, j, smoothed_value);
        }
    }
}

/// Normalize rows of the interpolation matrix to minimize energy.
fn minimize_energy(interpolation: &mut Mat<f64>, _matrix: &Mat<f64>) {
    let rows = interpolation.nrows();
    let cols = interpolation.ncols();
    for i in 0..rows {
        let mut row_sum = 0.0;
        for j in 0..cols {
            row_sum += interpolation.read(i, j).powi(2);
        }
        let norm_factor = if row_sum.abs() > 1e-14 {
            row_sum.sqrt()
        } else {
            1.0
        };
        for j in 0..cols {
            let value = interpolation.read(i, j) / norm_factor;
            interpolation.write(i, j, value);
        }
    }
}

/// Parallel mat-vec multiplication using rayon.
fn parallel_mat_vec(mat: &dyn Matrix<Scalar = f64>, vec: &[f64], result: &mut [f64]) {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let (vlen, rlen) = (vec.len(), result.len());

    assert_eq!(
        cols, vlen,
        "Dimension mismatch in parallel_mat_vec!\n \
         Matrix is {}x{}, but input vector length is {}.\n \
         (Matrix columns must match vector length.)",
        rows, cols, vlen,
    );
    assert_eq!(
        rows, rlen,
        "Dimension mismatch in parallel_mat_vec!\n \
         Matrix is {}x{}, but result length is {}.\n \
         (Matrix rows must match result length.)",
        rows, cols, rlen,
    );

    // Now do the parallel multiplication
    result
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, res)| {
            *res = (0..cols)
                .into_par_iter()
                .map(|j| mat.get(i, j) * vec[j])
                .sum();
        });
}


// ------------------- Helper Functions for Enhanced Coarsening -------------------

/// Compute strength of connection matrix S, based on the definition:
/// S(i, j) = |A_ij| / sqrt(|A_ii| * |A_jj|) if > threshold, else 0.
fn compute_strength_matrix(a: &Mat<f64>, threshold: f64) -> Mat<f64> {
    let n = a.nrows();
    let mut s = Mat::<f64>::zeros(n, n);

    let updates: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let a_ii = a.read(i, i).abs();
            (0..n)
                .filter_map(move |j| {
                    if i == j {
                        return Some((i, j, 0.0)); // Diagonal entry
                    }
                    let val = a.read(i, j);
                    let a_jj = a.read(j, j).abs();
                    if a_ii > 1e-14 && a_jj > 1e-14 {
                        let strength = (val.abs() / (a_ii * a_jj).sqrt()) as f64;
                        if strength > threshold {
                            return Some((i, j, strength));
                        }
                    }
                    None
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (i, j, value) in updates {
        s.write(i, j, value);
    }

    s
}


/// Perform double-pairwise aggregation:
/// 1. Pairwise aggregate the graph to form coarse nodes.
/// 2. On the coarse graph, perform another round of pairing to form larger aggregates.
/// This function returns a vector where `aggregates[i]` = aggregate index of node i.
fn double_pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    // First pass: pairwise aggregation
    let first_pass = pairwise_aggregation(s);

    // Construct a coarse-level graph and apply pairwise aggregation again
    let coarse_graph = build_coarse_graph(s, &first_pass);
    let second_pass = pairwise_aggregation(&coarse_graph);

    // Map the second pass results back to the fine level
    remap_aggregates(&first_pass, &second_pass)
}

/// Greedy aggregation based on strength of connection:
/// Each node finds its strongest neighbor and they form an aggregate.
/// If a node is already aggregated, skip it.
fn greedy_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut aggregates = vec![usize::MAX; n];
    let mut next_agg_id = 0;

    for i in 0..n {
        if aggregates[i] == usize::MAX {
            // Find strongest neighbor j
            let mut max_strength = 0.0;
            let mut strongest = i; // if no stronger neighbor found, the node stands alone
            for j in 0..n {
                let strength = s.read(i, j);
                if strength > max_strength && aggregates[j] == usize::MAX && i != j {
                    max_strength = strength;
                    strongest = j;
                }
            }
            // Assign both i and j to the same aggregate
            aggregates[i] = next_agg_id;
            if strongest != i {
                aggregates[strongest] = next_agg_id;
            }
            next_agg_id += 1;
        }
    }

    aggregates
}

/// Pairwise aggregate a given strength matrix. This is a helper for double_pairwise_aggregation.
fn pairwise_aggregation(s: &Mat<f64>) -> Vec<usize> {
    let n = s.nrows();
    let mut aggregates = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut aggregate_id = 0;

    for i in 0..n {
        if visited[i] {
            continue;
        }

        // Find the strongest unvisited neighbor
        let mut max_strength = 0.0;
        let mut strongest_neighbor = None;
        for j in 0..n {
            if i != j && !visited[j] {
                let strength = s.read(i, j);
                if strength > max_strength {
                    max_strength = strength;
                    strongest_neighbor = Some(j);
                }
            }
        }

        // Form an aggregate with the strongest neighbor
        if let Some(j) = strongest_neighbor {
            aggregates[i] = aggregate_id;
            aggregates[j] = aggregate_id;
            visited[i] = true;
            visited[j] = true;
            aggregate_id += 1;
        } else {
            // No neighbor found, form a singleton aggregate
            aggregates[i] = aggregate_id;
            visited[i] = true;
            aggregate_id += 1;
        }
    }

    aggregates
}

/// Build a coarse graph from fine-level aggregates.
/// Each aggregate forms a node in the coarse graph.
/// The weights of edges between coarse nodes can be inherited or averaged from the fine graph.
fn build_coarse_graph(s: &Mat<f64>, aggregates: &[usize]) -> Mat<f64> {
    let max_agg_id = *aggregates.iter().max().unwrap();
    let coarse_n = max_agg_id + 1;
    let mut coarse_mat = Mat::<f64>::zeros(coarse_n, coarse_n);

    // Accumulate strengths into coarse matrix
    for (fine_node_i, &agg_i) in aggregates.iter().enumerate() {
        for fine_node_j in 0..s.ncols() {
            let agg_j = aggregates[fine_node_j];
            if agg_j < usize::MAX {
                let val = s.read(fine_node_i, fine_node_j);
                if val != 0.0 {
                    // Accumulate connection strength between aggregates
                    let old_val = coarse_mat.read(agg_i, agg_j);
                    coarse_mat.write(agg_i, agg_j, old_val + val);
                }
            }
        }
    }

    coarse_mat
}

/// Remap second pass aggregates to fine-level nodes.
fn remap_aggregates(first_pass: &[usize], second_pass: &[usize]) -> Vec<usize> {
    let n = first_pass.len();
    let mut final_aggregates = vec![0; n];
    for i in 0..n {
        // The aggregate at the fine level i is mapped to second_pass aggregate
        let coarse_agg = first_pass[i];
        final_aggregates[i] = second_pass[coarse_agg];
    }
    final_aggregates
}

/// Construct the prolongation matrix P from the aggregate assignments.
/// For piecewise constant interpolation:
/// P_{ij} = 1 if node i is in aggregate j, else 0.
fn construct_prolongation(a: &Mat<f64>, aggregates: &[usize]) -> Mat<f64> {
    let n = a.nrows();
    let max_agg_id = *aggregates.iter().max().unwrap();
    let coarse_n = max_agg_id + 1;

    let mut p = Mat::<f64>::zeros(n, coarse_n);

    for i in 0..n {
        let agg_id = aggregates[i];
        p.write(i, agg_id, 1.0);
    }

    p
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
