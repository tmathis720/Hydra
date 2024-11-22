use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

/// Helper function to map parse errors to io::Error.
fn parse_error<E>(err: E) -> io::Error
where
    E: std::fmt::Debug,
{
    io::Error::new(io::ErrorKind::InvalidData, format!("{:?}", err))
}

/// Reads a MatrixMarket file and returns the data as a tuple:
/// (rows, cols, nonzeros, row_indices, col_indices, values).
pub fn read_matrix_market<P: AsRef<Path>>(
    file_path: P,
) -> io::Result<(usize, usize, usize, Vec<usize>, Vec<usize>, Vec<f64>)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Empty file"))??;

    // Check the MatrixMarket banner
    if !header.starts_with("%%MatrixMarket") {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid MatrixMarket banner"));
    }

    // Skip comments
    let size_line = lines
        .find(|line| {
            if let Ok(content) = line {
                !content.starts_with('%')
            } else {
                false
            }
        })
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing size information"))??;

    // Read matrix dimensions and non-zero count
    let size_parts: Vec<_> = size_line.split_whitespace().collect();
    if size_parts.len() != 3 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid size format"));
    }

    let rows = size_parts[0].parse::<usize>().map_err(parse_error)?;
    let cols = size_parts[1].parse::<usize>().map_err(parse_error)?;
    let nonzeros = size_parts[2].parse::<usize>().map_err(parse_error)?;

    let mut row_indices = Vec::with_capacity(nonzeros);
    let mut col_indices = Vec::with_capacity(nonzeros);
    let mut values = Vec::with_capacity(nonzeros);

    // Parse matrix data
    for line in lines {
        let line = line?;
        let parts: Vec<_> = line.split_whitespace().collect();

        if parts.len() < 2 {
            continue;
        }

        let row = parts[0].parse::<usize>().map_err(parse_error)? - 1; // Convert to 0-based index
        let col = parts[1].parse::<usize>().map_err(parse_error)? - 1; // Convert to 0-based index
        let value = if parts.len() > 2 {
            parts[2].parse::<f64>().map_err(parse_error)?
        } else {
            1.0 // Default value for pattern matrices
        };

        row_indices.push(row);
        col_indices.push(col);
        values.push(value);
    }

    Ok((rows, cols, nonzeros, row_indices, col_indices, values))
}

/// Writes a MatrixMarket file from given matrix data.
pub fn write_matrix_market<P: AsRef<Path>>(
    file_path: P,
    rows: usize,
    cols: usize,
    nonzeros: usize,
    row_indices: &[usize],
    col_indices: &[usize],
    values: &[f64],
) -> io::Result<()> {
    let mut file = File::create(file_path)?;

    // Write banner
    writeln!(file, "%%MatrixMarket matrix coordinate real general")?;

    // Write size line
    writeln!(file, "{} {} {}", rows, cols, nonzeros)?;

    // Write data
    for ((&row, &col), &value) in row_indices.iter().zip(col_indices).zip(values) {
        writeln!(file, "{} {} {}", row + 1, col + 1, value)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    /// Helper function to read a Matrix Market file and create the sparse matrix structure.
    fn read_matrix_market(file_path: &str) -> SparseMatrix {
        let contents = fs::read_to_string(file_path).expect("Failed to read Matrix Market file");
        parse_matrix_market(&contents).expect("Failed to parse Matrix Market data")
    }

    #[test]
    fn test_matrix_market_loading() {
        let file_path = "test_matrix.mtx";
        let matrix = read_matrix_market(file_path);

        // Validate dimensions
        assert_eq!(matrix.rows, 236, "Number of rows mismatch");
        assert_eq!(matrix.cols, 236, "Number of columns mismatch");

        // Validate number of non-zeros
        assert_eq!(matrix.nnz, 5856, "Number of non-zero elements mismatch");

        // Spot-check a few entries
        let expected_value = -5.3333331478961e-01; // Example from matrix data
        assert!(
            (matrix.get(7, 1) - expected_value).abs() < 1e-10,
            "Mismatch at (7, 1)"
        );
    }

    #[test]
    fn test_sparse_matrix_computation() {
        let file_path = "test_matrix.mtx";
        let matrix = read_matrix_market(file_path);

        // Perform a basic computation, e.g., matrix-vector multiplication
        let vector = vec![1.0; matrix.cols]; // Example vector
        let result = matrix.multiply_vector(&vector);

        // Validate the result's size
        assert_eq!(result.len(), matrix.rows, "Result vector length mismatch");

        // Add spot-check or property validations here
        let expected_result_sample = 8.5333331637906e+00; // Example expected value
        assert!(
            (result[0] - expected_result_sample).abs() < 1e-10,
            "Result mismatch at index 0"
        );
    }

    #[test]
    fn test_solver_accuracy() {
        let file_path = "test_matrix.mtx";
        let matrix = read_matrix_market(file_path);

        // Example: Solving Ax = b
        let b = vec![1.0; matrix.rows]; // Example right-hand side
        let x = matrix.solve(&b).expect("Solver failed");

        // Validate solution (e.g., using residual ||Ax - b||)
        let residual = matrix.residual(&x, &b);
        assert!(
            residual.norm() < 1e-6,
            "Solver residual too high: {}",
            residual.norm()
        );
    }
}
