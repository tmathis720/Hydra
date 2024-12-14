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

    // Determine the format (sparse or dense array)
    let is_array_format = header.contains("array");
    let is_coordinate_format = header.contains("coordinate");

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

    // Parse size line
    let size_parts: Vec<_> = size_line.split_whitespace().collect();
    let (rows, cols, nonzeros) = if is_array_format {
        // Array format specifies rows and columns, no explicit non-zero count
        if size_parts.len() != 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid size format for array"));
        }
        let rows = size_parts[0].parse::<usize>().map_err(parse_error)?;
        let cols = size_parts[1].parse::<usize>().map_err(parse_error)?;
        (rows, cols, rows * cols)
    } else if is_coordinate_format {
        // Coordinate format specifies rows, columns, and non-zero count
        if size_parts.len() != 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid size format for coordinate"));
        }
        let rows = size_parts[0].parse::<usize>().map_err(parse_error)?;
        let cols = size_parts[1].parse::<usize>().map_err(parse_error)?;
        let nonzeros = size_parts[2].parse::<usize>().map_err(parse_error)?;
        (rows, cols, nonzeros)
    } else {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported MatrixMarket format"));
    };

    let mut row_indices = Vec::with_capacity(nonzeros);
    let mut col_indices = Vec::with_capacity(nonzeros);
    let mut values = Vec::with_capacity(nonzeros);

    if is_array_format {
        // Parse dense array data
        for (i, line) in lines.enumerate() {
            let line = line?; // Bind the line to a variable
            let trimmed_line = line.trim(); // Create a separate binding for the trimmed string
            if trimmed_line.is_empty() {
                continue; // Skip empty lines
            }
            let value = trimmed_line.parse::<f64>().map_err(parse_error)?;
            let row = i / cols;
            let col = i % cols;
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }
    } else if is_coordinate_format {
        // Parse sparse coordinate data
        for line in lines {
            let line = line?; // Bind the line to a variable
            let trimmed_line = line.trim(); // Create a separate binding for the trimmed string
            if trimmed_line.is_empty() {
                continue; // Skip empty lines
            }
            let parts: Vec<_> = trimmed_line.split_whitespace().collect();
            if parts.len() < 2 {
                continue; // Skip invalid lines
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
    is_array_format: bool,
) -> io::Result<()> {
    let mut file = File::create(file_path)?;

    if is_array_format {
        // Write array header
        writeln!(file, "%%MatrixMarket matrix array real general")?;
        writeln!(file, "{} {}", rows, cols)?;

        // Write data
        for value in values {
            writeln!(file, "{}", value)?;
        }
    } else {
        // Write coordinate header
        writeln!(file, "%%MatrixMarket matrix coordinate real general")?;
        writeln!(file, "{} {} {}", rows, cols, nonzeros)?;

        // Write data
        for ((&row, &col), &value) in row_indices.iter().zip(col_indices).zip(values) {
            writeln!(file, "{} {} {}", row + 1, col + 1, value)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const MATRIX_FILE: &str = "inputs/matrix/e05r0000/e05r0000.mtx";
    const RHS_FILE: &str = "inputs/matrix/e05r0000/e05r0000_rhs1.mtx";
    const OUTPUT_FILE: &str = "outputs/output_test.mtx";

    #[test]
    fn test_read_sparse_matrix_market() {
        // Test reading the sparse matrix file
        let (rows, cols, nonzeros, row_indices, col_indices, values) =
            read_matrix_market(MATRIX_FILE).expect("Failed to read sparse matrix market file");

        // Assertions based on known metadata
        assert_eq!(rows, 236, "Unexpected number of rows");
        assert_eq!(cols, 236, "Unexpected number of columns");
        assert_eq!(nonzeros, 5856, "Unexpected number of non-zero entries");

        // Validate the first entry
        assert_eq!(row_indices[0], 6, "First row index mismatch"); // 0-based in Rust
        assert_eq!(col_indices[0], 0, "First column index mismatch"); // 0-based in Rust
        assert!((values[0] - -5.3333331478961e-01).abs() < 1e-12, "First value mismatch");
    }

    #[test]
    fn test_write_sparse_matrix_market() {
        // Read sparse matrix data
        let (rows, cols, nonzeros, row_indices, col_indices, values) =
            read_matrix_market(MATRIX_FILE).expect("Failed to read sparse matrix market file");

        // Write to a new file
        write_matrix_market(
            OUTPUT_FILE,
            rows,
            cols,
            nonzeros,
            &row_indices,
            &col_indices,
            &values,
            false, // Sparse format
        )
        .expect("Failed to write sparse matrix market file");

        // Re-read and validate
        let (w_rows, w_cols, w_nonzeros, w_row_indices, w_col_indices, w_values) =
            read_matrix_market(OUTPUT_FILE).expect("Failed to re-read written file");

        // Validate dimensions and counts
        assert_eq!(rows, w_rows, "Row count mismatch");
        assert_eq!(cols, w_cols, "Column count mismatch");
        assert_eq!(nonzeros, w_nonzeros, "Nonzero count mismatch");

        // Validate content equality
        assert_eq!(row_indices, w_row_indices, "Row indices mismatch after write-read cycle");
        assert_eq!(col_indices, w_col_indices, "Column indices mismatch after write-read cycle");
        assert_eq!(values, w_values, "Values mismatch after write-read cycle");

        // Clean up
        fs::remove_file(OUTPUT_FILE).expect("Failed to clean up output test file");
    }

    #[test]
    fn test_read_dense_matrix_market() {
        // Test reading the dense RHS file
        let (rows, cols, nonzeros, row_indices, col_indices, values) =
            read_matrix_market(RHS_FILE).expect("Failed to read dense matrix market file");

        // Assertions based on known metadata
        assert_eq!(rows, 236, "Unexpected number of rows");
        assert_eq!(cols, 1, "Unexpected number of columns");
        assert_eq!(nonzeros, 236, "Unexpected number of non-zero entries");

        // Validate the first entry
        assert_eq!(row_indices[0], 0, "First row index mismatch"); // 0-based in Rust
        assert_eq!(col_indices[0], 0, "First column index mismatch"); // 0-based in Rust
        assert!((values[0] - 0.0).abs() < 1e-12, "First value mismatch");
    }

 /*    #[test]
    fn test_write_dense_matrix_market() {
        // Read dense RHS data
        let (rows, cols, nonzeros, row_indices, col_indices, values) =
            read_matrix_market(RHS_FILE).expect("Failed to read dense matrix market file");

        // Write to a new file
        write_matrix_market(
            OUTPUT_FILE,
            rows,
            cols,
            nonzeros,
            &row_indices,
            &col_indices,
            &values,
            true, // Dense format
        )
        .expect("Failed to write dense matrix market file");

        // Re-read and validate
        let (w_rows, w_cols, w_nonzeros, w_row_indices, w_col_indices, w_values) =
            read_matrix_market(OUTPUT_FILE).expect("Failed to re-read written file");

        // Validate dimensions and counts
        assert_eq!(rows, w_rows, "Row count mismatch");
        assert_eq!(cols, w_cols, "Column count mismatch");
        assert_eq!(nonzeros, w_nonzeros, "Nonzero count mismatch");

        // Validate content equality
        assert_eq!(row_indices, w_row_indices, "Row indices mismatch after write-read cycle");
        assert_eq!(col_indices, w_col_indices, "Column indices mismatch after write-read cycle");
        assert_eq!(values, w_values, "Values mismatch after write-read cycle");

        // Clean up
        fs::remove_file(OUTPUT_FILE).expect("Failed to clean up output test file");
    } */
}
