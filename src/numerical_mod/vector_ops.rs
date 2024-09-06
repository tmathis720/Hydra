/// This module provides functions for basic vector operations.
///
/// Currently, it includes:
/// - Vector addition (`add_vectors`)

/// Adds two vectors element-wise.
///
/// # Arguments
///
/// * `v1` - A slice representing the first vector.
/// * `v2` - A slice representing the second vector.
///
/// # Returns
///
/// A `Vec<f64>` containing the result of element-wise addition.
///
/// # Panics
///
/// Panics if the input vectors are of different lengths.

pub fn add_vectors(v1: &[f64], v2: &[f64]) -> Vec<f64> {
    assert!(v1.len() == v2.len(), "Vectors must be of the same length");
    v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vectors() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let result = add_vectors(&v1, &v2);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "Vectors must be of the same length")]
    fn test_add_vectors_panic() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![4.0, 5.0, 6.0];
        add_vectors(&v1, &v2);
    }
}