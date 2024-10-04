// src/vector/tests.rs

use super::traits::Vector;
use super::vec_impl::*;
use super::mat_impl::*;
use faer::Mat;

/// Helper function to create a simple vector for testing.
fn create_test_vector() -> Vec<f64> {
    vec![1.0, 2.0, 3.0, 4.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for `len` method
    #[test]
    fn test_vector_len() {
        let vec = create_test_vector();
        assert_eq!(vec.len(), 4, "Vector length should be 4");
    }

    // Test for `get` method
    #[test]
    fn test_vector_get() {
        let vec = create_test_vector();
        assert_eq!(vec.get(0), 1.0, "First element should be 1.0");
        assert_eq!(vec.get(2), 3.0, "Third element should be 3.0");
    }

    // Test for `set` method
    #[test]
    fn test_vector_set() {
        let mut vec = create_test_vector();
        vec.set(1, 10.0); // Set the second element to 10.0
        assert_eq!(vec.get(1), 10.0, "Second element should be updated to 10.0");
    }

    // Test for `dot` product method
    #[test]
    fn test_vector_dot() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        
        let dot_product = vec1.dot(&vec2);
        assert_eq!(dot_product, 32.0, "Dot product should be 32.0 (1*4 + 2*5 + 3*6)");
    }

    // Test for `norm` method (Euclidean norm)
    #[test]
    fn test_vector_norm() {
        let vec = vec![3.0, 4.0];
        
        let norm = vec.norm();
        assert!((norm - 5.0).abs() < 1e-6, "Norm should be 5.0 (sqrt(3^2 + 4^2))");
    }

    // Test for `as_slice` method
    #[test]
    fn test_vector_as_slice() {
        let vec = create_test_vector();
        let slice = vec.as_slice();
        
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0], "Slice should match the original vector");
    }

    // Test for `scale` method
    #[test]
    fn test_vector_scale() {
        let mut vec = create_test_vector();
        vec.scale(2.0);  // Scale the vector by 2.0
        
        assert_eq!(vec.as_slice(), &[2.0, 4.0, 6.0, 8.0], "All elements should be scaled by 2.0");
    }

    #[test]
    fn test_vector_axpy() {
        let mut y = vec![1.0, 1.0, 1.0, 1.0];  // Y vector
        let x = vec![2.0, 2.0, 2.0, 2.0];      // X vector

        y.axpy(2.0, &x);  // Perform the operation: y = 2 * x + y

        assert_eq!(y.as_slice(), &[5.0, 5.0, 5.0, 5.0], "Result should be [5.0, 5.0, 5.0, 5.0]");
    }

    #[test]
    fn test_vector_element_wise_add() {
        let mut vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![4.0, 3.0, 2.0, 1.0];

        vec1.element_wise_add(&vec2);  // Perform element-wise addition

        assert_eq!(vec1.as_slice(), &[5.0, 5.0, 5.0, 5.0], "Result should be [5.0, 5.0, 5.0, 5.0]");
    }

    #[test]
    fn test_vector_element_wise_mul() {
        let mut vec1 = vec![1.0, 2.0, 3.0, 4.0];
        let vec2 = vec![4.0, 3.0, 2.0, 1.0];

        vec1.element_wise_mul(&vec2);  // Perform element-wise multiplication

        assert_eq!(vec1.as_slice(), &[4.0, 6.0, 6.0, 4.0], "Result should be [4.0, 6.0, 6.0, 4.0]");
    }

    #[test]
    fn test_vector_element_wise_div() {
        let mut vec1 = vec![4.0, 9.0, 16.0, 25.0];
        let vec2 = vec![2.0, 3.0, 4.0, 5.0];

        vec1.element_wise_div(&vec2);  // Perform element-wise division

        assert_eq!(vec1.as_slice(), &[2.0, 3.0, 4.0, 5.0], "Result should be [2.0, 3.0, 4.0, 5.0]");
    }

    // Test cross product for Vec<f64>
    #[test]
    fn test_vec_cross_product() {
        let mut vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        vec1.cross(&vec2).expect("Cross product should succeed");

        assert_eq!(vec1, vec![0.0, 0.0, 1.0], "Cross product should be [0.0, 0.0, 1.0]");
    }

    // Test cross product for faer::Mat<f64>
    #[test]
    fn test_mat_cross_product() {
        let mut mat1 = Mat::from_fn(3, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        let mat2 = Mat::from_fn(3, 1, |i, _| if i == 1 { 1.0 } else { 0.0 });
        mat1.cross(&mat2).expect("Cross product should succeed");

        assert_eq!(mat1.as_slice(), &[0.0, 0.0, 1.0], "Cross product should be [0.0, 0.0, 1.0]");
    }

    #[test]
    fn test_vec_sum() {
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(vec.sum(), 10.0, "Sum should be 10.0");
    }

    #[test]
    fn test_mat_sum() {
        let mat = Mat::from_fn(4, 1, |i, _| (i + 1) as f64); // Matrix with values [1.0, 2.0, 3.0, 4.0]
        assert_eq!(mat.sum(), 10.0, "Sum should be 10.0");
    }

    #[test]
    fn test_vec_max() {
        let vec = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(vec.max(), 5.0, "Max should be 5.0");
    }

    #[test]
    fn test_mat_max() {
        let mat = Mat::from_fn(4, 1, |i, _| (i as f64) * 2.0); // Matrix with values [0.0, 2.0, 4.0, 6.0]
        assert_eq!(mat.max(), 6.0, "Max should be 6.0");
    }

    #[test]
    fn test_vec_min() {
        let vec = vec![1.0, 3.0, 2.0, -5.0, 4.0];
        assert_eq!(vec.min(), -5.0, "Min should be -5.0");
    }

    #[test]
    fn test_mat_min() {
        let mat = Mat::from_fn(4, 1, |i, _| (i as f64) * 2.0 - 3.0); // Matrix with values [-3.0, -1.0, 1.0, 3.0]
        assert_eq!(mat.min(), -3.0, "Min should be -3.0");
    }

    #[test]
    fn test_vec_mean() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(vec.mean(), 3.0, "Mean should be 3.0");
    }

    #[test]
    fn test_mat_mean() {
        let mat = Mat::from_fn(5, 1, |i, _| (i as f64) + 1.0); // Matrix with values [1.0, 2.0, 3.0, 4.0, 5.0]
        assert_eq!(mat.mean(), 3.0, "Mean should be 3.0");
    }

    #[test]
    fn test_empty_vec_mean() {
        let vec: Vec<f64> = vec![];
        assert_eq!(vec.mean(), 0.0, "Mean should be 0.0 for empty vector");
    }

    #[test]
    fn test_empty_mat_mean() {
        let mat = Mat::from_fn(0, 1, |_i, _| 0.0); // Empty matrix
        assert_eq!(mat.mean(), 0.0, "Mean should be 0.0 for empty matrix");
    }

    #[test]
    fn test_vec_variance() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((vec.variance() - 2.0).abs() < 1e-6, "Variance should be 2.0");
    }

    #[test]
    fn test_mat_variance() {
        let mat = Mat::from_fn(5, 1, |i, _| (i as f64) + 1.0); // Matrix with values [1.0, 2.0, 3.0, 4.0, 5.0]
        assert!((mat.variance() - 2.0).abs() < 1e-6, "Variance should be 2.0");
    }

    #[test]
    fn test_empty_vec_variance() {
        let vec: Vec<f64> = vec![];
        assert_eq!(vec.variance(), 0.0, "Variance should be 0.0 for empty vector");
    }

    #[test]
    fn test_empty_mat_variance() {
        let mat = Mat::from_fn(0, 1, |_i, _| 0.0); // Empty matrix
        assert_eq!(mat.variance(), 0.0, "Variance should be 0.0 for empty matrix");
    }
}
