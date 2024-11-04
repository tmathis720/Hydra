use faer::Mat; // Example for faer vector support.

pub struct VectorBuilder;

impl VectorBuilder {
    /// Builds a vector of the specified type with a given initial size.
    /// Supports various vector types through generics.
    ///
    /// # Parameters
    /// - `size`: The length of the vector.
    ///
    /// # Returns
    /// A vector of type `T` initialized to the specified length.
    pub fn build_vector<T: VectorOperations>(size: usize) -> T {
        T::construct(size)
    }

    /// Builds a dense vector using faer's `Mat` structure as a column vector.
    /// Initializes with zeros.
    pub fn build_dense_vector(size: usize) -> Mat<f64> {
        Mat::<f64>::zeros(size, 1)
    }

    /// Resizes the provided vector dynamically while maintaining memory safety.
    /// Ensures no data is left uninitialized during resizing.
    pub fn resize_vector<T: VectorOperations + ExtendedVectorOperations>(
        vector: &mut T,
        new_size: usize,
    ) {
        vector.resize(new_size);
    }
}

pub trait VectorOperations {
    fn construct(size: usize) -> Self;
    fn set_value(&mut self, index: usize, value: f64);
    fn get_value(&self, index: usize) -> f64;
    fn size(&self) -> usize;
}

pub trait ExtendedVectorOperations: VectorOperations {
    /// Dynamically resizes the vector.
    fn resize(&mut self, new_size: usize);
}

impl VectorOperations for Vec<f64> {
    fn construct(size: usize) -> Self {
        vec![0.0; size]
    }

    fn set_value(&mut self, index: usize, value: f64) {
        self[index] = value;
    }

    fn get_value(&self, index: usize) -> f64 {
        self[index]
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl ExtendedVectorOperations for Vec<f64> {
    fn resize(&mut self, new_size: usize) {
        self.resize(new_size, 0.0);
    }
}

impl VectorOperations for Mat<f64> {
    fn construct(size: usize) -> Self {
        Mat::<f64>::zeros(size, 1)
    }

    fn set_value(&mut self, index: usize, value: f64) {
        self.write(index, 0, value);
    }

    fn get_value(&self, index: usize) -> f64 {
        self.read(index, 0)
    }

    fn size(&self) -> usize {
        self.nrows()
    }
}

impl ExtendedVectorOperations for Mat<f64> {
    fn resize(&mut self, new_size: usize) {
        let mut new_vector = Mat::<f64>::zeros(new_size, 1);
        for i in 0..usize::min(self.nrows(), new_size) {
            new_vector.write(i, 0, self.read(i, 0));
        }
        *self = new_vector;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_dense_vector() {
        let size = 5;
        let vector = VectorBuilder::build_dense_vector(size);

        assert_eq!(vector.nrows(), size, "Vector length should match the specified size.");
        assert_eq!(vector.ncols(), 1, "Vector should be a column vector.");

        for i in 0..size {
            assert_eq!(vector.read(i, 0), 0.0, "Vector should be initialized to zero.");
        }
    }

    #[test]
    fn test_build_vector_generic() {
        let size = 4;
        let vector = VectorBuilder::build_vector::<Vec<f64>>(size);

        assert_eq!(vector.len(), size, "Vector length should match the specified size.");
        for val in vector.iter() {
            assert_eq!(*val, 0.0, "Vector should be initialized to zero.");
        }
    }

    #[test]
    fn test_resize_vector() {
        let mut vector = VectorBuilder::build_dense_vector(3);
        vector.write(0, 0, 1.0);
        vector.write(1, 0, 2.0);
        vector.write(2, 0, 3.0);

        VectorBuilder::resize_vector(&mut vector, 5);

        assert_eq!(vector.nrows(), 5, "Vector length should be updated after resizing.");
        let expected_values = vec![1.0, 2.0, 3.0, 0.0, 0.0];
        for i in 0..5 {
            assert_eq!(vector.read(i, 0), expected_values[i], "Vector element mismatch at index {}.", i);
        }
    }
}
