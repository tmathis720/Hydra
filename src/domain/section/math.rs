use std::ops::{Add, Div, Neg, Sub};

use super::{data::Section, scalar::Scalar, vector::Vector3};
// Add for Section<Scalar>
impl Add for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements addition for `Section<Scalar>`.
    ///
    /// This operator performs a component-wise addition of two `Section<Scalar>` instances.
    /// If a key exists in both sections, their corresponding values are added. If a key exists
    /// in only one section, its value is copied to the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Scalar>` operand (consumed).
    /// - `rhs`: The second `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` containing the sum of the two sections.
    fn add(self, rhs: Self) -> Self::Output {
        let result = self.clone(); // Clone the first section to use as a base
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair from the second section
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 += value.0; // Add values if the key exists in both sections
            } else {
                result.set_data(*key, *value); // Insert the value if the key only exists in `rhs`
            }
        }
        result
    }
}

// Sub for Section<Scalar>
impl Sub for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements subtraction for `Section<Scalar>`.
    ///
    /// This operator performs a component-wise subtraction of two `Section<Scalar>` instances.
    /// If a key exists in both sections, their corresponding values are subtracted. If a key exists
    /// in only one section, its value is added or negated in the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Scalar>` operand (consumed).
    /// - `rhs`: The second `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` containing the difference of the two sections.
    fn sub(self, rhs: Self) -> Self::Output {
        let result = self.clone(); // Clone the first section to use as a base
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair(); // Access key-value pair from the second section
            if let Some(mut current) = result.data.get_mut(key) {
                current.value_mut().0 -= value.0; // Subtract values if the key exists in both sections
            } else {
                result.set_data(*key, Scalar(-value.0)); // Negate and insert the value if the key only exists in `rhs`
            }
        }
        result
    }
}

// Neg for Section<Scalar>
impl Neg for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements negation for `Section<Scalar>`.
    ///
    /// This operator negates each value in the `Section<Scalar>` component-wise.
    ///
    /// # Parameters
    /// - `self`: The `Section<Scalar>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Scalar>` with all values negated.
    fn neg(self) -> Self::Output {
        let result = self.clone(); // Clone the section to preserve original data
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 = -value.0; // Negate the scalar value
        }
        result
    }
}

// Div for Section<Scalar>
impl Div<f64> for Section<Scalar> {
    type Output = Section<Scalar>;

    /// Implements scalar division for `Section<Scalar>`.
    ///
    /// Divides each value in the `Section<Scalar>` by a scalar `rhs` component-wise.
    ///
    /// # Parameters
    /// - `self`: The `Section<Scalar>` operand (consumed).
    /// - `rhs`: A scalar `f64` divisor.
    ///
    /// # Returns
    /// A new `Section<Scalar>` with all values scaled by `1/rhs`.
    fn div(self, rhs: f64) -> Self::Output {
        let result = self.clone(); // Clone the section to preserve original data
        for mut entry in result.data.iter_mut() {
            let (_, value) = entry.pair_mut(); // Access mutable key-value pair
            value.0 /= rhs; // Divide the scalar value by `rhs`
        }
        result
    }
}

// Sub for Section<Vector3>
impl Sub for Section<Vector3> {
    type Output = Section<Vector3>;

    /// Implements subtraction for `Section<Vector3>`.
    ///
    /// This operator performs a component-wise subtraction of two `Section<Vector3>` instances.
    /// If a key exists in both sections, their corresponding vectors are subtracted. If a key exists
    /// in only one section, its value is added or negated in the result.
    ///
    /// # Parameters
    /// - `self`: The first `Section<Vector3>` operand (consumed).
    /// - `rhs`: The second `Section<Vector3>` operand (consumed).
    ///
    /// # Returns
    /// A new `Section<Vector3>` containing the difference of the two sections.
    fn sub(self, rhs: Self) -> Self::Output {
        let result = Section::new(); // Create a new section to hold the result

        // Process all keys from the `rhs` section
        for entry in rhs.data.iter() {
            let (key, value) = entry.pair();
            if let Some(current) = self.data.get(key) {
                result.set_data(*key, *current.value() - *value); // Subtract if the key exists in both sections
            } else {
                result.set_data(*key, -*value); // Negate and add if the key only exists in `rhs`
            }
        }

        // Process all keys from the `self` section that are not in `rhs`
        for entry in self.data.iter() {
            let (key, value) = entry.pair();
            if !rhs.data.contains_key(key) {
                result.set_data(*key, *value); // Add the value if the key only exists in `self`
            }
        }

        result
    }
}
