pub struct Node {
    pub id: u32,
    pub position: (f64, f64, f64), // Coordinates in 2D space
}

impl Node {
    pub fn distance(&self, other: &Node) -> f64 {
        let (x1, y1, _z1) = self.position;
        let (x2, y2, _z2) = other.position;
        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    }

    // Interpolate a scalar value from neighboring elements
    // `element_values` contains the values at neighboring elements, and `weights` is the set of interpolation weights
    pub fn interpolate_scalar(&self, element_values: &[f64], weights: &[f64]) -> f64 {
        assert_eq!(element_values.len(), weights.len(), "Mismatched lengths between element values and weights");

        element_values.iter().zip(weights.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    }
}
