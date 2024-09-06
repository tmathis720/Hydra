// Define a struct to represent elements (triangles in this case)
pub struct Element {
    pub id: usize,
    pub node_ids: [usize; 3],
    pub tags: Vec<usize>,  // Store the physical and elementary tags
    pub area: f64, // Store the computed area of the element
    pub neighbors: Vec<usize>, // Store neighbor element ids
    pub state: f64, // Example state variable (e.g., temperature or concentration)
    pub flux: f64, // simple flux accumulator
}