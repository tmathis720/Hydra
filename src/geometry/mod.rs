// Module for handling geometric data and computations
// 2D Shape Modules
pub mod quadrilateral;
pub mod triangle;
// 3D Shape Modules
pub mod tetrahedron;
pub mod hexahedron;
pub mod prism;
pub mod pyramid;

// This struct will store geometric information such as vertex coordinates,
// centroids, and more as we extend it later.
pub struct Geometry {
    pub vertices: Vec<[f64; 3]>,  // 3D coordinates for each vertex
    pub cell_centroids: Vec<[f64; 3]>, // Centroid positions for each cell
    pub cell_volumes: Vec<f64>,    // Volumes of each cell
}

#[derive(Debug, Clone, Copy)]
pub enum CellShape {
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

#[derive(Debug, Clone, Copy)]
pub enum FaceShape {
    Triangle,
    Quadrilateral,
}

// Initial implementation of the Geometry module
impl Geometry {
    // Initializes geometry with empty data
    pub fn new() -> Geometry {
        Geometry {
            vertices: Vec::new(),
            cell_centroids: Vec::new(),
            cell_volumes: Vec::new(),
        }
    }

    /// Adds or updates a vertex in the geometry.
    /// If the vertex already exists (based on ID or index), it updates its coordinates.
    /// Otherwise, it adds a new vertex.
    ///
    /// # Arguments
    /// * `vertex_index` - The index or ID of the vertex to be set or updated.
    /// * `coords` - The 3D coordinates of the vertex as an array of `[f64; 3]`.
    pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]) {
        // Ensure the vector has enough capacity to accommodate the vertex at the given index
        if vertex_index >= self.vertices.len() {
            // Resize the vertices vector to hold up to `vertex_index` and beyond
            self.vertices.resize(vertex_index + 1, [0.0, 0.0, 0.0]);
        }

        // Update the vertex at the specified index
        self.vertices[vertex_index] = coords;
    }
    
    /// Computes the centroid of a given cell based on its shape and vertices.
    ///
    /// # Arguments
    /// * `cell_shape` - Enum defining the shape of the cell (e.g., Tetrahedron, Hexahedron).
    /// * `cell_vertices` - A vector of vertices representing the 3D coordinates of the cell's vertices.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the cell centroid.
    pub fn compute_cell_centroid(&self, cell_shape: CellShape, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_centroid(cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_centroid(cell_vertices),
            CellShape::Prism => self.compute_prism_centroid(cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_centroid(cell_vertices),
        }
    }

    /// Computes the volume of a given cell based on its shape and vertex coordinates.
    ///
    /// # Arguments
    /// * `cell_shape` - Enum defining the shape of the cell (e.g., Tetrahedron, Hexahedron).
    /// * `cell_vertices` - A vector of vertices representing the 3D coordinates of the cell's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the cell.
    pub fn compute_cell_volume(&self, cell_shape: CellShape, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_volume(cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_volume(cell_vertices),
            CellShape::Prism => self.compute_prism_volume(cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_volume(cell_vertices),
        }
    }

    /// Computes the Euclidean distance between two points in 3D space.
    ///
    /// # Arguments
    /// * `p1` - The first point as an array of 3D coordinates `[f64; 3]`.
    /// * `p2` - The second point as an array of 3D coordinates `[f64; 3]`.
    ///
    /// # Returns
    /// * `f64` - The Euclidean distance between the two points.
    pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        // Compute the squared differences for each coordinate (x, y, z)
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];

        // Apply the Euclidean distance formula: sqrt(dx^2 + dy^2 + dz^2)
        (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt()
    }

    /// Computes the area of a 2D face based on its shape.
    ///
    /// # Arguments
    /// * `face_shape` - Enum defining the shape of the face (e.g., Triangle, Quadrilateral).
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the face.
    ///
    /// # Returns
    /// * `f64` - The area of the face.
    pub fn compute_face_area(&self, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64 {
        match face_shape {
            FaceShape::Triangle => self.compute_triangle_area(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_area(face_vertices),
        }
    }

    /// Computes the centroid of a 2D face based on its shape.
    ///
    /// # Arguments
    /// * `face_shape` - Enum defining the shape of the face (e.g., Triangle, Quadrilateral).
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the face.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the face centroid.
    pub fn compute_face_centroid(&self, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        match face_shape {
            FaceShape::Triangle => self.compute_triangle_centroid(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_centroid(face_vertices),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{Geometry, CellShape, FaceShape};

    #[test]
    fn test_set_vertex() {
        let mut geometry = Geometry::new();

        // Set vertex at index 0
        geometry.set_vertex(0, [1.0, 2.0, 3.0]);
        assert_eq!(geometry.vertices[0], [1.0, 2.0, 3.0]);

        // Update vertex at index 0
        geometry.set_vertex(0, [4.0, 5.0, 6.0]);
        assert_eq!(geometry.vertices[0], [4.0, 5.0, 6.0]);

        // Set vertex at a higher index
        geometry.set_vertex(3, [7.0, 8.0, 9.0]);
        assert_eq!(geometry.vertices[3], [7.0, 8.0, 9.0]);

        // Ensure the intermediate vertices are initialized to [0.0, 0.0, 0.0]
        assert_eq!(geometry.vertices[1], [0.0, 0.0, 0.0]);
        assert_eq!(geometry.vertices[2], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_distance() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [3.0, 4.0, 0.0];

        let distance = Geometry::compute_distance(&p1, &p2);

        // The expected distance is 5 (Pythagoras: sqrt(3^2 + 4^2))
        assert_eq!(distance, 5.0);
    }

    #[test]
    fn test_compute_cell_centroid_tetrahedron() {
        let geometry = Geometry::new();
        let cell_vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let centroid = geometry.compute_cell_centroid(CellShape::Tetrahedron, &cell_vertices);

        // Expected centroid is the average of all vertices: (0.25, 0.25, 0.25)
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_compute_cell_volume_hexahedron() {
        let geometry = Geometry::new();

        // Define a regular hexahedron (cube) in 3D space
        let hexahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.0, 0.0, 1.0], // vertex 5
            [1.0, 0.0, 1.0], // vertex 6
            [1.0, 1.0, 1.0], // vertex 7
            [0.0, 1.0, 1.0], // vertex 8
        ];

        let volume = geometry.compute_cell_volume(CellShape::Hexahedron, &hexahedron_vertices);

        // The volume of a cube with side length 1 is 1^3 = 1.0
        assert!((volume - 1.0).abs() < 1e-10, "Volume of the hexahedron is incorrect");
    }

    #[test]
    fn test_compute_face_area_triangle() {
        let geometry = Geometry::new();

        // Define a right-angled triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];

        let area = geometry.compute_face_area(FaceShape::Triangle, &triangle_vertices);

        // Expected area: 0.5 * base * height = 0.5 * 3.0 * 4.0 = 6.0
        assert_eq!(area, 6.0);
    }

    #[test]
    fn test_compute_face_centroid_quadrilateral() {
        let geometry = Geometry::new();

        // Define a square in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        let centroid = geometry.compute_face_centroid(FaceShape::Quadrilateral, &quad_vertices);

        // Expected centroid is the geometric center: (0.5, 0.5, 0.0)
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }
}
