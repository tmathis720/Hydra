Generate a detailed users guide for the `Geometry` module for Hydra. I am going to provide the code for all of the parts of the `Geometry` module below, and you can analyze and build the detailed outline based on this version of the source code.

`src/geometry/mod.rs`

```rust
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};
use std::sync::Mutex;

// Module for handling geometric data and computations
// 2D Shape Modules
pub mod quadrilateral;
pub mod triangle;
// 3D Shape Modules
pub mod tetrahedron;
pub mod hexahedron;
pub mod prism;
pub mod pyramid;

/// The `Geometry` struct stores geometric data for a mesh, including vertex coordinates, 
/// cell centroids, and volumes. It also maintains a cache of computed properties such as 
/// volume and centroid for reuse, optimizing performance by avoiding redundant calculations.
pub struct Geometry {
    pub vertices: Vec<[f64; 3]>,        // 3D coordinates for each vertex
    pub cell_centroids: Vec<[f64; 3]>,  // Centroid positions for each cell
    pub cell_volumes: Vec<f64>,         // Volumes of each cell
    pub cache: Mutex<FxHashMap<usize, GeometryCache>>, // Cache for computed properties, with thread safety
}

/// The `GeometryCache` struct stores computed properties of geometric entities, 
/// including volume, centroid, and area, with an optional "dirty" flag for lazy evaluation.
#[derive(Default)]
pub struct GeometryCache {
    pub volume: Option<f64>,
    pub centroid: Option<[f64; 3]>,
    pub area: Option<f64>,
    pub normal: Option<[f64; 3]>,  // Stores a precomputed normal vector for a face
}

/// `CellShape` enumerates the different cell shapes in a mesh, including:
/// * Tetrahedron
/// * Hexahedron
/// * Prism
/// * Pyramid
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellShape {
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

/// `FaceShape` enumerates the different face shapes in a mesh, including:
/// * Triangle
/// * Quadrilateral
#[derive(Debug, Clone, Copy)]
pub enum FaceShape {
    Triangle,
    Quadrilateral,
}

impl Geometry {
    /// Initializes a new `Geometry` instance with empty data.
    pub fn new() -> Geometry {
        Geometry {
            vertices: Vec::new(),
            cell_centroids: Vec::new(),
            cell_volumes: Vec::new(),
            cache: Mutex::new(FxHashMap::default()),
        }
    }

    /// Adds or updates a vertex in the geometry. If the vertex already exists,
    /// it updates its coordinates; otherwise, it adds a new vertex.
    ///
    /// # Arguments
    /// * `vertex_index` - The index of the vertex.
    /// * `coords` - The 3D coordinates of the vertex.
    pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]) {
        if vertex_index >= self.vertices.len() {
            self.vertices.resize(vertex_index + 1, [0.0, 0.0, 0.0]);
        }
        self.vertices[vertex_index] = coords;
        self.invalidate_cache();
    }

    /// Computes and returns the centroid of a specified cell using the cell's shape and vertices.
    /// Caches the result for reuse.
    pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> [f64; 3] {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.centroid) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let centroid = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_centroid(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_centroid(&cell_vertices),
            CellShape::Prism => self.compute_prism_centroid(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_centroid(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().centroid = Some(centroid);
        centroid
    }

    /// Computes the volume of a given cell using its shape and vertex coordinates.
    /// The computed volume is cached for efficiency.
    pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> f64 {
        let cell_id = cell.get_id();
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.volume) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell).expect("Cell shape not found");
        let cell_vertices = mesh.get_cell_vertices(cell);

        let volume = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_volume(&cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_volume(&cell_vertices),
            CellShape::Prism => self.compute_prism_volume(&cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_volume(&cell_vertices),
        };

        self.cache.lock().unwrap().entry(cell_id).or_default().volume = Some(volume);
        volume
    }

    /// Calculates Euclidean distance between two points in 3D space.
    pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt()
    }

    /// Computes the area of a 2D face based on its shape, caching the result.
    pub fn compute_face_area(&mut self, face_id: usize, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64 {
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.area) {
            return cached;
        }

        let area = match face_shape {
            FaceShape::Triangle => self.compute_triangle_area(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_area(face_vertices),
        };

        self.cache.lock().unwrap().entry(face_id).or_default().area = Some(area);
        area
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

    /// Computes and caches the normal vector for a face based on its shape.
    ///
    /// This function determines the face shape and calls the appropriate 
    /// function to compute the normal vector.
    ///
    /// # Arguments
    /// * `mesh` - A reference to the mesh.
    /// * `face` - The face entity for which to compute the normal.
    /// * `cell` - The cell associated with the face, used to determine the orientation.
    ///
    /// # Returns
    /// * `Option<[f64; 3]>` - The computed normal vector, or `None` if it could not be computed.
    pub fn compute_face_normal(
        &mut self,
        mesh: &Mesh,
        face: &MeshEntity,
        _cell: &MeshEntity,
    ) -> Option<[f64; 3]> {
        let face_id = face.get_id();

        // Check if the normal is already cached
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.normal) {
            return Some(cached);
        }

        let face_vertices = mesh.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let normal = match face_shape {
            FaceShape::Triangle => self.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_normal(&face_vertices),
        };

        // Cache the normal vector for future use
        self.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal);

        Some(normal)
    }

    /// Invalidate the cache when geometry changes (e.g., vertex updates).
    fn invalidate_cache(&mut self) {
        self.cache.lock().unwrap().clear();
    }

    /// Computes the total volume of all cells.
    pub fn compute_total_volume(&self) -> f64 {
        self.cell_volumes.par_iter().sum()
    }

    /// Updates all cell volumes in parallel using mesh information.
    pub fn update_all_cell_volumes(&mut self, mesh: &Mesh) {
        let new_volumes: Vec<f64> = mesh
            .get_cells()
            .par_iter()
            .map(|cell| {
                let mut temp_geometry = Geometry::new();
                temp_geometry.compute_cell_volume(mesh, cell)
            })
            .collect();

        self.cell_volumes = new_volumes;
    }

    /// Computes the total centroid of all cells.
    pub fn compute_total_centroid(&self) -> [f64; 3] {
        let total_centroid: [f64; 3] = self.cell_centroids
            .par_iter()
            .cloned()
            .reduce(
                || [0.0, 0.0, 0.0],
                |acc, centroid| [
                    acc[0] + centroid[0],
                    acc[1] + centroid[1],
                    acc[2] + centroid[2],
                ],
            );

        let num_centroids = self.cell_centroids.len() as f64;
        [
            total_centroid[0] / num_centroids,
            total_centroid[1] / num_centroids,
            total_centroid[2] / num_centroids,
        ]
    }
}


#[cfg(test)]
mod tests {
    use crate::geometry::{Geometry, CellShape, FaceShape};
    use crate::domain::{MeshEntity, mesh::Mesh, Sieve};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::sync::{Arc, RwLock};

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
        // Create a new Sieve and Mesh.
        let sieve = Sieve::new();
        let mut mesh = Mesh {
            sieve: Arc::new(sieve),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: None,
            boundary_data_receiver: None,
        };

        // Define vertices and a cell.
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);
        let cell = MeshEntity::Cell(1);

        // Set vertex coordinates.
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]);

        // Add entities to the mesh.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);
        mesh.add_entity(cell);

        // Establish relationships between the cell and vertices.
        mesh.add_arrow(cell, vertex1);
        mesh.add_arrow(cell, vertex2);
        mesh.add_arrow(cell, vertex3);
        mesh.add_arrow(cell, vertex4);

        // Verify that `get_cell_vertices` retrieves the correct vertices.
        let cell_vertices = mesh.get_cell_vertices(&cell);
        assert_eq!(cell_vertices.len(), 4, "Expected 4 vertices for a tetrahedron cell.");

        // Validate the shape before computing.
        assert_eq!(mesh.get_cell_shape(&cell), Ok(CellShape::Tetrahedron));

        // Create a Geometry instance and compute the centroid.
        let mut geometry = Geometry::new();
        let centroid = geometry.compute_cell_centroid(&mesh, &cell);

        // Expected centroid is the average of all vertices: (0.25, 0.25, 0.25)
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_compute_face_area_triangle() {
        let mut geometry = Geometry::new();

        // Define a right-angled triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];

        let area = geometry.compute_face_area(1, FaceShape::Triangle, &triangle_vertices);

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

    #[test]
    fn test_compute_total_volume() {
        let mut geometry = Geometry::new();
        let _mesh = Mesh::new();

        // Example setup: Define cells with known volumes
        // Here, you would typically define several cells and their volumes for the test
        geometry.cell_volumes = vec![1.0, 2.0, 3.0];

        // Expected total volume is the sum of individual cell volumes: 1.0 + 2.0 + 3.0 = 6.0
        assert_eq!(geometry.compute_total_volume(), 6.0);
    }

    #[test]
    fn test_compute_face_normal_triangle() {
        let geometry = Geometry::new();
        
        // Define vertices for a triangular face in the XY plane
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
        ];

        // Define the face as a triangle
        let _face = MeshEntity::Face(1);
        let _cell = MeshEntity::Cell(1);

        // Directly compute the normal without setting up mesh connectivity
        let normal = geometry.compute_triangle_normal(&vertices);

        // Expected normal for a triangle in the XY plane should be along the Z-axis
        let expected_normal = [0.0, 0.0, 1.0];
        
        // Check if the computed normal is correct
        for i in 0..3 {
            assert!((normal[i] - expected_normal[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_face_normal_quadrilateral() {
        let geometry = Geometry::new();

        // Define vertices for a quadrilateral face in the XY plane
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        // Define the face as a quadrilateral
        let _face = MeshEntity::Face(2);
        let _cell = MeshEntity::Cell(1);

        // Directly compute the normal for quadrilateral
        let normal = geometry.compute_quadrilateral_normal(&vertices);

        // Expected normal for a quadrilateral in the XY plane should be along the Z-axis
        let expected_normal = [0.0, 0.0, 1.0];
        
        // Check if the computed normal is correct
        for i in 0..3 {
            assert!((normal[i] - expected_normal[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_face_normal_caching() {
        let geometry = Geometry::new();

        // Define vertices for a triangular face
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
        ];

        let face_id = 3; // Unique identifier for caching
        let _face = MeshEntity::Face(face_id);
        let _cell = MeshEntity::Cell(1);

        // First computation to populate the cache
        let normal_first = geometry.compute_triangle_normal(&vertices);

        // Manually retrieve from cache to verify caching behavior
        geometry.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal_first);
        let cached_normal = geometry.cache.lock().unwrap().get(&face_id).and_then(|c| c.normal);

        // Verify that the cached value matches the first computed value
        assert_eq!(Some(normal_first), cached_normal);
    }

    #[test]
    fn test_compute_face_normal_unsupported_shape() {
        let geometry = Geometry::new();

        // Define vertices for a pentagon (unsupported)
        let vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.5, 0.5, 0.0], // vertex 5
        ];

        let _face = MeshEntity::Face(4);
        let _cell = MeshEntity::Cell(1);

        // Since compute_face_normal expects only triangles or quadrilaterals, it should return None
        let face_shape = match vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return, // Unsupported shape, skip test
        };

        // Attempt to compute the normal for an unsupported shape
        let normal = match face_shape {
            FaceShape::Triangle => Some(geometry.compute_triangle_normal(&vertices)),
            FaceShape::Quadrilateral => Some(geometry.compute_quadrilateral_normal(&vertices)),
        };

        // Assert that the function correctly handles unsupported shapes by skipping normal computation
        assert!(normal.is_none());
    }
}
```

---

`src/geometry/triangle.rs`

```rust
use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a triangular face.
    ///
    /// This function calculates the centroid of a triangle using its three vertices. The
    /// centroid is the point that is the average of the positions of the vertices.
    /// 
    /// # Arguments
    ///
    /// * `triangle_vertices` - A vector of 3D coordinates representing the vertices of the triangle.
    ///
    /// # Returns
    ///
    /// * `[f64; 3]` - The computed 3D coordinates of the triangle's centroid.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of vertices provided is not exactly three.
    pub fn compute_triangle_centroid(&self, triangle_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(triangle_vertices.len() == 3, "Triangle must have exactly 3 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for vertex in triangle_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        let num_vertices = triangle_vertices.len() as f64;
        [centroid[0] / num_vertices, centroid[1] / num_vertices, centroid[2] / num_vertices]
    }

    /// Computes the area of a triangular face.
    ///
    /// This function calculates the area of a triangle using the cross product of two edge vectors.
    /// The magnitude of the cross product gives twice the area of the triangle, and thus the final
    /// area is half of that magnitude.
    ///
    /// # Arguments
    ///
    /// * `triangle_vertices` - A vector of 3D coordinates representing the vertices of the triangle.
    ///
    /// # Returns
    ///
    /// * `f64` - The computed area of the triangle.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of vertices provided is not exactly three.
    pub fn compute_triangle_area(&self, triangle_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(
            triangle_vertices.len() == 3,
            "Triangle must have exactly 3 vertices"
        );

        let v0 = triangle_vertices[0];
        let v1 = triangle_vertices[1];
        let v2 = triangle_vertices[2];

        // Compute vectors v0->v1 and v0->v2
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Cross product of e1 and e2
        let cross_product = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Compute the magnitude of the cross product
        let cross_product_magnitude = (cross_product[0].powi(2)
            + cross_product[1].powi(2)
            + cross_product[2].powi(2))
        .sqrt();

        // Area is half the magnitude of the cross product
        0.5 * cross_product_magnitude
    }

    /// Computes the normal vector for a triangular face in 3D space.
    ///
    /// The normal vector is computed using the cross product of two edges from the triangle.
    /// The length of the normal vector will be proportional to the area of the triangle.
    ///
    /// # Arguments
    /// * `triangle_vertices` - A vector of 3D coordinates representing the vertices of the triangle.
    ///
    /// # Returns
    /// * `[f64; 3]` - The normal vector for the triangular face.
    pub fn compute_triangle_normal(&self, triangle_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(triangle_vertices.len() == 3, "Triangle must have exactly 3 vertices");

        let v0 = triangle_vertices[0];
        let v1 = triangle_vertices[1];
        let v2 = triangle_vertices[2];

        // Compute edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Compute the cross product to get the normal vector
        let normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        normal
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_triangle_normal() {
        let geometry = Geometry::new();

        let triangle_vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let normal = geometry.compute_triangle_normal(&triangle_vertices);

        // Expected normal should be [0.0, 0.0, 1.0] for this triangle in the XY plane
        assert!((normal[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_area() {
        let geometry = Geometry::new();

        // Define a right-angled triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];

        let area = geometry.compute_triangle_area(&triangle_vertices);

        // The area of this triangle is 0.5 * base * height = 0.5 * 3.0 * 4.0 = 6.0
        assert!((area - 6.0).abs() < 1e-10, "Area should be approximately 6.0");
    }

    #[test]
    fn test_triangle_centroid() {
        let geometry = Geometry::new();
        
        // Define a triangle in 3D space
        let triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [3.0, 0.0, 0.0], // vertex 2
            [0.0, 4.0, 0.0], // vertex 3
        ];
        
        let centroid = geometry.compute_triangle_centroid(&triangle_vertices);
        
        // The centroid of this triangle is the average of the vertices:
        // ([0.0, 0.0, 0.0] + [3.0, 0.0, 0.0] + [0.0, 4.0, 0.0]) / 3 = [1.0, 4.0 / 3.0, 0.0]
        assert_eq!(centroid, [1.0, 4.0 / 3.0, 0.0]);
    }

    #[test]
    fn test_triangle_area_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate triangle (all vertices are the same point)
        let degenerate_triangle_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
        ];

        let area = geometry.compute_triangle_area(&degenerate_triangle_vertices);

        // The area of a degenerate triangle is zero
        assert!(
            area.abs() < 1e-10,
            "Area should be approximately zero for a degenerate triangle"
        );
    }

    #[test]
    fn test_triangle_centroid_degenerate_case() {
        let geometry = Geometry::new();
        
        // Define a degenerate triangle (all vertices are the same point)
        let degenerate_triangle_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
        ];
        
        let centroid = geometry.compute_triangle_centroid(&degenerate_triangle_vertices);
        
        // The centroid of a degenerate triangle is the same as the vertex
        assert_eq!(centroid, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_triangle_area_colinear_points() {
        let geometry = Geometry::new();

        // Define a triangle with colinear points
        let colinear_triangle_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [2.0, 2.0, 2.0], // vertex 3
        ];

        let area = geometry.compute_triangle_area(&colinear_triangle_vertices);

        // The area should be zero for colinear points
        assert!(
            area.abs() < 1e-10,
            "Area should be approximately zero for colinear points"
        );
    }
}
```

---

`src/geometry/quadrilateral.rs`

```rust
use crate::geometry::Geometry;

impl Geometry {
    /// Computes the area of a quadrilateral face by dividing it into two triangles.
    ///
    /// This function computes the area of a quadrilateral in 3D space by splitting it into
    /// two triangles and summing their respective areas. The quadrilateral is assumed to have
    /// exactly four vertices.
    ///
    /// # Arguments
    ///
    /// * `quad_vertices` - A vector of 3D coordinates representing the vertices of the quadrilateral.
    ///
    /// # Returns
    ///
    /// * `f64` - The computed area of the quadrilateral.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of vertices provided is not exactly four.
    pub fn compute_quadrilateral_area(&self, quad_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(quad_vertices.len() == 4, "Quadrilateral must have exactly 4 vertices");

        // Split the quadrilateral into two triangles
        let triangle1 = vec![quad_vertices[0], quad_vertices[1], quad_vertices[2]];
        let triangle2 = vec![quad_vertices[2], quad_vertices[3], quad_vertices[0]];

        // Compute the area of the two triangles and sum them
        let area1 = self.compute_triangle_area(&triangle1);
        let area2 = self.compute_triangle_area(&triangle2);

        area1 + area2
    }

    /// Computes the centroid of a quadrilateral face.
    ///
    /// This function calculates the centroid (geometric center) of a quadrilateral by averaging the
    /// positions of its four vertices. The centroid is the point that minimizes the sum of the
    /// squared distances to all the vertices.
    ///
    /// # Arguments
    ///
    /// * `quad_vertices` - A vector of 3D coordinates representing the vertices of the quadrilateral.
    ///
    /// # Returns
    ///
    /// * `[f64; 3]` - The computed centroid of the quadrilateral.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of vertices provided is not exactly four.
    pub fn compute_quadrilateral_centroid(&self, quad_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(quad_vertices.len() == 4, "Quadrilateral must have exactly 4 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for vertex in quad_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        let num_vertices = quad_vertices.len() as f64;
        centroid[0] /= num_vertices;
        centroid[1] /= num_vertices;
        centroid[2] /= num_vertices;

        centroid
    }

    /// Computes the normal vector for a quadrilateral face in 3D space.
    ///
    /// The normal vector is approximated by dividing the quadrilateral into two triangles,
    /// calculating each triangle's normal, and averaging them.
    ///
    /// # Arguments
    /// * `quad_vertices` - A vector of 3D coordinates representing the vertices of the quadrilateral.
    ///
    /// # Returns
    /// * `[f64; 3]` - The normal vector for the quadrilateral face.
    pub fn compute_quadrilateral_normal(&self, quad_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(quad_vertices.len() == 4, "Quadrilateral must have exactly 4 vertices");

        // Divide the quadrilateral into two triangles
        let triangle1 = vec![quad_vertices[0], quad_vertices[1], quad_vertices[2]];
        let triangle2 = vec![quad_vertices[2], quad_vertices[3], quad_vertices[0]];

        // Compute normals for each triangle
        let normal1 = self.compute_triangle_normal(&triangle1);
        let normal2 = self.compute_triangle_normal(&triangle2);

        // Average the two normals
        [
            (normal1[0] + normal2[0]) / 2.0,
            (normal1[1] + normal2[1]) / 2.0,
            (normal1[2] + normal2[2]) / 2.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_quadrilateral_normal() {
        let geometry = Geometry::new();

        let quad_vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let normal = geometry.compute_quadrilateral_normal(&quad_vertices);

        // Expected normal should be [0.0, 0.0, 1.0] for this quadrilateral in the XY plane
        assert!((normal[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadrilateral_area_square() {
        let geometry = Geometry::new();

        // Define a square in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        let area = geometry.compute_quadrilateral_area(&quad_vertices);

        // The area of this square is 1.0 * 1.0 = 1.0
        assert_eq!(area, 1.0);
    }

    #[test]
    fn test_quadrilateral_area_non_planar() {
        let geometry = Geometry::new();

        // Define a non-planar quadrilateral in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0],  // vertex 1
            [1.0, 0.0, 0.0],  // vertex 2
            [1.0, 1.0, 1.0],  // vertex 3 (non-planar point)
            [0.0, 1.0, 0.0],  // vertex 4
        ];

        let area = geometry.compute_quadrilateral_area(&quad_vertices);

        // Since this quadrilateral is non-planar, the area should still be computed correctly
        assert!(area > 1.0, "Area of non-planar quadrilateral should be greater than 1.0");
    }

    #[test]
    fn test_quadrilateral_centroid_square() {
        let geometry = Geometry::new();

        // Define a square in 3D space
        let quad_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
        ];

        let centroid = geometry.compute_quadrilateral_centroid(&quad_vertices);

        // The centroid of a square is at (0.5, 0.5, 0.0)
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_quadrilateral_area_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate quadrilateral (all vertices are the same point)
        let degenerate_quad_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
            [1.0, 1.0, 1.0], // vertex 4
        ];

        let area = geometry.compute_quadrilateral_area(&degenerate_quad_vertices);

        // The area of a degenerate quadrilateral is zero
        assert_eq!(area, 0.0);
    }

    #[test]
    fn test_quadrilateral_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate quadrilateral (all vertices are the same point)
        let degenerate_quad_vertices = vec![
            [1.0, 1.0, 1.0], // vertex 1
            [1.0, 1.0, 1.0], // vertex 2
            [1.0, 1.0, 1.0], // vertex 3
            [1.0, 1.0, 1.0], // vertex 4
        ];

        let centroid = geometry.compute_quadrilateral_centroid(&degenerate_quad_vertices);

        // The centroid of a degenerate quadrilateral is the same as the vertex
        assert_eq!(centroid, [1.0, 1.0, 1.0]);
    }
}
```

---

`src/geometry/hexahedron.rs`

```rust
use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a hexahedral cell (e.g., a cube or cuboid).
    ///
    /// The centroid is calculated by averaging the positions of all 8 vertices of the hexahedron.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 8 vertices, each represented by a 3D coordinate `[f64; 3]`,
    ///   representing the vertices of the hexahedron.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the centroid of the hexahedron.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not exactly 8.
    pub fn compute_hexahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(cell_vertices.len() == 8, "Hexahedron must have exactly 8 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for vertex in cell_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        let num_vertices = cell_vertices.len() as f64;
        [
            centroid[0] / num_vertices,
            centroid[1] / num_vertices,
            centroid[2] / num_vertices,
        ]
    }

    /// Computes the volume of a hexahedral cell using tetrahedral decomposition.
    ///
    /// A hexahedron (e.g., a cube or cuboid) is decomposed into 5 tetrahedrons, and the volume
    /// of each tetrahedron is calculated. The sum of the volumes of all tetrahedrons gives the total
    /// volume of the hexahedron.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 8 vertices, each represented by a 3D coordinate `[f64; 3]`,
    ///   representing the vertices of the hexahedron.
    ///
    /// # Returns
    /// * `f64` - The volume of the hexahedron.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not exactly 8.
    pub fn compute_hexahedron_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(cell_vertices.len() == 8, "Hexahedron must have exactly 8 vertices");

        // Tetrahedral decomposition of a hexahedron (5 tetrahedrons).
        // We create the tetrahedrons using specific vertices from the hexahedron.
        let tets = vec![
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[3], cell_vertices[4]], // Tet 1
            vec![cell_vertices[1], cell_vertices[2], cell_vertices[3], cell_vertices[6]], // Tet 2
            vec![cell_vertices[1], cell_vertices[3], cell_vertices[4], cell_vertices[6]], // Tet 3
            vec![cell_vertices[3], cell_vertices[4], cell_vertices[6], cell_vertices[7]], // Tet 4
            vec![cell_vertices[1], cell_vertices[4], cell_vertices[5], cell_vertices[6]], // Tet 5
        ];

        // Sum the volume of each tetrahedron
        let mut total_volume = 0.0;
        for tet in tets {
            total_volume += self.compute_tetrahedron_volume(&tet);
        }

        total_volume
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_hexahedron_volume_regular() {
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

        let volume = geometry.compute_hexahedron_volume(&hexahedron_vertices);

        // The volume of a cube is side^3, here side = 1.0, so volume = 1.0^3 = 1.0
        assert!((volume - 1.0).abs() < 1e-10, "Volume of the hexahedron is incorrect");
    }

    #[test]
    fn test_hexahedron_centroid_regular() {
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

        let centroid = geometry.compute_hexahedron_centroid(&hexahedron_vertices);

        // The centroid of a unit cube is at the center: (0.5, 0.5, 0.5)
        assert_eq!(centroid, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_hexahedron_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate hexahedron where all vertices collapse into a plane
        let degenerate_hexahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.0, 0.0, 0.0], // vertex 5 (collapsing to the same plane)
            [1.0, 0.0, 0.0], // vertex 6 (collapsing to the same plane)
            [1.0, 1.0, 0.0], // vertex 7 (collapsing to the same plane)
            [0.0, 1.0, 0.0], // vertex 8 (collapsing to the same plane)
        ];

        let volume = geometry.compute_hexahedron_volume(&degenerate_hexahedron_vertices);

        // Since the hexahedron collapses into a plane, the volume should be zero
        assert_eq!(volume, 0.0);
    }

    #[test]
    fn test_hexahedron_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate hexahedron where all vertices collapse into a plane
        let degenerate_hexahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [1.0, 1.0, 0.0], // vertex 3
            [0.0, 1.0, 0.0], // vertex 4
            [0.0, 0.0, 0.0], // vertex 5 (collapsing to the same plane)
            [1.0, 0.0, 0.0], // vertex 6 (collapsing to the same plane)
            [1.0, 1.0, 0.0], // vertex 7 (collapsing to the same plane)
            [0.0, 1.0, 0.0], // vertex 8 (collapsing to the same plane)
        ];

        let centroid = geometry.compute_hexahedron_centroid(&degenerate_hexahedron_vertices);

        // The centroid should still be the average of the vertices, which lie in the XY plane
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }
}
```

---

`src/geometry/prism.rs`

```rust
use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a triangular prism.
    ///
    /// The prism is assumed to have two parallel triangular faces (top and bottom),
    /// each defined by 3 vertices. The centroid is calculated by averaging the centroids
    /// of the top and bottom triangles.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 6 vertices representing the 3D coordinates of the prism's vertices. 
    /// The first 3 vertices define the top triangle, and the last 3 vertices define the bottom triangle.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the centroid of the triangular prism.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not exactly 6.
    pub fn compute_prism_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(cell_vertices.len() == 6, "Triangular prism must have exactly 6 vertices");

        // Split into top and bottom triangles
        let top_triangle = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]];
        let bottom_triangle = vec![cell_vertices[3], cell_vertices[4], cell_vertices[5]];

        // Compute the centroids of both triangles
        let top_centroid = self.compute_triangle_centroid(&top_triangle);
        let bottom_centroid = self.compute_triangle_centroid(&bottom_triangle);

        // Compute the centroid of the prism by averaging the top and bottom centroids
        [
            (top_centroid[0] + bottom_centroid[0]) / 2.0,
            (top_centroid[1] + bottom_centroid[1]) / 2.0,
            (top_centroid[2] + bottom_centroid[2]) / 2.0,
        ]
    }

    /// Computes the volume of a triangular prism.
    ///
    /// The volume is calculated by multiplying the area of the triangular base (bottom face)
    /// with the height, which is the distance between the centroids of the top and bottom triangles.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 6 vertices representing the 3D coordinates of the prism's vertices. 
    /// The first 3 vertices define the top triangle, and the last 3 vertices define the bottom triangle.
    ///
    /// # Returns
    /// * `f64` - The volume of the triangular prism.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not exactly 6.
    pub fn compute_prism_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(cell_vertices.len() == 6, "Triangular prism must have exactly 6 vertices");

        // Split into top and bottom triangles
        let bottom_triangle = vec![cell_vertices[3], cell_vertices[4], cell_vertices[5]];

        // Compute the area of the base triangle (bottom triangle)
        let base_area = self.compute_triangle_area(&bottom_triangle);

        // Compute the height of the prism as the distance between the top and bottom triangle centroids
        let top_centroid = self.compute_triangle_centroid(&cell_vertices[0..3].to_vec());
        let bottom_centroid = self.compute_triangle_centroid(&bottom_triangle);
        let height = Geometry::compute_distance(&top_centroid, &bottom_centroid);

        // Volume of the prism = base area * height
        base_area * height
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_prism_volume_regular() {
        let geometry = Geometry::new();

        // Define a regular triangular prism in 3D space
        let prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 1.0], // bottom triangle vertex 1
            [1.0, 0.0, 1.0], // bottom triangle vertex 2
            [0.0, 1.0, 1.0], // bottom triangle vertex 3
        ];

        let volume = geometry.compute_prism_volume(&prism_vertices);

        // Base area = 0.5 (triangular area), height = 1.0 -> Volume = 0.5 * 1.0 = 0.5
        assert!((volume - 0.5).abs() < 1e-10, "Volume of the triangular prism is incorrect");
    }

    #[test]
    fn test_prism_centroid_regular() {
        let geometry = Geometry::new();

        // Define a regular triangular prism in 3D space
        let prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 1.0], // bottom triangle vertex 1
            [1.0, 0.0, 1.0], // bottom triangle vertex 2
            [0.0, 1.0, 1.0], // bottom triangle vertex 3
        ];

        let centroid = geometry.compute_prism_centroid(&prism_vertices);

        // The centroid of the top triangle = (1/3, 1/3, 0), bottom = (1/3, 1/3, 1) -> overall = (1/3, 1/3, 0.5)
        assert_eq!(centroid, [1.0 / 3.0, 1.0 / 3.0, 0.5]);
    }

    #[test]
    fn test_prism_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate triangular prism where all points lie on the same plane
        let degenerate_prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 0.0], // bottom triangle vertex 1 (collapsing onto the top triangle)
            [1.0, 0.0, 0.0], // bottom triangle vertex 2
            [0.0, 1.0, 0.0], // bottom triangle vertex 3
        ];

        let volume = geometry.compute_prism_volume(&degenerate_prism_vertices);

        // Since the prism collapses into a plane, the volume should be zero
        assert_eq!(volume, 0.0);
    }

    #[test]
    fn test_prism_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate triangular prism where all points lie on the same plane
        let degenerate_prism_vertices = vec![
            [0.0, 0.0, 0.0], // top triangle vertex 1
            [1.0, 0.0, 0.0], // top triangle vertex 2
            [0.0, 1.0, 0.0], // top triangle vertex 3
            [0.0, 0.0, 0.0], // bottom triangle vertex 1 (collapsing onto the top triangle)
            [1.0, 0.0, 0.0], // bottom triangle vertex 2
            [0.0, 1.0, 0.0], // bottom triangle vertex 3
        ];

        let centroid = geometry.compute_prism_centroid(&degenerate_prism_vertices);

        // The centroid is still computed as the average of all vertices
        assert_eq!(centroid, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    }
}
```

---

`src/geometry/pyramid.rs`

```rust
use crate::geometry::{Geometry, FaceShape};

impl Geometry {
    /// Computes the centroid of a pyramid cell (triangular or square base).
    ///
    /// The centroid is computed for a pyramid with either a triangular or square base.
    /// The pyramid is represented by 4 vertices (triangular base) or 5 vertices (square base).
    ///
    /// - For a triangular base, the pyramid is treated as a tetrahedron.
    /// - For a square base, the pyramid is split into two tetrahedrons, and the weighted centroid of both is calculated.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 4 vertices for a triangular-based pyramid, or 5 vertices for a square-based pyramid.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the pyramid centroid.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not 4 (for a triangular base) or 5 (for a square base).
    pub fn compute_pyramid_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(cell_vertices.len() == 4 || cell_vertices.len() == 5, "Pyramid must have 4 or 5 vertices");

        let apex = if cell_vertices.len() == 4 { cell_vertices[3] } else { cell_vertices[4] };
        let base_vertices = if cell_vertices.len() == 4 {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]]
        } else {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], cell_vertices[3]]
        };

        let base_centroid = self.compute_face_centroid(
            if cell_vertices.len() == 4 { FaceShape::Triangle } else { FaceShape::Quadrilateral },
            &base_vertices,
        );

        // Apply pyramid centroid formula: 3/4 * base_centroid + 1/4 * apex
        [
            (3.0 * base_centroid[0] + apex[0]) / 4.0,
            (3.0 * base_centroid[1] + apex[1]) / 4.0,
            (3.0 * base_centroid[2] + apex[2]) / 4.0,
        ]
    }

    /// Computes the volume of a pyramid cell (triangular or square base).
    ///
    /// The pyramid is represented by either 4 (triangular base) or 5 (square base) vertices.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 4 or 5 vertices representing the 3D coordinates of the pyramid's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the pyramid.
    ///
    /// # Panics
    /// This function will panic if the number of vertices is not 4 (triangular) or 5 (square).
    pub fn compute_pyramid_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(cell_vertices.len() == 4 || cell_vertices.len() == 5, "Pyramid must have 4 or 5 vertices");

        let apex = if cell_vertices.len() == 4 { cell_vertices[3] } else { cell_vertices[4] };
        
        if cell_vertices.len() == 5 {
            let tetra1 = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            let tetra2 = vec![cell_vertices[0], cell_vertices[2], cell_vertices[3], apex];

            let volume1 = self.compute_tetrahedron_volume(&tetra1);
            let volume2 = self.compute_tetrahedron_volume(&tetra2);

            volume1 + volume2
        } else {
            let tetra = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            self.compute_tetrahedron_volume(&tetra)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_pyramid_volume_square() {
        let geometry = Geometry::new();

        // Define a square pyramid in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [1.0, 1.0, 0.0], // base vertex 3
            [0.0, 1.0, 0.0], // base vertex 4
            [0.5, 0.5, 1.0], // apex
        ];

        let volume = geometry.compute_pyramid_volume(&pyramid_vertices);

        // Expected volume: (1/3) * base area * height, where base area = 1.0, height = 1.0
        assert!((volume - (1.0 / 3.0)).abs() < 1e-10, "Volume of the square pyramid is incorrect");
    }

    #[test]
    fn test_pyramid_volume_triangular() {
        let geometry = Geometry::new();

        // Define a triangular pyramid (tetrahedron) in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 1.0], // apex
        ];

        let volume = geometry.compute_pyramid_volume(&pyramid_vertices);

        // Expected volume: (1/3) * base area * height, where base area = 0.5, height = 1.0
        assert!((volume - (1.0 / 6.0)).abs() < 1e-10, "Volume of the triangular pyramid is incorrect");
    }

    #[test]
    fn test_pyramid_centroid_square() {
        let geometry = Geometry::new();

        // Define a square pyramid in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [1.0, 1.0, 0.0], // base vertex 3
            [0.0, 1.0, 0.0], // base vertex 4
            [0.5, 0.5, 1.0], // apex
        ];

        let centroid = geometry.compute_pyramid_centroid(&pyramid_vertices);

        // Expected centroid: (0.5, 0.5, 0.25)
        assert_eq!(centroid, [0.5, 0.5, 0.25]);
    }

    #[test]
    fn test_pyramid_centroid_triangular() {
        let geometry = Geometry::new();

        // Define a triangular pyramid in 3D space
        let pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 1.0], // apex
        ];

        let centroid = geometry.compute_pyramid_centroid(&pyramid_vertices);

        // Expected centroid: (0.375, 0.375, 0.25)
        assert_eq!(centroid, [0.375, 0.375, 0.25]);
    }

    #[test]
    fn test_pyramid_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate pyramid where all points are coplanar
        let degenerate_pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 0.0], // apex (degenerate, on the same plane as the base)
        ];

        let volume = geometry.compute_pyramid_volume(&degenerate_pyramid_vertices);

        // Volume of a degenerate pyramid should be zero
        assert_eq!(volume, 0.0);
    }

    #[test]
    fn test_pyramid_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate pyramid where all points are coplanar
        let degenerate_pyramid_vertices = vec![
            [0.0, 0.0, 0.0], // base vertex 1
            [1.0, 0.0, 0.0], // base vertex 2
            [0.0, 1.0, 0.0], // base vertex 3
            [0.5, 0.5, 0.0], // apex (degenerate, on the same plane as the base)
        ];

        let centroid = geometry.compute_pyramid_centroid(&degenerate_pyramid_vertices);

        // Expected centroid in the plane: (0.375, 0.375, 0.0)
        let expected_centroid = [0.375, 0.375, 0.0];
        assert_eq!(centroid, expected_centroid);
    }
}
```

---

`src/geometry/tetrahedron.rs`

```rust
use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a tetrahedral cell.
    ///
    /// This function calculates the centroid (geometric center) of a tetrahedron using
    /// the 3D coordinates of its four vertices. The centroid is the average of the
    /// positions of all vertices.
    ///
    /// # Arguments
    ///
    /// * `cell_vertices` - A vector of 3D coordinates representing the vertices of the tetrahedron.
    ///
    /// # Returns
    ///
    /// * `[f64; 3]` - The 3D coordinates of the computed centroid.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::geometry::Geometry;
    /// let geometry = Geometry::new();
    /// let vertices = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0]
    /// ];
    /// let centroid = geometry.compute_tetrahedron_centroid(&vertices);
    /// assert_eq!(centroid, [0.25, 0.25, 0.25]);
    /// ```
    pub fn compute_tetrahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert_eq!(cell_vertices.len(), 4, "Tetrahedron must have exactly 4 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for v in cell_vertices {
            centroid[0] += v[0];
            centroid[1] += v[1];
            centroid[2] += v[2];
        }

        // Average the vertex coordinates to get the centroid
        for i in 0..3 {
            centroid[i] /= 4.0;
        }

        centroid
    }

    /// Computes the volume of a tetrahedral cell.
    ///
    /// This function calculates the volume of a tetrahedron using the 3D coordinates of
    /// its four vertices. The volume is determined by computing the determinant of a matrix
    /// formed by three edges of the tetrahedron originating from a single vertex.
    ///
    /// # Arguments
    ///
    /// * `tet_vertices` - A vector of 4 vertices representing the 3D coordinates of the tetrahedron's vertices.
    ///
    /// # Returns
    ///
    /// * `f64` - The volume of the tetrahedron.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of vertices provided is not exactly four.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hydra::geometry::Geometry;
    /// let geometry = Geometry::new();
    /// let vertices = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0]
    /// ];
    /// let volume = geometry.compute_tetrahedron_volume(&vertices);
    /// assert!((volume - 1.0 / 6.0).abs() < 1e-10);
    /// ```
    pub fn compute_tetrahedron_volume(&self, tet_vertices: &Vec<[f64; 3]>) -> f64 {
        assert_eq!(tet_vertices.len(), 4, "Tetrahedron must have exactly 4 vertices");

        let v0 = tet_vertices[0];
        let v1 = tet_vertices[1];
        let v2 = tet_vertices[2];
        let v3 = tet_vertices[3];

        // Matrix formed by edges from v0 to the other vertices
        let matrix = [
            [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]],
            [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]],
            [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]],
        ];

        // Compute the determinant of the matrix
        let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

        det.abs() / 6.0
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

    #[test]
    fn test_tetrahedron_volume() {
        let geometry = Geometry::new();

        // Define a simple tetrahedron in 3D space
        let tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [0.0, 0.0, 1.0], // vertex 4
        ];

        let volume = geometry.compute_tetrahedron_volume(&tetrahedron_vertices);

        // The volume of this tetrahedron is 1/6, since it is a right-angled tetrahedron
        assert!((volume - 1.0 / 6.0).abs() < 1e-10, "Volume of the tetrahedron is incorrect");
    }

    #[test]
    fn test_tetrahedron_centroid() {
        let geometry = Geometry::new();

        // Define a simple tetrahedron in 3D space
        let tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [0.0, 0.0, 1.0], // vertex 4
        ];

        let centroid = geometry.compute_tetrahedron_centroid(&tetrahedron_vertices);

        // The centroid of a regular tetrahedron in this configuration is at [0.25, 0.25, 0.25]
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_tetrahedron_volume_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate tetrahedron where all vertices lie on the same plane
        let degenerate_tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [1.0, 1.0, 0.0], // vertex 4 (lies on the same plane as other vertices)
        ];

        let volume = geometry.compute_tetrahedron_volume(&degenerate_tetrahedron_vertices);

        // The volume of a degenerate tetrahedron is zero
        assert_eq!(volume, 0.0);
    }

    #[test]
    fn test_tetrahedron_centroid_degenerate_case() {
        let geometry = Geometry::new();

        // Define a degenerate tetrahedron where all vertices lie on the same plane
        let degenerate_tetrahedron_vertices = vec![
            [0.0, 0.0, 0.0], // vertex 1
            [1.0, 0.0, 0.0], // vertex 2
            [0.0, 1.0, 0.0], // vertex 3
            [1.0, 1.0, 0.0], // vertex 4 (lies on the same plane)
        ];

        let centroid = geometry.compute_tetrahedron_centroid(&degenerate_tetrahedron_vertices);

        // The centroid of this degenerate tetrahedron should still be calculated
        assert_eq!(centroid, [0.5, 0.5, 0.0]);
    }
}
```