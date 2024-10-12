use rayon::prelude::*;
use rustc_hash::FxHashMap;
use crate::domain::{mesh::Mesh, MeshEntity};

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
    pub cache: FxHashMap<usize, GeometryCache>, // Cache for computed properties
}

// Cache structure for storing computed properties of geometric entities.
#[derive(Default)]
pub struct GeometryCache {
    pub volume: Option<f64>,
    pub centroid: Option<[f64; 3]>,
    pub area: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
            cache: FxHashMap::default(),
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
        self.invalidate_cache(); // Invalidate cache when vertices change
    }
    
    /// Computes the centroid of a given cell based on its shape and vertices.
    ///
    /// # Arguments
    /// * `cell_id` - The ID of the cell to cache the result.
    /// * `cell_shape` - Enum defining the shape of the cell (e.g., Tetrahedron, Hexahedron).
    /// * `cell_vertices` - A vector of vertices representing the 3D coordinates of the cell's vertices.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the cell centroid.
    /// Computes the centroid of a given cell using data from the Mesh.
    pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> [f64; 3] {
        let cell_id = cell.id();
        if let Some(cached) = self.cache.get(&cell_id).and_then(|c| c.centroid) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell);
        let cell_vertices = mesh.get_cell_vertices(cell);

        let centroid = match cell_shape {
            Ok(CellShape::Tetrahedron) => self.compute_tetrahedron_centroid(&cell_vertices),
            Ok(CellShape::Hexahedron) => self.compute_hexahedron_centroid(&cell_vertices),
            Ok(CellShape::Prism) => self.compute_prism_centroid(&cell_vertices),
            Ok(CellShape::Pyramid) => self.compute_pyramid_centroid(&cell_vertices),
            Err(_) => todo!(),
        };

        self.cache.entry(cell_id).or_default().centroid = Some(centroid);
        centroid
    }

    /// Computes the volume of a given cell based on its shape and vertex coordinates.
    ///
    /// # Arguments
    /// * `cell_id` - The ID of the cell to cache the result.
    /// * `cell_shape` - Enum defining the shape of the cell (e.g., Tetrahedron, Hexahedron).
    /// * `cell_vertices` - A vector of vertices representing the 3D coordinates of the cell's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the cell.
    /// Computes the volume of a given cell using data from the Mesh.
    pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> f64 {
        let cell_id = cell.id();
        if let Some(cached) = self.cache.get(&cell_id).and_then(|c| c.volume) {
            return cached;
        }

        let cell_shape = mesh.get_cell_shape(cell);
        let cell_vertices = mesh.get_cell_vertices(cell);

        let volume = match cell_shape {
            Ok(CellShape::Tetrahedron) => self.compute_tetrahedron_volume(&cell_vertices),
            Ok(CellShape::Hexahedron) => self.compute_hexahedron_volume(&cell_vertices),
            Ok(CellShape::Prism) => self.compute_prism_volume(&cell_vertices),
            Ok(CellShape::Pyramid) => self.compute_pyramid_volume(&cell_vertices),
            Err(_) => todo!(),
        };

        self.cache.entry(cell_id).or_default().volume = Some(volume);
        volume
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
    /// * `face_id` - The ID of the face to cache the result.
    /// * `face_shape` - Enum defining the shape of the face (e.g., Triangle, Quadrilateral).
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the face.
    ///
    /// # Returns
    /// * `f64` - The area of the face.
    pub fn compute_face_area(&mut self, face_id: usize, face_shape: FaceShape, face_vertices: &Vec<[f64; 3]>) -> f64 {
        if let Some(cached) = self.cache.get(&face_id).and_then(|c| c.area) {
            return cached;
        }

        let area = match face_shape {
            FaceShape::Triangle => self.compute_triangle_area(face_vertices),
            FaceShape::Quadrilateral => self.compute_quadrilateral_area(face_vertices),
        };

        self.cache.entry(face_id).or_default().area = Some(area);
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

    /// Clears the cache when geometry changes, such as when vertices are updated.
    fn invalidate_cache(&mut self) {
        self.cache.clear();
    }

    /// Computes the total volume of all cells.
    pub fn compute_total_volume(&self) -> f64 {
        self.cell_volumes.par_iter().sum()
    }

    /// Updates all cell volumes in parallel.
pub fn update_all_cell_volumes(&mut self, mesh: &Mesh) {
    // Using a mutable reference inside the parallel iteration requires `Mutex` or similar for thread safety.
    let new_volumes: Vec<f64> = mesh
        .get_cells()
        .par_iter()
        .map(|cell| {
            // Use a temporary Geometry instance to compute the volume.
            let mut temp_geometry = Geometry::new();
            temp_geometry.compute_cell_volume(mesh, cell)
        })
        .collect();
    
    // Update the cell volumes after the parallel iteration.
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

        // Debug: Print the sieve structure.
        println!("Sieve contents: {:?}", mesh.sieve);

        // Verify that `get_cell_vertices` retrieves the correct vertices.
        let cell_vertices = mesh.get_cell_vertices(&cell);
        println!("Cell vertices: {:?}", cell_vertices);
        assert_eq!(cell_vertices.len(), 4, "Expected 4 vertices for a tetrahedron cell.");

        // Validate the shape before computing.
        assert_eq!(mesh.get_cell_shape(&cell), Ok(CellShape::Tetrahedron));

        // Create a Geometry instance and compute the centroid.
        let mut geometry = Geometry::new();
        let centroid = geometry.compute_cell_centroid(&mesh, &cell);

        // Expected centroid is the average of all vertices: (0.25, 0.25, 0.25)
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
    }

    /* #[test]
    fn test_compute_cell_volume_hexahedron() {
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
        let vertex5 = MeshEntity::Vertex(5);
        let vertex6 = MeshEntity::Vertex(6);
        let vertex7 = MeshEntity::Vertex(7);
        let vertex8 = MeshEntity::Vertex(8);
        let cell = MeshEntity::Cell(2);
    
        // Set vertex coordinates for a regular hexahedron (cube).
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [1.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(4, [0.0, 1.0, 0.0]);
        mesh.set_vertex_coordinates(5, [0.0, 0.0, 1.0]);
        mesh.set_vertex_coordinates(6, [1.0, 0.0, 1.0]);
        mesh.set_vertex_coordinates(7, [1.0, 1.0, 1.0]);
        mesh.set_vertex_coordinates(8, [0.0, 1.0, 1.0]);
    
        // Add entities to the mesh.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(vertex4);
        mesh.add_entity(vertex5);
        mesh.add_entity(vertex6);
        mesh.add_entity(vertex7);
        mesh.add_entity(vertex8);
        mesh.add_entity(cell);
    
        // Establish relationships between the cell and vertices.
        mesh.add_arrow(cell, vertex1);
        mesh.add_arrow(cell, vertex2);
        mesh.add_arrow(cell, vertex3);
        mesh.add_arrow(cell, vertex4);
        mesh.add_arrow(cell, vertex5);
        mesh.add_arrow(cell, vertex6);
        mesh.add_arrow(cell, vertex7);
        mesh.add_arrow(cell, vertex8);
    
        // Retrieve and verify the vertices of the cell.
        let cell_vertices = mesh.get_cell_vertices(&cell);
        println!("Retrieved cell vertices: {:?}", cell_vertices);
        assert_eq!(cell_vertices.len(), 8, "Expected 8 vertices for a hexahedron cell.");
    
        // Validate the shape before computing.
        let shape = mesh.get_cell_shape(&cell);
        println!("Determined cell shape: {:?}", shape);
        assert_eq!(shape, Ok(CellShape::Hexahedron), "Expected cell shape to be Hexahedron.");
    
        // Create a Geometry instance and compute the volume.
        let mut geometry = Geometry::new();
        let volume = geometry.compute_hexahedron_volume(&cell_vertices);
        println!("Computed volume: {}", volume);
    
        // The volume of a cube with side length 1 is 1^3 = 1.0.
        assert!((volume - 1.0).abs() < 1e-10, "Volume of the hexahedron is incorrect");
    } */

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
}
