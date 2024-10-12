Carefully analyze the below `Detailed Report on Leveraging Parallel Computation for Performance Gains` and then internalize this guidance such that you can apply it to the following source code of the `src/geometry/` module of Hydra. We will be revising the code for the `src/geometry/` module following this guidance. Your instructions are to provide complete revised code based on the following recommendations and source code provided below. You can simply provide revised code blocks based on the following guidance and recommended upgrades.

### Detailed Report on Leveraging Parallel Computation for Performance Gains

#### Overview
The recommendation to leverage parallel computation in the geometry module of the Hydra program focuses on improving the performance of computationally intensive tasks like volume calculation and centroid determination. This is particularly important when dealing with large meshes containing numerous geometric cells, where traditional sequential approaches can become a bottleneck. By employing parallelism, specifically using Rust's concurrency model, the module can utilize multiple CPU cores, significantly reducing execution time.

#### Key Concepts from Parallel Computational Geometry
1. **Divide-and-Conquer Strategies**:
   - These strategies break down a problem into smaller sub-problems, solve them independently, and then combine the results.
   - For example, calculating the volume of a large mesh can be divided into calculating the volumes of individual cells, and the results can be aggregated. This method suits parallel computation as each cell’s volume can be computed independently of others.

2. **Task-based Parallelism**:
   - Task-based parallelism is ideal for computations that involve iterating over a large set of data and performing similar operations on each item.
   - In the context of the Hydra geometry module, this could mean computing the centroid for each geometric cell or applying transformations to multiple vertex positions in parallel.

#### Parallelism in Rust: `Rayon` Crate
Rust’s `Rayon` crate is a library designed to enable easy parallel iteration. It simplifies the process of converting sequential operations into parallel ones, making it suitable for the geometry module in Hydra. The `Rayon` crate provides methods like `par_iter()` for parallel iteration over collections, and `map()`, `reduce()`, and `for_each()` for performing operations on elements in parallel.

##### Example: Volume Computation using `Rayon`
Here's how you might transform a volume computation function using `Rayon` for parallelism:

- **Original Sequential Implementation**:
    ```rust
    pub fn compute_total_volume(cells: &Vec<GeometryCell>) -> f64 {
        cells.iter().map(|cell| cell.compute_volume()).sum()
    }
    ```
    In this example, the function iterates over each cell in the mesh, calculates its volume using `compute_volume()`, and sums up the results. This is sequential, meaning that each volume calculation waits for the previous one to complete.

- **Parallel Implementation with `Rayon`**:
    ```rust
    use rayon::prelude::*;

    pub fn compute_total_volume(cells: &Vec<GeometryCell>) -> f64 {
        cells.par_iter().map(|cell| cell.compute_volume()).sum()
    }
    ```
    Here, `par_iter()` replaces `iter()`, enabling parallel iteration. Now, each cell’s volume can be calculated concurrently, utilizing multiple CPU cores. This approach is especially beneficial when `cells` is large, as it allows volume calculations for many cells to occur simultaneously.

##### Example: Parallel Transformation of Vertex Positions
When updating the positions of vertices in a geometry (e.g., during deformation or mesh adjustment), parallelizing the updates can improve performance:

- **Original Sequential Implementation**:
    ```rust
    pub fn update_vertices(vertices: &mut Vec<Point3D>, transform: &Transform) {
        for vertex in vertices.iter_mut() {
            *vertex = transform.apply(*vertex);
        }
    }
    ```

- **Parallel Implementation with `Rayon`**:
    ```rust
    use rayon::prelude::*;

    pub fn update_vertices(vertices: &mut Vec<Point3D>, transform: &Transform) {
        vertices.par_iter_mut().for_each(|vertex| {
            *vertex = transform.apply(*vertex);
        });
    }
    ```
    In this parallel version, `par_iter_mut()` allows each vertex update to be processed concurrently, distributing the work across multiple cores and thus speeding up the operation.

#### Guidance for Implementation in Hydra’s Geometry Module
1. **Integrating `Rayon`**:
   - Add `Rayon` as a dependency in the `Cargo.toml` file of the Hydra project:
     ```toml
     [dependencies]
     rayon = "1.6"
     ```
   - Use `par_iter()` for iterating over collections of geometric entities like vertices, cells, or faces. The change is typically minimal—replacing `iter()` with `par_iter()`—but can yield significant performance improvements for large datasets.

2. **Identifying Computational Hotspots**:
   - Before implementing parallelism, profile the existing code to identify the most time-consuming operations. This can be done using Rust profiling tools like `perf` or the `cargo-flamegraph` crate.
   - Focus on parallelizing the most computationally expensive functions first, such as those involving volume, surface area, or centroid calculations.

3. **Consider Load Balancing**:
   - When parallelizing tasks, ensure that the workload is evenly distributed across threads. `Rayon` manages load balancing automatically, but it’s important to ensure that each task (e.g., volume calculation for a cell) is not disproportionately more expensive than others.
   - If cells vary significantly in computational complexity (e.g., some cells have more vertices than others), consider using `rayon::scope()` to manually balance more complex tasks.

4. **Testing and Verification**:
   - Thoroughly test the parallelized functions to ensure correctness, as concurrency can introduce new challenges like race conditions if mutable state is shared improperly.
   - Use Rust’s strong type system and the `Sync` and `Send` traits to ensure that data types are safe to share across threads.

5. **Benchmarking Performance Gains**:
   - After implementing parallelism, benchmark the new functions against their sequential counterparts using Rust’s `criterion` crate. This will help quantify the performance improvements and ensure that the overhead of creating threads does not negate the benefits for smaller datasets.
   - Aim for a balance where the parallel implementation is significantly faster for large datasets without a noticeable performance hit for smaller ones.

#### Expected Benefits
- **Improved Scalability**: As the size of the mesh increases, the parallel approach will scale better than the sequential one, providing near-linear speedup for certain operations.
- **Enhanced User Experience**: For interactive applications like simulations or visualizations where real-time feedback is crucial, parallel computations can help maintain smooth performance.
- **Better Resource Utilization**: Utilizing all available CPU cores ensures that the program makes the most of modern multi-core processors, leading to a more efficient computational geometry engine.

#### Potential Challenges
- **Overhead of Parallelization**: For smaller meshes or simpler geometries, the overhead of spawning threads can outweigh the benefits of parallelism. It's essential to ensure that the data size justifies the parallel approach.
- **Concurrency Bugs**: Parallelism introduces risks like race conditions and deadlocks, though Rust’s ownership model mitigates many of these issues. Careful testing is still necessary.
- **Balancing Readability and Performance**: Introducing parallelism can make the codebase more complex. It’s important to maintain a balance between optimizing for performance and keeping the code maintainable for future developers.

### Conclusion
Integrating parallel computation into the Hydra geometry module using the `Rayon` crate offers substantial performance gains for large-scale simulations. By parallelizing key operations like volume and centroid calculations, the module can handle larger datasets more efficiently, enabling faster simulations and analyses. Implementing these changes thoughtfully, with attention to profiling, testing, and benchmarking, will ensure a robust and performant geometry module.

---

### Source Code

1. `src/geometry/mod.rs`

```rust

use rustc_hash::FxHashMap;

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
    pub fn compute_cell_centroid(&mut self, cell_id: usize, cell_shape: CellShape, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        if let Some(cached) = self.cache.get(&cell_id).and_then(|c| c.centroid) {
            return cached;
        }

        let centroid = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_centroid(cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_centroid(cell_vertices),
            CellShape::Prism => self.compute_prism_centroid(cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_centroid(cell_vertices),
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
    pub fn compute_cell_volume(&mut self, cell_id: usize, cell_shape: CellShape, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        if let Some(cached) = self.cache.get(&cell_id).and_then(|c| c.volume) {
            return cached;
        }

        let volume = match cell_shape {
            CellShape::Tetrahedron => self.compute_tetrahedron_volume(cell_vertices),
            CellShape::Hexahedron => self.compute_hexahedron_volume(cell_vertices),
            CellShape::Prism => self.compute_prism_volume(cell_vertices),
            CellShape::Pyramid => self.compute_pyramid_volume(cell_vertices),
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
        let mut geometry = Geometry::new();
        let cell_vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let centroid = geometry.compute_cell_centroid(1, CellShape::Tetrahedron, &cell_vertices);

        // Expected centroid is the average of all vertices: (0.25, 0.25, 0.25)
        assert_eq!(centroid, [0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_compute_cell_volume_hexahedron() {
        let mut geometry = Geometry::new();

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

        let volume = geometry.compute_cell_volume(2, CellShape::Hexahedron, &hexahedron_vertices);

        // The volume of a cube with side length 1 is 1^3 = 1.0
        assert!((volume - 1.0).abs() < 1e-10, "Volume of the hexahedron is incorrect");
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
}


```

2. `src/geometry/triangle.rs`

```rust

use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a triangular face.
    pub fn compute_triangle_centroid(&self, triangle_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(triangle_vertices.len() == 3, "Triangle must have exactly 3 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for vertex in triangle_vertices {
            centroid[0] += vertex[0];
            centroid[1] += vertex[1];
            centroid[2] += vertex[2];
        }

        let num_vertices = triangle_vertices.len() as f64;
        centroid[0] /= num_vertices;
        centroid[1] /= num_vertices;
        centroid[2] /= num_vertices;

        centroid
    }

    /// Computes the area of a triangular face.
    ///
    /// # Arguments
    /// * `triangle_vertices` - A vector of 3D coordinates for the triangle vertices.
    ///
    /// # Returns
    /// * `f64` - The area of the triangle.
    pub fn compute_triangle_area(&self, triangle_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(
            triangle_vertices.len() == 3,
            "Triangle must have exactly 3 vertices"
        );

        let v0 = triangle_vertices[0];
        let v1 = triangle_vertices[1];
        let v2 = triangle_vertices[2];

        // Compute vectors v0->v1 and v0->v2
        let e1 = [
            v1[0] - v0[0],
            v1[1] - v0[1],
            v1[2] - v0[2],
        ];
        let e2 = [
            v2[0] - v0[0],
            v2[1] - v0[1],
            v2[2] - v0[2],
        ];

        // Compute cross product of the two vectors
        let cross_product = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        // Compute the magnitude of the cross product vector
        let cross_product_magnitude = (cross_product[0].powi(2)
            + cross_product[1].powi(2)
            + cross_product[2].powi(2))
        .sqrt();

        // The area is half the magnitude of the cross product
        let area = 0.5 * cross_product_magnitude;
        area
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

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
        // ([0.0, 0.0, 0.0] + [3.0, 0.0, 0.0] + [0.0, 4.0, 0.0]) / 3 = [1.0, 1.3333, 0.0]
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

3. `src/geometry/quadrilateral.rs`

```rust

use crate::geometry::Geometry;

impl Geometry {
    /// Computes the area of a quadrilateral face.
    ///
    /// # Arguments
    /// * `quad_vertices` - A vector of 3D coordinates for the quadrilateral vertices.
    ///
    /// # Returns
    /// * `f64` - The area of the quadrilateral.
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
}

#[cfg(test)]
mod tests {
    use crate::geometry::Geometry;

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

        // The centroid of a square centered at (0.5, 0.5, 0.0)
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

4. `src/geometry/tetrahedron.rs`

```rust

use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a tetrahedral cell
    pub fn compute_tetrahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        let mut centroid = [0.0, 0.0, 0.0];
        for v in cell_vertices {
            centroid[0] += v[0];
            centroid[1] += v[1];
            centroid[2] += v[2];
        }
        let num_vertices = cell_vertices.len() as f64;
        centroid[0] /= num_vertices;
        centroid[1] /= num_vertices;
        centroid[2] /= num_vertices;
        centroid
    }

    /// Computes the volume of a tetrahedral cell
    ///
    /// # Arguments
    /// * `tet_vertices` - A vector of 4 vertices representing the 3D coordinates of the tetrahedron's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the tetrahedron.
    pub fn compute_tetrahedron_volume(&self, tet_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(tet_vertices.len() == 4, "Tetrahedron must have exactly 4 vertices");

        let v0 = tet_vertices[0];
        let v1 = tet_vertices[1];
        let v2 = tet_vertices[2];
        let v3 = tet_vertices[3];

        let matrix = [
            [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]],
            [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]],
            [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]],
        ];

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

5. `src/geometry/prism.rs`

```rust

use crate::geometry::Geometry;

impl Geometry {
    
    /// Computes the centroid of a prism cell (assuming a triangular prism).
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 6 vertices representing the 3D coordinates of the prism's vertices.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the prism centroid.
    pub fn compute_prism_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(cell_vertices.len() == 6, "Triangular prism must have exactly 6 vertices");

        // Split into top and bottom triangles
        let top_triangle = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]];
        let bottom_triangle = vec![cell_vertices[3], cell_vertices[4], cell_vertices[5]];

        // Compute the centroids of both triangles
        let top_centroid = self.compute_tetrahedron_centroid(&top_triangle);
        let bottom_centroid = self.compute_tetrahedron_centroid(&bottom_triangle);

        // Compute the centroid of the prism by averaging the top and bottom centroids
        let prism_centroid = [
            (top_centroid[0] + bottom_centroid[0]) / 2.0,
            (top_centroid[1] + bottom_centroid[1]) / 2.0,
            (top_centroid[2] + bottom_centroid[2]) / 2.0,
        ];

        prism_centroid
    }

    /// Computes the volume of a prism cell (assuming a triangular prism).
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 6 vertices representing the 3D coordinates of the prism's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the triangular prism.
    pub fn compute_prism_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(cell_vertices.len() == 6, "Triangular prism must have exactly 6 vertices");

        // Split into top and bottom triangles
        let top_triangle = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]];
        let bottom_triangle = vec![cell_vertices[3], cell_vertices[4], cell_vertices[5]];

        // Compute the area of the base triangle (bottom triangle)
        let base_area = self.compute_triangle_area(&bottom_triangle);

        // Compute the height of the prism as the distance between the top and bottom triangle centroids
        let top_centroid = self.compute_tetrahedron_centroid(&top_triangle);
        let bottom_centroid = self.compute_tetrahedron_centroid(&bottom_triangle);
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

6. `src/geometry/pyramid.rs`

```rust

use crate::geometry::{Geometry, FaceShape};

impl Geometry {
    /// Computes the centroid of a pyramid cell (triangular or square base).
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 4 (triangular base) or 5 (square base) vertices.
    ///
    /// # Returns
    /// * `[f64; 3]` - The 3D coordinates of the pyramid centroid.
    pub fn compute_pyramid_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        let mut _total_volume = 0.0;
        let mut weighted_centroid = [0.0, 0.0, 0.0];

        let num_vertices = cell_vertices.len();
        assert!(num_vertices == 4 || num_vertices == 5, "Pyramid must have 4 (triangular) or 5 (square) vertices");

        // Define the apex and base vertices
        let apex = if num_vertices == 4 { cell_vertices[3] } else { cell_vertices[4] };
        let base_vertices = if num_vertices == 4 {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]] // Triangular base
        } else {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], cell_vertices[3]] // Square base
        };

        // For a square-based pyramid (split into two tetrahedrons), adjust centroid calculation:
        if num_vertices == 5 {
            let tetra1 = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            let tetra2 = vec![cell_vertices[0], cell_vertices[2], cell_vertices[3], apex];

            // Compute volumes and centroids
            let volume1 = self.compute_tetrahedron_volume(&tetra1);
            let volume2 = self.compute_tetrahedron_volume(&tetra2);

            let centroid1 = self.compute_tetrahedron_centroid(&tetra1);
            let centroid2 = self.compute_tetrahedron_centroid(&tetra2);

            // Total volume of pyramid is sum of tetrahedron volumes
            _total_volume = volume1 + volume2;

            // Weighted centroid: combining centroids of both tetrahedrons
            weighted_centroid[0] = (centroid1[0] * volume1 + centroid2[0] * volume2) / _total_volume;
            weighted_centroid[1] = (centroid1[1] * volume1 + centroid2[1] * volume2) / _total_volume;
            weighted_centroid[2] = (centroid1[2] * volume1 + centroid2[2] * volume2) / _total_volume;
            
            // NOTE: Do not reapply the base-apex averaging rule here as weighted centroids are already applied.
        } else {
            // For triangular base (tetrahedron), no need for further splitting
            let tetra = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            let volume = self.compute_tetrahedron_volume(&tetra);
            let centroid = self.compute_tetrahedron_centroid(&tetra);

            _total_volume = volume;
            weighted_centroid = centroid;
        }

        // Correct the centroid by adjusting the weight between the base centroid and apex
        let base_centroid = self.compute_face_centroid(
            if num_vertices == 4 { FaceShape::Triangle } else { FaceShape::Quadrilateral },
            &base_vertices,
        );

        // Apply the correct weighting for the pyramid's centroid.
        // Pyramid centroid = 3/4 * base_centroid + 1/4 * apex
        weighted_centroid[0] = (3.0 * base_centroid[0] + apex[0]) / 4.0;
        weighted_centroid[1] = (3.0 * base_centroid[1] + apex[1]) / 4.0;
        weighted_centroid[2] = (3.0 * base_centroid[2] + apex[2]) / 4.0;

        println!("Adjusted weighted centroid: {:?}", weighted_centroid);

        weighted_centroid
    }

    /// Computes the volume of a pyramid cell (triangular or square base).
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 4 or 5 vertices representing the 3D coordinates of the pyramid's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the pyramid.
    pub fn compute_pyramid_volume(&self, cell_vertices: &Vec<[f64; 3]>) -> f64 {
        let mut _total_volume = 0.0;

        let num_vertices = cell_vertices.len();
        assert!(num_vertices == 4 || num_vertices == 5, "Pyramid must have 4 (triangular) or 5 (square) vertices");

        // Define the apex and base vertices
        let apex = if num_vertices == 4 { cell_vertices[3] } else { cell_vertices[4] };
        let _base_vertices = if num_vertices == 4 {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2]] // Triangular base
        } else {
            vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], cell_vertices[3]] // Square base
        };

        // Split into two tetrahedrons if square base
        if num_vertices == 5 {
            let tetra1 = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            let tetra2 = vec![cell_vertices[0], cell_vertices[2], cell_vertices[3], apex];

            // Compute volumes
            let volume1 = self.compute_tetrahedron_volume_local(&tetra1);
            let volume2 = self.compute_tetrahedron_volume_local(&tetra2);

            _total_volume = volume1 + volume2;
        } else {
            // Single tetrahedron for triangular base pyramid
            let tetra = vec![cell_vertices[0], cell_vertices[1], cell_vertices[2], apex];
            _total_volume = self.compute_tetrahedron_volume_local(&tetra);
        }

        println!("Total volume: {:?}", _total_volume);
        _total_volume
    }

    /// Computes the volume of a tetrahedron
    fn compute_tetrahedron_volume_local(&self, vertices: &Vec<[f64; 3]>) -> f64 {
        assert!(vertices.len() == 4, "Tetrahedron must have 4 vertices");

        let a = vertices[0];
        let b = vertices[1];
        let c = vertices[2];
        let d = vertices[3];

        let v1 = [b[0] - d[0], b[1] - d[1], b[2] - d[2]];
        let v2 = [c[0] - d[0], c[1] - d[1], c[2] - d[2]];
        let v3 = [a[0] - d[0], a[1] - d[1], a[2] - d[2]];

        // Volume formula: V = 1/6 * |v1 . (v2 x v3)|
        let cross_product = [
            v2[1] * v3[2] - v2[2] * v3[1],
            v2[2] * v3[0] - v2[0] * v3[2],
            v2[0] * v3[1] - v2[1] * v3[0],
        ];

        let dot_product = v1[0] * cross_product[0] + v1[1] * cross_product[1] + v1[2] * cross_product[2];
        (dot_product.abs() / 6.0).abs()
    }

    /// Computes the centroid of a tetrahedron
    fn _compute_tetrahedron_centroid_local(&self, vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        assert!(vertices.len() == 4, "Tetrahedron must have 4 vertices");

        let mut centroid = [0.0, 0.0, 0.0];
        for v in vertices {
            centroid[0] += v[0];
            centroid[1] += v[1];
            centroid[2] += v[2];
        }

        centroid[0] /= 4.0;
        centroid[1] /= 4.0;
        centroid[2] /= 4.0;

        centroid
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

        // The expected volume of the square pyramid is (1/3) * base area * height
        // Base area = 1.0, height = 1.0 -> Volume = (1/3) * 1.0 * 1.0 = 1/3
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

        // The expected volume of the triangular pyramid (tetrahedron) is (1/3) * base area * height
        // Base area = 0.5, height = 1.0 -> Volume = (1/3) * 0.5 * 1.0 = 1/6
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

        // The correct centroid is at (0.5, 0.5, 0.25)
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

        // The correct centroid is at (0.375, 0.375, 0.25)
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

        // The volume of a degenerate pyramid should be zero
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

        // The centroid of this degenerate pyramid should still be computed correctly
        let expected_centroid = [0.375, 0.375, 0.0];
        assert_eq!(centroid, expected_centroid);
    }
}

```

7. `src/geometry/hexahedron.rs`

```rust

use crate::geometry::Geometry;

impl Geometry {
    /// Computes the centroid of a hexahedral cell
    pub fn compute_hexahedron_centroid(&self, cell_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        let mut centroid = [0.0, 0.0, 0.0];
        for v in cell_vertices {
            centroid[0] += v[0];
            centroid[1] += v[1];
            centroid[2] += v[2];
        }
        let num_vertices = cell_vertices.len() as f64;
        centroid[0] /= num_vertices;
        centroid[1] /= num_vertices;
        centroid[2] /= num_vertices;
        centroid
    }

    /// Computes the volume of a hexahedral (cube-like) cell using tetrahedral decomposition.
    ///
    /// # Arguments
    /// * `cell_vertices` - A vector of 8 vertices representing the 3D coordinates of the hexahedron's vertices.
    ///
    /// # Returns
    /// * `f64` - The volume of the hexahedron.
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