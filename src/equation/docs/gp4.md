Please evaluate the below notes and the source code provided to determine the extent to which new functions or implementations might need to be defined, based on solutions above. If additional functions need to be implemented, generate a complete upgraded version of the source code file based on the provided starting source code provided after these notes.

**Note on Additional Functions:**

In the upgraded code, several methods and functions are used that may require implementation elsewhere:

1. **Mesh Methods**:
   - `domain.get_cells_sharing_face(&face)`
   - `domain.get_face_normal(&face, None)`
   - `domain.get_face_area(&face)`
   - Ensure these methods are implemented in the `Mesh` struct.

2. **Geometry Methods**:
   - `Geometry::new()`
   - Geometry computations like face normals and areas.

3. **BoundaryConditionHandler Methods**:
   - `boundary_handler.get_bc(&face)`
   - Boundary condition application logic may need to be fleshed out.

4. **Fields Struct Methods**:
   - Methods for getting and setting field values are implemented, but the `Section` struct's methods like `restrict` and `set_data` need to be properly defined.

5. **TimeStepper Trait Implementation**:
   - The `TimeStepper` trait and its methods like `current_time()`, `get_time_step()`, and `step()` must be implemented for the specific time-stepping scheme used.

6. **TimeDependentProblem Trait**:
   - The `compute_rhs`, `time_to_scalar`, and other methods in `TimeDependentProblem` require concrete implementations based on the problem.

7. **Vector Trait for Fields**:
   - Since `Fields` is used as the `State` in `TimeDependentProblem`, it may need to implement the `Vector` trait or adjust the implementation accordingly.

Please ensure these functions and methods are properly defined in their respective modules to complete the integration of the upgraded code.

---

`src/domain/mesh/geometry.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;

impl Mesh {
    /// Retrieves all the faces of a given cell, filtering only face entities.
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<DashMap<MeshEntity, ()>> {
        self.sieve.cone(cell).map(|set| {
            let faces = DashMap::new();
            set.into_iter()
                .filter(|entity| matches!(entity, MeshEntity::Face(_)))
                .for_each(|face| {
                    faces.insert(face, ());
                });
            faces
        })
    }

    /// Retrieves all the cells that share the given face, filtering only cell entities that are present in the mesh.
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let cells = DashMap::new();
        let entities = self.entities.read().unwrap();
        self.sieve
            .support(face)
            .into_iter()
            .filter(|entity| matches!(entity, MeshEntity::Cell(_)) && entities.contains(entity))
            .for_each(|cell| {
                cells.insert(cell, ());
            });
        cells
    }

    /// Computes the Euclidean distance between two cells based on their centroids.
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.
    pub fn get_face_area(&self, face: &MeshEntity) -> Option<f64> {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let mut geometry = Geometry::new();
        let face_id = face.get_id();
        Some(geometry.compute_face_area(face_id, face_shape, &face_vertices))
    }

    /// Computes the centroid of a cell based on its vertices.
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let _cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        let mut geometry = Geometry::new();
        geometry.compute_cell_centroid(self, cell)
    }

    /// Retrieves all vertices connected to the given vertex by shared cells.
    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let neighbors = DashMap::new();
        let connected_cells = self.sieve.support(vertex);

        connected_cells.into_iter().for_each(|cell| {
            if let Some(cell_vertices) = self.sieve.cone(&cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(v.clone(), ());
                    }
                }
            }
        });
        neighbors.into_iter().map(|(vertex, _)| vertex).collect()
    }

    /// Returns an iterator over the IDs of all vertices in the mesh.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on the number of vertices it has.
    pub fn get_cell_shape(&self, cell: &MeshEntity) -> Result<CellShape, String> {
        let cell_vertices = self.get_cell_vertices(cell);
        match cell_vertices.len() {
            4 => Ok(CellShape::Tetrahedron),
            5 => Ok(CellShape::Pyramid),
            6 => Ok(CellShape::Prism),
            8 => Ok(CellShape::Hexahedron),
            _ => Err(format!(
                "Unsupported cell shape with {} vertices. Expected 4, 5, 6, or 8 vertices.",
                cell_vertices.len()
            )),
        }
    }

    /// Retrieves the vertices of a cell and their coordinates, sorted by vertex ID.
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_entities) = self.sieve.cone(cell) {
            for entity in connected_entities {
                if let MeshEntity::Vertex(vertex_id) = entity {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                }
            }
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }

    /// Retrieves the vertices of a face and their coordinates, sorted by vertex ID.
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertex_ids_and_coords = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                }
            }
            vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);
        }
        vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect()
    }

    /// Computes the normal vector of a face based on its vertices and shape.
    ///
    /// This function calculates the outward normal vector for a face by leveraging the
    /// `Geometry` module. It determines the face shape and uses the vertices to compute
    /// the vector. The orientation of the normal can optionally depend on a neighboring cell.
    ///
    /// # Arguments
    /// * `face` - The face entity for which the normal is computed.
    /// * `reference_cell` - Optional cell entity to determine the normal orientation.
    ///
    /// # Returns
    /// * `Option<[f64; 3]>` - The computed normal vector if successful, otherwise `None`.
    pub fn get_face_normal(
        &self,
        face: &MeshEntity,
        reference_cell: Option<&MeshEntity>,
    ) -> Option<[f64; 3]> {
        // Retrieve face vertices
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => return None, // Unsupported face shape
        };

        let geometry = Geometry::new();
        let normal = match face_shape {
            FaceShape::Triangle => geometry.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => geometry.compute_quadrilateral_normal(&face_vertices),
        };

        // If a reference cell is provided, adjust the normal's orientation
        if let Some(cell) = reference_cell {
            let cell_centroid = self.get_cell_centroid(cell);
            let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            // Compute the vector from the face centroid to the cell centroid
            let to_cell_vector = [
                cell_centroid[0] - face_centroid[0],
                cell_centroid[1] - face_centroid[1],
                cell_centroid[2] - face_centroid[2],
            ];

            // Ensure the normal points outward by checking the dot product
            let dot_product = normal[0] * to_cell_vector[0]
                + normal[1] * to_cell_vector[1]
                + normal[2] * to_cell_vector[2];

            if dot_product < 0.0 {
                // Reverse the normal direction to make it outward-pointing
                return Some([-normal[0], -normal[1], -normal[2]]);
            }
        }

        Some(normal)
    }
}
```

---

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
```

---

`src/boundary/bc_handler.rs`

```rust
use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;
use crate::boundary::dirichlet::DirichletBC;
use crate::boundary::neumann::NeumannBC;
use crate::boundary::robin::RobinBC;
use crate::boundary::mixed::MixedBC;
use crate::boundary::cauchy::CauchyBC;
use faer::MatMut;

pub type BoundaryConditionFn = Arc<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// BoundaryCondition represents various types of boundary conditions
/// that can be applied to mesh entities.
#[derive(Clone)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Robin { alpha: f64, beta: f64 },
    Mixed { gamma: f64, delta: f64 },
    Cauchy { lambda: f64, mu: f64 },
    DirichletFn(BoundaryConditionFn),
    NeumannFn(BoundaryConditionFn),
}

/// The BoundaryConditionHandler struct is responsible for managing
/// boundary conditions associated with specific mesh entities.
pub struct BoundaryConditionHandler {
    conditions: DashMap<MeshEntity, BoundaryCondition>,
}

impl BoundaryConditionHandler {
    /// Creates a new BoundaryConditionHandler with an empty map to store boundary conditions.
    pub fn new() -> Self {
        Self {
            conditions: DashMap::new(),
        }
    }

    /// Sets a boundary condition for a specific mesh entity.
    pub fn set_bc(&self, entity: MeshEntity, condition: BoundaryCondition) {
        self.conditions.insert(entity, condition);
    }

    /// Retrieves the boundary condition applied to a specific mesh entity, if it exists.
    pub fn get_bc(&self, entity: &MeshEntity) -> Option<BoundaryCondition> {
        self.conditions.get(entity).map(|entry| entry.clone())
    }

    pub fn get_boundary_faces(&self) -> Vec<MeshEntity> {
        self.conditions.iter()
            .map(|entry| entry.key().clone()) // Extract the keys (MeshEntities) from the map
            .filter(|entity| matches!(entity, MeshEntity::Face(_))) // Filter for Face entities
            .collect()
    }

    /// Applies the boundary conditions to the system matrices and right-hand side vectors.
    pub fn apply_bc(
        &self,
        matrix: &mut MatMut<f64>,
        rhs: &mut MatMut<f64>,
        boundary_entities: &[MeshEntity],
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        for entity in boundary_entities {
            if let Some(bc) = self.get_bc(entity) {
                let index = *entity_to_index.get(entity).unwrap();
                match bc {
                    BoundaryCondition::Dirichlet(value) => {
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::Neumann(flux) => {
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, flux);
                    }
                    BoundaryCondition::Robin { alpha, beta } => {
                        let robin_bc = RobinBC::new();
                        robin_bc.apply_robin(matrix, rhs, index, alpha, beta);
                    }
                    BoundaryCondition::DirichletFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let dirichlet_bc = DirichletBC::new();
                        dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
                    }
                    BoundaryCondition::NeumannFn(fn_bc) => {
                        let coords = [0.0, 0.0, 0.0];
                        let value = fn_bc(time, &coords);
                        let neumann_bc = NeumannBC::new();
                        neumann_bc.apply_constant_neumann(rhs, index, value);
                    }
                    BoundaryCondition::Mixed { gamma, delta } => {
                        let mixed_bc = MixedBC::new();
                        mixed_bc.apply_mixed(matrix, rhs, index, gamma, delta);
                    }
                    BoundaryCondition::Cauchy { lambda, mu } => {
                        let cauchy_bc = CauchyBC::new();
                        cauchy_bc.apply_cauchy(matrix, rhs, index, lambda, mu);
                    }
                }
            }
        }
    }
}

/// The BoundaryConditionApply trait defines the `apply` method, which is used to apply 
/// a boundary condition to a given mesh entity.
pub trait BoundaryConditionApply {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    );
}

impl BoundaryConditionApply for BoundaryCondition {
    fn apply(
        &self,
        entity: &MeshEntity,
        rhs: &mut MatMut<f64>,
        matrix: &mut MatMut<f64>,
        entity_to_index: &DashMap<MeshEntity, usize>,
        time: f64,
    ) {
        let index = *entity_to_index.get(entity).unwrap();
        match self {
            BoundaryCondition::Dirichlet(value) => {
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, *value);
            }
            BoundaryCondition::Neumann(flux) => {
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, *flux);
            }
            BoundaryCondition::Robin { alpha, beta } => {
                let robin_bc = RobinBC::new();
                robin_bc.apply_robin(matrix, rhs, index, *alpha, *beta);
            }
            BoundaryCondition::DirichletFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let dirichlet_bc = DirichletBC::new();
                dirichlet_bc.apply_constant_dirichlet(matrix, rhs, index, value);
            }
            BoundaryCondition::NeumannFn(fn_bc) => {
                let coords = [0.0, 0.0, 0.0];
                let value = fn_bc(time, &coords);
                let neumann_bc = NeumannBC::new();
                neumann_bc.apply_constant_neumann(rhs, index, value);
            }
            BoundaryCondition::Mixed { gamma, delta } => {
                let mixed_bc = MixedBC::new();
                mixed_bc.apply_mixed(matrix, rhs, index, *gamma, *delta);
            }
            BoundaryCondition::Cauchy { lambda, mu } => {
                let cauchy_bc = CauchyBC::new();
                cauchy_bc.apply_cauchy(matrix, rhs, index, *lambda, *mu);
            }
        }
    }
}
```

---

`src/domain/section.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.  
/// It provides methods for setting, updating, and retrieving data, and supports  
/// parallel updates for performance improvements.  
///
/// Example usage:
///
///    let section = Section::new();  
///    let vertex = MeshEntity::Vertex(1);  
///    section.set_data(vertex, 42);  
///    assert_eq!(section.restrict(&vertex), Some(42));  
/// 
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.  
    pub data: DashMap<MeshEntity, T>,
}

impl<T> Section<T> {
    /// Creates a new `Section` with an empty data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn new() -> Self {
        Section {
            data: DashMap::new(),
        }
    }

    /// Sets the data associated with a given `MeshEntity`.  
    /// This method inserts the `entity` and its corresponding `value` into the data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    section.set_data(MeshEntity::Vertex(1), 10);  
    ///
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Restricts the data for a given `MeshEntity` by returning an immutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 42);  
    ///    assert_eq!(section.restrict(&vertex), Some(42));  
    ///
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Applies the given function in parallel to update all data values in the section.
    ///
    /// Example usage:
    ///
    ///    section.parallel_update(|v| *v += 1);  
    ///
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
        T: Send + Sync,
    {
        // Clone the keys to ensure safe access to each mutable entry in parallel.
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        // Apply the update function to each entry in parallel.
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Restricts the data for a given `MeshEntity` by returning a mutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 5);  
    ///    let mut value = section.restrict_mut(&vertex).unwrap();  
    ///    value = 10;  
    ///    section.set_data(vertex, value);  
    ///
    pub fn restrict_data_mut(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates the data for a specific `MeshEntity` by replacing the existing value  
    /// with the new value.  
    ///
    /// Example usage:
    ///
    ///    section.update_data(&MeshEntity::Vertex(1), 15);  
    ///
    pub fn update_data(&self, entity: &MeshEntity, new_value: T) {
        self.data.insert(*entity, new_value);
    }

    /// Clears all data from the section, removing all entity associations.  
    ///
    /// Example usage:
    ///
    ///    section.clear();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Retrieves all `MeshEntity` objects associated with the section.  
    ///
    /// Returns a vector containing all mesh entities currently stored in the section.  
    ///
    /// Example usage:
    ///
    ///    let entities = section.entities();  
    ///
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves all data stored in the section as immutable copies.  
    ///
    /// Returns a vector of data values.  
    ///
    /// Example usage:
    ///
    ///    let all_data = section.all_data();  
    ///
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Retrieves all data stored in the section with mutable access.  
    ///
    /// Returns a vector of data values that can be modified.  
    ///
    /// Example usage:
    ///
    ///    let all_data_mut = section.all_data_mut();  
    ///
    pub fn all_data_mut(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter_mut().map(|entry| entry.value().clone()).collect()
    }
}
```

---

`src/time_stepping/ts.rs`

```rust
use crate::linalg::Matrix;
use crate::linalg::Vector;

#[derive(Debug)]
pub enum TimeSteppingError {
    InvalidStep,
    SolverError(String),
}

pub trait TimeDependentProblem {
    type State: Vector + Clone;
    type Time: Copy + PartialOrd;

    fn compute_rhs(
        &self,
        time: Self::Time,
        state: &Self::State,
        derivative: &mut Self::State,
    ) -> Result<(), TimeSteppingError>;

    fn initial_state(&self) -> Self::State;

    fn time_to_scalar(&self, time: Self::Time) -> <Self::State as Vector>::Scalar;

    fn get_matrix(&self) -> Option<Box<dyn Matrix<Scalar = f64>>>;

    fn solve_linear_system(
        &self,
        matrix: &mut dyn Matrix<Scalar = f64>,
        state: &mut Self::State,
        rhs: &Self::State,
    ) -> Result<(), TimeSteppingError>;
}

pub trait TimeStepper<P>
where
    P: TimeDependentProblem + Sized,
{
    fn current_time(&self) -> P::Time;

    fn set_current_time(&mut self, time: P::Time);

    fn step(
        &mut self,
        problems: &[P], // Accept slice of `P`
        dt: P::Time,
        current_time: P::Time,
        state: &mut P::State,
    ) -> Result<(), TimeSteppingError>;

    fn adaptive_step(
        &mut self,
        problem: &P,
        state: &mut P::State,
    ) -> Result<P::Time, TimeSteppingError>;

    fn set_time_interval(&mut self, start_time: P::Time, end_time: P::Time);

    fn set_time_step(&mut self, dt: P::Time);

    fn get_time_step(&self) -> P::Time;
}

```

---

`src/linalg/vector/traits.rs`

```rust
// src/vector/traits.rs


/// Trait defining a set of common operations for vectors.
/// It abstracts over different vector types, enabling flexible implementations
/// for standard dense vectors or more complex matrix structures.
///
/// # Requirements:
/// Implementations of `Vector` must be thread-safe (`Send` and `Sync`).
pub trait Vector: Send + Sync {
    /// The scalar type of the vector elements.
    type Scalar: Copy + Send + Sync;

    /// Returns the length (number of elements) of the vector.
    fn len(&self) -> usize;

    /// Retrieves the element at index `i`.
    ///
    /// # Panics
    /// Panics if the index `i` is out of bounds.
    fn get(&self, i: usize) -> Self::Scalar;

    /// Sets the element at index `i` to `value`.
    ///
    /// # Panics
    /// Panics if the index `i` is out of bounds.
    fn set(&mut self, i: usize, value: Self::Scalar);

    /// Provides a slice of the underlying data.
    fn as_slice(&self) -> &[f64];

    /// Provides a mutable slice of the underlying data.
    fn as_mut_slice(&mut self) -> &mut [Self::Scalar];

    /// Computes the dot product of `self` with another vector `other`.
    ///
    /// # Example
    /// 
    /// ```rust
    /// use hydra::linalg::vector::traits::Vector;
    /// let vec1: Vec<f64> = vec![1.0, 2.0, 3.0];
    /// let vec2: Vec<f64> = vec![4.0, 5.0, 6.0];
    /// let dot_product = vec1.dot(&vec2);
    /// assert_eq!(dot_product, 32.0);
    /// ```
    fn dot(&self, other: &dyn Vector<Scalar = Self::Scalar>) -> Self::Scalar;

    /// Computes the Euclidean norm (L2 norm) of the vector.
    ///
    /// # Example
    /// ```rust
    /// use hydra::linalg::vector::traits::Vector;
    /// let vec: Vec<f64> = vec![3.0, 4.0];
    /// let norm = vec.norm();
    /// assert_eq!(norm, 5.0);
    /// ```
    fn norm(&self) -> Self::Scalar;

    /// Scales the vector by multiplying each element by the scalar `scalar`.
    fn scale(&mut self, scalar: Self::Scalar);

    /// Performs the operation `self = a * x + self`, also known as AXPY.
    fn axpy(&mut self, a: Self::Scalar, x: &dyn Vector<Scalar = Self::Scalar>);

    /// Adds another vector `other` to `self` element-wise.
    fn element_wise_add(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Multiplies `self` by another vector `other` element-wise.
    fn element_wise_mul(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Divides `self` by another vector `other` element-wise.
    fn element_wise_div(&mut self, other: &dyn Vector<Scalar = Self::Scalar>);

    /// Computes the cross product with another vector `other` (for 3D vectors only).
    ///
    /// # Errors
    /// Returns an error if the vectors are not 3-dimensional.
    fn cross(&mut self, other: &dyn Vector<Scalar = Self::Scalar>) -> Result<(), &'static str>;

    /// Computes the sum of all elements in the vector.
    fn sum(&self) -> Self::Scalar;

    /// Returns the maximum element of the vector.
    fn max(&self) -> Self::Scalar;

    /// Returns the minimum element of the vector.
    fn min(&self) -> Self::Scalar;

    /// Returns the mean value of the vector.
    fn mean(&self) -> Self::Scalar;

    /// Returns the variance of the vector.
    fn variance(&self) -> Self::Scalar;
}
```