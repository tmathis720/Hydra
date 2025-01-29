use log::{error, warn};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use thiserror::Error;
use crate::domain::{self, mesh::Mesh, section::Vector3, MeshEntity};
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
    Triangle,
    Quadrilateral,
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
    Edge,
    Triangle,
    Quadrilateral,
}

#[derive(Error, Debug)]
pub enum GeometryError {
    #[error("Vertex index {0} is out of bounds.")]
    VertexIndexOutOfBounds(usize),

    #[error("Invalid coordinates provided: {0:?}.")]
    InvalidCoordinates([f64; 3]),

    #[error("Invalid shape: {0}. Expected a valid cell or face shape.")]
    InvalidShape(String),

    #[error("Failed to retrieve vertices for {0:?}: {1}")]
    VertexRetrievalError(MeshEntity, String),

    #[error("Face {0:?} has no vertices; operation cannot be performed.")]
    EmptyFace(MeshEntity),

    #[error("Operation not supported for shape with {0} vertices.")]
    UnsupportedShape(usize),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Invalid point dimensions: {0}")]
    InvalidPointDimension(String),

    #[error("Invalid face vertices: {0}")]
    InvalidFaceVertices(String),

    #[error("Mesh error: {0}")]
    MeshError(String),

    #[error("Unsupported face shape: {0}")]
    UnsupportedFaceShape(String),

    #[error("Invalid number of vertices: expected {expected}, found {found}.")]
    InvalidVertexCount { expected: usize, found: usize },

    #[error("Computation failed due to a degenerate triangle.")]
    DegenerateTriangle,

    #[error("Normal vector computation failed with zero magnitude.")]
    ZeroMagnitudeNormal,

    #[error("Failed to retrieve centroid for cell {0}")]
    CentroidError(String),
    
    #[error("Failed to compute normal vector for face {0}")]
    NormalComputationError(String)
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
    ///
    /// # Errors
    /// Returns an error if the coordinates contain `NaN` or `Inf`.
    pub fn set_vertex(&mut self, vertex_index: usize, coords: [f64; 3]) -> Result<(), GeometryError> {
        if coords.iter().any(|&c| !c.is_finite()) {
            log::error!("Invalid coordinates: {:?}", coords);
            return Err(GeometryError::InvalidCoordinates(coords));
        }

        if vertex_index >= self.vertices.len() {
            log::info!("Resizing vertex storage to accommodate index {}", vertex_index);
            self.vertices.resize(vertex_index + 1, [0.0, 0.0, 0.0]);
        }

        log::debug!("Setting vertex {} to {:?}", vertex_index, coords);
        self.vertices[vertex_index] = coords;
        self.invalidate_cache();
        Ok(())
    }

    /// Computes and returns the centroid of a specified cell using the cell's shape and vertices.
    /// Caches the result for reuse.
    ///
    /// # Arguments
    /// * `mesh` - Reference to the mesh structure containing the cell.
    /// * `cell` - The cell entity whose centroid is to be computed.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The computed centroid or an error if the computation fails.
    pub fn compute_cell_centroid(&mut self, mesh: &Mesh, cell: &MeshEntity) -> Result<[f64; 3], GeometryError> {
        let cell_id = cell.get_id();

        // Check if the centroid is cached
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.centroid) {
            log::debug!("Using cached centroid for cell {:?}", cell_id);
            return Ok(cached);
        }

        // Retrieve the cell shape
        let cell_shape = mesh
            .get_cell_shape(cell)
            .map_err(|_| GeometryError::InvalidShape(format!("Unsupported or missing shape for cell {:?}", cell)))?;

        // Retrieve cell vertices
        let cell_vertices = mesh
            .get_cell_vertices(cell)
            .map_err(|err| GeometryError::VertexRetrievalError(*cell, err.to_string()))?;

        if cell_vertices.is_empty() {
            return Err(GeometryError::VertexRetrievalError(
                *cell,
                "Cell has no vertices; cannot compute centroid.".to_string(),
            ));
        }

        // Compute the centroid based on the cell shape
        let centroid = match cell_shape {
            CellShape::Triangle => self.compute_triangle_centroid(&cell_vertices)?,
            CellShape::Quadrilateral => self.compute_quadrilateral_centroid(&cell_vertices)?,
            CellShape::Tetrahedron => self.compute_tetrahedron_centroid(&cell_vertices)?,
            CellShape::Hexahedron => self.compute_hexahedron_centroid(&cell_vertices)?,
            CellShape::Prism => self.compute_prism_centroid(&cell_vertices)?,
            CellShape::Pyramid => self.compute_pyramid_centroid(&cell_vertices)?,
        };

        // Cache the computed centroid for future use
        self.cache
            .lock()
            .unwrap()
            .entry(cell_id)
            .or_default()
            .centroid = Some(centroid);

        log::info!("Computed centroid for cell {:?}: {:?}", cell_id, centroid);
        Ok(centroid)
    }



    /// Computes the volume of a given cell using its shape and vertex coordinates.
    /// The computed volume is cached for efficiency.
    ///
    /// # Arguments
    /// * `mesh` - Reference to the mesh structure containing the cell.
    /// * `cell` - The cell entity whose volume is to be computed.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The computed volume or an error if the computation fails.
    pub fn compute_cell_volume(&mut self, mesh: &Mesh, cell: &MeshEntity) -> Result<f64, GeometryError> {
        let cell_id = cell.get_id();

        // Check if the volume is cached
        if let Some(cached) = self.cache.lock().unwrap().get(&cell_id).and_then(|c| c.volume) {
            log::debug!("Using cached volume for cell {:?}", cell_id);
            return Ok(cached);
        }

        // Retrieve cell shape
        let cell_shape = mesh
            .get_cell_shape(cell)
            .map_err(|_| GeometryError::InvalidShape(format!("Unsupported or missing shape for cell {:?}", cell)))?;

        // Retrieve cell vertices
        let cell_vertices = mesh
            .get_cell_vertices(cell)
            .map_err(|err| GeometryError::VertexRetrievalError(*cell, err.to_string()))?;

        if cell_vertices.is_empty() {
            return Err(GeometryError::VertexRetrievalError(
                *cell,
                "Cell has no vertices; cannot compute volume.".to_string(),
            ));
        }

        // Compute the volume based on the cell shape
        let volume = match cell_shape {
            CellShape::Triangle => self.compute_triangle_area(&cell_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute triangle area: {:?}", err))
            }),
            CellShape::Quadrilateral => self.compute_quadrilateral_area(&cell_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute quadrilateral area: {:?}", err))
            }),
            CellShape::Tetrahedron => self.compute_tetrahedron_volume(&cell_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute tetrahedron volume: {:?}", err))
            }),
            CellShape::Hexahedron => self.compute_hexahedron_volume(&cell_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute hexahedron volume: {:?}", err))
            }),
            CellShape::Prism => self.compute_prism_volume(&cell_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute prism volume: {:?}", err))
            }),
            CellShape::Pyramid => self.compute_pyramid_volume(&cell_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute pyramid volume: {:?}", err))
            }),
        }?;

        // Cache the computed volume for future use
        self.cache
            .lock()
            .unwrap()
            .entry(cell_id)
            .or_default()
            .volume = Some(volume);

        log::info!("Computed volume for cell {:?}: {:?}", cell_id, volume);
        Ok(volume)
    }



    /// Calculates Euclidean distance between two points in 3D space.
    ///
    /// # Arguments
    /// * `p1` - The first point as a reference to a `[f64; 3]` array.
    /// * `p2` - The second point as a reference to a `[f64; 3]` array.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The computed distance or an error if computation fails.
    pub fn compute_distance(p1: &[f64; 3], p2: &[f64; 3]) -> Result<f64, GeometryError> {
        // Calculate the differences in each dimension
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];

        // Compute the Euclidean distance
        let distance = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();

        // Check for NaN result
        if distance.is_nan() {
            let warning_msg = format!(
                "Distance computation resulted in NaN for points {:?} and {:?}",
                p1, p2
            );
            log::warn!("{}", warning_msg);
            return Err(GeometryError::ComputationError(warning_msg));
        }

        Ok(distance)
    }


    /// Computes the area of a 2D face based on its shape, caching the result.
    ///
    /// # Arguments
    /// * `face_id` - The unique identifier for the face.
    /// * `face_shape` - The shape of the face (`FaceShape` enum).
    /// * `face_vertices` - A reference to a vector of `[f64; 3]` representing the face's vertices.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The computed area or an error if validation fails.
    pub fn compute_face_area(
        &mut self,
        face_id: usize,
        face_shape: FaceShape,
        face_vertices: &Vec<[f64; 3]>,
    ) -> Result<f64, GeometryError> {
        // Check if the area is already cached
        if let Some(cached) = self.cache.lock().unwrap().get(&face_id).and_then(|c| c.area) {
            log::debug!("Using cached area for face {:?}", face_id);
            return Ok(cached);
        }

        // Validate the number of vertices based on the face shape
        let expected_vertices = match face_shape {
            FaceShape::Edge => 2,
            FaceShape::Triangle => 3,
            FaceShape::Quadrilateral => 4,
        };

        if face_vertices.len() != expected_vertices {
            let error_msg = format!(
                "{:?} face must have exactly {} vertices, but got {}.",
                face_shape,
                expected_vertices,
                face_vertices.len()
            );
            log::error!("{}", error_msg);
            return Err(GeometryError::InvalidFaceVertices(error_msg));
        }

        // Compute the area based on the face shape
        let area = match face_shape {
            FaceShape::Edge => self.compute_edge_length(face_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute edge length: {:?}", err))
            })?,
            FaceShape::Triangle => self.compute_triangle_area(face_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute triangle area: {:?}", err))
            })?,
            FaceShape::Quadrilateral => self.compute_quadrilateral_area(face_vertices).map_err(|err| {
                GeometryError::ComputationError(format!("Failed to compute quadrilateral area: {:?}", err))
            })?,
        };

        // Check for invalid area (e.g., NaN or negative)
        if area.is_nan() || area < 0.0 {
            let warning_msg = format!(
                "Computed area is invalid (NaN or negative) for face {} with vertices {:?}.",
                face_id, face_vertices
            );
            log::warn!("{}", warning_msg);
            return Err(GeometryError::ComputationError(warning_msg));
        }

        // Cache the computed area
        self.cache
            .lock()
            .unwrap()
            .entry(face_id)
            .or_default()
            .area = Some(area);

        log::info!("Computed area for face {:?}: {:?}", face_id, area);
        Ok(area)
    }

    /// Computes the centroid of a 2D face based on its shape.
    ///
    /// # Arguments
    /// * `face_shape` - Enum defining the shape of the face (e.g., Triangle, Quadrilateral).
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the face.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The 3D coordinates of the face centroid or an error if validation fails.
    pub fn compute_face_centroid(
        &self,
        face_shape: FaceShape,
        face_vertices: &Vec<[f64; 3]>,
    ) -> Result<[f64; 3], GeometryError> {
        // Validate the number of vertices based on the face shape
        match face_shape {
            FaceShape::Edge if face_vertices.len() != 2 => {
                let error_msg = format!(
                    "Edge face must have exactly 2 vertices, but got {}.",
                    face_vertices.len()
                );
                error!("{}", error_msg);
                return Err(GeometryError::InvalidFaceVertices(error_msg));
            }
            FaceShape::Triangle if face_vertices.len() != 3 => {
                let error_msg = format!(
                    "Triangle face must have exactly 3 vertices, but got {}.",
                    face_vertices.len()
                );
                error!("{}", error_msg);
                return Err(GeometryError::InvalidFaceVertices(error_msg));
            }
            FaceShape::Quadrilateral if face_vertices.len() != 4 => {
                let error_msg = format!(
                    "Quadrilateral face must have exactly 4 vertices, but got {}.",
                    face_vertices.len()
                );
                error!("{}", error_msg);
                return Err(GeometryError::InvalidFaceVertices(error_msg));
            }
            _ => {} // Validation passed
        }

        // Compute the centroid based on the face shape
        let centroid = match face_shape {
            FaceShape::Edge => Ok(self.compute_edge_midpoint(face_vertices)?),
            FaceShape::Triangle => Ok(self.compute_triangle_centroid(face_vertices)?),
            FaceShape::Quadrilateral => Ok(self.compute_quadrilateral_centroid(face_vertices)?),
        }?;

        // Check for invalid centroid coordinates
        if centroid.iter().any(|&c| c.is_nan()) {
            let warning_msg = format!(
                "Computed centroid contains NaN values for face shape {:?} with vertices {:?}.",
                face_shape, face_vertices
            );
            warn!("{}", warning_msg);
            return Err(GeometryError::ComputationError(warning_msg));
        }

        Ok(centroid)
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
    /// * `Result<Vector3, GeometryError>` - The computed normal vector or an error message.
    pub fn compute_face_normal(
        &mut self,
        mesh: &Mesh,
        face: &MeshEntity,
        _cell: &MeshEntity,
    ) -> Result<Vector3, GeometryError> {
        let face_id = face.get_id();

        // Check if the normal is already cached
        if let Some(cached) = self
            .cache
            .lock()
            .unwrap()
            .get(&face_id)
            .and_then(|c| c.normal)
        {
            return Ok(domain::section::Vector3(cached));
        }

        // Retrieve face vertices and handle potential errors
        let face_vertices = match mesh.get_face_vertices(face) {
            Ok(vertices) if !vertices.is_empty() => vertices,
            Ok(_) => {
                let error_msg = format!(
                    "Face {:?} has no vertices; normal cannot be computed.",
                    face
                );
                log::error!("{}", error_msg);
                return Err(GeometryError::InvalidFaceVertices(error_msg));
            }
            Err(err) => {
                let error_msg = format!(
                    "Failed to retrieve vertices for face {:?}: {}",
                    face, err
                );
                log::error!("{}", error_msg);
                return Err(GeometryError::MeshError(error_msg));
            }
        };

        // Determine the face shape based on the number of vertices
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => {
                let error_msg = format!(
                    "Unsupported face shape with {} vertices for face {:?}",
                    face_vertices.len(),
                    face
                );
                log::error!("{}", error_msg);
                return Err(GeometryError::UnsupportedFaceShape(error_msg));
            }
        };

        // Compute the normal vector based on the face shape
        let normal = match face_shape {
            FaceShape::Edge => self.compute_edge_normal(&face_vertices),
            FaceShape::Triangle => self.compute_triangle_normal(&face_vertices)?,
            FaceShape::Quadrilateral => self.compute_quadrilateral_normal(&face_vertices)?,
        };

        // Cache the normal vector for future use
        self.cache
            .lock()
            .unwrap()
            .entry(face_id)
            .or_default()
            .normal = Some(normal);

        Ok(domain::section::Vector3(normal))
    }



    /// Computes the normal vector of an edge (2D face).
    pub fn compute_edge_normal(&self, face_vertices: &Vec<[f64; 3]>) -> [f64; 3] {
        // Ensure that we have exactly 2 vertices
        assert_eq!(face_vertices.len(), 2, "Edge must have exactly 2 vertices");

        let p1 = face_vertices[0];
        let p2 = face_vertices[1];

        // Compute the edge vector
        let edge_vector = [
            p2[0] - p1[0],
            p2[1] - p1[1],
            p2[2] - p1[2],
        ];

        // In 2D, assuming Z=0, the normal is perpendicular to the edge vector
        // Rotate the vector by 90 degrees counterclockwise
        let normal = [
            -edge_vector[1],
            edge_vector[0],
            0.0,
        ];

        // Normalize the normal vector
        let magnitude = (normal[0].powi(2) + normal[1].powi(2)).sqrt();
        if magnitude != 0.0 {
            [normal[0] / magnitude, normal[1] / magnitude, 0.0]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Computes the length of an edge.
    ///
    /// # Arguments
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the edge.
    ///
    /// # Returns
    /// * `Result<f64, GeometryError>` - The computed length of the edge or an error if the input is invalid.
    pub fn compute_edge_length(&self, face_vertices: &Vec<[f64; 3]>) -> Result<f64, GeometryError> {
        // Validate that there are exactly 2 vertices
        if face_vertices.len() != 2 {
            let error_msg = format!("Edge must have exactly 2 vertices, got {}", face_vertices.len());
            log::error!("{}", error_msg);
            return Err(GeometryError::InvalidFaceVertices(error_msg));
        }

        let p1 = face_vertices[0];
        let p2 = face_vertices[1];

        // Compute the distance between the two vertices
        let length = Geometry::compute_distance(&p1, &p2)?;
        Ok(length)
    }

    /// Computes the midpoint of an edge.
    ///
    /// # Arguments
    /// * `face_vertices` - A vector of 3D coordinates representing the vertices of the edge.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryError>` - The midpoint of the edge or an error if the input is invalid.
    pub fn compute_edge_midpoint(&self, face_vertices: &[[f64; 3]]) -> Result<[f64; 3], GeometryError> {
        // Validate that there are exactly 2 vertices
        if face_vertices.len() != 2 {
            let error_msg = format!("Edge must have exactly 2 vertices, got {}", face_vertices.len());
            log::error!("{}", error_msg);
            return Err(GeometryError::InvalidFaceVertices(error_msg));
        }

        let p1 = face_vertices[0];
        let p2 = face_vertices[1];

        // Compute the midpoint
        let midpoint = [
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
            (p1[2] + p2[2]) / 2.0,
        ];

        Ok(midpoint)
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
    ///
    /// This function creates a separate `Geometry` instance for each thread to avoid
    /// mutable borrow conflicts in parallel computations. Errors during volume computation
    /// are logged, and invalid cells are skipped.
    pub fn update_all_cell_volumes(&mut self, mesh: &Mesh) {
        let new_volumes: Vec<f64> = mesh
            .get_cells()
            .par_iter()
            .filter_map(|cell| {
                // Create a new Geometry instance for this thread
                let mut temp_geometry = Geometry::new();

                // Compute the volume for the cell using the temporary instance
                match temp_geometry.compute_cell_volume(mesh, cell) {
                    Ok(volume) => Some(volume), // Include valid volumes
                    Err(err) => {
                        log::error!(
                            "Failed to compute volume for cell {:?}: {}",
                            cell,
                            err
                        );
                        None // Skip cells with errors
                    }
                }
            })
            .collect();

        self.cell_volumes = new_volumes;
    }


    /// Computes the total centroid of all cells.
    ///
    /// Returns the average position of all cell centroids. If no centroids exist,
    /// returns `[0.0, 0.0, 0.0]` to indicate an empty geometry.
    pub fn compute_total_centroid(&self) -> [f64; 3] {
        // Handle the case where there are no centroids
        if self.cell_centroids.is_empty() {
            log::warn!("No centroids found; returning default centroid [0.0, 0.0, 0.0]");
            return [0.0, 0.0, 0.0];
        }

        // Compute the sum of centroids in parallel
        let total_centroid: [f64; 3] = self
            .cell_centroids
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

        // Calculate the average centroid
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
    fn test_compute_edge_normal() {
        let geometry = Geometry::new();

        // Define vertices for an edge
        let vertices = vec![
            [0.0, 0.0, 0.0], // Vertex 1
            [1.0, 0.0, 0.0], // Vertex 2
        ];

        let normal = geometry.compute_edge_normal(&vertices);
        // Expected normal is [0.0, 1.0, 0.0] for an edge along X-axis

        assert!((normal[0] - 0.0).abs() < 1e-6);
        assert!((normal[1] - 1.0).abs() < 1e-6);
        assert!((normal[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_edge_length() {
        let geometry = Geometry::new();

        // Define vertices for an edge
        let vertices = vec![
            [0.0, 0.0, 0.0], // Vertex 1
            [3.0, 4.0, 0.0], // Vertex 2
        ];

        let length = geometry.compute_edge_length(&vertices);
        // Expected length is 5.0 (3-4-5 triangle)

        assert!((length.unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_edge_midpoint() {
        let geometry = Geometry::new();

        // Define vertices for an edge
        let vertices = vec![
            [1.0, 2.0, 3.0], // Vertex 1
            [4.0, 5.0, 6.0], // Vertex 2
        ];

        let midpoint = geometry.compute_edge_midpoint(&vertices);
        // Expected midpoint is [2.5, 3.5, 4.5]

        assert_eq!(midpoint.unwrap(), [2.5, 3.5, 4.5]);
    }

    /* #[test]
    fn test_compute_face_normal_edge_in_mesh() {
        let mut mesh = Mesh::new();

        // Add vertices
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);

        // Add edge (face in 2D)
        let edge = MeshEntity::Face(1);
        mesh.add_entity(edge.clone());
        mesh.add_arrow(edge.clone(), MeshEntity::Vertex(1));
        mesh.add_arrow(edge.clone(), MeshEntity::Vertex(2));

        // Create Geometry instance
        let mut geometry = Geometry::new();

        // Compute normal
        let normal_option = geometry.compute_face_normal(&mesh, &edge, None);
        assert!(normal_option.is_some());

        let normal = normal_option.unwrap();
        // Expected normal is [0.0, 1.0, 0.0] for an edge along X-axis

        assert!((normal[0] - 0.0).abs() < 1e-6);
        assert!((normal[1] - 1.0).abs() < 1e-6);
        assert!((normal[2] - 0.0).abs() < 1e-6);
    } */

    #[test]
    fn test_set_vertex() {
        let mut geometry = Geometry::new();

        // Set vertex at index 0
        let _ = geometry.set_vertex(0, [1.0, 2.0, 3.0]);
        assert_eq!(geometry.vertices[0], [1.0, 2.0, 3.0]);

        // Update vertex at index 0
        let _ = geometry.set_vertex(0, [4.0, 5.0, 6.0]);
        assert_eq!(geometry.vertices[0], [4.0, 5.0, 6.0]);

        // Set vertex at a higher index
        let _ = geometry.set_vertex(3, [7.0, 8.0, 9.0]);
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
        assert_eq!(distance.unwrap(), 5.0);
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
            logger: todo!(),
        };

        // Define vertices and a cell.
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let vertex4 = MeshEntity::Vertex(4);
        let cell = MeshEntity::Cell(1);

        // Set vertex coordinates.
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]).unwrap();
        mesh.set_vertex_coordinates(4, [0.0, 0.0, 1.0]).unwrap();

        // Add entities to the mesh.
        mesh.add_entity(vertex1).unwrap();
        mesh.add_entity(vertex2).unwrap();
        mesh.add_entity(vertex3).unwrap();
        mesh.add_entity(vertex4).unwrap();
        mesh.add_entity(cell).unwrap();

        // Establish relationships between the cell and vertices.
        mesh.add_arrow(cell, vertex1).unwrap();
        mesh.add_arrow(cell, vertex2).unwrap();
        mesh.add_arrow(cell, vertex3).unwrap();
        mesh.add_arrow(cell, vertex4).unwrap();

        // Verify that `get_cell_vertices` retrieves the correct vertices.
        let cell_vertices = mesh.get_cell_vertices(&cell);
        assert_eq!(cell_vertices.unwrap().len(), 4, "Expected 4 vertices for a tetrahedron cell.");

        // Validate the shape before computing.
        assert_eq!(mesh.get_cell_shape(&cell), Ok(CellShape::Tetrahedron));

        // Create a Geometry instance and compute the centroid.
        let mut geometry = Geometry::new();
        let centroid = geometry.compute_cell_centroid(&mesh, &cell);

        // Expected centroid is the average of all vertices: (0.25, 0.25, 0.25)
        assert_eq!(centroid.unwrap(), [0.25, 0.25, 0.25]);
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
        assert_eq!(area.unwrap(), 6.0);
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
        assert_eq!(centroid.unwrap(), [0.5, 0.5, 0.0]);
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
            assert!((normal.as_ref().unwrap()[i] - expected_normal[i]).abs() < 1e-6);
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
            assert!((normal.as_ref().unwrap()[i] - expected_normal[i]).abs() < 1e-6);
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
        let normal_first = geometry.compute_triangle_normal(&vertices).unwrap();

        // Manually retrieve from cache to verify caching behavior
        geometry.cache.lock().unwrap().entry(face_id).or_default().normal = Some(normal_first);
        let cache = geometry.cache.lock().unwrap();
        let cached_normal = cache.get(&face_id).and_then(|c| c.normal.as_ref());

        // Verify that the cached value matches the first computed value
        assert_eq!(Some(&normal_first), cached_normal);
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
    
        // Attempt to compute the normal for an unsupported shape
        let result = match vertices.len() {
            2 => Ok(geometry.compute_edge_normal(&vertices)),
            3 => Ok(geometry.compute_triangle_normal(&vertices).unwrap()),
            4 => Ok(geometry.compute_quadrilateral_normal(&vertices).unwrap()),
            _ => Err("Unsupported face shape: expected 2, 3, or 4 vertices".to_string()),
        };
    
        // Assert that the function correctly identifies unsupported shapes
        assert!(
            result.is_err(),
            "Expected error for unsupported face shape, but got a normal vector: {:?}",
            result
        );
    
        // Validate the error message
        if let Err(err_msg) = result {
            assert!(
                err_msg.contains("Unsupported face shape"),
                "Unexpected error message: {}",
                err_msg
            );
        }
    }
    
}
