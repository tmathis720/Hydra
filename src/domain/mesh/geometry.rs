use super::geometry_validation::GeometryValidationError;
use super::Mesh;
use crate::domain;
use crate::domain::mesh::MeshError;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;
use crate::domain::section::Vector3;

impl Mesh {
    /// Retrieves all the faces of a given cell.
    ///
    /// This function collects entities connected to the provided cell and filters
    /// them to include only face entities. The result is returned as a `Result`.
    ///
    /// # Arguments
    /// * `cell` - A `MeshEntity` representing the cell whose faces are being retrieved.
    ///
    /// # Returns
    /// * `Result<DashMap<MeshEntity, ()>, GeometryValidationError>` - A map of face entities connected to the cell,
    ///   or a `GeometryValidationError` if the operation fails.
    pub fn get_faces_of_cell(
        &self,
        cell: &MeshEntity,
    ) -> Result<DashMap<MeshEntity, ()>, GeometryValidationError> {
        // Attempt to retrieve the cone (connected entities) for the cell.
        let connected_entities = self.sieve.cone(cell).map_err(|err| {
            log::error!(
                "Failed to retrieve connected entities for cell {:?}: {}",
                cell,
                err
            );
            GeometryValidationError::DistanceCalculationError(*cell, *cell, err.to_string())
        })?;

        let faces = DashMap::new();

        // Filter for face entities and insert them into the `faces` map.
        connected_entities
            .into_iter()
            .filter(|entity| matches!(entity, MeshEntity::Face(_)))
            .for_each(|face| {
                faces.insert(face, ());
            });

        if faces.is_empty() {
            log::warn!(
                "No faces found for cell {:?}. This may indicate an invalid topology.",
                cell
            );
            return Err(GeometryValidationError::MissingVertexCoordinates(cell.get_id() as u64));
        }

        Ok(faces)
    }


    /// Retrieves all the cells that share a given face.
    ///
    /// This function identifies all cell entities that share a specified face,
    /// filtering only valid cell entities present in the mesh.
    ///
    /// # Arguments
    /// * `face` - A `MeshEntity` representing the face.
    ///
    /// # Returns
    /// * `Result<DashMap<MeshEntity, ()>, GeometryValidationError>` - A map of cell entities sharing the face,
    ///   or a `GeometryValidationError` if the operation fails.
    pub fn get_cells_sharing_face(
        &self,
        face: &MeshEntity,
    ) -> Result<DashMap<MeshEntity, ()>, GeometryValidationError> {
        // Attempt to retrieve supporting entities for the face.
        let supporting_entities = self.sieve.support(face).map_err(|err| {
            log::error!(
                "Failed to retrieve supporting entities for face {:?}: {}",
                face,
                err
            );
            GeometryValidationError::TopologyError(format!(
                "Failed to retrieve supporting entities for face {:?}: {}",
                face, err
            ))
        })?;

        let cells = DashMap::new();

        let entities = self.entities.read().map_err(|_| {
            let message = format!(
                "Failed to acquire read lock on entities during cell sharing computation for face {:?}.",
                face
            );
            log::error!("{}", message);
            GeometryValidationError::EntityAccessError(message)
        })?;

        // Filter for valid cell entities and insert them into the `cells` map.
        supporting_entities
            .into_iter()
            .filter(|entity| matches!(entity, MeshEntity::Cell(_)) && entities.contains(entity))
            .for_each(|cell| {
                cells.insert(cell, ());
            });

        if cells.is_empty() {
            let message = format!(
                "No cells found sharing face {:?}. This may indicate an invalid topology.",
                face
            );
            log::warn!("{}", message);
            return Err(GeometryValidationError::TopologyError(message));
        }

        Ok(cells)
    }


    /// Computes the Euclidean distance between two cells based on their centroids.
    ///
    /// # Arguments
    /// * `cell_i` - The first cell entity.
    /// * `cell_j` - The second cell entity.
    ///
    /// # Returns
    /// * `Result<f64, GeometryValidationError>` - The computed distance between the centroids of the two cells,
    ///   or a `GeometryValidationError` if the centroids cannot be computed.
    pub fn get_distance_between_cells(
        &self,
        cell_i: &MeshEntity,
        cell_j: &MeshEntity,
    ) -> Result<f64, GeometryValidationError> {
        // Compute the centroid for the first cell.
        let centroid_i = self.get_cell_centroid(cell_i).map_err(|err| {
            log::error!(
                "Failed to compute centroid for cell {:?}: {}",
                cell_i,
                err
            );
            GeometryValidationError::CentroidError(format!(
                "Failed to compute centroid for cell {:?}: {}",
                cell_i, err
            ))
        })?;

        // Compute the centroid for the second cell.
        let centroid_j = self.get_cell_centroid(cell_j).map_err(|err| {
            log::error!(
                "Failed to compute centroid for cell {:?}: {}",
                cell_j,
                err
            );
            GeometryValidationError::CentroidError(format!(
                "Failed to compute centroid for cell {:?}: {}",
                cell_j, err
            ))
        })?;

        // Compute the Euclidean distance using the `Geometry` module.
        let distance = Geometry::compute_distance(&centroid_i, &centroid_j);
        log::info!(
            "Computed distance between cells {:?} and {:?}: {:.6}",
            cell_i,
            cell_j,
            distance
        );

        Ok(distance)
    }


    /// Computes the area of a face based on its geometric shape and vertices.
    ///
    /// # Arguments
    /// * `face` - The face entity for which to compute the area.
    ///
    /// # Returns
    /// * `Result<f64, GeometryValidationError>` - The area of the face, or a `GeometryValidationError`
    ///   if the face shape is unsupported or vertices cannot be retrieved.
    pub fn get_face_area(&self, face: &MeshEntity) -> Result<f64, GeometryValidationError> {
        // Retrieve the vertices of the face
        let face_vertices = self.get_face_vertices(face).map_err(|err| {
            log::error!(
                "Failed to retrieve vertices for face {:?}: {}",
                face,
                err
            );
            GeometryValidationError::VertexError(format!(
                "Failed to retrieve vertices for face {:?}: {}",
                face, err
            ))
        })?;

        // Determine the shape of the face based on the number of vertices
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => {
                let error_message = format!(
                    "Unsupported face shape with {} vertices. Expected 2, 3, or 4 vertices.",
                    face_vertices.len()
                );
                log::error!("Unsupported face shape: {:?}", error_message);
                return Err(GeometryValidationError::ShapeError(error_message));
            }
        };

        // Compute the area using the geometry utility
        let mut geometry = Geometry::new();
        let face_id = face.get_id();
        let area = geometry
            .compute_face_area(face_id, face_shape, &face_vertices);

        if area.is_nan() {
            let err_msg = format!(
                "Failed to compute area for face {:?} with shape {:?}: Computation resulted in NaN",
                face, face_shape
            );
            log::error!("{}", err_msg);
            return Err(GeometryValidationError::ComputationError(err_msg));
        }

        log::info!(
            "Computed area for face {:?} with shape {:?}: {:.6}",
            face,
            face_shape,
            area
        );

        Ok(area)
    }

    /// Computes the centroid of a cell based on its vertices.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which to compute the centroid.
    ///
    /// # Returns
    /// * `Result<[f64; 3], GeometryValidationError>` - The 3D coordinates of the cell's centroid or a structured error.
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> Result<[f64; 3], GeometryValidationError> {
        // Retrieve the vertices of the cell
        let cell_vertices = self.get_cell_vertices(cell).map_err(|err| {
            log::error!("Failed to retrieve vertices for cell {:?}: {}", cell, err);
            GeometryValidationError::VertexError(format!(
                "Failed to retrieve vertices for cell {:?}: {}",
                cell, err
            ))
        })?;

        // Validate the number of vertices for supported cell shapes
        if !matches!(cell_vertices.len(), 4 | 5 | 6 | 8) {
            let error_message = format!(
                "Unsupported cell shape with {} vertices for cell {:?}. Expected 4, 5, 6, or 8 vertices.",
                cell_vertices.len(),
                cell
            );
            log::error!("{}", error_message);
            return Err(GeometryValidationError::ShapeError(error_message));
        }

        // Compute the centroid using the Geometry module
        let mut geometry = Geometry::new();
        let centroid = geometry.compute_cell_centroid(self, cell); // Directly retrieve the centroid

        log::info!(
            "Computed centroid for cell {:?}: {:?}",
            cell,
            centroid
        );

        Ok(centroid) // Return the centroid directly
    }

    /// Retrieves all vertices connected to a given vertex via shared cells.
    ///
    /// # Arguments
    /// * `vertex` - The vertex entity for which to find neighboring vertices.
    ///
    /// # Returns
    /// * `Result<Vec<MeshEntity>, MeshError>` - A list of neighboring vertex entities,
    ///   or a structured error if the operation fails.
    pub fn get_neighboring_vertices(
        &self,
        vertex: &MeshEntity,
    ) -> Result<Vec<MeshEntity>, MeshError> {
        // Validate that the input entity is a vertex.
        if !matches!(vertex, MeshEntity::Vertex(_)) {
            let error_message = format!("Invalid input: {:?} is not a vertex entity.", vertex);
            log::error!("{}", error_message);
            return Err(MeshError::InvalidEntityType(error_message));
        }

        // Retrieve cells connected to the given vertex.
        let connected_cells = self.sieve.support(vertex).map_err(|err| {
            let error_message = format!(
                "Failed to retrieve supporting cells for vertex {:?}: {}",
                vertex, err
            );
            log::error!("{}", error_message);
            MeshError::ConnectivityQueryError(error_message)
        })?;

        let neighbors = DashMap::new();

        // Iterate over connected cells to find neighboring vertices.
        for cell in connected_cells {
            match self.sieve.cone(&cell) {
                Ok(cell_vertices) => {
                    for v in cell_vertices {
                        // Only add vertices other than the input vertex itself.
                        if v != *vertex && matches!(v, MeshEntity::Vertex(_)) {
                            neighbors.insert(v.clone(), ());
                        }
                    }
                }
                Err(err) => {
                    let error_message = format!(
                        "Failed to retrieve vertices for cell {:?} connected to vertex {:?}: {}",
                        cell, vertex, err
                    );
                    log::error!("{}", error_message);
                    return Err(MeshError::ConnectivityQueryError(error_message));
                }
            }
        }

        // Collect and return neighboring vertices.
        if neighbors.is_empty() {
            let warning_message = format!(
                "No neighboring vertices found for vertex {:?}.",
                vertex
            );
            log::warn!("{}", warning_message);
            Err(MeshError::NoNeighborsError(warning_message))
        } else {
            let neighbors_vec: Vec<MeshEntity> =
                neighbors.into_iter().map(|(vertex, _)| vertex).collect();
            log::info!(
                "Found {} neighboring vertices for vertex {:?}.",
                neighbors_vec.len(),
                vertex
            );
            Ok(neighbors_vec)
        }
    }


    /// Returns an iterator over all vertex IDs in the mesh.
    ///
    /// # Returns
    /// * `impl Iterator<Item = &usize>` - An iterator over vertex IDs.
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on its vertex count.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which to determine the shape.
    ///
    /// # Returns
    /// * `Result<CellShape, String>` - The determined cell shape or an error message if unsupported.
    pub fn get_cell_shape(&self, cell: &MeshEntity) -> Result<CellShape, String> {
        let cell_vertices = self.get_cell_vertices(cell)?;

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

    /// Retrieves the vertices of a cell, sorted by vertex ID.
    ///
    /// # Arguments
    /// * `cell` - The cell entity whose vertices are being retrieved.
    ///
    /// # Returns
    /// * `Result<Vec<[f64; 3]>, String>` - The 3D coordinates of the cell's vertices, sorted by ID,
    ///   or an error message if the operation fails.
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Result<Vec<[f64; 3]>, String> {
        // Ensure the input is a `Cell`.
        if !matches!(cell, MeshEntity::Cell(_)) {
            return Err(format!(
                "Invalid input: {:?} is not a cell entity.",
                cell
            ));
        }

        // Retrieve entities connected to the cell.
        let connected_entities = self
            .sieve
            .cone(cell)
            .map_err(|err| format!("Failed to retrieve connected entities for cell {:?}: {}", cell, err))?;

        let mut vertex_ids_and_coords = Vec::new();

        for entity in connected_entities {
            if let MeshEntity::Vertex(vertex_id) = entity {
                match self.get_vertex_coordinates(vertex_id) {
                    Some(coords) => {
                        vertex_ids_and_coords.push((vertex_id, coords));
                    }
                    None => {
                        return Err(format!(
                            "Failed to retrieve coordinates for vertex {:?} connected to cell {:?}.",
                            vertex_id, cell
                        ));
                    }
                }
            }
        }

        // Sort vertices by their IDs.
        vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);

        // Extract and return the coordinates of the vertices.
        Ok(vertex_ids_and_coords
            .into_iter()
            .map(|(_, coords)| coords)
            .collect())
    }

    /// Retrieves the vertices of a face, sorted by vertex ID.
    ///
    /// # Arguments
    /// * `face` - The face entity whose vertices are being retrieved.
    ///
    /// # Returns
    /// * `Result<Vec<[f64; 3]>, String>` - The 3D coordinates of the face's vertices, sorted by ID, or an error message.
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Result<Vec<[f64; 3]>, String> {
        let mut vertex_ids_and_coords = Vec::new();

        // Handle the `Result` returned by `self.sieve.cone(face)`
        let connected_vertices = self.sieve.cone(face)?;
        for vertex in connected_vertices {
            if let MeshEntity::Vertex(vertex_id) = vertex {
                // Retrieve vertex coordinates and handle missing cases
                if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                    vertex_ids_and_coords.push((vertex_id, coords));
                } else {
                    return Err(format!("Vertex ID {} has no associated coordinates.", vertex_id));
                }
            }
        }

        // Sort vertices by their ID
        vertex_ids_and_coords.sort_by_key(|&(vertex_id, _)| vertex_id);

        // Collect and return sorted coordinates
        Ok(vertex_ids_and_coords.into_iter().map(|(_, coords)| coords).collect())
    }


    /// Computes the outward normal vector for a face based on its shape and vertices.
    ///
    /// Optionally adjusts the normal's orientation based on a reference cell's centroid.
    ///
    /// # Arguments
    /// * `face` - The face entity for which to compute the normal.
    /// * `reference_cell` - An optional reference cell entity to adjust the orientation.
    ///
    /// # Returns
    /// * `Result<Vector3, String>` - The computed normal vector or an error message.
    pub fn get_face_normal(
        &self,
        face: &MeshEntity,
        reference_cell: Option<&MeshEntity>,
    ) -> Result<Vector3, String> {
        // Retrieve vertices of the face
        let face_vertices = self.get_face_vertices(face).map_err(|err| {
            format!(
                "Failed to retrieve vertices for face {:?}: {}",
                face, err
            )
        })?;

        // Determine the shape of the face based on the number of vertices
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => {
                return Err(format!(
                    "Unsupported face shape with {} vertices for face {:?}",
                    face_vertices.len(),
                    face
                ));
            }
        };

        // Compute the normal vector based on the face shape
        let geometry = Geometry::new();
        let normal = match face_shape {
            FaceShape::Edge => geometry.compute_edge_normal(&face_vertices),
            FaceShape::Triangle => geometry.compute_triangle_normal(&face_vertices),
            FaceShape::Quadrilateral => geometry.compute_quadrilateral_normal(&face_vertices),
        };

        // Adjust the normal orientation if a reference cell is provided
        if let Some(cell) = reference_cell {
            let cell_centroid = self.get_cell_centroid(cell).map_err(|err| {
                format!(
                    "Failed to retrieve centroid for reference cell {:?}: {}",
                    cell, err
                )
            })?;

            let face_centroid = geometry.compute_face_centroid(face_shape, &face_vertices);

            let to_cell_vector = [
                cell_centroid[0] - face_centroid[0],
                cell_centroid[1] - face_centroid[1],
                cell_centroid[2] - face_centroid[2],
            ];

            let dot_product = normal[0] * to_cell_vector[0]
                + normal[1] * to_cell_vector[1]
                + normal[2] * to_cell_vector[2];

            if dot_product < 0.0 {
                // Reverse the normal if it points inward
                return Ok(domain::section::Vector3([-normal[0], -normal[1], -normal[2]]));
            }
        }

        Ok(domain::section::Vector3(normal))
    }
}
