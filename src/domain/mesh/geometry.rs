use super::Mesh;
use crate::domain;
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
    /// * `Result<DashMap<MeshEntity, ()>, String>` - A map of face entities connected to the cell,
    ///   or an error message if the operation fails.
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Result<DashMap<MeshEntity, ()>, String> {
        // Attempt to retrieve the cone (connected entities) for the cell.
        let cone_result = self.sieve.cone(cell);
        match cone_result {
            Ok(connected_entities) => {
                let faces = DashMap::new();

                // Filter for face entities and insert them into the `faces` map.
                connected_entities
                    .into_iter()
                    .filter(|entity| matches!(entity, MeshEntity::Face(_)))
                    .for_each(|face| {
                        faces.insert(face, ());
                    });

                if faces.is_empty() {
                    Err(format!(
                        "No faces found for cell {:?}. This may indicate an invalid topology.",
                        cell
                    ))
                } else {
                    Ok(faces)
                }
            }
            Err(err) => {
                // Log and return the error if the cone retrieval fails.
                Err(format!(
                    "Failed to retrieve connected entities for cell {:?}: {}",
                    cell, err
                ))
            }
        }
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
    /// * `Result<DashMap<MeshEntity, ()>, String>` - A map of cell entities sharing the face,
    ///   or an error message if the operation fails.
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> Result<DashMap<MeshEntity, ()>, String> {
        // Attempt to retrieve supporting entities for the face.
        match self.sieve.support(face) {
            Ok(supporting_entities) => {
                let cells = DashMap::new();
                let entities = self.entities.read().map_err(|_| {
                    "Failed to acquire read lock on entities during cell sharing computation.".to_string()
                })?;

                // Filter for valid cell entities and insert them into the `cells` map.
                supporting_entities
                    .into_iter()
                    .filter(|entity| matches!(entity, MeshEntity::Cell(_)) && entities.contains(entity))
                    .for_each(|cell| {
                        cells.insert(cell, ());
                    });

                if cells.is_empty() {
                    Err(format!(
                        "No cells found sharing face {:?}. This may indicate an invalid topology.",
                        face
                    ))
                } else {
                    Ok(cells)
                }
            }
            Err(err) => {
                // Log and return the error if the support retrieval fails.
                Err(format!(
                    "Failed to retrieve supporting entities for face {:?}: {}",
                    face, err
                ))
            }
        }
    }


    /// Computes the Euclidean distance between two cells based on their centroids.
    ///
    /// # Arguments
    /// * `cell_i` - The first cell entity.
    /// * `cell_j` - The second cell entity.
    ///
    /// # Returns
    /// * `Result<f64, String>` - The computed distance between the centroids of the two cells,
    ///   or an error message if the centroids cannot be computed.
    pub fn get_distance_between_cells(
        &self,
        cell_i: &MeshEntity,
        cell_j: &MeshEntity,
    ) -> Result<f64, String> {
        // Compute the centroids of the two cells.
        let centroid_i = self
            .get_cell_centroid(cell_i)
            .map_err(|err| format!("Failed to compute centroid for cell {:?}: {}", cell_i, err))?;
        let centroid_j = self
            .get_cell_centroid(cell_j)
            .map_err(|err| format!("Failed to compute centroid for cell {:?}: {}", cell_j, err))?;

        // Compute the distance using the Geometry module.
        Ok(Geometry::compute_distance(&centroid_i, &centroid_j))
    }


    /// Computes the area of a face based on its geometric shape and vertices.
    ///
    /// # Arguments
    /// * `face` - The face entity for which to compute the area.
    ///
    /// # Returns
    /// * `Result<f64, String>` - The area of the face, or an error message if the face shape is unsupported.
    pub fn get_face_area(&self, face: &MeshEntity) -> Result<f64, String> {
        // Retrieve the vertices of the face
        let face_vertices = self.get_face_vertices(face)?;

        // Determine the shape of the face based on the number of vertices
        let face_shape = match face_vertices.len() {
            2 => FaceShape::Edge,
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => {
                return Err(format!(
                    "Unsupported face shape with {} vertices. Expected 2, 3, or 4 vertices.",
                    face_vertices.len()
                ));
            }
        };

        // Compute the area using the geometry utility
        let mut geometry = Geometry::new();
        let face_id = face.get_id();

        Ok(geometry.compute_face_area(face_id, face_shape, &face_vertices))
    }

    /// Computes the centroid of a cell based on its vertices.
    ///
    /// # Arguments
    /// * `cell` - The cell entity for which to compute the centroid.
    ///
    /// # Returns
    /// * `Result<[f64; 3], String>` - The 3D coordinates of the cell's centroid or an error message.
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> Result<[f64; 3], String> {
        // Retrieve the vertices of the cell.
        let cell_vertices = self.get_cell_vertices(cell).map_err(|err| {
            format!(
                "Failed to retrieve vertices for cell {:?}: {}",
                cell, err
            )
        })?;

        // Determine the shape of the cell based on the number of vertices.
        match cell_vertices.len() {
            4 | 5 | 6 | 8 => {
                // Compute the centroid using the geometry module.
                let mut geometry = Geometry::new();
                let centroid = geometry.compute_cell_centroid(self, cell);

                Ok(centroid) // Return the computed centroid directly
            }
            _ => Err(format!(
                "Unsupported cell shape with {} vertices for cell {:?}",
                cell_vertices.len(),
                cell
            )),
        }
    }

    /// Retrieves all vertices connected to a given vertex via shared cells.
    ///
    /// # Arguments
    /// * `vertex` - The vertex entity for which to find neighboring vertices.
    ///
    /// # Returns
    /// * `Result<Vec<MeshEntity>, String>` - A list of neighboring vertex entities,
    ///   or an error message if the operation fails.
    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Result<Vec<MeshEntity>, String> {
        // Ensure the input entity is a vertex.
        if !matches!(vertex, MeshEntity::Vertex(_)) {
            return Err(format!(
                "Invalid input: {:?} is not a vertex entity.",
                vertex
            ));
        }

        // Retrieve cells connected to the given vertex.
        let connected_cells = self
            .sieve
            .support(vertex)
            .map_err(|err| format!("Failed to retrieve supporting cells for vertex {:?}: {}", vertex, err))?;

        let neighbors = DashMap::new();

        // Iterate over connected cells and find neighboring vertices.
        for cell in connected_cells {
            match self.sieve.cone(&cell) {
                Ok(cell_vertices) => {
                    for v in cell_vertices {
                        if v != *vertex && matches!(v, MeshEntity::Vertex(_)) {
                            neighbors.insert(v.clone(), ());
                        }
                    }
                }
                Err(err) => {
                    return Err(format!(
                        "Failed to retrieve vertices for cell {:?} connected to vertex {:?}: {}",
                        cell, vertex, err
                    ));
                }
            }
        }

        // Collect and return neighboring vertices.
        if neighbors.is_empty() {
            Err(format!(
                "No neighboring vertices found for vertex {:?}.",
                vertex
            ))
        } else {
            Ok(neighbors.into_iter().map(|(vertex, _)| vertex).collect())
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
