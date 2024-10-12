use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use rustc_hash::FxHashSet;

impl Mesh {
    /// Retrieves all the faces of a given cell.  
    ///
    /// This method uses the `cone` function of the sieve to obtain all the faces  
    /// connected to the given cell.  
    ///
    /// Returns a set of `MeshEntity` representing the faces of the cell, or  
    /// `None` if the cell has no connected faces.  
    ///
    /// Example usage:
    /// 
    ///    let faces = mesh.get_faces_of_cell(&cell);  
    ///
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<FxHashSet<MeshEntity>> {
        self.sieve.cone(cell).map(|set| set.clone())
    }

    /// Retrieves all the cells that share the given face.  
    ///
    /// This method uses the `support` function of the sieve to obtain all the cells  
    /// that are connected to the given face.  
    ///
    /// Returns a set of `MeshEntity` representing the neighboring cells.  
    ///
    /// Example usage:
    /// 
    ///    let cells = mesh.get_cells_sharing_face(&face);  
    ///
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> FxHashSet<MeshEntity> {
        self.sieve.support(face)
    }

    /// Computes the Euclidean distance between two cells based on their centroids.  
    ///
    /// This method calculates the centroids of both cells and then uses the `Geometry`  
    /// module to compute the distance between these centroids.  
    ///
    /// Example usage:
    /// 
    ///    let distance = mesh.get_distance_between_cells(&cell1, &cell2);  
    ///
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.  
    ///
    /// This method determines the face shape (triangle or quadrilateral) and  
    /// uses the `Geometry` module to compute the area.  
    ///
    /// Panics if the face has an unsupported number of vertices.  
    ///
    /// Example usage:
    /// 
    ///    let area = mesh.get_face_area(&face);  
    ///
    pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };

        // Use a new or existing geometry instance, providing a unique ID for caching.
        let mut geometry = Geometry::new();
        let face_id = face.id(); // Assuming `id` method provides a unique ID for the face.
        geometry.compute_face_area(face_id, face_shape, &face_vertices)
    }

    /// Computes the centroid of a cell based on its vertices.  
    ///
    /// This method determines the cell shape (tetrahedron, pyramid, prism, or hexahedron)  
    /// and uses the `Geometry` module to compute the centroid.  
    ///
    /// Panics if the cell has an unsupported number of vertices.  
    ///
    /// Example usage:
    /// 
    ///    let centroid = mesh.get_cell_centroid(&cell);  
    ///
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let _cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        // Use a new or existing Geometry instance, providing a unique ID for caching.
        let mut geometry = Geometry::new();
        geometry.compute_cell_centroid(self, cell)
    }

    /// Retrieves all vertices connected to the given vertex by shared cells.  
    ///
    /// This method uses the `support` function of the sieve to find cells that  
    /// contain the given vertex and then retrieves all other vertices in those cells.  
    ///
    /// Example usage:
    /// 
    ///    let neighbors = mesh.get_neighboring_vertices(&vertex);  
    ///
    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let mut neighbors = FxHashSet::default();
        let connected_cells = self.sieve.support(vertex);

        for cell in &connected_cells {
            if let Some(cell_vertices) = self.sieve.cone(cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(*v);
                    }
                }
            } else {
                panic!("Cell {:?} has no connected vertices", cell);
            }
        }
        neighbors.into_iter().collect()
    }

    /// Returns an iterator over the IDs of all vertices in the mesh.  
    ///
    /// Example usage:
    /// 
    ///    for vertex_id in mesh.iter_vertices() {  
    ///        println!("Vertex ID: {}", vertex_id);  
    ///    }  
    ///
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on the number of vertices it has.  
    ///
    /// This method supports tetrahedrons, pyramids, prisms, and hexahedrons.  
    /// Returns an error if the cell has an unsupported number of vertices.  
    ///
    /// Example usage:
    /// 
    ///    let cell_shape = mesh.get_cell_shape(&cell).unwrap();  
    ///
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

    /// Retrieves the vertices of a cell and their coordinates.  
    ///
    /// This method uses the sieve to find all vertices connected to the cell,  
    /// and retrieves their 3D coordinates from the `vertex_coordinates` map.  
    ///
    /// Panics if the coordinates for a vertex are not found.  
    ///
    /// Example usage:
    /// 
    ///    let vertices = mesh.get_cell_vertices(&cell);  
    ///
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
    
        // Retrieve entities connected to the cell.
        if let Some(connected_entities) = self.sieve.cone(cell) {
            for entity in connected_entities {
                // Check if the entity is a vertex and extract its coordinates.
                if let MeshEntity::Vertex(vertex_id) = entity {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    } else {
                        panic!("Coordinates for vertex {} not found", vertex_id);
                    }
                }
            }
        }
    
        // Sort and deduplicate vertices to ensure unique entries.
        vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vertices.dedup();
        vertices
    }

    /// Retrieves the vertices of a face and their coordinates.  
    ///
    /// This method uses the sieve to find all vertices connected to the face,  
    /// and retrieves their 3D coordinates from the `vertex_coordinates` map.  
    ///
    /// Example usage:
    /// 
    ///    let face_vertices = mesh.get_face_vertices(&face);  
    ///
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    }
                }
            }
        }
        vertices
    }
}
