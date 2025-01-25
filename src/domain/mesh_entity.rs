/// Represents an entity in a mesh, such as a vertex, edge, face, or cell, 
/// using a unique identifier for each entity type.
/// 
/// The `MeshEntity` enum defines four types of mesh entities:
/// - `Vertex`: Represents a point in the mesh.
/// - `Edge`: Represents a connection between two vertices.
/// - `Face`: Represents a polygonal area bounded by edges.
/// - `Cell`: Represents a volumetric region of the mesh.
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Clone, Copy)]
pub enum MeshEntity {
    Vertex(usize),  // Identifier for a vertex
    Edge(usize),    // Identifier for an edge
    Face(usize),    // Identifier for a face
    Cell(usize),    // Identifier for a cell
}

impl MeshEntity {
    /// Retrieves the unique identifier for the `MeshEntity`.
    ///
    /// Matches the entity type (Vertex, Edge, Face, or Cell) and returns its id.
    /// This function is often used to distinguish between specific instances of entities.
    pub fn get_id(&self) -> usize {
        match *self {
            MeshEntity::Vertex(id) => id,
            MeshEntity::Edge(id) => id,
            MeshEntity::Face(id) => id,
            MeshEntity::Cell(id) => id,
        }
    }

    /// Retrieves the entity type as a string (`"Vertex"`, `"Edge"`, `"Face"`, or `"Cell"`).
    ///
    /// This is useful for logging, debugging, or general introspection of the entity type.
    pub fn get_entity_type(&self) -> &str {
        match *self {
            MeshEntity::Vertex(_) => "Vertex",
            MeshEntity::Edge(_) => "Edge",
            MeshEntity::Face(_) => "Face",
            MeshEntity::Cell(_) => "Cell",
        }
    }

    /// Creates a new `MeshEntity` with a specified identifier.
    ///
    /// # Arguments
    /// - `new_id`: The new identifier for the entity. Must be non-zero.
    ///
    /// # Returns
    /// - `Ok(MeshEntity)`: If the `new_id` is valid.
    /// - `Err(String)`: If the `new_id` is invalid.
    pub fn with_id(&self, new_id: usize) -> Result<Self, String> {
        if new_id == 0 {
            return Err(format!(
                "Invalid ID: {}. Entity IDs must be non-zero.",
                new_id
            ));
        }
        Ok(match *self {
            MeshEntity::Vertex(_) => MeshEntity::Vertex(new_id),
            MeshEntity::Edge(_) => MeshEntity::Edge(new_id),
            MeshEntity::Face(_) => MeshEntity::Face(new_id),
            MeshEntity::Cell(_) => MeshEntity::Cell(new_id),
        })
    }
}

/// A struct representing a directed relationship between two `MeshEntity` elements,
/// referred to as an `Arrow`. It stores a "from" entity and a "to" entity, signifying
/// a directed connection or dependency between the two.
pub struct Arrow {
    pub from: MeshEntity,  // Source entity of the relationship
    pub to: MeshEntity,    // Target entity of the relationship
}

impl Arrow {
    /// Creates a new `Arrow` between two `MeshEntity` instances.
    ///
    /// # Arguments
    /// - `from`: The starting point of the arrow.
    /// - `to`: The ending point of the arrow.
    ///
    /// # Returns
    /// - `Ok(Arrow)`: If `from` and `to` are valid and not the same.
    /// - `Err(String)`: If `from` and `to` are identical.
    pub fn new(from: MeshEntity, to: MeshEntity) -> Result<Self, String> {
        if from == to {
            return Err(format!(
                "Invalid Arrow: 'from' and 'to' entities cannot be the same: {:?}",
                from
            ));
        }
        Ok(Arrow { from, to })
    }

    /// Converts any type implementing `Into<MeshEntity>` into a `MeshEntity`.
    ///
    /// This function is a utility for constructing mesh entities from other data types
    /// that can be converted into `MeshEntity`. It simplifies entity creation in contexts
    /// where generic or derived data types are used.
    pub fn add_entity<T: Into<MeshEntity>>(entity: T) -> MeshEntity {
        entity.into()
    }

    /// Returns a reference to the `from` and `to` entities of the `Arrow`.
    ///
    /// This function is commonly used to inspect the relationship stored in the arrow.
    pub fn get_relation(&self) -> (&MeshEntity, &MeshEntity) {
        (&self.from, &self.to)
    }

    /// Updates the `from` entity in the `Arrow` with a new `MeshEntity`.
    ///
    /// This allows modifying the source of the arrow after its creation.
    pub fn set_from(&mut self, from: MeshEntity) {
        self.from = from;
    }

    /// Updates the `to` entity in the `Arrow` with a new `MeshEntity`.
    ///
    /// This allows modifying the destination of the arrow after its creation.
    pub fn set_to(&mut self, to: MeshEntity) {
        self.to = to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the correct id and type are returned for various `MeshEntity` instances.
    #[test]
    fn test_entity_id_and_type() {
        let vertex = MeshEntity::Vertex(1);
        assert_eq!(vertex.get_id(), 1); // ID check
        assert_eq!(vertex.get_entity_type(), "Vertex"); // Type check
    }

    /// Verifies the creation of an `Arrow` and the retrieval of its relationship.
    #[test]
    fn test_arrow_creation_and_relation() {
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(2);

        // Create a new arrow between a vertex and an edge
        let arrow = Arrow::new(vertex, edge).unwrap();

        // Retrieve and verify the relationship
        let (from, to) = arrow.get_relation();
        assert_eq!(*from, MeshEntity::Vertex(1));
        assert_eq!(*to, MeshEntity::Edge(2));
    }

    /// Verifies the addition of a new entity using the `add_entity` utility function.
    #[test]
    fn test_add_entity() {
        let vertex = MeshEntity::Vertex(5);

        // Convert into a `MeshEntity` using the utility function
        let added_entity = Arrow::add_entity(vertex);

        assert_eq!(added_entity.get_id(), 5); // ID check
        assert_eq!(added_entity.get_entity_type(), "Vertex"); // Type check
    }

    /// Verifies that a new `MeshEntity` can be created with a different id using `with_id`.
    #[test]
    fn test_with_id() {
        let edge = MeshEntity::Edge(5);

        // Create a new edge with a different id
        let new_edge = edge.with_id(10).unwrap();

        assert_eq!(new_edge.get_id(), 10); // New ID check
        assert_eq!(new_edge.get_entity_type(), "Edge"); // Type remains unchanged
    }
}
