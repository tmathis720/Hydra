// src/domain/mesh_entity.rs

#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Clone, Copy)]
pub enum MeshEntity {
    Vertex(usize),  // Vertex id
    Edge(usize),    // Edge id
    Face(usize),    // Face id
    Cell(usize),    // Cell id
}

impl MeshEntity {
    // Returns the unique identifier of the mesh entity
    pub fn id(&self) -> usize {
        match *self {
            MeshEntity::Vertex(id) => id,
            MeshEntity::Edge(id) => id,
            MeshEntity::Face(id) => id,
            MeshEntity::Cell(id) => id,
        }
    }

    // Returns a human-readable name for the entity type
    pub fn entity_type(&self) -> &str {
        match *self {
            MeshEntity::Vertex(_) => "Vertex",
            MeshEntity::Edge(_) => "Edge",
            MeshEntity::Face(_) => "Face",
            MeshEntity::Cell(_) => "Cell",
        }
    }
}

pub struct Arrow {
    pub from: MeshEntity,
    pub to: MeshEntity,
}

impl Arrow {
    // Constructor for creating a new Arrow
    pub fn new(from: MeshEntity, to: MeshEntity) -> Self {
        Arrow { from, to }
    }

    // Add a new mesh entity and relate it to another entity through an arrow
    pub fn add_entity<T: Into<MeshEntity>>(entity: T) -> MeshEntity {
        let mesh_entity = entity.into();
        println!(
            "Adding entity: {} with id: {}",
            mesh_entity.entity_type(),
            mesh_entity.id()
        );
        mesh_entity
    }

    // Get the "from" and "to" points of the arrow
    pub fn get_relation(&self) -> (&MeshEntity, &MeshEntity) {
        (&self.from, &self.to)
    }
}

// Tests for verifying the functionality of MeshEntity and Arrow

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_id_and_type() {
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(2);
        let face = MeshEntity::Face(3);
        let cell = MeshEntity::Cell(4);

        assert_eq!(vertex.id(), 1);
        assert_eq!(edge.entity_type(), "Edge");
        assert_eq!(face.id(), 3);
        assert_eq!(cell.entity_type(), "Cell");
    }

    #[test]
    fn test_arrow_creation_and_relation() {
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(2);

        let arrow = Arrow::new(vertex, edge);
        let (from, to) = arrow.get_relation();

        assert_eq!(*from, MeshEntity::Vertex(1));
        assert_eq!(*to, MeshEntity::Edge(2));
    }

    #[test]
    fn test_add_entity() {
        let vertex = MeshEntity::Vertex(5);
        let added_entity = Arrow::add_entity(vertex);

        assert_eq!(added_entity.id(), 5);
        assert_eq!(added_entity.entity_type(), "Vertex");
    }
}
