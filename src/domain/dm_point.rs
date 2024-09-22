// src/domain/dm_DPoint.rs

/// Represents a DPoint in the Hasse diagram,
/// either an element, vertex, face, etc.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum DPointType {
    Vertex,
    Edge,
    Face,
    Cell,
}

#[derive(Debug, Clone)]
pub struct DPoint {
    pub id: usize,                    // Unique identifier for the DPoint
    pub point_type: DPointType,        // Type of the DPoint: Vertex, Edge, Face, or Cell
    pub cone: Vec<usize>,              // References to DPoints that "cover" this DPoint (in the Hasse diagram)
    pub support: Vec<usize>,           // References to DPoints that this DPoint covers (dual of cone)
    pub num_fields: usize,             // Number of prognostic variables or fields associated with this DPoint
    pub offset: usize,                 // The starting index in the global data array for this DPoint's data
}

impl DPoint {
    // Create a new DPoint with a given id, DPoint type, and number of fields
    pub fn new(id: usize, point_type: DPointType, num_fields: usize) -> Self {
        DPoint {
            id,
            point_type,
            cone: Vec::new(),
            support: Vec::new(),
            num_fields,
            offset: 0,  // This will be updated when the mesh is initialized
        }
    }

    // Set the cone (covering DPoints) for this DPoint
    pub fn set_cone(&mut self, covering_dpoints: Vec<usize>) {
        self.cone = covering_dpoints;
    }

    // Add a single covering DPoint to the cone
    pub fn add_to_cone(&mut self, dpoint_id: usize) {
        self.cone.push(dpoint_id);
    }

    // Set the support DPoints that this DPoint covers
    pub fn set_support(&mut self, supported_dpoints: Vec<usize>) {
        self.support = supported_dpoints;
    }

    // Add a single support DPoint to this DPoint's support list
    pub fn add_to_support(&mut self, dpoint_id: usize) {
        self.support.push(dpoint_id);
    }

    // Set the offset for this DPoint in the global data array
    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }

    // Get the offset for this DPoint
    pub fn get_offset(&self) -> usize {
        self.offset
    }

    // Get the number of fields associated with this DPoint
    pub fn get_num_fields(&self) -> usize {
        self.num_fields
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dpoint() {
        // Create a vertex DPoint
        let dpoint = DPoint::new(0, DPointType::Vertex, 2);
        assert_eq!(dpoint.id, 0);
        assert_eq!(dpoint.point_type, DPointType::Vertex);
        assert!(dpoint.cone.is_empty());
        assert!(dpoint.support.is_empty());
        assert_eq!(dpoint.num_fields, 2);
    }

    #[test]
    fn test_set_cone() {
        // Create an edge DPoint and set its cone
        let mut dpoint = DPoint::new(1, DPointType::Edge, 1);
        dpoint.set_cone(vec![0, 2]);

        assert_eq!(dpoint.cone, vec![0, 2]);
        assert!(dpoint.support.is_empty()); // Support should remain empty
    }

    #[test]
    fn test_add_to_cone() {
        // Create a face DPoint and add to its cone incrementally
        let mut dpoint = DPoint::new(2, DPointType::Face, 3);
        dpoint.add_to_cone(0);
        dpoint.add_to_cone(1);

        assert_eq!(dpoint.cone, vec![0, 1]);
        assert!(dpoint.support.is_empty());
    }

    #[test]
    fn test_set_support() {
        // Create a cell DPoint and set its support
        let mut dpoint = DPoint::new(3, DPointType::Cell, 1);
        dpoint.set_support(vec![1, 2]);

        assert_eq!(dpoint.support, vec![1, 2]);
        assert!(dpoint.cone.is_empty()); // Cone should remain empty
    }

    #[test]
    fn test_add_to_support() {
        // Create a vertex DPoint and add to its support incrementally
        let mut dpoint = DPoint::new(4, DPointType::Vertex, 2);
        dpoint.add_to_support(1);
        dpoint.add_to_support(2);

        assert_eq!(dpoint.support, vec![1, 2]);
        assert!(dpoint.cone.is_empty());
    }

    #[test]
    fn test_set_and_get_offset() {
        // Create a DPoint and set its offset
        let mut dpoint = DPoint::new(5, DPointType::Edge, 2);
        dpoint.set_offset(10);

        assert_eq!(dpoint.get_offset(), 10);
    }

    #[test]
    fn test_get_num_fields() {
        // Verify that the number of fields for a DPoint is correct
        let dpoint = DPoint::new(6, DPointType::Face, 3);
        assert_eq!(dpoint.get_num_fields(), 3);
    }
}
