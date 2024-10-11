#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel::unbounded;
    use crate::domain::mesh::Mesh;

    #[test]
    fn test_send_receive_boundary_data() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        // Set up vertex coordinates.
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(2, [4.0, 5.0, 6.0]);

        // Add boundary entities.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);

        // Set up a separate sender and receiver for testing.
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate sending the boundary data.
        mesh.send_boundary_data();

        // Create a second mesh instance to simulate the receiver.
        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(mesh.boundary_data_sender.clone().unwrap(), mesh.boundary_data_receiver.clone().unwrap());

        // Simulate receiving the boundary data.
        mesh_receiver.receive_boundary_data();

        // Verify that the receiver mesh has the updated vertex coordinates.
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[4.0, 5.0, 6.0]));
    }

    /* #[test]
    fn test_receive_empty_data() {
        let mut mesh = Mesh::new();
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate receiving without sending any data.
        mesh.receive_boundary_data();

        // Ensure no data has been added.
        assert!(mesh.vertex_coordinates.is_empty());
    } */

    #[test]
    fn test_send_without_receiver() {
        let mut mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(3);
        mesh.vertex_coordinates.insert(3, [7.0, 8.0, 9.0]);
        mesh.add_entity(vertex);

        // Simulate sending the boundary data without setting a receiver.
        mesh.send_boundary_data();

        // No receiver to process, but this should not panic or fail.
        assert!(mesh.vertex_coordinates.get(&3).is_some());
    }

    #[test]
    fn test_add_entity() {
        let mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(1);
        mesh.add_entity(vertex);
        assert!(mesh.entities.read().unwrap().contains(&vertex));
    }

    #[test]
    fn test_iter_vertices() {
        let mut mesh = Mesh::new();
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        let vertices: Vec<_> = mesh.iter_vertices().collect();
        assert_eq!(vertices, vec![&1]);
    }
}
